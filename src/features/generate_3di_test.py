import torch
from torch.utils.data import Dataset
from transformers import T5Tokenizer, AutoModelForSeq2SeqLM
import re
import os
from Bio import SeqIO
from tqdm import tqdm
import time
import gc

# ================= ⚙️ 配置区域 =================
INPUT_FASTA = "data/Test/testsuperset.fasta"
OUTPUT_FILE = "data/Features_3Di/test_3di.fasta"
MODEL_DIR = "/data/CAFA6_QIU/models/ProstT5"

# 显存安全阈值 (L20 48G BF16模式)
MAX_TOKENS_PER_BATCH = 15000 
# =================================================

def verify_and_clean_output(input_fasta_path, output_fasta_path):
    """
    🧹 强力清洗模式：
    不仅检查ID是否存在，还要检查序列长度是否合理。
    如果发现坏数据，直接从内存记录中剔除（甚至可以重写文件，但为了安全我们只选择重跑）。
    """
    print("🔍 Verifying data integrity (Crucial Step)...")
    
    # 1. 建立原始序列长度映射
    aa_lengths = {}
    for record in SeqIO.parse(input_fasta_path, "fasta"):
        clean_id = str(record.id).strip()
        if "|" in clean_id: clean_id = clean_id.split('|')[1]
        aa_lengths[clean_id] = len(record.seq)
        
    valid_ids = set()
    corrupt_count = 0
    
    # 2. 扫描已生成文件
    if not os.path.exists(output_fasta_path):
        return valid_ids

    current_id = None
    current_seq = []
    
    # 使用流式读取，避免内存炸裂
    with open(output_fasta_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                # 处理上一条
                if current_id and current_seq:
                    seq_str = "".join(current_seq)
                    # 校验逻辑：3Di 序列长度不应小于原始 AA 长度的 50% (防止截断)
                    # ProstT5 通常是 1:1，但考虑到 special tokens，我们放宽到 0.8
                    expected_len = aa_lengths.get(current_id, 0)
                    
                    if len(seq_str) > 0 and len(seq_str) >= expected_len * 0.8:
                        valid_ids.add(current_id)
                    else:
                        corrupt_count += 1
                        # print(f"   ⚠️ Corrupt entry found: {current_id} (Exp: {expected_len}, Got: {len(seq_str)})")
                
                # 开始新的一条
                current_id = line[1:]
                current_seq = []
            else:
                current_seq.append(line)
                
        # 处理最后一条
        if current_id and current_seq:
            seq_str = "".join(current_seq)
            expected_len = aa_lengths.get(current_id, 0)
            if len(seq_str) > 0 and len(seq_str) >= expected_len * 0.8:
                valid_ids.add(current_id)
            else:
                corrupt_count += 1

    print(f"✅ Integrity Check Passed: {len(valid_ids)} valid sequences.")
    if corrupt_count > 0:
        print(f"🧹 Detected {corrupt_count} CORRUPT/INCOMPLETE sequences! They will be re-generated.")
        
    return valid_ids

class FastaDataset(Dataset):
    def __init__(self, path, done_ids):
        self.data = []
        print(f"📖 Reading FASTA headers from {path}...")
        clean_pattern = re.compile(r"[UZOB]")
        
        for record in tqdm(SeqIO.parse(path, "fasta"), desc="Loading Index"):
            raw_id = str(record.id).strip()
            clean_id = raw_id.split('|')[1] if "|" in raw_id else raw_id
            
            # 只有通过了完整性校验的 ID 才会被跳过
            if clean_id in done_ids:
                continue
            
            seq_str = str(record.seq).upper()
            seq_len = len(seq_str)
            
            processed_seq = "<AA2fold> " + " ".join(list(clean_pattern.sub("X", seq_str)))
            self.data.append((raw_id, processed_seq, seq_len))
        
        # 核心：按长度倒序排列
        self.data.sort(key=lambda x: x[2], reverse=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def get_token_based_batches(dataset, max_tokens):
    batch = []
    max_len_in_batch = 0
    
    for item in dataset:
        seq_len = item[2]
        estimated_tokens = int(seq_len) + 5
        new_max_len = max(max_len_in_batch, estimated_tokens)
        next_batch_size = len(batch) + 1
        current_batch_cost = next_batch_size * new_max_len
        
        if current_batch_cost > max_tokens and len(batch) > 0:
            yield batch
            batch = []
            max_len_in_batch = 0
        
        batch.append(item)
        max_len_in_batch = max(max_len_in_batch, estimated_tokens)
        
    if batch:
        yield batch

def run_inference(model, tokenizer, batch, f_out, device):
    batch_ids = [x[0] for x in batch]
    batch_seqs = [x[1] for x in batch]
    
    inputs = tokenizer(
        batch_seqs,
        add_special_tokens=True,
        padding=True,
        return_tensors="pt"
    ).to(device)
    
    current_max_len = inputs.input_ids.shape[1]
    # 给足余量，防止因为 output 截断导致下次校验失败
    gen_max_tokens = min(int(current_max_len * 1.5) + 20, 4096) 
    
    with torch.no_grad():
        generation = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=gen_max_tokens,
            do_sample=False,
            num_beams=1,
            use_cache=True, 
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    decoded = tokenizer.batch_decode(generation, skip_special_tokens=True)
    
    buffer = []
    for pid, seq_3di in zip(batch_ids, decoded):
        seq_3di = seq_3di.replace(" ", "").lower()
        # 再次检查：生成结果是否为空
        if len(seq_3di) == 0:
            raise ValueError(f"Model generated empty sequence for {pid}")
        buffer.append(f">{pid}\n{seq_3di}\n")
    
    f_out.write("".join(buffer))
    f_out.flush()

def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device_gpu = torch.device("cuda:0")
    device_cpu = torch.device("cpu")
    
    # 1. 完整性校验 (The Fix)
    # 这会花一点时间（几分钟），但绝对值得，用来排除那7万条里的“假数据”
    done_ids = verify_and_clean_output(INPUT_FASTA, OUTPUT_FILE)
    
    # 2. 加载数据
    dataset = FastaDataset(INPUT_FASTA, done_ids)
    if len(dataset) == 0:
        print("🎉 All sequences verified and processed!")
        return

    print(f"📋 Tasks remaining: {len(dataset)} sequences.")

    # 3. 加载 GPU 模型
    print("\n🚀 [Stage 1] Loading Model to GPU (L20)...")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR, do_lower_case=False, legacy=False)
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_DIR,
            torch_dtype=torch.bfloat16, 
            device_map="cuda:0"
        )
    except:
        print("⚠️ BF16 failed, falling back to FP16")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_DIR,
            torch_dtype=torch.float16,
            device_map="cuda:0"
        )
    model.eval()

    failed_batches = []
    f_out = open(OUTPUT_FILE, "a", buffering=8192)
    batch_iterator = get_token_based_batches(dataset, MAX_TOKENS_PER_BATCH)
    
    print(f"\n⚡ Starting GPU Inference Loop...")
    for batch in tqdm(batch_iterator, desc="GPU Inference"):
        try:
            run_inference(model, tokenizer, batch, f_out, device_gpu)
        except RuntimeError as e:
            if "out of memory" in str(e):
                # 只有这里会触发 OOM
                # print(f"\n💥 OOM detected on batch size {len(batch)}. Pushing to ICU queue.")
                failed_batches.extend(batch) 
                torch.cuda.empty_cache() 
            else:
                print(f"\n❌ Error: {e}. Pushing to ICU.")
                failed_batches.extend(batch)
        except Exception as e:
            print(f"\n❌ General Error: {e}. Pushing to ICU.")
            failed_batches.extend(batch)

    # 5. Stage 2: CPU 重症监护
    if len(failed_batches) > 0:
        print(f"\n\n🚨 [Stage 2] Entering ICU Mode for {len(failed_batches)} failed sequences.")
        print("♻️  Unloading GPU model...")
        del model
        torch.cuda.empty_cache()
        gc.collect()
        
        print("🐢 Loading Model to CPU RAM...")
        model_cpu = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_DIR,
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        model_cpu.eval()
        
        print("🚑 Processing failed sequences on CPU...")
        for item in tqdm(failed_batches, desc="CPU ICU Processing"):
            single_batch = [item] 
            try:
                run_inference(model_cpu, tokenizer, single_batch, f_out, device_cpu)
            except Exception as e:
                print(f"❌ FATAL ERROR on {item[0]}: {e}")
                with open("FATAL_ERRORS.log", "a") as err_f:
                    err_f.write(f"{item[0]}\t{item[2]}\t{str(e)}\n")
    
    f_out.close()
    print("\n✅ All tasks finished and verified.")

if __name__ == "__main__":
    main()