import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
import re
import os
from Bio import SeqIO
from tqdm import tqdm

# ================= ⚙️ L20 (48GB) 专属配置 =================
# L20 显存比 A800 小，处理 2048 长度序列时，16 是安全水位
# 如果跑得动且显存有富余，可以尝试改到 20 或 24，但 16 绝对稳。
BATCH_SIZE = 16      
MAX_LEN = 2048        

# 任务列表：只跑训练集
TASKS = [
    {
        "name": "TRAIN_SET",
        "input": "/data/CAFA6_QIU/data/Train/train_sequences.fasta",
        "output": "/data/CAFA6_QIU/data/Features_3Di/train_3di.fasta"
    }
]

MODEL_DIR = "/data/CAFA6_QIU/models/ProstT5"
# 🔥 指定使用 GPU 1
DEVICE = torch.device("cuda:1")

class ProstDataset(Dataset):
    def __init__(self, fasta_file, done_ids):
        self.data = []
        print(f"📖 Reading: {os.path.basename(fasta_file)}")
        clean_pattern = re.compile(r"[UZOB]")
        
        for record in SeqIO.parse(fasta_file, "fasta"):
            raw_id = str(record.id).strip()
            # ID 清洗逻辑：保持与 A800 脚本完全一致
            clean_id = raw_id.split('|')[1] if "|" in raw_id else raw_id
            
            if clean_id in done_ids: continue
            
            seq = clean_pattern.sub("X", str(record.seq))[:MAX_LEN]
            self.data.append({
                "id": raw_id, # 输出原始ID
                "seq": "<AA2fold> " + " ".join(list(seq)),
                "len": len(seq)
            })
            
        # 按长度排序 (提速关键)
        self.data.sort(key=lambda x: x["len"])
        print(f"✅ Loaded {len(self.data)} sequences")

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

def collate_fn(batch):
    return {
        "ids": [item['id'] for item in batch],
        "seqs": [item['seq'] for item in batch]
    }

def main():
    print(f"🚀 Loading ProstT5 (BFloat16) on L20 (GPU: {DEVICE})...")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR, do_lower_case=False, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16).to(DEVICE)
    model.eval()
    
    # 编译加速
    try:
        model = torch.compile(model)
        print("🔌 Torch Compile Enabled")
    except: pass

    for task in TASKS:
        print(f"\n{'='*40}")
        print(f"🔥 Starting Task: {task['name']}")
        print(f"{'='*40}")
        
        if not os.path.exists(task['input']):
            print(f"❌ Error: Input file not found: {task['input']}")
            continue

        # 断点扫描逻辑
        done_ids = set()
        if os.path.exists(task['output']):
            with open(task['output'], "r") as f:
                for line in f:
                    if line.startswith(">"):
                        done_ids.add(line.strip()[1:].split('|')[1] if "|" in line else line.strip()[1:])
        
        dataset = ProstDataset(task['input'], done_ids)
        if len(dataset) == 0:
            print("   Task already completed!")
            continue

        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=8, collate_fn=collate_fn, pin_memory=True)

        f = open(task['output'], "a", buffering=8192)
        try:
            for batch in tqdm(loader, desc=task['name']):
                try:
                    inputs = tokenizer(batch["seqs"], add_special_tokens=True, 
                                     padding="longest", return_tensors="pt").to(DEVICE)
                    
                    with torch.no_grad():
                        gen = model.generate(
                            input_ids=inputs.input_ids, 
                            attention_mask=inputs.attention_mask,
                            max_length=MAX_LEN+50, 
                            do_sample=False, num_beams=1
                        )
                    
                    decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
                    
                    buffer = []
                    for pid, seq_3di in zip(batch["ids"], decoded):
                        clean_3di = seq_3di.replace(" ", "").lower()
                        buffer.append(f">{pid}\n{clean_3di}\n")
                    f.write("".join(buffer))
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print("⚠️ OOM! Skipping batch.")
                        torch.cuda.empty_cache()
                        continue
                    raise e
        finally:
            f.close()
            print(f"✅ {task['name']} Finished!")

    print("\n🎉🎉 Train Set Processing Completed on L20!")

if __name__ == "__main__":
    main()