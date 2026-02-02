import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import numpy as np
from Bio import SeqIO
from tqdm import tqdm
import pickle
import gzip
import gc

# === 配置 ===
MODEL_PATH = "models/esm2_t36_3B_UR50D"
ADAPTER_PATH = "models/checkpoints_esm2_3b_asl/final_esm2_3b_asl" 
TEST_FASTA = "data/Test/testsuperset.fasta"
LABEL_MAP = "models/checkpoints_esm2_3b_asl/label_map.pkl"
# 输出改为 .tsv.gz
OUTPUT_FILE = "predictions/esm2_raw.tsv.gz"
BATCH_SIZE = 64
MAX_LEN = 1024 
# 阈值设极低，保留所有可能的信号供后续融合和传播使用
THRESHOLD = 0.0001 

def predict():
    print(f"🚀 加载 ESM2 模型...")
    with open(LABEL_MAP, "rb") as f:
        term2idx = pickle.load(f)
    idx2term = {v: k for k, v in term2idx.items()}
    num_labels = len(term2idx)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH, num_labels=num_labels, torch_dtype=torch.bfloat16, device_map="cuda:0"
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    print("📥 读取并排序序列...")
    sequences, ids = [], []
    for record in SeqIO.parse(TEST_FASTA, "fasta"):
        # ID 清洗逻辑
        header = record.id
        pid = header.split('|')[1] if "|" in header else header.split()[0]
        sequences.append(str(record.seq)[:MAX_LEN])
        ids.append(pid)
    
    # 长度排序提速
    sorted_indices = np.argsort([len(s) for s in sequences])
    sequences = [sequences[i] for i in sorted_indices]
    ids = [ids[i] for i in sorted_indices]
    
    print(f"⚡ 开始推理 -> {OUTPUT_FILE}")
    
    # 使用 gzip 写入，文本模式 (wt)
    with gzip.open(OUTPUT_FILE, "wt") as f_out:
        with torch.no_grad():
            for i in tqdm(range(0, len(sequences), BATCH_SIZE)):
                batch_seqs = sequences[i:i+BATCH_SIZE]
                batch_ids = ids[i:i+BATCH_SIZE]

                inputs = tokenizer(batch_seqs, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to("cuda")
                logits = model(**inputs).logits
                probs = torch.sigmoid(logits).float().cpu().numpy()

                lines = []
                for j, pid in enumerate(batch_ids):
                    # 只要大于 0.0001 的都留着
                    indices = np.where(probs[j] > THRESHOLD)[0]
                    for idx in indices:
                        score = probs[j][idx]
                        lines.append(f"{pid}\t{idx2term[idx]}\t{score:.5f}\n")
                
                f_out.writelines(lines)
                del inputs, logits, probs
                if i % 1000 == 0: gc.collect()

if __name__ == "__main__":
    predict()