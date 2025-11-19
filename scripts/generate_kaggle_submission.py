import os
import torch
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm

# ================= 配置 =================
MODEL_PATH = './models/m2_esm2_hyper.pth'
EMBEDDING_PATH = './cache/esm2-650M_embeddings.pkl'
VOCAB_PATH = './models/vocab.pkl'    # <--- 读取刚才训练保存的词表 (关键!)
TEST_FASTA_PATH = 'data/Test/testsuperset.fasta'
OUTPUT_FILE = 'submission.tsv'       

DEVICE = 'cuda'
BATCH_SIZE = 4096 

class ESM2Predictor(torch.nn.Module):
    def __init__(self, n_labels, esm_embedding_dim=1280):
        super().__init__()
        self.head = torch.nn.Sequential(
            torch.nn.Linear(esm_embedding_dim, 2048),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(2048, n_labels)
        )
    def forward(self, x):
        return self.head(x)

def parse_submission_id(header):
    """解析 Fasta Header，获取提交所需的 Protein ID"""
    clean = header.strip()[1:] 
    return clean.split()[0] 

def parse_cache_key(header):
    """解析 Fasta Header，获取 Embedding Cache 的 Key"""
    # 必须与你生成 Cache 时的逻辑一致
    return header.strip()[1:].split()[0]

def main():
    # 1. 加载 Label 映射 (必须与训练一致)
    if not os.path.exists(VOCAB_PATH):
        raise FileNotFoundError(f"未找到 {VOCAB_PATH}！请先运行训练脚本以生成词表。")
        
    print(f">>> Loading Vocab from {VOCAB_PATH}...")
    with open(VOCAB_PATH, 'rb') as f:
        selected_terms = pickle.load(f)
    idx_to_term = {i: t for i, t in enumerate(selected_terms)}
    num_labels = len(selected_terms)
    print(f"Vocab loaded: {num_labels} terms")

    # 2. 加载 Embeddings
    print(">>> Loading Embedding Cache...")
    with open(EMBEDDING_PATH, 'rb') as f:
        embeddings_dict = pickle.load(f)

    # 3. 匹配测试集
    submission_ids = []
    X_list = []
    
    print(">>> Matching Test Sequences...")
    with open(TEST_FASTA_PATH, 'r') as f:
        for line in tqdm(f):
            if line.startswith('>'):
                cache_key = parse_cache_key(line)
                sub_id = parse_submission_id(line)
                
                if cache_key in embeddings_dict:
                    X_list.append(embeddings_dict[cache_key])
                    submission_ids.append(sub_id)
                # else: print(f"Missing: {cache_key}") # 调试用
    
    print(f"Stacking {len(X_list)} embeddings...")
    # L20 显存大，可以直接转 Tensor
    X_test = torch.tensor(np.stack(X_list)).float().to(DEVICE)
    
    # 4. 推理
    print(">>> Inference...")
    model = ESM2Predictor(num_labels).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    
    all_probs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(X_test), BATCH_SIZE)):
            batch = X_test[i:i+BATCH_SIZE]
            logits = model(batch)
            probs = torch.sigmoid(logits).cpu().numpy() # 移回 CPU
            all_probs.append(probs)
            
    all_probs = np.vstack(all_probs)
    
    # 5. 写入原始预测
    print(">>> Writing Submission...")
    with open(OUTPUT_FILE, 'w') as f:
        for i, pid in enumerate(tqdm(submission_ids)):
            # 保留概率 > 0.005 的项，减少文件体积
            indices = np.where(all_probs[i] > 0.005)[0] 
            for idx in indices:
                # 格式: ProteinID <tab> GoID <tab> Score
                f.write(f"{pid}\t{idx_to_term[idx]}\t{all_probs[i][idx]:.3f}\n")
    
    print(f"✅ 推理完成: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()