import os
import torch
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm

# ================= 配置 =================
MODEL_PATH = './models/m2_esm2_hyper.pth'
EMBEDDING_PATH = './cache/esm2-650M_embeddings.pkl'
LABELS_PATH = 'data/Train/train_terms.tsv'
TRAIN_FASTA_PATH = 'data/Train/train_sequences.fasta'
FOLD_NPY_PATH = '/data/CAFA6_QIU/folds/fold_0_val_idx.npy'
OUTPUT_FILE = 'predictions/m2_submission_fold0.tsv'
MAX_LABELS = 26125
DEVICE = 'cuda'
BATCH_SIZE = 4096

# ================= ID 解析工具 =================
def parse_long_id(header):
    # 用于从 Cache 查找: sp|A0A0C5B5G6|MOTSC_HUMAN
    return header.strip()[1:].split()[0]

def parse_short_id(header):
    # 用于提交文件: A0A0C5B5G6
    clean = header.strip()[1:]
    if '|' in clean:
        return clean.split('|')[1]
    return clean.split()[0]

# ================= 主逻辑 =================
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

def generate_submission():
    # 1. 加载 Label 映射
    df = pd.read_csv(LABELS_PATH, sep='\t')
    selected_terms = df['term'].value_counts().head(MAX_LABELS).index.tolist()
    idx_to_term = {i: t for i, t in enumerate(selected_terms)}

    # 2. 读取 FASTA 并提取 Fold 0 的 ID 对 (Long, Short)
    print("Parsing FASTA for IDs...")
    all_pairs = []
    with open(TRAIN_FASTA_PATH, 'r') as f:
        for line in f:
            if line.startswith('>'):
                # 同时保存 (CacheKey, SubmissionID)
                all_pairs.append((parse_long_id(line), parse_short_id(line)))
    
    fold_indices = np.load(FOLD_NPY_PATH)
    target_pairs = [all_pairs[i] for i in fold_indices] # 只取 Fold 0
    
    # 3. 加载 Embeddings
    print("Loading embeddings...")
    with open(EMBEDDING_PATH, 'rb') as f:
        embeddings_dict = pickle.load(f)

    # 4. 准备数据
    valid_pids = [] # 用于提交的短 ID
    X_list = []
    
    print("Matching embeddings...")
    for long_id, short_id in target_pairs:
        if long_id in embeddings_dict:
            X_list.append(embeddings_dict[long_id])
            valid_pids.append(short_id)
        else:
            # 尝试用 Short ID 找找看 (防备万一)
            pass 

    X_val = torch.stack(X_list).float().to(DEVICE)
    print(f"Inference on {len(X_val)} proteins...")

    # 5. 推理
    model = ESM2Predictor(MAX_LABELS).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(X_val), BATCH_SIZE):
            batch = X_val[i:i+BATCH_SIZE]
            logits = model(batch)
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
    all_probs = np.vstack(all_probs)

    # 6. 写入
    print(f"Writing to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        for i, pid in enumerate(tqdm(valid_pids)):
            indices = np.where(all_probs[i] > 0.01)[0]
            for idx in indices:
                f.write(f"{pid}\t{idx_to_term[idx]}\t{all_probs[i][idx]:.3f}\n")
                
    print("✅ Done!")

if __name__ == "__main__":
    os.makedirs('predictions', exist_ok=True)
    generate_submission()