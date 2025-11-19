import os
import torch
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm

# ================= 配置 =================
MODEL_PATH = './models/m2_esm2_hyper.pth'
EMBEDDING_PATH = './cache/esm2-650M_embeddings.pkl'
VOCAB_PATH = './models/vocab.pkl'
TEST_FASTA_PATH = 'data/Test/testsuperset.fasta'
OUTPUT_FILE = 'submission.tsv'
DEVICE = 'cuda'
BATCH_SIZE = 4096

# 🔥 推理参数
BASE_THRESHOLD = 0.01  # 基础阈值
MAX_PREDS_PER_PROTEIN = 1500  # CAFA规定

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
    """提取提交用的Protein ID"""
    clean = header.strip()[1:]
    return clean.split()[0]

def parse_cache_key(header):
    """提取Cache Key（必须与生成cache时一致）"""
    return header.strip()[1:].split()[0]

def main():
    print("="*60)
    print("CAFA6 M2 Inference Pipeline")
    print("="*60)
    
    # 1. 加载词表
    print(f"\n>>> Loading Vocab from {VOCAB_PATH}...")
    if not os.path.exists(VOCAB_PATH):
        raise FileNotFoundError(f"❌ Vocab file not found! Please run train_m2.py first.")
    
    with open(VOCAB_PATH, 'rb') as f:
        selected_terms = pickle.load(f)
    
    idx_to_term = {i: t for i, t in enumerate(selected_terms)}
    num_labels = len(selected_terms)
    print(f"✅ Vocab loaded: {num_labels} terms")
    
    # 2. 加载 Embeddings
    print(f"\n>>> Loading Embedding Cache from {EMBEDDING_PATH}...")
    with open(EMBEDDING_PATH, 'rb') as f:
        embeddings_dict = pickle.load(f)
    print(f"✅ Cache loaded: {len(embeddings_dict)} proteins")
    
    # 3. 匹配测试集
    submission_ids = []
    X_list = []
    missing_count = 0
    
    print(f"\n>>> Matching Test Sequences from {TEST_FASTA_PATH}...")
    with open(TEST_FASTA_PATH, 'r') as f:
        for line in tqdm(f):
            if line.startswith('>'):
                cache_key = parse_cache_key(line)
                sub_id = parse_submission_id(line)
                
                if cache_key in embeddings_dict:
                    X_list.append(embeddings_dict[cache_key])
                    submission_ids.append(sub_id)
                else:
                    missing_count += 1
    
    print(f"✅ Matched: {len(X_list)} proteins")
    if missing_count > 0:
        print(f"⚠️  Missing: {missing_count} proteins (not in cache)")
    
    print(f"\n>>> Stacking {len(X_list)} embeddings to GPU...")
    X_test = torch.tensor(np.stack(X_list)).float().to(DEVICE)
    print(f"✅ Tensor shape: {X_test.shape}")
    
    # 4. 加载模型并推理
    print(f"\n>>> Loading Model from {MODEL_PATH}...")
    model = ESM2Predictor(num_labels).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    print("✅ Model loaded")
    
    print(f"\n>>> Running Inference (Batch Size: {BATCH_SIZE})...")
    all_probs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(X_test), BATCH_SIZE)):
            batch = X_test[i:i+BATCH_SIZE]
            logits = model(batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
    
    all_probs = np.vstack(all_probs)
    print(f"✅ Inference complete: {all_probs.shape}")
    
    # 5. 生成提交文件
    print(f"\n>>> Writing Submission to {OUTPUT_FILE}...")
    print(f"    Threshold: {BASE_THRESHOLD}")
    print(f"    Max predictions/protein: {MAX_PREDS_PER_PROTEIN}")
    
    with open(OUTPUT_FILE, 'w') as f:
        for i, pid in enumerate(tqdm(submission_ids)):
            scores = all_probs[i]
            
            # 阈值过滤
            indices = np.where(scores >= BASE_THRESHOLD)[0]
            
            # 限制最多1500个
            if len(indices) > MAX_PREDS_PER_PROTEIN:
                candidate_scores = scores[indices]
                sorted_positions = np.argsort(candidate_scores)[::-1]
                indices = indices[sorted_positions[:MAX_PREDS_PER_PROTEIN]]
            
            # 按分数排序
            indices = indices[np.argsort(scores[indices])[::-1]]
            
            # 写入
            for idx in indices:
                score = scores[idx]
                f.write(f"{pid}\t{idx_to_term[idx]}\t{score:.3f}\n")
    
    print(f"✅ Submission file created: {OUTPUT_FILE}")
    
    # 6. 验证输出
    print("\n" + "="*60)
    print("VALIDATION REPORT")
    print("="*60)
    
    df_check = pd.read_csv(OUTPUT_FILE, sep='\t', names=['id', 'term', 'score'])
    
    print(f"\n📊 Statistics:")
    print(f"  Total predictions: {len(df_check):,}")
    print(f"  Unique proteins: {df_check['id'].nunique():,}")
    print(f"  Avg preds/protein: {len(df_check) / df_check['id'].nunique():.1f}")
    print(f"  Score range: [{df_check['score'].min():.3f}, {df_check['score'].max():.3f}]")
    print(f"  File size: {os.path.getsize(OUTPUT_FILE) / (1024*1024):.1f} MB")
    
    counts = df_check.groupby('id').size()
    print(f"\n📈 Predictions per protein:")
    print(f"  Min: {counts.min()}")
    print(f"  Max: {counts.max()}")
    print(f"  Median: {counts.median():.0f}")
    print(f"  Mean: {counts.mean():.1f}")
    
    # 检查合规性
    print(f"\n✅ Compliance Check:")
    if counts.max() <= 1500:
        print("  ✅ All proteins within 1500 prediction limit")
    else:
        print(f"  ❌ {(counts > 1500).sum()} proteins exceed 1500!")
    
    too_few = (counts < 10).sum()
    if too_few > 0:
        print(f"  ⚠️  {too_few} proteins have < 10 predictions")
    
    print("\n" + "="*60)
    print(f"✅ READY FOR PROPAGATION: Run python propagate.py")
    print("="*60)

if __name__ == "__main__":
    main()