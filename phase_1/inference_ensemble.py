#!/usr/bin/env python3
"""
CAFA6 Ensemble Inference - M2 (ESM-2) + M3 (ProtTrans)
策略：Logits 平均 (Soft Voting) -> 生成 Submission -> 之后再跑 Propagation
"""

import os
import torch
import torch.nn as nn
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

# ================= 配置 =================
# 自动寻找最新的 fold 模型
M2_PATHS = [f'./models/m2_esm2_fold{i}_fmax.pth' for i in range(3)]
M3_PATHS = [f'./models/m3_prottrans_fold{i}_fmax.pth' for i in range(3)]

# 缓存路径
CACHE_ESM2 = './cache/esm2-650M_embeddings.pkl'
CACHE_PROT = './cache/prottrans-bert_embeddings.pkl'
VOCAB_PATH = './models/vocab.pkl'
TEST_FASTA = 'data/Test/testsuperset.fasta'

OUTPUT_FILE = 'submission_ensemble.tsv' # 输出给 propagation.py 用

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 2048

# 极低阈值，保留尽可能多的原始数据给 Propagation 脚本处理
INFERENCE_THRESHOLD = 0.001 
MAX_PREDS = 2000

# ================= 模型定义 (必须与训练一致) =================
class ESM2PredictorUltimate(nn.Module):
    def __init__(self, n_labels, esm_embedding_dim=1280):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(esm_embedding_dim, 2560), nn.BatchNorm1d(2560), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(2560, 2048), nn.BatchNorm1d(2048), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(2048, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(1024, n_labels)
        )
        self.temperature = nn.Parameter(torch.ones(1) * 0.4)
    def forward(self, x):
        return self.head(x) / self.temperature.clamp(min=0.2, max=1.0)

class ProtTransPredictor(nn.Module):
    def __init__(self, n_labels, embedding_dim=1024):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embedding_dim, 2560), nn.BatchNorm1d(2560), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(2560, 2048), nn.BatchNorm1d(2048), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(2048, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(1024, n_labels)
        )
        self.temperature = nn.Parameter(torch.ones(1) * 0.4)
    def forward(self, x):
        return self.head(x) / self.temperature.clamp(min=0.2, max=1.0)

# ================= 工具函数 =================
def parse_id(header):
    return header.strip().lstrip('>').split()[0]

def load_cache(path):
    print(f"Loading {path}...")
    with open(path, 'rb') as f:
        data = pickle.load(f)
    # 建立 pure_id -> key 映射
    mapping = {}
    for k in data.keys():
        pure = k.strip().lstrip('>').split()[0]
        mapping[pure] = k
    return data, mapping

# ================= 主流程 =================
def main():
    print("="*80)
    print("🚀 CAFA6 Ensemble Inference (M2 + M3)")
    print("="*80)

    # 1. 加载词表
    with open(VOCAB_PATH, 'rb') as f:
        terms = pickle.load(f)
    term_map = {i: t for i, t in enumerate(terms)}
    n_classes = len(terms)
    print(f"✅ Vocab: {n_classes} terms")

    # 2. 加载 Embeddings
    esm_data, esm_map = load_cache(CACHE_ESM2)
    prot_data, prot_map = load_cache(CACHE_PROT)

    # 3. 匹配测试集
    test_ids = []
    X_esm = []
    X_prot = []
    
    print("Matching test sequences...")
    with open(TEST_FASTA) as f:
        for line in f:
            if line.startswith('>'):
                pid = parse_id(line)
                if pid in esm_map and pid in prot_map:
                    test_ids.append(pid)
                    X_esm.append(esm_data[esm_map[pid]])
                    X_prot.append(prot_data[prot_map[pid]])
    
    print(f"✅ Matched {len(test_ids)} proteins with dual embeddings")
    
    # 转Tensor
    X_esm = torch.stack([torch.tensor(x) if not isinstance(x, torch.Tensor) else x for x in X_esm]).float().to(DEVICE)
    X_prot = torch.stack([torch.tensor(x) if not isinstance(x, torch.Tensor) else x for x in X_prot]).float().to(DEVICE)

    # 4. 加载模型
    models_m2 = []
    models_m3 = []

    print("\nLoading M2 (ESM-2) Models...")
    for p in M2_PATHS:
        if os.path.exists(p):
            ckpt = torch.load(p, map_location=DEVICE)
            m = ESM2PredictorUltimate(n_classes).to(DEVICE)
            m.load_state_dict(ckpt['model_state_dict'])
            m.eval()
            models_m2.append(m)
            print(f"  Loaded {p} (F1_G: {ckpt.get('best_f1_global', 0):.4f})")
            
    print("\nLoading M3 (ProtTrans) Models...")
    for p in M3_PATHS:
        if os.path.exists(p):
            ckpt = torch.load(p, map_location=DEVICE)
            m = ProtTransPredictor(n_classes).to(DEVICE)
            m.load_state_dict(ckpt['model_state_dict'])
            m.eval()
            models_m3.append(m)
            print(f"  Loaded {p} (F1_G: {ckpt.get('best_f1_global', 0):.4f})")

    if not models_m2 and not models_m3:
        print("❌ No models found!")
        return

    # 5. 推理 & 集成
    print(f"\nRunning Ensemble Inference...")
    all_scores = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(test_ids), BATCH_SIZE)):
            batch_esm = X_esm[i:i+BATCH_SIZE]
            batch_prot = X_prot[i:i+BATCH_SIZE]
            
            # M2 推理 (Logits)
            logits_m2 = []
            for m in models_m2:
                logits_m2.append(m(batch_esm))
            
            # M3 推理 (Logits)
            logits_m3 = []
            for m in models_m3:
                logits_m3.append(m(batch_prot))
            
            # 汇总所有可用模型的 Logits
            all_logits = logits_m2 + logits_m3
            avg_logits = torch.stack(all_logits).mean(dim=0)
            
            # Sigmoid 转概率
            probs = torch.sigmoid(avg_logits).cpu().numpy()
            all_scores.append(probs)

    all_scores = np.vstack(all_scores)

    # 6. 写入文件
    print(f"\nWriting to {OUTPUT_FILE} (Threshold > {INFERENCE_THRESHOLD})...")
    count = 0
    with open(OUTPUT_FILE, 'w') as f:
        for i, pid in enumerate(tqdm(test_ids)):
            scores = all_scores[i]
            # 过滤
            indices = np.where(scores > INFERENCE_THRESHOLD)[0]
            # 排序取 Top K
            if len(indices) > MAX_PREDS:
                top_k_idx = np.argsort(scores[indices])[::-1][:MAX_PREDS]
                indices = indices[top_k_idx]
            
            for idx in indices:
                f.write(f"{pid}\t{term_map[idx]}\t{scores[idx]:.4f}\n")
                count += 1
                
    print(f"✅ Done! Total predictions: {count:,}")
    print(f"   Next step: Modify 'propagation.py' to use input='{OUTPUT_FILE}'")

if __name__ == "__main__":
    main()