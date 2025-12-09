#!/usr/bin/env python3
"""
CAFA6 推理脚本 - 使用Global F-max最佳阈值

关键：
- 训练时Global F-max最佳阈值约0.35
- 每蛋白预测数约5-6个
- 不要用低阈值，会产生大量假阳性
"""

import os
import torch
import torch.nn as nn
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm

# ================= 配置 =================
MODEL_PATHS = [
    './models/m2_esm2_fold0_fmax.pth',
    './models/m2_esm2_fold1_fmax.pth',
    './models/m2_esm2_fold2_fmax.pth'
]

# 备选路径（旧模型）
OLD_MODEL_PATHS = [
    './models/m2_esm2_fold0_ultimate.pth',
    './models/m2_esm2_fold1_ultimate.pth',
    './models/m2_esm2_fold2_ultimate.pth'
]

EMBEDDING_PATH = './cache/esm2-650M_embeddings.pkl'
VOCAB_PATH = './models/vocab.pkl'
TEST_FASTA_PATH = 'data/Test/testsuperset.fasta'
OUTPUT_FILE = 'submission.tsv'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 4096


DEFAULT_THRESHOLD = 0.02         
MAX_PREDS_PER_PROTEIN = 150        
MIN_PREDS_PER_PROTEIN = 30        


# ================= 模型定义 =================
class ESM2PredictorUltimate(nn.Module):
    """带Temperature的模型"""
    def __init__(self, n_labels, esm_embedding_dim=1280):
        super().__init__()
        
        self.head = nn.Sequential(
            nn.Linear(esm_embedding_dim, 2560),
            nn.BatchNorm1d(2560),
            nn.ReLU(),
            nn.Dropout(0.25),
            
            nn.Linear(2560, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.15),
            
            nn.Linear(1024, n_labels)
        )
        
        self.temperature = nn.Parameter(torch.ones(1) * 0.4)
    
    def forward(self, x):
        logits = self.head(x)
        temp = self.temperature.clamp(min=0.2, max=1.0)
        return logits / temp


def parse_protein_id(header):
    clean = header.strip()
    if clean.startswith('>'):
        clean = clean[1:]
    return clean.split()[0]


def find_in_cache(pid, embeddings_dict):
    possible_keys = [pid, f">{pid}", f"{pid} 9615", f">{pid} 9615"]
    
    for key in possible_keys:
        if key in embeddings_dict:
            return key
    
    for cache_key in embeddings_dict.keys():
        if isinstance(cache_key, str) and pid in cache_key:
            return cache_key
    
    return None


def filter_predictions(scores, threshold, max_preds, min_preds):
    """
    严格的 Top-K 过滤，控制文件体积
    """
    # 1. 先拿到所有大于阈值的索引
    indices = np.where(scores >= threshold)[0]
    
    # 2. 按照分数从高到低排序
    sorted_indices_raw = indices[np.argsort(scores[indices])[::-1]]
    
    # 3. 逻辑分支：
    if len(sorted_indices_raw) > max_preds:
        # 情况A: 预测太多 -> 只要前 MAX 个 (比如前300个)
        final_indices = sorted_indices_raw[:max_preds]
        
    elif len(sorted_indices_raw) < min_preds:
        # 情况B: 预测太少 (甚至可能是0) -> 强行去原始分数里找 Top-MIN 个
        # 注意：这里要回到原始 scores 数组去抓，不管阈值
        top_k_all = np.argsort(scores)[::-1][:min_preds]
        final_indices = top_k_all
        
    else:
        # 情况C: 数量适中 -> 保持原样
        final_indices = sorted_indices_raw
        
    return final_indices

def main():
    print("\n" + "="*80)
    print("🎯 CAFA6 推理 - 使用Global F-max最佳阈值")
    print("="*80)
    
    print(f"\n🖥️  Device: {DEVICE}")
    print(f"📋 Configuration:")
    print(f"   Default threshold: {DEFAULT_THRESHOLD}")
    print(f"   Max preds/protein: {MAX_PREDS_PER_PROTEIN}")
    print(f"   Min preds/protein: {MIN_PREDS_PER_PROTEIN}")
    
    # ==================== 检查文件 ====================
    print("\n>>> Checking files...")
    
    available_models = []
    
    # 先找新模型
    for path in MODEL_PATHS:
        if os.path.exists(path):
            available_models.append(path)
            print(f"  ✅ {path}")
    
    # 如果没有，找旧模型
    if not available_models:
        print("  New models not found, trying old paths...")
        for path in OLD_MODEL_PATHS:
            if os.path.exists(path):
                available_models.append(path)
                print(f"  ✅ {path}")
    
    if not available_models:
        print("❌ No models found!")
        return
    
    for path in [EMBEDDING_PATH, VOCAB_PATH, TEST_FASTA_PATH]:
        if not os.path.exists(path):
            print(f"❌ Not found: {path}")
            return
    
    # ==================== 加载数据 ====================
    print(f"\n>>> Loading vocabulary...")
    with open(VOCAB_PATH, 'rb') as f:
        selected_terms = pickle.load(f)
    idx_to_term = {i: t for i, t in enumerate(selected_terms)}
    num_labels = len(selected_terms)
    print(f"✅ {num_labels:,} GO terms")
    
    print(f"\n>>> Loading embeddings...")
    with open(EMBEDDING_PATH, 'rb') as f:
        embeddings_dict = pickle.load(f)
    print(f"✅ {len(embeddings_dict):,} proteins in cache")
    
    print(f"\n>>> Matching test proteins...")
    test_proteins = []
    X_list = []
    
    with open(TEST_FASTA_PATH, 'r') as f:
        for line in tqdm(f, desc="Reading"):
            if line.startswith('>'):
                pid = parse_protein_id(line)
                cache_key = find_in_cache(pid, embeddings_dict)
                if cache_key:
                    X_list.append(embeddings_dict[cache_key])
                    test_proteins.append(pid)
    
    print(f"✅ Matched {len(test_proteins):,} proteins")
    
    X_test = torch.tensor(np.stack(X_list)).float().to(DEVICE)
    
    # ==================== 加载模型 ====================
    print(f"\n>>> Loading {len(available_models)} model(s)...")
    
    models = []
    thresholds_global = []
    
    for model_path in available_models:
        checkpoint = torch.load(model_path, map_location=DEVICE)
        
        model = ESM2PredictorUltimate(num_labels).to(DEVICE)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # 🔥 读取Global最佳阈值
            thresh_g = checkpoint.get('best_threshold_global', None)
            thresh_s = checkpoint.get('best_threshold_sample', checkpoint.get('best_threshold', None))
            f1_g = checkpoint.get('best_f1_global', None)
            f1_s = checkpoint.get('best_f1_sample', checkpoint.get('best_f1', None))
            
            print(f"  {model_path}:")
            print(f"    F1(G)={f1_g}, ThG={thresh_g}")
            print(f"    F1(S)={f1_s}, ThS={thresh_s}")
            
            if thresh_g:
                thresholds_global.append(thresh_g)
            elif thresh_s:
                thresholds_global.append(thresh_s)
        else:
            model.load_state_dict(checkpoint)
            print(f"  {model_path}: loaded (no metadata)")
        
        model.eval()
        models.append(model)

    THRESHOLD = DEFAULT_THRESHOLD  # 即 0.001
    print(f"\n🎯 [FIXED] Forcing inference threshold: {THRESHOLD:.3f} (ignoring training threshold)")
    
    # ==================== 推理 ====================
    print(f"\n>>> Running inference...")
    
    all_probs = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(X_test), BATCH_SIZE), desc="Inference"):
            batch = X_test[i:i+BATCH_SIZE]
            
            logits_sum = None
            for model in models:
                logits = model(batch)
                if logits_sum is None:
                    logits_sum = logits
                else:
                    logits_sum += logits
            
            avg_logits = logits_sum / len(models)
            probs = torch.sigmoid(avg_logits).cpu().numpy()
            all_probs.append(probs)
    
    all_probs = np.vstack(all_probs)
    
    # 概率分布
    print(f"\n📊 Probability Distribution:")
    print(f"   Mean: {all_probs.mean():.4f}, Max: {all_probs.max():.4f}")
    print(f"   >0.1: {(all_probs > 0.1).mean():.2%}")
    print(f"   >0.3: {(all_probs > 0.3).mean():.4%}")
    print(f"   >0.5: {(all_probs > 0.5).mean():.6%}")
    
    # ==================== 写文件 ====================
    print(f"\n>>> Writing submission (threshold={THRESHOLD:.3f})...")
    
    total_preds = 0
    pred_counts = []
    
    with open(OUTPUT_FILE, 'w') as f:
        for i, pid in enumerate(tqdm(test_proteins, desc="Writing")):
            scores = all_probs[i]
            
            indices = filter_predictions(
                scores, 
                threshold=THRESHOLD,
                max_preds=MAX_PREDS_PER_PROTEIN,
                min_preds=MIN_PREDS_PER_PROTEIN
            )
            
            pred_counts.append(len(indices))
            
            for idx in indices:
                f.write(f"{pid}\t{idx_to_term[idx]}\t{scores[idx]:.3f}\n")
                total_preds += 1
    
    print(f"✅ Created: {OUTPUT_FILE}")
    
    # ==================== 统计 ====================
    print("\n" + "="*80)
    print("📊 SUBMISSION STATISTICS")
    print("="*80)
    
    df = pd.read_csv(OUTPUT_FILE, sep='\t', names=['id', 'term', 'score'])
    file_size = os.path.getsize(OUTPUT_FILE) / (1024*1024)
    
    print(f"\n📈 Summary:")
    print(f"   Predictions: {len(df):,}")
    print(f"   Proteins:    {df['id'].nunique():,}")
    print(f"   File size:   {file_size:.1f} MB")
    
    counts = df.groupby('id').size()
    print(f"\n📊 Predictions per protein:")
    print(f"   Min={counts.min()}, Median={counts.median():.0f}, "
          f"Mean={counts.mean():.1f}, Max={counts.max()}")
    
    print(f"\n📊 Score distribution:")
    print(f"   Min={df['score'].min():.3f}, Median={df['score'].median():.3f}, "
          f"Max={df['score'].max():.3f}")
    
    # 预估性能
    print(f"\n🎯 Expected Performance:")
    print(f"   Training F1(G): ~0.21")
    print(f"   With threshold {THRESHOLD:.2f} and avg {counts.mean():.1f} preds/protein")
    print(f"   Expected Kaggle F-max: 0.18-0.25")
    print(f"   After GO propagation: 0.22-0.30")
    
    print("\n" + "="*80)
    print("✅ DONE!")
    print(f"   Next: python propagation_fixed.py")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()