#!/usr/bin/env python3
"""
CAFA6 - Optimized 3-Fold Ensemble Inference
重点优化：控制文件大小，避免7GB爆炸

关键改进：
1. 自适应阈值（高分蛋白用低阈值，低分蛋白用高阈值）
2. 严格限制预测数量
3. Top-K截断
"""

import os
import torch
import torch.nn as nn
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm

# ================= 🔥 优化后的配置 =================
MODEL_PATHS = [
    './models/m2_esm2_fold0_ultimate.pth',
    './models/m2_esm2_fold1_ultimate.pth',
    './models/m2_esm2_fold2_ultimate.pth'
]
EMBEDDING_PATH = './cache/esm2-650M_embeddings.pkl'
VOCAB_PATH = './models/vocab.pkl'
TEST_FASTA_PATH = 'data/Test/testsuperset.fasta'
OUTPUT_FILE = 'submission.tsv'

DEVICE = 'cuda'
BATCH_SIZE = 4096

# 🔥 激进的过滤策略（控制文件大小）
GLOBAL_MIN_THRESHOLD = 0.05  # ← 从0.01提升到0.05
MAX_PREDS_PER_PROTEIN = 500  # ← 从1500降到500
TOP_K_CUTOFF = 800           # ← 即使>0.05，也只保留Top-800

# 自适应阈值策略
ADAPTIVE_THRESHOLDS = {
    'high_confidence': 0.03,   # 如果max_score > 0.7
    'medium_confidence': 0.05,  # 如果max_score > 0.5
    'low_confidence': 0.10      # 如果max_score < 0.5
}


class ESM2PredictorUltimate(nn.Module):
    """和训练时一致的模型架构"""
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
        scaled_logits = logits / temp
        return scaled_logits
    
    def get_temperature(self):
        return self.temperature.clamp(min=0.2, max=1.0).item()


def parse_protein_id(header):
    """从FASTA header提取protein ID"""
    clean = header.strip()
    if clean.startswith('>'):
        clean = clean[1:]
    return clean.split()[0]


def get_adaptive_threshold(max_score):
    """根据最高分数动态调整阈值"""
    if max_score > 0.7:
        return ADAPTIVE_THRESHOLDS['high_confidence']
    elif max_score > 0.5:
        return ADAPTIVE_THRESHOLDS['medium_confidence']
    else:
        return ADAPTIVE_THRESHOLDS['low_confidence']


def filter_predictions_smart(scores, idx_to_term):
    """
    智能过滤策略：
    1. 自适应阈值
    2. Top-K截断
    3. 严格数量限制
    """
    max_score = scores.max()
    
    # 策略1: 自适应阈值
    adaptive_threshold = get_adaptive_threshold(max_score)
    threshold = max(GLOBAL_MIN_THRESHOLD, adaptive_threshold)
    
    # 策略2: 阈值过滤
    indices = np.where(scores >= threshold)[0]
    
    # 策略3: Top-K截断（即使超过阈值也不要太多）
    if len(indices) > TOP_K_CUTOFF:
        candidate_scores = scores[indices]
        sorted_positions = np.argsort(candidate_scores)[::-1]
        indices = indices[sorted_positions[:TOP_K_CUTOFF]]
    
    # 策略4: 最终数量限制
    if len(indices) > MAX_PREDS_PER_PROTEIN:
        candidate_scores = scores[indices]
        sorted_positions = np.argsort(candidate_scores)[::-1]
        indices = indices[sorted_positions[:MAX_PREDS_PER_PROTEIN]]
    
    # 按分数排序
    indices = indices[np.argsort(scores[indices])[::-1]]
    
    return indices, threshold


def main():
    print("\n" + "="*80)
    print("🚀 CAFA6 - Optimized Ensemble Inference (File Size Control)")
    print("="*80)
    
    print("\n🔥 Filtering Strategy:")
    print(f"   Global min threshold:     {GLOBAL_MIN_THRESHOLD}")
    print(f"   Adaptive thresholds:      {ADAPTIVE_THRESHOLDS}")
    print(f"   Max preds/protein:        {MAX_PREDS_PER_PROTEIN}")
    print(f"   Top-K cutoff:             {TOP_K_CUTOFF}")
    print(f"   Target file size:         < 500 MB")
    
    # ==================== 1. 检查文件 ====================
    print("\n>>> Checking files...")
    
    missing_models = []
    for i, path in enumerate(MODEL_PATHS):
        if os.path.exists(path):
            print(f"  ✅ Fold {i}: {path}")
        else:
            print(f"  ❌ Fold {i}: {path} NOT FOUND!")
            missing_models.append(i)
    
    if missing_models:
        print(f"\n❌ Missing models for folds: {missing_models}")
        print("📝 Tip: If you only have fold0, edit MODEL_PATHS to use single model")
        return
    
    # ==================== 2. 加载词表 ====================
    print(f"\n>>> Loading Vocabulary...")
    with open(VOCAB_PATH, 'rb') as f:
        selected_terms = pickle.load(f)
    
    idx_to_term = {i: t for i, t in enumerate(selected_terms)}
    num_labels = len(selected_terms)
    print(f"✅ Vocab: {num_labels:,} GO terms")
    
    # ==================== 3. 加载Embeddings ====================
    print(f"\n>>> Loading ESM2 Embeddings Cache...")
    with open(EMBEDDING_PATH, 'rb') as f:
        embeddings_dict = pickle.load(f)
    print(f"✅ Cache: {len(embeddings_dict):,} proteins")
    
    # ==================== 4. 匹配测试集 ====================
    print(f"\n>>> Matching Test Sequences...")
    
    test_proteins = []
    X_list = []
    missing_count = 0
    
    with open(TEST_FASTA_PATH, 'r') as f:
        for line in tqdm(f, desc="Reading FASTA"):
            if line.startswith('>'):
                pid = parse_protein_id(line)
                
                # 多种key格式尝试
                cache_key = None
                for possible_key in [pid, f"{pid} 9615", f">{pid}", f">{pid} 9615", line.strip()]:
                    if possible_key in embeddings_dict:
                        cache_key = possible_key
                        break
                
                if cache_key:
                    X_list.append(embeddings_dict[cache_key])
                    test_proteins.append(pid)
                else:
                    missing_count += 1
    
    print(f"✅ Matched: {len(test_proteins):,} proteins")
    if missing_count > 0:
        print(f"⚠️  Missing: {missing_count} proteins")
    
    if len(X_list) == 0:
        print("❌ No proteins matched! Run debug script to check cache keys.")
        return
    
    # Stack到GPU
    print(f"\n>>> Preparing GPU tensors...")
    X_test = torch.tensor(np.stack(X_list)).float().to(DEVICE)
    print(f"✅ Tensor shape: {X_test.shape}")
    
    # ==================== 5. 加载模型 ====================
    print(f"\n>>> Loading {len(MODEL_PATHS)} Fold Models...")
    
    models = []
    
    for fold_idx, model_path in enumerate(MODEL_PATHS):
        print(f"  Loading Fold {fold_idx}...", end=" ")
        
        model = ESM2PredictorUltimate(num_labels).to(DEVICE)
        checkpoint = torch.load(model_path)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ (F1: {checkpoint.get('best_f1', 'N/A'):.4f})")
        else:
            model.load_state_dict(checkpoint)
            print("✅")
        
        model.eval()
        models.append(model)
    
    print(f"✅ All models loaded")
    
    # ==================== 6. Ensemble推理 ====================
    print(f"\n>>> Running Ensemble Inference...")
    
    all_probs = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(X_test), BATCH_SIZE), desc="Inference"):
            batch = X_test[i:i+BATCH_SIZE]
            
            # 平均所有fold的logits
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
    print(f"✅ Inference complete: {all_probs.shape}")
    
    # 概率分布分析
    print(f"\n📊 Probability Distribution:")
    print(f"   Mean:   {all_probs.mean():.6f}")
    print(f"   Median: {np.median(all_probs):.6f}")
    print(f"   Max:    {all_probs.max():.6f}")
    print(f"   >0.05:  {(all_probs > 0.05).mean():.4%}")
    print(f"   >0.10:  {(all_probs > 0.10).mean():.4%}")
    print(f"   >0.50:  {(all_probs > 0.50).mean():.6%}")
    
    # ==================== 7. 智能过滤 + 写文件 ====================
    print(f"\n>>> Writing Submission with Smart Filtering...")
    
    total_predictions = 0
    threshold_stats = {'high': 0, 'medium': 0, 'low': 0}
    
    with open(OUTPUT_FILE, 'w') as f:
        for i, pid in enumerate(tqdm(test_proteins, desc="Writing")):
            scores = all_probs[i]
            
            # 智能过滤
            indices, used_threshold = filter_predictions_smart(scores, idx_to_term)
            
            # 统计阈值使用
            if used_threshold <= 0.03:
                threshold_stats['high'] += 1
            elif used_threshold <= 0.05:
                threshold_stats['medium'] += 1
            else:
                threshold_stats['low'] += 1
            
            # 写入
            for idx in indices:
                score = scores[idx]
                f.write(f"{pid}\t{idx_to_term[idx]}\t{score:.3f}\n")
                total_predictions += 1
    
    print(f"✅ Submission file created: {OUTPUT_FILE}")
    print(f"   Total predictions: {total_predictions:,}")
    
    # 阈值使用统计
    print(f"\n📊 Adaptive Threshold Usage:")
    total_proteins = len(test_proteins)
    print(f"   High conf (≤0.03): {threshold_stats['high']:,} ({threshold_stats['high']/total_proteins:.1%})")
    print(f"   Med conf  (≤0.05): {threshold_stats['medium']:,} ({threshold_stats['medium']/total_proteins:.1%})")
    print(f"   Low conf  (>0.05): {threshold_stats['low']:,} ({threshold_stats['low']/total_proteins:.1%})")
    
    # ==================== 8. 验证 ====================
    print("\n" + "="*80)
    print("📊 SUBMISSION VALIDATION")
    print("="*80)
    
    df_check = pd.read_csv(OUTPUT_FILE, sep='\t', names=['id', 'term', 'score'])
    
    file_size_mb = os.path.getsize(OUTPUT_FILE) / (1024*1024)
    
    print(f"\n📈 File Statistics:")
    print(f"   Total predictions:     {len(df_check):,}")
    print(f"   Unique proteins:       {df_check['id'].nunique():,}")
    print(f"   Unique GO terms:       {df_check['term'].nunique():,}")
    print(f"   Avg preds/protein:     {len(df_check) / df_check['id'].nunique():.1f}")
    print(f"   Score range:           [{df_check['score'].min():.3f}, {df_check['score'].max():.3f}]")
    print(f"   📁 File size:          {file_size_mb:.1f} MB")
    
    # 文件大小判断
    if file_size_mb > 1000:
        print(f"   ❌ TOO LARGE! Still > 1GB")
    elif file_size_mb > 500:
        print(f"   ⚠️  Large (but acceptable)")
    else:
        print(f"   ✅ Good size!")
    
    # 每个蛋白的预测数量
    counts = df_check.groupby('id').size()
    print(f"\n📊 Predictions per Protein:")
    print(f"   Min:     {counts.min()}")
    print(f"   25%:     {counts.quantile(0.25):.0f}")
    print(f"   Median:  {counts.median():.0f}")
    print(f"   75%:     {counts.quantile(0.75):.0f}")
    print(f"   Max:     {counts.max()}")
    print(f"   Mean:    {counts.mean():.1f}")
    
    # 合规性检查
    print(f"\n✅ Compliance Checks:")
    
    if counts.max() <= 1500:
        print(f"   ✅ Max predictions: {counts.max()} ≤ 1500")
    else:
        print(f"   ❌ {(counts > 1500).sum()} proteins exceed 1500!")
    
    if df_check['score'].min() > 0 and df_check['score'].max() <= 1.0:
        print(f"   ✅ Score range valid")
    else:
        print(f"   ❌ Invalid scores!")
    
    coverage = df_check['id'].nunique() / len(test_proteins) * 100
    print(f"   ✅ Coverage: {coverage:.1f}%")
    
    print("\n" + "="*80)
    print("✅ OPTIMIZED INFERENCE COMPLETE!")
    print("="*80)
    print("\n📝 Next Steps:")
    if file_size_mb < 500:
        print("  ✅ File size OK! Run: python propagate.py")
    else:
        print("  ⚠️  File still large. Consider:")
        print("     - Increase GLOBAL_MIN_THRESHOLD to 0.08")
        print("     - Decrease MAX_PREDS_PER_PROTEIN to 300")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()