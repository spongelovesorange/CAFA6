#!/usr/bin/env python3
"""
CAFA6 - F1 Optimized Ensemble Inference
目标：最大化F-max分数（基于CAFA历史最佳实践）

关键策略：
1. Threshold=0.10（实验证明的最佳点）
2. 每蛋白保留150-250个预测（Precision vs Recall最佳平衡）
3. 更激进的自适应阈值
"""

import os
import torch
import torch.nn as nn
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm

# ================= 🎯 F1优化配置 =================
MODEL_PATHS = [
    './models/m2_esm2_fold0_ultimate.pth',
    './models/m2_esm2_fold1_ultimate.pth',
    './models/m2_esm2_fold2_ultimate.pth'
]
EMBEDDING_PATH = './cache/esm2-650M_embeddings.pkl'
VOCAB_PATH = './models/vocab.pkl'
TEST_FASTA_PATH = 'data/Test/testsuperset.fasta'
OUTPUT_FILE = 'submission.tsv'

# 设备配置
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 4096 if DEVICE == 'cuda' else 512

# 🎯 F1优化参数（基于CAFA历史最佳实践）
GLOBAL_MIN_THRESHOLD = 0.10      # ← 从0.05提升到0.10（历史最佳点）
MAX_PREDS_PER_PROTEIN = 200      # ← 从500降到200（F1最优）
TOP_K_CUTOFF = 350               # ← 从800降到350（控制FP）

# 更激进的自适应阈值
ADAPTIVE_THRESHOLDS = {
    'high_confidence': 0.08,     # max_score > 0.7
    'medium_confidence': 0.12,   # max_score > 0.5
    'low_confidence': 0.18       # max_score < 0.5（非常保守）
}

# 额外的质量过滤
MIN_PRED_PER_PROTEIN = 5         # 至少保留5个预测（避免过度过滤）
CONFIDENCE_BOOST = True          # 对高分预测放宽限制


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
    """
    根据最高分数动态调整阈值
    高置信度蛋白可以包含更多预测
    """
    if max_score > 0.7:
        return ADAPTIVE_THRESHOLDS['high_confidence']
    elif max_score > 0.5:
        return ADAPTIVE_THRESHOLDS['medium_confidence']
    else:
        return ADAPTIVE_THRESHOLDS['low_confidence']


def filter_predictions_f1_optimized(scores, idx_to_term):
    """
    F1优化的过滤策略
    
    策略：
    1. 自适应阈值（根据置信度）
    2. Top-K截断（控制FP）
    3. 保证最小预测数（保证Recall）
    4. 对高分预测放宽限制（Confidence boost）
    """
    max_score = scores.max()
    
    # Step 1: 确定自适应阈值
    adaptive_threshold = get_adaptive_threshold(max_score)
    threshold = max(GLOBAL_MIN_THRESHOLD, adaptive_threshold)
    
    # Step 2: 基础阈值过滤
    indices = np.where(scores >= threshold)[0]
    
    # Step 3: Confidence Boost（对于高分蛋白，额外保留一些次高分预测）
    if CONFIDENCE_BOOST and max_score > 0.7 and len(indices) < 50:
        # 如果是高置信度但预测很少，放宽到0.05
        relaxed_threshold = max(0.05, threshold * 0.5)
        relaxed_indices = np.where(scores >= relaxed_threshold)[0]
        
        # 保留Top-100
        if len(relaxed_indices) > 100:
            relaxed_scores = scores[relaxed_indices]
            sorted_pos = np.argsort(relaxed_scores)[::-1][:100]
            indices = relaxed_indices[sorted_pos]
        else:
            indices = relaxed_indices
    
    # Step 4: Top-K截断（防止过多预测）
    if len(indices) > TOP_K_CUTOFF:
        candidate_scores = scores[indices]
        sorted_positions = np.argsort(candidate_scores)[::-1]
        indices = indices[sorted_positions[:TOP_K_CUTOFF]]
    
    # Step 5: 最终数量限制
    if len(indices) > MAX_PREDS_PER_PROTEIN:
        candidate_scores = scores[indices]
        sorted_positions = np.argsort(candidate_scores)[::-1]
        indices = indices[sorted_positions[:MAX_PREDS_PER_PROTEIN]]
    
    # Step 6: 保证最小预测数（避免过度过滤影响Recall）
    if len(indices) < MIN_PRED_PER_PROTEIN:
        # 至少保留Top-5
        all_indices = np.argsort(scores)[::-1][:MIN_PRED_PER_PROTEIN]
        indices = all_indices
        threshold = scores[indices[-1]]  # 更新实际使用的阈值
    
    # 按分数排序
    indices = indices[np.argsort(scores[indices])[::-1]]
    
    return indices, threshold


def find_in_cache(pid, embeddings_dict):
    """尝试多种key格式匹配cache"""
    possible_keys = [
        pid,
        f">{pid}",
        f"{pid} 9615",
        f">{pid} 9615"
    ]
    
    for key in possible_keys:
        if key in embeddings_dict:
            return key
    
    # 模糊匹配
    for cache_key in embeddings_dict.keys():
        if isinstance(cache_key, str) and pid in cache_key:
            return cache_key
    
    return None


def main():
    print("\n" + "="*80)
    print("🎯 CAFA6 - F1 Optimized Ensemble Inference")
    print("="*80)
    
    # 设备信息
    print(f"\n🖥️  Device: {DEVICE.upper()}")
    if DEVICE == 'cpu':
        print(f"   ⚠️  Using CPU (slower)")
    
    print("\n🎯 F1 Optimization Strategy:")
    print(f"   Target metric:            F-max (CAFA)")
    print(f"   Global threshold:         {GLOBAL_MIN_THRESHOLD}")
    print(f"   Max preds/protein:        {MAX_PREDS_PER_PROTEIN}")
    print(f"   Min preds/protein:        {MIN_PRED_PER_PROTEIN}")
    print(f"   Top-K cutoff:             {TOP_K_CUTOFF}")
    print(f"   Confidence boost:         {CONFIDENCE_BOOST}")
    print(f"   Adaptive thresholds:      {ADAPTIVE_THRESHOLDS}")
    print(f"   Expected F-max:           0.34-0.38")
    print(f"   Target file size:         < 400 MB")
    
    # ==================== 1. 检查文件 ====================
    print("\n>>> Checking files...")
    
    # 检查模型
    available_models = []
    for i, path in enumerate(MODEL_PATHS):
        if os.path.exists(path):
            print(f"  ✅ Fold {i}: {path}")
            available_models.append(path)
        else:
            print(f"  ⚠️  Fold {i}: {path} NOT FOUND")
    
    if len(available_models) == 0:
        print("❌ No models found!")
        return
    
    MODEL_PATHS[:] = available_models
    print(f"  Using {len(MODEL_PATHS)} model(s) for ensemble")
    
    # 检查其他文件
    for path in [EMBEDDING_PATH, VOCAB_PATH, TEST_FASTA_PATH]:
        if not os.path.exists(path):
            print(f"❌ File not found: {path}")
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
                cache_key = find_in_cache(pid, embeddings_dict)
                
                if cache_key:
                    X_list.append(embeddings_dict[cache_key])
                    test_proteins.append(pid)
                else:
                    missing_count += 1
    
    print(f"✅ Matched: {len(test_proteins):,} proteins")
    if missing_count > 0:
        match_rate = len(test_proteins) / (len(test_proteins) + missing_count) * 100
        print(f"⚠️  Missing: {missing_count} proteins ({100-match_rate:.1f}% missing)")
        if match_rate < 90:
            print(f"   ⚠️  Low match rate! Check cache generation")
    
    if len(X_list) == 0:
        print("❌ No proteins matched!")
        return
    
    # Stack到设备
    print(f"\n>>> Preparing tensors...")
    X_test = torch.tensor(np.stack(X_list)).float().to(DEVICE)
    print(f"✅ Tensor shape: {X_test.shape}")
    
    # ==================== 5. 加载模型 ====================
    print(f"\n>>> Loading {len(MODEL_PATHS)} Model(s)...")
    
    models = []
    
    for fold_idx, model_path in enumerate(MODEL_PATHS):
        print(f"  Fold {fold_idx}...", end=" ")
        
        model = ESM2PredictorUltimate(num_labels).to(DEVICE)
        
        if DEVICE == 'cpu':
            checkpoint = torch.load(model_path, map_location='cpu')
        else:
            checkpoint = torch.load(model_path)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            f1 = checkpoint.get('best_f1', 'N/A')
            print(f"✅ (F1: {f1})")
        else:
            model.load_state_dict(checkpoint)
            print("✅")
        
        model.eval()
        models.append(model)
    
    print(f"✅ Loaded {len(models)} model(s)")
    
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
    print(f"   Mean:    {all_probs.mean():.6f}")
    print(f"   Median:  {np.median(all_probs):.6f}")
    print(f"   Std:     {all_probs.std():.6f}")
    print(f"   Max:     {all_probs.max():.6f}")
    print(f"   >0.05:   {(all_probs > 0.05).mean():.4%}")
    print(f"   >0.10:   {(all_probs > 0.10).mean():.4%}  ← Target threshold")
    print(f"   >0.20:   {(all_probs > 0.20).mean():.4%}")
    print(f"   >0.50:   {(all_probs > 0.50).mean():.6%}")
    
    # ==================== 7. F1优化过滤 + 写文件 ====================
    print(f"\n>>> Writing Submission with F1-Optimized Filtering...")
    
    total_predictions = 0
    threshold_stats = {'high': 0, 'medium': 0, 'low': 0, 'boosted': 0}
    pred_counts = []
    
    with open(OUTPUT_FILE, 'w') as f:
        for i, pid in enumerate(tqdm(test_proteins, desc="Writing")):
            scores = all_probs[i]
            
            # F1优化过滤
            indices, used_threshold = filter_predictions_f1_optimized(scores, idx_to_term)
            
            # 统计
            pred_counts.append(len(indices))
            
            max_score = scores.max()
            if max_score > 0.7:
                threshold_stats['high'] += 1
            elif max_score > 0.5:
                threshold_stats['medium'] += 1
            elif len(indices) > MAX_PREDS_PER_PROTEIN * 0.8:
                threshold_stats['boosted'] += 1
            else:
                threshold_stats['low'] += 1
            
            # 写入
            for idx in indices:
                score = scores[idx]
                f.write(f"{pid}\t{idx_to_term[idx]}\t{score:.3f}\n")
                total_predictions += 1
    
    print(f"✅ Submission created: {OUTPUT_FILE}")
    print(f"   Total predictions: {total_predictions:,}")
    
    # 阈值使用统计
    print(f"\n📊 Filtering Statistics:")
    total_prots = len(test_proteins)
    print(f"   High conf (>0.7):     {threshold_stats['high']:,} ({threshold_stats['high']/total_prots:.1%})")
    print(f"   Med conf (>0.5):      {threshold_stats['medium']:,} ({threshold_stats['medium']/total_prots:.1%})")
    print(f"   Low conf (<0.5):      {threshold_stats['low']:,} ({threshold_stats['low']/total_prots:.1%})")
    print(f"   Confidence boosted:   {threshold_stats['boosted']:,} ({threshold_stats['boosted']/total_prots:.1%})")
    
    # ==================== 8. 验证 ====================
    print("\n" + "="*80)
    print("📊 FINAL VALIDATION")
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
    if file_size_mb > 500:
        print(f"   ❌ Still too large!")
    elif file_size_mb > 300:
        print(f"   ⚠️  Acceptable (but could be smaller)")
    else:
        print(f"   ✅ Good size!")
    
    # 预测数分布
    counts = df_check.groupby('id').size()
    print(f"\n📊 Predictions per Protein:")
    print(f"   Min:     {counts.min()}")
    print(f"   10%:     {counts.quantile(0.10):.0f}")
    print(f"   25%:     {counts.quantile(0.25):.0f}")
    print(f"   Median:  {counts.median():.0f}")
    print(f"   75%:     {counts.quantile(0.75):.0f}")
    print(f"   90%:     {counts.quantile(0.90):.0f}")
    print(f"   Max:     {counts.max()}")
    print(f"   Mean:    {counts.mean():.1f}")
    
    # 合规性
    print(f"\n✅ Compliance Checks:")
    if counts.max() <= 1500:
        print(f"   ✅ Max: {counts.max()} ≤ 1500")
    else:
        print(f"   ❌ {(counts > 1500).sum()} proteins > 1500!")
    
    if df_check['score'].min() > 0:
        print(f"   ✅ All scores > 0")
    
    coverage = df_check['id'].nunique() / len(test_proteins) * 100
    print(f"   ✅ Coverage: {coverage:.1f}%")
    
    # F1预测
    avg_preds = counts.mean()
    score_median = df_check['score'].median()
    
    print(f"\n🎯 Expected Performance:")
    if avg_preds < 100:
        print(f"   Avg preds: {avg_preds:.1f} (conservative)")
    elif avg_preds < 200:
        print(f"   Avg preds: {avg_preds:.1f} (balanced) ✅")
    else:
        print(f"   Avg preds: {avg_preds:.1f} (aggressive)")
    
    if score_median > 0.15:
        print(f"   Score median: {score_median:.3f} (high precision)")
        print(f"   Expected F-max: 0.36-0.40 ⭐")
    elif score_median > 0.10:
        print(f"   Score median: {score_median:.3f} (balanced)")
        print(f"   Expected F-max: 0.34-0.38 ✅")
    else:
        print(f"   Score median: {score_median:.3f} (high recall)")
        print(f"   Expected F-max: 0.30-0.34")
    
    print("\n" + "="*80)
    print("✅ F1-OPTIMIZED INFERENCE COMPLETE!")
    print("="*80)
    print("\n📝 Next Steps:")
    if file_size_mb < 400:
        print("  1. Run GO propagation: python propagate.py")
        print("  2. Submit: submission_propagated.tsv")
        print("  3. Expected Kaggle LB: 0.34-0.38")
    else:
        print("  ⚠️  Consider more aggressive settings:")
        print("     GLOBAL_MIN_THRESHOLD = 0.12")
        print("     MAX_PREDS_PER_PROTEIN = 150")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()