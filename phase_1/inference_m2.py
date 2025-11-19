#!/usr/bin/env python3
"""
CAFA6 - 3-Fold Ensemble Inference
输入: 3个fold的模型checkpoint
输出: submission.tsv (ensemble后的预测)
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
BASE_THRESHOLD = 0.01
MAX_PREDS_PER_PROTEIN = 1500

# ================= 🔥 复制模型定义（和训练时一致）=================
class ESM2PredictorUltimate(nn.Module):
    """必须和训练时的架构完全一致！"""
    def __init__(self, n_labels, esm_embedding_dim=1280):
        super().__init__()
        
        self.head = nn.Sequential(
            # Layer 1
            nn.Linear(esm_embedding_dim, 2560),
            nn.BatchNorm1d(2560),
            nn.ReLU(),
            nn.Dropout(0.25),
            
            # Layer 2
            nn.Linear(2560, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Layer 3
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.15),
            
            # Output layer
            nn.Linear(1024, n_labels)
        )
        
        # Temperature (推理时不需要，但要保持架构一致)
        self.temperature = nn.Parameter(torch.ones(1) * 0.4)
    
    def forward(self, x):
        logits = self.head(x)
        # 推理时使用训练好的temperature
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
    
    # testsuperset.fasta格式: >A0A009IHW8 9615
    return clean.split()[0]


def main():
    print("\n" + "="*80)
    print("🚀 CAFA6 - 3-Fold Ensemble Inference Pipeline")
    print("="*80)
    
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
        print("Please train all folds first!")
        return
    
    for path in [EMBEDDING_PATH, VOCAB_PATH, TEST_FASTA_PATH]:
        if not os.path.exists(path):
            print(f"❌ File not found: {path}")
            return
        print(f"  ✅ {path}")
    
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
                
                # 尝试在cache中查找
                # testsuperset可能的key格式: "A0A009IHW8", ">A0A009IHW8 9615"
                cache_key = None
                for possible_key in [pid, f"{pid} 9615", f">{pid}", f">{pid} 9615"]:
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
        print(f"⚠️  Missing: {missing_count} proteins (not in cache)")
    
    if len(X_list) == 0:
        print("❌ No proteins matched! Check cache key format.")
        return
    
    # Stack到GPU
    print(f"\n>>> Preparing GPU tensors...")
    X_test = torch.tensor(np.stack(X_list)).float().to(DEVICE)
    print(f"✅ Tensor shape: {X_test.shape}")
    
    # ==================== 5. 加载3个模型 ====================
    print(f"\n>>> Loading 3 Fold Models...")
    
    models = []
    temperatures = []
    
    for fold_idx, model_path in enumerate(MODEL_PATHS):
        print(f"\n  Loading Fold {fold_idx}...")
        
        # 创建模型
        model = ESM2PredictorUltimate(num_labels).to(DEVICE)
        
        # 加载checkpoint
        checkpoint = torch.load(model_path)
        
        # 处理不同的checkpoint格式
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            best_f1 = checkpoint.get('best_f1', 'N/A')
            epoch = checkpoint.get('epoch', 'N/A')
            temp = checkpoint.get('temperature', model.get_temperature())
            print(f"    Epoch: {epoch}, F1: {best_f1}, Temp: {temp:.3f}")
        else:
            model.load_state_dict(checkpoint)
            print(f"    Loaded (no metadata)")
        
        model.eval()
        models.append(model)
        temperatures.append(model.get_temperature())
    
    print(f"\n✅ All 3 models loaded")
    print(f"   Temperatures: {[f'{t:.3f}' for t in temperatures]}")
    
    # ==================== 6. Ensemble推理 ====================
    print(f"\n>>> Running Ensemble Inference...")
    print(f"   Strategy: Average logits from 3 models")
    print(f"   Batch size: {BATCH_SIZE}")
    
    all_probs = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(X_test), BATCH_SIZE), desc="Inference"):
            batch = X_test[i:i+BATCH_SIZE]
            
            # 🔥 关键：平均logits（不是概率）
            logits_fold0 = models[0](batch)
            logits_fold1 = models[1](batch)
            logits_fold2 = models[2](batch)
            
            avg_logits = (logits_fold0 + logits_fold1 + logits_fold2) / 3
            
            # 转成概率
            probs = torch.sigmoid(avg_logits).cpu().numpy()
            all_probs.append(probs)
    
    all_probs = np.vstack(all_probs)
    print(f"✅ Inference complete: {all_probs.shape}")
    
    # 统计概率分布
    print(f"\n📊 Probability Distribution:")
    print(f"   Mean:  {all_probs.mean():.6f}")
    print(f"   Std:   {all_probs.std():.6f}")
    print(f"   Max:   {all_probs.max():.6f}")
    print(f"   >0.01: {(all_probs > 0.01).mean():.4%}")
    print(f"   >0.1:  {(all_probs > 0.1).mean():.4%}")
    print(f"   >0.5:  {(all_probs > 0.5).mean():.6%}")
    
    # ==================== 7. 生成提交文件 ====================
    print(f"\n>>> Writing Submission File: {OUTPUT_FILE}")
    print(f"   Threshold: {BASE_THRESHOLD}")
    print(f"   Max predictions/protein: {MAX_PREDS_PER_PROTEIN}")
    
    total_predictions = 0
    
    with open(OUTPUT_FILE, 'w') as f:
        for i, pid in enumerate(tqdm(test_proteins, desc="Writing")):
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
            
            # 写入（格式：protein\tGO_term\tscore）
            for idx in indices:
                score = scores[idx]
                f.write(f"{pid}\t{idx_to_term[idx]}\t{score:.3f}\n")
                total_predictions += 1
    
    print(f"✅ Submission file created")
    
    # ==================== 8. 验证 ====================
    print("\n" + "="*80)
    print("📊 SUBMISSION VALIDATION")
    print("="*80)
    
    df_check = pd.read_csv(OUTPUT_FILE, sep='\t', names=['id', 'term', 'score'])
    
    print(f"\n📈 Basic Statistics:")
    print(f"   Total predictions:     {len(df_check):,}")
    print(f"   Unique proteins:       {df_check['id'].nunique():,}")
    print(f"   Unique GO terms:       {df_check['term'].nunique():,}")
    print(f"   Avg preds/protein:     {len(df_check) / df_check['id'].nunique():.1f}")
    print(f"   Score range:           [{df_check['score'].min():.3f}, {df_check['score'].max():.3f}]")
    print(f"   File size:             {os.path.getsize(OUTPUT_FILE) / (1024*1024):.2f} MB")
    
    # 每个蛋白的预测数量分布
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
    
    # 检查1: 最大预测数
    if counts.max() <= MAX_PREDS_PER_PROTEIN:
        print(f"   ✅ Max predictions: {counts.max()} ≤ {MAX_PREDS_PER_PROTEIN}")
    else:
        over_limit = (counts > MAX_PREDS_PER_PROTEIN).sum()
        print(f"   ❌ {over_limit} proteins exceed {MAX_PREDS_PER_PROTEIN} limit!")
    
    # 检查2: 分数范围
    if df_check['score'].min() > 0 and df_check['score'].max() <= 1.0:
        print(f"   ✅ Score range: (0, 1.0]")
    else:
        print(f"   ❌ Invalid scores detected!")
    
    # 检查3: 格式
    print(f"   ✅ Format: TSV (no header)")
    
    # 检查4: 覆盖率
    expected_proteins = len(test_proteins)
    actual_proteins = df_check['id'].nunique()
    coverage = actual_proteins / expected_proteins * 100
    print(f"   ✅ Coverage: {actual_proteins}/{expected_proteins} ({coverage:.1f}%)")
    
    # 警告
    too_few = (counts < 5).sum()
    if too_few > 0:
        print(f"\n⚠️  Warning: {too_few} proteins have < 5 predictions (may be too conservative)")
    
    print("\n" + "="*80)
    print("✅ ENSEMBLE INFERENCE COMPLETE!")
    print("="*80)
    print("\nNext Steps:")
    print("  1. Run GO propagation:  python propagate.py")
    print("  2. Submit to Kaggle:    submission_propagated.tsv")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()