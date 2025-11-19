#!/usr/bin/env python3
"""
独立测试脚本 - 测试不同阈值下的性能
不需要import训练脚本
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ================= 配置 =================
CURRENT_FOLD = 0
PATHS = {
    'embeddings': './cache/esm2-650M_embeddings.pkl',
    'train_terms': 'data/Train/train_terms.tsv',
    'train_fasta': 'data/Train/train_sequences.fasta',
    'model_path': f'./models/m2_esm2_fold{CURRENT_FOLD}_v2.pth',
    'vocab': './models/vocab.pkl',
}
FOLD_DIR = './folds'

# ================= 复制必要的类 =================
class ESM2Predictor(nn.Module):
    def __init__(self, n_labels, esm_embedding_dim=1280):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(esm_embedding_dim, 2048),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, n_labels)
        )

    def forward(self, x):
        return self.head(x)


class UltraFastDataset(Dataset):
    def __init__(self, embedding_tensor, labels_list, num_classes):
        self.embeddings = embedding_tensor
        self.labels = labels_list
        self.num_classes = num_classes

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        emb = self.embeddings[idx]
        label_indices = self.labels[idx]
        label_vec = torch.zeros(self.num_classes, dtype=torch.float32)
        if len(label_indices) > 0:
            label_vec[label_indices] = 1.0
        return emb, label_vec


def parse_protein_id(header_line):
    header = header_line.strip()
    if header.startswith('>'):
        header = header[1:]
    if '|' in header:
        parts = header.split('|')
        if len(parts) >= 2:
            return parts[1]
    return header.split()[0]


def load_protein_ids_from_fasta(fasta_path):
    protein_ids = []
    with open(fasta_path, 'r') as f:
        for line in f:
            if line.startswith('>'):
                pid = parse_protein_id(line)
                protein_ids.append(pid)
    return protein_ids


def load_val_data(device):
    """加载验证集数据"""
    print(">>> Loading validation data...")
    
    # 加载embeddings
    with open(PATHS['embeddings'], 'rb') as f:
        embeddings_dict = pickle.load(f)
    print(f"    ✓ Loaded {len(embeddings_dict)} protein embeddings")
    
    # ID映射
    pure_id_to_cache_key = {}
    for cache_key in embeddings_dict.keys():
        if '|' in cache_key:
            pure_id = cache_key.split('|')[1]
        else:
            pure_id = cache_key.split()[0]
        pure_id_to_cache_key[pure_id] = cache_key
    
    # 加载vocab
    with open(PATHS['vocab'], 'rb') as f:
        selected_terms = pickle.load(f)
    term_to_idx = {term: i for i, term in enumerate(selected_terms)}
    num_classes = len(selected_terms)
    print(f"    ✓ Loaded vocab: {num_classes} terms")
    
    # 加载标签
    df = pd.read_csv(PATHS['train_terms'], sep='\t')
    valid_pure_ids = set(pure_id_to_cache_key.keys())
    df = df[df['EntryID'].isin(valid_pure_ids) & df['term'].isin(set(selected_terms))]
    temp_dict = df.groupby('EntryID')['term'].apply(list).to_dict()
    
    # 加载FASTA
    fasta_proteins = load_protein_ids_from_fasta(PATHS['train_fasta'])
    
    # 加载fold indices
    val_idx = np.load(os.path.join(FOLD_DIR, f'fold_{CURRENT_FOLD}_val_idx.npy'))
    
    # 构建验证集
    val_pids = []
    for idx in val_idx:
        if idx < len(fasta_proteins):
            pid = fasta_proteins[idx]
            if pid in temp_dict and pid in pure_id_to_cache_key:
                val_pids.append(pid)
    
    print(f"    ✓ Validation set: {len(val_pids)} proteins")
    
    # 构建数据
    val_features = []
    val_labels = []
    
    for pid in tqdm(val_pids, desc="    Loading"):
        cache_key = pure_id_to_cache_key[pid]
        val_features.append(embeddings_dict[cache_key])
        labels = [term_to_idx[t] for t in temp_dict[pid]]
        val_labels.append(labels)
    
    X_val = torch.stack(val_features).to(device)
    
    return X_val, val_labels, num_classes


def test_thresholds(model, val_loader, val_targets_tensor, device):
    """测试不同阈值"""
    print("\n>>> Generating predictions...")
    model.eval()
    
    val_logits_list = []
    with torch.no_grad():
        for batch_emb, _ in tqdm(val_loader):
            outputs = model(batch_emb)
            val_logits_list.append(outputs)
    
    val_logits = torch.cat(val_logits_list)
    probs = torch.sigmoid(val_logits)
    
    print("\n>>> Probability Distribution:")
    print(f"    Min:  {probs.min():.6f}")
    print(f"    Max:  {probs.max():.6f}")
    print(f"    Mean: {probs.mean():.6f}")
    print(f"    Std:  {probs.std():.6f}")
    print(f"    >0.001: {(probs > 0.001).float().mean():.4f}")
    print(f"    >0.01:  {(probs > 0.01).float().mean():.4f}")
    print(f"    >0.1:   {(probs > 0.1).float().mean():.4f}")
    print(f"    >0.5:   {(probs > 0.5).float().mean():.6f}")
    
    # 测试不同阈值
    print("\n" + "="*80)
    print("Testing Different Thresholds")
    print("="*80)
    print(f"{'Thresh':<8} | {'F1':<8} | {'Precision':<10} | {'Recall':<8} | {'Pred':<8} | {'True':<8}")
    print("-"*80)
    
    thresholds = [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
    
    best_f1 = 0
    best_thresh = 0
    results = []
    
    for thresh in thresholds:
        preds = (probs > thresh).float()
        
        tp = (preds * val_targets_tensor).sum(dim=1)
        fp = (preds * (1 - val_targets_tensor)).sum(dim=1)
        fn = ((1 - preds) * val_targets_tensor).sum(dim=1)
        
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * tp / (2 * tp + fp + fn + 1e-6)
        
        avg_f1 = f1.mean().item()
        avg_precision = precision.mean().item()
        avg_recall = recall.mean().item()
        avg_pred = preds.sum(dim=1).mean().item()
        avg_true = val_targets_tensor.sum(dim=1).mean().item()
        
        print(f"{thresh:<8.3f} | {avg_f1:<8.4f} | {avg_precision:<10.4f} | {avg_recall:<8.4f} | "
              f"{avg_pred:<8.1f} | {avg_true:<8.1f}")
        
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_thresh = thresh
        
        results.append({
            'threshold': thresh,
            'f1': avg_f1,
            'precision': avg_precision,
            'recall': avg_recall,
            'pred_count': avg_pred,
            'true_count': avg_true
        })
    
    print("="*80)
    print(f"🎯 Best Threshold: {best_thresh:.3f} (F1: {best_f1:.4f})")
    print("="*80)
    
    # 保存结果
    df = pd.DataFrame(results)
    output_path = f'./threshold_analysis_fold{CURRENT_FOLD}.csv'
    df.to_csv(output_path, index=False)
    print(f"\n✅ Results saved to: {output_path}")
    
    return results


def main():
    print("\n" + "="*80)
    print("CAFA6 - Threshold Analysis Tool")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # 检查模型文件
    if not os.path.exists(PATHS['model_path']):
        print(f"❌ Model not found: {PATHS['model_path']}")
        print("Please train the model first!")
        return
    
    # 加载数据
    X_val, val_labels, num_classes = load_val_data(device)
    
    # 构建dataset
    val_dataset = UltraFastDataset(X_val, val_labels, num_classes)
    val_loader = DataLoader(val_dataset, batch_size=2048, shuffle=False, num_workers=0)
    
    # 构建targets tensor (用于计算metrics)
    val_targets_list = []
    for _, labels in val_loader:
        val_targets_list.append(labels.to(device))
    val_targets_tensor = torch.cat(val_targets_list)
    
    print(f"✅ Validation data ready: {X_val.shape}")
    
    # 加载模型
    print(f"\n>>> Loading model from {PATHS['model_path']}...")
    model = ESM2Predictor(num_classes).to(device)
    model.load_state_dict(torch.load(PATHS['model_path']))
    model.eval()
    print(f"✅ Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # 测试阈值
    results = test_thresholds(model, val_loader, val_targets_tensor, device)
    
    # 可视化建议
    print("\n" + "="*80)
    print("📊 Recommendations")
    print("="*80)
    
    df = pd.DataFrame(results)
    
    # 找到F1最高的点
    best_row = df.loc[df['f1'].idxmax()]
    print(f"\n1️⃣  Best F1 Score:")
    print(f"    Threshold: {best_row['threshold']:.3f}")
    print(f"    F1: {best_row['f1']:.4f}")
    print(f"    Precision: {best_row['precision']:.4f}")
    print(f"    Recall: {best_row['recall']:.4f}")
    
    # 找到Pred/True最接近1.0的点
    df['ratio'] = df['pred_count'] / df['true_count']
    df['ratio_diff'] = abs(df['ratio'] - 1.0)
    balanced_row = df.loc[df['ratio_diff'].idxmin()]
    print(f"\n2️⃣  Most Balanced Predictions:")
    print(f"    Threshold: {balanced_row['threshold']:.3f}")
    print(f"    F1: {balanced_row['f1']:.4f}")
    print(f"    Pred/True: {balanced_row['pred_count']:.1f}/{balanced_row['true_count']:.1f}")
    
    # 找到Recall最高的点(F1 > 0.10)
    high_recall = df[df['f1'] > 0.10].loc[df[df['f1'] > 0.10]['recall'].idxmax()]
    print(f"\n3️⃣  Highest Recall (F1>0.10):")
    print(f"    Threshold: {high_recall['threshold']:.3f}")
    print(f"    F1: {high_recall['f1']:.4f}")
    print(f"    Recall: {high_recall['recall']:.4f}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()