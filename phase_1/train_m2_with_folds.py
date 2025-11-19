#!/usr/bin/env python3
"""
CAFA6 M2 Training - Focal Loss版本
解决极度类别不平衡问题
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# ================= 配置 =================
BATCH_SIZE = 2048
LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 100
PATIENCE = 15
MAX_LABELS = 26125

# 🔥 Focal Loss参数
FOCAL_ALPHA = 0.25  # 正样本权重
FOCAL_GAMMA = 2.0   # 聚焦参数

CURRENT_FOLD = int(os.environ.get('CURRENT_FOLD', 0))
FOLD_DIR = './folds'

PATHS = {
    'embeddings': './cache/esm2-650M_embeddings.pkl',
    'train_terms': 'data/Train/train_terms.tsv',
    'train_fasta': 'data/Train/train_sequences.fasta',
    'model_save': f'./models/m2_esm2_fold{CURRENT_FOLD}_focal.pth',
    'vocab_save': './models/vocab.pkl',
    'log_file': f'./models/training_log_fold{CURRENT_FOLD}_focal.csv'
}
os.makedirs('./models', exist_ok=True)

# ================= 🔥 Focal Loss =================
class FocalLoss(nn.Module):
    """
    Focal Loss for multi-label classification
    
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    
    Args:
        alpha: 正样本权重，默认0.25
        gamma: 聚焦参数，默认2.0。越大越聚焦于难分类样本
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, targets):
        """
        Args:
            logits: [batch_size, num_classes]
            targets: [batch_size, num_classes] (0或1)
        """
        # 计算BCE
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        
        # 计算预测概率
        probs = torch.sigmoid(logits)
        
        # 计算p_t：正样本用p，负样本用1-p
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # 计算alpha_t：正样本用alpha，负样本用1-alpha
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Focal Loss = alpha_t * (1 - p_t)^gamma * BCE
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        focal_loss = focal_weight * bce_loss
        
        return focal_loss.mean()

# ================= 模型 =================
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

def calculate_metrics_gpu(y_true_tensor, y_logits_tensor):
    probs = torch.sigmoid(y_logits_tensor)
    best_f1 = 0.0
    best_metrics = {}
    
    thresholds = [0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25]  # 🔥 添加更低阈值
    
    for thresh in thresholds:
        preds = (probs > thresh).float()
        tp = (preds * y_true_tensor).sum(dim=1)
        fp = (preds * (1 - y_true_tensor)).sum(dim=1)
        fn = ((1 - preds) * y_true_tensor).sum(dim=1)
        
        f1 = 2 * tp / (2 * tp + fp + fn + 1e-6)
        avg_f1 = f1.mean().item()
        
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            precision = tp / (tp + fp + 1e-6)
            recall = tp / (tp + fn + 1e-6)
            best_metrics = {
                'f1': avg_f1,
                'precision': precision.mean().item(),
                'recall': recall.mean().item(),
                'avg_pred': preds.sum(dim=1).mean().item(),
                'avg_true': y_true_tensor.sum(dim=1).mean().item(),
                'best_thresh': thresh
            }
    return best_metrics

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

def load_data_to_gpu(device, fold_indices):
    print(">>> Loading Data to GPU Memory...")
    
    with open(PATHS['embeddings'], 'rb') as f:
        embeddings_dict = pickle.load(f)
    print(f"    ✓ Loaded {len(embeddings_dict)} protein embeddings")
    
    print(">>> Creating ID mapping...")
    pure_id_to_cache_key = {}
    for cache_key in embeddings_dict.keys():
        if '|' in cache_key:
            pure_id = cache_key.split('|')[1]
        else:
            pure_id = cache_key.split()[0]
        pure_id_to_cache_key[pure_id] = cache_key
    
    df = pd.read_csv(PATHS['train_terms'], sep='\t')
    print(f"    ✓ Loaded {len(df)} annotations")
    
    term_counts = df['term'].value_counts()
    selected_terms = term_counts.head(MAX_LABELS).index.tolist()
    
    if CURRENT_FOLD == 0 or not os.path.exists(PATHS['vocab_save']):
        with open(PATHS['vocab_save'], 'wb') as f:
            pickle.dump(selected_terms, f)

    term_to_idx = {term: i for i, term in enumerate(selected_terms)}
    num_classes = len(selected_terms)
    
    valid_pure_ids = set(pure_id_to_cache_key.keys())
    df = df[df['EntryID'].isin(valid_pure_ids) & df['term'].isin(set(selected_terms))]
    temp_dict = df.groupby('EntryID')['term'].apply(list).to_dict()
    print(f"    ✓ {len(temp_dict)} proteins with both embeddings and labels")
    
    fasta_proteins = load_protein_ids_from_fasta(PATHS['train_fasta'])
    print(f"    ✓ Found {len(fasta_proteins)} proteins in FASTA")
    
    train_idx, val_idx = fold_indices
    
    train_pids = []
    for idx in train_idx:
        if idx < len(fasta_proteins):
            pid = fasta_proteins[idx]
            if pid in temp_dict and pid in pure_id_to_cache_key:
                train_pids.append(pid)
    
    val_pids = []
    for idx in val_idx:
        if idx < len(fasta_proteins):
            pid = fasta_proteins[idx]
            if pid in temp_dict and pid in pure_id_to_cache_key:
                val_pids.append(pid)
    
    print(f"    ✓ Fold split: {len(train_pids)} train, {len(val_pids)} val")
    
    print(">>> Building datasets...")
    train_features = []
    train_labels = []
    for pid in tqdm(train_pids, desc="    Train"):
        cache_key = pure_id_to_cache_key[pid]
        train_features.append(embeddings_dict[cache_key])
        train_labels.append([term_to_idx[t] for t in temp_dict[pid]])
    
    val_features = []
    val_labels = []
    for pid in tqdm(val_pids, desc="    Val"):
        cache_key = pure_id_to_cache_key[pid]
        val_features.append(embeddings_dict[cache_key])
        val_labels.append([term_to_idx[t] for t in temp_dict[pid]])
    
    X_train = torch.stack(train_features).to(device)
    X_val = torch.stack(val_features).to(device)
    
    print(f"✅ Data ready: Train {X_train.shape}, Val {X_val.shape}")
    
    return X_train, X_val, train_labels, val_labels, num_classes

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "="*80)
    print(f"CAFA6 M2 Training - Fold {CURRENT_FOLD} (FOCAL LOSS)")
    print("="*80)
    print(f"Batch: {BATCH_SIZE} | LR: {LR}")
    print(f"Focal Alpha: {FOCAL_ALPHA} | Focal Gamma: {FOCAL_GAMMA}")
    print("="*80 + "\n")
    
    train_idx_path = os.path.join(FOLD_DIR, f'fold_{CURRENT_FOLD}_train_idx.npy')
    val_idx_path = os.path.join(FOLD_DIR, f'fold_{CURRENT_FOLD}_val_idx.npy')
    
    train_idx = np.load(train_idx_path)
    val_idx = np.load(val_idx_path)
    
    X_train, X_val, train_labels, val_labels, num_classes = load_data_to_gpu(
        device, (train_idx, val_idx)
    )
    
    train_dataset = UltraFastDataset(X_train, train_labels, num_classes)
    val_dataset = UltraFastDataset(X_val, val_labels, num_classes)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    model = ESM2Predictor(num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    scaler = GradScaler()
    
    # 🔥 使用Focal Loss
    criterion = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
    
    print(f">>> Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f">>> Loss: Focal Loss (alpha={FOCAL_ALPHA}, gamma={FOCAL_GAMMA})\n")
    
    best_val_f1 = 0.0
    patience_counter = 0
    log_data = []
    
    print("🚀 Training Start")
    print(f"{'Epoch':<6} | {'Loss':<8} | {'Val F1':<8} | {'Pre/Rec':<12} | {'Pred/True':<12} | {'Thresh'}")
    print("-" * 80)

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for batch_emb, batch_labels in tqdm(train_loader, desc=f"Ep {epoch+1}", leave=False):
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(batch_emb)
                loss = criterion(outputs, batch_labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_logits_list = []
        val_targets_list = []
        
        with torch.no_grad():
            for batch_emb, batch_labels in val_loader:
                batch_labels = batch_labels.to(device)
                outputs = model(batch_emb)
                val_logits_list.append(outputs)
                val_targets_list.append(batch_labels)
        
        metrics = calculate_metrics_gpu(torch.cat(val_targets_list), torch.cat(val_logits_list))
        
        print(f"{epoch+1:<6} | {avg_train_loss:.4f}   | {metrics['f1']:.4f}   | "
              f"{metrics['precision']:.2f}/{metrics['recall']:.2f}   | "
              f"{metrics['avg_pred']:.1f}/{metrics['avg_true']:.1f}   | {metrics['best_thresh']:.2f}")
        
        log_data.append({
            'epoch': epoch+1,
            'train_loss': avg_train_loss,
            'val_f1': metrics['f1'],
            'val_precision': metrics['precision'],
            'val_recall': metrics['recall']
        })
        
        scheduler.step(metrics['f1'])
        
        if metrics['f1'] > best_val_f1:
            best_val_f1 = metrics['f1']
            torch.save(model.state_dict(), PATHS['model_save'])
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"🛑 Early stopping at epoch {epoch+1}")
                break
    
    pd.DataFrame(log_data).to_csv(PATHS['log_file'], index=False)
    
    print("\n" + "="*80)
    print(f"✅ Complete! Best F1: {best_val_f1:.4f}")
    print(f"    Model: {PATHS['model_save']}")
    print("="*80 + "\n")

if __name__ == "__main__":
    train()