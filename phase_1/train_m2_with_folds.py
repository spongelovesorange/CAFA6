#!/usr/bin/env python3
"""
CAFA6 M2 Training - 使用正确的CAFA F-max评估
关键修复：同时计算Sample-wise F1和Global F-max，用Global F-max选择最佳模型
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
import math

# ================= 配置 =================
BATCH_SIZE = 2048
LR = 5e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 150
PATIENCE = 25
MAX_LABELS = 26125
WARMUP_EPOCHS = 5
GRAD_CLIP = 1.0

CURRENT_FOLD = int(os.environ.get('CURRENT_FOLD', 0))
FOLD_DIR = './folds'

PATHS = {
    'embeddings': './cache/esm2-650M_embeddings.pkl',
    'train_terms': 'data/Train/train_terms.tsv',
    'train_fasta': 'data/Train/train_sequences.fasta',
    'model_save': f'./models/m2_esm2_fold{CURRENT_FOLD}_fmax.pth',
    'vocab_save': './models/vocab.pkl',
    'log_file': f'./models/training_log_fold{CURRENT_FOLD}_fmax.csv'
}
os.makedirs('./models', exist_ok=True)


# ================= Loss（保持原版） =================
class AdaptiveFocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma_init=2.5, gamma_adaptive=True):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma_init
        self.gamma_adaptive = gamma_adaptive
        print(f"[Loss] AdaptiveFocalLoss(alpha={alpha}, gamma={gamma_init})")
    
    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        if self.gamma_adaptive and self.training:
            avg_confidence = p_t.mean().item()
            gamma = self.gamma * (1.0 - avg_confidence * 0.3)
        else:
            gamma = self.gamma
        
        focal_weight = alpha_t * (1 - p_t) ** gamma
        focal_loss = focal_weight * bce
        
        pos_count = targets.sum(dim=1, keepdim=True).clamp(min=1)
        sample_weight = torch.sqrt(6.5 / pos_count)
        
        loss_per_sample = focal_loss.mean(dim=1)
        weighted_loss = (loss_per_sample * sample_weight.squeeze()).mean()
        
        return weighted_loss


# ================= 模型（保持原版） =================
class ESM2PredictorUltimate(nn.Module):
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
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        logits = self.head(x)
        temp = self.temperature.clamp(min=0.2, max=1.0)
        return logits / temp
    
    def get_temperature(self):
        return self.temperature.clamp(min=0.2, max=1.0).item()


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


# ================= 🔥 修复的评估函数：同时计算两种F1 =================
def calculate_metrics_both(y_true_tensor, y_logits_tensor, verbose=False):
    """
    同时计算:
    1. Sample-wise F1 (你原来的方式)
    2. Global F-max (CAFA官方方式)
    
    返回两个最佳阈值，可能不同！
    """
    probs = torch.sigmoid(y_logits_tensor)
    
    thresholds = [0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
    
    best_sample_f1 = 0.0
    best_sample_thresh = 0.0
    best_global_f1 = 0.0
    best_global_thresh = 0.0
    
    results = []
    
    if verbose:
        print(f"\n[Prob Distribution]")
        print(f"  min={probs.min():.4f}, max={probs.max():.4f}, mean={probs.mean():.4f}")
    
    for thresh in thresholds:
        preds = (probs > thresh).float()
        
        # ========== Sample-wise F1 (原来的方式) ==========
        tp_s = (preds * y_true_tensor).sum(dim=1)
        fp_s = (preds * (1 - y_true_tensor)).sum(dim=1)
        fn_s = ((1 - preds) * y_true_tensor).sum(dim=1)
        
        precision_s = tp_s / (tp_s + fp_s + 1e-6)
        recall_s = tp_s / (tp_s + fn_s + 1e-6)
        f1_s = 2 * tp_s / (2 * tp_s + fp_s + fn_s + 1e-6)
        
        avg_f1_s = f1_s.mean().item()
        avg_p_s = precision_s.mean().item()
        avg_r_s = recall_s.mean().item()
        avg_pred = preds.sum(dim=1).mean().item()
        
        # ========== Global F-max (CAFA官方方式) ==========
        tp_g = (preds * y_true_tensor).sum().item()
        fp_g = (preds * (1 - y_true_tensor)).sum().item()
        fn_g = ((1 - preds) * y_true_tensor).sum().item()
        
        precision_g = tp_g / (tp_g + fp_g + 1e-6)
        recall_g = tp_g / (tp_g + fn_g + 1e-6)
        f1_g = 2 * precision_g * recall_g / (precision_g + recall_g + 1e-6)
        
        results.append({
            'thresh': thresh,
            'f1_sample': avg_f1_s, 'p_sample': avg_p_s, 'r_sample': avg_r_s,
            'f1_global': f1_g, 'p_global': precision_g, 'r_global': recall_g,
            'avg_pred': avg_pred
        })
        
        if avg_f1_s > best_sample_f1:
            best_sample_f1 = avg_f1_s
            best_sample_thresh = thresh
        
        if f1_g > best_global_f1:
            best_global_f1 = f1_g
            best_global_thresh = thresh
    
    if verbose:
        print(f"\n[Threshold Comparison - Top 8 by Global F-max]")
        print(f"  {'Th':<6} {'F1(G)':<8} {'P(G)':<8} {'R(G)':<8} | {'F1(S)':<8} {'Pred':<6}")
        print(f"  {'-'*60}")
        for r in sorted(results, key=lambda x: x['f1_global'], reverse=True)[:8]:
            marker = " ←G" if r['thresh'] == best_global_thresh else ""
            marker += " ←S" if r['thresh'] == best_sample_thresh else ""
            print(f"  {r['thresh']:<6.2f} {r['f1_global']:<8.4f} {r['p_global']:<8.4f} "
                  f"{r['r_global']:<8.4f} | {r['f1_sample']:<8.4f} {r['avg_pred']:<6.1f}{marker}")
        
        if best_sample_thresh != best_global_thresh:
            print(f"\n  ⚠️  注意: Sample最佳阈值({best_sample_thresh}) ≠ Global最佳阈值({best_global_thresh})")
    
    # 获取最佳global阈值对应的详细指标
    best_result = next(r for r in results if r['thresh'] == best_global_thresh)
    
    return {
        'f1_sample': best_sample_f1,
        'f1_global': best_global_f1,
        'thresh_sample': best_sample_thresh,
        'thresh_global': best_global_thresh,
        'precision': best_result['p_global'],
        'recall': best_result['r_global'],
        'avg_pred': best_result['avg_pred'],
        'avg_true': y_true_tensor.sum(dim=1).mean().item()
    }


# ================= 学习率调度 =================
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_epoch = 0
    
    def step(self):
        if self.current_epoch < self.warmup_epochs:
            lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
        else:
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.current_epoch += 1
        return lr


# ================= 数据加载 =================
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
    
    train_features, train_labels = [], []
    for pid in tqdm(train_pids, desc="    Train"):
        cache_key = pure_id_to_cache_key[pid]
        train_features.append(embeddings_dict[cache_key])
        train_labels.append([term_to_idx[t] for t in temp_dict[pid]])
    
    val_features, val_labels = [], []
    for pid in tqdm(val_pids, desc="    Val"):
        cache_key = pure_id_to_cache_key[pid]
        val_features.append(embeddings_dict[cache_key])
        val_labels.append([term_to_idx[t] for t in temp_dict[pid]])
    
    X_train = torch.stack(train_features).to(device)
    X_val = torch.stack(val_features).to(device)
    
    train_label_counts = [len(l) for l in train_labels]
    print(f"\n>>> Dataset Statistics:")
    print(f"    Labels/protein: mean={np.mean(train_label_counts):.1f}, std={np.std(train_label_counts):.1f}")
    print(f"    Positive rate: {np.mean(train_label_counts) / num_classes * 100:.4f}%")
    
    return X_train, X_val, train_labels, val_labels, num_classes


# ================= 训练 =================
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "="*100)
    print(f"🚀 CAFA6 M2 Training - Fold {CURRENT_FOLD} (正确F-max评估)")
    print("="*100)
    print(f"Config: BS={BATCH_SIZE}, LR={LR}, Epochs={EPOCHS}, Patience={PATIENCE}")
    print("="*100 + "\n")
    
    train_idx = np.load(os.path.join(FOLD_DIR, f'fold_{CURRENT_FOLD}_train_idx.npy'))
    val_idx = np.load(os.path.join(FOLD_DIR, f'fold_{CURRENT_FOLD}_val_idx.npy'))
    
    X_train, X_val, train_labels, val_labels, num_classes = load_data_to_gpu(
        device, (train_idx, val_idx)
    )
    
    train_dataset = UltraFastDataset(X_train, train_labels, num_classes)
    val_dataset = UltraFastDataset(X_val, val_labels, num_classes)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    model = ESM2PredictorUltimate(num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = WarmupCosineScheduler(optimizer, WARMUP_EPOCHS, EPOCHS, LR, 1e-6)
    scaler = GradScaler()
    criterion = AdaptiveFocalLoss(alpha=0.75, gamma_init=2.5, gamma_adaptive=True)
    
    print(f">>> Model: {sum(p.numel() for p in model.parameters()):,} params")
    
    best_global_f1 = 0.0
    best_sample_f1 = 0.0
    patience_counter = 0
    log_data = []
    
    print("\n" + "="*100)
    print("🚀 Training Start")
    print("="*100)
    print(f"{'Ep':<4} | {'Loss':<8} | {'F1(G)':<8} {'ThG':<5} | {'F1(S)':<8} {'ThS':<5} | "
          f"{'P':<6} {'R':<6} | {'Pred':<5} | {'LR':<9}")
    print("-"*100)

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        for batch_emb, batch_labels in tqdm(train_loader, desc=f"Ep {epoch+1:3d}", leave=False):
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(batch_emb)
                loss = criterion(outputs, batch_labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
        
        avg_loss = train_loss / len(train_loader)
        current_lr = scheduler.step()
        
        # 验证
        model.eval()
        val_logits, val_targets = [], []
        
        with torch.no_grad():
            for batch_emb, batch_labels in val_loader:
                batch_labels = batch_labels.to(device)
                outputs = model(batch_emb)
                val_logits.append(outputs)
                val_targets.append(batch_labels)
        
        val_logits = torch.cat(val_logits)
        val_targets = torch.cat(val_targets)
        
        verbose = (epoch % 10 == 0 or epoch < 5)
        metrics = calculate_metrics_both(val_targets, val_logits, verbose=verbose)
        
        # 打印 - 显示两种F1和各自的最佳阈值
        print(f"{epoch+1:<4} | {avg_loss:<8.4f} | {metrics['f1_global']:<8.4f} "
              f"{metrics['thresh_global']:<5.2f} | {metrics['f1_sample']:<8.4f} "
              f"{metrics['thresh_sample']:<5.2f} | {metrics['precision']:<6.3f} "
              f"{metrics['recall']:<6.3f} | {metrics['avg_pred']:<5.1f} | {current_lr:<9.2e}")
        
        log_data.append({
            'epoch': epoch+1, 'loss': avg_loss,
            'f1_global': metrics['f1_global'], 'thresh_global': metrics['thresh_global'],
            'f1_sample': metrics['f1_sample'], 'thresh_sample': metrics['thresh_sample'],
            'precision': metrics['precision'], 'recall': metrics['recall'],
            'lr': current_lr
        })
        
        # 🔥 用Global F-max选择最佳模型（这是CAFA评估方式）
        if metrics['f1_global'] > best_global_f1:
            improvement = metrics['f1_global'] - best_global_f1
            best_global_f1 = metrics['f1_global']
            best_sample_f1 = metrics['f1_sample']
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'best_f1_global': best_global_f1,
                'best_f1_sample': best_sample_f1,
                'best_threshold_global': metrics['thresh_global'],
                'best_threshold_sample': metrics['thresh_sample'],
                'temperature': model.get_temperature()
            }, PATHS['model_save'])
            patience_counter = 0
            print(f"         ⭐ New best! F1(G)={best_global_f1:.4f} (+{improvement:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n🛑 Early stopping at epoch {epoch+1}")
                break
    
    pd.DataFrame(log_data).to_csv(PATHS['log_file'], index=False)
    
    print("\n" + "="*100)
    print(f"✅ Training Complete!")
    print(f"   Best Global F-max: {best_global_f1:.4f}")
    print(f"   Best Sample F1:    {best_sample_f1:.4f}")
    print(f"   Model: {PATHS['model_save']}")
    print("="*100 + "\n")


if __name__ == "__main__":
    train()