#!/usr/bin/env python3
"""
CAFA6 M3 Training - ProtTrans-BERT Version
架构与M2完全一致，仅修改输入维度为1024适配ProtTrans Embedding
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
# ProtTrans 维度通常为 1024
PROTTRANS_DIM = 1024  
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

# ⚠️ 注意：这里修改了路径指向 ProtTrans 的缓存
PATHS = {
    'embeddings': './cache/prottrans-bert_embeddings.pkl', 
    'train_terms': 'data/Train/train_terms.tsv',
    'train_fasta': 'data/Train/train_sequences.fasta',
    'model_save': f'./models/m3_prottrans_fold{CURRENT_FOLD}_fmax.pth', # 保存为 M3
    'vocab_save': './models/vocab.pkl',
    'log_file': f'./models/training_log_m3_fold{CURRENT_FOLD}_fmax.csv'
}
os.makedirs('./models', exist_ok=True)


# ================= Loss（保持原版） =================
class AdaptiveFocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma_init=2.5, gamma_adaptive=True):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma_init
        self.gamma_adaptive = gamma_adaptive
    
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


# ================= 模型（修改输入层为1024） =================
class ProtTransPredictor(nn.Module):
    def __init__(self, n_labels, embedding_dim=1024): # 默认改为1024
        super().__init__()
        
        self.head = nn.Sequential(
            nn.Linear(embedding_dim, 2560), # 输入层适配
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


# ================= 评估函数 (保持一致) =================
def calculate_metrics_both(y_true_tensor, y_logits_tensor, verbose=False):
    probs = torch.sigmoid(y_logits_tensor)
    thresholds = [0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
    
    best_sample_f1 = 0.0
    best_sample_thresh = 0.0
    best_global_f1 = 0.0
    best_global_thresh = 0.0
    results = []
    
    for thresh in thresholds:
        preds = (probs > thresh).float()
        
        # Sample-wise
        tp_s = (preds * y_true_tensor).sum(dim=1)
        fp_s = (preds * (1 - y_true_tensor)).sum(dim=1)
        fn_s = ((1 - preds) * y_true_tensor).sum(dim=1)
        f1_s = (2 * tp_s / (2 * tp_s + fp_s + fn_s + 1e-6)).mean().item()
        
        # Global
        tp_g = (preds * y_true_tensor).sum().item()
        fp_g = (preds * (1 - y_true_tensor)).sum().item()
        fn_g = ((1 - preds) * y_true_tensor).sum().item()
        precision_g = tp_g / (tp_g + fp_g + 1e-6)
        recall_g = tp_g / (tp_g + fn_g + 1e-6)
        f1_g = 2 * precision_g * recall_g / (precision_g + recall_g + 1e-6)
        
        avg_pred = preds.sum(dim=1).mean().item()
        
        if f1_s > best_sample_f1: best_sample_f1 = f1_s; best_sample_thresh = thresh
        if f1_g > best_global_f1: best_global_f1 = f1_g; best_global_thresh = thresh
        
        results.append({'thresh': thresh, 'f1_global': f1_g, 'p_global': precision_g, 'r_global': recall_g, 'avg_pred': avg_pred})

    best_result = next(r for r in results if r['thresh'] == best_global_thresh)
    
    return {
        'f1_sample': best_sample_f1, 'thresh_sample': best_sample_thresh,
        'f1_global': best_global_f1, 'thresh_global': best_global_thresh,
        'precision': best_result['p_global'], 'recall': best_result['r_global'], 'avg_pred': best_result['avg_pred']
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
        for param_group in self.optimizer.param_groups: param_group['lr'] = lr
        self.current_epoch += 1
        return lr


# ================= 数据加载 =================
def parse_protein_id(header_line):
    header = header_line.strip()
    if header.startswith('>'): header = header[1:]
    return header.split()[0] # 简化处理，ProtTrans通常只有ID

def load_data_to_gpu(device, fold_indices):
    print(">>> Loading ProtTrans Embeddings to GPU Memory...")
    
    # 加载 Embedding
    with open(PATHS['embeddings'], 'rb') as f:
        embeddings_dict = pickle.load(f)
    
    # 检查维度
    first_key = next(iter(embeddings_dict))
    emb_dim = embeddings_dict[first_key].shape[0]
    print(f"    ✓ Loaded {len(embeddings_dict)} protein embeddings. Dimension: {emb_dim}")
    
    if emb_dim != PROTTRANS_DIM:
        print(f"    ⚠️ WARNING: Configured dimension {PROTTRANS_DIM} but found {emb_dim} in file!")
    
    # 建立 ID 映射 (ProtTrans 缓存的 Key 可能是 'P12345' 或 '>P12345')
    pure_id_to_cache_key = {}
    for cache_key in embeddings_dict.keys():
        pure_id = cache_key.strip()
        if pure_id.startswith('>'): pure_id = pure_id[1:]
        pure_id = pure_id.split()[0] # 取第一个空格前
        pure_id_to_cache_key[pure_id] = cache_key
        
    df = pd.read_csv(PATHS['train_terms'], sep='\t')
    
    # 加载或创建 Vocab (必须与 M2 一致!)
    if os.path.exists(PATHS['vocab_save']):
        with open(PATHS['vocab_save'], 'rb') as f:
            selected_terms = pickle.load(f)
        print(f"    ✓ Loaded existing vocabulary ({len(selected_terms)} terms)")
    else:
        print("    ❌ Vocab not found! M2 should be run first to generate vocab.")
        return None

    term_to_idx = {term: i for i, term in enumerate(selected_terms)}
    num_classes = len(selected_terms)
    
    valid_pure_ids = set(pure_id_to_cache_key.keys())
    df = df[df['EntryID'].isin(valid_pure_ids) & df['term'].isin(set(selected_terms))]
    temp_dict = df.groupby('EntryID')['term'].apply(list).to_dict()
    
    # 加载 fold
    fasta_proteins = []
    with open(PATHS['train_fasta'], 'r') as f:
        for line in f:
            if line.startswith('>'): fasta_proteins.append(parse_protein_id(line))
            
    train_idx, val_idx = fold_indices
    
    # 准备数据
    def prepare_subset(indices, name):
        features, labels = [], []
        for idx in tqdm(indices, desc=f"    {name}"):
            if idx >= len(fasta_proteins): continue
            pid = fasta_proteins[idx]
            if pid in temp_dict and pid in pure_id_to_cache_key:
                cache_key = pure_id_to_cache_key[pid]
                features.append(embeddings_dict[cache_key])
                labels.append([term_to_idx[t] for t in temp_dict[pid]])
        return features, labels

    print("    Matching data...")
    train_features_list, train_labels = prepare_subset(train_idx, "Train")
    val_features_list, val_labels = prepare_subset(val_idx, "Val")
    
    # 转换为 Tensor
    X_train = torch.stack([torch.tensor(f) if not isinstance(f, torch.Tensor) else f for f in train_features_list]).to(device)
    X_val = torch.stack([torch.tensor(f) if not isinstance(f, torch.Tensor) else f for f in val_features_list]).to(device)
    
    return X_train, X_val, train_labels, val_labels, num_classes, emb_dim


# ================= 训练 =================
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "="*100)
    print(f"🚀 CAFA6 M3 (ProtTrans) Training - Fold {CURRENT_FOLD}")
    print("="*100)
    
    train_idx = np.load(os.path.join(FOLD_DIR, f'fold_{CURRENT_FOLD}_train_idx.npy'))
    val_idx = np.load(os.path.join(FOLD_DIR, f'fold_{CURRENT_FOLD}_val_idx.npy'))
    
    data = load_data_to_gpu(device, (train_idx, val_idx))
    if data is None: return
    X_train, X_val, train_labels, val_labels, num_classes, actual_dim = data
    
    train_dataset = UltraFastDataset(X_train, train_labels, num_classes)
    val_dataset = UltraFastDataset(X_val, val_labels, num_classes)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # 初始化模型（使用实际检测到的维度）
    model = ProtTransPredictor(num_classes, embedding_dim=actual_dim).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = WarmupCosineScheduler(optimizer, WARMUP_EPOCHS, EPOCHS, LR, 1e-6)
    scaler = GradScaler()
    criterion = AdaptiveFocalLoss(alpha=0.75, gamma_init=2.5, gamma_adaptive=True)
    
    print(f">>> M3 Model Params: {sum(p.numel() for p in model.parameters()):,}")
    
    best_global_f1 = 0.0
    best_sample_f1 = 0.0
    patience_counter = 0
    log_data = []

    print(f"{'Ep':<4} | {'Loss':<8} | {'F1(G)':<8} {'ThG':<5} | {'F1(S)':<8} | {'P':<6} {'R':<6} | {'LR':<9}")
    print("-"*90)

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
                outputs = model(batch_emb)
                val_logits.append(outputs)
                val_targets.append(batch_labels.to(device))
        
        val_logits = torch.cat(val_logits)
        val_targets = torch.cat(val_targets)
        
        metrics = calculate_metrics_both(val_targets, val_logits)
        
        print(f"{epoch+1:<4} | {avg_loss:<8.4f} | {metrics['f1_global']:<8.4f} {metrics['thresh_global']:<5.2f} | "
              f"{metrics['f1_sample']:<8.4f} | {metrics['precision']:<6.3f} {metrics['recall']:<6.3f} | {current_lr:<9.2e}")
        
        log_data.append({**metrics, 'epoch': epoch+1, 'loss': avg_loss})
        
        if metrics['f1_global'] > best_global_f1:
            best_global_f1 = metrics['f1_global']
            best_sample_f1 = metrics['f1_sample']
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'best_f1_global': best_global_f1,
                'best_threshold_global': metrics['thresh_global'],
                'temperature': model.get_temperature()
            }, PATHS['model_save'])
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"🛑 Early stopping at epoch {epoch+1}")
                break
    
    pd.DataFrame(log_data).to_csv(PATHS['log_file'], index=False)
    print(f"\n✅ M3 Training Complete! Best F1(G): {best_global_f1:.4f}")

if __name__ == "__main__":
    train()