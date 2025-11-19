#!/usr/bin/env python3
"""
CAFA6 M2 Training - Ultimate Optimized Version
主要优化：
1. Temperature Scaling层（可学习的概率校准）
2. Adaptive Focal Loss（动态gamma）
3. 降低dropout + 增加网络容量
4. 学习率warmup + cosine annealing
5. Gradient clipping
6. 扩展threshold范围（0.001-0.6）
7. 改进的early stopping
8. 详细的训练监控
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
LR = 5e-4                    # 🔥 降低学习率
WEIGHT_DECAY = 1e-4
EPOCHS = 150                 # 🔥 增加最大epochs
PATIENCE = 25                # 🔥 更大的patience
MAX_LABELS = 26125
WARMUP_EPOCHS = 5            # 🔥 学习率warmup
GRAD_CLIP = 1.0              # 🔥 梯度裁剪

CURRENT_FOLD = int(os.environ.get('CURRENT_FOLD', 0))
FOLD_DIR = './folds'

PATHS = {
    'embeddings': './cache/esm2-650M_embeddings.pkl',
    'train_terms': 'data/Train/train_terms.tsv',
    'train_fasta': 'data/Train/train_sequences.fasta',
    'model_save': f'./models/m2_esm2_fold{CURRENT_FOLD}_ultimate.pth',
    'vocab_save': './models/vocab.pkl',
    'log_file': f'./models/training_log_fold{CURRENT_FOLD}_ultimate.csv'
}
os.makedirs('./models', exist_ok=True)

# ================= 🔥 终极Loss Function =================
class AdaptiveFocalLoss(nn.Module):
    """
    自适应Focal Loss
    - 动态调整gamma（难样本多时gamma高，少时gamma低）
    - 高alpha给正样本更多权重
    - 添加样本级别的权重
    """
    def __init__(self, alpha=0.75, gamma_init=2.5, gamma_adaptive=True):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma_init
        self.gamma_adaptive = gamma_adaptive
        print(f"[Loss] AdaptiveFocalLoss(alpha={alpha}, gamma_init={gamma_init}, adaptive={gamma_adaptive})")
    
    def forward(self, logits, targets):
        """
        Args:
            logits: [batch_size, num_classes]
            targets: [batch_size, num_classes]
        """
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        
        probs = torch.sigmoid(logits)
        
        # 计算p_t（正确类别的预测概率）
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # alpha_t权重
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # 🔥 自适应gamma
        if self.gamma_adaptive and self.training:
            # 根据当前batch的平均置信度动态调整gamma
            # 如果模型已经很自信，降低gamma；否则保持高gamma
            avg_confidence = p_t.mean().item()
            gamma = self.gamma * (1.0 - avg_confidence * 0.3)  # 动态范围: [0.7*gamma, gamma]
        else:
            gamma = self.gamma
        
        # Focal weight
        focal_weight = alpha_t * (1 - p_t) ** gamma
        
        # 最终loss
        focal_loss = focal_weight * bce
        
        # 🔥 样本级别权重（正样本少的蛋白质加权）
        pos_count = targets.sum(dim=1, keepdim=True).clamp(min=1)
        sample_weight = torch.sqrt(6.5 / pos_count)  # 归一化到平均正样本数
        
        loss_per_sample = focal_loss.mean(dim=1)
        weighted_loss = (loss_per_sample * sample_weight.squeeze()).mean()
        
        return weighted_loss


# ================= 🔥 优化的模型架构 =================
class ESM2PredictorUltimate(nn.Module):
    """
    终极优化版模型
    1. 更深的网络（3层→4层）
    2. 降低dropout（更自信）
    3. Temperature Scaling层（可学习的概率校准）
    4. Batch Normalization（稳定训练）
    """
    def __init__(self, n_labels, esm_embedding_dim=1280):
        super().__init__()
        
        # 🔥 更深更宽的网络
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
        
        # 🔥 可学习的temperature参数
        # 初始值0.4表示让输出更自信
        self.temperature = nn.Parameter(torch.ones(1) * 0.4)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """Xavier初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        logits = self.head(x)
        
        # 🔥 Temperature scaling
        # temperature在[0.2, 1.0]之间
        # temperature < 1会让概率更极端（更自信）
        temp = self.temperature.clamp(min=0.2, max=1.0)
        scaled_logits = logits / temp
        
        return scaled_logits
    
    def get_temperature(self):
        """获取当前temperature值"""
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


# ================= 🔥 终极评估函数 =================
def calculate_metrics_gpu(y_true_tensor, y_logits_tensor, verbose=False):
    """
    扩展threshold范围 + 详细分析
    """
    probs = torch.sigmoid(y_logits_tensor)
    best_f1 = 0.0
    best_metrics = {}
    
    # 🔥 扩展到0.6，加密高阈值区间
    thresholds = [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2, 
                  0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
    
    if verbose:
        print(f"\n[Metrics Debug]")
        print(f"  Prob stats: min={probs.min():.6f}, max={probs.max():.6f}, "
              f"mean={probs.mean():.6f}, std={probs.std():.6f}")
        print(f"  Probs >0.001: {(probs > 0.001).float().mean():.4f}")
        print(f"  Probs >0.01:  {(probs > 0.01).float().mean():.4f}")
        print(f"  Probs >0.1:   {(probs > 0.1).float().mean():.4f}")
        print(f"  Probs >0.5:   {(probs > 0.5).float().mean():.6f}")
    
    threshold_results = []
    
    for thresh in thresholds:
        preds = (probs > thresh).float()
        tp = (preds * y_true_tensor).sum(dim=1)
        fp = (preds * (1 - y_true_tensor)).sum(dim=1)
        fn = ((1 - preds) * y_true_tensor).sum(dim=1)
        
        f1 = 2 * tp / (2 * tp + fp + fn + 1e-6)
        avg_f1 = f1.mean().item()
        
        if verbose:
            pred_count = preds.sum(dim=1).mean().item()
            threshold_results.append((thresh, avg_f1, pred_count))
        
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
    
    if verbose and len(threshold_results) > 0:
        print(f"\n[Threshold Scan - Top 8]")
        print(f"  {'Thresh':<8} {'F1':<8} {'Avg Pred':<10}")
        # 按F1排序，显示前8个
        sorted_results = sorted(threshold_results, key=lambda x: x[1], reverse=True)
        for t, f, p in sorted_results[:8]:
            print(f"  {t:<8.3f} {f:<8.4f} {p:<10.1f}")
    
    return best_metrics


# ================= 🔥 学习率调度器 =================
class WarmupCosineScheduler:
    """
    Warmup + Cosine Annealing学习率调度
    """
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_epoch = 0
    
    def step(self):
        """更新学习率"""
        if self.current_epoch < self.warmup_epochs:
            # Warmup阶段：线性增长
            lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing阶段
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
    
    print("\n>>> Checking for data leakage...")
    train_set = set(train_pids)
    val_set = set(val_pids)
    overlap = train_set & val_set
    if len(overlap) > 0:
        print(f"    ⚠️  WARNING: {len(overlap)} proteins overlap!")
    else:
        print(f"    ✓ No overlap detected")
    
    print(">>> Building datasets...")
    train_features = []
    train_labels = []
    train_label_counts = []
    
    for pid in tqdm(train_pids, desc="    Train"):
        cache_key = pure_id_to_cache_key[pid]
        train_features.append(embeddings_dict[cache_key])
        labels = [term_to_idx[t] for t in temp_dict[pid]]
        train_labels.append(labels)
        train_label_counts.append(len(labels))
    
    val_features = []
    val_labels = []
    val_label_counts = []
    
    for pid in tqdm(val_pids, desc="    Val"):
        cache_key = pure_id_to_cache_key[pid]
        val_features.append(embeddings_dict[cache_key])
        labels = [term_to_idx[t] for t in temp_dict[pid]]
        val_labels.append(labels)
        val_label_counts.append(len(labels))
    
    X_train = torch.stack(train_features).to(device)
    X_val = torch.stack(val_features).to(device)
    
    print(f"\n>>> Dataset Statistics:")
    print(f"    Train labels: mean={np.mean(train_label_counts):.1f}, std={np.std(train_label_counts):.1f}")
    print(f"    Val labels:   mean={np.mean(val_label_counts):.1f}, std={np.std(val_label_counts):.1f}")
    print(f"    Positive rate: {np.mean(train_label_counts) / num_classes * 100:.4f}%")
    
    print(f"✅ Data ready: Train {X_train.shape}, Val {X_val.shape}")
    
    return X_train, X_val, train_labels, val_labels, num_classes


# ================= 🔥 训练主函数 =================
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "="*90)
    print(f"🚀 CAFA6 M2 Training - Fold {CURRENT_FOLD} (ULTIMATE OPTIMIZED)")
    print("="*90)
    print(f"Config:")
    print(f"  Batch Size:     {BATCH_SIZE}")
    print(f"  Learning Rate:  {LR}")
    print(f"  Max Epochs:     {EPOCHS}")
    print(f"  Warmup Epochs:  {WARMUP_EPOCHS}")
    print(f"  Patience:       {PATIENCE}")
    print(f"  Grad Clip:      {GRAD_CLIP}")
    print("="*90 + "\n")
    
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
    
    # 🔥 使用终极优化模型
    model = ESM2PredictorUltimate(num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    # 🔥 使用自定义学习率调度器
    scheduler = WarmupCosineScheduler(
        optimizer, 
        warmup_epochs=WARMUP_EPOCHS,
        total_epochs=EPOCHS,
        base_lr=LR,
        min_lr=1e-6
    )
    
    scaler = GradScaler()
    
    # 🔥 使用自适应Focal Loss
    criterion = AdaptiveFocalLoss(alpha=0.75, gamma_init=2.5, gamma_adaptive=True)
    
    print(f"\n>>> Model Architecture:")
    print(f"    Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"    Loss: {criterion.__class__.__name__}")
    print(f"    Optimizer: AdamW(lr={LR}, wd={WEIGHT_DECAY})")
    print(f"    Scheduler: WarmupCosine(warmup={WARMUP_EPOCHS}, total={EPOCHS})")
    
    best_val_f1 = 0.0
    patience_counter = 0
    log_data = []
    
    print("\n" + "="*90)
    print("🚀 Training Start")
    print("="*90)
    print(f"{'Ep':<4} | {'Loss':<8} | {'F1':<8} | {'P':<6} {'R':<6} | "
          f"{'Pred':<6} {'True':<6} | {'Th':<6} | {'Temp':<6} | {'LR':<9}")
    print("-"*90)

    for epoch in range(EPOCHS):
        # ==================== Training ====================
        model.train()
        train_loss = 0
        for batch_emb, batch_labels in tqdm(train_loader, desc=f"Ep {epoch+1:3d}", leave=False):
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(batch_emb)
                loss = criterion(outputs, batch_labels)
            
            scaler.scale(loss).backward()
            
            # 🔥 梯度裁剪
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # 🔥 更新学习率
        current_lr = scheduler.step()

        # ==================== Validation ====================
        model.eval()
        val_logits_list = []
        val_targets_list = []
        
        with torch.no_grad():
            for batch_emb, batch_labels in val_loader:
                batch_labels = batch_labels.to(device)
                outputs = model(batch_emb)
                val_logits_list.append(outputs)
                val_targets_list.append(batch_labels)
        
        val_logits = torch.cat(val_logits_list)
        val_targets = torch.cat(val_targets_list)
        
        # 🔥 详细评估（每10个epoch）
        verbose = (epoch % 10 == 0 or epoch < 5)
        metrics = calculate_metrics_gpu(val_targets, val_logits, verbose=verbose)
        
        # 获取当前temperature
        current_temp = model.get_temperature()
        
        # 打印训练状态
        print(f"{epoch+1:<4} | {avg_train_loss:<8.4f} | {metrics['f1']:<8.4f} | "
              f"{metrics['precision']:<6.3f} {metrics['recall']:<6.3f} | "
              f"{metrics['avg_pred']:<6.1f} {metrics['avg_true']:<6.1f} | "
              f"{metrics['best_thresh']:<6.3f} | {current_temp:<6.3f} | {current_lr:<9.2e}")
        
        # 记录日志
        log_data.append({
            'epoch': epoch+1,
            'train_loss': avg_train_loss,
            'val_f1': metrics['f1'],
            'val_precision': metrics['precision'],
            'val_recall': metrics['recall'],
            'best_threshold': metrics['best_thresh'],
            'temperature': current_temp,
            'learning_rate': current_lr
        })
        
        # ==================== Model Saving & Early Stopping ====================
        if metrics['f1'] > best_val_f1:
            improvement = metrics['f1'] - best_val_f1
            best_val_f1 = metrics['f1']
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_val_f1,
                'temperature': current_temp
            }, PATHS['model_save'])
            patience_counter = 0
            print(f"         ⭐ New best! (+{improvement:.4f}) [Saved]")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n🛑 Early stopping at epoch {epoch+1}")
                print(f"   Best F1: {best_val_f1:.4f} (stopped after {PATIENCE} epochs without improvement)")
                break
        
        # 🔥 学习率过低警告
        if current_lr < 1e-6:
            print(f"⚠️  Learning rate too low ({current_lr:.2e})")
    
    # ==================== 保存训练日志 ====================
    pd.DataFrame(log_data).to_csv(PATHS['log_file'], index=False)
    
    print("\n" + "="*90)
    print("✅ Training Complete!")
    print("="*90)
    print(f"📊 Final Results:")
    print(f"   Best Val F1:  {best_val_f1:.4f}")
    print(f"   Total Epochs: {epoch+1}/{EPOCHS}")
    print(f"   Model saved:  {PATHS['model_save']}")
    print(f"   Log saved:    {PATHS['log_file']}")
    print("="*90 + "\n")


if __name__ == "__main__":
    train()