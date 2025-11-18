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
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score

# ================= 配置区域 (Configuration) =================
# [Resource Strategy] 基于 Task 0.4 测试结果优化
BATCH_SIZE = 2048      # 你的 48GB 显存完全吃得消
LR = 1e-3              # 初始学习率
EPOCHS = 50            # 总轮次
PATIENCE = 5           # 早停耐心值 (几轮不提升就停止)
MAX_LABELS = 40000     # 尽可能覆盖全量标签

PATHS = {
    'embeddings': './cache/esm2-650M_embeddings.pkl',
    'train_terms': 'data/Train/train_terms.tsv',
    'model_save': './models/m2_esm2_best.pth',
    'log_file': './models/training_log.csv'
}
os.makedirs('./models', exist_ok=True)

# ================= 1. 组件定义 =================

# [Source: Listing 11] IC加权 Loss - 核心竞争力来源
class ICWeightedBCELoss(nn.Module):
    def __init__(self, ic_weights, device='cuda'):
        super().__init__()
        self.ic_weights = torch.tensor(ic_weights).float().to(device)

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        # 广播权重: [1, n_labels] * [batch, n_labels]
        weighted_bce = bce_loss * self.ic_weights.unsqueeze(0)
        return weighted_bce.mean()

# [Source: Listing 11] 模型架构
class ESM2Predictor(nn.Module):
    def __init__(self, n_labels, esm_embedding_dim=1280):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(esm_embedding_dim, 2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, n_labels)
        )

    def forward(self, x):
        return self.head(x)

class CachedEmbeddingDataset(Dataset):
    def __init__(self, protein_ids, embeddings_dict, labels_dict, num_classes):
        self.protein_ids = protein_ids
        self.embeddings = embeddings_dict
        self.labels = labels_dict
        self.num_classes = num_classes

    def __len__(self):
        return len(self.protein_ids)

    def __getitem__(self, idx):
        pid = self.protein_ids[idx]
        emb = self.embeddings[pid]
        label_indices = self.labels.get(pid, [])
        label_vec = torch.zeros(self.num_classes, dtype=torch.float32)
        if len(label_indices) > 0:
            label_vec[label_indices] = 1.0
        return emb, label_vec

# ================= 2. 增强监控工具 =================
def calculate_metrics(y_true, y_pred_logits, threshold=0.3):
    """
    计算多维度健康指标
    y_true: numpy array (N, Labels)
    y_pred_logits: numpy array (N, Labels)
    """
    # Sigmoid 转概率
    probs = 1 / (1 + np.exp(-y_pred_logits))
    preds = (probs > threshold).astype(int)
    
    # 1. Sample F1 (最接近 CAFA 评分)
    sample_f1 = f1_score(y_true, preds, average='samples', zero_division=0)
    
    # 2. Macro F1 (关注稀有类别表现)
    macro_f1 = f1_score(y_true, preds, average='macro', zero_division=0)
    
    # 3. Precision / Recall (查准 vs 查全)
    precision = precision_score(y_true, preds, average='samples', zero_division=0)
    recall = recall_score(y_true, preds, average='samples', zero_division=0)
    
    # 4. [关键监控] 正例率 (Positive Rate)
    # 监控模型预测了多少个 1。如果太低，说明模型在躺平；如果太高，说明在乱猜。
    # 正常的蛋白质通常有 10-50 个 GO Term，对于 One-hot 向量来说非常稀疏
    avg_pred_count = preds.sum(axis=1).mean()
    avg_true_count = y_true.sum(axis=1).mean()
    
    return {
        'f1_sample': sample_f1,
        'f1_macro': macro_f1,
        'precision': precision,
        'recall': recall,
        'avg_pred_terms': avg_pred_count, # 预测平均词数
        'avg_true_terms': avg_true_count  # 真实平均词数
    }

# ================= 3. 数据加载与预处理 =================
def load_data():
    print(">>> Loading Data & Computing IC Weights...")
    with open(PATHS['embeddings'], 'rb') as f:
        embeddings = pickle.load(f)
    
    df = pd.read_csv(PATHS['train_terms'], sep='\t')
    
    # 标签截断策略：覆盖 MAX_LABELS
    term_counts = df['term'].value_counts()
    selected_terms = term_counts.head(MAX_LABELS).index.tolist()
    term_to_idx = {term: i for i, term in enumerate(selected_terms)}
    num_classes = len(selected_terms)
    
    # 计算 IC 权重 (Information Content)
    # IC(t) = -log2(P(t))
    total_annots = len(df)
    counts = term_counts.head(MAX_LABELS).values
    probs = (counts + 1) / (total_annots + num_classes)
    ic_weights = -np.log2(probs)
    ic_weights = ic_weights / ic_weights.mean() # 归一化
    
    print(f"Target Labels: {num_classes}")
    print(f"IC Weights stats: Min={ic_weights.min():.2f}, Max={ic_weights.max():.2f}, Mean={ic_weights.mean():.2f}")
    
    # 构建映射
    labels_dict = {}
    valid_proteins = set(embeddings.keys())
    
    # 优化 Groupby 速度
    print("Mapping annotations...")
    # 只保留有效数据
    df = df[df['EntryID'].isin(valid_proteins) & df['term'].isin(set(selected_terms))]
    temp_dict = df.groupby('EntryID')['term'].apply(list).to_dict()
    
    for pid, terms in tqdm(temp_dict.items()):
        labels_dict[pid] = [term_to_idx[t] for t in terms]
        
    return embeddings, labels_dict, list(labels_dict.keys()), num_classes, ic_weights

# ================= 4. 主训练循环 (Pro版) =================
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device} (Batch Size: {BATCH_SIZE})")

    embeddings, labels_dict, all_pids, num_classes, ic_weights = load_data()
    
    # 划分数据集
    train_pids, val_pids = train_test_split(all_pids, test_size=0.1, random_state=42)
    
    train_dataset = CachedEmbeddingDataset(train_pids, embeddings, labels_dict, num_classes)
    val_dataset = CachedEmbeddingDataset(val_pids, embeddings, labels_dict, num_classes)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

    # 初始化
    model = ESM2Predictor(num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    
    # 学习率调度器: 当验证集 Loss 不再下降时，降低学习率
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)
    
    scaler = GradScaler()
    criterion = ICWeightedBCELoss(ic_weights, device)

    # 状态追踪
    best_val_f1 = 0.0
    patience_counter = 0
    logs = []

    print(f"\n🔥 Starting M2 Training (Optimized for L20 48GB)")
    print(f"{'Epoch':<6} | {'Loss':<8} | {'Val F1':<8} | {'Pre/Rec':<14} | {'Active Terms (Pred/True)':<25}")
    print("-" * 80)

    for epoch in range(EPOCHS):
        # --- Train ---
        model.train()
        train_loss = 0
        
        for batch_emb, batch_labels in tqdm(train_loader, desc=f"Ep {epoch+1} Train", leave=False):
            batch_emb, batch_labels = batch_emb.to(device, non_blocking=True), batch_labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            with autocast():
                outputs = model(batch_emb)
                loss = criterion(outputs, batch_labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)

        # --- Validation (Full Metrics) ---
        model.eval()
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for batch_emb, batch_labels in val_loader:
                batch_emb = batch_emb.to(device, non_blocking=True)
                outputs = model(batch_emb)
                # 收集 Logits 和 Labels 到 CPU 进行一次性计算
                val_preds.append(outputs.cpu().numpy())
                val_targets.append(batch_labels.numpy())
        
        # 拼接大矩阵
        val_preds = np.vstack(val_preds)
        val_targets = np.vstack(val_targets)
        
        # 计算详细指标
        metrics = calculate_metrics(val_targets, val_preds, threshold=0.3)
        
        # 打印监控面板
        print(f"{epoch+1:<6} | {avg_train_loss:.4f}   | {metrics['f1_sample']:.4f}   | "
              f"{metrics['precision']:.2f}/{metrics['recall']:.2f}   | "
              f"{metrics['avg_pred_terms']:.1f} / {metrics['avg_true_terms']:.1f}")

        # --- Scheduler & Checkpoint ---
        # 根据 F1 调整学习率
        scheduler.step(metrics['f1_sample'])
        
        # 保存最佳模型
        if metrics['f1_sample'] > best_val_f1:
            best_val_f1 = metrics['f1_sample']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'f1': best_val_f1,
            }, PATHS['model_save'])
            print(f"    >>> ⭐ New Best Model! Saved.")
            patience_counter = 0
        else:
            patience_counter += 1
            
        # 记录日志
        logs.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            **metrics
        })
        pd.DataFrame(logs).to_csv(PATHS['log_file'], index=False)
        
        # --- Early Stopping ---
        if patience_counter >= PATIENCE:
            print(f"\n🛑 Early stopping triggered at epoch {epoch+1}. Best F1: {best_val_f1:.4f}")
            break

    print("\n✅ Training Complete.")

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    train()