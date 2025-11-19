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

# ================= HYPER 配置 (L20 48GB 专用) =================
BATCH_SIZE = 4096      
LR = 2e-3              
EPOCHS = 50
PATIENCE = 8           
MAX_LABELS = 26125     

PATHS = {
    'embeddings': './cache/esm2-650M_embeddings.pkl',
    'train_terms': 'data/Train/train_terms.tsv',
    'model_save': './models/m2_esm2_hyper.pth',
    'vocab_save': './models/vocab.pkl',          # <--- 新增：保存词表路径
    'log_file': './models/training_log.csv'
}
os.makedirs('./models', exist_ok=True)

# ================= 1. 核心组件 =================
class ICWeightedBCELoss(nn.Module):
    def __init__(self, ic_weights, device='cuda'):
        super().__init__()
        self.ic_weights = torch.tensor(ic_weights).float().to(device)

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        weighted_bce = bce_loss * self.ic_weights.unsqueeze(0)
        return weighted_bce.mean()

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

# ================= 2. GPU 加速指标计算 =================
def calculate_metrics_gpu(y_true_tensor, y_logits_tensor):
    probs = torch.sigmoid(y_logits_tensor)
    best_f1 = 0.0
    best_metrics = {}
    
    # 模拟 CAFA 阈值扫描
    thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
    
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

# ================= 3. 数据加载 (关键修正：保存 Vocab) =================
def load_data_to_gpu(device):
    print(">>> Loading Data to GPU Memory...")
    with open(PATHS['embeddings'], 'rb') as f:
        embeddings_dict = pickle.load(f)
    
    df = pd.read_csv(PATHS['train_terms'], sep='\t')
    
    # 计算 Top Terms
    term_counts = df['term'].value_counts()
    selected_terms = term_counts.head(MAX_LABELS).index.tolist()
    
    # --- 🔥 关键修正：保存这个列表供推理使用 ---
    print(f">>> Saving vocabulary ({len(selected_terms)} terms) to {PATHS['vocab_save']}...")
    with open(PATHS['vocab_save'], 'wb') as f:
        pickle.dump(selected_terms, f)
    # ---------------------------------------

    term_to_idx = {term: i for i, term in enumerate(selected_terms)}
    num_classes = len(selected_terms)
    
    # 计算 IC 权重
    total = len(df)
    counts = term_counts.head(MAX_LABELS).values
    probs = (counts + 1) / (total + num_classes)
    ic_weights = -np.log2(probs)
    ic_weights = ic_weights / ic_weights.mean()
    
    # 过滤有效数据
    valid_proteins = set(embeddings_dict.keys())
    df = df[df['EntryID'].isin(valid_proteins) & df['term'].isin(set(selected_terms))]
    
    temp_dict = df.groupby('EntryID')['term'].apply(list).to_dict()
    all_pids = list(temp_dict.keys())
    
    # 构建 Tensor
    feature_list = []
    label_list = []
    for pid in tqdm(all_pids, desc="Preparing Tensors"):
        feature_list.append(embeddings_dict[pid])
        label_list.append([term_to_idx[t] for t in temp_dict[pid]])
        
    X_tensor = torch.stack(feature_list).to(device)
    print(f"Data Loaded on GPU: {X_tensor.shape}")
    
    return X_tensor, label_list, num_classes, ic_weights

# ================= 4. 主训练循环 =================
def train():
    device = torch.device('cuda')
    print(f"🔥 HYPER Training Mode (Batch: {BATCH_SIZE})")

    X_tensor, label_list, num_classes, ic_weights = load_data_to_gpu(device)
    
    indices = np.arange(len(label_list))
    train_idx, val_idx = train_test_split(indices, test_size=0.1, random_state=42)
    
    train_dataset = UltraFastDataset(X_tensor[train_idx], [label_list[i] for i in train_idx], num_classes)
    val_dataset = UltraFastDataset(X_tensor[val_idx], [label_list[i] for i in val_idx], num_classes)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = ESM2Predictor(num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    scaler = GradScaler()
    criterion = ICWeightedBCELoss(ic_weights, device)

    best_val_f1 = 0.0
    patience_counter = 0
    
    print("\n🚀 Training Start")
    print(f"{'Epoch':<6} | {'Loss':<8} | {'Val F1':<8} | {'Pre/Rec':<12} | {'Pred/True':<12} | {'Best Thresh'}")
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
        
        scheduler.step(metrics['f1'])
        if metrics['f1'] > best_val_f1:
            best_val_f1 = metrics['f1']
            torch.save(model.state_dict(), PATHS['model_save'])
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"🛑 Early Stopping. Best F1: {best_val_f1:.4f}")
                break

if __name__ == "__main__":
    train()