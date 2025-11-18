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

# ================= 配置区域 =================
# [优化] 保持大 Batch Size (基于 Task 0.4 显存测试结果)
BATCH_SIZE = 2048  
LR = 1e-3          
EPOCHS = 50
EMBEDDING_DIM = 1280   

# [计划要求] 尽可能覆盖所有 Terms，Listing 12 建议 40000
# 实际数据中可能只有 31000 左右个唯一 GO Term
MAX_LABELS = 40000 

PATHS = {
    'embeddings': './cache/esm2-650M_embeddings.pkl',
    'train_terms': 'data/Train/train_terms.tsv',
    'model_save': './models/m2_esm2_strict.pth'
}
os.makedirs('./models', exist_ok=True)

# ================= 1. 核心组件：符合计划的 Loss 函数 =================
# [Source: Listing 11 in CAFA 6 Project Plan]
class ICWeightedBCELoss(nn.Module):
    """
    Binary cross-entropy weighted by Information Content (IC).
    计划书中明确要求的 Loss，用于提升加权 F1 分数。
    """
    def __init__(self, ic_weights, device='cuda'):
        super().__init__()
        # ic_weights 应该是一个形状为 [n_labels] 的 tensor
        self.ic_weights = torch.tensor(ic_weights).float().to(device)

    def forward(self, logits, targets):
        # standard BCE (no reduction yet)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # [计划核心逻辑] 加权: loss * ic_weight
        # 扩展权重维度以匹配 batch: [1, n_labels]
        weighted_bce = bce_loss * self.ic_weights.unsqueeze(0)
        
        return weighted_bce.mean()

# ================= 2. 模型定义 =================
# [Source: Listing 11 in CAFA 6 Project Plan]
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

# ================= 3. 数据处理 =================
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
        
        # Create Multi-hot Label
        label_vec = torch.zeros(self.num_classes, dtype=torch.float32)
        if len(label_indices) > 0:
            label_vec[label_indices] = 1.0
        return emb, label_vec

def load_and_process_data():
    print(f"Loading embeddings...")
    with open(PATHS['embeddings'], 'rb') as f:
        embeddings = pickle.load(f)
    
    print(f"Loading annotations...")
    df = pd.read_csv(PATHS['train_terms'], sep='\t')
    
    # 1. 确定 Label 空间 (覆盖前 N 个最常见的词)
    term_counts = df['term'].value_counts()
    selected_terms = term_counts.head(MAX_LABELS).index.tolist()
    term_to_idx = {term: i for i, term in enumerate(selected_terms)}
    num_classes = len(selected_terms)
    
    print(f"Target Labels: {num_classes} (Coverage of annotations: {term_counts.head(MAX_LABELS).sum() / term_counts.sum():.2%})")

    # 2. 计算简易 IC 权重 (Information Content)
    # IC(t) = -log2(P(t)), 这里用频率代替概率 P(t) = count(t) / total_proteins
    # 越罕见的词，权重越高
    print("Computing IC weights...")
    total_annots = len(df)
    counts = term_counts.head(MAX_LABELS).values
    # 加上平滑项防止 log(0)
    probs = (counts + 1) / (total_annots + num_classes) 
    ic_weights = -np.log2(probs)
    
    # 归一化权重，防止 Loss 爆炸
    ic_weights = ic_weights / ic_weights.mean()
    
    # 3. 构建 Protein -> Label 映射
    labels_dict = {}
    valid_proteins = set(embeddings.keys())
    
    # 只处理有 Embedding 的数据
    df_filtered = df[df['EntryID'].isin(valid_proteins) & df['term'].isin(set(selected_terms))]
    
    # 快速分组
    # 使用 pandas group 可能会慢，这里用简单的循环优化
    temp_dict = df_filtered.groupby('EntryID')['term'].apply(list).to_dict()
    
    for pid, terms in tqdm(temp_dict.items(), desc="Mapping Labels"):
        indices = [term_to_idx[t] for t in terms]
        labels_dict[pid] = indices
        
    train_pids = list(labels_dict.keys())
    print(f"Training samples: {len(train_pids)}")
    
    return embeddings, labels_dict, train_pids, num_classes, ic_weights

# ================= 4. 主训练程序 =================
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. 准备数据
    embeddings, labels_dict, all_pids, num_classes, ic_weights = load_and_process_data()
    
    train_pids, val_pids = train_test_split(all_pids, test_size=0.1, random_state=42)
    
    train_dataset = CachedEmbeddingDataset(train_pids, embeddings, labels_dict, num_classes)
    val_dataset = CachedEmbeddingDataset(val_pids, embeddings, labels_dict, num_classes)
    
    # Pin_memory=True 加速数据传输
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

    # 2. 模型与 Loss
    model = ESM2Predictor(n_labels=num_classes, esm_embedding_dim=EMBEDDING_DIM).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = GradScaler()

    # [Strict Compliance] 使用加权 Loss
    print("Initializing ICWeightedBCELoss...")
    criterion = ICWeightedBCELoss(ic_weights, device=device)

    best_val_loss = float('inf')

    print(f"\n🔥 M2 Strict Training (Labels={num_classes}, Batch={BATCH_SIZE})")
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        for batch_emb, batch_labels in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            batch_emb = batch_emb.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(batch_emb)
                loss = criterion(outputs, batch_labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_emb, batch_labels in val_loader:
                batch_emb = batch_emb.to(device, non_blocking=True)
                batch_labels = batch_labels.to(device, non_blocking=True)
                outputs = model(batch_emb)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()
        
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}: Train Loss={avg_train:.4f}, Val Loss={avg_val:.4f}")
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), PATHS['model_save'])
            print(f"  --> Model Saved (Val Loss: {best_val_loss:.4f})")

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    train()