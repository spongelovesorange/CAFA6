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
from sklearn.metrics import f1_score

# ================= 配置区域 (Configuration) =================
# [优化] 针对 L20 48GB 的激进配置
BATCH_SIZE = 2048      # 显存只有 1.7G 占用，我们可以大胆开到 2048 甚至 4096
LR = 1e-3              # 大 Batch Size 通常配合稍大的学习率
EPOCHS = 50
EMBEDDING_DIM = 1280   # ESM-2 650M 的维度
NUM_LABELS = 1500      # 演示用：取频率最高的 Top 1500 个 GO Term (CAFA通常只评估高频词)
                       # 实际比赛中你可能需要预测 3000-5000 个，视你的显存和策略而定

PATHS = {
    'embeddings': './cache/esm2-650M_embeddings.pkl', # 你刚刚生成的文件
    'train_terms': 'data/Train/train_terms.tsv',      # 比赛提供的标签文件
    'model_save': './models/m2_esm2_mlp.pth'
}
os.makedirs('./models', exist_ok=True)

# ================= 1. 数据集定义 (Dataset) =================
class CachedEmbeddingDataset(Dataset):
    """直接从内存读取缓存的 Dataset"""
    def __init__(self, protein_ids, embeddings_dict, labels_dict, num_classes):
        self.protein_ids = protein_ids
        self.embeddings = embeddings_dict
        self.labels = labels_dict
        self.num_classes = num_classes

    def __len__(self):
        return len(self.protein_ids)

    def __getitem__(self, idx):
        pid = self.protein_ids[idx]
        # 获取 Embedding (1280,)
        emb = self.embeddings[pid] 
        
        # 获取 Label (Multi-hot encoding)
        label_indices = self.labels.get(pid, [])
        label_vec = torch.zeros(self.num_classes, dtype=torch.float32)
        if len(label_indices) > 0:
            label_vec[label_indices] = 1.0
            
        return emb, label_vec

# ================= 2. 模型定义 (Model) [基于 Listing 11] =================
class ESM2Predictor(nn.Module):
    def __init__(self, n_labels, esm_embedding_dim=1280):
        super().__init__()
        # 简单的 MLP 头，计划书中的设计
        self.head = nn.Sequential(
            nn.Linear(esm_embedding_dim, 2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, n_labels)
        )

    def forward(self, x):
        return self.head(x)

# ================= 3. 辅助函数：加载数据 =================
def load_data():
    print(f"Loading embeddings from {PATHS['embeddings']}...")
    with open(PATHS['embeddings'], 'rb') as f:
        embeddings = pickle.load(f)
    print(f"Loaded {len(embeddings)} embeddings.")

    print(f"Loading labels from {PATHS['train_terms']}...")
    # 读取 CAFA 提供的 train_terms.tsv
    # 格式通常是: Protein_ID, GO_Term, Aspect
    df = pd.read_csv(PATHS['train_terms'], sep='\t')
    
    # 筛选 Top N 高频 GO Terms 作为训练目标
    top_terms = df['term'].value_counts().head(NUM_LABELS).index.tolist()
    term_to_idx = {term: i for i, term in enumerate(top_terms)}
    
    print(f"Selected Top {NUM_LABELS} frequent GO terms for training.")
    
    # 构建 Protein -> Label Indices 的映射
    labels_dict = {}
    # 只保留有 embedding 的蛋白质
    valid_proteins = set(embeddings.keys())
    
    # 过滤数据：只保留我们关心的 Top Terms 和 有 Embedding 的蛋白质
    df_filtered = df[df['term'].isin(set(top_terms)) & df['EntryID'].isin(valid_proteins)]
    
    for pid, group in tqdm(df_filtered.groupby('EntryID'), desc="Grouping Labels"):
        indices = [term_to_idx[t] for t in group['term']]
        labels_dict[pid] = indices
        
    # 获取最终用于训练的 ID 列表 (即有 Embedding 也有 Label 的交集)
    train_pids = list(labels_dict.keys())
    print(f"Final training set size: {len(train_pids)} proteins.")
    
    return embeddings, labels_dict, train_pids, term_to_idx

# ================= 4. 主训练循环 =================
def train():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. 准备数据
    embeddings, labels_dict, all_pids, term_mapping = load_data()
    
    # 简单划分训练集/验证集 (80/20) - 对应 Task 0.1 的简化版
    train_pids, val_pids = train_test_split(all_pids, test_size=0.2, random_state=42)
    
    train_dataset = CachedEmbeddingDataset(train_pids, embeddings, labels_dict, NUM_LABELS)
    val_dataset = CachedEmbeddingDataset(val_pids, embeddings, labels_dict, NUM_LABELS)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # 2. 初始化模型
    model = ESM2Predictor(n_labels=NUM_LABELS, esm_embedding_dim=EMBEDDING_DIM).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    
    # 损失函数 (BCEWithLogitsLoss 自带 Sigmoid，数值更稳定)
    criterion = nn.BCEWithLogitsLoss() 
    scaler = GradScaler() # 混合精度

    best_val_loss = float('inf')

    print("\n" + "="*30)
    print("🔥 Starting Phase 1: M2 Training")
    print(f"Batch Size: {BATCH_SIZE}")
    print("="*30 + "\n")

    for epoch in range(EPOCHS):
        # --- 训练阶段 ---
        model.train()
        train_loss = 0
        for batch_emb, batch_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
            batch_emb, batch_labels = batch_emb.to(device), batch_labels.to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(batch_emb)
                loss = criterion(outputs, batch_labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)

        # --- 验证阶段 ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_emb, batch_labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                batch_emb, batch_labels = batch_emb.to(device), batch_labels.to(device)
                outputs = model(batch_emb)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

        # --- 保存最佳模型 ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), PATHS['model_save'])
            print(f"  --> New Best Model Saved! (Loss: {best_val_loss:.4f})")

    print("\n✅ M2 Training Complete!")

if __name__ == "__main__":
    train()