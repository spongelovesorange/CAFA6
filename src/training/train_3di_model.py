import os
# === 显卡设置：使用空闲的 GPU 0 ===
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from Bio import SeqIO
from tqdm import tqdm
import pickle
import math
import sys

# ================= ⚙️ 配置区域 =================
# 路径配置
TRAIN_3DI_FASTA = "data/Features_3Di/train_3di.fasta" 
TRAIN_TERMS = "data/Train/train_terms.tsv"
LABEL_MAP = "models/checkpoints_esm2_3b_qlora/label_map.pkl"
SAVE_DIR = "models/checkpoints_3di_transformer"

# 数据集划分
SPLIT_TRAIN_IDX = "folds/train_ids_final.npy"
SPLIT_VALID_IDX = "folds/valid_ids_final.npy"

# 模型超参数
MAX_LEN = 1024       
BATCH_SIZE = 128     # L20 显存大，直接拉满
LR = 5e-4            
EPOCHS = 15          # 结构特征收敛快，15轮足够
EMBED_DIM = 256      
NUM_HEADS = 4        
NUM_LAYERS = 4       
DROPOUT = 0.1

# 3Di 词汇表 (FoldSeek 标准)
VOCAB_3DI = "dpqvlcaskghnmwtryfei"
CHAR_TO_IDX = {c: i+1 for i, c in enumerate(VOCAB_3DI)} # 0 is Padding

# ================= 🛠️ 工具函数 =================
def smart_load_map(path):
    """ 能够处理各种诡异 pickle 结构的加载器 """
    print(f"📂 读取 Label Map: {path}")
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict): return data
    if isinstance(data, (list, tuple)):
        # 尝试寻找里面的字典
        for item in data:
            if isinstance(item, dict) and len(item) > 0:
                # 检查 key 是否为 str (GO term)
                k = next(iter(item.keys()))
                if isinstance(k, str): return item
    raise ValueError(f"❌ 无法从 {path} 解析出 Label Map (dict)")

# ================= 🧠 模型定义 =================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x): return x + self.pe[:x.size(1), :]

class StructTransformer(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.embedding = nn.Embedding(len(VOCAB_3DI)+1, EMBED_DIM, padding_idx=0)
        self.pos_encoder = PositionalEncoding(EMBED_DIM, MAX_LEN)
        encoder_layer = nn.TransformerEncoderLayer(d_model=EMBED_DIM, nhead=NUM_HEADS, dim_feedforward=EMBED_DIM*4, dropout=DROPOUT, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)
        self.fc = nn.Linear(EMBED_DIM, num_labels)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        key_padding_mask = (x == 0)
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)
        
        # Mean Pooling (Masked)
        mask_expanded = (~key_padding_mask).unsqueeze(-1).float()
        x = x * mask_expanded
        sum_embeddings = x.sum(dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        x = sum_embeddings / sum_mask
        
        x = self.dropout(x)
        logits = self.fc(x)
        return logits

# ================= 📂 数据加载 =================
class StructureDataset(Dataset):
    def __init__(self, fasta_path, target_ids, id2labels):
        self.data = []
        print(f"📖 解析 3Di Fasta: {fasta_path}")
        if not os.path.exists(fasta_path):
            raise FileNotFoundError(f"找不到 {fasta_path}！")
            
        # 建立快速查找集
        target_ids_set = set(target_ids)
        
        for record in SeqIO.parse(fasta_path, "fasta"):
            pid = str(record.id).strip()
            if "|" in pid: pid = pid.split("|")[1]
            
            if pid in target_ids_set and pid in id2labels:
                seq_str = str(record.seq).lower().replace("\n", "").strip()
                indices = [CHAR_TO_IDX.get(c, 0) for c in seq_str][:MAX_LEN]
                if len(indices) < MAX_LEN: indices += [0] * (MAX_LEN - len(indices))
                
                self.data.append({
                    "input_ids": np.array(indices, dtype=np.int64),
                    "labels": id2labels[pid]
                })
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.data[idx]['input_ids']),
            "labels": torch.tensor(self.data[idx]['labels'], dtype=torch.float32)
        }

# ================= 🚀 训练主流程 =================
def train():
    # 1. 加载 Label Map
    term2idx = smart_load_map(LABEL_MAP)
    num_classes = len(term2idx)
    print(f"🎯 目标分类数: {num_classes}")

    # 2. 准备 One-Hot 标签
    print("构建标签矩阵...")
    terms_df = pd.read_csv(TRAIN_TERMS, sep="\t", dtype={'EntryID': str})
    terms_df['EntryID'] = terms_df['EntryID'].astype(str).str.strip()
    
    train_ids = set(str(x).strip() for x in np.load(SPLIT_TRAIN_IDX, allow_pickle=True))
    valid_ids = set(str(x).strip() for x in np.load(SPLIT_VALID_IDX, allow_pickle=True))
    
    id2labels = {}
    filtered_terms = terms_df[terms_df['term'].isin(term2idx.keys())]
    for entry, group in tqdm(filtered_terms.groupby('EntryID'), desc="Grouping"):
        if entry in train_ids or entry in valid_ids:
            idxs = [term2idx[t] for t in group['term']]
            vec = np.zeros(num_classes, dtype=np.float32)
            vec[idxs] = 1.0
            id2labels[entry] = vec

    # 3. DataLoader
    train_ds = StructureDataset(TRAIN_3DI_FASTA, train_ids, id2labels)
    valid_ds = StructureDataset(TRAIN_3DI_FASTA, valid_ids, id2labels)
    
    if len(train_ds) == 0: 
        print("❌ 错误：训练集为空！"); return

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    valid_dl = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    # 4. 模型初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training on {device}")
    model = StructTransformer(num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    best_val_loss = float('inf')

    # 5. Loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        pbar = tqdm(train_dl, desc=f"Ep {epoch+1}/{EPOCHS}")
        
        for batch in pbar:
            inputs = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in valid_dl:
                inputs = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                val_loss += criterion(model(inputs), labels).item()
        
        avg_val_loss = val_loss / len(valid_dl)
        print(f"Epoch {epoch+1} | Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"{SAVE_DIR}/best_model.pth")
            print(f"💾 Saved Best Model (Loss: {best_val_loss:.4f})")

if __name__ == "__main__":
    train()