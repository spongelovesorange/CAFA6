import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # 保持单卡独占

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType
import pandas as pd
import numpy as np
from Bio import SeqIO
from tqdm import tqdm
import pickle
import random

# === 核心配置 ===
MODEL_PATH = "models/esm2_t36_3B_UR50D"
SAVE_DIR = "models/checkpoints_esm2_3b_asl"
TRAIN_FASTA = "data/Train/train_sequences.fasta"
TRAIN_TERMS = "data/Train/train_terms.tsv"
SPLIT_TRAIN_IDX = "folds/train_ids_final.npy"
SPLIT_VALID_IDX = "folds/valid_ids_final.npy"

# === ⚡ 极速配置 ===
# 开启 Gradient Checkpointing 后，BS=16 是非常安全的
BATCH_SIZE = 16          
GRAD_ACCUMULATION = 2    # 等效 BS = 32
LR = 1e-4
EPOCHS = 8
TOP_K = 3000
MAX_LEN = 1024           

# === 🛡️ 核心修复：自定义整理器 (Custom Collator) ===
# 专门解决 DataCollator 乱动 labels 的问题
class CustomCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        # 1. 拆分数据：文本归文本，标签归标签
        inputs = [{"input_ids": item["input_ids"], "attention_mask": item["attention_mask"]} for item in batch]
        labels = [item["labels"] for item in batch]

        # 2. 文本动态填充 (Dynamic Padding)
        # 只让 Tokenizer 处理 input_ids 和 attention_mask
        batch_out = self.tokenizer.pad(inputs, padding="longest", return_tensors="pt")

        # 3. 标签直接堆叠 (Stack)
        # 既然 labels 已经是 tensor 且长度固定 (3000)，直接堆叠最安全
        batch_out["labels"] = torch.stack(labels)

        return batch_out

# === Asymmetric Loss ===
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=0, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            loss = -torch.pow(1 - (xs_pos * y + xs_neg * (1 - y)), self.gamma_pos * y + self.gamma_neg * (1 - y)) * (los_pos + los_neg)
        else:
            loss = -(los_pos + los_neg)
        return loss.sum()

# === 智能长度采样器 ===
class LengthGroupedSampler(Sampler):
    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size
        self.lengths = [len(x['input_ids']) for x in data_source]
        
    def __iter__(self):
        indices = np.argsort(self.lengths)
        batches = [indices[i:i + self.batch_size] for i in range(0, len(indices), self.batch_size)]
        if len(batches[-1]) < self.batch_size:
            batches = batches[:-1]
        random.shuffle(batches) 
        return iter([idx for batch in batches for idx in batch])

    def __len__(self):
        return len(self.data_source) // self.batch_size * self.batch_size

class ProteinDataset(Dataset):
    def __init__(self, fasta_file, target_ids, id2labels, tokenizer):
        self.data = []
        self.tokenizer = tokenizer
        print(f"解析 FASTA: {fasta_file} ...")
        target_ids_set = set(target_ids)
        for record in SeqIO.parse(fasta_file, "fasta"):
            pid = record.id.split("|")[1] if "|" in record.id else record.id
            if pid in target_ids_set and pid in id2labels:
                seq = str(record.seq)[:MAX_LEN]
                # 预先 Tokenize，但不填充(Padding=False)
                enc = tokenizer(seq, truncation=True, max_length=MAX_LEN, padding=False)
                
                self.data.append({
                    "input_ids": enc['input_ids'], 
                    "attention_mask": enc['attention_mask'],
                    "labels": id2labels[pid]
                })

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "input_ids": item['input_ids'],
            "attention_mask": item['attention_mask'],
            # ✅ 显式转为 Tensor (float32)，配合 CustomCollator 使用
            "labels": torch.tensor(item['labels'], dtype=torch.float32)
        }

def train():
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 1. Label Map
    terms_df = pd.read_csv(TRAIN_TERMS, sep="\t", dtype={'EntryID': str})
    terms_df['EntryID'] = terms_df['EntryID'].str.strip()
    top_terms = terms_df['term'].value_counts().head(TOP_K).index.tolist()
    term2idx = {t: i for i, t in enumerate(top_terms)}
    
    with open(f"{SAVE_DIR}/label_map.pkl", "wb") as f:
        pickle.dump(term2idx, f)
        
    train_ids = np.load(SPLIT_TRAIN_IDX, allow_pickle=True)
    valid_ids = np.load(SPLIT_VALID_IDX, allow_pickle=True)
    train_ids_set = set(str(x).strip() for x in train_ids)
    valid_ids_set = set(str(x).strip() for x in valid_ids)
    
    print("构建标签矩阵...")
    id2labels = {}
    filtered_df = terms_df[terms_df['term'].isin(set(top_terms))]
    for pid, group in tqdm(filtered_df.groupby('EntryID')):
        if pid in train_ids_set or pid in valid_ids_set:
            lbl = np.zeros(TOP_K, dtype=np.float32)
            indices = [term2idx[t] for t in group['term']]
            lbl[indices] = 1.0
            id2labels[pid] = lbl

    # 2. Model
    print("加载 ESM-2 (Eager Mode)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH, 
        num_labels=TOP_K, 
        torch_dtype=torch.bfloat16, 
        device_map="cuda:0", 
        attn_implementation="eager"
    )
    
    # ✅ 必须开启 Checkpointing，否则 BS=16 会 OOM
    base_model.gradient_checkpointing_enable() 
    print("⚡ Gradient Checkpointing: ON")
    
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, 
        r=16, lora_alpha=32, lora_dropout=0.1,
        target_modules=["query", "key", "value", "dense", "up_proj", "down_proj"],
        modules_to_save=["classifier"] 
    )
    model = get_peft_model(base_model, peft_config)
    
    train_ds = ProteinDataset(TRAIN_FASTA, train_ids_set, id2labels, tokenizer)
    valid_ds = ProteinDataset(TRAIN_FASTA, valid_ids_set, id2labels, tokenizer)
    
    # ✅ 使用自定义整理器
    my_collator = CustomCollator(tokenizer)
    train_sampler = LengthGroupedSampler(train_ds, BATCH_SIZE)
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        sampler=train_sampler, 
        collate_fn=my_collator, # 替换为自定义的
        num_workers=4, 
        pin_memory=True
    )
    
    valid_loader = DataLoader(
        valid_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        collate_fn=my_collator, # 替换为自定义的
        num_workers=4, 
        pin_memory=True
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = AsymmetricLoss(gamma_neg=2, gamma_pos=0, clip=0.05)
    best_val_loss = float('inf')
    
    print(f"🚀 冲刺模式 (BS={BATCH_SIZE}, Dynamic Padding, Checkpointing=ON)...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for step, batch in enumerate(pbar):
            input_ids = batch['input_ids'].cuda()
            mask = batch['attention_mask'].cuda()
            labels = batch['labels'].cuda()
            
            outputs = model(input_ids, attention_mask=mask)
            
            loss = criterion(outputs.logits, labels)
            (loss / GRAD_ACCUMULATION).backward()
            
            if (step+1) % GRAD_ACCUMULATION == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                
            train_loss += loss.item()
            
            current_len = input_ids.shape[1]
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'len': current_len})
        
        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in valid_loader:
                input_ids = batch['input_ids'].cuda()
                mask = batch['attention_mask'].cuda()
                labels = batch['labels'].cuda()
                outputs = model(input_ids, attention_mask=mask)
                val_loss += criterion(outputs.logits, labels).item()
        
        avg_val = val_loss / len(valid_loader)
        print(f"Epoch {epoch+1} Valid Loss: {avg_val:.4f}")
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            model.save_pretrained(f"{SAVE_DIR}/best_checkpoint")
            print(f"💾 Loss 创新低 ({best_val_loss:.4f})")

if __name__ == "__main__":
    train()