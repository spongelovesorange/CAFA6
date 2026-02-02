import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from Bio import SeqIO
from tqdm import tqdm
import pickle
import numpy as np
import math

# === 配置 ===
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # 用空闲的卡
TEST_3DI_FASTA = "data/Features_3Di/test_3di.fasta" # 必须是生成好的测试集 3Di
MODEL_PATH = "models/checkpoints_3di_transformer/best_model.pth"
LABEL_MAP = "models/checkpoints_esm2_3b_qlora/label_map.pkl"
OUTPUT_CSV = "data/Predictions/3di_preds.csv"
MAX_LEN = 1024
BATCH_SIZE = 128
EMBED_DIM = 256
NUM_HEADS = 4
NUM_LAYERS = 4
VOCAB_3DI = "dpqvlcaskghnmwtryfei"
CHAR_TO_IDX = {c: i+1 for i, c in enumerate(VOCAB_3DI)}

# === 模型定义 (必须与训练一致) ===
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
        encoder_layer = nn.TransformerEncoderLayer(d_model=EMBED_DIM, nhead=NUM_HEADS, dim_feedforward=EMBED_DIM*4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)
        self.fc = nn.Linear(EMBED_DIM, num_labels)
    def forward(self, x):
        key_padding_mask = (x == 0)
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)
        mask_expanded = (~key_padding_mask).unsqueeze(-1).float()
        x = x * mask_expanded
        x = x.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
        return self.fc(x)

# === 推理逻辑 ===
def predict():
    # 1. 加载 Label Map
    with open(LABEL_MAP, "rb") as f:
        data = pickle.load(f)
    # 简单粗暴的识别
    if isinstance(data, dict): term2idx = data
    else: 
        for item in data:
            if isinstance(item, dict) and isinstance(list(item.keys())[0], str):
                term2idx = item; break
    num_classes = len(term2idx)
    print(f"标签数: {num_classes}")

    # 2. 加载数据
    data = []
    ids = []
    print("读取 FASTA...")
    for record in SeqIO.parse(TEST_3DI_FASTA, "fasta"):
        pid = str(record.id).strip()
        if "|" in pid: pid = pid.split("|")[1]
        ids.append(pid)
        seq_str = str(record.seq).lower().replace("\n", "").strip()
        indices = [CHAR_TO_IDX.get(c, 0) for c in seq_str][:MAX_LEN]
        if len(indices) < MAX_LEN: indices += [0] * (MAX_LEN - len(indices))
        data.append(np.array(indices))
    
    print(f"待预测序列: {len(data)}")
    data_tensor = torch.tensor(np.array(data), dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(data_tensor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 3. 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StructTransformer(num_classes).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # 4. 预测
    print("开始预测...")
    with open(OUTPUT_CSV, "w") as f:
        f.write("id,logits\n") # Header
        
    with open(OUTPUT_CSV, "a") as f:
        with torch.no_grad():
            batch_idx = 0
            for batch in tqdm(loader):
                inputs = batch[0].to(device)
                logits = model(inputs).cpu().numpy()
                
                # 写入
                buffer = []
                start_id = batch_idx * BATCH_SIZE
                current_ids = ids[start_id : start_id + len(logits)]
                
                for pid, row in zip(current_ids, logits):
                    logit_str = ",".join([f"{x:.4f}" for x in row])
                    buffer.append(f"{pid},{logit_str}\n")
                f.write("".join(buffer))
                batch_idx += 1

    print(f"🎉 预测完成！结果保存在 {OUTPUT_CSV}")

if __name__ == "__main__":
    predict()