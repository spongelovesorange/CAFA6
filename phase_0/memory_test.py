import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import sys
import os

# --- 模拟 M2 (ESM-2 Predictor) 的架构 ---
# 这里的架构必须和你计划中 M2 的定义一致 (Listing 11)
class ESM2Predictor(nn.Module):
    def __init__(self, n_labels=40000, esm_embedding_dim=1280):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(esm_embedding_dim, 2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, n_labels)
        )

    def forward(self, cached_embedding_batch):
        return self.head(cached_embedding_batch)

def profile_memory_footprint():
    print("="*50)
    print("Task 0.4: GPU Memory Profiling (Stress Test)")
    print("="*50)
    
    device = 'cuda'
    if not torch.cuda.is_available():
        print("!! 错误: 未检测到 CUDA 设备！")
        return

    # 1. 模拟最坏情况下的数据加载
    # 假设 batch_size = 16 (计划书中的安全值，虽然我们在生成时用了256，但训练时显存压力更大)
    batch_size = 16 
    n_labels = 40000 # GO Term 的数量级
    embedding_dim = 1280 # ESM-2 650M 的维度
    
    print(f"Testing M2 (ESM-2 Predictor) Training Loop...")
    print(f"Configurations: Batch={batch_size}, Labels={n_labels}, Dim={embedding_dim}")

    # 2. 初始化模型
    try:
        model = ESM2Predictor(n_labels=n_labels, esm_embedding_dim=embedding_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters())
        scaler = GradScaler() # 混合精度训练必需
        
        print("Model loaded to GPU.")
    except Exception as e:
        print(f"!! 模型初始化失败: {e}")
        return

    # 3. 模拟训练步骤
    torch.cuda.reset_peak_memory_stats()
    
    # 生成虚拟数据
    dummy_embeddings = torch.randn(batch_size, embedding_dim).to(device)
    dummy_labels = torch.randn(batch_size, n_labels).to(device) # 使用 float 模拟 logits
    
    print("Starting forward/backward pass...")
    
    try:
        # 开启混合精度 (Mixed Precision)
        with autocast():
            outputs = model(dummy_embeddings)
            # 使用 BCEWithLogitsLoss 模拟多标签分类损失
            loss = F.binary_cross_entropy_with_logits(outputs, dummy_labels)
        
        # 反向传播
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # 4. 检查显存峰值
        peak_memory = torch.cuda.max_memory_allocated() / 1e9
        print("-" * 30)
        print(f"✅ M2 Peak Memory Usage: {peak_memory:.2f} GB")
        print("-" * 30)
        
        # 5. 验证结果
        limit = 46.0 # L20 48GB 留一点余量
        if peak_memory < limit:
            print(f"[PASS] 显存占用在安全范围内 (< {limit} GB)。")
            print("结论: Phase 1 M2 训练可以安全启动。")
        else:
            print(f"!! [CRITICAL WARNING] 显存超出安全阈值！")
            print("建议: 减小训练时的 batch_size 或检查模型结构。")

    except RuntimeError as e:
        print(f"!! [OOM ERROR] 显存溢出: {e}")
    
    # 清理
    del model, optimizer, scaler, dummy_embeddings, dummy_labels
    torch.cuda.empty_cache()

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    profile_memory_footprint()