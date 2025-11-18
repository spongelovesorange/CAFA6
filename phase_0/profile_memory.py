import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import os

# 确保 PyTorch 使用 GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class ESM2Predictor(nn.Module):
    def __init__(self, n_labels, esm_embedding_dim=1280):
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
    """测试 M2 模型的近似最坏情况内存使用"""
    
    device = 'cuda' 

    try:
        # --- Test M2 (ESM-2 预测器) ---
        print("Testing M2 (ESM-2 predictor)...")
        
        # 假设 40000 个标签
        model = ESM2Predictor(n_labels=40000).to(device)
        optimizer = torch.optim.Adam(model.parameters())
        scaler = GradScaler()
        
        # 你的计划中定义的实际批量大小 (来自 Listing 12)
        actual_batch_size = 16 
        
        # 重置峰值内存统计数据
        torch.cuda.reset_peak_memory_stats(device)
        
        # 模拟数据
        dummy_embeddings = torch.randn(actual_batch_size, 1280).to(device)
        dummy_labels = torch.randn(actual_batch_size, 40000).to(device)
        
        print(f"Simulating training step with batch size {actual_batch_size}...")
        
        # 模拟训练步骤 (混合精度)
        with autocast():
            outputs = model(dummy_embeddings)
            loss = F.binary_cross_entropy_with_logits(outputs, dummy_labels)
        
        # 反向传播
        scaler.scale(loss).backward()
        
        # 获取峰值内存 (GB)
        peak_memory = torch.cuda.max_memory_allocated(device) / 1e9
        
        print(f"M2 Peak Memory: {peak_memory:.2f} GB")
        
        # 关键断言：必须小于 L20 的容量 (计划中设为 46GB)
        assert peak_memory < 46.0, f"CRITICAL: M2 Memory ({peak_memory:.2f} GB) exceeds L20 capacity!"
        
        del model, optimizer, scaler, dummy_embeddings, dummy_labels
        torch.cuda.empty_cache()
        
        print("\n[PASS] All memory tests passed!")

    except AssertionError as e:
        print(f"\n[FAIL] Memory Test Failed: {e}")
    except Exception as e:
        print(f"\n[ERROR] An error occurred during memory profiling: {e}")

# 运行分析
if __name__ == "__main__":
    profile_memory_footprint()