import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
import os

# --- 配置 (请确保路径正确) ---
# [!! 已修正!!] 告诉脚本在 data/ 目录中查找文件
CLUSTER_FILE = 'data/clusterRes_cluster.tsv' 
FASTA_FILE = 'data/Train/train_sequences.fasta'
N_SPLITS = 3
OUTPUT_DIR = 'folds' # 将在此处创建新目录

# --- 辅助函数 (来自你的计划) ---
def save_fold(fold_idx, train_idx, val_idx):
    """保存训练和验证索引到 .npy 文件"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    train_path = os.path.join(OUTPUT_DIR, f'fold_{fold_idx}_train_idx.npy')
    val_path = os.path.join(OUTPUT_DIR, f'fold_{fold_idx}_val_idx.npy')
    
    np.save(train_path, train_idx)
    np.save(val_path, val_idx)
    
    print(f"Saved Fold {fold_idx}: Train={len(train_idx)}, Val={len(val_idx)}")
    print(f"  -> {train_path}")
    print(f"  -> {val_path}")


# --- 主脚本 (基于 Listing 2) ---
def generate_cv_sim_splits():
    print(f"--- 任务 0.1: 生成 CV-Sim 折叠 (Listing 2) ---")

    # 1. 从 FASTA 文件加载所有蛋白质 ID
    print(f"Loading all protein IDs from {FASTA_FILE}...")
    all_proteins = []
    try:
        with open(FASTA_FILE, 'r') as f:
            for line in f:
                if line.startswith('>'):
                    # 蛋白质 ID 是 '>' 之后到第一个空格前的部分
                    protein_id = line.split()[0][1:]
                    all_proteins.append(protein_id)
    except FileNotFoundError:
        print(f"!! 错误: {FASTA_FILE} 未找到。")
        print("!! 请确保你在 /data/CAFA6_QIU 目录中运行此脚本。")
        return
    
    if not all_proteins:
        print("!! 错误: 在 FASTA 文件中未找到蛋白质。")
        return
        
    print(f"Found {len(all_proteins)} total proteins.")
    
    # 创建一个占位符 X，其长度与 all_proteins 相同
    X = np.empty(len(all_proteins)) 

    # 2. 加载聚类结果
    print(f"Loading clustering results from {CLUSTER_FILE}...")
    try:
        clusters_df = pd.read_csv(
            CLUSTER_FILE, 
            sep='\t', 
            header=None, 
            names=['cluster_head', 'protein_id']
        )
    except FileNotFoundError:
        print(f"!! 错误: {CLUSTER_FILE} 未找到。")
        print("!! 'mmseqs' 命令是否成功运行并在该路径生成了文件？")
        return

    # 创建从 protein_id 到其 cluster_head (组) 的映射
    protein_to_cluster = pd.Series(
        clusters_df.cluster_head.values,
        index=clusters_df.protein_id
    ).to_dict()

    # 3. 创建 GroupKFold (确保序列在同一个簇中保持在一起)
    # 为每个蛋白质分配一个组 (cluster_head)。
    # 如果一个蛋白质没有被聚类 (例如，单例)，它将使用自己的 ID 作为组。
    groups = [protein_to_cluster.get(pid, pid) for pid in all_proteins]
    
    gkf = GroupKFold(n_splits=N_SPLITS)

    print(f"Generating {N_SPLITS} folds using GroupKFold...")
    
    # 4. 生成并保存拆分
    # 我们根据 'all_proteins' 列表的索引进行拆分
    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, groups=groups)):
        # train_idx 和 val_idx 是来自 all_proteins 列表的整数索引 (位置)
        
        # 调用你计划中的 save_fold 函数
        save_fold(fold_idx, train_idx, val_idx)
        
    print(f"\n[SUCCESS] CV-Sim Folds 已保存到 '{OUTPUT_DIR}' 目录。")

if __name__ == "__main__":
    generate_cv_sim_splits()