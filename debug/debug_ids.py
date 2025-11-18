import pandas as pd
import numpy as np

# 配置路径
FASTA_PATH = 'data/Train/train_sequences.fasta'
TERMS_PATH = 'data/Train/train_terms.tsv'
FOLD_PATH = '/data/CAFA6_QIU/folds/fold_0_val_idx.npy'

def check_ids():
    print("=== ID 格式诊断 ===")
    
    # 1. 检查 FASTA ID 解析逻辑
    print(f"\n[1] 读取 FASTA: {FASTA_PATH}")
    fasta_ids = []
    raw_headers = []
    try:
        with open(FASTA_PATH, 'r') as f:
            for _ in range(100): # 只读前100行
                line = f.readline()
                if not line: break
                if line.startswith('>'):
                    raw_headers.append(line.strip())
                    # 模拟之前的解析逻辑
                    parsed_id = line.strip()[1:].split()[0]
                    fasta_ids.append(parsed_id)
    except FileNotFoundError:
        print("!! 错误：找不到 FASTA 文件")
        return

    print(f"原始 Header (前3个): {raw_headers[:3]}")
    print(f"解析后 ID (前3个):   {fasta_ids[:3]}")

    # 2. 检查 Train Terms ID
    print(f"\n[2] 读取 Terms: {TERMS_PATH}")
    try:
        df = pd.read_csv(TERMS_PATH, sep='\t', nrows=10)
        term_ids = df['EntryID'].astype(str).tolist()
        print(f"Terms ID (前3个):     {term_ids[:3]}")
    except Exception as e:
        print(f"!! 错误：读取 Terms 失败 - {e}")
        return

    # 3. 检查 Fold 索引
    print(f"\n[3] 检查 Fold 索引: {FOLD_PATH}")
    try:
        indices = np.load(FOLD_PATH)
        print(f"Fold 0 包含 {len(indices)} 个索引")
        print(f"示例索引: {indices[:5]}")
        
        # 尝试映射
        if indices.max() < 50: # 如果索引很小，可能我们读取的 FASTA 还没到那
             # 这里假设 fasta_ids 包含了所有数据，实际我们只读了部分，所以这里跳过完整性检查
             pass
    except Exception as e:
         print(f"!! 错误：读取 Fold 失败 - {e}")

    # 4. 尝试匹配
    # 重新读取完整的 IDs 进行一次小规模匹配测试
    print("\n[4] 尝试寻找共同 ID...")
    # 为了快速，我们假设格式问题是显而易见的
    match = False
    for fid in fasta_ids:
        if fid in term_ids: # 这里只对比了前10个，可能不准，但在 debug 模式下主要看格式
            match = True
            break
            
    if not match:
        print("❌ 前几个样本 ID 不匹配！请对比 [1] 和 [2] 的格式差异。")
        if '|' in fasta_ids[0] and '|' not in term_ids[0]:
            print("💡 提示: FASTA ID 包含 '|'，可能需要只取中间一段 (如 Uniprot ID)。")
    else:
        print("✅ ID 格式看起来是一致的。")

if __name__ == "__main__":
    check_ids()