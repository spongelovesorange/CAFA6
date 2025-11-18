import pandas as pd
import numpy as np
import os

# ================= 配置 =================
FOLD_NPY_PATH = '/data/CAFA6_QIU/folds/fold_0_val_idx.npy'
TRAIN_TERMS_PATH = 'data/Train/train_terms.tsv'
TRAIN_FASTA_PATH = 'data/Train/train_sequences.fasta'
OUTPUT_DIR = 'data/Test'
OUTPUT_GT_PATH = os.path.join(OUTPUT_DIR, 'ground_truth.tsv')

# ================= 1. 精准解析 Uniprot ID =================
def parse_uniprot_id(header_line):
    """
    从 >sp|A0A0C5B5G6|MOTSC_HUMAN 中提取 A0A0C5B5G6
    """
    clean_line = header_line.strip()[1:] # 去掉 '>'
    
    if '|' in clean_line:
        parts = clean_line.split('|')
        # Uniprot 格式通常是 db|ID|Name，取中间的 ID
        if len(parts) >= 2:
            return parts[1]
            
    # 如果没有竖线，回退到取第一个空格前的内容
    return clean_line.split()[0]

def load_protein_ids(fasta_path):
    print(f"Loading protein IDs from {fasta_path}...")
    ids = []
    with open(fasta_path, 'r') as f:
        for line in f:
            if line.startswith('>'):
                ids.append(parse_uniprot_id(line))
    return np.array(ids)

# ================= 2. 主程序 =================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. 加载并解析 ID
    all_protein_ids = load_protein_ids(TRAIN_FASTA_PATH)
    print(f"Sample parsed IDs: {all_protein_ids[:3]}") # 确认是否干净 (无 sp| )
    
    # 2. 加载验证集索引
    val_indices = np.load(FOLD_NPY_PATH)
    val_ids = all_protein_ids[val_indices]
    print(f"Fold 0 Validation IDs: {len(val_ids)}")
    
    # 3. 匹配标签
    print(f"Reading Terms from {TRAIN_TERMS_PATH}...")
    df_terms = pd.read_csv(TRAIN_TERMS_PATH, sep='\t')
    
    # 强制转为字符串进行匹配
    df_terms['EntryID'] = df_terms['EntryID'].astype(str)
    val_ids_set = set(val_ids)
    
    # 过滤
    gt_subset = df_terms[df_terms['EntryID'].isin(val_ids_set)]
    
    print("-" * 30)
    print(f"Matched annotations: {len(gt_subset)}")
    
    if len(gt_subset) == 0:
        print("❌ 依然为空！请检查上面的 Sample IDs 是否与 terms 文件里的 ID 长得一样。")
    else:
        # 4. 保存
        gt_subset.to_csv(OUTPUT_GT_PATH, sep='\t', index=False)
        print(f"✅ Success! Ground Truth saved to: {OUTPUT_GT_PATH}")

if __name__ == "__main__":
    main()