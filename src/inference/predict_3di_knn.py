import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import os

# ================= 路径配置 =================
BASE_DIR = "."  
TRAIN_FASTA = os.path.join(BASE_DIR, "Features_3Di", "train_3di.fasta")
TEST_FASTA = os.path.join(BASE_DIR, "Features_3Di", "test_3di.fasta")
TRAIN_TERMS = os.path.join(BASE_DIR, "Train", "train_terms.tsv")
OUTPUT_FILE = "prediction_3di_knn_fixed.tsv"
# ===========================================

def clean_id(raw_header):
    """
    关键修复：处理 ID 格式不匹配
    如果 header 是 'sp|P12345|NAME' -> 提取 'P12345'
    如果 header 是 'P12345' -> 保持不变
    """
    # 移除开头的 > (如果是直接传进来 header行的话，不过下面的逻辑已经去掉了)
    raw_header = raw_header.strip()
    
    # 尝试按 | 分割 (UniProt 标准格式)
    if "|" in raw_header:
        parts = raw_header.split('|')
        # 通常 ID 在中间，如 sp|ID|Name，取 index 1
        if len(parts) >= 2:
            return parts[1]
    
    # 如果没有 |，或者分割失败，尝试取空格前的第一个词
    return raw_header.split()[0]

def parse_fasta(file_path):
    seqs = {}
    current_id = None
    current_seq = []
    
    print(f"正在读取 {file_path} ...")
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_id:
                    seqs[current_id] = "".join(current_seq)
                
                # === 这里调用 ID 清洗 ===
                raw_id_part = line[1:] # 去掉 >
                current_id = clean_id(raw_id_part) 
                # ======================
                
                current_seq = []
            else:
                current_seq.append(line)
        if current_id:
            seqs[current_id] = "".join(current_seq)
    return seqs

def load_train_terms(file_path):
    print(f"正在读取标签 {file_path} ...")
    try:
        # 尝试读取，如果不带表头，手动加名字
        df = pd.read_csv(file_path, sep='\t')
        
        # 简单检查第一行是否看起来像 header
        # 如果第一列包含 'EntryID' 字样，则是 header，否则假设无 header
        if 'EntryID' not in df.columns:
             # 重新读取，假设没有 header (视具体 TSV 格式而定)
             # 通常 CAFA train_terms.tsv 是: EntryID, term, aspect
             df = pd.read_csv(file_path, sep='\t', header=None, names=['EntryID', 'term', 'aspect'])
        
        # 再次清洗 EntryID 列，以防 TSV 里也有脏数据
        # (虽然通常 TSV 是干净的)
        df['EntryID'] = df['EntryID'].astype(str).apply(lambda x: x.strip())
        
        term_dict = df.groupby('EntryID')['term'].apply(set).to_dict()
        return term_dict
    except Exception as e:
        print(f"读取 TSV 出错: {e}")
        return {}

# --- 主程序 ---

print(">>> 第一步：加载并清洗数据")
train_seqs = parse_fasta(TRAIN_FASTA)
# 打印前 5 个 ID 进行调试
print(f"DEBUG (Train FASTA ID 样例): {list(train_seqs.keys())[:5]}")

train_terms = load_train_terms(TRAIN_TERMS)
print(f"DEBUG (Train Terms ID 样例): {list(train_terms.keys())[:5]}")

# 检查交集
common_ids = set(train_seqs.keys()) & set(train_terms.keys())
print(f"=== 诊断结果: 交集大小为 {len(common_ids)} ===")

if len(common_ids) == 0:
    print("!!! 警告：交集仍然为 0。请检查上方打印的 ID 样例是否一致。")
    # 如果还是 0，不要往下跑了，直接停止避免浪费时间
    exit()

# 如果交集正常，继续后续步骤
train_ids_list = list(common_ids)
train_corpus = [train_seqs[uid] for uid in train_ids_list]

test_seqs = parse_fasta(TEST_FASTA)
test_ids_list = list(test_seqs.keys())
test_corpus = [test_seqs[uid] for uid in test_ids_list]

print(f"最终用于检索库的大小: {len(train_corpus)}")

# 向量化 (TF-IDF)
print(">>> 第二步：TF-IDF 结构向量化")
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 3), max_features=15000)
vectorizer.fit(train_corpus + test_corpus)
X_train = vectorizer.transform(train_corpus)
X_test = vectorizer.transform(test_corpus)

# KNN 检索
print(">>> 第三步：KNN 结构检索")
knn = NearestNeighbors(n_neighbors=20, metric='cosine', n_jobs=-1)
knn.fit(X_train)

# 预测与输出
print(">>> 第四步：生成预测结果")
results = []
BATCH_SIZE = 1000
num_batches = int(np.ceil(X_test.shape[0] / BATCH_SIZE))

for i in tqdm(range(num_batches), desc="Processing Batches"):
    start = i * BATCH_SIZE
    end = min((i + 1) * BATCH_SIZE, X_test.shape[0])
    
    batch_X = X_test[start:end]
    batch_ids = test_ids_list[start:end]
    
    distances, indices = knn.kneighbors(batch_X)
    
    for j, (dist_row, idx_row) in enumerate(zip(distances, indices)):
        target_id = batch_ids[j]
        weights = 1 - dist_row
        total_weight = np.sum(weights) + 1e-9
        
        go_counter = {}
        for k, neighbor_idx in enumerate(idx_row):
            neighbor_real_id = train_ids_list[neighbor_idx]
            neighbor_gos = train_terms.get(neighbor_real_id, set())
            w = weights[k]
            for go_id in neighbor_gos:
                go_counter[go_id] = go_counter.get(go_id, 0.0) + w
        
        sorted_gos = sorted(go_counter.items(), key=lambda x: x[1], reverse=True)[:60]
        for go_id, score in sorted_gos:
            final_score = score / total_weight
            results.append(f"{target_id}\t{go_id}\t{final_score:.4f}")

print(f">>> 第五步：写入文件 {OUTPUT_FILE}")
with open(OUTPUT_FILE, 'w') as f:
    for line in results:
        f.write(line + "\n")

print("完成！请检查 prediction_3di_knn_fixed.tsv")