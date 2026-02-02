import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import csv

# === ⚙️ 配置 ===
# 原始 Diamond 比对结果 (Query, Subject, Bitscore)
RAW_DIAMOND_FILE = "/data/CAFA6_QIU/results/diamond_baseline.tsv"
# 训练集真实标签 (用来查询 Subject 对应的 GO Terms)
TRAIN_TERMS_FILE = "/data/CAFA6_QIU/data/Train/train_terms.tsv"
# 输出文件
OUTPUT_CSV = "/data/CAFA6_QIU/predictions/diamond_scores_normalized.csv"

# 归一化参数
# 如果 Diamond 结果只有 Top 1，直接给 1.0
# 如果有多个，我们要把 Bitscore 转换成概率。
# 简单策略：Score = Bitscore / Max_Bitscore_of_Query
NORMALIZE_METHOD = "MAX_SCALE" 

def main():
    print(f"🚀 开始处理 Diamond 结果...")
    
    # 1. 加载训练集标签 (构建 "蛋白ID -> GO列表" 的字典)
    print(f"📖 加载训练集标签: {TRAIN_TERMS_FILE}")
    # 格式: EntryID <tab> term <tab> aspect
    train_terms = pd.read_csv(TRAIN_TERMS_FILE, sep="\t", dtype={'EntryID': str, 'term': str})
    
    # 优化：只保留我们关心的 EntryID，减少内存
    # 转换为字典: {'P12345': {'GO:001', 'GO:002'}, ...}
    annot_map = {}
    for pid, group in tqdm(train_terms.groupby('EntryID'), desc="构建注释库"):
        annot_map[pid] = set(group['term'].values)
        
    print(f"✅ 注释库构建完成，包含 {len(annot_map)} 个已知蛋白。")

    # 2. 处理 Diamond 原始输出
    # Diamond 格式通常是: qseqid, sseqid, bitscore (或者 pident)
    # 你的 head 显示: A0A0C5B5G6   sp|A0A0C5B5G6|MOTSC_HUMAN   38.9
    
    print(f"📖 逐行处理比对结果: {RAW_DIAMOND_FILE}")
    
    # 存储结果: query_id -> {term: max_score}
    query_preds = {}
    
    with open(RAW_DIAMOND_FILE, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        
        for row in tqdm(reader, desc="Processing Alignments"):
            if len(row) < 3: continue
            
            q_id = row[0].strip()
            s_id_raw = row[1].strip()
            try:
                score = float(row[2])
            except:
                continue
                
            # 解析 Subject ID (去掉 sp|...|...)
            # 如果 s_id 是 "sp|P12345|NAME"，我们需要 "P12345"
            if "|" in s_id_raw:
                s_id = s_id_raw.split('|')[1]
            else:
                s_id = s_id_raw
                
            # 如果这个 Subject 蛋白在训练集里有功能注释
            if s_id in annot_map:
                # 获取它所有的 GO term
                terms = annot_map[s_id]
                
                if q_id not in query_preds:
                    query_preds[q_id] = {}
                
                # 将分数赋予给这些 GO term
                # 如果同一个 Query 对同一个 Term 有多次命中（来自不同 Subject），取最高分
                for term in terms:
                    if term not in query_preds[q_id]:
                        query_preds[q_id][term] = score
                    else:
                        query_preds[q_id][term] = max(query_preds[q_id][term], score)

    # 3. 归一化并写入 CSV
    print(f"💾 正在写入最终结果: {OUTPUT_CSV}")
    
    with open(OUTPUT_CSV, 'w') as out_f:
        out_f.write("id,term,score\n")
        
        for q_id, term_scores in tqdm(query_preds.items(), desc="Normalizing"):
            if not term_scores: continue
            
            # 找到该 Query 的最大 Bitscore 作为分母
            max_score = max(term_scores.values())
            
            # 避免除以 0
            if max_score <= 0: max_score = 1.0
            
            for term, raw_score in term_scores.items():
                # 归一化：将 Bitscore 映射到 [0, 1]
                # 策略：相对分数。最匹配的那个 Subject 带来的 Term 置信度为 1.0
                final_score = raw_score / max_score
                
                # 过滤极低分，减小文件体积
                if final_score > 0.01:
                    out_f.write(f"{q_id},{term},{final_score:.4f}\n")

    print("🎉 Diamond 数据清洗完成！现在它是 100% 可靠的了。")

if __name__ == "__main__":
    main()