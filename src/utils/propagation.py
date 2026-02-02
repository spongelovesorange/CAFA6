#!/usr/bin/env python3
"""
CAFA6 Fast Propagation - Multiprocessing Version
利用服务器多核 CPU 加速传播过程 (从 1.5h -> 5min)
"""

import os
import pandas as pd
import numpy as np
import csv
from goatools.obo_parser import GODag
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import functools

# ================= 配置 =================
# ⚠️ 确保这里是你刚刚生成的集成文件
INPUT_SUBMISSION = 'submission_ensemble.tsv' 
OUTPUT_SUBMISSION = 'submission.tsv'         # 最终提交文件

OBO_PATH = 'data/Train/go-basic.obo'
FINAL_THRESHOLD = 0.001
MAX_PREDS_PER_PROTEIN = 1500

# 全局变量 (用于多进程共享)
global_go_dag = None

def load_go_dag():
    """每个子进程初始化时加载一次，或者利用 Fork 机制共享"""
    global global_go_dag
    if global_go_dag is None:
        global_go_dag = GODag(OBO_PATH)

def get_ancestors(term, dag):
    try:
        if term in dag:
            return dag[term].get_all_parents()
        return set()
    except:
        return set()

def process_protein_group(data):
    """
    处理单个蛋白的所有预测
    data: (protein_id, list_of_terms, list_of_scores)
    """
    pid, terms, scores = data
    
    # 这一步依赖全局 dag，Linux 下 fork 模式可以直接访问
    # 如果是 Windows 需要在函数内重新加载 (服务器通常是 Linux)
    dag = global_go_dag 
    
    # 1. 原始分数映射
    final_scores = dict(zip(terms, scores))
    
    # 2. 传播 (Max Rule)
    updates = {}
    for term, score in final_scores.items():
        if term not in dag: continue
        
        # 获取所有祖先
        ancestors = dag[term].get_all_parents()
        for ancestor in ancestors:
            updates[ancestor] = max(updates.get(ancestor, 0), score)
            
    # 合并
    for term, score in updates.items():
        final_scores[term] = max(final_scores.get(term, 0), score)
        
    # 3. 过滤和截断
    filtered_items = [
        (t, s) for t, s in final_scores.items() 
        if s >= FINAL_THRESHOLD
    ]
    
    # Top-K 截断
    if len(filtered_items) > MAX_PREDS_PER_PROTEIN:
        filtered_items.sort(key=lambda x: x[1], reverse=True)
        filtered_items = filtered_items[:MAX_PREDS_PER_PROTEIN]
        
    # 格式化输出
    results = []
    # 再次排序保证输出美观 (虽然不是必须)
    filtered_items.sort(key=lambda x: x[1], reverse=True)
    
    for term, score in filtered_items:
        # 使用 3 位小数，节省空间
        results.append(f"{pid}\t{term}\t{score:.3f}\n")
        
    return results

def main():
    print("="*80)
    print("🚀 CAFA6 Fast Propagation (Multiprocessing)")
    print("="*80)
    
    # 1. 加载 GO 图 (主进程)
    print(f">>> Loading GO DAG from {OBO_PATH}...")
    global global_go_dag
    global_go_dag = GODag(OBO_PATH)
    print(f"✅ GO DAG loaded: {len(global_go_dag)} terms")
    
    # 2. 读取数据
    print(f">>> Reading input: {INPUT_SUBMISSION}...")
    current_input = INPUT_SUBMISSION
    if not os.path.exists(current_input):
        print(f"❌ File not found: {current_input}")
        # 尝试回退到 submission.tsv
        if os.path.exists('submission.tsv'):
            print("⚠️ Falling back to 'submission.tsv'...")
            current_input = 'submission.tsv'
        else:
            return

    # 使用 pandas 读取，然后转换为列表格式以便分发
    df = pd.read_csv(current_input, sep='\t', names=['id', 'term', 'score'])
    print(f"✅ Loaded {len(df):,} rows")
    
    # 3. 准备数据包
    print(">>> Grouping data by protein...")
    # 这种转换方式比 groupby 快
    # 先排序
    df.sort_values('id', inplace=True)
    
    # 提取 numpy 数组加速处理
    ids = df['id'].values
    terms = df['term'].values
    scores = df['score'].values
    
    # 快速分组算法
    tasks = []
    
    # 找出每个 ID 的起始和结束位置
    unique_ids, indices = np.unique(ids, return_index=True)
    # 添加最后一个索引
    indices = np.append(indices, len(ids))
    
    print(f"✅ Prepared {len(unique_ids):,} proteins for processing")
    
    for i in tqdm(range(len(unique_ids)), desc="Building tasks"):
        start_idx = indices[i]
        end_idx = indices[i+1]
        
        p_id = ids[start_idx]
        p_terms = terms[start_idx:end_idx]
        p_scores = scores[start_idx:end_idx]
        
        tasks.append((p_id, p_terms, p_scores))
        
    del df, ids, terms, scores # 释放内存
    
    # 4. 多进程并行
    n_cpu = max(1, int(cpu_count() * 0.9)) # 使用 90% 的核心
    print(f"\n>>> Starting Pool with {n_cpu} cores...")
    
    results = []
    with Pool(processes=n_cpu) as pool:
        # 使用 chunksize 优化通信开销
        chunksize = max(1, len(tasks) // (n_cpu * 4))
        for res in tqdm(pool.imap(process_protein_group, tasks, chunksize=chunksize), 
                       total=len(tasks), desc="Propagating"):
            results.extend(res)
            
    # 5. 写入
    print(f"\n>>> Writing to {OUTPUT_SUBMISSION}...")
    with open(OUTPUT_SUBMISSION, 'w') as f:
        f.writelines(results)
        
    print(f"✅ Done! Saved to {OUTPUT_SUBMISSION}")
    print(f"📦 File size: {os.path.getsize(OUTPUT_SUBMISSION) / (1024*1024):.1f} MB")
    
    # 6. 自动压缩建议
    print("\n💡 Tip: Run this to compress for Kaggle:")
    print(f"   zip submission.zip {OUTPUT_SUBMISSION}")

if __name__ == "__main__":
    main()