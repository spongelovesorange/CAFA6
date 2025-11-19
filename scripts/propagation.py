import os
import pandas as pd
import numpy as np
import csv
from goatools.obo_parser import GODag
from tqdm import tqdm

# ================= 配置 =================
INPUT_SUBMISSION = 'submission.tsv'
OUTPUT_SUBMISSION = 'submission_propagated.tsv'
OBO_PATH = 'data/Train/go-basic.obo'

# 传播参数
MAX_PREDS_PER_PROTEIN = 1500
PROPAGATION_DECAY = 0.95  # 父节点继承95%的子节点分数
FINAL_THRESHOLD = 0.01    # 最终阈值

def main():
    print("="*60)
    print("CAFA6 GO Hierarchy Propagation")
    print("="*60)
    
    # 1. 加载GO图
    print(f"\n>>> Loading GO DAG from {OBO_PATH}...")
    if not os.path.exists(OBO_PATH):
        raise FileNotFoundError(f"❌ GO OBO file not found: {OBO_PATH}")
    
    go_dag = GODag(OBO_PATH)
    print(f"✅ GO DAG loaded: {len(go_dag)} terms")
    
    # 2. 预计算祖先（加速）
    print("\n>>> Pre-computing ancestors map...")
    term_ancestors = {}
    for term in tqdm(go_dag):
        term_ancestors[term] = go_dag[term].get_all_parents()
    print(f"✅ Ancestors computed")
    
    # 3. 读取原始提交
    print(f"\n>>> Reading raw submission: {INPUT_SUBMISSION}...")
    if not os.path.exists(INPUT_SUBMISSION):
        raise FileNotFoundError(f"❌ Input file not found! Please run inference_m2.py first.")
    
    df = pd.read_csv(INPUT_SUBMISSION, sep='\t', names=['id', 'term', 'score'])
    print(f"✅ Input: {len(df):,} predictions for {df['id'].nunique():,} proteins")
    
    # 4. 过滤无效GO terms
    valid_terms = set(term_ancestors.keys())
    before_filter = len(df)
    df = df[df['term'].isin(valid_terms)]
    after_filter = len(df)
    
    if before_filter != after_filter:
        print(f"⚠️  Filtered {before_filter - after_filter} predictions with invalid GO terms")
    
    # 5. 传播
    grouped = df.groupby('id')
    new_rows = []
    
    print(f"\n>>> Propagating scores (Child → Parent with {PROPAGATION_DECAY} decay)...")
    for pid, group in tqdm(grouped, total=len(grouped)):
        # 当前蛋白的预测
        scores = dict(zip(group['term'], group['score']))
        final_scores = scores.copy()
        
        # 传播到祖先
        predicted_terms = list(scores.keys())
        for term in predicted_terms:
            score = scores[term]
            parents = term_ancestors.get(term, [])
            
            for parent in parents:
                # 父节点分数 = max(当前分数, 子节点分数 × 衰减系数)
                propagated_score = score * PROPAGATION_DECAY
                final_scores[parent] = max(final_scores.get(parent, 0.0), propagated_score)
        
        # 排序并限制数量
        sorted_terms = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        sorted_terms = sorted_terms[:MAX_PREDS_PER_PROTEIN]
        
        # 应用最终阈值
        for term, score in sorted_terms:
            if score >= FINAL_THRESHOLD:
                new_rows.append([pid, term, f"{score:.3f}"])
    
    print(f"✅ Propagation complete")
    
    # 6. 保存结果
    print(f"\n>>> Saving propagated results to {OUTPUT_SUBMISSION}...")
    with open(OUTPUT_SUBMISSION, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(new_rows)
    print(f"✅ File saved")
    
    # 7. 验证
    print("\n" + "="*60)
    print("VALIDATION REPORT")
    print("="*60)
    
    df_final = pd.read_csv(OUTPUT_SUBMISSION, sep='\t', names=['id', 'term', 'score'])
    
    print(f"\n📊 Statistics:")
    print(f"  Total predictions: {len(df_final):,}")
    print(f"  Unique proteins: {df_final['id'].nunique():,}")
    print(f"  Avg preds/protein: {len(df_final) / df_final['id'].nunique():.1f}")
    print(f"  Score range: [{df_final['score'].min():.3f}, {df_final['score'].max():.3f}]")
    print(f"  File size: {os.path.getsize(OUTPUT_SUBMISSION) / (1024*1024):.1f} MB")
    
    counts = df_final.groupby('id').size()
    print(f"\n📈 Predictions per protein:")
    print(f"  Min: {counts.min()}")
    print(f"  Max: {counts.max()}")
    print(f"  Median: {counts.median():.0f}")
    print(f"  Mean: {counts.mean():.1f}")
    
    # 对比传播前后
    print(f"\n📊 Before vs After Propagation:")
    print(f"  Before: {len(df):,} predictions")
    print(f"  After:  {len(df_final):,} predictions")
    print(f"  Change: +{len(df_final) - len(df):,} ({(len(df_final)/len(df) - 1)*100:+.1f}%)")
    
    # 合规性检查
    print(f"\n✅ Compliance Check:")
    if counts.max() <= 1500:
        print("  ✅ All proteins within 1500 prediction limit")
    else:
        print(f"  ❌ {(counts > 1500).sum()} proteins exceed 1500!")
    
    print("\n" + "="*60)
    print(f"✅ SUBMISSION READY: {OUTPUT_SUBMISSION}")
    print("="*60)
    print("\nNext steps:")
    print("  1. Download submission_propagated.tsv")
    print("  2. Submit to Kaggle")
    print("  3. Wait for evaluation")

if __name__ == "__main__":
    main()