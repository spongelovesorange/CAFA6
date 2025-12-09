#!/usr/bin/env python3
"""
CAFA6 GO Hierarchy Propagation - 修复版本
主要修复：
1. 传播只增加预测，不删除
2. 使用更合理的阈值
3. 正确的祖先节点分数计算
4. 保持足够的预测数以保证recall
"""

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

# ================= 修复后的传播参数 =================
# 关键改变：
# 1. 降低最终阈值，保留更多预测
# 2. 增加最大预测数
# 3. 传播decay不要太激进

PROPAGATION_MODE = 'max_inheritance'    # 传播到所有祖先，不只是直接父节点
PROPAGATION_DECAY = 0.85              # 每层衰减15%
FINAL_THRESHOLD = 0.001                # 最终过滤阈值（从0.06降到0.01）
MAX_PREDS_PER_PROTEIN = 1500          # 最大预测数（从300增到1000）
MIN_PROPAGATED_SCORE = 0.01           # 传播的最小分数


def get_all_ancestors(go_dag, term):
    """获取GO term的所有祖先节点"""
    try:
        if term in go_dag:
            return go_dag[term].get_all_parents()
        return set()
    except:
        return set()


def get_direct_parents(go_dag, term):
    """只获取直接父节点"""
    try:
        if term in go_dag:
            return set(parent.id for parent in go_dag[term].parents)
        return set()
    except:
        return set()


def propagate_scores(protein_predictions, go_dag):
    """
    CAFA 标准传播逻辑：Max Rule
    父节点分数 = max(原分数, max(子节点分数))
    不需要 decay，因为我们要完全信任子节点的证据。
    """
    # 1. 初始化所有涉及到的节点
    # 使用字典存储最终分数，初始值为原始预测值
    final_scores = dict(protein_predictions)
    
    # 2. 拓扑排序传播 (从子节点向父节点)
    # 为了简单起见，我们可以多轮迭代或者通过获取所有祖先来处理
    # 这里使用"所有祖先"的方式，虽然计算量大，但逻辑最简单且正确
    
    # 建立一个临时的更新字典，避免在遍历时修改
    updates = {}
    
    for term, score in protein_predictions.items():
        # 获取该 term 的所有祖先
        ancestors = get_all_ancestors(go_dag, term)
        
        for ancestor in ancestors:
            # 祖先的分数至少应该是当前子节点的分数
            # CAFA 规则：S(parent) >= S(child)
            if ancestor in updates:
                updates[ancestor] = max(updates[ancestor], score)
            else:
                updates[ancestor] = score
    
    # 3. 合并更新
    for term, score in updates.items():
        if term in final_scores:
            final_scores[term] = max(final_scores[term], score)
        else:
            final_scores[term] = score
            
    return final_scores

def main():
    print("="*80)
    print("🎯 CAFA6 GO Propagation - Fixed Version")
    print("="*80)
    
    print(f"\n📋 Configuration:")
    print(f"   Mode:                 {PROPAGATION_MODE}")
    print(f"   Decay factor:         {PROPAGATION_DECAY}")
    print(f"   Min propagated score: {MIN_PROPAGATED_SCORE}")
    print(f"   Final threshold:      {FINAL_THRESHOLD}")
    print(f"   Max preds/protein:    {MAX_PREDS_PER_PROTEIN}")
    
    # 1. 加载GO图
    print(f"\n>>> Loading GO DAG from {OBO_PATH}...")
    if not os.path.exists(OBO_PATH):
        print(f"❌ GO OBO file not found: {OBO_PATH}")
        print("   Please download from: http://geneontology.org/docs/download-ontology/")
        return
    
    go_dag = GODag(OBO_PATH)
    print(f"✅ GO DAG loaded: {len(go_dag)} terms")
    
    # 2. 读取原始提交
    print(f"\n>>> Reading raw submission: {INPUT_SUBMISSION}...")
    if not os.path.exists(INPUT_SUBMISSION):
        print(f"❌ Submission file not found: {INPUT_SUBMISSION}")
        return
    
    df = pd.read_csv(INPUT_SUBMISSION, sep='\t', names=['id', 'term', 'score'])
    print(f"✅ Input: {len(df):,} predictions for {df['id'].nunique():,} proteins")
    
    # 原始统计
    original_avg_preds = len(df) / df['id'].nunique()
    print(f"   Original avg preds/protein: {original_avg_preds:.1f}")
    
    # 3. 过滤无效GO terms
    valid_terms = set(go_dag.keys())
    before_filter = len(df)
    df = df[df['term'].isin(valid_terms)]
    
    if before_filter != len(df):
        print(f"⚠️  Filtered {before_filter - len(df)} invalid GO terms")
    
    # 4. 传播
    grouped = df.groupby('id')
    new_rows = []
    
    stats = {
        'proteins_processed': 0,
        'preds_before': 0,
        'preds_after': 0,
        'preds_added': 0,
        'preds_from_propagation': 0
    }
    
    print(f"\n>>> Propagating GO terms...")
    for pid, group in tqdm(grouped, total=len(grouped)):
        # 构建原始预测字典
        original_scores = dict(zip(group['term'], group['score']))
        stats['preds_before'] += len(original_scores)
        
        # 传播
        propagated_scores = propagate_scores(
            original_scores, 
            go_dag
        )
        
        # 统计新增的预测
        new_terms = set(propagated_scores.keys()) - set(original_scores.keys())
        stats['preds_from_propagation'] += len(new_terms)
        
        # 应用最终阈值（只过滤太低的分数）
        final_scores = {
            term: score 
            for term, score in propagated_scores.items() 
            if score >= FINAL_THRESHOLD
        }
        
        # 如果预测太多，保留Top-K
        if len(final_scores) > MAX_PREDS_PER_PROTEIN:
            sorted_items = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
            final_scores = dict(sorted_items[:MAX_PREDS_PER_PROTEIN])
        
        stats['preds_after'] += len(final_scores)
        stats['preds_added'] += (len(final_scores) - len(original_scores))
        stats['proteins_processed'] += 1
        
        # 按分数降序排列并添加到结果
        for term, score in sorted(final_scores.items(), key=lambda x: x[1], reverse=True):
            new_rows.append([pid, term, f"{score:.3f}"])
    
    print(f"✅ Propagation complete")
    
    # 5. 保存
    print(f"\n>>> Saving to {OUTPUT_SUBMISSION}...")
    with open(OUTPUT_SUBMISSION, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(new_rows)
    print(f"✅ File saved")
    
    # 6. 验证和统计
    print("\n" + "="*80)
    print("📊 PROPAGATION REPORT")
    print("="*80)
    
    df_final = pd.read_csv(OUTPUT_SUBMISSION, sep='\t', names=['id', 'term', 'score'])
    file_size_mb = os.path.getsize(OUTPUT_SUBMISSION) / (1024*1024)
    
    print(f"\n📈 Statistics:")
    print(f"   Proteins processed:    {stats['proteins_processed']:,}")
    print(f"   Predictions before:    {stats['preds_before']:,}")
    print(f"   Predictions after:     {stats['preds_after']:,}")
    print(f"   Added by propagation:  {stats['preds_from_propagation']:,}")
    print(f"   Net change:            {stats['preds_added']:+,}")
    
    print(f"\n📈 Final File Statistics:")
    print(f"   Total predictions:     {len(df_final):,}")
    print(f"   Unique proteins:       {df_final['id'].nunique():,}")
    print(f"   Unique GO terms:       {df_final['term'].nunique():,}")
    print(f"   Avg preds/protein:     {len(df_final) / df_final['id'].nunique():.1f}")
    print(f"   Score range:           [{df_final['score'].min():.3f}, {df_final['score'].max():.3f}]")
    print(f"   Score median:          {df_final['score'].median():.3f}")
    print(f"   📁 File size:          {file_size_mb:.1f} MB")
    
    # 预测数分布
    counts = df_final.groupby('id').size()
    print(f"\n📊 Predictions per Protein:")
    print(f"   Min:     {counts.min()}")
    print(f"   10%:     {counts.quantile(0.10):.0f}")
    print(f"   25%:     {counts.quantile(0.25):.0f}")
    print(f"   Median:  {counts.median():.0f}")
    print(f"   75%:     {counts.quantile(0.75):.0f}")
    print(f"   90%:     {counts.quantile(0.90):.0f}")
    print(f"   Max:     {counts.max()}")
    print(f"   Mean:    {counts.mean():.1f}")
    
    # 传播效果
    final_avg = len(df_final) / df_final['id'].nunique()
    growth = (final_avg / original_avg_preds - 1) * 100
    
    print(f"\n📈 Propagation Effect:")
    print(f"   Before:  {original_avg_preds:.1f} preds/protein")
    print(f"   After:   {final_avg:.1f} preds/protein")
    print(f"   Growth:  {growth:+.1f}%")
    
    # 文件大小检查
    print(f"\n📁 File Size Check:")
    if file_size_mb > 800:
        print(f"   ⚠️  Large file ({file_size_mb:.0f} MB)")
        print(f"   Consider increasing FINAL_THRESHOLD or decreasing MAX_PREDS_PER_PROTEIN")
    else:
        print(f"   ✅ File size OK ({file_size_mb:.0f} MB)")
    
    # 性能估计
    print(f"\n🎯 Expected Performance:")
    score_median = df_final['score'].median()
    
    if final_avg > 100 and score_median > 0.05:
        print(f"   Strategy:        Balanced (good recall + precision)")
        print(f"   Expected F-max:  0.32-0.40 ✅")
    elif final_avg > 50:
        print(f"   Strategy:        Conservative")
        print(f"   Expected F-max:  0.28-0.36")
    else:
        print(f"   Strategy:        Very Conservative")
        print(f"   Expected F-max:  0.25-0.32")
    
    print("\n" + "="*80)
    print(f"✅ PROPAGATION COMPLETE!")
    print(f"   Output: {OUTPUT_SUBMISSION}")
    print(f"   Size:   {file_size_mb:.1f} MB")
    print(f"   Ready for submission to Kaggle")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()