import pandas as pd
import numpy as np
import csv
from goatools.obo_parser import GODag
from tqdm import tqdm

# ================= 配置 =================
INPUT_SUBMISSION = 'submission.tsv'       
OUTPUT_SUBMISSION = 'submission_propagated.tsv' 
OBO_PATH = 'data/Train/go-basic.obo'

def main():
    print(f"Loading GO DAG from {OBO_PATH}...")
    go_dag = GODag(OBO_PATH)
    
    print("Pre-computing ancestors map (Optimization)...")
    # 预计算所有 Term 的祖先，避免在主循环中重复搜索
    term_ancestors = {}
    for term in go_dag:
        term_ancestors[term] = go_dag[term].get_all_parents()

    print(f"Reading raw submission: {INPUT_SUBMISSION}...")
    # 假设文件无 header，列为 id, term, score
    df = pd.read_csv(INPUT_SUBMISSION, sep='\t', names=['id', 'term', 'score'])
    
    # 仅保留在 OBO 文件中的 Term (过滤无效预测)
    valid_terms = set(term_ancestors.keys())
    df = df[df['term'].isin(valid_terms)]
    
    grouped = df.groupby('id')
    new_rows = []
    
    print("Propagating scores (Child -> Parent)...")
    # 使用 tqdm 显示进度
    for pid, group in tqdm(grouped, total=len(grouped)):
        # 当前蛋白的预测：{term: score}
        scores = dict(zip(group['term'], group['score']))
        final_scores = scores.copy()
        
        predicted_terms = list(scores.keys())
        
        for term in predicted_terms:
            score = scores[term]
            # 获取该 Term 的所有祖先
            parents = term_ancestors.get(term, [])
            for p in parents:
                # 传播逻辑：父节点分数 = max(当前父节点分数, 子节点分数)
                final_scores[p] = max(final_scores.get(p, 0.0), score)
        
        # 收集结果
        for term, score in final_scores.items():
            if score >= 0.01: # 再次过滤低分，确保提交文件合规
                new_rows.append([pid, term, f"{score:.3f}"])
                
    print(f"Saving propagated results to {OUTPUT_SUBMISSION}...")
    with open(OUTPUT_SUBMISSION, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(new_rows)
        
    print("✅ Done! 请提交 submission_propagated.tsv")

if __name__ == "__main__":
    main()