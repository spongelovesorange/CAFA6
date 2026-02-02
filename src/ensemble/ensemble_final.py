import gzip
import math
import gc
from tqdm import tqdm
import sys

# ================= 配置 =================
INPUT_FILES = {
    "esm": "/data/CAFA6_QIU/predictions/esm2_raw.tsv.gz",
    "knn": "/data/CAFA6_QIU/predictions/prediction_3di_knn.tsv",
    "dia": "/data/CAFA6_QIU/predictions/diamond_scores_normalized.csv"
}
OUTPUT_FILE = "submission_final_cafa6_v2_optimized.tsv.gz"
OBO_FILE = "/data/CAFA6_QIU/data/go-basic.obo"

# 权重 (0.1, 0.5, 0.4 for MF)
WEIGHTS = {
    'MF': [0.1, 0.5, 0.4], # [esm, knn, dia]
    'BP': [0.5, 0.2, 0.3],
    'CC': [0.2, 0.2, 0.6]
}

# 过滤阈值
MIN_SCORE_RAW = 0.001 
MAX_TERMS = 700 # CAFA 规则上限是 1500，稍微放宽一点，最后再切

def load_ontology(obo_path):
    print("📚 加载 OBO...")
    ns_map = {}
    with open(obo_path, 'r') as f:
        curr = None
        for line in f:
            if line.startswith("id: GO:"): curr = line.strip()[4:]
            elif line.startswith("namespace:") and curr:
                if 'molecular_function' in line: ns_map[curr] = 'MF'
                elif 'biological_process' in line: ns_map[curr] = 'BP'
                elif 'cellular_component' in line: ns_map[curr] = 'CC'
    return ns_map

def main():
    # 1. 加载 OBO
    ns_map = load_ontology(OBO_FILE)
    
    # 2. 内存优化核心：只加载 KNN 和 Diamond 到内存
    # 数据结构: aux_data[pid][go] = [knn_score, dia_score]
    # 我们使用 compact 的方式
    aux_data = {} 
    
    # --- 加载 Diamond (Index 1) ---
    print("🔹 加载 Diamond 到内存 (1/2)...")
    with open(INPUT_FILES['dia'], 'r') as f:
        for line in tqdm(f):
            if "score" in line.lower() or "target" in line.lower(): continue
            p = line.strip().split(',')
            if len(p) < 3: continue
            
            pid = sys.intern(p[0].split('|')[1]) if '|' in p[0] else sys.intern(p[0])
            go = sys.intern(p[1])
            try:
                s = float(p[2])
            except: continue
            
            if pid not in aux_data: aux_data[pid] = {}
            if go not in aux_data[pid]: aux_data[pid][go] = [0.0, 0.0]
            aux_data[pid][go][1] = s # Index 1 is Diamond

    gc.collect() # 强制回收临时垃圾

    # --- 加载 KNN (Index 0) ---
    print("🔹 加载 KNN 到内存 (2/2)...")
    with open(INPUT_FILES['knn'], 'r') as f:
        for line in tqdm(f):
            if "score" in line.lower(): continue
            p = line.strip().split('\t')
            if len(p) < 3: continue
            
            pid = sys.intern(p[0].split('|')[1]) if '|' in p[0] else sys.intern(p[0])
            go = sys.intern(p[1])
            try:
                s = float(p[2])
            except: continue

            if pid not in aux_data: aux_data[pid] = {}
            if go not in aux_data[pid]: aux_data[pid][go] = [0.0, 0.0]
            aux_data[pid][go][0] = s # Index 0 is KNN

    print(f"✅ 辅助数据加载完成，内存中包含 {len(aux_data)} 个蛋白质的数据。")
    
    # 3. 流式处理 ESM 并融合
    print("🚀 开始流式处理 ESM 并生成最终文件...")
    
    # 为了防止 aux_data 里有 ESM 没有覆盖到的蛋白 (仅有 KNN/DIA 的情况)
    # 我们需要记录哪些蛋白已经被处理过了
    processed_pids = set()
    
    current_pid = None
    current_terms = {} # {go: esm_score}
    
    count = 0
    
    # 打开输出文件
    out_f = gzip.open(OUTPUT_FILE, 'wt')

    def flush_protein(pid, terms_dict):
        """融合并写入单个蛋白的数据"""
        nonlocal count
        final_scores = []
        
        # 1. 获取该蛋白的辅助数据 (KNN, DIA)
        aux = aux_data.get(pid, {})
        
        # 2. 合并所有涉及的 GO Term
        all_gos = set(terms_dict.keys()) | set(aux.keys())
        
        for go in all_gos:
            ns = ns_map.get(go, 'MF')
            w = WEIGHTS[ns] # [esm, knn, dia]
            
            s_esm = terms_dict.get(go, 0.0)
            s_knn = aux.get(go, [0.0, 0.0])[0]
            s_dia = aux.get(go, [0.0, 0.0])[1]
            
            # 融合公式
            raw_score = (w[0] * s_esm) + (w[1] * s_knn) + (w[2] * s_dia)
            
            if raw_score < MIN_SCORE_RAW: continue
            
            # Sqrt Trick
            final = math.sqrt(raw_score)
            if final > 1.0: final = 1.0
            
            final_scores.append((go, final))
        
        # 排序截断并写入
        final_scores.sort(key=lambda x: x[1], reverse=True)
        for go, s in final_scores[:MAX_TERMS]:
            out_f.write(f"{pid}\t{go}\t{s:.3f}\n")
            count += 1
            
        # 标记为已处理，并从内存中删除以释放空间（关键！）
        if pid in aux_data:
            del aux_data[pid]

    # --- 遍历 ESM 文件 ---
    with gzip.open(INPUT_FILES['esm'], 'rt') as f:
        for line in tqdm(f):
            if "score" in line.lower(): continue
            p = line.strip().split('\t')
            if len(p) < 3: continue
            
            pid_raw = p[0]
            pid = pid_raw.split('|')[1] if '|' in pid_raw else pid_raw
            go = p[1]
            try:
                s = float(p[2])
            except: continue
            
            # 状态机：如果换了新蛋白，就结算上一个
            if pid != current_pid:
                if current_pid is not None:
                    flush_protein(current_pid, current_terms)
                
                current_pid = pid
                current_terms = {} # 重置
            
            current_terms[go] = s
            
    # 结算最后一个 ESM 蛋白
    if current_pid is not None:
        flush_protein(current_pid, current_terms)
        
    print("Processing remaining proteins (KNN/DIA only)...")
    # --- 处理剩下那些 ESM 里没出现，但 KNN/DIA 里有的蛋白 ---
    # 因为 flush_protein 里面会 del aux_data[pid]，���以现在的 aux_data 里剩下的就是 ESM 没覆盖的
    remaining_pids = list(aux_data.keys())
    for pid in tqdm(remaining_pids):
        flush_protein(pid, {}) # 传入空的 ESM dict

    out_f.close()
    print(f"✅ 完成！共写入 {count} 行预测。")

if __name__ == "__main__":
    main()