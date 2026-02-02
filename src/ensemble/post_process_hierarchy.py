import pandas as pd
import numpy as np
import pickle
import os
import networkx as nx
from tqdm import tqdm

class HierarchyEnforcer:
    def __init__(self, label_map_path="/data/CAFA6_QIU/models/checkpoints_esm2_3b_qlora/label_map.pkl", obo_path="data/go-basic.obo"):
        self.label_map_path = label_map_path
        self.obo_path = obo_path
        self.graph = None
        self.term2idx = None
        self.idx2term = None
        
        # 自动初始化
        self.load_map()
        self.build_graph()

    def load_map(self):
        # 智能加载 Label Map
        with open(self.label_map_path, "rb") as f:
            data = pickle.load(f)
        
        if isinstance(data, dict):
            self.term2idx = data
        elif isinstance(data, (list, tuple)):
            for item in data:
                if isinstance(item, dict) and isinstance(list(item.keys())[0], str):
                    self.term2idx = item
                    break
        
        if self.term2idx is None:
            raise ValueError("❌ 无法加载 Label Map，层级修正初始化失败！")
            
        self.idx2term = {v: k for k, v in self.term2idx.items()}
        print(f"✅ [Hierarchy] 已加载 {len(self.term2idx)} 个标签定义")

    def build_graph(self):
        # 使用 networkx 构建 GO 图谱
        # 解析 OBO 文件
        print(f"🔄 [Hierarchy] 正在解析 {self.obo_path} 构建 DAG...")
        self.graph = nx.DiGraph()
        
        if not os.path.exists(self.obo_path):
            print("⚠️ 警告: 找不到 go-basic.obo，将无法执行层级修正！")
            return

        term_id = None
        with open(self.obo_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith("[Term]"):
                    term_id = None
                elif line.startswith("id: GO:"):
                    term_id = line[4:]
                elif line.startswith("is_a:") and term_id:
                    parent_id = line[5:].split(' ! ')[0]
                    # 在图中，边是从 Child -> Parent (is_a 关系)
                    # 但为了传播分数，我们需要 Parent -> Child 的路径，或者反向传播
                    # 这里我们添加 Child -> Parent 的边
                    self.graph.add_edge(term_id, parent_id)
        
        print(f"✅ DAG 构建完成，包含 {self.graph.number_of_nodes()} 个节点")

    def apply_max_propagation(self, scores_matrix):
        """
        执行 Max-Propagation: 
        Parent Score = max(Parent Score, All Children Scores)
        这保证了如果子节点得分高，父节点一定高。
        """
        if self.graph is None:
            return scores_matrix

        print("🚀 执行 Max-Propagation (这可能需要一点时间)...")
        num_targets, num_classes = scores_matrix.shape
        
        # 预先计算拓扑排序，确保从子节点向父节点传播的顺序正确
        # 但由于 GO 图很大，我们只关心 label_map 里的那 3000 个词
        
        # 优化策略：只针对我们在预测矩阵里有的列进行传播
        # 建立索引映射：Parent Index -> List of Child Indices
        # 这比遍历图快得多
        
        # 1. 找到所有我们关心的词及其父子关系
        # 我们需要一个 'Child -> Parents' 的映射，但在矩阵操作中，
        # 我们通常希望：Score[Parent] = max(Score[Parent], Score[Child])
        # 所以我们需要遍历所有 Child，更新其 Parent
        
        # 为了高效，我们将 Graph 转换为矩阵操作所需的邻接表
        # 只保留 matrix 中存在的节点
        relevant_terms = set(self.term2idx.keys())
        
        # 查找所有存在于 map 中的 (child, parent) 对
        propagation_pairs = []
        for term in relevant_terms:
            if term in self.graph:
                # 获取该 term 的所有直接父节点
                parents = list(self.graph.successors(term)) # is_a 指向父节点
                child_idx = self.term2idx[term]
                
                for p in parents:
                    if p in self.term2idx:
                        parent_idx = self.term2idx[p]
                        propagation_pairs.append((child_idx, parent_idx))
        
        print(f"   需要维护 {len(propagation_pairs)} 条层级约束边")

        # 2. 迭代传播 (通常 2-3 次迭代足以覆盖大部分深度)
        # 因为 Python 循环慢，这里我们用 NumPy 的 fancy indexing 甚至都不够快
        # 最快的方法是按层级顺序，但这里我们用简单的多次迭代
        
        updated_scores = scores_matrix.copy()
        
        for i in range(3): # 传播 3 次，足以覆盖大多数 ontology 深度
            changes = 0
            for child_idx, parent_idx in propagation_pairs:
                # Parent 分数 = max(Parent, Child)
                # 使用 np.maximum 进行向量化操作 (一次更新所有样本)
                # updated_scores[:, parent_idx] = np.maximum(updated_scores[:, parent_idx], updated_scores[:, child_idx])
                
                # 为了极致速度，提取出来比较
                parent_vec = updated_scores[:, parent_idx]
                child_vec = updated_scores[:, child_idx]
                
                # 只有当 child > parent 时才更新
                mask = child_vec > parent_vec
                if np.any(mask):
                    updated_scores[mask, parent_idx] = child_vec[mask]
                    changes += 1
            
            print(f"   Iter {i+1}: 更新了 {changes} 条边的约束")
            if changes == 0:
                break
                
        return updated_scores

    def apply(self, scores):
        return self.apply_max_propagation(scores)