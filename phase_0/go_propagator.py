import os
import pickle
import torch
from transformers import AutoTokenizer, EsmModel, AutoModel
from tqdm import tqdm
import sys

# --- 配置 (请确保路径正确) ---
TRAIN_FASTA = 'data/Train/train_sequences.fasta'
TEST_FASTA = 'data/Test/testsuperset.fasta'
CACHE_DIR = './cache'

# [!! 新增 !!] 指向你上传的本地模型路径
MODEL_PATHS = {
    'esm2-650M': '/data/CAFA6_QIU/models/esm2_t33_650M_UR50D',
    'prottrans-bert': '/data/CAFA6_QIU/models/prot_bert'
}

# --- 辅助函数：加载所有序列 ---
# [基于 Listing 7: load_all_sequences]
def load_all_sequences(train_path, test_path):
    """从训练和测试 FASTA 文件中加载所有唯一的蛋白质序列"""
    sequences_dict = {}
    
    def parse_fasta(file_path):
        """解析 FASTA 文件并填充 sequences_dict"""
        try:
            with open(file_path, 'r') as f:
                protein_id = None
                sequence = []
                for line in f:
                    line = line.strip()
                    if line.startswith('>'):
                        if protein_id:
                            sequences_dict[protein_id] = "".join(sequence)
                        protein_id = line.split()[0][1:]
                        sequence = []
                    else:
                        sequence.append(line)
                if protein_id:
                    sequences_dict[protein_id] = "".join(sequence)
        except FileNotFoundError:
            print(f"!! 警告: 文件未找到 {file_path}")

    print(f"Loading sequences from {train_path}...")
    parse_fasta(train_path)
    print(f"Loading sequences from {test_path}...")
    parse_fasta(test_path)
    
    return sequences_dict

# --- 缓存管理器 (基于 Listing 6) ---

class EmbeddingCache:
    """
    [基于 Listing 6]
    蛋白质嵌入的智能缓存管理器。
    """
    
    def __init__(self, cache_dir='./cache'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def get_or_compute(self, model_name, sequences_dict, batch_size=32, device='cuda'):
        """
        [基于 Listing 6: get_or_compute]
        """
        
        cache_file = f"{self.cache_dir}/{model_name}_embeddings.pkl"
        
        if os.path.exists(cache_file):
            print(f"\n[Cache Hit] 从磁盘加载 {model_name}...")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        print(f"\n[Cache Miss] 开始计算 {model_name} 的嵌入...")
        embeddings = self.compute_embeddings(
            model_name, sequences_dict, batch_size, device
        )
        
        print(f"\n[Cache Save] 保存到 {cache_file}...")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(embeddings, f)
            print(f"[Cache Save] 成功保存 {len(embeddings)} 个嵌入。")
        except Exception as e:
            print(f"!! 错误: 无法保存 pickle 文件: {e}")

        return embeddings

    def compute_embeddings(self, model_name, sequences_dict, batch_size, device):
        """
        [基于 Listing 6: compute_embeddings]
        [!! 已修改 !!] 从本地路径加载模型
        """
        
        local_path = MODEL_PATHS.get(model_name.lower())
        if not local_path or not os.path.exists(local_path):
            raise FileNotFoundError(f"!! 错误: 找不到 {model_name} 的本地模型路径: {local_path}")

        print(f"  -> 正在从本地路径加载模型 {model_name}: {local_path}")
        
        # 1. 加载模型和 Tokenizer (从本地路径)
        if 'esm2-650m' in model_name.lower():
            # [!! 已修改 !!]
            model = EsmModel.from_pretrained(local_path, local_files_only=True).to(device)
            tokenizer = AutoTokenizer.from_pretrained(local_path, local_files_only=True)
        elif 'prottrans-bert' in model_name.lower():
            # [!! 已修改 !!]
            model = AutoModel.from_pretrained(local_path, local_files_only=True).to(device)
            tokenizer = AutoTokenizer.from_pretrained(local_path, local_files_only=True)
        else:
            raise ValueError(f"未知的模型: {model_name}")

        model.eval()
        embeddings = {}
        protein_ids = list(sequences_dict.keys())
        
        print(f"  -> 开始为 {len(protein_ids)} 个序列计算嵌入 (批量大小: {batch_size})...")

        # 2. 迭代所有批次
        with torch.no_grad():
            for i in tqdm(range(0, len(protein_ids), batch_size), desc=f"计算 {model_name}"):
                batch_ids = protein_ids[i:i+batch_size]
                batch_seqs = [sequences_dict[pid] for pid in batch_ids]
                
                inputs = tokenizer(
                    batch_seqs, 
                    return_tensors='pt', 
                    padding=True, 
                    truncation=True, 
                    max_length=1024
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                outputs = model(**inputs)
                
                batch_embeddings = outputs.last_hidden_state.mean(dim=1)
                
                for pid, emb in zip(batch_ids, batch_embeddings):
                    embeddings[pid] = emb.cpu()

        del model
        torch.cuda.empty_cache()
        print(f"  -> {model_name} 计算完成。")
        return embeddings

# --- 执行缓存 (基于 Listing 7) ---
def main():
    print("="*50)
    print(f"任务 0.3: 嵌入缓存系统 (Listing 6 & 7)")
    print("="*50)

    # 1. 加载所有序列
    all_sequences = load_all_sequences(TRAIN_FASTA, TEST_FASTA)
    print(f"\n[总计] 加载了 {len(all_sequences)} 个唯一的蛋白质序列。")

    cache = EmbeddingCache(cache_dir=CACHE_DIR)

    # 2. 缓存 ESM-2 650M
    print("="*50)
    print("Caching ESM-2 650M...")
    esm2_embeddings = cache.get_or_compute(
        model_name='esm2-650M',
        sequences_dict=all_sequences,
        batch_size=32 
    )
    print(f"ESM-2 缓存大小: {len(esm2_embeddings)} 个蛋白质")

    # 3. 缓存 ProtTrans-BERT
    print("="*50)
    print("Caching ProtTrans-BERT...")
    prottrans_embeddings = cache.get_or_compute(
        model_name='prottrans-bert',
        sequences_dict=all_sequences,
        batch_size=32
    )
    print(f"ProtTrans 缓存大小: {len(prottrans_embeddings)} 个蛋白质")

    print("="*50)
    print("[SUCCESS] Phase 0 Caching Complete!")

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()