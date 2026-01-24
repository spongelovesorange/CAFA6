
import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

def parse_protein_id(header_line):
    header = header_line.strip()
    if header.startswith('>'): header = header[1:]
    return header.split()[0]

def debug_data():
    PATHS = {
        'embeddings': './cache/prottrans-bert_embeddings.pkl', 
        'train_terms': 'data/Train/train_terms.tsv',
        'train_fasta': 'data/Train/train_sequences.fasta',
        'vocab_save': './models/vocab.pkl',
    }
    
    print("Checking paths...")
    for k, v in PATHS.items():
        print(f"{k}: {v} -> {os.path.exists(v)}")

    print("\nLoading Embeddings keys...")
    with open(PATHS['embeddings'], 'rb') as f:
        embeddings_dict = pickle.load(f)
    print(f"Num embeddings: {len(embeddings_dict)}")
    sample_key = next(iter(embeddings_dict))
    print(f"Sample embedding key: '{sample_key}'")

    pure_id_to_cache_key = {}
    for cache_key in embeddings_dict.keys():
        pure_id = cache_key.strip()
        if pure_id.startswith('>'): pure_id = pure_id[1:]
        pure_id = pure_id.split()[0]
        pure_id_to_cache_key[pure_id] = cache_key
    
    print(f"Num pure_id_to_cache_key: {len(pure_id_to_cache_key)}")
    print(f"Sample pure_id: '{next(iter(pure_id_to_cache_key))}'")

    print("\nLoading Vocab...")
    with open(PATHS['vocab_save'], 'rb') as f:
        selected_terms = pickle.load(f)
    print(f"Vocab size: {len(selected_terms)}")

    print("\nLoading Train Terms...")
    df = pd.read_csv(PATHS['train_terms'], sep='\t')
    print(f"Total rows in train_terms: {len(df)}")
    
    valid_pure_ids = set(pure_id_to_cache_key.keys())
    # Check intersection of EntryID and valid_pure_ids
    entry_ids = set(df['EntryID'].unique())
    print(f"Unique EntryIDs in df: {len(entry_ids)}")
    intersection_ids = entry_ids.intersection(valid_pure_ids)
    print(f"Intersection of EntryIDs and Embedding IDs: {len(intersection_ids)}")
    
    if len(intersection_ids) == 0:
        print("CRITICAL: No intersection between Train Terms EntryIDs and Embedding IDs!")
        print(f"Sample EntryIDs: {list(entry_ids)[:5]}")
        print(f"Sample Embedding Keys (processed): {list(valid_pure_ids)[:5]}")

    df = df[df['EntryID'].isin(valid_pure_ids) & df['term'].isin(set(selected_terms))]
    print(f"Filtered df rows: {len(df)}")
    
    temp_dict = df.groupby('EntryID')['term'].apply(list).to_dict()
    print(f"Available proteins with terms after filtering: {len(temp_dict)}")

    print("\nLoading Fasta...")
    fasta_proteins = []
    with open(PATHS['train_fasta'], 'r') as f:
        for line in f:
            if line.startswith('>'): fasta_proteins.append(parse_protein_id(line))
    print(f"Num proteins in Fasta: {len(fasta_proteins)}")
    print(f"Sample Fasta protein: '{fasta_proteins[0] if fasta_proteins else 'None'}'")

    # Check intersection fasta vs embeddings
    fasta_set = set(fasta_proteins)
    print(f"Intersection Fasta vs Embeddings: {len(fasta_set.intersection(valid_pure_ids))}")
    print(f"Intersection Fasta vs Filtered Terms proteins: {len(fasta_set.intersection(set(temp_dict.keys())))}")

    print("\nLoading Folds...")
    try:
        train_idx = np.load('./folds/fold_0_train_idx.npy')
        print(f"Fold 0 Train indices: {len(train_idx)}")
        print(f"Max index: {train_idx.max()}")
        
        # Simulate the Loop
        count_matched = 0
        count_idx_oob = 0
        count_missing_features = 0 # Not in embeddings
        count_missing_labels = 0 # Not in temp_dict (filtered terms)

        for idx in train_idx[:1000]: # Check first 1000
            if idx >= len(fasta_proteins):
                count_idx_oob += 1
                continue
            pid = fasta_proteins[idx]
            
            has_emb = pid in pure_id_to_cache_key
            has_labels = pid in temp_dict
            
            if has_emb and has_labels:
                count_matched += 1
            else:
                if not has_emb: count_missing_features += 1
                if not has_labels: count_missing_labels += 1
        
        print(f"Simulation on first 1000 indices:")
        print(f"  Matched: {count_matched}")
        print(f"  Index OOB: {count_idx_oob}")
        print(f"  Missing Embeddings: {count_missing_features}")
        print(f"  Missing Labels: {count_missing_labels}")

    except Exception as e:
        print(f"Error loading folds: {e}")

if __name__ == "__main__":
    debug_data()
