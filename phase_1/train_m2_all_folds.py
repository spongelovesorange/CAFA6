#!/usr/bin/env python3
"""
CAFA6 - 训练所有Fold的循环脚本
自动训练fold 0, 1, 2，并生成汇总报告
"""

import os
import sys
import subprocess
import time
import pandas as pd

# ================= 配置 =================
N_FOLDS = 3
TRAIN_SCRIPT = 'phase_1/train_m2_with_folds.py'
MODELS_DIR = './models'

def check_prerequisites():
    """检查运行前提条件"""
    print("\n>>> Checking Prerequisites...")
    
    # 检查训练脚本
    if not os.path.exists(TRAIN_SCRIPT):
        print(f"❌ Error: {TRAIN_SCRIPT} not found!")
        return False
    
    # 检查fold文件
    for i in range(N_FOLDS):
        train_idx = f'./folds/fold_{i}_train_idx.npy'
        val_idx = f'./folds/fold_{i}_val_idx.npy'
        if not os.path.exists(train_idx) or not os.path.exists(val_idx):
            print(f"❌ Error: Fold {i} files not found!")
            return False
    
    # 检查embeddings
    if not os.path.exists('./cache/esm2-650M_embeddings.pkl'):
        print("❌ Error: Embeddings cache not found!")
        return False
    
    # 检查GPU
    try:
        import torch
        if not torch.cuda.is_available():
            print("⚠️  Warning: CUDA not available!")
            response = input("Continue with CPU? (y/n): ")
            if response.lower() != 'y':
                return False
    except ImportError:
        print("❌ Error: PyTorch not installed!")
        return False
    
    print("✅ All prerequisites checked")
    return True

def train_fold(fold_idx):
    """训练单个fold"""
    print("\n" + "="*80)
    print(f"🔥 STARTING FOLD {fold_idx}/{N_FOLDS-1}")
    print("="*80 + "\n")
    
    start_time = time.time()
    
    # 设置环境变量来传递fold索引
    env = os.environ.copy()
    env['CURRENT_FOLD'] = str(fold_idx)
    
    # 运行训练脚本
    try:
        result = subprocess.run(
            [sys.executable, TRAIN_SCRIPT],
            env=env,
            check=False,  # 不自动抛出异常
            text=True
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"\n✅ Fold {fold_idx} completed in {elapsed/60:.1f} minutes")
            return True, elapsed
        else:
            print(f"\n❌ Fold {fold_idx} FAILED with return code {result.returncode}")
            return False, elapsed
            
    except Exception as e:
        print(f"\n❌ Fold {fold_idx} CRASHED: {e}")
        elapsed = time.time() - start_time
        return False, elapsed

def generate_summary():
    """生成训练汇总报告"""
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    
    summary_data = []
    
    for fold_idx in range(N_FOLDS):
        log_path = f'{MODELS_DIR}/training_log_fold{fold_idx}.csv'
        model_path = f'{MODELS_DIR}/m2_esm2_fold{fold_idx}.pth'
        
        fold_info = {'fold': fold_idx}
        
        # 检查模型文件
        if os.path.exists(model_path):
            size_mb = os.path.getsize(model_path) / (1024*1024)
            fold_info['model_exists'] = True
            fold_info['model_size_mb'] = size_mb
        else:
            fold_info['model_exists'] = False
            fold_info['model_size_mb'] = 0
        
        # 读取训练日志
        if os.path.exists(log_path):
            try:
                df = pd.read_csv(log_path)
                fold_info['best_f1'] = df['val_f1'].max()
                fold_info['final_epoch'] = df['epoch'].max()
                fold_info['best_epoch'] = df.loc[df['val_f1'].idxmax(), 'epoch']
            except Exception as e:
                print(f"⚠️  Warning: Could not read log for fold {fold_idx}: {e}")
                fold_info['best_f1'] = None
                fold_info['final_epoch'] = None
                fold_info['best_epoch'] = None
        else:
            fold_info['best_f1'] = None
            fold_info['final_epoch'] = None
            fold_info['best_epoch'] = None
        
        summary_data.append(fold_info)
    
    # 打印表格
    print(f"\n{'Fold':<6} {'Status':<12} {'Best F1':<10} {'Best Epoch':<12} {'Model Size':<12}")
    print("-" * 80)
    
    for info in summary_data:
        status = "✅ Success" if info['model_exists'] else "❌ Failed"
        f1_str = f"{info['best_f1']:.4f}" if info['best_f1'] is not None else "N/A"
        epoch_str = f"{info['best_epoch']}" if info['best_epoch'] is not None else "N/A"
        size_str = f"{info['model_size_mb']:.1f} MB" if info['model_exists'] else "N/A"
        
        print(f"{info['fold']:<6} {status:<12} {f1_str:<10} {epoch_str:<12} {size_str:<12}")
    
    # 统计
    successful = sum(1 for info in summary_data if info['model_exists'])
    
    if successful > 0:
        avg_f1 = sum(info['best_f1'] for info in summary_data if info['best_f1'] is not None) / successful
        print(f"\n📊 Statistics:")
        print(f"  Successful folds: {successful}/{N_FOLDS}")
        print(f"  Average F1: {avg_f1:.4f}")
        
        best_fold = max((info for info in summary_data if info['best_f1'] is not None), 
                       key=lambda x: x['best_f1'])
        print(f"  Best fold: {best_fold['fold']} (F1: {best_fold['best_f1']:.4f})")
    
    return successful == N_FOLDS

def main():
    print("\n" + "="*80)
    print("CAFA6 - Training All Folds (CV-Sim)")
    print("="*80)
    print(f"Total folds: {N_FOLDS}")
    print(f"Training script: {TRAIN_SCRIPT}")
    print(f"Models directory: {MODELS_DIR}")
    
    # 检查前提条件
    if not check_prerequisites():
        print("\n❌ Prerequisites check failed. Exiting.")
        return
    
    # 确认开始
    print("\n" + "="*80)
    print("⚠️  This will train 3 folds. Estimated time: 3-6 hours")
    print("="*80)
    response = input("\nDo you want to continue? (y/n): ")
    if response.lower() != 'y':
        print("Training cancelled by user.")
        return
    
    # 记录结果
    results = {}
    times = {}
    total_start = time.time()
    
    # 训练每个fold
    for fold_idx in range(N_FOLDS):
        success, elapsed = train_fold(fold_idx)
        results[fold_idx] = success
        times[fold_idx] = elapsed
        
        if not success:
            print(f"\n⚠️  Fold {fold_idx} failed. Do you want to continue? (y/n)")
            response = input().strip().lower()
            if response != 'y':
                print("Training stopped by user.")
                break
    
    total_elapsed = time.time() - total_start
    
    # 生成汇总
    all_success = generate_summary()
    
    # 时间统计
    print("\n" + "="*80)
    print("TIME BREAKDOWN")
    print("="*80)
    for fold_idx, elapsed in times.items():
        status = "✅" if results[fold_idx] else "❌"
        print(f"{status} Fold {fold_idx}: {elapsed/60:.1f} minutes")
    print(f"\n⏱️  Total time: {total_elapsed/3600:.2f} hours")
    
    # 最终状态
    print("\n" + "="*80)
    if all_success:
        print("✅ ALL FOLDS TRAINED SUCCESSFULLY!")
        print("\nNext steps:")
        print("  1. Run: python phase_1/inference_ensemble.py")
        print("  2. Or test individual fold: python phase_1/inference_m2.py --fold 0")
    else:
        successful = sum(results.values())
        print(f"⚠️  Only {successful}/{N_FOLDS} folds completed successfully")
        print("\nYou can:")
        print("  1. Check the logs in models/training_log_fold*.csv")
        print("  2. Re-run this script to retry failed folds")
        print("  3. Or proceed with successful folds only")
    print("="*80 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user (Ctrl+C)")
        print("Partial progress has been saved in ./models/")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()