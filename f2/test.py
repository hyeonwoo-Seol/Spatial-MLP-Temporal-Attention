import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# config.py가 같은 폴더에 있다고 가정하고 import 시도
try:
    import config
    NUM_COORDS = config.NUM_COORDS
except ImportError:
    print("Warning: config.py not found. Using default NUM_COORDS=12.")
    NUM_COORDS = 12

# =========================================================
# 설정 (preprocess_ntu_data.py와 동일한 상대 경로 사용)
# =========================================================
DATA_PATH = '../nturgbd_processed_12D_Norm/'  # 전처리된 .pt 파일들이 있는 폴더
TRAINING_CAMERAS = [2, 3]  # Source Domain (Train)
OUTPUT_IMAGE_NAME = 'domain_stats_comparison.png' # 저장될 그래프 이미지 파일명

def get_domain_stats(filenames, domain_name):
    """
    파일 리스트의 데이터를 로드하여 채널별 평균(Mean)과 표준편차(Std)를 계산
    """
    cnt = np.zeros(NUM_COORDS)
    sum_val = np.zeros(NUM_COORDS)
    sq_sum_val = np.zeros(NUM_COORDS)
    
    print(f"--- Calculating stats for {domain_name} ({len(filenames)} files) ---")
    
    # 진행률 표시를 위해 tqdm 사용
    for filename in tqdm(filenames):
        path = os.path.join(DATA_PATH, filename)
        if not os.path.exists(path): continue
        
        try:
            # .pt 파일 로드
            datum = torch.load(path)
            data = datum['data'].numpy() # Shape: (Max_Frames, Num_Joints, Num_Coords)
            
            # (Total_Frames * Num_Joints, Num_Coords) 형태로 변환하여 통계 계산
            data_flat = data.reshape(-1, NUM_COORDS)
            
            # 유효 데이터 마스킹 (모든 값이 0인 더미 데이터 제외)
            mask = np.abs(data_flat).sum(axis=1) > 1e-6
            valid_data = data_flat[mask]
            
            if valid_data.shape[0] == 0: continue
            
            # 누적 계산
            c = valid_data.shape[0]
            s = valid_data.sum(axis=0)
            ss = np.sum(valid_data**2, axis=0)
            
            cnt += c
            sum_val += s
            sq_sum_val += ss
            
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue

    # 평균 및 표준편차 최종 계산
    # (cnt가 0일 경우를 대비해 1e-8 추가)
    mean = sum_val / (cnt + 1e-8)
    var = (sq_sum_val / (cnt + 1e-8)) - mean**2
    std = np.sqrt(np.maximum(var, 0))
    
    return mean, std

def main():
    if not os.path.exists(DATA_PATH):
        print(f"Error: Path '{DATA_PATH}' does not exist.")
        print("Please check if 'preprocess_ntu_data.py' has finished successfully.")
        return

    # 1. 파일 목록 로드 및 분류
    all_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.pt')]
    
    if not all_files:
        print(f"No .pt files found in {DATA_PATH}")
        return

    source_files = [] # Camera 2, 3
    target_files = [] # Camera 1
    
    print("Classifying files into Source and Target domains...")
    for f in all_files:
        # 파일명 형식: S001C001P001R001A001.pt
        try:
            # 파일명에서 Camera ID (C00x) 추출 (인덱스 5~8)
            cid = int(f[5:8]) 
            
            if cid in TRAINING_CAMERAS:
                source_files.append(f)
            else:
                target_files.append(f)
        except Exception as e:
            continue
            
    print(f"Found {len(source_files)} Source files (Cam 2, 3)")
    print(f"Found {len(target_files)} Target files (Cam 1)")
    
    if len(source_files) == 0 or len(target_files) == 0:
        print("Error: Not enough files to compare. Check TRAINING_CAMERAS or DATA_PATH.")
        return
    
    # 2. 통계 계산 실행
    src_mean, src_std = get_domain_stats(source_files, "Source (Train)")
    tgt_mean, tgt_std = get_domain_stats(target_files, "Target (Eval)")
    
    # 3. 결과 텍스트 출력
    print("\n" + "="*60)
    print(f"{'Dim':<5} | {'Src Mean':<10} | {'Tgt Mean':<10} | {'Diff':<10} | {'Gap Level'}")
    print("-" * 60)
    
    diffs = []
    for i in range(NUM_COORDS):
        diff = abs(src_mean[i] - tgt_mean[i])
        diffs.append(diff)
        
        # 차이가 큰 경우 시각적으로 강조
        gap_level = ""
        if diff > 0.1: gap_level = "!!! (Large)"
        elif diff > 0.05: gap_level = "! (Mid)"
            
        print(f"{i:<5} | {src_mean[i]:<10.4f} | {tgt_mean[i]:<10.4f} | {diff:<10.4f} | {gap_level}")
    print("="*60)
    print(f"Average Diff: {np.mean(diffs):.4f}")

    # 4. 결과 시각화 및 저장
    x = np.arange(NUM_COORDS)
    width = 0.35
    
    fig, ax = plt.subplots(2, 1, figsize=(12, 10))
    
    # 평균 비교 그래프
    ax[0].bar(x - width/2, src_mean, width, label='Source (Cam 2,3)', alpha=0.7, color='royalblue')
    ax[0].bar(x + width/2, tgt_mean, width, label='Target (Cam 1)', alpha=0.7, color='darkorange')
    ax[0].set_ylabel('Mean Value')
    ax[0].set_title('Feature Mean Comparison: Source vs Target')
    ax[0].set_xticks(x)
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    # 표준편차 비교 그래프
    ax[1].bar(x - width/2, src_std, width, label='Source (Cam 2,3)', alpha=0.7, color='forestgreen')
    ax[1].bar(x + width/2, tgt_std, width, label='Target (Cam 1)', alpha=0.7, color='firebrick')
    ax[1].set_ylabel('Std Value')
    ax[1].set_title('Feature Std Comparison: Source vs Target')
    ax[1].set_xticks(x)
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE_NAME)
    print(f"\nGraph saved to '{OUTPUT_IMAGE_NAME}'")
    print("Please open the image to verify the domain shift.")

if __name__ == "__main__":
    main()
