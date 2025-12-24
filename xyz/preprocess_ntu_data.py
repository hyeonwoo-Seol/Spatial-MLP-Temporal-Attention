# preprocess_ntu_data.py
# ##--------------------------------------------------------------------------
# [Step 2: Raw XYZ 3D 좌표 추출]
# - 복잡한 Hand-crafted Feature(Bone, Velocity 등) 제거
# - 순수 (x, y, z) 3차원 좌표만 추출
# - 2명의 사람(25 joints) -> 50 joints로 병합
# - 입력 차원: (T, 50, 3)
# ##----------------------------------------------------------------------------

import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import config
from multiprocessing import Pool, cpu_count
import traceback


SOURCE_DATA_PATH = '../../paper-review/Action_Recognition/Code/nturgbd01/' 
TARGET_DATA_PATH = config.DATASET_PATH

# 통계 파일 경로 (XYZ용으로 새로 생성됨)
STATS_FILE_XSUB = '../stats_xsub_xyz.npz'
STATS_FILE_XVIEW = '../stats_xview_xyz.npz'

MAX_FRAMES = config.MAX_FRAMES 
NUM_JOINTS = config.NUM_JOINTS # 50
BASE_NUM_JOINTS = 25
NUM_COORDS = 3  # (x, y, z)

# Protocol Definitions
TRAINING_SUBJECTS = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
TRAINING_CAMERAS = [2, 3] 

def _read_skeleton_file(filepath):
    """기존과 동일한 스켈레톤 파일 읽기 함수"""
    try:
        with open(filepath, 'r') as f:
            first_line = f.readline()
            if not first_line: return np.zeros((0, 2, BASE_NUM_JOINTS, 3))
            num_frames = int(first_line)
    except Exception:
        return np.zeros((0, 2, BASE_NUM_JOINTS, 3))

    if num_frames == 0:
        return np.zeros((0, 2, BASE_NUM_JOINTS, 3))

    # 1차 스캔: Actor 찾기 (가장 움직임이 많은 2명 선별 로직 등 기존 유지)
    body_id_counts = {}
    try:
        with open(filepath, 'r') as f:
            f.readline()
            while True:
                line = f.readline()
                if not line: break
                try:
                    num_bodies = int(line.strip())
                except ValueError: continue
                
                for _ in range(num_bodies):
                    body_info = f.readline().strip().split()
                    if not body_info: continue
                    body_id = body_info[0]
                    num_joints = int(f.readline().strip())
                    
                    has_data = False
                    for _ in range(num_joints):
                        if any(float(x) != 0 for x in f.readline().split()[:3]):
                            has_data = True
                    
                    if has_data:
                        body_id_counts[body_id] = body_id_counts.get(body_id, 0) + 1
    except Exception:
        return np.zeros((0, 2, BASE_NUM_JOINTS, 3))

    sorted_ids = sorted(body_id_counts.items(), key=lambda x: x[1], reverse=True)
    targets = [x[0] for x in sorted_ids[:2]]
    
    # 2차 스캔: 좌표 추출
    final_coords = np.zeros((num_frames, 2, BASE_NUM_JOINTS, 3))
    try:
        with open(filepath, 'r') as f:
            f.readline()
            f_idx = 0
            while f_idx < num_frames:
                line = f.readline()
                if not line: break
                try:
                    num_bodies = int(line.strip())
                except ValueError: continue
                
                for _ in range(num_bodies):
                    body_info = f.readline().split()
                    if not body_info: continue
                    curr_id = body_info[0]
                    num_joints = int(f.readline().strip())
                    
                    p_idx = -1
                    if len(targets) > 0 and curr_id == targets[0]: p_idx = 0
                    elif len(targets) > 1 and curr_id == targets[1]: p_idx = 1
                    
                    for j in range(num_joints):
                        coords = list(map(float, f.readline().split()[:3]))
                        if p_idx != -1 and j < BASE_NUM_JOINTS:
                            final_coords[f_idx, p_idx, j] = coords
                f_idx += 1
    except Exception:
        pass
        
    return final_coords

def resize_data_skateformer_style(data_numpy, target_frames=MAX_FRAMES):
    """
    [Temporal Resizing]
    유효 구간을 찾아 Target Frames 길이로 선형 보간합니다.
    """
    T, M, V, C = data_numpy.shape
    
    # 0이 아닌 구간 찾기
    valid_mask = np.sum(np.abs(data_numpy), axis=(1, 2, 3)) != 0
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) == 0:
        return np.zeros((target_frames, M, V, C))
    
    begin = valid_indices[0]
    end = valid_indices[-1] + 1
    
    valid_data = data_numpy[begin:end] # (Real_T, M, V, C)
    
    # Linear Interpolation
    data_torch = torch.from_numpy(valid_data).float()
    # (Real_T, M, V, C) -> (1, M*V*C, Real_T)
    data_torch = data_torch.permute(1, 2, 3, 0).contiguous().view(1, M * V * C, -1)
    
    data_resized = F.interpolate(data_torch, size=target_frames, mode='linear', align_corners=False)
    
    # (1, M*V*C, Target_T) -> (Target_T, M, V, C)
    data_resized = data_resized.view(M, V, C, target_frames).permute(3, 0, 1, 2).contiguous()
    
    return data_resized.numpy()

def _get_raw_xyz(coords):
    """
    [New Function for Step 2]
    단순 XYZ 좌표 추출 함수
    Input: (T, 2, 25, 3)
    Output: (T, 50, 3)
    """
    T = coords.shape[0]
    if T == 0:
        return np.zeros((0, NUM_JOINTS, 3))

    # Person 1 (T, 25, 3)
    p1 = coords[:, 0, :, :]
    # Person 2 (T, 25, 3)
    p2 = coords[:, 1, :, :]
    
    # 두 사람을 관절 차원(V)으로 병합 -> (T, 50, 3)
    # 별도의 정규화(척추 길이 나눗셈 등) 없이 좌표 그대로 사용
    final_features = np.concatenate((p1, p2), axis=1)
    
    return final_features

def process_file_for_stats(filename):
    if not filename.endswith('.skeleton'): return None
    
    try:
        sid = int(filename[9:12])
        cid = int(filename[5:8])
        
        is_xsub_train = sid in TRAINING_SUBJECTS
        is_xview_train = cid in TRAINING_CAMERAS
        
        if not is_xsub_train and not is_xview_train: return None
        
        path = os.path.join(SOURCE_DATA_PATH, filename)
        coords = _read_skeleton_file(path)
        if coords.shape[0] == 0: return None
        
        # Resize
        resized_coords = resize_data_skateformer_style(coords, target_frames=MAX_FRAMES)
        
        # Extract Raw XYZ
        features = _get_raw_xyz(resized_coords) # (T, 50, 3)
        
        # 통계 계산을 위해 (N, 3) 형태로 Flatten
        features_flat = features.reshape(-1, 3)
        
        # 0이 아닌 값만 통계에 사용 (Zero-padding 제외)
        mask = np.abs(features_flat).sum(axis=1) > 1e-6
        valid_data = features_flat[mask]
        
        if valid_data.shape[0] == 0: return None
        
        # 개수, 합, 제곱합 반환 (차원별 계산)
        return (valid_data.shape[0], valid_data.sum(axis=0), np.sum(valid_data**2, axis=0), is_xsub_train, is_xview_train)
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None

def calculate_and_save_stats():
    print("--- Calculating Stats for Raw XYZ Features (X, Y, Z) ---")
    filenames = os.listdir(SOURCE_DATA_PATH)
    
    # 3차원 통계 변수 초기화
    cnt_sub, sum_sub, ss_sub = np.zeros(3), np.zeros(3), np.zeros(3)
    cnt_view, sum_view, ss_view = np.zeros(3), np.zeros(3), np.zeros(3)
    
    num_cores = cpu_count() - 1 if cpu_count() > 1 else 1
    
    with Pool(num_cores) as pool:
        for res in tqdm(pool.imap_unordered(process_file_for_stats, filenames), total=len(filenames)):
            if res:
                c, s, ss, is_xsub, is_xview = res
                if is_xsub:
                    cnt_sub += c; sum_sub += s; ss_sub += ss
                if is_xview:
                    cnt_view += c; sum_view += s; ss_view += ss

    # Mean & Std 계산
    mean_sub = sum_sub / (cnt_sub + 1e-8)
    std_sub = np.sqrt(np.maximum((ss_sub / (cnt_sub + 1e-8)) - mean_sub**2, 0)) + 1e-8
    np.savez(STATS_FILE_XSUB, mean=mean_sub, std=std_sub)
    print(f"X-Sub XYZ Stats saved: {STATS_FILE_XSUB} (Mean: {mean_sub})")

    mean_view = sum_view / (cnt_view + 1e-8)
    std_view = np.sqrt(np.maximum((ss_view / (cnt_view + 1e-8)) - mean_view**2, 0)) + 1e-8
    np.savez(STATS_FILE_XVIEW, mean=mean_view, std=std_view)
    print(f"X-View XYZ Stats saved: {STATS_FILE_XVIEW} (Mean: {mean_view})")

def process_and_save_file(filename):
    if not filename.endswith('.skeleton'): return
    
    try:
        path = os.path.join(SOURCE_DATA_PATH, filename)
        coords = _read_skeleton_file(path)
        label = int(filename[17:20]) - 1

        if coords.shape[0] == 0:
            feat = np.zeros((MAX_FRAMES, NUM_JOINTS, 3))
        else:
            # 1. Resize
            resized_coords = resize_data_skateformer_style(coords, target_frames=MAX_FRAMES)
            
            # 2. Extract Raw XYZ
            feat = _get_raw_xyz(resized_coords) 

        # 저장
        torch.save({
            'data': torch.from_numpy(feat).float(),
            'label': label
        }, os.path.join(TARGET_DATA_PATH, filename.replace('.skeleton', '.pt')))
    except Exception as e:
        print(f"Error saving {filename}: {e}")

def main():
    if not os.path.exists(TARGET_DATA_PATH):
        os.makedirs(TARGET_DATA_PATH)
        
    # 통계 파일이 없거나 새로 만들어야 할 때 계산
    # (주의: 이전 12D 통계 파일과 이름이 다릅니다. stats_xsub_xyz.npz)
    if not os.path.exists(STATS_FILE_XSUB) or not os.path.exists(STATS_FILE_XVIEW):
        calculate_and_save_stats()
        
    print(f"--- Processing Files for Raw XYZ (Target: {TARGET_DATA_PATH}) ---")
    filenames = os.listdir(SOURCE_DATA_PATH)
    num_cores = cpu_count() - 1 if cpu_count() > 1 else 1
    
    with Pool(num_cores) as pool:
        list(tqdm(pool.imap_unordered(process_and_save_file, filenames), total=len(filenames)))

if __name__ == '__main__':
    main()
