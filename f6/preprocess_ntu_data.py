# preprocess_ntu_data.py
# ##--------------------------------------------------------------------------
# 1. 12차원 입력 (All Vectors)
#    - Group A: Normalized Bone Vector (3)
#    - Group B: Normalized Velocity Vector (3)
#    - Group C: Relative Center Vector (3) - 상호작용
#    - Group D: Relative To Other Vector (3) - 디테일
# 2. 척추 길이 정규화 (Normalization) 복구
# 3. log1p 제거 (물리적 선형성 유지)
# ##----------------------------------------------------------------------------
# [수정 사항 2025-12-07] (Critical Bug Fix)
# - resize_data_skateformer_style 함수 내 차원 축소 오류 수정.
# - np.sum(..., axis=0)을 제거하고 axis=(1,2,3)만 합산하여 Time 축 보존.
# - Multiprocessing 안전성 강화를 위한 예외 처리 추가.
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
TARGET_DATA_PATH = '../nturgbd_processed_12D_Norm/' 

# 통계 파일 경로 분리
STATS_FILE_XSUB = '../stats_xsub.npz'
STATS_FILE_XVIEW = '../stats_xview.npz'

MAX_FRAMES = config.MAX_FRAMES 
NUM_JOINTS = config.NUM_JOINTS
BASE_NUM_JOINTS = 25

# Protocol Definitions
TRAINING_SUBJECTS = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
TRAINING_CAMERAS = [2, 3] 

# 뼈 연결 정보
SKELETON_BONES = [
    (20, 1), (1, 0), (20, 2), (2, 3),
    (20, 4), (4, 5), (5, 6), (6, 7), (7, 21), (7, 22),
    (20, 8), (8, 9), (9, 10), (10, 11), (11, 23), (11, 24),
    (0, 12), (12, 13), (13, 14), (14, 15),
    (0, 16), (16, 17), (17, 18), (18, 19)
]

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

    # 1차 스캔: Actor 찾기
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
    [SkateFormer Strategy Implementation]
    1. 전체 데이터에서 움직임이 있는(0이 아닌) 유효 구간(Valid Interval)을 찾습니다.
    2. 해당 구간을 Target Frames(config.MAX_FRAMES) 길이로 선형 보간(Resize)합니다.
    Input: (T, M, V, C) numpy array
    Output: (target_frames, M, V, C) numpy array
    """
    T, M, V, C = data_numpy.shape
    
    valid_mask = np.sum(np.abs(data_numpy), axis=(1, 2, 3)) != 0
    
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) == 0:
        return np.zeros((target_frames, M, V, C))
    
    # 유효 구간의 시작과 끝
    begin = valid_indices[0]
    end = valid_indices[-1] + 1
    
    # 유효 구간 Crop (자르기)
    valid_data = data_numpy[begin:end] # Shape: (Real_T, M, V, C)
    
    # 2. Resize (Linear Interpolation)
    # PyTorch interpolate 사용을 위해 차원 변환: (Batch, Channel, Length)
    # 여기서는 Batch=1, Channel=All Features(M*V*C), Length=Time
    data_torch = torch.from_numpy(valid_data).float()
    
    # (Real_T, M, V, C) -> (1, M, V, C, Real_T) 아님. Permute 주의.
    # permute(1, 2, 3, 0) -> (M, V, C, Real_T)
    # view(1, -1, Real_T) -> (1, M*V*C, Real_T) == (Batch, Channel, Length)
    data_torch = data_torch.permute(1, 2, 3, 0).contiguous().view(1, M * V * C, -1)
    
    # 보간 수행
    data_resized = F.interpolate(data_torch, size=target_frames, mode='linear', align_corners=False)
    
    # 원래 차원으로 복구: (1, M*V*C, Target_T) -> (Target_T, M, V, C)
    data_resized = data_resized.view(M, V, C, target_frames).permute(3, 0, 1, 2).contiguous()
    
    return data_resized.numpy()

def _calculate_features(coords):
    """12차원 벡터 특징 계산 함수"""
    T = coords.shape[0]
    if T == 0:
        return np.zeros((0, NUM_JOINTS, config.NUM_COORDS))

    # 1. 척추 길이 계산
    spine_vec = coords[:, :, 20, :] - coords[:, :, 0, :]
    spine_len = np.linalg.norm(spine_vec, axis=-1, keepdims=True)
    
    valid_mask = (spine_len > 0.01).astype(np.float32)
    safe_spine_len = spine_len + 1e-8
    safe_spine_len = safe_spine_len[..., np.newaxis] 

    # A. Bone Vector
    bone_features = np.zeros((T, 2, BASE_NUM_JOINTS, 3))
    for parent, child in SKELETON_BONES:
        vec = coords[:, :, child, :] - coords[:, :, parent, :]
        bone_features[:, :, child, :] = vec
    bone_features = bone_features / safe_spine_len

    # B. Velocity Vector
    # Resize된 좌표 상에서의 속도이므로 시간 간격이 정규화된 상태의 변화량입니다.
    velocity = np.zeros_like(coords)
    velocity[1:] = coords[1:] - coords[:-1]
    velocity_features = velocity / safe_spine_len

    # C. Relative Center
    p0_center = coords[:, 0, 0, :] 
    p1_center = coords[:, 1, 0, :] 
    
    rel_center_0to1 = (p1_center - p0_center)
    rel_center_1to0 = (p0_center - p1_center)
    
    rel_center_0to1 = rel_center_0to1[:, np.newaxis, :] / safe_spine_len[:, 0, :, :]
    rel_center_1to0 = rel_center_1to0[:, np.newaxis, :] / safe_spine_len[:, 1, :, :]
    
    rc_feat_p0 = np.broadcast_to(rel_center_0to1, (T, BASE_NUM_JOINTS, 3))
    rc_feat_p1 = np.broadcast_to(rel_center_1to0, (T, BASE_NUM_JOINTS, 3))
    rel_center_features = np.stack([rc_feat_p0, rc_feat_p1], axis=1)

    # D. Relative To Other
    rel_other_p0 = (coords[:, 0, :, :] - p1_center[:, np.newaxis, :])
    rel_other_p1 = (coords[:, 1, :, :] - p0_center[:, np.newaxis, :])
    
    rel_other_p0 = rel_other_p0 / safe_spine_len[:, 0, :, :]
    rel_other_p1 = rel_other_p1 / safe_spine_len[:, 1, :, :]
    rel_other_features = np.stack([rel_other_p0, rel_other_p1], axis=1)

    # Concatenation
    final_features_per_person = np.concatenate([
        bone_features,
        velocity_features,
        rel_center_features,
        rel_other_features
    ], axis=-1)

    final_features_per_person *= valid_mask[..., np.newaxis]

    person1 = final_features_per_person[:, 0, :, :]
    person2 = final_features_per_person[:, 1, :, :]
    
    return np.concatenate((person1, person2), axis=1)

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
        
        # [Corrected Logic] 전체 좌표를 넘겨서 Resize 수행 (config.MAX_FRAMES)
        resized_coords = resize_data_skateformer_style(coords, target_frames=MAX_FRAMES)
        features = _calculate_features(resized_coords)
        
        features_flat = features.reshape(-1, config.NUM_COORDS)
        mask = np.abs(features_flat).sum(axis=1) > 1e-6
        valid_data = features_flat[mask]
        
        if valid_data.shape[0] == 0: return None
        
        return (valid_data.shape[0], valid_data.sum(axis=0), np.sum(valid_data**2, axis=0), is_xsub_train, is_xview_train)
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None

def calculate_and_save_stats():
    print("--- Calculating Stats for 12D Features (Separate for X-Sub / X-View) ---")
    filenames = os.listdir(SOURCE_DATA_PATH)
    
    cnt_sub, sum_sub, ss_sub = np.zeros(config.NUM_COORDS), np.zeros(config.NUM_COORDS), np.zeros(config.NUM_COORDS)
    cnt_view, sum_view, ss_view = np.zeros(config.NUM_COORDS), np.zeros(config.NUM_COORDS), np.zeros(config.NUM_COORDS)
    
    num_cores = cpu_count() - 1 if cpu_count() > 1 else 1
    
    with Pool(num_cores) as pool:
        for res in tqdm(pool.imap_unordered(process_file_for_stats, filenames), total=len(filenames)):
            if res:
                c, s, ss, is_xsub, is_xview = res
                if is_xsub:
                    cnt_sub += c; sum_sub += s; ss_sub += ss
                if is_xview:
                    cnt_view += c; sum_view += s; ss_view += ss

    mean_sub = sum_sub / (cnt_sub + 1e-8)
    std_sub = np.sqrt(np.maximum((ss_sub / (cnt_sub + 1e-8)) - mean_sub**2, 0)) + 1e-8
    np.savez(STATS_FILE_XSUB, mean=mean_sub, std=std_sub)
    print(f"X-Sub Stats saved: {STATS_FILE_XSUB}")

    mean_view = sum_view / (cnt_view + 1e-8)
    std_view = np.sqrt(np.maximum((ss_view / (cnt_view + 1e-8)) - mean_view**2, 0)) + 1e-8
    np.savez(STATS_FILE_XVIEW, mean=mean_view, std=std_view)
    print(f"X-View Stats saved: {STATS_FILE_XVIEW}")

def process_and_save_file(filename):
    if not filename.endswith('.skeleton'): return
    
    try:
        path = os.path.join(SOURCE_DATA_PATH, filename)
        coords = _read_skeleton_file(path)
        label = int(filename[17:20]) - 1

        if coords.shape[0] == 0:
            feat = np.zeros((MAX_FRAMES, NUM_JOINTS, config.NUM_COORDS))
        else:
            # [Corrected Logic] config.MAX_FRAMES에 맞춰 전체 시퀀스 Resize
            resized_coords = resize_data_skateformer_style(coords, target_frames=MAX_FRAMES)
            
            # Resize된 좌표로 특징 계산
            feat = _calculate_features(resized_coords) 

        torch.save({
            'data': torch.from_numpy(feat).float(),
            'label': label
        }, os.path.join(TARGET_DATA_PATH, filename.replace('.skeleton', '.pt')))
    except Exception as e:
        # 하나가 실패해도 멈추지 않고 로그만 남김
        print(f"Error saving {filename}: {e}")
        # traceback.print_exc() # 상세 에러 필요시 주석 해제

def main():
    if not os.path.exists(TARGET_DATA_PATH):
        os.makedirs(TARGET_DATA_PATH)
        
    if not os.path.exists(STATS_FILE_XSUB) or not os.path.exists(STATS_FILE_XVIEW):
        calculate_and_save_stats()
        
    print(f"--- Processing Files (Strategy: Full Sequence Resize to {MAX_FRAMES} frames) ---")
    filenames = os.listdir(SOURCE_DATA_PATH)
    num_cores = cpu_count() - 1 if cpu_count() > 1 else 1
    
    with Pool(num_cores) as pool:
        list(tqdm(pool.imap_unordered(process_and_save_file, filenames), total=len(filenames)))

if __name__ == '__main__':
    main()
