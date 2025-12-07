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
# [수정 사항 2025-11-27]
# - 효율성 개선: 좌표 단계에서 먼저 200프레임으로 자르고 계산.
# - [Data Leakage Fix] X-Sub와 X-View 프로토콜의 통계를 분리하여 계산 및 저장.
# [수정 사항 2025-12-06] (현우님 요청)
# - Temporal Downsampling [::2] 제거: 원본 30fps 데이터를 그대로 보존.
# - 모델 내부에서 Downsampling을 수행하기 위함.
# ##----------------------------------------------------------------------------

import os
import numpy as np
import torch
from tqdm import tqdm
import config
from multiprocessing import Pool, cpu_count

# >> 경로 설정 (config와 일치시켜주세요)
SOURCE_DATA_PATH = '../../paper-review/Action_Recognition/Code/nturgbd01/' 
TARGET_DATA_PATH = '../nturgbd_processed_12D_Norm/' 

# [수정] 통계 파일 경로 분리
STATS_FILE_XSUB = '../stats_xsub.npz'
STATS_FILE_XVIEW = '../stats_xview.npz'

MAX_FRAMES = config.MAX_FRAMES
NUM_JOINTS = config.NUM_JOINTS
BASE_NUM_JOINTS = 25

# Protocol Definitions
TRAINING_SUBJECTS = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
TRAINING_CAMERAS = [2, 3] # [수정] X-View Training Cameras

# 뼈 연결 정보 (부모, 자식)
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

    # --- 1차 스캔: 메인 Actor 2명 찾기 (투표) ---
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
    
    # --- 2차 스캔: 좌표 추출 ---
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

def _calculate_features(coords):
    """12차원 벡터 특징 계산 함수"""
    T = coords.shape[0]
    if T == 0:
        return np.zeros((0, NUM_JOINTS, config.NUM_COORDS))

    # 1. 척추 길이 계산 (Scale Normalization Factor)
    spine_vec = coords[:, :, 20, :] - coords[:, :, 0, :]
    spine_len = np.linalg.norm(spine_vec, axis=-1, keepdims=True) # (T, 2, 1)
    
    # 노이즈 제거
    valid_mask = (spine_len > 0.01).astype(np.float32)
    
    # 나눗셈 안정성
    safe_spine_len = spine_len + 1e-8
    safe_spine_len = safe_spine_len[..., np.newaxis] 

    # --- Group A: Normalized Bone Vector (3ch) ---
    bone_features = np.zeros((T, 2, BASE_NUM_JOINTS, 3))
    
    for parent, child in SKELETON_BONES:
        vec = coords[:, :, child, :] - coords[:, :, parent, :]
        bone_features[:, :, child, :] = vec
        
    bone_features = bone_features / safe_spine_len

    # --- Group B: Normalized Velocity Vector (3ch) ---
    velocity = np.zeros_like(coords)
    velocity[1:] = coords[1:] - coords[:-1]
    
    velocity_features = velocity / safe_spine_len

    # --- Group C: Relative Center Vector (3ch) ---
    p0_center = coords[:, 0, 0, :] 
    p1_center = coords[:, 1, 0, :] 
    
    rel_center_0to1 = (p1_center - p0_center)
    rel_center_1to0 = (p0_center - p1_center)
    
    rel_center_0to1 = rel_center_0to1[:, np.newaxis, :] / safe_spine_len[:, 0, :, :]
    rel_center_1to0 = rel_center_1to0[:, np.newaxis, :] / safe_spine_len[:, 1, :, :]
    
    rc_feat_p0 = np.broadcast_to(rel_center_0to1, (T, BASE_NUM_JOINTS, 3))
    rc_feat_p1 = np.broadcast_to(rel_center_1to0, (T, BASE_NUM_JOINTS, 3))
    
    rel_center_features = np.stack([rc_feat_p0, rc_feat_p1], axis=1)

    # --- Group D: Relative To Other Vector (3ch) ---
    rel_other_p0 = (coords[:, 0, :, :] - p1_center[:, np.newaxis, :])
    rel_other_p1 = (coords[:, 1, :, :] - p0_center[:, np.newaxis, :])
    
    rel_other_p0 = rel_other_p0 / safe_spine_len[:, 0, :, :]
    rel_other_p1 = rel_other_p1 / safe_spine_len[:, 1, :, :]
    
    rel_other_features = np.stack([rel_other_p0, rel_other_p1], axis=1)

    # --- Final Concatenation (12ch) ---
    final_features_per_person = np.concatenate([
        bone_features,
        velocity_features,
        rel_center_features,
        rel_other_features
    ], axis=-1)

    # 마스킹 적용
    final_features_per_person *= valid_mask[..., np.newaxis]

    # (T, 50, 12) 형태로 병합
    person1 = final_features_per_person[:, 0, :, :]
    person2 = final_features_per_person[:, 1, :, :]
    
    return np.concatenate((person1, person2), axis=1)

def process_file_for_stats(filename):
    """[수정됨] 통계 계산용 워커 함수: X-Sub, X-View 여부를 함께 반환"""
    if not filename.endswith('.skeleton'): return None
    
    # Protocol 판별
    sid = int(filename[9:12])
    cid = int(filename[5:8])
    
    is_xsub_train = sid in TRAINING_SUBJECTS
    is_xview_train = cid in TRAINING_CAMERAS
    
    # 학습 데이터가 아니라면 계산할 필요 없음
    if not is_xsub_train and not is_xview_train: return None
    
    path = os.path.join(SOURCE_DATA_PATH, filename)
    coords = _read_skeleton_file(path)
    if coords.shape[0] == 0: return None
    
    # [수정] 2배수 제한 제거 -> 1배수 (MAX_FRAMES)
    # 왜냐하면 이제 [::2] 다운샘플링을 하지 않기 때문입니다.
    # 하지만 통계 계산 시에는 모든 유효 프레임을 다 쓰는 것이 좋으므로
    # MAX_FRAMES 제한을 넉넉하게 두거나 제거해도 되지만, 일관성을 위해 유지합니다.
    # 기존: limit_frames = MAX_FRAMES * 2
    limit_frames = MAX_FRAMES 
    
    cropped_coords = coords[:limit_frames]
    
    # [수정] [::2] 제거하여 모든 프레임 사용
    features = _calculate_features(cropped_coords)
    
    features_flat = features.reshape(-1, config.NUM_COORDS)
    
    mask = np.abs(features_flat).sum(axis=1) > 1e-6
    valid_data = features_flat[mask]
    
    if valid_data.shape[0] == 0: return None
    
    # (카운트, 합, 제곱합, X-Sub여부, X-View여부) 반환
    return (valid_data.shape[0], valid_data.sum(axis=0), np.sum(valid_data**2, axis=0), is_xsub_train, is_xview_train)

def calculate_and_save_stats():
    """[수정됨] 통계(Mean, Std) 계산 및 분리 저장 (X-Sub / X-View)"""
    print("--- Calculating Stats for 12D Features (Separate for X-Sub / X-View) ---")
    filenames = os.listdir(SOURCE_DATA_PATH)
    
    # X-Sub Accumulators
    cnt_sub = np.zeros(config.NUM_COORDS)
    sum_sub = np.zeros(config.NUM_COORDS)
    ss_sub  = np.zeros(config.NUM_COORDS)
    
    # X-View Accumulators
    cnt_view = np.zeros(config.NUM_COORDS)
    sum_view = np.zeros(config.NUM_COORDS)
    ss_view  = np.zeros(config.NUM_COORDS)
    
    num_cores = cpu_count() - 1 if cpu_count() > 1 else 1
    
    with Pool(num_cores) as pool:
        for res in tqdm(pool.imap_unordered(process_file_for_stats, filenames), total=len(filenames)):
            if res:
                c, s, ss, is_xsub, is_xview = res
                
                if is_xsub:
                    cnt_sub += c
                    sum_sub += s
                    ss_sub  += ss
                
                if is_xview:
                    cnt_view += c
                    sum_view += s
                    ss_view  += ss

    # --- Save X-Sub Stats ---
    mean_sub = sum_sub / (cnt_sub + 1e-8)
    var_sub = (ss_sub / (cnt_sub + 1e-8)) - mean_sub**2
    std_sub = np.sqrt(np.maximum(var_sub, 0)) + 1e-8
    np.savez(STATS_FILE_XSUB, mean=mean_sub, std=std_sub)
    print(f"X-Sub Stats saved to {STATS_FILE_XSUB}")

    # --- Save X-View Stats ---
    mean_view = sum_view / (cnt_view + 1e-8)
    var_view = (ss_view / (cnt_view + 1e-8)) - mean_view**2
    std_view = np.sqrt(np.maximum(var_view, 0)) + 1e-8
    np.savez(STATS_FILE_XVIEW, mean=mean_view, std=std_view)
    print(f"X-View Stats saved to {STATS_FILE_XVIEW}")

def process_and_save_file(filename):
    """최종 전처리 및 저장"""
    if not filename.endswith('.skeleton'): return
    
    path = os.path.join(SOURCE_DATA_PATH, filename)
    coords = _read_skeleton_file(path)
    
    if coords.shape[0] == 0:
        feat = np.zeros((MAX_FRAMES, NUM_JOINTS, config.NUM_COORDS))
        label = 0
    else:
        # [수정] 2배수 제거, [::2] 제거
        limit_frames = MAX_FRAMES 
        cropped_coords = coords[:limit_frames]
        feat_raw = _calculate_features(cropped_coords) # Full Frame
        
        T = feat_raw.shape[0]
        if T < MAX_FRAMES:
            pad = np.zeros((MAX_FRAMES - T, NUM_JOINTS, config.NUM_COORDS))
            feat = np.concatenate([feat_raw, pad], axis=0)
        else:
            feat = feat_raw[:MAX_FRAMES]
            
        label = int(filename[17:20]) - 1

    torch.save({
        'data': torch.from_numpy(feat).float(),
        'label': label
    }, os.path.join(TARGET_DATA_PATH, filename.replace('.skeleton', '.pt')))

def main():
    if not os.path.exists(TARGET_DATA_PATH):
        os.makedirs(TARGET_DATA_PATH)
        
    # 통계 파일이 하나라도 없으면 재계산
    if not os.path.exists(STATS_FILE_XSUB) or not os.path.exists(STATS_FILE_XVIEW):
        calculate_and_save_stats()
        
    print("--- Processing All Files (Full Resolution, No [::2]) ---")
    filenames = os.listdir(SOURCE_DATA_PATH)
    num_cores = cpu_count() - 1 if cpu_count() > 1 else 1
    
    with Pool(num_cores) as pool:
        list(tqdm(pool.imap_unordered(process_and_save_file, filenames), total=len(filenames)))

if __name__ == '__main__':
    main()
