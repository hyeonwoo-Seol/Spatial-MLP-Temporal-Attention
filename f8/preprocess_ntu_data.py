# >> preprocess_ntu_data.py

# ##--------------------------------------------------------------------------
# 1. 12차원 입력 (All Vectors)
#    - Group A: Normalized Bone Vector (3)
#    - Group B: Normalized Velocity Vector (3)
#    - Group C: Relative Center Vector (3) - 상호작용
#    - Group D: Relative To Other Vector (3) - 디테일
# 2. 척추 길이 정규화 (Normalization) 복구
# 3. log1p 제거 (물리적 선형성 유지)
# ##----------------------------------------------------------------------------


import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import config
from multiprocessing import Pool, cpu_count
import traceback

# >> 데이터 파일 경로를 설정한다.
SOURCE_DATA_PATH = '../../paper-review/Action_Recognition/Code/nturgbd01/' 
TARGET_DATA_PATH = '../nturgbd_processed_12D_Norm_SKF/' 

# >> 통계 파일 경로를 분리하여 설정한다.
STATS_FILE_XSUB = '../stats_xsub_SKF.npz'
STATS_FILE_XVIEW = '../stats_xview_SKF.npz'

MAX_FRAMES = config.MAX_FRAMES 
NUM_JOINTS = config.NUM_JOINTS
BASE_NUM_JOINTS = 25

# >> 학습에 사용할 Subject와 Camera 정보를 정의한다.
TRAINING_SUBJECTS = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
TRAINING_CAMERAS = [2, 3] 

# >> 관절 간의 연결(Bone) 정보를 정의한다.
SKELETON_BONES = [
    (20, 1), (1, 0), (20, 2), (2, 3),
    (20, 4), (4, 5), (5, 6), (6, 7), (7, 21), (7, 22),
    (20, 8), (8, 9), (9, 10), (10, 11), (11, 23), (11, 24),
    (0, 12), (12, 13), (13, 14), (14, 15),
    (0, 16), (16, 17), (17, 18), (18, 19)
]

# >> 스켈레톤 파일을 읽어 좌표 데이터를 추출한다.
def _read_skeleton_file(filepath):
    try:
        with open(filepath, 'r') as f:
            first_line = f.readline()
            if not first_line: return np.zeros((0, 2, BASE_NUM_JOINTS, 3))
            num_frames = int(first_line)
    except Exception:
        return np.zeros((0, 2, BASE_NUM_JOINTS, 3))

    if num_frames == 0:
        return np.zeros((0, 2, BASE_NUM_JOINTS, 3))

    # >> 1차 스캔: 등장 빈도가 높은 상위 2명의 Actor를 찾는다.
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
    
    # >> 2차 스캔: 선별된 Actor의 관절 좌표를 추출한다.
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

# >> 전체 데이터에서 움직임이 있는 유효 구간을 찾아 MAX_FRAMES로 선형 보간한다.
# >> INPUT: (T, M, V, C) 넘파이 배열
# >> OUTPUT: (config.MAX_FRAMES, M, V, C) 넘파이 배열
def resize_data_skateformer_style(data_numpy, target_frames=MAX_FRAMES):
    T, M, V, C = data_numpy.shape
    
    # >> 움직임이 있는(0이 아닌) 프레임을 식별한다.
    valid_mask = np.sum(np.abs(data_numpy), axis=(1, 2, 3)) != 0
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) == 0:
        return np.zeros((target_frames, M, V, C))
    
    # >> 유효 구간의 시작과 끝 인덱스를 설정한다.
    begin = valid_indices[0]
    end = valid_indices[-1] + 1
    
    # >> 유효 구간만 잘라낸다(Crop).
    valid_data = data_numpy[begin:end] # Shape: (Real_T, M, V, C)
    
    # >> PyTorch의 interpolate 함수 사용을 위해 차원을 변환한다.
    # >> (Real_T, M, V, C) -> (1, M*V*C, Real_T)
    data_torch = torch.from_numpy(valid_data).float()
    data_torch = data_torch.permute(1, 2, 3, 0).contiguous().view(1, M * V * C, -1)
    
    # >> 선형 보간(Linear Interpolation)을 수행하여 프레임 수를 맞춘다.
    data_resized = F.interpolate(data_torch, size=target_frames, mode='linear', align_corners=False)
    
    # >> 원래 차원으로 복구한다. (Target_T, M, V, C)
    data_resized = data_resized.view(M, V, C, target_frames).permute(3, 0, 1, 2).contiguous()
    
    return data_resized.numpy()

# >> 12차원 특징 벡터를 계산하고 정규화를 수행한다.
def _calculate_features(coords):
    T = coords.shape[0]
    if T == 0:
        return np.zeros((0, NUM_JOINTS, config.NUM_COORDS))

    # >> 1. 척추 벡터와 길이를 계산한다.
    spine_vec = coords[:, :, 20, :] - coords[:, :, 0, :]
    spine_len = np.linalg.norm(spine_vec, axis=-1, keepdims=True)
    
    # >> 척추 길이가 감지되지 않은 프레임(Noise)을 선형 보간으로 복구한다.
    for m in range(spine_len.shape[1]): # 각 사람에 대해 반복
        sl = spine_len[:, m, 0]
        invalid_idx = sl <= 0.01 # 척추 길이가 0에 가까운 프레임 식별
        valid_idx = ~invalid_idx
        
        # >> 유효한 프레임과 무효한 프레임이 공존할 경우 보간을 수행한다.
        if valid_idx.any() and invalid_idx.any():
            x_valid = np.where(valid_idx)[0]
            y_valid = sl[valid_idx]
            x_invalid = np.where(invalid_idx)[0]
            
            # >> 결측치(invalid)를 유효값(valid) 기반으로 채운다.
            sl[x_invalid] = np.interp(x_invalid, x_valid, y_valid)
            spine_len[:, m, 0] = sl

    valid_mask = (spine_len > 0.01).astype(np.float32)
    
    # >> 0으로 나누는 것을 방지하기 위해 엡실론을 더한다.
    safe_spine_len = spine_len + 1e-8
    safe_spine_len = safe_spine_len[..., np.newaxis] 

    # >> A. 척추 길이를 기준으로 정규화된 Bone Vector를 계산한다.
    bone_features = np.zeros((T, 2, BASE_NUM_JOINTS, 3))
    for parent, child in SKELETON_BONES:
        vec = coords[:, :, child, :] - coords[:, :, parent, :]
        bone_features[:, :, child, :] = vec
    bone_features = bone_features / safe_spine_len

    # >> B. 척추 길이를 기준으로 정규화된 Velocity Vector를 계산한다.
    velocity = np.zeros_like(coords)
    velocity[1:] = coords[1:] - coords[:-1]
    velocity_features = velocity / safe_spine_len

    # >> C. 두 사람 간의 상대적 중심 위치(Relative Center)를 계산한다.
    p0_center = coords[:, 0, 0, :] 
    p1_center = coords[:, 1, 0, :] 
    
    rel_center_0to1 = (p1_center - p0_center)
    rel_center_1to0 = (p0_center - p1_center)
    
    rel_center_0to1 = rel_center_0to1[:, np.newaxis, :] / safe_spine_len[:, 0, :, :]
    rel_center_1to0 = rel_center_1to0[:, np.newaxis, :] / safe_spine_len[:, 1, :, :]
    
    rc_feat_p0 = np.broadcast_to(rel_center_0to1, (T, BASE_NUM_JOINTS, 3))
    rc_feat_p1 = np.broadcast_to(rel_center_1to0, (T, BASE_NUM_JOINTS, 3))
    rel_center_features = np.stack([rc_feat_p0, rc_feat_p1], axis=1)

    # >> D. 상대방 중심에 대한 나의 관절 위치(Relative To Other)를 계산한다.
    rel_other_p0 = (coords[:, 0, :, :] - p1_center[:, np.newaxis, :])
    rel_other_p1 = (coords[:, 1, :, :] - p0_center[:, np.newaxis, :])
    
    rel_other_p0 = rel_other_p0 / safe_spine_len[:, 0, :, :]
    rel_other_p1 = rel_other_p1 / safe_spine_len[:, 1, :, :]
    rel_other_features = np.stack([rel_other_p0, rel_other_p1], axis=1)

    # >> 4가지 특징 그룹을 연결(Concatenate)한다.
    final_features_per_person = np.concatenate([
        bone_features,
        velocity_features,
        rel_center_features,
        rel_other_features
    ], axis=-1)

    # >> 유효하지 않은 데이터는 마스킹 처리한다.
    final_features_per_person *= valid_mask[..., np.newaxis]

    person1 = final_features_per_person[:, 0, :, :]
    person2 = final_features_per_person[:, 1, :, :]
    
    return np.concatenate((person1, person2), axis=1)

# >> 통계 계산을 위해 개별 파일을 처리한다.
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
        
        # >> Resize를 수행한 후 특징을 추출한다.
        resized_coords = resize_data_skateformer_style(coords, target_frames=MAX_FRAMES)
        features = _calculate_features(resized_coords)
        
        # >> 통계 계산을 위해 유효한 데이터만 필터링한다.
        features_flat = features.reshape(-1, config.NUM_COORDS)
        mask = np.abs(features_flat).sum(axis=1) > 1e-6
        valid_data = features_flat[mask]
        
        if valid_data.shape[0] == 0: return None
        
        return (valid_data.shape[0], valid_data.sum(axis=0), np.sum(valid_data**2, axis=0), is_xsub_train, is_xview_train)
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None

# >> 전체 데이터셋에 대한 평균과 표준편차를 계산하고 저장한다.
def calculate_and_save_stats():
    print("--- Calculating Stats for 12D Features (Separate for X-Sub / X-View) ---")
    filenames = os.listdir(SOURCE_DATA_PATH)
    
    cnt_sub, sum_sub, ss_sub = np.zeros(config.NUM_COORDS), np.zeros(config.NUM_COORDS), np.zeros(config.NUM_COORDS)
    cnt_view, sum_view, ss_view = np.zeros(config.NUM_COORDS), np.zeros(config.NUM_COORDS), np.zeros(config.NUM_COORDS)
    
    num_cores = cpu_count() - 1 if cpu_count() > 1 else 1
    
    # >> 멀티프로세싱을 사용하여 병렬로 통계를 집계한다.
    with Pool(num_cores) as pool:
        for res in tqdm(pool.imap_unordered(process_file_for_stats, filenames), total=len(filenames)):
            if res:
                c, s, ss, is_xsub, is_xview = res
                if is_xsub:
                    cnt_sub += c; sum_sub += s; ss_sub += ss
                if is_xview:
                    cnt_view += c; sum_view += s; ss_view += ss

    # >> X-Sub 프로토콜에 대한 통계를 계산하고 저장한다.
    mean_sub = sum_sub / (cnt_sub + 1e-8)
    std_sub = np.sqrt(np.maximum((ss_sub / (cnt_sub + 1e-8)) - mean_sub**2, 0)) + 1e-8
    np.savez(STATS_FILE_XSUB, mean=mean_sub, std=std_sub)
    print(f"X-Sub Stats saved: {STATS_FILE_XSUB}")

    # >> X-View 프로토콜에 대한 통계를 계산하고 저장한다.
    mean_view = sum_view / (cnt_view + 1e-8)
    std_view = np.sqrt(np.maximum((ss_view / (cnt_view + 1e-8)) - mean_view**2, 0)) + 1e-8
    np.savez(STATS_FILE_XVIEW, mean=mean_view, std=std_view)
    print(f"X-View Stats saved: {STATS_FILE_XVIEW}")

# >> 개별 파일을 전처리하고 .pt 형식으로 저장한다.
def process_and_save_file(filename):
    if not filename.endswith('.skeleton'): return
    
    try:
        path = os.path.join(SOURCE_DATA_PATH, filename)
        coords = _read_skeleton_file(path)
        label = int(filename[17:20]) - 1

        if coords.shape[0] == 0:
            feat = np.zeros((MAX_FRAMES, NUM_JOINTS, config.NUM_COORDS))
        else:
            # >> 전체 시퀀스를 MAX_FRAMES 길이로 Resize한다.
            resized_coords = resize_data_skateformer_style(coords, target_frames=MAX_FRAMES)
            
            # >> Resize된 좌표를 기반으로 특징을 계산한다.
            feat = _calculate_features(resized_coords) 

        torch.save({
            'data': torch.from_numpy(feat).float(),
            'label': label
        }, os.path.join(TARGET_DATA_PATH, filename.replace('.skeleton', '.pt')))
    except Exception as e:
        # >> 에러 발생 시 로그를 출력하고 처리를 계속한다.
        print(f"Error saving {filename}: {e}")
        # traceback.print_exc() # 상세 에러 필요시 주석 해제

# >> 메인 실행 함수이다.
def main():
    if not os.path.exists(TARGET_DATA_PATH):
        os.makedirs(TARGET_DATA_PATH)
        
    # >> 통계 파일이 없으면 새로 계산한다.
    if not os.path.exists(STATS_FILE_XSUB) or not os.path.exists(STATS_FILE_XVIEW):
        calculate_and_save_stats()
        
    print(f"--- Processing Files (Strategy: Full Sequence Resize to {MAX_FRAMES} frames) ---")
    filenames = os.listdir(SOURCE_DATA_PATH)
    num_cores = cpu_count() - 1 if cpu_count() > 1 else 1
    
    # >> 멀티프로세싱을 사용하여 전체 파일을 병렬로 처리한다.
    with Pool(num_cores) as pool:
        list(tqdm(pool.imap_unordered(process_and_save_file, filenames), total=len(filenames)))

if __name__ == '__main__':
    main()
