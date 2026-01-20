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

# >> 통계 파일 경로 분리
STATS_FILE_XSUB = '../stats_xsub.npz'
STATS_FILE_XVIEW = '../stats_xview.npz'

MAX_FRAMES = config.MAX_FRAMES 
NUM_JOINTS = config.NUM_JOINTS
BASE_NUM_JOINTS = 25

# >> Protocol Definitions
TRAINING_SUBJECTS = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
TRAINING_CAMERAS = [2, 3] 

# >> 뼈 연결 정보
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

    # >> 1차 스캔: Actor 찾기
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
    
    # >> 2차 스캔: 좌표 추출
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
    
    # >> 유효 구간의 시작과 끝
    begin = valid_indices[0]
    end = valid_indices[-1] + 1
    
    # >> 유효 구간 Crop (자르기)
    valid_data = data_numpy[begin:end] # Shape: (Real_T, M, V, C)
    
    # >> Resize (Linear Interpolation)
    # >> PyTorch interpolate 사용을 위해 차원 변환: (Batch, Channel, Length)
    # >> 여기서는 Batch=1, Channel=All Features(M*V*C), Length=Time
    data_torch = torch.from_numpy(valid_data).float()
    
    # >> (Real_T, M, V, C) -> (1, M, V, C, Real_T) 아님. Permute 주의.
    # >> permute(1, 2, 3, 0) -> (M, V, C, Real_T)
    # >> view(1, -1, Real_T) -> (1, M*V*C, Real_T) == (Batch, Channel, Length)
    data_torch = data_torch.permute(1, 2, 3, 0).contiguous().view(1, M * V * C, -1)
    
    # >> 보간 수행
    data_resized = F.interpolate(data_torch, size=target_frames, mode='linear', align_corners=False)
    
    # >> 원래 차원으로 복구: (1, M*V*C, Target_T) -> (Target_T, M, V, C)
    data_resized = data_resized.view(M, V, C, target_frames).permute(3, 0, 1, 2).contiguous()
    
    return data_resized.numpy()

def _calculate_features(coords):
    """12차원 벡터 특징 계산 함수"""
    T = coords.shape[0]
    if T == 0: # 데이터가 비어 있는 경우,
        return np.zeros((0, NUM_JOINTS, config.NUM_COORDS))

    # >> 1. 척추 길이 계산
    spine_vec = coords[:, :, 20, :] - coords[:, :, 0, :] # 20번 - 0번 해서 척추 벡터 계산
    spine_len = np.linalg.norm(spine_vec, axis=-1, keepdims=True) # 척추 벡터의 유클리드 거리 계산
    
    valid_mask = (spine_len > 0.01).astype(np.float32) # 척추 길이가 0.01보다 크면 1.0을 반환. 아니라면 0.0을 반환
    safe_spine_len = spine_len + 1e-8 # 척추 길이에 작은 값을 더해서 0으로 나누는 것을 방지
    safe_spine_len = safe_spine_len[..., np.newaxis] # 브로드캐스팅 연산을 위해 차원을 하나 추가

    # >> A. Bone Vector
    bone_features = np.zeros((T, 2, BASE_NUM_JOINTS, 3)) # 0으로 초기화
    for parent, child in SKELETON_BONES:
        vec = coords[:, :, child, :] - coords[:, :, parent, :]
        bone_features[:, :, child, :] = vec
    bone_features = bone_features / safe_spine_len # 척추 길이로 나누어 정규화

    # >> B. Velocity Vector
    # >> Resize된 좌표 상에서의 속도이므로 시간 간격이 정규화된 상태의 변화량이다.
    velocity = np.zeros_like(coords)
    velocity[1:] = coords[1:] - coords[:-1] # 현재 프레임에서 이전 프레임을 뺀다
    velocity_features = velocity / safe_spine_len # 척추 길이로 나누어 정규화

    # >> C. Relative Center
    p0_center = coords[:, 0, 0, :] # 사람 0의 0번 관절
    p1_center = coords[:, 1, 0, :] # 사람 1의 0번 관절
    
    rel_center_0to1 = (p1_center - p0_center) # 사람 0에서 사람 1로 향하는 벡터
    rel_center_1to0 = (p0_center - p1_center) # 사람 1에서 사람 0으로 향하는 벡터
    
    rel_center_0to1 = rel_center_0to1[:, np.newaxis, :] / safe_spine_len[:, 0, :, :] # 정규화
    rel_center_1to0 = rel_center_1to0[:, np.newaxis, :] / safe_spine_len[:, 1, :, :] # 정규화 
    
    rc_feat_p0 = np.broadcast_to(rel_center_0to1, (T, BASE_NUM_JOINTS, 3)) # 브로드캐스트를 통해 모든 관절에 값을 복사 
    rc_feat_p1 = np.broadcast_to(rel_center_1to0, (T, BASE_NUM_JOINTS, 3))
    rel_center_features = np.stack([rc_feat_p0, rc_feat_p1], axis=1) # 두 사람의 특징을 합쳐서 (T, 2, 25, 3) 형태로 만들기 

    # >> D. Relative To Other
    rel_other_p0 = (coords[:, 0, :, :] - p1_center[:, np.newaxis, :]) # 사람 0의 모든 관절과 사람 1의 중심 좌표를 뺀다
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

    final_features_per_person *= valid_mask[..., np.newaxis] # 유효성 마스크를 곱해서 척추 길이가 0.01 이하인 것의 프레임 데이터를 0으로 처리 

    person1 = final_features_per_person[:, 0, :, :] # (T, 2, 25, 12)를 (T, 50, 12)으로 만들기 위해 사람별로 분리 
    person2 = final_features_per_person[:, 1, :, :]
    
    return np.concatenate((person1, person2), axis=1) # (T, 50, 12)로 만들기 

def process_file_for_stats(filename):
    if not filename.endswith('.skeleton'): return None # 파일 확장자가 .skeleton인지 검사 
    
    try:
        sid = int(filename[9:12]) # 9,10,11 자리 이름을 sid에 저장 
        cid = int(filename[5:8]) # 5,6,7 자리 이름을 cid에 저장 
        
        is_xsub_train = sid in TRAINING_SUBJECTS # sid가 Training subject 리스트에 포함되면 is_xsub_train을 True로 설정 
        is_xview_train = cid in TRAINING_CAMERAS
        
        if not is_xsub_train and not is_xview_train: return None # sid, cid 모두 충족하지 않으면 none 반환 
        
        path = os.path.join(SOURCE_DATA_PATH, filename)
        coords = _read_skeleton_file(path)
        if coords.shape[0] == 0: return None
        
        # >> 전체 좌표를 넘겨서 Resize 수행 (config.MAX_FRAMES)
        resized_coords = resize_data_skateformer_style(coords, target_frames=MAX_FRAMES) # MAX_FRAMES 길이로 보간된 데이터를 기준으로 통계를 계산 
        features = _calculate_features(resized_coords) # 12차원 특징 추출 
        
        features_flat = features.reshape(-1, config.NUM_COORDS) # 데이터를 (T * J, C) 형태로 평탄화 
        mask = np.abs(features_flat).sum(axis=1) > 1e-6 # 특징 값의 합이 1e-6보다 작으면 계산에서 제외
        valid_data = features_flat[mask]
        
        if valid_data.shape[0] == 0: return None

        # >> valid_data.shape[0]은 유효한 데이터 샘플의 개수
        # >> valid_data.sum(axis=0)은 각 채널별 값의 합계
        # >> np.sum(valid_data**2, axis=0)은 각 채널별 값의 제곱 합
        # >> is_xsub_train, is_xview_train은 XSUB, XVIEW 데이터의 여부
        return (valid_data.shape[0], valid_data.sum(axis=0), np.sum(valid_data**2, axis=0), is_xsub_train, is_xview_train)
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None

def calculate_and_save_stats():
    print("--- Calculating Stats for 12D Features (Separate for X-Sub / X-View) ---")
    filenames = os.listdir(SOURCE_DATA_PATH) # PATH에 있는 모든 파일 목록을 가져오기 

    # >> XSUB, XVIEW에서 쓸 누적 변수를 0으로 초기화
    cnt_sub, sum_sub, ss_sub = np.zeros(config.NUM_COORDS), np.zeros(config.NUM_COORDS), np.zeros(config.NUM_COORDS)
    cnt_view, sum_view, ss_view = np.zeros(config.NUM_COORDS), np.zeros(config.NUM_COORDS), np.zeros(config.NUM_COORDS)

    # >> CPU를 최대한 많이 사용하도록 함 
    num_cores = cpu_count() - 1 if cpu_count() > 1 else 1

    # >> 병렬처리를 하고 결과를 합침 
    with Pool(num_cores) as pool:
        for res in tqdm(pool.imap_unordered(process_file_for_stats, filenames), total=len(filenames)):
            if res: # 반환 결과가 0이 아니면, 즉 학습 데이터가 유효하면 
                c, s, ss, is_xsub, is_xview = res
                if is_xsub:
                    cnt_sub += c; sum_sub += s; ss_sub += ss
                if is_xview:
                    cnt_view += c; sum_view += s; ss_view += ss

    mean_sub = sum_sub / (cnt_sub + 1e-8) # 평균 계산 
    std_sub = np.sqrt(np.maximum((ss_sub / (cnt_sub + 1e-8)) - mean_sub**2, 0)) + 1e-8 # 표준편차 계산 
    np.savez(STATS_FILE_XSUB, mean=mean_sub, std=std_sub) # .npz 파일로 저장 
    print(f"X-Sub Stats saved: {STATS_FILE_XSUB}")

    mean_view = sum_view / (cnt_view + 1e-8)
    std_view = np.sqrt(np.maximum((ss_view / (cnt_view + 1e-8)) - mean_view**2, 0)) + 1e-8
    np.savez(STATS_FILE_XVIEW, mean=mean_view, std=std_view)
    print(f"X-View Stats saved: {STATS_FILE_XVIEW}")


    
def process_and_save_file(filename):
    if not filename.endswith('.skeleton'): return # 확장자가 .skeleton으로 끝나는지 검사 
    
    try: # 파일 처리 중간에 에러가 발생해도 멈추지 않도록 try 안에 작성
        path = os.path.join(SOURCE_DATA_PATH, filename) # Source_data_path와 filename을 합쳐서 절대경로를 만든다
        coords = _read_skeleton_file(path) # 함수 호출을 해서 3D 관절 좌표 데이터를 읽어온다 
        label = int(filename[17:20]) - 1 # 정답 레이블을 추출한다 

        if coords.shape[0] == 0: # 프레임 수가 0이라서 파일이 손상된 경우 모든 값이 0인 더미 데이터를 생성. 에러는 발생하지 않음 
            feat = np.zeros((MAX_FRAMES, NUM_JOINTS, config.NUM_COORDS))
        else:
            # >> config.MAX_FRAMES에 맞춰 전체 시퀀스 Resize
            resized_coords = resize_data_skateformer_style(coords, target_frames=MAX_FRAMES)
            
            # >> Resize된 좌표로 특징 계산
            feat = _calculate_features(resized_coords) 

        # >> 처리된 데이터를 최종적으로 저장 
        torch.save({ # 딕셔너리를 직렬화해서 저장 
            'data': torch.from_numpy(feat).float(),
            'label': label
        }, os.path.join(TARGET_DATA_PATH, filename.replace('.skeleton', '.pt'))) # 확장자를 .skeleton에서 .pt로 변환 
    except Exception as e:
        # >> 하나가 실패해도 멈추지 않고 로그만 남김
        print(f"Error saving {filename}: {e}")

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
