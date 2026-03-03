import os
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn.functional as F

# ==========================================
# 1. 설정 및 데이터 경로 (수정 필요)
# ==========================================
# 실제 .skeleton 파일들이 들어있는 폴더 경로로 변경해주세요.
SOURCE_DATA_PATH = '../../paper-review/Action_Recognition/Code/nturgbd01/'
MAX_FRAMES = 64 
BASE_NUM_JOINTS = 25

# preprocess_ntu_data.py에 정의된 뼈대 연결 정보
SKELETON_BONES = [
    (20, 1), (1, 0), (20, 2), (2, 3),
    (20, 4), (4, 5), (5, 6), (6, 7), (7, 21), (7, 22),
    (20, 8), (8, 9), (9, 10), (10, 11), (11, 23), (11, 24),
    (0, 12), (12, 13), (13, 14), (14, 15),
    (0, 16), (16, 17), (17, 18), (18, 19)
]

# ==========================================
# 2. Helper Functions (preprocess_ntu_data.py 기반)
# ==========================================

def read_skeleton_file(filepath):
    """원본 .skeleton 파일을 읽어오는 함수 (핵심 로직만 간소화)"""
    if not os.path.exists(filepath):
        return None
        
    with open(filepath, 'r') as f:
        first_line = f.readline()
        if not first_line: return None
        num_frames = int(first_line)
        
        # 데이터 담을 배열: (Frame, Body(max 2), Joint(25), XYZ(3))
        # 편의상 Body는 최대 2명까지만 읽습니다.
        data = np.zeros((num_frames, 2, BASE_NUM_JOINTS, 3))
        
        for t in range(num_frames):
            line = f.readline()
            if not line: break
            num_bodies = int(line.strip())
            
            for b in range(num_bodies):
                body_info = f.readline().strip().split()
                body_id = body_info[0] # 사용하지 않음
                num_joints = int(f.readline().strip())
                
                # 2명 이상이면 앞의 2명만 저장 (간소화)
                if b >= 2: 
                    for _ in range(num_joints): f.readline()
                    continue
                
                for j in range(num_joints):
                    coords = list(map(float, f.readline().split()[:3]))
                    if j < BASE_NUM_JOINTS:
                        data[t, b, j] = coords
                        
    return data

def resize_data(data_numpy, target_frames=MAX_FRAMES):
    """선형 보간 (Linear Interpolation) 함수"""
    T, M, V, C = data_numpy.shape
    
    # 유효 구간 찾기 (모두 0인 프레임 제거)
    valid_mask = np.sum(np.abs(data_numpy), axis=(1, 2, 3)) != 0
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) == 0:
        return np.zeros((target_frames, M, V, C))
    
    begin = valid_indices[0]
    end = valid_indices[-1] + 1
    valid_data = data_numpy[begin:end]
    
    # Torch Tensor로 변환 및 차원 변경
    data_torch = torch.from_numpy(valid_data).float()
    # (T, M, V, C) -> (1, M*V*C, T)
    data_torch = data_torch.permute(1, 2, 3, 0).contiguous().view(1, M * V * C, -1)
    
    # Interpolate
    data_resized = F.interpolate(data_torch, size=target_frames, mode='linear', align_corners=False)
    
    # 복구: (1, M*V*C, T) -> (T, M, V, C)
    data_resized = data_resized.view(M, V, C, target_frames).permute(3, 0, 1, 2).contiguous()
    
    return data_resized.numpy()

def normalize_skeleton(coords):
    """
    척추 길이 정규화 적용 함수
    (preprocess_ntu_data.py의 _calculate_features 로직 일부 차용)
    """
    # 1. 척추 길이 계산 (Joint 20 - Joint 0)
    # coords shape: (T, M, V, C)
    spine_vec = coords[:, :, 20, :] - coords[:, :, 0, :]
    spine_len = np.linalg.norm(spine_vec, axis=-1, keepdims=True)
    safe_spine_len = spine_len + 1e-8

    safe_spine_len = safe_spine_len[..., np.newaxis]
    
    # 2. 정규화: 모든 좌표를 척추 길이로 나눔
    # 시각화를 위해, 중심점(Joint 0: SpineBase)을 원점으로 이동시킨 후 스케일링
    center = coords[:, :, 0:1, :] # (T, M, 1, 3)
    normalized_coords = (coords - center) / safe_spine_len
    
    return normalized_coords

# ==========================================
# 3. Visualization Function
# ==========================================
def visualize_comparison(raw_data, processed_data, file_name):
    fig = plt.figure(figsize=(14, 7))
    
    # --- 1) Left Plot: Raw Data ---
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title(f"Raw Data (Original Scale)\nFile: {file_name}")
    
    # Raw Data의 중간 프레임 선택
    frame_idx_raw = raw_data.shape[0] // 2
    skeleton_raw = raw_data[frame_idx_raw] # (M, V, C)
    
    for m in range(skeleton_raw.shape[0]): # 사람 수 만큼 반복 (보통 1~2명)
        # 모든 관절이 0인 경우(사람 없음) 건너뛰기
        if np.sum(np.abs(skeleton_raw[m])) == 0: continue
            
        x = skeleton_raw[m, :, 0]
        y = skeleton_raw[m, :, 1]
        z = skeleton_raw[m, :, 2]
        
        # 관절 점 찍기
        ax1.scatter(x, z, y, c='r', marker='o', s=20) # Y와 Z를 바꿔서 그리는 경우가 많음 (Kinect 좌표계 특성)
        
        # 뼈대 연결하기
        for start, end in SKELETON_BONES:
            if start < 25 and end < 25:
                ax1.plot([x[start], x[end]], [z[start], z[end]], [y[start], y[end]], c='b')

    ax1.set_xlabel('X')
    ax1.set_ylabel('Z (Depth)')
    ax1.set_zlabel('Y (Height)')
    # 비율 고정 (형태 왜곡 방지)
    # ax1.set_box_aspect([1,1,1]) 

    # --- 2) Right Plot: Processed Data ---
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title(f"Processed (Interpolated & Normalized)\nFrames: {MAX_FRAMES}, Unit Scale")
    
    # Processed Data의 중간 프레임 선택
    frame_idx_proc = processed_data.shape[0] // 2
    skeleton_proc = processed_data[frame_idx_proc] # (M, V, C)

    for m in range(skeleton_proc.shape[0]):
        if np.sum(np.abs(skeleton_proc[m])) == 0: continue
            
        x = skeleton_proc[m, :, 0]
        y = skeleton_proc[m, :, 1]
        z = skeleton_proc[m, :, 2]
        
        ax2.scatter(x, z, y, c='g', marker='o', s=20)
        
        for start, end in SKELETON_BONES:
            if start < 25 and end < 25:
                ax2.plot([x[start], x[end]], [z[start], z[end]], [y[start], y[end]], c='k')

    ax2.set_xlabel('X (Norm)')
    ax2.set_ylabel('Z (Norm)')
    ax2.set_zlabel('Y (Norm)')
    
    # 정규화된 데이터는 범위가 작으므로 축 범위를 -1 ~ 1 정도로 제한하면 보기 좋음
    ax2.set_xlim(-1, 1)
    ax2.set_ylim(-1, 1)
    ax2.set_zlim(-1, 1)

    plt.tight_layout()
    plt.show()

# ==========================================
# 4. Main Execution
# ==========================================
def main():
    if not os.path.exists(SOURCE_DATA_PATH):
        print(f"Error: Path '{SOURCE_DATA_PATH}' not found. Please update 'SOURCE_DATA_PATH'.")
        return

    # 1. 파일 목록 가져오기
    files = [f for f in os.listdir(SOURCE_DATA_PATH) if f.endswith('.skeleton')]
    if not files:
        print("No .skeleton files found.")
        return

    # 2. 랜덤 샘플링
    target_file = random.choice(files)
    file_path = os.path.join(SOURCE_DATA_PATH, target_file)
    print(f"Selected Sample: {target_file}")

    # 3. Raw Data 읽기
    raw_data = read_skeleton_file(file_path)
    if raw_data is None:
        print("Failed to read file.")
        return

    # 4. Process Data (보간 -> 정규화)
    # (1) Resize (Linear Interpolation)
    resized_data = resize_data(raw_data, target_frames=MAX_FRAMES)
    # (2) Normalize (Spine Length)
    normalized_data = normalize_skeleton(resized_data)

    # 5. 시각화
    visualize_comparison(raw_data, normalized_data, target_file)

if __name__ == "__main__":
    main()
