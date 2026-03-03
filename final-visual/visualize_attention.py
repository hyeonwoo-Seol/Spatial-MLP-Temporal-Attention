import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import config
from model import ST_Model
from ntu_data_loader import NTURGBDDataset

# ==========================================
# 1. 설정
# ==========================================
# 학습된 모델 경로가 있다면 수정하세요. 없으면 랜덤 가중치로 실행됩니다.
CHECKPOINT_PATH = 'best_model.pth.tar' 
sample_index = 0  # 시각화할 샘플 인덱스 (변경 가능)

# 뼈대 연결 정보 (preprocess_ntu_data.py 기준)
SKELETON_BONES = [
    (20, 1), (1, 0), (20, 2), (2, 3), (20, 4), (4, 5), (5, 6), (6, 7), 
    (7, 21), (7, 22), (20, 8), (8, 9), (9, 10), (10, 11), (11, 23), (11, 24),
    (0, 12), (12, 13), (13, 14), (14, 15), (0, 16), (16, 17), (17, 18), (18, 19)
]

# ==========================================
# 2. Hook 설정 (중요: 내부 Weight 추출)
# ==========================================
attention_weights = []

def get_attention_weights_hook(module, input, output):
    """
    AttentivePooling의 forward 내부 변수를 가져오기 어렵기 때문에,
    model.py를 수정하지 않고 입력값(x)을 이용해 가중치를 재계산하거나,
    AttentivePooling이 weights를 리턴하도록 수정해야 합니다.
    
    하지만 현우님은 model.py를 수정할 수 있으므로, 
    가장 확실한 방법은 model.py의 AttentivePooling을 잠시 수정하거나
    아래처럼 '재계산' 로직을 훅에서 수행하는 것입니다.
    """
    # input[0]은 pooling 직전의 feature (N, T, D) 입니다.
    x = input[0]
    
    # AttentivePooling의 로직을 그대로 재현하여 weights 추출
    # self.proxy_classifier는 module.proxy_classifier로 접근
    logits = module.proxy_classifier(x)
    probs = F.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
    confidence = 1.0 / (1.0 + entropy)
    weights = 1.0 + confidence
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
    
    # 추출된 weights를 전역 리스트에 저장 (N, T)
    attention_weights.append(weights.detach().cpu().numpy())

# ==========================================
# 3. 모델 및 데이터 로드
# ==========================================
device = config.DEVICE
model = ST_Model(num_classes=config.NUM_CLASSES).to(device)

# Hook 등록
model.attentive_pooling.register_forward_hook(get_attention_weights_hook)

# 체크포인트 로드 시도
try:
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    # state_dict 키 매칭 (module. 접두사 처리 등)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint from {CHECKPOINT_PATH}")
except FileNotFoundError:
    print("Checkpoint not found. Using random weights for demonstration.")

model.eval()

# 데이터셋 로드 (Validation set)
dataset = NTURGBDDataset(data_path=config.DATASET_PATH, split='val', protocol='xview')
features, label, _ = dataset[sample_index]

# ==========================================
# 4. Inference 및 가중치 추출
# ==========================================
input_tensor = features.unsqueeze(0).to(device) # (1, C, T, V)

with torch.no_grad():
    output = model(input_tensor) # Hook이 여기서 실행됨

# 추출된 가중치 가져오기
# shape: (1, T_downsampled) -> 보통 (1, 32)
att_w = attention_weights[0][0]  # 첫번째 배치의 가중치
T_down = len(att_w)
T_original = config.MAX_FRAMES

# 시간축 Upsampling (32 -> 64)
# 단순 반복(repeat)보다는 선형 보간이 시각적으로 부드러움
att_w_tensor = torch.tensor(att_w).view(1, 1, -1) # (1, 1, 32)
att_w_upsampled = F.interpolate(att_w_tensor, size=T_original, mode='linear', align_corners=False)
att_w_final = att_w_upsampled.squeeze().numpy() # (64,)

# 정규화 (그래프 그리기 좋게 0~1로 스케일링, 선택사항)
att_w_final = (att_w_final - att_w_final.min()) / (att_w_final.max() - att_w_final.min() + 1e-8)

print(f"Action Label: {label}")
print(f"Attention Weights Shape: {att_w_final.shape}")

# ==========================================
# 5. 시각화 (Skeleton + Bar Graph)
# ==========================================
# 데이터 차원 변환: (C, T, V) -> (T, V, C) -> numpy
skeleton_data = features.permute(1, 2, 0).numpy()

fig = plt.figure(figsize=(10, 8))
gs = GridSpec(3, 1, figure=fig)

# 상단: 3D Skeleton
ax_3d = fig.add_subplot(gs[0:2, :], projection='3d')
# 하단: Attention Weights Bar
ax_chart = fig.add_subplot(gs[2, :])

def update(frame):
    ax_3d.clear()
    ax_chart.clear()
    
    # --- 1. Skeleton 그리기 ---
    current_skeleton = skeleton_data[frame] # (V, 3)
    
    # 사람이 2명(50 joints)일 경우 분리해서 그림
    num_people = 2 if current_skeleton.shape[0] == 50 else 1
    
    for p in range(num_people):
        start_idx = p * 25
        end_idx = (p + 1) * 25
        person_skeleton = current_skeleton[start_idx:end_idx]
        
        # 0인 데이터(사람 없음) 건너뛰기
        if np.sum(np.abs(person_skeleton)) < 1e-5: continue

        x = person_skeleton[:, 0]
        y = person_skeleton[:, 1]
        z = person_skeleton[:, 2]
        
        # Skeleton Plot (Kinect 좌표계 고려: y와 z를 바꿔서 그리는 경우가 많음)
        ax_3d.scatter(x, z, y, c='b' if p==0 else 'r', s=20)
        
        for s, e in SKELETON_BONES:
            if s < 25 and e < 25:
                ax_3d.plot([x[s], x[e]], [z[s], z[e]], [y[s], y[e]], c='k')

    ax_3d.set_xlim(-1, 1)
    ax_3d.set_ylim(-1, 1)
    ax_3d.set_zlim(-1, 1)
    ax_3d.set_title(f"Frame {frame}/{T_original} | Action Label: {label}")
    ax_3d.set_xlabel('X')
    ax_3d.set_ylabel('Z') # Depth
    ax_3d.set_zlabel('Y') # Height

    # --- 2. Bar Chart 그리기 ---
    # 전체 프레임의 중요도를 회색으로 깔아둠
    ax_chart.bar(range(T_original), att_w_final, color='lightgray', width=1.0)
    # 현재 프레임을 빨간색으로 강조
    ax_chart.bar(frame, att_w_final[frame], color='red', width=1.0)
    
    ax_chart.set_xlim(0, T_original)
    ax_chart.set_ylim(0, 1.1)
    ax_chart.set_title("Temporal Attention Weight (Importance)")
    ax_chart.set_xlabel("Frame Index")
    ax_chart.set_ylabel("Normalized Weight")

ani = animation.FuncAnimation(fig, update, frames=T_original, interval=100)

# 저장 (GIF 또는 MP4)
save_path = 'attention_vis.gif'
ani.save(save_path, writer='pillow', fps=15)
print(f"Visualization saved to {save_path}")
plt.close()
