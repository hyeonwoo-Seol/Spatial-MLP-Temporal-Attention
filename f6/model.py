# >> model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import config

# >> RMSNorm
# >> 평균을 뺀 분산 대신 제곱 평균을 사용하는 정규화 기법
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps # 0으로 나누는 것을 방지하는 값
        self.weight = nn.Parameter(torch.ones(d)) # 학습 가능한 파라미터를 생성하고 1로 초기화

    def forward(self, x):
        rsqrt = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) # 입력x를 제곱하고 마지막 차원(-1)에 대해 평균을 구한 뒤 e를 더하고 역제곱근(sqrt)를 계산
        return x * rsqrt * self.weight # 입력 x에 정규화 계수를 곱하고 학습 가능한 가중치 weight를 곱하여 반환

# >> Temporal Downsampling Layer
# >> 시간을 압축하고 연산량을 줄이기 위해 사용한다
class TemporalDownsample(nn.Module):
    """
    Applies strided 1D convolution to reduce temporal resolution by half.
    Input: (B, T, D) where B is usually N * V
    Output: (B, T//2, D)
    """
    def __init__(self, dim, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=kernel_size, stride=stride, padding=padding) # 1D 합성곱 정의
        self.norm = RMSNorm(dim) # 합성곱 통과 후 적용할 정규화 층 정의

    def forward(self, x):
        # x: (B, T, D)
        B, T, D = x.shape # 입력 텐서의 크기를 배치, 시간, 차원으로 분해
        
        x = x.transpose(1, 2) # (B,T,D)를 (B,D,T)로 전치
        x = self.conv(x) # 1D 합성곱 통과해서 시간 축을 절반으로 줄이기
        
        x = x.transpose(1, 2) # 다시 (B,T,D) 형태로 전치
        x = self.norm(x) # 정규화 수행
        return x

# >> Embedding Layer
# >> 원본 좌표 데이터를 모델의 은닉 차원으로 변환하고 위치 정보를 더해준다.
class SpatioTemporalEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_joints, max_frames, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim) # 입력 좌표를 모델 내부 차원으로 선형 변환한다
        
        self.spatial_pe = nn.Parameter(torch.zeros(1, 1, num_joints, hidden_dim)) # 공간적 위치 인코딩을 위한 학습 가능한 파라미터 (1, 1, V, D)
        
        self.dropout = nn.Dropout(dropout) # 과적합 방지를 위한 Dropout
        self._init_weights() # 가중치 초기화

    def _init_weights(self):
        nn.init.trunc_normal_(self.spatial_pe, std=0.02) # 위치인코딩 값을 표준편차 0.02의 절단 정규분포로 추기화

    def forward(self, x):
        # x: (N, C, T, V) -> (N, T, V, C)
        N, C, T, V = x.shape # 입력 데이터에서 (배치, 채널, 시간, 관절)을 가져온다
        x = x.permute(0, 2, 3, 1).contiguous() # 차원 순서를 (배치, 시간, 관절, 채널)으로 변경한다
        x = self.input_proj(x) # 좌표 채널 차원을 은닉 차원으로 변환한다
        x = x + self.spatial_pe[:, :, :V, :] # 각 관절 위치에 해당하는 학습된 위치 정보를 더한다
        return self.dropout(x) # Dropout 적용한다.

# >> Spatial Stream: Pure MLP Mixer
# >> 관절 간의 관계를 학습한다.
class SpatialMixerBlock(nn.Module):
    def __init__(self, dim, num_joints, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_joints = num_joints
        
        # Token Mixing MLP: 관절(V) 간의 정보 교환하는 MLP다.
        self.norm1 = RMSNorm(dim)
        self.token_mixing_mlp = nn.Sequential(
            nn.Linear(num_joints, num_joints),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(num_joints, num_joints)
        )
        
        # Channel Mixing MLP: 채널(D) 간의 정보 교환하는 MLP다.
        self.norm2 = RMSNorm(dim)
        self.channel_mixing_mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        N, T, V, D = x.shape # 입력을 가져온다.
        
        # 1. Token Mixing (Spatial Interaction)
        # (N, T, V, D) -> (N*T, D, V) -> MLP(V->V) -> (N, T, V, D)
        residual = x # 잔차 연결을 위해 입력을 저장한다.
        x = self.norm1(x) # 정규화를 수행한다.
        x = x.reshape(N * T, V, D).permute(0, 2, 1) # 관절 축이 마지막 차원에 오도록 형태를 변형한다. MLP는 마지막 차원에 대해 연산하기 때문
        x = self.token_mixing_mlp(x) # MLP를 통해 관절 간 정보를 교환한다.
        x = x.permute(0, 2, 1).reshape(N, T, V, D) # 원래 모양으로 복구한다
        x = residual + x # 잔차 연결을 수행한다
        
        # 2. Channel Mixing
        residual = x
        x = self.norm2(x)
        x = self.channel_mixing_mlp(x) # 채널 간 정보를 교환하는 MLP를 통과한다
        x = residual + x
        
        return x

# >> Temporal Stream: Multi-Scale Swin Transformer
class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, dropout=0.1):
        super().__init__()
        self.dim = dim # 입력 차원 크기
        self.window_size = window_size # 윈도우 크기
        self.num_heads = num_heads # 멀티 헤드 어탠션 개수
        head_dim = dim // num_heads # 전체 차원을 헤드 수로 나누어 각 헤드가 담당할 차원 크기
        self.scale = head_dim ** -0.5 # 어텐션 점수 스케일링 팩터를 계산. 내적 값이 너무 커지는 것을 방지

        # >> 윈도우 내의 상대적 위치에 따른 편향을 학습하는 파라미터 테이블이다.
        # >> 0으로 초기화한다
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(2 * window_size - 1, num_heads)
        )
        
        coords = torch.arange(self.window_size) # 0부터 윈도우 사이즈 -1까지의 정수 좌표를 생성한다
        relative_coords = coords[:, None] - coords[None, :] # 브로드캐스팅을 이용해 모든 위치 쌍 간의 상대좌표를 계산한다
        relative_coords += self.window_size - 1 # 계산된 좌표에 윈도우 사이즈 -1 을 더해서 모든 값을 0 이상의 양수로 만든다
        self.register_buffer("relative_position_index", relative_coords) 

        self.qkv = nn.Linear(dim, dim * 3, bias=True) # 입력 dim을 dim*3으로 변환하는 선형 레이어를 정의한다
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim) # 어텐션 연산 후 결과를 다시 원래 차원으로 섞어주는 출력 선형 레이어다.
        self.proj_drop = nn.Dropout(dropout)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02) # 위치 편향 테이블을 표준편차 0.002의 절단 정규분포로 초기화한다
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        # >> 선형 변환을 수행한다.
        # >> 변환 결과를 (B_, N, 3, 헤드 수, 헤드 차원)으로 변형한다
        # >> 차원 순서를 (3, B_, 헤드 수, N, 헤드 차원)으로 변환한다.
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # 텐서를 첫 번째 차원을 기준으로 분리해서 각 Query, Key, Value에 할당한다

        q = q * self.scale # Query에 스케일링 팩터를 곱해 내적 값의 크기를 조절한다
        attn = (q @ k.transpose(-2, -1)) # Qeury와 전치된 Key를 행렬곱 해서 어텐션 점수를 계산한다

        # >> view(-1)로 인덱스를 1차원으로 펼친 뒤 테이블에서 값을 조회한다 그리고 다시 .view를 호출해 형상을 맞춘다
        # >> 조회한 편향 값을 (윈도우크기, 윈도우 크기, 헤드 수) 형태로 변형한다
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size, self.window_size, -1
        )
        # >> 편향 텐서의 차원 수를 (헤드 수, 윈도우 크기, 윈도우 크기)로 변경하고 메모리에 연속적으로 배치한다
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0) # 계산된 어탠션 점수에 위치 편향을 더한다.

        # >> 마스크가 None이 아닐 경우
        if mask is not None:
            nW = mask.shape[0] # 마스크 텐서의 첫 번째 차원을 가져온다
            attn = attn.reshape(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0) # 어탠션 점수를 윈도우 단위로 분리한 뒤 마스크를 더한다
            attn = attn.reshape(-1, self.num_heads, N, N) # 마스킹 후 다시 원래 차원으로 되돌린다.
            
        attn = self.softmax(attn) # 어탠션 점수에 Softmax를 적용해서 확률값으로 변환한다
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTemporalBlock(nn.Module):
    def __init__(self, dim, num_heads, window_sizes=[10, 20], shift_sizes=[0, 0], dropout=0.1, bottleneck_ratio=0.5):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        
        # [Multi-Scale] 서로 다른 Window Size와 Shift Size를 리스트로 받음
        self.window_sizes = window_sizes
        self.shift_sizes = shift_sizes
        
        # Bottleneck: 채널 축소 (예: 128 -> 64)
        self.dim_inner = int(dim * bottleneck_ratio)
        
        # 각 브랜치(헤드 그룹)별 채널 및 헤드 수 (반으로 분할)
        self.dim_branch = self.dim_inner // 2
        self.heads_branch = num_heads // 2
        
        # Linear Projection for Bottleneck (Down / Up)
        self.proj_down = nn.Linear(dim, self.dim_inner)
        self.proj_up = nn.Linear(self.dim_inner, dim)
        
        self.norm1 = RMSNorm(self.dim_inner)
        
        # [Multi-Scale] 두 개의 서로 다른 Window Attention 생성
        # Branch 1: window_sizes[0] (예: 10)
        self.attn1 = WindowAttention(
            self.dim_branch, window_sizes[0], self.heads_branch, dropout
        )
        # Branch 2: window_sizes[1] (예: 20)
        self.attn2 = WindowAttention(
            self.dim_branch, window_sizes[1], self.heads_branch, dropout
        )
        
        self.norm2 = RMSNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout)
        )

    def _process_branch(self, x, attn_layer, window_size, shift_size):
        """
        각 브랜치별로 Padding -> Shift -> Window Partition -> Attention -> Merge -> Reverse Shift 수행
        """
        B, T, C = x.shape
        
        # 1. Padding
        pad_t = (window_size - T % window_size) % window_size
        if pad_t > 0:
            x = F.pad(x, (0, 0, 0, pad_t))
        
        Bp, Tp, Cp = x.shape

        # 2. Cyclic Shift & Mask Generation
        if shift_size > 0:
            shifted_x = torch.roll(x, shifts=-shift_size, dims=1)
            
            # Mask 생성 (Dynamic T 고려)
            img_mask = torch.zeros((1, Tp, 1), device=x.device)
            t_slices = (
                slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None)
            )
            cnt = 0
            for s in t_slices:
                img_mask[:, s, :] = cnt
                cnt += 1
            
            mask_windows = img_mask.reshape(-1, window_size, 1)
            attn_mask = mask_windows - mask_windows.transpose(1, 2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            shifted_x = x
            attn_mask = None

        # 3. Window Partition
        x_windows = shifted_x.reshape(Bp, Tp // window_size, window_size, Cp)
        x_windows = x_windows.reshape(-1, window_size, Cp)
        
        # 4. Attention
        attn_windows = attn_layer(x_windows, mask=attn_mask)
        
        # 5. Merge Windows
        shifted_x = attn_windows.reshape(Bp, Tp, Cp)
        
        # 6. Reverse Shift
        if shift_size > 0:
            x_out = torch.roll(shifted_x, shifts=shift_size, dims=1)
        else:
            x_out = shifted_x

        # 7. Remove Padding
        if pad_t > 0:
            x_out = x_out[:, :T, :]
            
        return x_out

    def forward(self, x):
        # x: (N*V, T, D)
        residual = x
        
        # --- 1. Bottleneck Down ---
        # (B, T, 128) -> (B, T, 64)
        x_inner = self.proj_down(x)
        x_inner = self.norm1(x_inner)

        # --- 2. Channel Split & Multi-Scale Processing ---
        # 채널을 반으로 나눔 (예: 64 -> 32, 32)
        x1, x2 = torch.split(x_inner, self.dim_branch, dim=-1)
        
        # Branch 1 (Window Size 1)
        out1 = self._process_branch(x1, self.attn1, self.window_sizes[0], self.shift_sizes[0])
        
        # Branch 2 (Window Size 2)
        out2 = self._process_branch(x2, self.attn2, self.window_sizes[1], self.shift_sizes[1])
        
        # --- 3. Concatenation ---
        x_inner = torch.cat([out1, out2], dim=-1)

        # --- 4. Bottleneck Up ---
        # (B, T, 64) -> (B, T, 128)
        x = self.proj_up(x_inner)

        x = residual + x
        x = x + self.ffn(self.norm2(x))
        return x

# >> Attentive Pooling
class AttentivePooling(nn.Module):
    def __init__(self, dim, num_classes):
        super().__init__()
        self.dim = dim
        self.proxy_classifier = nn.Linear(dim, num_classes)
        
    def forward(self, x):
        logits = self.proxy_classifier(x)
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        confidence = 1.0 / (1.0 + entropy)
        weights = 1.0 + confidence
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        x_pooled = torch.sum(weights.unsqueeze(-1) * x, dim=1)
        return x_pooled

# >> Main Model: ST_Model (GRL Removed)
class ST_Model(nn.Module):
    def __init__(self, 
                 num_joints=config.NUM_JOINTS, 
                 num_coords=config.NUM_COORDS, 
                 num_classes=config.NUM_CLASSES, 
                 hidden_dim=128,
                 window_size=config.WINDOW_SIZE, # 기본 10
                 dropout=config.DROPOUT,
                 **kwargs):
        
        super().__init__()
        self.embedding = SpatioTemporalEmbedding(
            num_coords, hidden_dim, num_joints, config.MAX_FRAMES, dropout
        )
        
        # Multi-Scale Window 설정
        # 기본 윈도우(10)와 2배 크기 윈도우(20)를 동시에 사용
        ws_small = window_size
        ws_large = window_size * 2
        window_sizes = [ws_small, ws_large]
        
        # Shift Size 설정 (각 윈도우의 절반)
        ss_small = ws_small // 2
        ss_large = ws_large // 2

        # --- Stage 1 ---
        self.spatial_1 = SpatialMixerBlock(hidden_dim, num_joints, dropout=dropout)
        # Shift 없음: [0, 0]
        self.temporal_1 = SwinTemporalBlock(hidden_dim, num_heads=4, window_sizes=window_sizes, shift_sizes=[0, 0], dropout=dropout, bottleneck_ratio=0.5)
        
        # --- Early Downsampling (New) ---
        self.downsample = TemporalDownsample(hidden_dim, kernel_size=3, stride=2, padding=1)
        
        # --- Stage 2 ---
        self.spatial_2 = SpatialMixerBlock(hidden_dim, num_joints, dropout=dropout)
        # Shift 적용: [5, 10]
        self.temporal_2 = SwinTemporalBlock(hidden_dim, num_heads=4, window_sizes=window_sizes, shift_sizes=[ss_small, ss_large], dropout=dropout, bottleneck_ratio=0.5)
        
        # --- Stage 3 ---
        self.spatial_3 = SpatialMixerBlock(hidden_dim, num_joints, dropout=dropout)
        # Shift 없음: [0, 0]
        self.temporal_3 = SwinTemporalBlock(hidden_dim, num_heads=4, window_sizes=window_sizes, shift_sizes=[0, 0], dropout=dropout, bottleneck_ratio=0.5)

        self.attentive_pooling = AttentivePooling(hidden_dim, num_classes)

        # 1. Main Action Classifier
        self.action_head = nn.Sequential(
            RMSNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        # GRL and Aux Classifier Removed

    def forward(self, x):
        # x: (N, C, T, V)
        N, C, T, V = x.shape
        x = self.embedding(x)
        
        # --- Layer 1 ---
        x = self.spatial_1(x) 
        # Transform for Temporal: (N*V, T, D)
        x_temporal = x.permute(0, 2, 1, 3).contiguous().reshape(N * V, T, -1)
        x_temporal = self.temporal_1(x_temporal)
        
        # --- Downsampling ---
        x_temporal = self.downsample(x_temporal) # (N*V, T/2, D)
        
        # Get new T (Dynamic Shape)
        # x_temporal is (Batch_Size, Time, Channels)
        _, current_T, _ = x_temporal.shape
        
        # Reshape back to Spatial: (N, T', V, D)
        x = x_temporal.reshape(N, V, current_T, -1).permute(0, 2, 1, 3)

        # --- Layer 2 ---
        x = self.spatial_2(x)
        # Use current_T for reshape
        x_temporal = x.permute(0, 2, 1, 3).contiguous().reshape(N * V, current_T, -1)
        x_temporal = self.temporal_2(x_temporal)
        x = x_temporal.reshape(N, V, current_T, -1).permute(0, 2, 1, 3)

        # --- Layer 3 ---
        x = self.spatial_3(x)
        x_temporal = x.permute(0, 2, 1, 3).contiguous().reshape(N * V, current_T, -1)
        x_temporal = self.temporal_3(x_temporal)
        
        # Final Feature Aggregation
        final_features = x_temporal.reshape(N, V, current_T, -1)
        frame_features_mean = final_features.mean(dim=1)
        
        # (N, D)
        pooled_features = self.attentive_pooling(frame_features_mean) 
        
        # --- Branch 1: Main Task (Action Recognition) ---
        action_logits = self.action_head(pooled_features)
        
        # Return only action logits
        return action_logits
