# >> model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import config


# >> Utils & Layers
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        # >> Root Mean Square Layer Normalization을 계산한다.
        rsqrt = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rsqrt * self.weight


# >> Temporal Downsampling Layer
class TemporalDownsample(nn.Module):
    # >> 1D 합성곱을 사용하여 시간 해상도를 절반으로 줄인다.
    # >> 입력은 (B, T, D)이고 출력은 (B, T//2, D)이다.
    def __init__(self, dim, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = RMSNorm(dim)

    def forward(self, x):
        # >> 입력 텐서의 형상은 (B, T, D)이다.
        B, T, D = x.shape
        
        # >> Conv1d는 (Batch, Channel, Length)를 기대하므로 (B, D, T)로 차원을 변환한다.
        x = x.transpose(1, 2)
        x = self.conv(x)
        
        # >> 다시 (B, T', D) 형태로 복구하고 정규화를 수행한다.
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x


# >> Embedding Layer
class SpatioTemporalEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_joints, max_frames, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # >> 공간적 위치 인코딩(Spatial PE) 파라미터를 (1, 1, V, D) 형태로 초기화한다.
        self.spatial_pe = nn.Parameter(torch.zeros(1, 1, num_joints, hidden_dim))
        
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.spatial_pe, std=0.02)

    def forward(self, x):
        # >> 입력 텐서 (N, C, T, V)를 (N, T, V, C)로 변환한다.
        N, C, T, V = x.shape
        x = x.permute(0, 2, 3, 1).contiguous() 
        x = self.input_proj(x) # (N, T, V, D)
        
        # >> 공간적 위치 정보를 더한다.
        x = x + self.spatial_pe[:, :, :V, :]
        return self.dropout(x)


# >> Spatial Stream: Pure MLP Mixer (Lightweight)
class SpatialMixerBlock(nn.Module):
    def __init__(self, dim, num_joints, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_joints = num_joints
        
        # >> 관절(V) 간의 정보를 교환하는 Token Mixing MLP를 정의한다.
        self.norm1 = RMSNorm(dim)
        self.token_mixing_mlp = nn.Sequential(
            nn.Linear(num_joints, num_joints),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(num_joints, num_joints)
        )
        
        # >> 채널(D) 간의 정보를 교환하는 Channel Mixing MLP를 정의한다.
        self.norm2 = RMSNorm(dim)
        self.channel_mixing_mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # >> 입력 텐서의 형상은 (N, T, V, D)이다.
        N, T, V, D = x.shape
        
        # >> 1. Token Mixing (Spatial Interaction)을 수행한다.
        # >> 차원을 (N*T, D, V)로 변환하여 관절 축에 대해 MLP를 적용한다.
        residual = x
        x = self.norm1(x)
        x = x.reshape(N * T, V, D).permute(0, 2, 1) # (N*T, D, V)
        x = self.token_mixing_mlp(x)
        x = x.permute(0, 2, 1).reshape(N, T, V, D)
        x = residual + x
        
        # >> 2. Channel Mixing을 수행한다.
        residual = x
        x = self.norm2(x)
        x = self.channel_mixing_mlp(x)
        x = residual + x
        
        return x


# >> Temporal Stream: Multi-Scale Swin Transformer (Modified)
class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # >> 상대적 위치 편향(Relative Position Bias) 테이블을 초기화한다.
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(2 * window_size - 1, num_heads)
        )
        coords = torch.arange(self.window_size)
        relative_coords = coords[:, None] - coords[None, :]
        relative_coords += self.window_size - 1
        self.register_buffer("relative_position_index", relative_coords)

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        # >> Q, K, V를 계산하고 헤드 별로 차원을 분리한다.
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # >> 상대적 위치 편향을 어텐션 점수에 더한다.
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size, self.window_size, -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        # >> 마스킹이 필요한 경우 마스크 값을 더해 어텐션을 제한한다.
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.reshape(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.reshape(-1, self.num_heads, N, N)
            
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        # >> 어텐션 값과 V를 곱하고 차원을 원래대로 복구한다.
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTemporalBlock(nn.Module):
    def __init__(self, dim, num_heads, window_sizes=[10, 20], shift_sizes=[0, 0], dropout=0.1, bottleneck_ratio=0.5):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        
        # >> Multi-Scale 처리를 위해 서로 다른 Window Size와 Shift Size를 리스트로 받는다.
        self.window_sizes = window_sizes
        self.shift_sizes = shift_sizes
        
        # >> 병목 구조를 사용하여 연산량을 줄인다 (예: 128 -> 64).
        self.dim_inner = int(dim * bottleneck_ratio)
        
        # >> 각 브랜치(헤드 그룹)별 채널 및 헤드 수를 절반으로 나눈다.
        self.dim_branch = self.dim_inner // 2
        self.heads_branch = num_heads // 2
        
        # >> 채널 축소 및 확대를 위한 선형 레이어를 정의한다.
        self.proj_down = nn.Linear(dim, self.dim_inner)
        self.proj_up = nn.Linear(self.dim_inner, dim)
        
        self.norm1 = RMSNorm(self.dim_inner)
        
        # >> 두 개의 서로 다른 Window Attention을 생성한다.
        # >> Branch 1: window_sizes[0] 사용
        self.attn1 = WindowAttention(
            self.dim_branch, window_sizes[0], self.heads_branch, dropout
        )
        # >> Branch 2: window_sizes[1] 사용
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
        # >> 각 브랜치별로 Padding, Shift, Window Partition, Attention, Merge, Reverse Shift를 수행한다.
        B, T, C = x.shape
        
        # >> 1. 윈도우 크기에 맞춰 패딩을 적용한다.
        pad_t = (window_size - T % window_size) % window_size
        if pad_t > 0:
            x = F.pad(x, (0, 0, 0, pad_t))
        
        Bp, Tp, Cp = x.shape

        # >> 2. Cyclic Shift를 수행하고 어텐션 마스크를 생성한다.
        if shift_size > 0:
            shifted_x = torch.roll(x, shifts=-shift_size, dims=1)
            
            # >> Dynamic T를 고려하여 마스크를 생성한다.
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

        # >> 3. 윈도우 단위로 텐서를 분할한다.
        x_windows = shifted_x.reshape(Bp, Tp // window_size, window_size, Cp)
        x_windows = x_windows.reshape(-1, window_size, Cp)
        
        # >> 4. 윈도우 어텐션을 수행한다.
        attn_windows = attn_layer(x_windows, mask=attn_mask)
        
        # >> 5. 분할된 윈도우를 다시 병합한다.
        shifted_x = attn_windows.reshape(Bp, Tp, Cp)
        
        # >> 6. 시프트된 텐서를 원래 위치로 되돌린다 (Reverse Shift).
        if shift_size > 0:
            x_out = torch.roll(shifted_x, shifts=shift_size, dims=1)
        else:
            x_out = shifted_x

        # >> 7. 패딩을 제거하여 원래 크기로 복구한다.
        if pad_t > 0:
            x_out = x_out[:, :T, :]
            
        return x_out

    def forward(self, x):
        # >> 입력 텐서의 형상은 (N*V, T, D)이다.
        residual = x
        
        # >> 1. 병목 구조를 통해 채널 차원을 축소한다.
        # >> (B, T, 128) -> (B, T, 64)
        x_inner = self.proj_down(x)
        x_inner = self.norm1(x_inner)

        # >> 2. 채널을 반으로 나누어 다중 스케일 처리를 준비한다.
        x1, x2 = torch.split(x_inner, self.dim_branch, dim=-1)
        
        # >> Branch 1 (Window Size 1)을 처리한다.
        out1 = self._process_branch(x1, self.attn1, self.window_sizes[0], self.shift_sizes[0])
        
        # >> Branch 2 (Window Size 2)을 처리한다.
        out2 = self._process_branch(x2, self.attn2, self.window_sizes[1], self.shift_sizes[1])
        
        # >> 3. 두 브랜치의 결과를 연결(Concatenation)한다.
        x_inner = torch.cat([out1, out2], dim=-1)

        # >> 4. 채널 차원을 원래대로 복구한다.
        # >> (B, T, 64) -> (B, T, 128)
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
        # >> 프레임별 중요도(가중치)를 계산하여 풀링을 수행한다.
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
        # >> 시공간 임베딩 레이어를 초기화한다.
        self.embedding = SpatioTemporalEmbedding(
            num_coords, hidden_dim, num_joints, config.MAX_FRAMES, dropout
        )
        
        # >> Multi-Scale Window 설정을 정의한다.
        # >> 기본 윈도우(10)와 2배 크기 윈도우(20)를 동시에 사용한다.
        ws_small = window_size
        ws_large = window_size * 2
        window_sizes = [ws_small, ws_large]
        
        # >> Shift Size를 설정한다 (각 윈도우의 절반).
        ss_small = ws_small // 2
        ss_large = ws_large // 2

        # >> 시간적 다운샘플링 레이어를 정의한다.
        self.downsample1 = TemporalDownsample(hidden_dim, kernel_size=3, stride=2, padding=1)
        self.downsample2 = TemporalDownsample(hidden_dim, kernel_size=3, stride=2, padding=1)

        # >> Stage 1 레이어들을 정의한다. Shift는 없다.
        self.spatial_1 = SpatialMixerBlock(hidden_dim, num_joints, dropout=dropout)
        self.temporal_1 = SwinTemporalBlock(hidden_dim, num_heads=4, window_sizes=window_sizes, shift_sizes=[0, 0], dropout=dropout, bottleneck_ratio=0.5)
        
        # >> Stage 2 레이어들을 정의한다. Shift를 적용한다.
        self.spatial_2 = SpatialMixerBlock(hidden_dim, num_joints, dropout=dropout)
        self.temporal_2 = SwinTemporalBlock(hidden_dim, num_heads=4, window_sizes=window_sizes, shift_sizes=[ss_small, ss_large], dropout=dropout, bottleneck_ratio=0.5)
        
        # >> Stage 3 레이어들을 정의한다. Shift는 없다.
        self.spatial_3 = SpatialMixerBlock(hidden_dim, num_joints, dropout=dropout)
        self.temporal_3 = SwinTemporalBlock(hidden_dim, num_heads=4, window_sizes=window_sizes, shift_sizes=[0, 0], dropout=dropout, bottleneck_ratio=0.5)

        # >> Stage 4 레이어들을 정의한다. Shift를 적용한다.
        self.spatial_4 = SpatialMixerBlock(hidden_dim, num_joints, dropout=dropout)
        self.temporal_4 = SwinTemporalBlock(hidden_dim, num_heads=4, window_sizes=window_sizes, shift_sizes=[ss_small, ss_large], dropout=dropout, bottleneck_ratio=0.5)

        # >> 어텐티브 풀링 레이어를 정의한다.
        self.attentive_pooling = AttentivePooling(hidden_dim, num_classes)

        # >> 최종 행동 분류를 위한 헤드를 정의한다.
        self.action_head = nn.Sequential(
            RMSNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # >> 입력 텐서 (N, C, T, V)를 처리한다.
        N, C, T, V = x.shape
        x = self.embedding(x)

        # >> 첫 번째 다운샘플링을 수행한다.
        x_flat = x.permute(0, 2, 1, 3).contiguous().reshape(N * V, T, -1)
        x_flat = self.downsample1(x_flat) # (N*V, 64, D)

        # >> 새로운 시간 차원 T를 얻고 텐서를 재구성한다.
        _, current_T, _ = x_flat.shape
        x = x_flat.reshape(N, V, current_T, -1).permute(0, 2, 1, 3) # (N, 64, V, D)
        
        # >> Layer 1 (Stage 1)을 수행한다.
        x = self.spatial_1(x) 
        # >> 시간적 처리를 위해 차원을 변환한다: (N*V, T, D)
        x_temporal = x.permute(0, 2, 1, 3).contiguous().reshape(N * V, current_T, -1)
        x_temporal = self.temporal_1(x_temporal)
        x = x_temporal.reshape(N, V, current_T, -1).permute(0, 2, 1, 3)


        # >> 두 번째 다운샘플링을 수행한다.
        x_flat = x.permute(0, 2, 1, 3).contiguous().reshape(N * V, current_T, -1)
        x_flat = self.downsample2(x_flat) # (N*V, 32, D)

        # >> 새로운 시간 차원 T를 얻고 텐서를 재구성한다.
        _, current_T, _ = x_flat.shape
        x = x_flat.reshape(N, V, current_T, -1).permute(0, 2, 1, 3) # (N, 32, V, D)

        
        # >> Layer 2 (Stage 2)를 수행한다.
        x = self.spatial_2(x)
        x_temporal = x.permute(0, 2, 1, 3).contiguous().reshape(N * V, current_T, -1)
        x_temporal = self.temporal_2(x_temporal)
        x = x_temporal.reshape(N, V, current_T, -1).permute(0, 2, 1, 3)

                
        # >> Layer 3 (Stage 3)를 수행한다.
        x = self.spatial_3(x)
        x_temporal = x.permute(0, 2, 1, 3).contiguous().reshape(N * V, current_T, -1)
        x_temporal = self.temporal_3(x_temporal)
        x = x_temporal.reshape(N, V, current_T, -1).permute(0, 2, 1, 3)

        # >> Layer 4 (Stage 4)를 수행한다.
        x = self.spatial_4(x)
        x_temporal = x.permute(0, 2, 1, 3).contiguous().reshape(N * V, current_T, -1)
        x_temporal = self.temporal_4(x_temporal)
        
        # >> 최종 특징을 집계한다.
        final_features = x_temporal.reshape(N, V, current_T, -1)
        frame_features_mean = final_features.mean(dim=1)
        
        # >> 어텐티브 풀링을 통해 (N, D) 크기의 벡터를 생성한다.
        pooled_features = self.attentive_pooling(frame_features_mean) 
        
        # >> 행동 분류 로짓을 계산한다.
        action_logits = self.action_head(pooled_features)
        
        # >> 최종 로짓만 반환한다.
        return action_logits
