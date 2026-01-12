# >> model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import config

# --------------------------------------------------------------------------
# Utils & Layers
# --------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        rsqrt = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rsqrt * self.weight

# --------------------------------------------------------------------------
# Temporal Downsampling Layer
# --------------------------------------------------------------------------
class TemporalDownsample(nn.Module):
    """
    Applies strided 1D convolution to reduce temporal resolution by half.
    Input: (B, T, D) where B is usually N * V
    Output: (B, T//2, D)
    """
    def __init__(self, dim, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = RMSNorm(dim)

    def forward(self, x):
        # x: (B, T, D)
        B, T, D = x.shape
        
        # Conv1d expects (Batch, Channel, Length) -> Transpose to (B, D, T)
        x = x.transpose(1, 2)
        x = self.conv(x)
        
        # Back to (B, T', D)
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x

# --------------------------------------------------------------------------
# 1. Embedding Layer
# --------------------------------------------------------------------------
class SpatioTemporalEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_joints, max_frames, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Spatial PE: (1, 1, V, D)
        self.spatial_pe = nn.Parameter(torch.zeros(1, 1, num_joints, hidden_dim))
        
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.spatial_pe, std=0.02)

    def forward(self, x):
        # x: (N, C, T, V) -> (N, T, V, C)
        N, C, T, V = x.shape
        x = x.permute(0, 2, 3, 1).contiguous() 
        x = self.input_proj(x) # (N, T, V, D)
        x = x + self.spatial_pe[:, :, :V, :]
        return self.dropout(x)

# --------------------------------------------------------------------------
# 2. Spatial Stream: Pure MLP Mixer (Lightweight)
# --------------------------------------------------------------------------
class SpatialMixerBlock(nn.Module):
    def __init__(self, dim, num_joints, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_joints = num_joints
        
        # Token Mixing MLP: 관절(V) 간의 정보 교환
        self.norm1 = RMSNorm(dim)
        self.token_mixing_mlp = nn.Sequential(
            nn.Linear(num_joints, num_joints),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(num_joints, num_joints)
        )
        
        # Channel Mixing MLP: 채널(D) 간의 정보 교환
        self.norm2 = RMSNorm(dim)
        self.channel_mixing_mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: (N, T, V, D)
        N, T, V, D = x.shape
        
        # 1. Token Mixing (Spatial Interaction)
        # (N, T, V, D) -> (N*T, D, V) -> MLP(V->V) -> (N, T, V, D)
        residual = x
        x = self.norm1(x)
        x = x.reshape(N * T, V, D).permute(0, 2, 1) # (N*T, D, V)
        x = self.token_mixing_mlp(x)
        x = x.permute(0, 2, 1).reshape(N, T, V, D)
        x = residual + x
        
        # 2. Channel Mixing
        residual = x
        x = self.norm2(x)
        x = self.channel_mixing_mlp(x)
        x = residual + x
        
        return x

# --------------------------------------------------------------------------
# 3. Temporal Stream: Multi-Scale Swin Transformer (Modified)
# --------------------------------------------------------------------------
class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Relative Position Bias
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
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size, self.window_size, -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.reshape(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.reshape(-1, self.num_heads, N, N)
            
        attn = self.softmax(attn)
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

# --------------------------------------------------------------------------
# 4. Attentive Pooling
# --------------------------------------------------------------------------
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

# --------------------------------------------------------------------------
# Main Model: ST_Model (GRL Removed)
# --------------------------------------------------------------------------
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
