import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import config
import math
from torch.autograd import Function

# --------------------------------------------------------------------------
# Utils & Layers
# --------------------------------------------------------------------------

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, input, alpha):
        ctx.alpha = alpha
        return input.clone()

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = -ctx.alpha * grad_output
        return grad_input, None

class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super(GradientReversalLayer, self).__init__()
        self.alpha = alpha

    def forward(self, input):
        return GradientReversalFunction.apply(input, self.alpha)

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        rsqrt = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rsqrt * self.weight

# --------------------------------------------------------------------------
# 1. Embedding Layer
# --------------------------------------------------------------------------
class SpatioTemporalEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_joints, max_frames, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Spatial PE: (1, 1, V, D) - 관절의 정체성
        self.spatial_pe = nn.Parameter(torch.zeros(1, 1, num_joints, hidden_dim))
        
        # Temporal PE: (1, T, 1, D) - 시간적 순서
        self.temporal_pe = nn.Parameter(torch.zeros(1, max_frames, 1, hidden_dim))
        
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.spatial_pe, std=0.02)
        nn.init.trunc_normal_(self.temporal_pe, std=0.02)

    def forward(self, x):
        # x: (N, C, T, V) -> (N, T, V, C)로 변환 후 투영
        N, C, T, V = x.shape
        x = x.permute(0, 2, 3, 1).contiguous() # (N, T, V, C)
        
        x = self.input_proj(x) # (N, T, V, D)
        
        # PE 더하기 (Broadcasting)
        x = x + self.spatial_pe[:, :, :V, :]
        x = x + self.temporal_pe[:, :T, :, :]
        
        return self.dropout(x)

# --------------------------------------------------------------------------
# 2. Spatial Stream: MLP Mixer + SSA
# --------------------------------------------------------------------------
class SpatialMixerBlock(nn.Module):
    def __init__(self, dim, num_joints, mlp_ratio=0.75, dropout=0.1):
        super().__init__()
        
        self.dim = dim
        self.num_joints = num_joints
        
        # 채널 분할
        self.dim_mlp = int(dim * mlp_ratio)
        self.dim_attn = dim - self.dim_mlp
        
        # Branch 1: Token Mixing MLP (관절 간 정보 교환)
        # 입력: (N, T, V, C_mlp) -> Transpose -> (N, T, C_mlp, V) -> FC(V->V)
        self.token_mixing_mlp = nn.Sequential(
            nn.Linear(num_joints, num_joints),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(num_joints, num_joints)
        )
        
        # Branch 2: Spatial Self-Attention (SSA)
        self.norm_attn = RMSNorm(self.dim_attn)
        self.ssa = nn.MultiheadAttention(embed_dim=self.dim_attn, num_heads=4, batch_first=True, dropout=dropout)
        
        # Fusion: Channel Mixing MLP
        # Concat된 특징을 다시 섞어줌
        self.norm_fusion = RMSNorm(dim)
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
        
        # 1. Split
        x_mlp_in, x_attn_in = torch.split(x, [self.dim_mlp, self.dim_attn], dim=-1)
        
        # 2. Branch 1: MLP (Spatial Mixing)
        # (N, T, V, Cm) -> (N*T, Cm, V)
        x_mlp = x_mlp_in.reshape(N * T, V, self.dim_mlp).permute(0, 2, 1)
        x_mlp = self.token_mixing_mlp(x_mlp) # V축에 대해 동작
        x_mlp = x_mlp.permute(0, 2, 1).reshape(N, T, V, self.dim_mlp)
        
        # 3. Branch 2: SSA
        # (N, T, V, Ca) -> (N*T, V, Ca)
        x_attn_reshaped = x_attn_in.reshape(N * T, V, self.dim_attn)
        x_attn_norm = self.norm_attn(x_attn_reshaped)
        # Self-Attention
        x_attn_out, _ = self.ssa(x_attn_norm, x_attn_norm, x_attn_norm)
        x_attn = x_attn_out.reshape(N, T, V, self.dim_attn)
        
        # 4. Concatenate (Residual connection for each branch implies adding input, but here we construct new features)
        x_concat = torch.cat([x_mlp, x_attn], dim=-1) # (N, T, V, D)
        
        # 5. Channel Mixing
        x_out = self.norm_fusion(x_concat)
        x_out = self.channel_mixing_mlp(x_out)
        
        # Global Residual
        return x + x_out

# --------------------------------------------------------------------------
# 3. Temporal Stream: Factorized Temporal Attention
# --------------------------------------------------------------------------
class TemporalFactorizedBlock(nn.Module):
    def __init__(self, dim, num_heads=4, window_size=10, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        
        self.norm1 = RMSNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        
        self.norm2 = RMSNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout)
        )

    def _create_factorized_mask(self, T, device):
        # T: 실제 프레임 수
        # 시퀀스 길이: T + 1 (Global Token 포함)
        # Global Token Index: 0 (맨 앞이라고 가정)
        
        size = T + 1
        mask = torch.full((size, size), float('-inf'), device=device)
        
        # 1. Global Token (Index 0)은 모든 프레임을 봄
        mask[0, :] = 0.0
        
        # 2. 모든 프레임은 Global Token을 봄
        mask[:, 0] = 0.0
        
        # 3. Sliding Window (Local)
        # 인덱스 1부터 T까지는 실제 프레임
        # i, j가 1 이상일 때, |i - j| <= window_size 이면 0.0
        indices = torch.arange(1, size, device=device)
        # Broadcasting으로 차이 계산
        diff = torch.abs(indices.unsqueeze(0) - indices.unsqueeze(1))
        
        # Window 마스크 생성
        window_mask = torch.where(diff <= self.window_size, 0.0, float('-inf'))
        
        # 전체 마스크에 적용
        mask[1:, 1:] = window_mask
        
        return mask

    def forward(self, x):
        # x: (N*V, T+1, D) - 이미 Global Token이 붙어서 들어옴
        B, SeqLen, D = x.shape
        T = SeqLen - 1
        
        residual = x
        x = self.norm1(x)
        
        # Mask 생성
        attn_mask = self._create_factorized_mask(T, x.device)
        
        # Multi-head Attention with Mask
        x, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = residual + x
        
        # FFN
        x = x + self.ffn(self.norm2(x))
        
        return x

# --------------------------------------------------------------------------
# 4. Attentive Pooling
# --------------------------------------------------------------------------
class AttentivePooling(nn.Module):
    def __init__(self, dim, num_classes):
        super().__init__()
        self.dim = dim
        
        # 불확실성을 계산하기 위한 가벼운 Classifier
        self.proxy_classifier = nn.Linear(dim, num_classes)
        
    def forward(self, x):
        # x: (B, T, D) - Global Token을 제외한 실제 프레임 특징들
        
        # 1. 각 프레임별 예측 확률 계산
        logits = self.proxy_classifier(x) # (B, T, NumClasses)
        probs = F.softmax(logits, dim=-1)
        
        # 2. Entropy(불확실성) 계산
        # H(x) = -sum(p * log(p))
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1) # (B, T)
        
        # 3. Confidence(확신도) 기반 가중치
        confidence = 1.0 / (1.0 + entropy)
        weights = 1.0 + confidence # (B, T)
        
        # 가중치 정규화
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        
        # 4. Weighted Sum
        # (B, T, 1) * (B, T, D) -> sum -> (B, D)
        x_pooled = torch.sum(weights.unsqueeze(-1) * x, dim=1)
        
        return x_pooled

# --------------------------------------------------------------------------
# Main Model
# --------------------------------------------------------------------------
class ST_GRL_Model(nn.Module):
    def __init__(self, 
                 num_joints=config.NUM_JOINTS, 
                 num_coords=config.NUM_COORDS, 
                 num_classes=config.NUM_CLASSES, 
                 hidden_dim=128,
                 spatial_depth=3,
                 temporal_depth=3,
                 window_size=10,
                 dropout=config.DROPOUT):
        
        super().__init__()
        
        # 1. Embedding
        self.embedding = SpatioTemporalEmbedding(
            num_coords, hidden_dim, num_joints, config.MAX_FRAMES, dropout
        )

        # 2. [CLS] Token 초기화
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # 3. Block
        # Block 1
        self.spatial_1 = SpatialMixerBlock(hidden_dim, num_joints, dropout=dropout)
        self.temporal_1 = TemporalFactorizedBlock(hidden_dim, window_size=window_size, dropout=dropout)
        # Block 2
        self.spatial_2 = SpatialMixerBlock(hidden_dim, num_joints, dropout=dropout)
        self.temporal_2 = TemporalFactorizedBlock(hidden_dim, window_size=window_size, dropout=dropout)
        
        # Block 3
        self.spatial_3 = SpatialMixerBlock(hidden_dim, num_joints, dropout=dropout)
        self.temporal_3 = TemporalFactorizedBlock(hidden_dim, window_size=window_size, dropout=dropout)

        # 4. Pooling
        self.attentive_pooling = AttentivePooling(hidden_dim, num_classes)

        # 5. Output Heads
        self.action_head = nn.Sequential(
            RMSNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        self.grl = GradientReversalLayer(alpha=1.0)
        self.domain_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1), 
        )
        

    def forward(self, x, alpha=1.0):
        # x: (N, C, T, V)
        N, C, T, V = x.shape
        
        # 1. Embedding
        x = self.embedding(x)
        
        # [CLS] 토큰 준비 (N*V, 1, D)
        # 초기에는 Parameter를 복사해서 사용하지만, 층을 지나면서 업데이트된 값을 유지해야 함
        cls_tokens = self.cls_token.expand(N * V, -1, -1)
        
        # ---------------------------------------------------------
        # Hard-coded Forward Pass
        # ---------------------------------------------------------
        
        # === Layer 1 ===
        # 1-1. Spatial (Frames only)
        x = self.spatial_1(x) # (N, T, V, D)
        
        # [Transform for Temporal]
        # (N, T, V, D) -> (N, V, T, D) -> (N*V, T, D)
        x_flat = x.permute(0, 2, 1, 3).reshape(N * V, T, -1)
        # Attach CLS: (N*V, 1, D) + (N*V, T, D) -> (N*V, T+1, D)
        x_temporal_in = torch.cat((cls_tokens, x_flat), dim=1)
        
        # 1-2. Temporal
        x_temporal_out = self.temporal_1(x_temporal_in)
        
        # [Detach CLS & Transform for Spatial]
        # Split: CLS는 업데이트된 상태로 보존, Frames는 다음 Spatial로 전달
        cls_tokens = x_temporal_out[:, 0:1, :] # Update CLS for next layer
        x_flat = x_temporal_out[:, 1:, :]      # Frames
        
        # (N*V, T, D) -> (N, V, T, D) -> (N, T, V, D)
        x = x_flat.view(N, V, T, -1).permute(0, 2, 1, 3)

        # === Layer 2 ===
        # 2-1. Spatial
        x = self.spatial_2(x)
        
        # [Transform]
        x_flat = x.permute(0, 2, 1, 3).reshape(N * V, T, -1)
        x_temporal_in = torch.cat((cls_tokens, x_flat), dim=1) # Use updated CLS
        
        # 2-2. Temporal
        x_temporal_out = self.temporal_2(x_temporal_in)
        
        # [Detach]
        cls_tokens = x_temporal_out[:, 0:1, :]
        x_flat = x_temporal_out[:, 1:, :]
        x = x_flat.view(N, V, T, -1).permute(0, 2, 1, 3)

        # === Layer 3 ===
        # 3-1. Spatial
        x = self.spatial_3(x)
        
        # [Transform]
        x_flat = x.permute(0, 2, 1, 3).reshape(N * V, T, -1)
        x_temporal_in = torch.cat((cls_tokens, x_flat), dim=1)
        
        # 3-2. Temporal
        x_temporal_out = self.temporal_3(x_temporal_in)

        # Final Feature Preparation
        # Temporal Block을 마지막으로 통과했으므로 x_temporal_out 사용
        

        # Frames만 추출
        frame_features = x_temporal_out[:, 1:, :] # (N*V, T, D)
        
        # Attentive Pooling은 (B, T, D) 입력을 받음. 여기서 B는 N이어야 의미가 있음.
        # 기존 코드는 (N, V, T+1, D) -> mean(V) -> (N, T+1, D) 였습니다.
        # 관절(V) 차원에 대해 평균을 내서 비디오 레벨 특징으로 만듭니다.
        
        # (N*V, T, D) -> (N, V, T, D)
        frame_features = frame_features.view(N, V, T, -1)
        # 관절 차원 평균 -> (N, T, D)
        frame_features_mean = frame_features.mean(dim=1)
        
        # 6. Attentive Pooling -> (N, D)
        pooled_features = self.attentive_pooling(frame_features_mean) 
        
        # 7. Output Heads
        action_logits = self.action_head(pooled_features)
        
        # Domain Head
        self.grl.alpha = alpha 
        domain_feat = self.grl(pooled_features)
        domain_logits = self.domain_head(domain_feat)
        
        return action_logits, domain_logits, pooled_features
