import torch
import torch.nn as nn
import os
import torch.nn.functional as F

def calculate_accuracy(outputs, labels):
    """
    모델의 출력(logits)과 실제 레이블을 기반으로 정확도를 계산
    """
    _, predicted = torch.max(outputs.data, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    return correct / total

def save_checkpoint(state, directory, filename="checkpoint.pth.tar"):
    """
    체크포인트 저장 (임시 파일 생성 후 rename하여 안전성 확보)
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    filepath = os.path.join(directory, filename)
    filepath_tmp = filepath + ".tmp"

    try:
        torch.save(state, filepath_tmp)
        os.rename(filepath_tmp, filepath)
    except Exception as e:
        print(f"Error saving checkpoint: {e}")
        if os.path.exists(filepath_tmp):
            os.remove(filepath_tmp)

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, device='cpu'):
    """
    저장된 체크포인트 복원
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at '{checkpoint_path}'")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['state_dict'])
    
    if optimizer and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

    if scheduler and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
        
    print(f"Model checkpoint loaded from '{checkpoint_path}'")
    return checkpoint

class FeatureConsistencyLoss(nn.Module):
    """
    [Feature Consistency Loss]
    두 개의 뷰(View1, View2)에서 나온 특징 벡터 간의 상관관계 행렬(Correlation Matrix)이
    단위 행렬(Identity Matrix)에 가까워지도록 유도하여,
    1. 동일한 샘플의 특징은 유사하게 (Diagonal -> 1)
    2. 서로 다른 차원의 특징은 독립적이게 (Off-diagonal -> 0) 만듦.
    """
    def __init__(self, lambd=0.0051):
        super().__init__()
        self.lambd = lambd

    def forward(self, z1, z2):
        # z1, z2: (N, D)
        N, D = z1.size()

        # 1. Normalize along the batch dimension
        # 각 차원의 평균을 0, 표준편차를 1로 만듦
        bn = nn.BatchNorm1d(D, affine=False).to(z1.device)
        z1_norm = bn(z1)
        z2_norm = bn(z2)

        # 2. Cross-correlation matrix (D x D)
        c = torch.mm(z1_norm.T, z2_norm) / N

        # 3. Identity matrix (D x D)
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum() # 대각 성분은 1이 되어야 함 ((C_ii - 1)^2)
        off_diag = off_diagonal(c).pow_(2).sum()           # 비대각 성분은 0이 되어야 함 (C_ij^2)

        loss = on_diag + self.lambd * off_diag
        return loss

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
