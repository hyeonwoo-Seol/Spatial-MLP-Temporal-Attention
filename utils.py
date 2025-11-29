import torch
import torch.nn as nn
import os
import torch.nn.functional as F



# >> 모델의 logits과 실제 레이블을 기반으로 정확도를 계산한다.
def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    return correct / total



# >> 체크포인트를 저장한다.
def save_checkpoint(state, directory, filename="checkpoint.pth.tar"):
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



# >> 저장된 체크포인트를 복원한다.
def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, device='cpu'):
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


# >> 2개의 View에서 나온 특징 벡터 사이의 상관관계 행렬이 단위 행렬에 가까워지도록 유도한다.
class FeatureConsistencyLoss(nn.Module):
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
