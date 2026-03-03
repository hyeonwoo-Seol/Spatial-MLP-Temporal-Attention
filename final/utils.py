# >> utils.py

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
