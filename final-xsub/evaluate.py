# >> evaluate.py
# >> python evaluate.py --checkpoint checkpoints/GRL_XView/trial_1/best_model.pth.tar --protocol xview

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import os
import argparse
import numpy as np
import matplotlib
# 서버 환경에서 GUI 없이 실행하기 위해 Agg 백엔드 설정
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

import config
from ntu_data_loader import NTURGBDDataset
from model import ST_Model
from utils import load_checkpoint

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def plot_confusion_matrix(labels, preds, num_classes, save_dir, prefix='eval'):
    # 저장 디렉토리 생성
    os.makedirs(save_dir, exist_ok=True)

    # Confusion Matrix 계산
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for l, p in zip(labels, preds):
        cm[l, p] += 1
    
    # Class Accuracy 계산
    with np.errstate(divide='ignore', invalid='ignore'):
        class_acc = cm.diagonal() / cm.sum(axis=1)
        class_acc = np.nan_to_num(class_acc)

    # Class Accuracy 텍스트 파일 저장
    acc_save_path = os.path.join(save_dir, f'{prefix}_class_accuracy.txt')
    with open(acc_save_path, 'w') as f:
        f.write(f"Evaluation Class-wise Accuracy:\n")
        for i, acc in enumerate(class_acc):
            f.write(f"Class {i+1}: {acc:.4f}\n")
    print(f"[Info] Class accuracy saved to {acc_save_path}")

    # Confusion Matrix 이미지 저장
    plt.figure(figsize=(20, 20))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix ({prefix})')
    plt.colorbar()
    
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, tick_marks + 1, rotation=90, fontsize=8)
    plt.yticks(tick_marks, tick_marks + 1, fontsize=8)
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    
    cm_save_path = os.path.join(save_dir, f'{prefix}_confusion_matrix.png')
    plt.savefig(cm_save_path)
    plt.close()
    print(f"[Info] Confusion matrix saved to {cm_save_path}")

def evaluate(model, loader, device):
    model.eval()
    correct_action = 0
    total_samples = 0
    
    all_preds = []
    all_labels = []
    
    print(f"[Info] Starting inference on validation set...")
    with torch.no_grad():
        for features, labels, _ in tqdm(loader, desc="[Eval]", colour='cyan'):
            features = features.to(device)
            labels = labels.to(device)
            
            with autocast('cuda'):
                action_logits = model(features)
                
            _, predicted = torch.max(action_logits, 1)
            correct_action += (predicted == labels).sum().item()
            total_samples += features.size(0)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    total_acc = correct_action / total_samples
    return total_acc, all_preds, all_labels

def main():
    parser = argparse.ArgumentParser(description="Evaluate existing checkpoint")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the checkpoint file (e.g., checkpoints/.../best_model.pth.tar)")
    parser.add_argument('--protocol', type=str, default='xsub', choices=['xsub', 'xview'], help="Evaluation protocol")
    parser.add_argument('--batch-size', type=int, default=config.BATCH_SIZE, help="Batch size for evaluation")
    parser.add_argument('--save-dir', type=str, default='eval_results', help="Directory to save evaluation results")
    
    args = parser.parse_args()
    
    # 1. 설정 및 시드 고정
    device = config.DEVICE
    set_seed(config.SEED)
    print(f"\n[Info] Evaluation Device: {device}")
    print(f"[Info] Protocol: {args.protocol}")
    print(f"[Info] Loading Checkpoint: {args.checkpoint}")

    # 2. 데이터셋 로드 (Validation Split만 로드)
    # 데이터 로딩 시 Generator에 시드를 설정하여 일관성 유지
    g = torch.Generator()
    g.manual_seed(config.SEED)

    val_dataset = NTURGBDDataset(config.DATASET_PATH, split='val', max_frames=config.MAX_FRAMES, protocol=args.protocol)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, 
        shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY, generator=g
    )
    print(f"[Info] Validation Samples: {len(val_dataset)}")

    # 3. 모델 초기화
    model = ST_Model(
        num_joints=config.NUM_JOINTS,
        num_coords=config.NUM_COORDS,
        num_classes=config.NUM_CLASSES,
        hidden_dim=config.HIDDEN_DIM,
        window_size=config.WINDOW_SIZE,
        dropout=config.DROPOUT
    ).to(device)

    # 4. 체크포인트 로드
    # utils.py의 load_checkpoint 함수 사용 (Optimizer/Scheduler는 로드할 필요 없음)
    try:
        load_checkpoint(args.checkpoint, model, device=device)
    except Exception as e:
        print(f"[Error] Failed to load checkpoint: {e}")
        return

    # 5. 검증 수행
    acc, preds, labels = evaluate(model, val_loader, device)
    
    print(f"\n=============================================")
    print(f" Final Evaluation Accuracy: {acc:.4f}")
    print(f"=============================================\n")

    # 6. 결과 시각화 및 저장
    # 파일명 prefix를 체크포인트 이름이나 프로토콜로 설정
    ckpt_name = os.path.basename(args.checkpoint).replace('.pth.tar', '')
    prefix = f"{args.protocol}_{ckpt_name}"
    
    plot_confusion_matrix(labels, preds, config.NUM_CLASSES, args.save_dir, prefix=prefix)
    print(f"[Info] Evaluation Complete.")

if __name__ == '__main__':
    main()
