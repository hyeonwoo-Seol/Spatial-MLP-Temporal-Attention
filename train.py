import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, CosineAnnealingWarmRestarts, SequentialLR
from tqdm import tqdm
import os
import random
import numpy as np
import sys
import argparse
import time
import config

# 모듈 import
from ntu_data_loader import NTURGBDDataset
from model import ST_GRL_Model
from utils import calculate_accuracy, save_checkpoint, load_checkpoint, FeatureConsistencyLoss

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_scheduler(optimizer, scheduler_name, total_epochs, warmup_epochs):
    print(f"Using '{scheduler_name}' scheduler.")
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
    
    if scheduler_name == 'cosine_decay':
        main_scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs, eta_min=config.ETA_MIN)
    elif scheduler_name == 'cosine_restarts':
        main_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=config.T_0, T_mult=config.T_MULT, eta_min=config.ETA_MIN)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

    return SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_epochs])

def train_one_epoch(model, loader, criterion_cls, criterion_domain, criterion_fc, optimizer, device, scaler, epoch, args):
    model.train()
    
    running_loss = 0.0
    correct_action = 0
    total_samples = 0
    
    # 진행률 표시
    desc = f"[Train Ep {epoch+1}]"
    pbar = tqdm(loader, desc=desc, leave=False, colour='green')
    
    for view1, view2, action_labels, _, _ in pbar:
        # 데이터 이동
        view1 = view1.to(device) # (N, C, T, V)
        view2 = view2.to(device) # (N, C, T, V)
        action_labels = action_labels.to(device)
        
        # Domain Labels 생성 (Source = 0)
        # 추후 Target 데이터가 추가되면 Target은 1로 설정해야 함
        batch_size = view1.size(0)
        domain_labels_source = torch.zeros(batch_size, 1).to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            # 1. Forward Pass (View 1 - Main)
            # action_logits: (N, Class)
            # domain_logits: (N, 1) -> GRL 통과된 값
            # feat1: (N, D) -> Feature Consistency용
            act_logits1, dom_logits1, feat1 = model(view1, alpha=args.alpha)
            
            # 2. Forward Pass (View 2 - Consistency)
            # View 2는 학습용(backprop)으로 쓸 때, Action Loss는 계산 안 하거나 View1과 동일하게 취급 가능
            # 여기서는 Feature Consistency를 위해 Feature만 추출하는 것이 주 목적이지만,
            # 데이터 증강 효과를 위해 Action Classification에도 참여시킬 수 있음.
            # (논문 구현에 따라 다르지만, 보통 View1, View2 모두 Action Loss를 계산하면 성능이 더 좋음)
            act_logits2, dom_logits2, feat2 = model(view2, alpha=args.alpha)
            
            # --- Loss Calculation ---
            
            # A. Action Classification Loss (Source Data Only)
            loss_cls = (criterion_cls(act_logits1, action_labels) + criterion_cls(act_logits2, action_labels)) * 0.5
            
            # B. Feature Consistency Loss
            loss_fc = criterion_fc(feat1, feat2)
            
            # C. Domain Discriminator Loss (GRL)
            # 현재는 Source만 있으므로 Source를 0으로 예측하도록 학습
            # (Target 데이터가 없으므로 Discriminator가 0으로만 편향될 수 있음. 
            #  본격적인 GRL 학습 시에는 Target DataLoader에서 데이터를 가져와 함께 학습해야 함)
            loss_dom = criterion_domain(dom_logits1, domain_labels_source) 
            
            # Total Loss
            # loss_dom은 Target 데이터가 없을 땐 큰 의미가 없으므로 가중치를 낮추거나 0으로 둘 수 있음
            # 여기서는 구조를 갖추기 위해 포함시킴
            loss = loss_cls + (args.lambda_fc * loss_fc) + (args.lambda_dom * loss_dom)

        # Backward
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.GRAD_CLIP_NORM)
        scaler.step(optimizer)
        scaler.update()
        
        # Stats Update
        running_loss += loss.item() * batch_size
        total_samples += batch_size
        
        # Acc 계산 (View1 기준)
        _, predicted = torch.max(act_logits1, 1)
        correct_action += (predicted == action_labels).sum().item()
        
        pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'Acc': f"{correct_action/total_samples:.4f}"})
        
    return running_loss / total_samples, correct_action / total_samples

def validate_one_epoch(model, loader, criterion_cls, device):
    model.eval()
    running_loss = 0.0
    correct_action = 0
    total_samples = 0
    
    with torch.no_grad():
        for view1, _, action_labels, _, _ in tqdm(loader, desc="[Val]", leave=False, colour='cyan'):
            view1 = view1.to(device)
            action_labels = action_labels.to(device)
            
            with autocast():
                # Validation에서는 View1(원본)만 사용
                act_logits, _, _ = model(view1, alpha=0.0) # Alpha 0 for Eval
                loss = criterion_cls(act_logits, action_labels)
                
            running_loss += loss.item() * view1.size(0)
            _, predicted = torch.max(act_logits, 1)
            correct_action += (predicted == action_labels).sum().item()
            total_samples += view1.size(0)
            
    return running_loss / total_samples, correct_action / total_samples

def run_training(args):
    """
    Optuna 등 외부에서 호출 가능하도록 학습 로직을 함수로 분리
    """
    # 설정 업데이트
    set_seed(config.SEED)
    config.LEARNING_RATE = args.lr
    config.DROPOUT = args.dropout
    config.ADVERSARIAL_ALPHA = args.alpha
    config.PROB = args.prob
    config.ADAMW_WEIGHT_DECAY = args.weight_decay
    config.LABEL_SMOOTHING = args.smoothing
    
    device = config.DEVICE
    print(f"\n[Info] Device: {device}, Protocol: {args.protocol}")
    
    # 1. Dataset & Loader
    train_dataset = NTURGBDDataset(config.DATASET_PATH, split='train', max_frames=config.MAX_FRAMES, protocol=args.protocol)
    val_dataset = NTURGBDDataset(config.DATASET_PATH, split='val', max_frames=config.MAX_FRAMES, protocol=args.protocol)
    
    # Optuna 시 worker seed 고정을 위한 generator
    g = torch.Generator()
    g.manual_seed(config.SEED)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY, generator=g
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY, generator=g
    )
    
    # 2. Model Init
    model = ST_GRL_Model(
        num_joints=config.NUM_JOINTS,
        num_coords=config.NUM_COORDS,
        num_classes=config.NUM_CLASSES,
        hidden_dim=config.HIDDEN_DIM,
        spatial_depth=config.SPATIAL_DEPTH,
        temporal_depth=config.TEMPORAL_DEPTH,
        window_size=config.WINDOW_SIZE,
        dropout=config.DROPOUT
    ).to(device)
    
    # 3. Optim & Loss
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.ADAMW_WEIGHT_DECAY)
    scheduler = get_scheduler(optimizer, args.scheduler, config.EPOCHS, config.WARMUP_EPOCHS)
    scaler = GradScaler()
    
    criterion_cls = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
    criterion_domain = nn.BCELoss() # Binary Cross Entropy for Source(0) vs Target(1)
    criterion_fc = FeatureConsistencyLoss(lambd=0.005) # 논문 권장값 참조 or 튜닝 필요
    
    # 4. Training Loop
    best_acc = 0.0
    trial_save_dir = os.path.join(config.SAVE_DIR, args.study_name, f"trial_{args.trial_number}")
    os.makedirs(trial_save_dir, exist_ok=True)
    
    print(f"Start Training: {config.EPOCHS} Epochs")
    for epoch in range(config.EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion_cls, criterion_domain, criterion_fc, 
            optimizer, device, scaler, epoch, args
        )
        
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion_cls, device)
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Ep [{epoch+1}/{config.EPOCHS}] LR: {current_lr:.6f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
            }, directory=trial_save_dir, filename="best_model.pth.tar")
            
    print(f"Training Finished. Best Val Acc: {best_acc:.4f}")
    return best_acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scheduler', type=str, default='cosine_decay', choices=['cosine_decay', 'cosine_restarts'])
    parser.add_argument('--protocol', type=str, default='xsub', choices=['xsub', 'xview'])
    parser.add_argument('--study-name', type=str, default='default_study')
    parser.add_argument('--trial-number', type=int, default=0)
    
    # Hyperparameters (Defaults from config)
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE)
    parser.add_argument('--dropout', type=float, default=config.DROPOUT)
    parser.add_argument('--alpha', type=float, default=config.ADVERSARIAL_ALPHA)
    parser.add_argument('--prob', type=float, default=config.PROB)
    parser.add_argument('--weight-decay', type=float, default=config.ADAMW_WEIGHT_DECAY)
    parser.add_argument('--smoothing', type=float, default=config.LABEL_SMOOTHING)
    
    # Loss Weights
    parser.add_argument('--lambda-fc', type=float, default=0.1, help="Weight for Feature Consistency Loss")
    parser.add_argument('--lambda-dom', type=float, default=0.1, help="Weight for Domain Loss")

    args = parser.parse_args()
    
    run_training(args)

if __name__ == '__main__':
    main()
