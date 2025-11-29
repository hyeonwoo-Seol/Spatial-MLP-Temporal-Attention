import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm
import os
import random
import numpy as np
import sys
import argparse
import time
import matplotlib.pyplot as plt # 그래프 작성을 위한 라이브러리
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


    
def get_scheduler(optimizer, total_epochs, warmup_epochs):
    print(f"Using 'Warmup + Cosine Annealing' scheduler.")
    
    # 1. Warmup Scheduler
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
    
    # 2. Main Scheduler (Cosine Decay)
    main_scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs, eta_min=config.ETA_MIN)

    # 3. Sequential (Warmup -> Main)
    return SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_epochs])



def plot_training_results(train_losses, val_losses, train_accs, val_accs, save_dir):
    """
    학습 결과를 그래프로 그리고 저장하는 함수
    """
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # 1. Loss Graph
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss')
    plt.plot(epochs, val_losses, 'r-', label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper left') # 범례 위치: 왼쪽 상단
    plt.grid(True)

    # 2. Accuracy Graph
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b-', label='Train Acc')
    plt.plot(epochs, val_accs, 'r-', label='Val Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper left') # 범례 위치: 왼쪽 상단
    plt.grid(True)

    plt.tight_layout()
    
    # 그래프 저장
    save_path = os.path.join(save_dir, 'training_result.png')
    plt.savefig(save_path)
    print(f"Training graph saved to '{save_path}'")
    plt.close()



def train_one_epoch(model, source_loader, target_loader, criterion_cls, criterion_domain, criterion_fc, optimizer, device, scaler, epoch, args, global_step, total_steps):
    model.train()
    
    running_loss = 0.0
    correct_action = 0
    total_samples = 0
    
    desc = f"[Train Ep {epoch+1}]"
    
    # zip을 사용하여 배치 쌍 생성 (길이는 짧은 쪽 기준)
    min_len = min(len(source_loader), len(target_loader))
    pbar = tqdm(zip(source_loader, target_loader), total=min_len, desc=desc, leave=False, colour='green')
    
    for (src_data), (tgt_data) in pbar:
        # --- Alpha Scheduling (GRL) ---
        # 수식: alpha = 2 / (1 + exp(-10 * p)) - 1
        # p: 학습 진행률 (0 ~ 1)
        p = float(global_step) / total_steps
        alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1
        
        # --- 1. Source Data Unpacking ---
        src_view1, src_view2, src_labels, _, _ = src_data
        src_view1 = src_view1.to(device)
        src_view2 = src_view2.to(device)
        src_labels = src_labels.to(device)
        
        # --- 2. Target Data Unpacking ---
        tgt_view1, tgt_view2, _, _, _ = tgt_data 
        tgt_view1 = tgt_view1.to(device)
        
        batch_size_src = src_view1.size(0)
        batch_size_tgt = tgt_view1.size(0)
        
        # --- 3. Domain Labels ---
        domain_label_src = torch.zeros(batch_size_src, 1).to(device)
        domain_label_tgt = torch.ones(batch_size_tgt, 1).to(device)
        
        optimizer.zero_grad()
        
        with autocast('cuda'):
            # ====================================================
            # (A) Source Stream Forward
            # ====================================================
            src_logits1, src_dom_logits1, src_feat1 = model(src_view1, alpha=alpha)
            src_logits2, src_dom_logits2, src_feat2 = model(src_view2, alpha=alpha)
            
            # 1. Action Classification Loss
            loss_cls = (criterion_cls(src_logits1, src_labels) + criterion_cls(src_logits2, src_labels)) * 0.5
            
            # 2. Feature Consistency Loss
            loss_fc_src = criterion_fc(src_feat1, src_feat2)
            
            # 3. Domain Loss (Source)
            loss_dom_src = criterion_domain(src_dom_logits1, domain_label_src)

            # ====================================================
            # (B) Target Stream Forward
            # ====================================================
            tgt_logits1, tgt_dom_logits1, tgt_feat1 = model(tgt_view1, alpha=alpha)
            
            # 4. Domain Loss (Target)
            loss_dom_tgt = criterion_domain(tgt_dom_logits1, domain_label_tgt)
            
            # ====================================================
            # (C) Total Loss
            # ====================================================
            loss_dom_total = (loss_dom_src + loss_dom_tgt) * 0.5
            
            # 최종 Loss 합산
            loss = loss_cls + \
                   (args.lambda_fc * loss_fc_src) + \
                   (args.lambda_dom * loss_dom_total)

        # Backward
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.GRAD_CLIP_NORM)
        scaler.step(optimizer)
        scaler.update()
        
        # Stats Update
        running_loss += loss.item() * batch_size_src
        total_samples += batch_size_src
        
        # Acc 계산
        _, predicted = torch.max(src_logits1, 1)
        correct_action += (predicted == src_labels).sum().item()
        
        # Global Step 증가
        global_step += 1
        
        # Logging
        pbar.set_postfix({
            'L_All': f"{loss.item():.3f}", 
            'Acc': f"{correct_action/total_samples:.3f}",
            'Alpha': f"{alpha:.3f}"
        })
        
    return running_loss / total_samples, correct_action / total_samples, global_step



def validate_one_epoch(model, loader, criterion_cls, device):
    model.eval()
    running_loss = 0.0
    correct_action = 0
    total_samples = 0
    
    with torch.no_grad():
        for view1, _, action_labels, _, _ in tqdm(loader, desc="[Val]", leave=False, colour='cyan'):
            view1 = view1.to(device)
            action_labels = action_labels.to(device)
            
            with autocast('cuda'):
                # Validation에서는 Alpha=0
                act_logits, _, _ = model(view1, alpha=0.0) 
                loss = criterion_cls(act_logits, action_labels)
                
            running_loss += loss.item() * view1.size(0)
            _, predicted = torch.max(act_logits, 1)
            correct_action += (predicted == action_labels).sum().item()
            total_samples += view1.size(0)
            
    return running_loss / total_samples, correct_action / total_samples



def run_training(args):
    # 설정 업데이트
    set_seed(config.SEED)
    config.LEARNING_RATE = args.lr
    config.DROPOUT = args.dropout
    # Alpha는 동적 스케줄링으로 사용됨
    
    config.PROB = args.prob
    config.ADAMW_WEIGHT_DECAY = args.weight_decay
    config.LABEL_SMOOTHING = args.smoothing
    
    device = config.DEVICE
    print(f"\n[Info] Device: {device}, Protocol: {args.protocol}")
    
    # ----------------------------------------------------------------------
    # 1. Dataset & Loader Setup
    # ----------------------------------------------------------------------
    g = torch.Generator()
    g.manual_seed(config.SEED)

    source_dataset = NTURGBDDataset(config.DATASET_PATH, split='train', max_frames=config.MAX_FRAMES, protocol=args.protocol)
    source_loader = torch.utils.data.DataLoader(
        source_dataset, batch_size=config.BATCH_SIZE // 2, 
        shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY, generator=g, drop_last=True
    )
    
    target_dataset = NTURGBDDataset(config.DATASET_PATH, split='target_train', max_frames=config.MAX_FRAMES, protocol=args.protocol)
    target_loader = torch.utils.data.DataLoader(
        target_dataset, batch_size=config.BATCH_SIZE // 2, 
        shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY, generator=g, drop_last=True
    )

    val_dataset = NTURGBDDataset(config.DATASET_PATH, split='val', max_frames=config.MAX_FRAMES, protocol=args.protocol)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, 
        shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY, generator=g
    )
    
    print(f"Source Samples: {len(source_dataset)}, Target Samples: {len(target_dataset)}, Val Samples: {len(val_dataset)}")

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
    
    # 스케줄러 선택 로직 제거하고 표준 Warmup+CosineDecay 사용
    scheduler = get_scheduler(optimizer, config.EPOCHS, config.WARMUP_EPOCHS)
    
    scaler = GradScaler('cuda')
    
    criterion_cls = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
    criterion_domain = nn.BCEWithLogitsLoss()
    criterion_fc = FeatureConsistencyLoss(lambd=0.005) 
    
    # 4. Training Loop Prep
    best_acc = 0.0
    start_epoch = 0 # 학습 시작 Epoch
    trial_save_dir = os.path.join(config.SAVE_DIR, args.study_name, f"trial_{args.trial_number}")
    os.makedirs(trial_save_dir, exist_ok=True)
    
    # [추가] 기록 저장을 위한 리스트 초기화
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    # --- Auto Resume Logic (수정됨) ---
    # args.resume이 없더라도 --auto-resume이 켜져있고 파일이 존재하면 로드
    default_ckpt_path = os.path.join(trial_save_dir, "best_model.pth.tar")
    
    if args.auto_resume and not args.resume and os.path.exists(default_ckpt_path):
        print(f"[Auto-Resume] Found existing checkpoint at '{default_ckpt_path}'. Resuming...")
        args.resume = default_ckpt_path
    elif not args.resume and os.path.exists(default_ckpt_path):
        print(f"[Warning] Found existing checkpoint at '{default_ckpt_path}' but NOT loading it.")
        print(f"          If you want to resume, use --resume or --auto-resume.")
    
    # --- Resume Logic (체크포인트 불러오기) ---
    if args.resume and os.path.exists(args.resume):
        print(f"Loading checkpoint from '{args.resume}'...")
        # load_checkpoint 함수가 optimizer와 scheduler 상태도 로드하도록 utils.py에 구현되어 있어야 함
        checkpoint = load_checkpoint(args.resume, model, optimizer, scheduler, device)
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint.get('best_acc', 0.0)
        
        # [추가] 이전 학습 기록 복구
        train_losses = checkpoint.get('train_losses', [])
        val_losses = checkpoint.get('val_losses', [])
        train_accs = checkpoint.get('train_accs', [])
        val_accs = checkpoint.get('val_accs', [])
        
        print(f"Resumed from Epoch {start_epoch}, Best Acc: {best_acc:.4f}")

    steps_per_epoch = min(len(source_loader), len(target_loader))
    total_steps = config.EPOCHS * steps_per_epoch
    
    # Global Step 계산 (Resume 시 이전 Step부터 이어서 계산)
    global_step = start_epoch * steps_per_epoch
    
    print(f"Start Training: {config.EPOCHS} Epochs (Total Steps: {total_steps})")
    
    # Epoch Loop (start_epoch부터 시작)
    for epoch in range(start_epoch, config.EPOCHS):
        
        train_loss, train_acc, global_step = train_one_epoch(
            model, source_loader, target_loader, 
            criterion_cls, criterion_domain, criterion_fc, 
            optimizer, device, scaler, epoch, args, 
            global_step, total_steps
        )
        
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion_cls, device)
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # [추가] 결과 기록
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f"Ep [{epoch+1}/{config.EPOCHS}] LR: {current_lr:.6f} | Train Acc(Src): {train_acc:.4f} | Val Acc(Tgt): {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            # --- Checkpoint Saving (Optimizer & Scheduler & History 포함) ---
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                # [추가] 학습 기록 저장
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accs': train_accs,
                'val_accs': val_accs
            }, directory=trial_save_dir, filename="best_model.pth.tar")
            
    # [추가] 모든 학습 종료 후 결과 그래프 생성
    print("Generating training graphs...")
    plot_training_results(train_losses, val_losses, train_accs, val_accs, trial_save_dir)
            
    print(f"Training Finished. Best Val Acc: {best_acc:.4f}")
    return best_acc





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=str, default='xsub', choices=['xsub', 'xview'])
    parser.add_argument('--study-name', type=str, default='default_study')
    parser.add_argument('--trial-number', type=int, default=0)
    
    # [추가] Resume Argument
    parser.add_argument('--resume', type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument('--auto-resume', action='store_true', help="Automatically resume from current trial dir if checkpoint exists")
    
    # Hyperparameters
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
