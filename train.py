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

def train_one_epoch(model, source_loader, target_loader, criterion_cls, criterion_domain, criterion_fc, optimizer, device, scaler, epoch, args):
    model.train()
    
    running_loss = 0.0
    correct_action = 0
    total_samples = 0
    
    # 두 로더 중 길이가 짧은 쪽에 맞추거나, 긴 쪽에 맞추기 위해 zip 사용
    # 일반적으로 Target 데이터가 적을 수 있으므로 itertools.cycle을 쓰기도 하지만, 
    # 여기서는 간단히 zip을 사용하여 미니배치 쌍을 만듭니다.
    # Source와 Target 배치가 1:1로 매핑되어 들어갑니다.
    
    desc = f"[Train Ep {epoch+1}]"
    # zip의 길이는 더 짧은 로더 기준이 되므로, 긴 로더의 데이터 일부는 이 에폭에서 안 쓰일 수 있음.
    # 하지만 매 에폭마다 shuffle=True이므로 전체적으로는 골고루 학습됨.
    min_len = min(len(source_loader), len(target_loader))
    pbar = tqdm(zip(source_loader, target_loader), total=min_len, desc=desc, leave=False, colour='green')
    
    for (src_data), (tgt_data) in pbar:
        # --- 1. Source Data Unpacking ---
        src_view1, src_view2, src_labels, _, _ = src_data
        src_view1 = src_view1.to(device)
        src_view2 = src_view2.to(device)
        src_labels = src_labels.to(device)
        
        # --- 2. Target Data Unpacking ---
        tgt_view1, tgt_view2, _, _, _ = tgt_data # Target Label은 학습에 사용하지 않음 (Unsupervised)
        tgt_view1 = tgt_view1.to(device)
        # Target Consistency를 쓰고 싶다면 tgt_view2도 사용 가능, 여기선 GRL 기본에 집중
        
        batch_size_src = src_view1.size(0)
        batch_size_tgt = tgt_view1.size(0)
        
        # --- 3. Domain Labels ---
        # Source = 0, Target = 1
        domain_label_src = torch.zeros(batch_size_src, 1).to(device)
        domain_label_tgt = torch.ones(batch_size_tgt, 1).to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            # ====================================================
            # (A) Source Stream Forward
            # ====================================================
            # Action Classifier & Domain Discriminator for Source
            # alpha는 Gradient Reversal Layer의 강도 (스케줄링 될 수 있음)
            src_logits1, src_dom_logits1, src_feat1 = model(src_view1, alpha=args.alpha)
            src_logits2, src_dom_logits2, src_feat2 = model(src_view2, alpha=args.alpha)
            
            # 1. Action Classification Loss (Source Only)
            loss_cls = (criterion_cls(src_logits1, src_labels) + criterion_cls(src_logits2, src_labels)) * 0.5
            
            # 2. Feature Consistency Loss (Source)
            loss_fc_src = criterion_fc(src_feat1, src_feat2)
            
            # 3. Domain Loss (Source -> 0)
            loss_dom_src = criterion_domain(src_dom_logits1, domain_label_src)

            # ====================================================
            # (B) Target Stream Forward
            # ====================================================
            # Target 데이터는 Action Label이 없으므로 Classification Loss 계산 안 함
            # 오직 Domain Discriminator와 (선택적으로) Consistency Loss만 계산
            tgt_logits1, tgt_dom_logits1, tgt_feat1 = model(tgt_view1, alpha=args.alpha)
            
            # 4. Domain Loss (Target -> 1)
            loss_dom_tgt = criterion_domain(tgt_dom_logits1, domain_label_tgt)
            
            # (Optional) Target Feature Consistency
            # 만약 Target에 대해서도 Temporal Robustness를 주고 싶다면 계산 (Unsupervised Augmentation)
            # tgt_logits2, _, tgt_feat2 = model(tgt_view2.to(device), alpha=args.alpha)
            # loss_fc_tgt = criterion_fc(tgt_feat1, tgt_feat2)
            # 여기서는 논문(GRL) 구현의 핵심인 Domain Loss에 집중하기 위해 생략하거나 가중치를 낮게 줄 수 있음
            
            # ====================================================
            # (C) Total Loss
            # ====================================================
            # Total Domain Loss
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
        
        # Acc 계산 (Source View1 기준)
        _, predicted = torch.max(src_logits1, 1)
        correct_action += (predicted == src_labels).sum().item()
        
        # Logging
        pbar.set_postfix({
            'L_All': f"{loss.item():.3f}", 
            'L_Cls': f"{loss_cls.item():.3f}",
            'L_Dom': f"{loss_dom_total.item():.3f}",
            'Acc': f"{correct_action/total_samples:.3f}"
        })
        
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
                # Validation에서는 Alpha=0 (GRL 영향 없음)
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
    config.ADVERSARIAL_ALPHA = args.alpha
    config.PROB = args.prob
    config.ADAMW_WEIGHT_DECAY = args.weight_decay
    config.LABEL_SMOOTHING = args.smoothing
    
    device = config.DEVICE
    print(f"\n[Info] Device: {device}, Protocol: {args.protocol}")
    
    # ----------------------------------------------------------------------
    # 1. Dataset & Loader Setup (Source & Target)
    # ----------------------------------------------------------------------
    g = torch.Generator()
    g.manual_seed(config.SEED)

    # A. Source Loader (Labeled, Train Split)
    source_dataset = NTURGBDDataset(config.DATASET_PATH, split='train', max_frames=config.MAX_FRAMES, protocol=args.protocol)
    source_loader = torch.utils.data.DataLoader(
        source_dataset, batch_size=config.BATCH_SIZE // 2, # 배치 사이즈를 반으로 나누어 Source/Target 합쳐서 원래 배치 크기 유지 권장
        shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY, generator=g, drop_last=True
    )
    
    # B. Target Loader (Unlabeled, Target Train Split) -> GRL 학습용
    target_dataset = NTURGBDDataset(config.DATASET_PATH, split='target_train', max_frames=config.MAX_FRAMES, protocol=args.protocol)
    target_loader = torch.utils.data.DataLoader(
        target_dataset, batch_size=config.BATCH_SIZE // 2, 
        shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY, generator=g, drop_last=True
    )

    # C. Validation Loader (Labeled Target, Val Split) -> 평가용
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
    scheduler = get_scheduler(optimizer, args.scheduler, config.EPOCHS, config.WARMUP_EPOCHS)
    scaler = GradScaler()
    
    criterion_cls = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
    criterion_domain = nn.BCELoss() 
    criterion_fc = FeatureConsistencyLoss(lambd=0.005) 
    
    # 4. Training Loop
    best_acc = 0.0
    trial_save_dir = os.path.join(config.SAVE_DIR, args.study_name, f"trial_{args.trial_number}")
    os.makedirs(trial_save_dir, exist_ok=True)
    
    print(f"Start Training: {config.EPOCHS} Epochs")
    for epoch in range(config.EPOCHS):
        
        # Train one epoch with Source & Target
        train_loss, train_acc = train_one_epoch(
            model, source_loader, target_loader, 
            criterion_cls, criterion_domain, criterion_fc, 
            optimizer, device, scaler, epoch, args
        )
        
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion_cls, device)
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Ep [{epoch+1}/{config.EPOCHS}] LR: {current_lr:.6f} | Train Acc(Src): {train_acc:.4f} | Val Acc(Tgt): {val_acc:.4f}")
        
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
