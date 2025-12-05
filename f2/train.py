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
import matplotlib.pyplot as plt
import config

# 모듈 import
from ntu_data_loader import NTURGBDDataset
from model import ST_GRL_Model
from utils import calculate_accuracy, save_checkpoint, load_checkpoint

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
    if len(train_losses) == 0:
        return

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # 1. Loss Graph
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss')
    plt.plot(epochs, val_losses, 'r-', label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True)

    # 2. Accuracy Graph
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b-', label='Train Acc')
    plt.plot(epochs, val_accs, 'r-', label='Val Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True)

    plt.tight_layout()
    
    # 그래프 저장
    save_path = os.path.join(save_dir, 'training_result.png')
    plt.savefig(save_path)
    plt.close()



def train_one_epoch(model, source_loader, criterion_cls, optimizer, device, scaler, epoch, args):
    model.train()
    
    running_loss = 0.0
    correct_action = 0
    total_samples = 0
    
    desc = f"[Train Ep {epoch+1}]"
    
    # 단일 source_loader 순회
    pbar = tqdm(source_loader, desc=desc, leave=False, colour='green')
    
    for batch in pbar:
        features, labels = batch
        features = features.to(device)
        labels = labels.to(device)
        
        batch_size = features.size(0)
        
        optimizer.zero_grad()
        
        with autocast('cuda'):
            # 모델 Forward: alpha 인자 제거
            logits = model(features)
            
            # Loss 계산: 오직 Classification Loss만 계산
            loss = criterion_cls(logits, labels)

        # Backward
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.GRAD_CLIP_NORM)
        scaler.step(optimizer)
        scaler.update()
        
        # Stats Update
        running_loss += loss.item() * batch_size
        total_samples += batch_size
        
        # Acc 계산
        _, predicted = torch.max(logits, 1)
        correct_action += (predicted == labels).sum().item()
        
        # Logging
        pbar.set_postfix({
            'Loss': f"{loss.item():.3f}", 
            'Acc': f"{correct_action/total_samples:.3f}"
        })
        
    return running_loss / total_samples, correct_action / total_samples



def validate_one_epoch(model, loader, criterion_cls, device):
    model.eval()
    running_loss = 0.0
    correct_action = 0
    total_samples = 0
    
    with torch.no_grad():
        for features, labels in tqdm(loader, desc="[Val]", leave=False, colour='cyan'):
            features = features.to(device)
            labels = labels.to(device)
            
            with autocast('cuda'):
                logits = model(features) 
                loss = criterion_cls(logits, labels)
                
            running_loss += loss.item() * features.size(0)
            _, predicted = torch.max(logits, 1)
            correct_action += (predicted == labels).sum().item()
            total_samples += features.size(0)
            
    return running_loss / total_samples, correct_action / total_samples



def run_training(args):
    set_seed(config.SEED)
    config.LEARNING_RATE = args.lr
    config.DROPOUT = args.dropout
    
    config.PROB = args.prob
    config.ADAMW_WEIGHT_DECAY = args.weight_decay
    config.LABEL_SMOOTHING = args.smoothing
    
    device = config.DEVICE
    print(f"\n[Info] Device: {device}, Protocol: {args.protocol}")
    print(f"[Info] Mode: Pure Supervised Learning (No UDA/GRL)")
    
    # ----------------------------------------------------------------------
    # 1. Dataset & Loader Setup
    # ----------------------------------------------------------------------
    g = torch.Generator()
    g.manual_seed(config.SEED)

    # Source Dataset (Train)
    source_dataset = NTURGBDDataset(config.DATASET_PATH, split='train', max_frames=config.MAX_FRAMES, protocol=args.protocol)
    source_loader = torch.utils.data.DataLoader(
        source_dataset, batch_size=config.BATCH_SIZE, # [수정] Batch Size 나누기(/2) 제거
        shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY, generator=g, drop_last=True
    )
    
    # Validation Dataset
    val_dataset = NTURGBDDataset(config.DATASET_PATH, split='val', max_frames=config.MAX_FRAMES, protocol=args.protocol)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, 
        shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY, generator=g
    )
    
    print(f"Training Samples: {len(source_dataset)}, Validation Samples: {len(val_dataset)}")

    # 2. Model Init
    model = ST_GRL_Model(
        num_joints=config.NUM_JOINTS,
        num_coords=config.NUM_COORDS,
        num_classes=config.NUM_CLASSES,
        hidden_dim=config.HIDDEN_DIM,
        window_size=config.WINDOW_SIZE,
        dropout=config.DROPOUT
    ).to(device)
    
    # 3. Optim & Loss
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.ADAMW_WEIGHT_DECAY)
    
    scheduler = get_scheduler(optimizer, config.EPOCHS, config.WARMUP_EPOCHS)
    
    scaler = GradScaler('cuda')
    
    criterion_cls = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
    
    # 4. Training Loop Prep
    best_acc = 0.0
    start_epoch = 0 
    trial_save_dir = os.path.join(config.SAVE_DIR, args.study_name, f"trial_{args.trial_number}")
    os.makedirs(trial_save_dir, exist_ok=True)
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    # --- Resume Logic ---
    ckpt_path = None
    last_ckpt_path = os.path.join(trial_save_dir, "last_model.pth.tar")
    best_ckpt_path = os.path.join(trial_save_dir, "best_model.pth.tar")

    if args.resume:
        ckpt_path = args.resume
    elif args.auto_resume:
        if os.path.exists(last_ckpt_path):
            ckpt_path = last_ckpt_path
            print(f"[Auto-Resume] Found 'last_model'. Resuming...")
        elif os.path.exists(best_ckpt_path):
            ckpt_path = best_ckpt_path
            print(f"[Auto-Resume] Found 'best_model'. Resuming...")
    
    if ckpt_path and os.path.exists(ckpt_path):
        print(f"Loading checkpoint from '{ckpt_path}'...")
        checkpoint = load_checkpoint(ckpt_path, model, optimizer, scheduler, device)
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint.get('best_acc', 0.0)
        
        train_losses = checkpoint.get('train_losses', [])
        val_losses = checkpoint.get('val_losses', [])
        train_accs = checkpoint.get('train_accs', [])
        val_accs = checkpoint.get('val_accs', [])
        
        if len(train_losses) > start_epoch:
            train_losses = train_losses[:start_epoch]
            val_losses = val_losses[:start_epoch]
            train_accs = train_accs[:start_epoch]
            val_accs = val_accs[:start_epoch]
            
        print(f"Resumed from Epoch {start_epoch}, Best Acc: {best_acc:.4f}")

    print(f"Start Training: {config.EPOCHS} Epochs")
    
    try:
        for epoch in range(start_epoch, config.EPOCHS):
            train_loss, train_acc = train_one_epoch(
                model, source_loader, criterion_cls, 
                optimizer, device, scaler, epoch, args
            )
            
            val_loss, val_acc = validate_one_epoch(model, val_loader, criterion_cls, device)
            
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            print(f"Ep [{epoch+1}/{config.EPOCHS}] LR: {current_lr:.6f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
            
            plot_training_results(train_losses, val_losses, train_accs, val_accs, trial_save_dir)
            
            save_dict = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accs': train_accs,
                'val_accs': val_accs
            }

            if val_acc > best_acc:
                best_acc = val_acc
                save_dict['best_acc'] = best_acc 
                save_checkpoint(save_dict, directory=trial_save_dir, filename="best_model.pth.tar")
                
            save_checkpoint(save_dict, directory=trial_save_dir, filename="last_model.pth.tar")
                
    except KeyboardInterrupt:
        print("\n\n[Warning] Training interrupted by user.")
        plot_training_results(train_losses, val_losses, train_accs, val_accs, trial_save_dir)
        sys.exit(0)
            
    print("Generating final graphs...")
    plot_training_results(train_losses, val_losses, train_accs, val_accs, trial_save_dir)
            
    print(f"Training Finished. Best Val Acc: {best_acc:.4f}")
    return best_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=str, default='xsub', choices=['xsub', 'xview'])
    parser.add_argument('--study-name', type=str, default='default_study')
    parser.add_argument('--trial-number', type=int, default=0)
    
    # Resume Argument
    parser.add_argument('--resume', type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument('--auto-resume', action='store_true', help="Automatically resume")
    
    # Hyperparameters
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE)
    parser.add_argument('--dropout', type=float, default=config.DROPOUT)
    parser.add_argument('--prob', type=float, default=config.PROB)
    parser.add_argument('--weight-decay', type=float, default=config.ADAMW_WEIGHT_DECAY)
    parser.add_argument('--smoothing', type=float, default=config.LABEL_SMOOTHING)
    

    args = parser.parse_args()
    
    run_training(args)

if __name__ == '__main__':
    main()
