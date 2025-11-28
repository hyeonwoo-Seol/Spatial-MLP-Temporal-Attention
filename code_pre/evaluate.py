# evaluate.py
# >> X-SUB 프로토콜로 평가하기
# python evaluate.py -c checkpoints/best_model.pth.tar
# >> X-SUB 평가하고 t-SNE 시각화하기
# python evaluate.py -c checkpoints/best_model.pth.tar --protocol xsub --tsne
# >> X-View 평가하기
# python evaluate.py -c checkpoints/best_model.pth.tar --protocol xview
# >> X-View 평가하고 t-SNE 시각화하기
# python evaluate.py -c checkpoints/best_model.pth.tar --protocol xview --tsne

import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
from torch.amp import autocast
import numpy as np
import os

from thop import profile, clever_format
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# 직접 작성한 파일들 임포트
import config
from ntu_data_loader import NTURGBDDataset, DataLoader
# SlowFast_Transformer
from model import SlowFast_Transformer 
from utils import load_checkpoint

## #--------------------------------------------------------------------
# 모델의 Top-K 정확도를 계산한다.
## #--------------------------------------------------------------------
def calculate_topk_accuracy(outputs, labels, k=5):
    """Top-K 정확도 계산"""
    with torch.no_grad():
        batch_size = labels.size(0)
        _, pred = outputs.topk(k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(labels.view(1, -1).expand_as(pred))
        
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        return correct_k.item()


## #--------------------------------------------------------------------
# t-SNE 플롯을 생성하고 저장한다.
## #--------------------------------------------------------------------
def generate_tsne_plot(features_list, action_labels_list, domain_labels_list, protocol):
    print("\n--- Generating t-SNE Visualization ---")
    print("This may take a few minutes...")

    # 1. 데이터를 하나의 NumPy 배열로 결합
    # Hook에서 (N, C) 형태의 텐서 리스트로 수집됨
    all_features = torch.cat(features_list, dim=0).numpy()
    all_action_labels = torch.cat(action_labels_list, dim=0).numpy()
    all_domain_labels = torch.cat(domain_labels_list, dim=0).numpy()

    # 2. 특징 스케일링 (t-SNE는 스케일에 민감함)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(all_features)

    # 3. t-SNE 실행
    tsne = TSNE(n_components=2, 
                perplexity=30, # 데이터 밀도에 따라 조절 가능
                n_iter=1000, 
                random_state=config.SEED,
                n_jobs=-1) # 모든 CPU 코어 사용
    
    features_2d = tsne.fit_transform(features_scaled)

    # 4. 시각화 (2개 플롯)
    print("Plotting...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    num_classes = config.NUM_CLASSES
    num_domains = 2 # 훈련 그룹(0) / 검증 그룹(1)

    # --- 플롯 1: 행동 클래스(Action) 기준 ---
    sns.scatterplot(
        x=features_2d[:, 0],
        y=features_2d[:, 1],
        hue=all_action_labels,
        palette=sns.color_palette("hsv", num_classes),
        s=10, # 점 크기
        alpha=0.7,
        legend=None, # 범례가 너무 많아 제거
        ax=ax1
    )
    ax1.set_title(f't-SNE colored by Action Labels (K={num_classes})', fontsize=16)

    # --- 플롯 2: 도메인(Subject/View) 기준 ---
    # [수정됨] GRL의 목표는 이 플롯에서 색이 섞여 보이게 하는 것
    domain_names = {0: 'Source Domain (Train)', 1: 'Target Domain (Val)'}
    domain_labels_named = [domain_names[label] for label in all_domain_labels]
    
    sns.scatterplot(
        x=features_2d[:, 0],
        y=features_2d[:, 1],
        hue=domain_labels_named,
        palette=['blue', 'orange'],
        s=10,
        alpha=0.7,
        legend="full",
        ax=ax2
    )
    ax2.set_title(f't-SNE colored by Domain ({protocol})', fontsize=16)

    # --- 파일 저장 ---
    save_path = f"tsne_plot_{protocol}.png"
    plt.savefig(save_path, dpi=150)
    print(f"t-SNE visualization saved to '{save_path}'")
    plt.close()

def calculate_flops(model, device):

    print("\n--- Calculating FLOPs (using thop) ---")
    model.eval() 
    T_fast = config.MAX_FRAMES
    T_slow = config.MAX_FRAMES // 2 
    
    # 더미 입력 생성
    sample_fast = torch.randn(1, config.NUM_COORDS, T_fast, config.NUM_JOINTS).to(device)
    sample_slow = torch.randn(1, config.NUM_COORDS, T_slow, config.NUM_JOINTS).to(device)
    
    # thop.profile은 입력을 리스트 또는 튜플로 받습니다.
    inputs = (sample_fast, sample_slow) 
    
    try:
        # thop.profile은 (total_ops, total_params)를 반환합니다.
        total_ops, total_params = profile(model, inputs=inputs, verbose=False)
        
        # thop는 MACs(Multiply-Accumulate)를 계산합니다.
        # GFLOPs는 G-MACs * 2 로 계산하는 것이 표준입니다.
        gflops = (total_ops * 2) / 1e9
        
        print(f"Model GFLOPs (MACs * 2): {gflops:.2f} G")
        print(f"(thop이 계산한 G-MACs: {total_ops / 1e9:.2f} G)")

    except Exception as e:
        print(f"Error during FLOPs calculation with thop: {e}")
        print("이것은 'thop'가 사용자님의 모델에 있는 커스텀 모듈을")
        print("인식하지 못할 때 발생할 수 있습니다. 이 경우 FLOPs 계산은 실패합니다.")
    print("---------------------------\n") # 파라미터 계산 후 출력으로 이동

## #--------------------------------------------------------------------
# 모델의 파라미터 크기 (Params)를 계산한다.
## #--------------------------------------------------------------------
def calculate_params(model):
    print("--- Calculating Params ---")
    try:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total Parameters: {total_params / 1e6:.2f} M")
        print(f"Trainable Parameters: {trainable_params / 1e6:.2f} M")
    except Exception as e:
        print(f"Error during parameter calculation: {e}")
    print("---------------------------\n")

## #--------------------------------------------------------------------
# 저장된 모델을 불러와 검증 데이터셋으로 성능을 평가한다.
## #--------------------------------------------------------------------
def evaluate_model(checkpoint_path, protocol, run_tsne):
    
    # --- X-View 경고 및 설정 ---
    if protocol == 'xview':
        print("--- [X-View] Evaluation Mode ---")
    else: # 'xsub'
        print("--- [X-Sub] Evaluation Mode ---")

    split_name = 'val'
    
    # >> 설정값 불러오기
    device = config.DEVICE
    print(f"Using device: {device}")

    # 1. 체크포인트를 먼저 로드해서 하이퍼파라미터를 확인
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at '{checkpoint_path}'")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 2. 저장된 Optuna 파라미터로 config 값을 덮어쓰기 (매우 중요!)
    if 'optuna_params' in checkpoint:
        print("--- Loading Hyperparameters from Checkpoint ---")
        for key, value in checkpoint['optuna_params'].items():
            if hasattr(config, key):
                print(f"Overriding config: {key} = {value}")
                setattr(config, key, value) # config.DROPOUT = value
            else:
                print(f"Warning: Param '{key}' not found in config.py")
    else:
        print("Warning: 'optuna_params' not found in checkpoint. Using default config.")
        
    # >> 데이터 로딩
    try:
        val_dataset = NTURGBDDataset(
            data_path=config.DATASET_PATH, 
            split=split_name,
            max_frames=config.MAX_FRAMES,
            protocol = protocol
        )
    except FileNotFoundError:
        print(f"오류: `split='{split_name}'`에 해당하는 데이터를 찾을 수 없습니다.")
        print("X-View를 실행하려면 `ntu_data_loader.py`를 먼저 수정해야 합니다.")
        return

    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )

    alpha_from_checkpoint = checkpoint.get('optuna_params', {}).get('ADVERSARIAL_ALPHA', 1.0)
    print(f"Loading model with GRL alpha = {alpha_from_checkpoint:.3f}")
    
    # 모델 초기화
    model = SlowFast_Transformer(
        num_joints=config.NUM_JOINTS,
        num_coords=config.NUM_COORDS,
        num_classes=config.NUM_CLASSES,
        fast_dims=config.FAST_DIMS,
        slow_dims=config.SLOW_DIMS,
        num_subjects=config.NUM_SUBJECTS,
        alpha=alpha_from_checkpoint
    ).to(device)

    # 4. 모델 가중치 로드
    model.load_state_dict(checkpoint['state_dict'])
    print(f"Model weights loaded from '{checkpoint_path}'")

    # FLOPs 및 Params 계산
    calculate_params(model)
    
    calculate_flops(model, device)
    
    criterion = nn.CrossEntropyLoss()

    # --- t-SNE를 위한 Hook 준비 ---
    all_features_list = []
    all_action_labels_list = []
    all_domain_labels_list = []

    def hook_fn(module, input, output):
        # GRL의 입력(input[0])이 GRL 직전의 `combined_summary` 특징
        all_features_list.append(input[0].detach().cpu())

    if run_tsne:
        print("Registering t-SNE hook on 'model.grad_reversal' layer...")
        # GRL 레이어에 forward hook 등록
        model.grad_reversal.register_forward_hook(hook_fn)

    # --- 평가 루프 ---
    model.eval()
    total_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0 # Top-5
    correct_subject = 0
    total_samples = 0
    
    eval_bar = tqdm(val_loader, desc=f"[Evaluate {protocol.upper()}]", colour="yellow")
    with torch.no_grad():
        for data_fast, data_slow, action_labels, subject_labels in eval_bar:
            
            data_fast = data_fast.to(device)
            data_slow = data_slow.to(device)
            action_labels = action_labels.to(device)
            subject_labels = subject_labels.to(device)
            
            with autocast(device_type=device):
                outputs_action, outputs_subject = model(data_fast, data_slow)
                loss = criterion(outputs_action, action_labels)
            
            total_loss += loss.item() * data_fast.size(0)
            
            # --- 정확도 계산 ---
            total_samples += action_labels.size(0)
            
            # Top-1
            _, predicted_action = torch.max(outputs_action.data, 1)
            correct_top1 += (predicted_action == action_labels).sum().item()
            
            # Top-5
            correct_top5 += calculate_topk_accuracy(outputs_action, action_labels, k=5)

            # Subject
            _, predicted_subject = torch.max(outputs_subject.data, 1)
            correct_subject += (predicted_subject == subject_labels).sum().item()

            # --- t-SNE 데이터 수집 (Hook이 자동으로 수집) ---
            if run_tsne:
                all_action_labels_list.append(action_labels.cpu())
                if protocol == 'xview':
                    # xview는 0(Train) 또는 1(Val)을 사용합니다.
                    all_domain_labels_list.append(subject_labels.cpu())
                else: # xsub
                    # xsub 평가(val_loader)는 모두 검증용 피실험자(Target Domain)입니다.
                    # 따라서 '1'로 채워진 텐서를 추가합니다.
                    domain_labels = torch.ones_like(action_labels, dtype=torch.int)
                    all_domain_labels_list.append(domain_labels.cpu())

            # --- 진행률 표시 ---
            avg_loss = total_loss / total_samples
            avg_acc_t1 = correct_top1 / total_samples
            avg_acc_t5 = correct_top5 / total_samples # Top-5 추가
            avg_sub_acc = correct_subject / total_samples
            
            eval_bar.set_postfix(
                loss=f"{avg_loss:.4f}", 
                acc_T1=f"{avg_acc_t1:.4f}", # T1
                acc_T5=f"{avg_acc_t5:.4f}", # T5
                acc_SUB=f"{avg_sub_acc:.4f}"
            )

    # --- 최종 결과 출력 ---
    final_loss = total_loss / total_samples
    final_acc_top1 = correct_top1 / total_samples
    final_acc_top5 = correct_top5 / total_samples
    final_subject_acc = correct_subject / total_samples
    
    print(f"\n--- Evaluation Finished ({protocol.upper()}) ---")
    print(f"Average Loss: {final_loss:.4f}")
    print(f"Top-1 Action Accuracy: {final_acc_top1 * 100:.2f}%") # Top-1
    print(f"Top-5 Action Accuracy: {final_acc_top5 * 100:.2f}%") # Top-5
    print(f"Subject Accuracy: {final_subject_acc * 100:.2f}% (GRL 성공 시 낮은 값)")

    # --- t-SNE 시각화 실행 ---
    if run_tsne:
        generate_tsne_plot(
            all_features_list, 
            all_action_labels_list, 
            all_domain_labels_list,
            protocol
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate the SlowFast Transformer model.")
    parser.add_argument('-c', '--checkpoint', type=str, required=True,
                        help="Path to the model checkpoint file (e.g., 'checkpoints/best_model.pth.tar')")
    # [추가됨] X-Sub / X-View 선택
    parser.add_argument('-p', '--protocol', type=str, default='xsub', choices=['xsub', 'xview'],
                        help="Evaluation protocol: 'xsub' (Cross-Subject) or 'xview' (Cross-View). Default: 'xsub'")
    # [추가됨] t-SNE 실행 여부
    parser.add_argument('--tsne', action='store_true',
                        help="Run t-SNE visualization. (May take several minutes and high memory)")
    
    args = parser.parse_args()
    
    evaluate_model(args.checkpoint, args.protocol, args.tsne)
