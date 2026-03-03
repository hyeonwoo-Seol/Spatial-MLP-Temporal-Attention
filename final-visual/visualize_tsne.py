import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import os
import argparse
from tqdm import tqdm

import config
from model import ST_Model
from ntu_data_loader import NTURGBDDataset

# [설정] 시각화할 클래스 개수 (너무 많으면 안 보임)
# 예: 0~9번 클래스 (10개)만 시각화하여 군집성을 강조
TARGET_CLASSES = list(range(0, 10)) 
# 혹은 None으로 두면 전체 클래스 시각화 (비추천, 너무 복잡함)
# TARGET_CLASSES = None 

def extract_features(model, loader, device):
    model.eval()
    features_list = []
    labels_list = []
    
    print("--- Extracting Features for t-SNE ---")
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting"):
            inputs, labels, _ = batch
            inputs = inputs.to(device)
            
            # 1. 모델에서 Features 추출 (return_features=True)
            _, features = model(inputs, return_features=True)
            
            features_list.append(features.cpu().numpy())
            labels_list.append(labels.numpy())
            
    # 리스트를 하나의 거대한 Numpy 배열로 변환
    features_all = np.concatenate(features_list, axis=0)
    labels_all = np.concatenate(labels_list, axis=0)
    
    return features_all, labels_all

def visualize_tsne(features, labels, save_path):
    print("--- Running t-SNE (This may take a while) ---")
    
    # 2. t-SNE 차원 축소 (128차원 -> 2차원)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_results = tsne.fit_transform(features)
    
    # 3. 데이터프레임 생성 (시각화 용이성)
    df = pd.DataFrame({
        'x': tsne_results[:, 0],
        'y': tsne_results[:, 1],
        'label': labels
    })
    
    # 4. 필터링 (원하는 클래스만 선택)
    if TARGET_CLASSES:
        df = df[df['label'].isin(TARGET_CLASSES)]
        print(f"Visualizing only classes: {TARGET_CLASSES}")

    plt.figure(figsize=(12, 10))
    
    # 5. Scatter Plot 그리기
    # palette='tab10'은 색상이 뚜렷해서 구분하기 좋음
    sns.scatterplot(
        x='x', y='y',
        hue='label',
        palette=sns.color_palette("hsv", len(df['label'].unique())),
        data=df,
        legend="full",
        alpha=0.7
    )
    
    plt.title('t-SNE Visualization of Action Features', fontsize=16)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300)
    print(f"Saved t-SNE plot to {save_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    # 학습된 모델 경로 (Best Model 권장)
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to best_model.pth.tar')
    parser.add_argument('--protocol', type=str, default='xsub')
    args = parser.parse_args()

    device = config.DEVICE
    
    # 1. 데이터셋 로드 (Validation Set 사용)
    val_dataset = NTURGBDDataset(config.DATASET_PATH, split='val', protocol=args.protocol)
    # 셔플을 꺼야 라벨 순서가 맞지만, 추출만 하므로 상관없음. 다만 섞어서 뽑는게 좋음.
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, 
        num_workers=config.NUM_WORKERS, pin_memory=True
    )
    
    # 2. 모델 초기화 및 가중치 로드
    model = ST_Model(
        num_joints=config.NUM_JOINTS,
        num_coords=config.NUM_COORDS,
        num_classes=config.NUM_CLASSES,
        hidden_dim=config.HIDDEN_DIM,
        window_size=config.WINDOW_SIZE,
        dropout=config.DROPOUT
    ).to(device)
    
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    
    # 3. Feature 추출
    features, labels = extract_features(model, val_loader, device)
    
    # 4. 시각화 실행
    save_dir = os.path.dirname(args.checkpoint)
    save_path = os.path.join(save_dir, 'tsne_visualization.png')
    visualize_tsne(features, labels, save_path)

if __name__ == '__main__':
    main()
