# ntu_data_loader.py

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import config

# Protocol Definitions
TRAINING_SUBJECTS = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
TRAINING_CAMERAS = [2, 3]

class NTURGBDDataset(Dataset):
    def __init__(self, data_path, split='train', max_frames=300, protocol='xsub'):
        self.data_path = data_path
        self.split = split
        self.max_frames = max_frames
        self.protocol = protocol
        self.training_subjects = TRAINING_SUBJECTS
        
        # 샘플 리스트 로드
        self.samples = []
        self._load_data_path()

        # 통계치 로드 (Protocol에 따라 분기하여 Data Leakage 방지)
        base_dir = os.path.dirname(data_path.rstrip('/'))
        
        if self.protocol == 'xsub':
            stats_filename = 'stats_xsub.npz'
        elif self.protocol == 'xview':
            stats_filename = 'stats_xview.npz'
        else:
            stats_filename = 'stats_xsub.npz' # Fallback
            
        stats_path = os.path.join(base_dir, stats_filename)
        
        if os.path.exists(stats_path):
            stats = np.load(stats_path)
            self.mean = torch.from_numpy(stats['mean'].flatten()).float()
            self.std = torch.from_numpy(stats['std'].flatten()).float()
            print(f"[{split.upper()}] Normalization stats loaded from {stats_filename} (Protocol: {protocol}).")
        else:
            print(f"Warning: Stats not found at {stats_path}. Using identity.")
            self.mean = torch.zeros(12) # config.NUM_COORDS 대신 하드코딩된 차원 사용 가능 시 대체
            self.std = torch.ones(12)

    def _load_data_path(self):
        if not os.path.exists(self.data_path):
            print(f"Error: Data path {self.data_path} does not exist.")
            return

        filenames = sorted(os.listdir(self.data_path))
        for filename in filenames:
            if not filename.endswith('.pt'): continue
            
            # 파일명에서 정보 추출 (예: S001C001P001R001A001)
            # S: Setup, C: Camera, P: Performer, R: Replication, A: Action
            try:
                sid = int(filename[9:12]) # Subject ID
                cid = int(filename[5:8])  # Camera ID
            except ValueError:
                continue
            
            # Protocol Check
            if self.protocol == 'xsub':
                is_train_sample = sid in self.training_subjects
            elif self.protocol == 'xview':
                is_train_sample = cid in TRAINING_CAMERAS
            else:
                raise ValueError(f"Unknown protocol: {self.protocol}")
            
            # Split 분기
            if self.split == 'train':
                if is_train_sample:
                    self.samples.append(os.path.join(self.data_path, filename))
            else: # val
                if not is_train_sample:
                    self.samples.append(os.path.join(self.data_path, filename))

    def __len__(self):
        return len(self.samples)

    def _random_temporal_crop(self, feature, valid_length):
        """
        Feature Consistency를 위한 랜덤 시간 크롭 (View 2 생성용)
        valid_length 구간 내에서 랜덤하게 일부를 잘라내고, 나머지는 0으로 채움.
        """
        T, V, C = feature.shape
        
        # 유효 길이가 너무 짧으면 크롭하지 않음
        if valid_length < 10: 
            return feature.clone(), valid_length
        
        # 크롭 비율 설정 (예: 70% ~ 90%)
        crop_ratio = np.random.uniform(0.7, 0.9)
        crop_len = int(valid_length * crop_ratio)
        
        # 시작 지점 랜덤 설정
        max_start = valid_length - crop_len
        start = np.random.randint(0, max_start + 1)
        
        # 잘라내기
        cropped = feature[start : start + crop_len, :, :] # (crop_len, V, C)
        
        # 다시 원래 크기(MAX_FRAMES)로 패딩
        new_feat = torch.zeros_like(feature)
        new_feat[:crop_len, :, :] = cropped
        
        return new_feat, crop_len

    def __getitem__(self, index):
        # 데이터 로드
        data = torch.load(self.samples[index])
        features = data['data'] # Shape: (MAX_FRAMES, 50, 12) -> (T, V, C)
        action_label = data['label']
        
        filename = os.path.basename(self.samples[index])
        

        # 1. 유효 프레임 길이(Valid Length) 계산
        # 전처리 단계에서 0으로 패딩된 부분을 제외한 실제 길이
        # (T, V, C) -> (T, V*C) -> 각 시간 t에 대해 합이 0보다 큰지 확인
        valid_mask_t = torch.sum(torch.abs(features.view(self.max_frames, -1)), dim=1) > 1e-6
        real_len = int(torch.sum(valid_mask_t).item())
        if real_len == 0: real_len = self.max_frames # 예외 처리

        # --- Data Augmentation (Train Only) ---
        if self.split == 'train':
            # Scaling (크기 변환)
            if np.random.rand() < config.PROB:
                scale = np.random.uniform(0.9, 1.1)
                features[:real_len] *= scale
            
            # Time Masking (일부 구간 0으로)
            if np.random.rand() < config.PROB:
                mask_len = np.random.randint(5, 20)
                # 유효 구간 내에서 마스킹
                if real_len > mask_len:
                    start = np.random.randint(0, real_len - mask_len)
                    features[start:start+mask_len] = 0

        # --- Normalization ---
        # (X - Mean) / Std
        # 패딩된 0값들도 정규화되면서 0이 아닌 값으로 변함 -> 다시 0으로 만들어야 함
        features = (features - self.mean) / (self.std + 1e-8)
        
        # Re-masking: 유효 길이 이후의 값들을 다시 강제로 0으로 초기화
        if real_len < self.max_frames:
            features[real_len:] = 0.0

        # --- View Generation (For Feature Consistency) ---
        
        # View 1: 원본 시퀀스 (전체)
        view1_feat = features.clone()
        view1_len = real_len

        # View 2: 변형된 시퀀스 (Random Crop or Partial)
        if self.split == 'train':
            view2_feat, view2_len = self._random_temporal_crop(features, real_len)
        else:
            # 검증/테스트 시에는 동일하게 설정
            view2_feat = features.clone()
            view2_len = real_len

        # 차원 변환: (T, V, C) -> (C, T, V)
        # PyTorch Conv1d나 일반적인 이미지 포맷은 (Batch, Channel, Height, Width)를 따름
        # 여기서 Height=Time, Width=Vertex 로 간주
        view1_out = view1_feat.permute(2, 0, 1) # (12, T, 50)
        view2_out = view2_feat.permute(2, 0, 1) # (12, T, 50)

        return view1_out, view2_out, action_label, view1_len, view2_len
