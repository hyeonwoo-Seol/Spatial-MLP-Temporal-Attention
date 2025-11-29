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
            self.mean = torch.zeros(config.NUM_COORDS) 
            self.std = torch.ones(config.NUM_COORDS)

    def _load_data_path(self):
        if not os.path.exists(self.data_path):
            print(f"Error: Data path {self.data_path} does not exist.")
            return

        filenames = sorted(os.listdir(self.data_path))
        for filename in filenames:
            if not filename.endswith('.pt'): continue
            
            # 파일명에서 정보 추출 (예: S001C001P001R001A001)
            try:
                sid = int(filename[9:12]) # Subject ID
                cid = int(filename[5:8])  # Camera ID
            except ValueError:
                continue
            
            # Protocol Check (Source vs Target 구분 기준)
            if self.protocol == 'xsub':
                is_source_domain = sid in self.training_subjects
            elif self.protocol == 'xview':
                is_source_domain = cid in TRAINING_CAMERAS
            else:
                raise ValueError(f"Unknown protocol: {self.protocol}")
            
            # Split 분기 로직 수정
            if self.split == 'train':
                # Source Domain (Labeled) - 학습용
                if is_source_domain:
                    self.samples.append(os.path.join(self.data_path, filename))
            
            elif self.split == 'val':
                # Target Domain (Labeled) - 평가용 (Source 데이터가 섞이면 안됨)
                if not is_source_domain:
                    self.samples.append(os.path.join(self.data_path, filename))
            
            elif self.split == 'target_train':
                # Target Domain (Unlabeled) - 학습용 (GRL 적응용)
                # 'val'과 데이터는 같지만, 목적이 다름 (Augmentation 적용 대상)
                if not is_source_domain:
                    self.samples.append(os.path.join(self.data_path, filename))

    def __len__(self):
        return len(self.samples)


    # >> Feature Consistency를 위한 랜덤 시간 크롭(View 2)
    def _random_temporal_crop(self, feature, valid_length):
        T, V, C = feature.shape
        
        if valid_length < 10: 
            return feature.clone(), valid_length
        
        crop_ratio = np.random.uniform(0.7, 0.9)
        crop_len = int(valid_length * crop_ratio)
        
        max_start = valid_length - crop_len
        start = np.random.randint(0, max_start + 1)
        
        cropped = feature[start : start + crop_len, :, :] 
        
        new_feat = torch.zeros_like(feature)
        new_feat[:crop_len, :, :] = cropped
        
        return new_feat, crop_len

    def __getitem__(self, index):
        # 데이터 로드
        data = torch.load(self.samples[index])
        features = data['data'] # Shape: (T, V, C)
        if features.shape[0] > self.max_frames:
            features = features[:self.max_frames]
        action_label = data['label']
        
        # 1. 유효 프레임 길이(Valid Length) 계산
        valid_mask_t = torch.sum(torch.abs(features.view(self.max_frames, -1)), dim=1) > 1e-6
        real_len = int(torch.sum(valid_mask_t).item())
        if real_len == 0: real_len = self.max_frames

        # --- Data Augmentation ---
        # 'train' (Source) 뿐만 아니라 'target_train' (Unlabeled Target) 도 학습 시에는 증강 수행
        if self.split in ['train', 'target_train']:
            # Scaling
            if np.random.rand() < config.PROB:
                scale = np.random.uniform(0.9, 1.1)
                features[:real_len] *= scale
            
            # Time Masking
            if np.random.rand() < config.PROB:
                mask_len = np.random.randint(5, 20)
                if real_len > mask_len:
                    start = np.random.randint(0, real_len - mask_len)
                    features[start:start+mask_len] = 0

        # --- Normalization ---
        features = (features - self.mean) / (self.std + 1e-8)
        
        # Re-masking
        if real_len < self.max_frames:
            features[real_len:] = 0.0

        # --- View Generation ---
        
        # View 1: 원본 (Source/Target 모두 사용)
        view1_feat = features.clone()
        view1_len = real_len

        # View 2: 변형 (Source는 Consistency Loss용, Target도 필요 시 사용 가능)
        # 학습 모드('train', 'target_train')이면 Random Crop 적용
        if self.split in ['train', 'target_train']:
            view2_feat, view2_len = self._random_temporal_crop(features, real_len)
        else:
            view2_feat = features.clone()
            view2_len = real_len

        # 차원 변환: (T, V, C) -> (C, T, V)
        view1_out = view1_feat.permute(2, 0, 1) # (12, T, 50)
        view2_out = view2_feat.permute(2, 0, 1) # (12, T, 50)

        return view1_out, view2_out, action_label, view1_len, view2_len
