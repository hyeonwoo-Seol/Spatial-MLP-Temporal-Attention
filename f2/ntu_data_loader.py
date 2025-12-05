import os
import numpy as np
import torch
from torch.utils.data import Dataset
import config

# Protocol Definitions
# NTU RGB+D Benchmark Protocols
TRAINING_SUBJECTS = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
TRAINING_CAMERAS = [2, 3]

class NTURGBDDataset(Dataset):
    def __init__(self, data_path, split='train', max_frames=100, protocol='xsub'):
        self.data_path = data_path
        self.split = split
        self.max_frames = max_frames
        self.protocol = protocol
        self.training_subjects = TRAINING_SUBJECTS
        self.training_cameras = TRAINING_CAMERAS
        
        # 샘플 리스트 로드
        self.samples = []
        self._load_data_path()

        # 통계치 로드
        base_dir = os.path.dirname(data_path.rstrip('/'))
        
        if self.protocol == 'xsub':
            stats_filename = 'stats_xsub.npz'
        elif self.protocol == 'xview':
            stats_filename = 'stats_xview.npz'
        else:
            stats_filename = 'stats_xsub.npz' 
            
        stats_path = os.path.join(base_dir, stats_filename)
        
        if os.path.exists(stats_path):
            stats = np.load(stats_path)
            self.mean = torch.from_numpy(stats['mean'].flatten()).float()
            self.std = torch.from_numpy(stats['std'].flatten()).float()
            print(f"[{split.upper()}] Normalization stats loaded from {stats_filename} (Protocol: {protocol}).")
        else:
            print(f"[Warning] Stats not found at {stats_path}. Using identity normalization.")
            self.mean = torch.zeros(config.NUM_COORDS) 
            self.std = torch.ones(config.NUM_COORDS)

    def _load_data_path(self):
        if not os.path.exists(self.data_path):
            print(f"Error: Data path {self.data_path} does not exist.")
            return

        filenames = sorted(os.listdir(self.data_path))
        for filename in filenames:
            if not filename.endswith('.pt'): continue
            
            # 파일명 파싱 (예: S001C001P001R001A001.pt)
            try:
                sid = int(filename[9:12]) # Subject ID
                cid = int(filename[5:8])  # Camera ID
            except ValueError:
                continue
            
            # Protocol에 따른 구분 (Train vs Test)
            if self.protocol == 'xsub':
                is_train_sample = sid in self.training_subjects
            elif self.protocol == 'xview':
                is_train_sample = cid in self.training_cameras
            else:
                raise ValueError(f"Unknown protocol: {self.protocol}")
            
            # Split 로직 단순화 (Target Unlabeled 제거)
            if self.split == 'train':
                # 학습 데이터 (Source Domain)
                if is_train_sample:
                    self.samples.append(os.path.join(self.data_path, filename))
            
            elif self.split == 'val':
                # 평가 데이터 (Target Domain / Validation Set)
                if not is_train_sample:
                    self.samples.append(os.path.join(self.data_path, filename))
            
            

    def __len__(self):
        return len(self.samples)

    def _get_augmented_view(self, features, valid_len):
        # 원본 데이터 보호를 위해 복제
        feat = features.clone()
        
        # 1. Random Rotation (3D Y-axis)
        if np.random.rand() < config.PROB:
            angle = np.random.uniform(-15, 15) * np.pi / 180.0
            c, s = np.cos(angle), np.sin(angle)
            rot_mat = torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=torch.float32)
            
            reshaped = feat.view(self.max_frames, config.NUM_JOINTS, 4, 3)
            rotated = torch.matmul(reshaped, rot_mat.T)
            feat = rotated.view(self.max_frames, config.NUM_JOINTS, 12)

        # 2. Random Scaling
        if np.random.rand() < config.PROB:
            scale = np.random.uniform(0.9, 1.1)
            feat[:valid_len] *= scale
            
        # 3. Time Masking (Random Zeroing)
        if np.random.rand() < config.PROB:
            mask_len = np.random.randint(5, 20)
            if valid_len > mask_len:
                start = np.random.randint(0, valid_len - mask_len)
                feat[start : start + mask_len] = 0.0
                
        # 4. Random Temporal Crop
        if np.random.rand() < config.PROB:
            crop_ratio = np.random.uniform(0.8, 1.0)
            new_len = int(valid_len * crop_ratio)
            if new_len > 10:
                start = np.random.randint(0, valid_len - new_len)
                cropped = feat[start : start + new_len, :, :]
                new_feat = torch.zeros_like(feat)
                new_feat[:new_len, :, :] = cropped
                feat = new_feat
                valid_len = new_len

        return feat, valid_len

    def __getitem__(self, index):
        # 1. 파일 로드
        data_path = self.samples[index]
        data = torch.load(data_path)
        
        raw_features = data['data'] # Shape: (T, V, C)
        action_label = data['label']
        
        # Max Frame 길이 맞추기
        if raw_features.shape[0] > self.max_frames:
            raw_features = raw_features[:self.max_frames]
        elif raw_features.shape[0] < self.max_frames:
            pad_len = self.max_frames - raw_features.shape[0]
            padding = torch.zeros((pad_len, config.NUM_JOINTS, config.NUM_COORDS))
            raw_features = torch.cat([raw_features, padding], dim=0)

        # 유효 길이 계산
        valid_mask = torch.sum(torch.abs(raw_features.view(self.max_frames, -1)), dim=1) > 1e-5
        real_len = int(torch.sum(valid_mask).item())
        if real_len == 0: real_len = self.max_frames

        
        if self.split == 'train':
            # 학습 시: 데이터 증강 적용 (1개의 View만 생성)
            features, feat_len = self._get_augmented_view(raw_features, real_len)
        else:
            # 검증 시: 원본 그대로 사용
            features = raw_features.clone()
            feat_len = real_len

        # ----------------------------------------------------------------------
        # Normalization (Z-score)
        # ----------------------------------------------------------------------
        features = (features - self.mean) / (self.std + 1e-8)

        # Zero-Padding 복구
        if feat_len < self.max_frames:
            features[feat_len:] = 0.0

        # ----------------------------------------------------------------------
        # Final Format: (C, T, V)
        # ----------------------------------------------------------------------
        features = features.permute(2, 0, 1) # (12, T, 50)

        
        return features, action_label
