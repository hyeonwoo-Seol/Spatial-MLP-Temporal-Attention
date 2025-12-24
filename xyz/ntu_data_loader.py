import os
import numpy as np
import torch
from torch.utils.data import Dataset
import config

# Protocol Definitions
TRAINING_SUBJECTS = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
TRAINING_CAMERAS = [2, 3]

class NTURGBDDataset(Dataset):
    def __init__(self, data_path, split='train', max_frames=config.MAX_FRAMES, protocol='xsub'):
        self.data_path = data_path
        self.split = split
        self.max_frames = max_frames
        self.protocol = protocol
        self.training_subjects = TRAINING_SUBJECTS
        self.training_cameras = TRAINING_CAMERAS
        
        self.samples = []
        self._load_data_path()

        # 통계치 로드 (Raw XYZ용 파일)
        base_dir = os.path.dirname(data_path.rstrip('/'))
        
        if self.protocol == 'xsub':
            stats_filename = 'stats_xsub_xyz.npz'
        elif self.protocol == 'xview':
            stats_filename = 'stats_xview_xyz.npz'
        else:
            stats_filename = 'stats_xsub_xyz.npz' 
            
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
            
            try:
                sid = int(filename[9:12]) 
                cid = int(filename[5:8])  
            except ValueError:
                continue
            
            if self.protocol == 'xsub':
                is_train_sample = sid in self.training_subjects
            elif self.protocol == 'xview':
                is_train_sample = cid in self.training_cameras
            else:
                raise ValueError(f"Unknown protocol: {self.protocol}")
            
            if self.split == 'train':
                if is_train_sample:
                    self.samples.append(os.path.join(self.data_path, filename))
            elif self.split == 'val':
                if not is_train_sample:
                    self.samples.append(os.path.join(self.data_path, filename))

    def __len__(self):
        return len(self.samples)

    def _get_augmented_view(self, features):
        """
        [3D XYZ Augmentation Strategy]
        Input features: (T, V, 3)
        """
        feat = features.clone()
        T, V, C = feat.shape
        
        # 1. Random Rotation (3D Y-axis)
        # 3차원 좌표이므로 단순 행렬 곱으로 처리 가능
        if np.random.rand() < config.PROB:
            angle = np.random.uniform(-15, 15) * np.pi / 180.0
            c, s = np.cos(angle), np.sin(angle)
            # Y축 기준 회전 행렬
            rot_mat = torch.tensor([
                [c, 0, s], 
                [0, 1, 0], 
                [-s, 0, c]
            ], dtype=torch.float32)
            
            # (T, V, 3) @ (3, 3).T -> (T, V, 3)
            feat = torch.matmul(feat, rot_mat.T)

        # 2. Random Scaling
        if np.random.rand() < config.PROB:
            scale = np.random.uniform(0.9, 1.1)
            feat *= scale
            
        # 3. Time Masking (일부 구간 0으로)
        if np.random.rand() < config.PROB:
            mask_len = np.random.randint(5, 15)
            if T > mask_len:
                start = np.random.randint(0, T - mask_len)
                feat[start : start + mask_len] = 0.0

        return feat

    def __getitem__(self, index):
        data_path = self.samples[index]
        filename = os.path.basename(data_path)
        
        try:
            data = torch.load(data_path)
        except Exception as e:
            print(f"Error loading {data_path}: {e}")
            # 에러 발생 시 0으로 채운 더미 데이터 반환 (학습 중단 방지)
            dummy = torch.zeros((config.NUM_COORDS, self.max_frames, config.NUM_JOINTS))
            return dummy, 0, 0

        # 전처리된 데이터: (MAX_FRAMES, V, 3)
        features = data['data'] 
        action_label = data['label']
        
        if self.protocol == 'xsub':
            sid = int(filename[9:12])
            aux_label = sid - 1 
        elif self.protocol == 'xview':
            cid = int(filename[5:8])
            aux_label = cid - 1 
        else:
            aux_label = 0 

        # Train 시에만 Augmentation 적용
        if self.split == 'train':
            features = self._get_augmented_view(features)
        
        # Normalization (Standard Scaling)
        # (T, V, C) - (C,) -> Broadcasting (T, V, C)
        features = (features - self.mean) / (self.std + 1e-8)

        # Permute for Model Input
        # (T, V, C) -> (C, T, V)
        features = features.permute(2, 0, 1) 

        return features, action_label, aux_label
