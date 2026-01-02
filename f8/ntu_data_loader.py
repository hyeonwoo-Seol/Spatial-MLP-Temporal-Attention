# >> ntu_data_loader.py


import os
import numpy as np
import torch
from torch.utils.data import Dataset
import config

# >> Protocol Definitions
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

        # >> 데이터 정규화를 위한 평균과 표준편차를 로드한다.
        base_dir = os.path.dirname(data_path.rstrip('/'))
        
        if self.protocol == 'xsub':
            stats_filename = 'stats_xsub_SKF.npz'
        elif self.protocol == 'xview':
            stats_filename = 'stats_xview_SKF.npz'
        else:
            stats_filename = 'stats_xsub_SKF.npz' 
            
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
        # >> 데이터 경로가 존재하는지 확인한다.
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
            
            # >> 프로토콜(X-Sub, X-View)에 따라 학습 샘플 여부를 결정한다.
            if self.protocol == 'xsub':
                is_train_sample = sid in self.training_subjects
            elif self.protocol == 'xview':
                is_train_sample = cid in self.training_cameras
            else:
                raise ValueError(f"Unknown protocol: {self.protocol}")
            
            # >> Split(train/val)에 맞는 파일만 리스트에 추가한다.
            if self.split == 'train':
                if is_train_sample:
                    self.samples.append(os.path.join(self.data_path, filename))
            elif self.split == 'val':
                if not is_train_sample:
                    self.samples.append(os.path.join(self.data_path, filename))

    def __len__(self):
        return len(self.samples)

    def _get_augmented_view(self, features):
        # >> 학습 시 데이터 증강을 적용한다.
        # >> 시간 차원은 이미 압축되었으므로 Temporal Crop은 생략한다.
        feat = features.clone()
        T = feat.shape[0]

        # >> 1. Random Rotation (Y축 회전)을 적용한다.
        if np.random.rand() < config.PROB:
            angle = np.random.uniform(-15, 15) * np.pi / 180.0
            c, s = np.cos(angle), np.sin(angle)
            rot_mat = torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=torch.float32)
            
            reshaped = feat.view(T, config.NUM_JOINTS, 4, 3) 
            rotated = torch.matmul(reshaped, rot_mat.T)
            feat = rotated.view(T, config.NUM_JOINTS, 12)

        # >> 2. Random Scaling을 적용하여 크기를 조절한다.
        if np.random.rand() < config.PROB:
            scale = np.random.uniform(0.9, 1.1)
            feat *= scale
            
        # >> 3. Time Masking을 통해 일부 구간을 0으로 만든다.
        if np.random.rand() < config.PROB:
            mask_len = np.random.randint(5, 15)
            if T > mask_len:
                start = np.random.randint(0, T - mask_len)
                feat[start : start + mask_len] = 0.0

        return feat

    def __getitem__(self, index):
        data_path = self.samples[index]
        filename = os.path.basename(data_path)
        
        data = torch.load(data_path)
        
        # >> 전처리된 데이터(config.MAX_FRAMES, V, C)와 라벨을 로드한다.
        features = data['data'] 
        action_label = data['label']
        
        # >> 보조 라벨(Subject ID 또는 Camera ID)을 설정한다.
        if self.protocol == 'xsub':
            sid = int(filename[9:12])
            aux_label = sid - 1 
        elif self.protocol == 'xview':
            cid = int(filename[5:8])
            aux_label = cid - 1 
        else:
            aux_label = 0 

        # >> 학습 모드일 경우에만 데이터 증강을 수행한다.
        if self.split == 'train':
            features = self._get_augmented_view(features)
        
        # >> 로드한 통계치로 정규화를 수행한다.
        features = (features - self.mean) / (self.std + 1e-8)

        # >> 모델 입력 형태에 맞춰 차원을 변환한다: (T, V, C) -> (C, T, V)
        features = features.permute(2, 0, 1) 

        return features, action_label, aux_label
