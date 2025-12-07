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
    def __init__(self, data_path, split='train', max_frames=64, protocol='xsub'):
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

    def _get_augmented_view(self, features):
        # [SkateFormer Strategy]
        # 이미 64프레임으로 압축된 Global View이므로, 
        # Temporal Crop을 수행하여 프레임 수를 줄이는 것은 지양합니다.
        # 대신 Spatial Augmentation(Rotation, Scaling)을 적용합니다.
        
        # 원본 데이터 보호를 위해 복제
        feat = features.clone()
        T = feat.shape[0]

        # 1. Random Rotation (3D Y-axis)
        if np.random.rand() < config.PROB:
            angle = np.random.uniform(-15, 15) * np.pi / 180.0
            c, s = np.cos(angle), np.sin(angle)
            rot_mat = torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=torch.float32)
            
            reshaped = feat.view(T, config.NUM_JOINTS, 4, 3) # (T, V, 4, 3)
            rotated = torch.matmul(reshaped, rot_mat.T)
            feat = rotated.view(T, config.NUM_JOINTS, 12)

        # 2. Random Scaling
        if np.random.rand() < config.PROB:
            scale = np.random.uniform(0.9, 1.1)
            feat *= scale
            
        # 3. Time Masking (Random Zeroing) - 길이는 유지하되 일부 구간만 가림
        if np.random.rand() < config.PROB:
            mask_len = np.random.randint(5, 15) # 마스킹 길이도 전체 길이에 맞춰 조정
            if T > mask_len:
                start = np.random.randint(0, T - mask_len)
                feat[start : start + mask_len] = 0.0
                
        # [Deleted] Random Temporal Crop
        # SkateFormer의 Resize 전략은 '전체 동작'을 보는 것이 핵심이므로,
        # 이미 압축된 시퀀스를 또 잘라내는 것은 정보 손실이 큽니다.

        return feat

    def __getitem__(self, index):
        # 1. 파일 로드
        data_path = self.samples[index]
        filename = os.path.basename(data_path)
        
        data = torch.load(data_path)
        
        # Preprocess 단계에서 이미 (MAX_FRAMES, V, C)로 Resize되어 저장됨
        features = data['data'] # Shape: (MAX_FRAMES, V, C)
        action_label = data['label']
        
        # 파일명 형식: S001C001P001R001A001.pt
        # S: Subject (9:12), C: Camera (5:8)
        
        if self.protocol == 'xsub':
            # Subject Classification Task
            sid = int(filename[9:12])
            aux_label = sid - 1 
        elif self.protocol == 'xview':
            # Camera View Classification Task
            cid = int(filename[5:8])
            aux_label = cid - 1 
        else:
            aux_label = 0 

        # 이미 전처리된 데이터는 고정 길이를 보장하므로
        # 패딩이나 자르기 로직(Padding/Slicing)은 제거합니다.
        
        if self.split == 'train':
            # 학습 시: 데이터 증강 적용
            features = self._get_augmented_view(features)
        else:
            # 검증 시: 원본 그대로 사용
            pass

        # ----------------------------------------------------------------------
        # Normalization (Z-score)
        # ----------------------------------------------------------------------
        features = (features - self.mean) / (self.std + 1e-8)

        # ----------------------------------------------------------------------
        # Final Format: (C, T, V)
        # ----------------------------------------------------------------------
        features = features.permute(2, 0, 1) # (12, 64, 50)

        return features, action_label, aux_label
