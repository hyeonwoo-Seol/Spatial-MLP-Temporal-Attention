# >> ntu_data_loader.py

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import config

# >> Protocol Definitions
TRAINING_SUBJECTS = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38] # 학습용으로 사용될 사람 ID 
TRAINING_CAMERAS = [2, 3] # 학습용으로 사용될 카메라 번호 

class NTURGBDDataset(Dataset):
    def __init__(self, data_path, split='train', max_frames=config.MAX_FRAMES, protocol='xsub'):
        self.data_path = data_path # 클래스 생성 시 경로 
        self.split = split # 분할 모드 
        self.max_frames = max_frames # 최대 프레임 수 
        self.protocol = protocol # 프로토콜 
        self.training_subjects = TRAINING_SUBJECTS
        self.training_cameras = TRAINING_CAMERAS
        
        self.samples = []
        self._load_data_path() # 실제 데이터 파일 경로들을 리스트에 채운다

        # >> 통계치 로드
        base_dir = os.path.dirname(data_path.rstrip('/')) # datapath의 끝의 /를 제거한다. 통계파일이 상위 디렉터리에 있기 때문이다.
        
        if self.protocol == 'xsub':
            stats_filename = 'stats_xsub.npz'
        elif self.protocol == 'xview':
            stats_filename = 'stats_xview.npz'
        else:
            stats_filename = 'stats_xsub.npz' 
            
        stats_path = os.path.join(base_dir, stats_filename)
        
        if os.path.exists(stats_path): # 파일이 존재하면 mean과 std를 텐서로 변환해 메모리에 올린다 
            stats = np.load(stats_path) # npz 형식의 통계 파일을 불러온다
            self.mean = torch.from_numpy(stats['mean'].flatten()).float()
            self.std = torch.from_numpy(stats['std'].flatten()).float()
            print(f"[{split.upper()}] Normalization stats loaded from {stats_filename} (Protocol: {protocol}).")
        else:
            print(f"[Warning] Stats not found at {stats_path}. Using identity normalization.")
            self.mean = torch.zeros(config.NUM_COORDS) # 평균을 0으로 설정한다 
            self.std = torch.ones(config.NUM_COORDS) # 표준편차를 0으로 설정한다 

    # >> 데이터 경로 불러오는 함수 
    def _load_data_path(self):
        if not os.path.exists(self.data_path):
            print(f"Error: Data path {self.data_path} does not exist.")
            return

        filenames = sorted(os.listdir(self.data_path)) # 디렉토리 내의 모든 파일 목록을 가져와서 정렬한다
        for filename in filenames:
            if not filename.endswith('.pt'): continue
            
            try:
                sid = int(filename[9:12]) # 9 ~ 11번째 글자를 가져와 Subject ID로 추출한다 
                cid = int(filename[5:8])
            except ValueError:
                continue
            
            # >> 프로토콜이 xsub, xview이면 학습을 진행하도록 true를 반환한다  
            if self.protocol == 'xsub':
                is_train_sample = sid in self.training_subjects
            elif self.protocol == 'xview':
                is_train_sample = cid in self.training_cameras
            else:
                raise ValueError(f"Unknown protocol: {self.protocol}")

            # >> 학습 모드에 따라 파일 경로를 불러온다 
            if self.split == 'train':
                if is_train_sample:
                    self.samples.append(os.path.join(self.data_path, filename))
            elif self.split == 'val':
                if not is_train_sample:
                    self.samples.append(os.path.join(self.data_path, filename))

    # >> 데이터 길이를 변환한다 
    def __len__(self):
        return len(self.samples)

    # >> Bone Length AdaIN (Inter-instance Augmentation) 구현
    def _bone_length_adain(self, feat):
        """
        현재 샘플(feat)의 뼈 벡터 길이를 다른 랜덤 샘플의 뼈 벡터 길이로 교체합니다.
        feat shape: (T, V, C=12)
        preprocess_ntu_data.py에 따르면 채널 0~3이 Bone Vector입니다.
        """
        # 1. 랜덤하게 다른 샘플 선택 (Target)
        rand_idx = np.random.randint(0, len(self.samples))
        target_path = self.samples[rand_idx]
        
        try:
            target_data = torch.load(target_path)
            target_feat = target_data['data'] # (MAX_FRAMES, Joint, C)
        except:
            return feat # 로드 실패 시 원본 반환

        # 2. Bone Vector 추출 (채널 0:3)
        # feat: (T, V, 12)
        source_bone_vec = feat[:, :, :3]
        target_bone_vec = target_feat[:, :, :3]

        # 타겟 데이터의 프레임 수가 다를 경우 보간하거나 잘라냄 (현재는 모두 MAX_FRAMES로 맞춰져 있다고 가정)
        # 만약 차이가 난다면 안전하게 shape 맞춤
        if target_bone_vec.shape[0] != source_bone_vec.shape[0]:
            # 간단하게 타겟을 소스 길이에 맞게 자르거나 반복
            tgt_len = target_bone_vec.shape[0]
            src_len = source_bone_vec.shape[0]
            if tgt_len > src_len:
                target_bone_vec = target_bone_vec[:src_len]
            else:
                # 짧으면 반복
                pad_len = src_len - tgt_len
                target_bone_vec = torch.cat([target_bone_vec, target_bone_vec[:pad_len]], dim=0)

        # 3. 길이(Length) 계산
        # eps를 더해 0으로 나누기 방지
        source_len = torch.norm(source_bone_vec, dim=-1, keepdim=True) + 1e-8
        target_len = torch.norm(target_bone_vec, dim=-1, keepdim=True) + 1e-8

        # 4. AdaIN 적용 (Swap)
        # 방향(Direction)은 원본 유지, 크기(Length)는 타겟 적용
        # Source Direction = Source Vector / Source Length
        # New Vector = Source Direction * Target Length
        source_direction = source_bone_vec / source_len
        new_bone_vec = source_direction * target_len

        # 5. 특징 맵 업데이트
        # 원본 feat을 복사하지 않고 직접 수정 (호출하는 곳에서 clone 처리됨)
        feat[:, :, :3] = new_bone_vec
        
        return feat

    # >> 데이터를 증강한다
    def _get_augmented_view(self, features):
        feat = features.clone() # 원본 데이터를 보존하기 위해 deep copy를 한다 
        T = feat.shape[0] # 텐서의 첫 번째 차원인 프레임 수 정보를 T에 저장한다 

        # >> [NEW] 0. Bone Length AdaIN (Inter-instance Augmentation)
        # >> 다른 사람의 신체 비율(뼈 길이)을 현재 동작에 입힘
        if np.random.rand() < config.PROB:
            feat = self._bone_length_adain(feat)

        # >> 1. Random Rotation (3D Y-axis)
        if np.random.rand() < config.PROB:
            angle = np.random.uniform(-15, 15) * np.pi / 180.0 # -15 ~ 15도 사이의 값을 라디안으로 변환하여 랜덤 각도를 생성한다 
            c, s = np.cos(angle), np.sin(angle) # 해당 각도의 코사인과 사인 값을 계산한다 
            rot_mat = torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=torch.float32) # y축을 기준으로 하는 3x3 회전 행렬을 텐서로 생성한다 
            
            reshaped = feat.view(T, config.NUM_JOINTS, 4, 3) # 12차원 특징 벡터를 회전 연산이 가능하도록 (시간, 관절, 그룹 수, 3차원 좌표) 형태로 차원 변경한다. 우리는 특징 그룹 수가 4이다. Group 1 ~ 4 
            rotated = torch.matmul(reshaped, rot_mat.T) # 회전 행렬을 곱해서 좌표로 회전시킨다 
            feat = rotated.view(T, config.NUM_JOINTS, 12) # 연산이 끝난 텐서를 다시 원래의 12차원 형태로 되돌린다 

        # >> 2. Random Scaling
        if np.random.rand() < config.PROB:
            scale = np.random.uniform(0.9, 1.1) # 0.9 ~ 1.1 사이의 랜덤한 실수 값을 생성한다 
            feat *= scale # 특징 벡터 전체에 스케일 값을 곱해서 크기를 조절한다 
            
        # >> 3. Time Masking (일부 구간 0으로)
        if np.random.rand() < config.PROB:
            mask_len = np.random.randint(5, 15) # 마스킹할 프레임의 길이를 5 ~ 15 사이로 랜덤하게 고른다 
            if T > mask_len: # 전체 프레임 길이가 마스크 길이보다 큰 경우에만 수행 
                start = np.random.randint(0, T - mask_len) # 마스킹을 시작할 지점을 랜덤하게 선택 
                feat[start : start + mask_len] = 0.0 # 선택된 구간의 모든 데이터를 0으로 만들기 

        return feat

    # >> 인덱스를 받아 실제 데이터를 반환하는 메서드
    def __getitem__(self, index):
        data_path = self.samples[index] # 주어진 인덱스에 해당하는 파일 경로 가져오기 
        filename = os.path.basename(data_path) # 경로에서 파일명만 추출하기 
        
        data = torch.load(data_path)
        
        # >> 전처리된 데이터는 이미 (config.MAX_FRAMES, Joint, C) 형태이다.
        features = data['data'] # 딕셔너리에서 data 키에 해당하는 특징 텐서를 꺼내기 
        action_label = data['label'] # 딕셔너리에서 label에 해당하는 행동 라벨 꺼내기 

        # >> 1을 빼서 0부터 시작하는 보조 라벨을 만든다 
        if self.protocol == 'xsub':
            sid = int(filename[9:12])
            aux_label = sid - 1 
        elif self.protocol == 'xview':
            cid = int(filename[5:8])
            aux_label = cid - 1 
        else:
            aux_label = 0 

        # >> Train 시에만 Augmentation 적용
        if self.split == 'train':
            features = self._get_augmented_view(features)
        
        # >> Normalization
        features = (features - self.mean) / (self.std + 1e-8)

        # (Time, Joint, Channel) -> (Channel, Time, Joint)
        features = features.permute(2, 0, 1) 

        return features, action_label, aux_label
