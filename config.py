# config.py

import torch

# --------------------------------------------------------------------------
# 1. 경로 설정
# --------------------------------------------------------------------------
# >> NTU RGB+D 60 데이터셋의 .np 파일들이 있는 디렉토리 경로
DATASET_PATH = '../nturgbd_processed_12D_Norm/'

# >> 학습된 모델 가중치(체크포인트)를 저장할 디렉토리
SAVE_DIR = 'checkpoints/'



# --------------------------------------------------------------------------
# 2. 데이터 및 로더 설정
# --------------------------------------------------------------------------
# >> 재현성을 위해 시드 설정
SEED = 42

MAX_FRAMES = 200  # 시퀀스의 최대 길이
BATCH_SIZE = 64   # 배치 크기
NUM_WORKERS = 6   # 데이터를 불러올 때 사용할 CPU 프로세서 수 
PIN_MEMORY = True # GPU 사용 시 데이터 전송 속도를 높이기 위한 설정

NUM_JOINTS = 50   # 관절 수 (25 joints * 2 persons)
NUM_COORDS = 12   # 입력 차원 (Preprocess에서 생성된 12차원 벡터)
NUM_CLASSES = 60  # 행동 클래스 수 (NTU RGB+D 60)

# >> 데이터 증강 확률 (Scaling, Time Masking 등)
PROB = 0.8



# --------------------------------------------------------------------------
# 3. 모델 하이퍼파라미터 (Light-weight Transformer)
# --------------------------------------------------------------------------
HIDDEN_DIM = 128
SPATIAL_DEPTH = 4    # Spatial Mixer Block 깊이
TEMPORAL_DEPTH = 4   # Temporal Factorized Block 깊이
WINDOW_SIZE = 20     # Factorized Attention의 Local Window 크기
DROPOUT = 0.4



# --------------------------------------------------------------------------
# 4. 학습 하이퍼파라미터
# --------------------------------------------------------------------------
EPOCHS = 40               # 총 학습 에폭
LEARNING_RATE = 0.000465   
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# >> Optimizer & Scheduler
WARMUP_EPOCHS = 7         # Linear Warmup 기간
GRAD_CLIP_NORM = 1.0      # Gradient Clipping Threshold
ADAMW_WEIGHT_DECAY = 0.0153 
LABEL_SMOOTHING = 0.104   


ETA_MIN = 1e-6



# --------------------------------------------------------------------------
# 5. Domain Adaptation (GRL)
# --------------------------------------------------------------------------
ADVERSARIAL_ALPHA = 1.0   # GRL Alpha (train.py argparse default 값으로 사용됨)

