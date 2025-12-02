# config.py

import torch

DATASET_PATH = '../nturgbd_processed_12D_Norm/'


SAVE_DIR = 'checkpoints/'


SEED = 42

MAX_FRAMES = 100  # 시퀀스의 최대 길이
BATCH_SIZE = 32   # 배치 크기
NUM_WORKERS = 6   # 데이터를 불러올 때 사용할 CPU 프로세서 수 
PIN_MEMORY = True # GPU 사용 시 데이터 전송 속도를 높이기 위한 설정

NUM_JOINTS = 50   # 관절 수 (25 joints * 2 persons)
NUM_COORDS = 12   # 입력 차원 (Preprocess에서 생성된 12차원 벡터)
NUM_CLASSES = 60  # 행동 클래스 수 (NTU RGB+D 60)

PROB = 0.5 # 데이터 증강 확률

HIDDEN_DIM = 128
WINDOW_SIZE = 20     # Factorized Attention의 Local Window 크기
DROPOUT = 0.4

EPOCHS = 30               # 총 학습 에폭
LEARNING_RATE = 0.000465   
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

WARMUP_EPOCHS = 7         # Linear Warmup 기간
GRAD_CLIP_NORM = 1.0      # Gradient Clipping Threshold
ADAMW_WEIGHT_DECAY = 0.0153 
LABEL_SMOOTHING = 0.113   

ETA_MIN = 1e-6

ADVERSARIAL_ALPHA = 0.5   # GRL Alpha (train.py argparse default 값으로 사용됨)

