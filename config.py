# config.py

import torch


# >> NTU RGB+D 60 데이터셋의 .np 파일들이 있는 디렉토리 경로
# >> 이 경로에는 preprocess_ntu_data.py를 실행하고 생성된 .np 파일들이 있는 경로로 지정해야 한다.
DATASET_PATH = '../nturgbd_processed_12D_Norm/'

# >> 학습된 모델 가중치(체크포인트)를 저장할 디렉토리
SAVE_DIR = 'checkpoints/'

# >> 재현성을 위해 시드 설정
SEED = 42

# >> 데이터 로더 설정
MAX_FRAMES = 200  # 시퀀스의 최대 길이
BATCH_SIZE = 64   # 배치 크기
NUM_WORKERS = 6   # 데이터를 불러올 때 사용할 CPU 프로세서 수 
PIN_MEMORY = True # GPU 사용 시 데이터 전송 속도를 높이기 위한 설정

# >> 모델 하이퍼파라미터
NUM_JOINTS = 50  # 관절 수
NUM_COORDS = 12
NUM_CLASSES = 60 # 행동 클래스 수 (NTU RGB+D 60)
PROB = 0.8
NUM_SUBJECTS = 40
NUM_CAMERAS = 3

# >> 학습 하이퍼파라미터
EPOCHS = 40               # 총 학습 에폭
LEARNING_RATE = 0.000465   
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' # 학습 장치
WARMUP_EPOCHS = 7         # 학습 초기에 학습률을 서서히 증가시키는 웜업 에폭 수 
GRAD_CLIP_NORM = 1.0       # 그레이디언트 폭발을 막기 위한 클리핑 최대 L2 Norm 값 
ADAMW_WEIGHT_DECAY = 0.0153  # AdamW weight decay , L2 정규화의 강도 설정
PATIENCE = 10              # 조기종료 변수
LABEL_SMOOTHING = 0.104     # Loss Function CrossEntropy의 label smoothing
DROPOUT = 0.4
ETA_MIN = 1e-6

# >> 모델 구조
HIDDEN_DIM = 128
SPATIAL_DEPTH = 4
TEMPORAL_DEPTH = 4
WINDOW_SIZE = 20
