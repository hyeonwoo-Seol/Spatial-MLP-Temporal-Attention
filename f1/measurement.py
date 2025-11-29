import torch
import time
import numpy as np
import config
from model import ST_GRL_Model
from thop import profile

def measure_model_performance():
    # ----------------------------------------------------------------------
    # 1. 설정 (Edge Device 환경 가정)
    # ----------------------------------------------------------------------
    # 엣지 디바이스에서는 보통 배치 사이즈를 1로 두고 실시간 처리를 합니다.
    TEST_BATCH_SIZE = 1
    
    # 로봇 환경에 맞는 프레임 수 설정 (학습 때 80으로 줄였다면 여기서도 80으로 테스트)
    TEST_FRAMES = config.MAX_FRAMES 
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[Environment] Device: {device}")
    print(f"[Settings] Batch: {TEST_BATCH_SIZE}, Frames: {TEST_FRAMES}, Joints: {config.NUM_JOINTS}")

    # ----------------------------------------------------------------------
    # 2. 모델 로드
    # ----------------------------------------------------------------------
    model = ST_GRL_Model(
        num_joints=config.NUM_JOINTS,
        num_coords=config.NUM_COORDS,
        num_classes=config.NUM_CLASSES,
        hidden_dim=config.HIDDEN_DIM,
        spatial_depth=config.SPATIAL_DEPTH,
        temporal_depth=config.TEMPORAL_DEPTH,
        window_size=config.WINDOW_SIZE,
        dropout=0.0 # 측정 시 Dropout은 끔
    ).to(device)
    model.eval()

    # 더미 입력 데이터 생성 (N, C, T, V)
    dummy_input = torch.randn(
        TEST_BATCH_SIZE, 
        config.NUM_COORDS, 
        TEST_FRAMES, 
        config.NUM_JOINTS
    ).to(device)

    # ----------------------------------------------------------------------
    # 3. Parameters (모델 크기) 측정
    # ----------------------------------------------------------------------
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[1] Model Size")
    print(f"    - Parameters: {num_params / 1e6:.2f} M (Million)")
    
    # ----------------------------------------------------------------------
    # 4. FLOPs (연산량) 측정
    # ----------------------------------------------------------------------
    # thop은 입력 튜플을 받습니다.
    macs, params = profile(model, inputs=(dummy_input, ), verbose=False)
    flops = macs * 2 # 통상적으로 1 MAC = 2 FLOPs로 계산
    
    print(f"\n[2] Computational Cost")
    print(f"    - MACs (Multiply-Accumulates): {macs / 1e9:.3f} G (Giga)")
    print(f"    - FLOPs (Floating Point Ops) : {flops / 1e9:.3f} G (Giga)")
    print(f"    * Note: 젯슨 나노급은 보통 1~5 GFLOPs 이내 권장")

    # ----------------------------------------------------------------------
    # 5. Latency (지연 시간) 및 FPS 측정
    # ----------------------------------------------------------------------
    print(f"\n[3] Latency & FPS Measurement (Warmup + 100 runs)")
    
    # GPU Warmup (초기 캐싱으로 인한 지연 제거)
    with torch.no_grad():
        for _ in range(20):
            _ = model(dummy_input)
    
    # 시간 측정
    repetitions = 100
    timings = np.zeros((repetitions, 1))
    
    # 동기화 객체 (GPU 시간 측정 정확도를 위함)
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            
            # GPU 연산이 끝날 때까지 대기
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender) # 밀리초(ms) 단위 반환
            timings[rep] = curr_time

    avg_latency_ms = np.mean(timings)
    std_latency_ms = np.std(timings)
    fps = 1000 / avg_latency_ms
    
    print(f"    - Avg Latency: {avg_latency_ms:.4f} ms (+/- {std_latency_ms:.4f})")
    print(f"    - Throughput : {fps:.2f} FPS")
    
    # ----------------------------------------------------------------------
    # 6. Peak Memory (최대 메모리 사용량)
    # ----------------------------------------------------------------------
    if torch.cuda.is_available():
        max_mem = torch.cuda.max_memory_allocated() / 1024 / 1024 # MB
        print(f"\n[4] Peak Memory Usage (Inference)")
        print(f"    - VRAM: {max_mem:.2f} MB")

    # ----------------------------------------------------------------------
    # 결과 해석 가이드
    # ----------------------------------------------------------------------
    print("\n" + "="*50)
    print(" [ Robot / Edge Device Suitability Check ]")
    print("="*50)
    
    # 1. 파라미터 기준
    if num_params < 5e6: msg_p = "Excellent (Very Light)"
    elif num_params < 10e6: msg_p = "Good (Fit for Mobile)"
    else: msg_p = "Heavy (Might need optimization)"
    print(f" 1. Size     : {msg_p}")
    
    # 2. FPS 기준 (일반적인 로봇 제어 루프 기준)
    if fps >= 30: msg_f = "Perfect (Real-time)"
    elif fps >= 15: msg_f = "Good (Usable)"
    else: msg_f = "Slow (Might feel laggy)"
    print(f" 2. Speed    : {msg_f}")
    
    print("="*50)

if __name__ == "__main__":
    measure_model_performance()
