import torch
import time
import numpy as np
from ptflops import get_model_complexity_info
from model import ST_GRL_Model  # 변경: 현재 모델 import
import config

def input_constructor(input_res):
    """
    ptflops가 모델에 입력을 넣을 때 사용하는 생성자 함수입니다.
    ST_GRL_Model은 단일 입력 (N, C, T, V)를 받습니다.
    """
    batch_size = 1
    # config에 정의된 차원 사용
    # 입력 형태: (Batch, Channels, Frames, Joints)
    input_shape = (batch_size, config.NUM_COORDS, config.MAX_FRAMES, config.NUM_JOINTS)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dummy_input = torch.randn(input_shape).to(device)
    
    # 모델의 forward(*args)에 들어갈 인자들을 튜플 또는 딕셔너리로 반환
    # forward(self, x)
    return {"x": dummy_input}

def measure_efficiency():
    # 1. 모델 준비
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ST_GRL_Model 인스턴스화 (기본 config 값 사용)
    # num_aux_classes는 측정 시 중요하지 않으므로 기본값(0) 또는 임의값 사용
    model = ST_GRL_Model().to(device)
    model.eval()

    # ----------------------------------------------------------------------
    # 2. FLOPs 및 Parameters 측정 (ptflops 사용)
    # ----------------------------------------------------------------------
    print("\n[1] Measuring FLOPs & Params with ptflops...")
    
    try:
        macs, params = get_model_complexity_info(
            model, 
            (config.NUM_COORDS, config.MAX_FRAMES, config.NUM_JOINTS), # input_res (사용되지 않으나 형식상 전달)
            as_strings=False, 
            print_per_layer_stat=False, 
            verbose=False,
            input_constructor=input_constructor
        )
        
        # 일반적으로 1 MAC = 2 FLOPs로 계산
        flops = macs * 2

        print(f" - Parameters : {params / 1e6:.2f} M")
        print(f" - MACs       : {macs / 1e9:.2f} G")
        print(f" - FLOPs      : {flops / 1e9:.2f} G (Calculated as MACs * 2)")
        
    except Exception as e:
        print(f"Error measuring FLOPs: {e}")
        print("Skipping FLOPs measurement...")
        params = sum(p.numel() for p in model.parameters()) # Fallback for params
        flops = 0.0

    # ----------------------------------------------------------------------
    # 3. Latency 및 FPS 측정
    # ----------------------------------------------------------------------
    print("\n[2] Measuring Latency & FPS...")
    
    # 더미 입력 생성 (Batch Size = 1)
    dummy_input = torch.randn(1, config.NUM_COORDS, config.MAX_FRAMES, config.NUM_JOINTS).to(device)

    # GPU Warm-up (초기 오버헤드 제거)
    warmup_iters = 50
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(dummy_input)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # 실제 측정
    test_iters = 1000
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(test_iters):
            _ = model(dummy_input)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_latency = total_time / test_iters # 초 단위
    fps = 1.0 / avg_latency

    print(f" - Batch Size : 1")
    print(f" - Latency    : {avg_latency * 1000:.2f} ms")
    print(f" - FPS        : {fps:.2f} frames/sec")

    print("\n" + "="*40)
    print(f"Summary for Report:")
    print(f"Params: {params / 1e6:.2f} M")
    if flops > 0:
        print(f"FLOPs : {flops / 1e9:.2f} G")
    print(f"Time  : {avg_latency * 1000:.2f} ms")
    print("="*40)

if __name__ == "__main__":
    measure_efficiency()
