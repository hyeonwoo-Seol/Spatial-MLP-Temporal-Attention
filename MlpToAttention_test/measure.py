import torch
import time
import numpy as np
from ptflops import get_model_complexity_info
import config
from model import ST_Model

def input_constructor(input_res):
    """
    ptflops가 모델에 입력을 넣을 때 사용하는 생성자 함수입니다.
    ST_Model은 단일 입력 (N, C, T, V)를 받습니다.
    """
    batch_size = 1
    # config에 정의된 차원 사용
    input_shape = (batch_size, config.NUM_COORDS, config.MAX_FRAMES, config.NUM_JOINTS)
    
    device = config.DEVICE
    dummy_input = torch.randn(input_shape).to(device)
    
    # 모델의 forward(*args)에 들어갈 인자들을 딕셔너리로 반환
    return {"x": dummy_input}

def measure_efficiency():
    # 1. 모델 준비
    device = config.DEVICE
    print(f"Using device: {device}")

    # ST_Model 인스턴스화 (num_aux_classes 제거)
    model = ST_Model().to(device)
    model.eval()

    # ----------------------------------------------------------------------
    # 2. FLOPs 및 Parameters 측정 (ptflops 사용)
    # ----------------------------------------------------------------------
    print("\n[1] Measuring FLOPs & Params with ptflops...")
    
    try:
        macs, params = get_model_complexity_info(
            model, 
            (config.NUM_COORDS, config.MAX_FRAMES, config.NUM_JOINTS), 
            as_strings=False, 
            print_per_layer_stat=False, 
            verbose=False,
            input_constructor=input_constructor
        )
        
        # 1 MAC = 2 FLOPs
        flops = macs * 2

        print(f" - Parameters : {params / 1e6:.2f} M")
        print(f" - MACs       : {macs / 1e9:.2f} G")
        print(f" - FLOPs      : {flops / 1e9:.2f} G")
        
    except Exception as e:
        # 모델 구조 호환성 문제 등으로 인한 런타임 에러 시 처리
        print(f"Error measuring FLOPs: {e}")
        print("Skipping FLOPs measurement...")
        params = sum(p.numel() for p in model.parameters()) 
        flops = 0.0

    # ----------------------------------------------------------------------
    # 3. Latency 및 FPS 측정
    # ----------------------------------------------------------------------
    print("\n[2] Measuring Latency & FPS...")
    
    dummy_input = torch.randn(1, config.NUM_COORDS, config.MAX_FRAMES, config.NUM_JOINTS).to(device)

    # GPU Warm-up
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
    avg_latency = total_time / test_iters 
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
