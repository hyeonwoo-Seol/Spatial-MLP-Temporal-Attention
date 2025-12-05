import torch
import numpy as np
import config
from model import ST_GRL_Model # [수정] 모델 변경

def measure_inference_speed():
    # ------------------------------------------------------------------------
    # 1. 환경 설정 (Setup)
    # ------------------------------------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    if device.type == 'cpu':
        print("Warning: CPU 측정은 정확한 FPS 비교가 어렵습니다. GPU 사용을 권장합니다.")

    # [수정] 모델 로드 (Evaluation 모드 필수)
    # 실제 학습시 사용했던 aux_classes 개수 등 파라미터를 맞춰주는 것이 좋습니다.
    model = ST_GRL_Model(num_aux_classes=40).to(device)
    model.eval()

    # ------------------------------------------------------------------------
    # 2. 더미 데이터 준비 (Batch Size = 1)
    # ------------------------------------------------------------------------
    # [수정] 현재 모델(ST_GRL)은 단일 입력을 받습니다.
    # Shape: (Batch, Channels, Frames, Joints) -> (1, 12, 100, 50)
    dummy_input = torch.randn(1, config.NUM_COORDS, config.MAX_FRAMES, config.NUM_JOINTS).to(device)

    # ------------------------------------------------------------------------
    # 3. 웜업 (Warm-up)
    # ------------------------------------------------------------------------
    print("warming up...", end=" ")
    with torch.no_grad():
        for _ in range(50):
            # [수정] 단일 입력 전달
            _ = model(dummy_input)
    print("Done.")

    # ------------------------------------------------------------------------
    # 4. 실제 측정 (Measurement) - torch.cuda.Event 사용
    # ------------------------------------------------------------------------
    repetitions = 1000
    timings = []

    # GPU 타이머 이벤트 생성 (GPU일 때만 유효, CPU면 time.time 사용 필요하나 여기선 GPU 기준 작성)
    if device.type == 'cuda':
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
    
    print(f"Measuring latency over {repetitions} runs...")
    
    with torch.no_grad():
        for _ in range(repetitions):
            if device.type == 'cuda':
                starter.record()
                _ = model(dummy_input) # [수정] 추론 실행
                ender.record()
                torch.cuda.synchronize() # 대기
                curr_time = starter.elapsed_time(ender) # ms 단위
                timings.append(curr_time)
            else:
                # CPU Fallback (정밀도는 떨어짐)
                import time
                start_t = time.time()
                _ = model(dummy_input)
                end_t = time.time()
                timings.append((end_t - start_t) * 1000) # s -> ms

    # ------------------------------------------------------------------------
    # 5. 결과 계산 (Metrics)
    # ------------------------------------------------------------------------
    mean_latency = np.mean(timings)
    std_latency = np.std(timings)
    
    fps = 1000 / mean_latency

    print("\n" + "="*40)
    print(f" Model Speed Benchmark (Batch Size=1)")
    print(f" Model: ST_GRL_Model")
    print("="*40)
    print(f" Latency : {mean_latency:.4f} ms ± {std_latency:.2f}")
    print(f" FPS     : {fps:.2f} (samples/sec)")
    print("="*40)

if __name__ == "__main__":
    measure_inference_speed()
