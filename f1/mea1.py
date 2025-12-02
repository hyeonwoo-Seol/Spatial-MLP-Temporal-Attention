import torch
import time
import numpy as np
import config
from model import ST_GRL_Model

# THOP 라이브러리 및 커스텀 핸들러 설정
try:
    from thop import profile
    # thop은 기본적으로 nn.MultiheadAttention의 내부 연산을 무시합니다.
    # 따라서 공정한 비교를 위해 MHA의 연산량을 강제로 계산하는 핸들러를 정의해야 합니다.
    THOP_AVAILABLE = True
except ImportError:
    print("[Warning] 'thop' library not found. Please install it via 'pip install thop'")
    THOP_AVAILABLE = False

def count_multihead_attention(m, x, y):
    """
    nn.MultiheadAttention 모듈에 대한 정밀 FLOPs 계산 핸들러
    이 핸들러가 없으면 예전 모델(TemporalFactorizedBlock)의 FLOPs가 0에 가깝게 나옵니다.
    """
    # 입력 x[0]: (Batch, Seq_Len, Embed_Dim) 또는 (Seq_Len, Batch, Embed_Dim)
    input_tensor = x[0]
    
    # 배치 차원 확인 (batch_first=True/False)
    if m.batch_first:
        batch_size = input_tensor.shape[0]
        seq_len = input_tensor.shape[1]
    else:
        seq_len = input_tensor.shape[0]
        batch_size = input_tensor.shape[1]
        
    embed_dim = m.embed_dim
    num_heads = m.num_heads
    
    # 1. Linear Projections (Q, K, V): 3 * (B * Seq * Dim^2)
    linear_ops = 3 * batch_size * seq_len * embed_dim * embed_dim
    
    # 2. Attention Score (Q @ K.T): B * Heads * Seq^2 * Head_Dim
    head_dim = embed_dim // num_heads
    attn_score_ops = batch_size * num_heads * seq_len * seq_len * head_dim
    
    # 3. Weighted Sum (Score @ V): B * Heads * Seq^2 * Head_Dim
    weighted_sum_ops = batch_size * num_heads * seq_len * seq_len * head_dim
    
    # 4. Output Projection: B * Seq * Dim^2
    output_ops = batch_size * seq_len * embed_dim * embed_dim
    
    total_ops = linear_ops + attn_score_ops + weighted_sum_ops + output_ops
    m.total_ops += torch.DoubleTensor([int(total_ops)])

def measure_model_performance():
    # ----------------------------------------------------------------------
    # 1. 설정
    # ----------------------------------------------------------------------
    TEST_BATCH_SIZE = 1
    TEST_FRAMES = config.MAX_FRAMES 
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[Environment] Device: {device}")
    print(f"[Settings] Batch: {TEST_BATCH_SIZE}, Frames: {TEST_FRAMES}, Joints: {config.NUM_JOINTS}")

    # ----------------------------------------------------------------------
    # 2. 모델 로드
    # ----------------------------------------------------------------------
    print("\n[Loading Model]...")
    try:
        model = ST_GRL_Model(
            num_joints=config.NUM_JOINTS,
            num_coords=config.NUM_COORDS,
            num_classes=config.NUM_CLASSES,
            hidden_dim=config.HIDDEN_DIM,
            window_size=config.WINDOW_SIZE,
            dropout=0.0
        ).to(device)
        model.eval()
    except Exception as e:
        print(f"\n[Error] Model Init Failed: {e}")
        return

    dummy_input = torch.randn(TEST_BATCH_SIZE, config.NUM_COORDS, TEST_FRAMES, config.NUM_JOINTS).to(device)

    # Sanity Check
    try:
        with torch.no_grad():
            _ = model(dummy_input)
        print(" -> Forward pass successful.")
    except RuntimeError as e:
        print(f"\n[CRITICAL ERROR] Model Forward Failed: {e}")
        return

    # ----------------------------------------------------------------------
    # 3. FLOPs (연산량) 정밀 측정
    # ----------------------------------------------------------------------
    if THOP_AVAILABLE:
        print("\n[Profiling FLOPs with Custom Handlers...]")
        
        # 커스텀 핸들러 등록: nn.MultiheadAttention을 만났을 때 정확히 계산
        custom_ops = {
            torch.nn.MultiheadAttention: count_multihead_attention
        }
        
        try:
            macs, params = profile(model, inputs=(dummy_input, ), custom_ops=custom_ops, verbose=False)
            flops = macs * 2 
            print(f"\n[2] Computational Cost")
            print(f"    - FLOPs : {flops / 1e9:.3f} G")
            print(f"    - Params: {params / 1e6:.2f} M")
            
        except Exception as e:
            print(f"    - Failed to measure FLOPs: {e}")
    else:
        print("\n[2] Computational Cost")
        print("    - Skipped (thop not installed)")

    # ----------------------------------------------------------------------
    # 4. 실제 속도 (FPS)
    # ----------------------------------------------------------------------
    print(f"\n[3] Latency & FPS Measurement (Warmup + 100 runs)")
    
    # Warmup
    with torch.no_grad():
        for _ in range(10): _ = model(dummy_input)
    
    repetitions = 100
    timings = []
    
    if torch.cuda.is_available():
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    
    with torch.no_grad():
        for _ in range(repetitions):
            if torch.cuda.is_available():
                starter.record()
                _ = model(dummy_input)
                ender.record()
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
            else:
                st = time.time()
                _ = model(dummy_input)
                et = time.time()
                curr_time = (et - st) * 1000
            timings.append(curr_time)

    avg_latency = np.mean(timings)
    fps = 1000 / avg_latency
    
    print(f"    - Avg Latency: {avg_latency:.4f} ms")
    print(f"    - Throughput : {fps:.2f} FPS")
    print("="*50)

if __name__ == "__main__":
    measure_model_performance()
