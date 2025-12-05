import torch
from thop import profile
from fvcore.nn import FlopCountAnalysis
from model import ST_GRL_Model  # [수정] 모델 변경
import config

def compare_flops_measurement():
    # 1. 모델 및 데이터 준비
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # [수정] ST_GRL_Model 초기화
    # num_aux_classes는 FLOPs 계산에 영향을 주므로 임의값(예: 40) 설정
    model = ST_GRL_Model(num_aux_classes=40).to(device)
    model.eval()

    # [수정] 입력 데이터 생성 (Batch=1, C=12, T=100, V=50)
    # config.NUM_COORDS=12, config.MAX_FRAMES=100, config.NUM_JOINTS=50
    dummy_input = torch.randn(1, config.NUM_COORDS, config.MAX_FRAMES, config.NUM_JOINTS).to(device)
    
    # thop은 입력을 튜플 형태로 받기를 권장합니다.
    inputs = (dummy_input, )

    print(f"--- FLOPs Measurement Comparison (ST_GRL_Model) ---")

    # ---------------------------------------------------
    # Method 1: thop
    # ---------------------------------------------------
    # [수정] inputs 인자 전달 방식 변경
    macs_thop, params_thop = profile(model, inputs=inputs, verbose=False)
    flops_thop = macs_thop * 2 # MACs -> FLOPs 변환
    
    print(f"[thop]")
    print(f" - Parameters: {params_thop / 1e6:.2f} M")
    print(f" - GFLOPs    : {flops_thop / 1e9:.2f} G")

    # ---------------------------------------------------
    # Method 2: fvcore
    # ---------------------------------------------------
    # [수정] fvcore는 단일 텐서 입력 시 튜플로 감싸지 않아도 되지만, 
    # 모델이 튜플을 반환하더라도 내부적으로 처리 가능합니다.
    flops_analyzer = FlopCountAnalysis(model, dummy_input)
    
    flops_fvcore_raw = flops_analyzer.total()
    
    unsupported = flops_analyzer.unsupported_ops()
    if unsupported:
        print(f" * Warning: fvcore found unsupported ops: {unsupported}")

    # fvcore 결과 변환 (MACs -> FLOPs)
    final_flops_fvcore = flops_fvcore_raw * 2
    
    print(f"[fvcore]")
    print(f" - GFLOPs    : {final_flops_fvcore / 1e9:.2f} G")
    
    # ---------------------------------------------------
    # 비교 결론
    # ---------------------------------------------------
    diff = abs(flops_thop - final_flops_fvcore)
    print(f"\n[Conclusion]")
    print(f"Difference: {diff / 1e9:.4f} G")
    
    if flops_thop > 0:
        error_rate = diff / flops_thop
        if error_rate < 0.05: # 5% 미만 차이
            print(">> 두 도구의 결과가 거의 일치합니다. (thop 신뢰 가능)")
        else:
            print(f">> 차이가 있습니다 (오차율 {error_rate*100:.1f}%).")
            print("   fvcore 경고 메시지(unsupported ops)를 확인해보세요.")

if __name__ == "__main__":
    compare_flops_measurement()
