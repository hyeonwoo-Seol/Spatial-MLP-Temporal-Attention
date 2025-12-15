import optuna
import argparse
import torch
import train  # train.py 모듈 임포트
import config
import sys

# train.py의 args와 호환되는 설정을 담을 클래스
class TrainArgs:
    def __init__(self, trial, base_args):
        # 1. 탐색할 하이퍼파라미터 범위 설정 (Trial 객체 사용)
        self.lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
        self.dropout = trial.suggest_float("dropout", 0.1, 0.5)
        self.alpha = trial.suggest_float("alpha", 0.1, 1.0)
        self.prob = trial.suggest_float("prob", 0.3, 0.8) # Augmentation 확률
        self.weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-3, log=True)
        self.smoothing = trial.suggest_float("smoothing", 0.0, 0.2)
        

        # 2. 고정 파라미터 (커맨드라인에서 받은 값 유지)
        self.protocol = base_args.protocol
        self.study_name = base_args.study_name
        self.trial_number = trial.number
        
        # 3. train.py 호환성을 위한 필수 파라미터 추가 (기본값 설정)
        # train.py의 run_training 함수가 args.resume과 args.auto_resume을 참조하므로 필수입니다.
        self.resume = None
        self.auto_resume = False
        
        # 기타 필요한 설정들 (train.py의 main 참조)
        # train.py 내부 로직에서 args.xxx 형태로 접근하는 모든 변수가 여기 있어야 함

def objective(trial, base_args):
    # Trial별 하이퍼파라미터 생성
    args = TrainArgs(trial, base_args)
    
    # 학습 실행 및 결과 반환
    try:
        # train.py의 run_training 함수 호출
        # run_training은 내부적으로 best_acc를 반환함
        best_acc = train.run_training(args)
        
        return best_acc
    
    except RuntimeError as e:
        # OOM(Out of Memory) 발생 시 해당 Trial 가지치기(Pruning) 처리
        if "CUDA out of memory" in str(e):
            print(f"[Trial {trial.number}] CUDA OOM. Pruning trial.")
            torch.cuda.empty_cache()
            raise optuna.exceptions.TrialPruned()
        else:
            raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna Hyperparameter Search")
    parser.add_argument('--protocol', type=str, default='xsub', choices=['xsub', 'xview'])
    parser.add_argument('--study-name', type=str, default='ntu_optimization')
    parser.add_argument('--n-trials', type=int, default=50, help="Number of trials")
    parser.add_argument('--storage', type=str, default='sqlite:///optuna_study.db', help="Database storage URL")
    
    base_args = parser.parse_args()

    # Study 생성 또는 로드
    study = optuna.create_study(
        study_name=base_args.study_name,
        storage=base_args.storage,
        direction="maximize", # 정확도(Acc)는 높을수록 좋음
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner() # 성능이 낮은 Trial 조기 종료
    )

    print(f"Start Optimization: {base_args.study_name}")
    
    # 최적화 수행
    # lambda 함수를 사용하여 base_args를 objective 함수에 전달
    study.optimize(lambda trial: objective(trial, base_args), n_trials=base_args.n_trials)

    # 결과 출력
    print("\n[Best Trial]")
    trial = study.best_trial
    print(f"  Value (Acc): {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # 베스트 파라미터 저장 (옵션)
    with open(f"{base_args.study_name}_best_params.txt", "w") as f:
        f.write(f"Best Accuracy: {trial.value}\n")
        f.write(str(trial.params))
