import optuna
import argparse
import torch
import train
import config
import sys
import os

# train.py의 args와 호환되는 설정을 담을 클래스
class TrainArgs:
    def __init__(self, trial, base_args):
        # 1. 탐색할 하이퍼파라미터 범위 설정 (Trial 객체 사용)
        self.lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
        self.dropout = trial.suggest_float("dropout", 0.1, 0.5)
        self.prob = trial.suggest_float("prob", 0.3, 0.8) # Augmentation 확률
        self.weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-3, log=True)
        self.smoothing = trial.suggest_float("smoothing", 0.0, 0.2)
        
        # 2. 고정 파라미터 (커맨드라인에서 받은 값 유지)
        self.protocol = base_args.protocol
        self.study_name = base_args.study_name
        self.trial_number = trial.number
        
        # 3. train.py 호환성을 위한 필수 파라미터 추가
        self.resume = None
        self.auto_resume = False
        
        # GRL 관련 파라미터가 train.py에서 요구되지 않으므로 alpha는 제외함

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
            # Optuna에게 이 Trial이 실패했음을 알리고 다음으로 넘어감
            raise optuna.exceptions.TrialPruned()
        else:
            raise e
    except KeyboardInterrupt:
        # 사용자가 Ctrl+C를 누르면 즉시 종료 (DB 저장 상태 보존)
        print("\n[Optuna] Interrupted by user. Exiting...")
        sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna Hyperparameter Search")
    parser.add_argument('--protocol', type=str, default='xsub', choices=['xsub', 'xview'])
    parser.add_argument('--study-name', type=str, default='ntu_optimization')
    parser.add_argument('--n-trials', type=int, default=50, help="Number of trials")
    # DB 파일 저장 경로 설정
    parser.add_argument('--storage', type=str, default='sqlite:///optuna_study.db', help="Database storage URL")
    
    base_args = parser.parse_args()

    # DB 디렉터리 생성 (없을 경우 에러 방지)
    if base_args.storage.startswith('sqlite:///'):
        db_path = base_args.storage.replace('sqlite:///', '')
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)

    # Study 생성 또는 로드
    study = optuna.create_study(
        study_name=base_args.study_name,
        storage=base_args.storage,
        direction="maximize", # 정확도(Acc)는 높을수록 좋음
        load_if_exists=True,  # 이미 존재하는 Study가 있으면 로드 (Resume 기능의 핵심)
        pruner=optuna.pruners.MedianPruner() # 성능이 낮은 Trial 조기 종료
    )

    print(f"Start Optimization: {base_args.study_name}")
    print(f"Storage: {base_args.storage}")
    print(f"Remaining Trials: {base_args.n_trials - len(study.trials)} (Total requested: {base_args.n_trials})")
    
    # 최적화 수행
    try:
        study.optimize(lambda trial: objective(trial, base_args), n_trials=base_args.n_trials)
    except KeyboardInterrupt:
        print("\n[Optuna] Optimization interrupted. Saving current progress...")
        
    # 결과 출력 (하나라도 완료된 Trial이 있을 경우)
    if len(study.trials) > 0:
        print("\n[Best Trial]")
        try:
            trial = study.best_trial
            print(f"  Value (Acc): {trial.value}")
            print("  Params: ")
            for key, value in trial.params.items():
                print(f"    {key}: {value}")

            # 베스트 파라미터 텍스트 파일 저장
            with open(f"{base_args.study_name}_best_params.txt", "w") as f:
                f.write(f"Best Accuracy: {trial.value}\n")
                f.write(str(trial.params))
        except ValueError:
            print("No completed trials yet.")
    else:
        print("No trials completed.")
