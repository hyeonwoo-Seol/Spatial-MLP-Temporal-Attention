# >> optuna_search.py


import optuna
import argparse
import torch
import train
import config
import sys
import os

# >> train.py의 args와 호환되는 설정을 담을 클래스이다.
class TrainArgs:
    def __init__(self, trial, base_args):
        # >> 1. 탐색할 하이퍼파라미터 범위를 설정한다 (Trial 객체 사용).
        self.lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
        self.dropout = trial.suggest_float("dropout", 0.1, 0.5)
        self.prob = trial.suggest_float("prob", 0.3, 0.8) 
        self.weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-3, log=True)
        self.smoothing = trial.suggest_float("smoothing", 0.0, 0.2)
        
        # >> 2. 고정 파라미터는 커맨드라인에서 받은 값을 유지한다.
        self.protocol = base_args.protocol
        self.study_name = base_args.study_name
        self.trial_number = trial.number
        
        # >> 3. train.py 호환성을 위한 필수 파라미터를 추가한다.
        self.resume = None
        self.auto_resume = False
        
        # >> GRL 관련 파라미터는 train.py에서 제거되었으므로 여기서도 제외한다.

def objective(trial, base_args):
    # >> Trial별 하이퍼파라미터를 생성한다.
    args = TrainArgs(trial, base_args)
    
    # >> 학습을 실행하고 결과를 반환한다.
    try:
        # >> train.py의 run_training 함수를 호출하여 best_acc를 얻는다.
        best_acc = train.run_training(args)
        
        return best_acc
    
    except RuntimeError as e:
        # >> OOM(Out of Memory) 발생 시 해당 Trial을 가지치기(Pruning) 처리한다.
        if "CUDA out of memory" in str(e):
            print(f"[Trial {trial.number}] CUDA OOM. Pruning trial.")
            torch.cuda.empty_cache()
            # >> Optuna에게 이 Trial이 실패했음을 알리고 다음으로 넘어간다.
            raise optuna.exceptions.TrialPruned()
        else:
            raise e
    except KeyboardInterrupt:
        # >> 사용자가 중단할 경우 DB 저장 상태를 보존하며 종료한다.
        print("\n[Optuna] Interrupted by user. Exiting...")
        sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna Hyperparameter Search")
    parser.add_argument('--protocol', type=str, default='xsub', choices=['xsub', 'xview'])
    parser.add_argument('--study-name', type=str, default='ntu_optimization')
    parser.add_argument('--n-trials', type=int, default=50, help="Number of trials")
    # >> DB 파일 저장 경로를 설정한다.
    parser.add_argument('--storage', type=str, default='sqlite:///optuna_study.db', help="Database storage URL")
    
    base_args = parser.parse_args()

    # >> DB 디렉터리를 생성한다 (없을 경우 에러 방지).
    if base_args.storage.startswith('sqlite:///'):
        db_path = base_args.storage.replace('sqlite:///', '')
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)

    # >> Study를 생성하거나 로드한다.
    study = optuna.create_study(
        study_name=base_args.study_name,
        storage=base_args.storage,
        direction="maximize", # >> 정확도는 높을수록 좋다.
        load_if_exists=True,  # >> 이미 존재하는 Study가 있으면 로드하여 이어한다.
        pruner=optuna.pruners.MedianPruner() # >> 성능이 낮은 Trial을 조기 종료한다.
    )

    print(f"Start Optimization: {base_args.study_name}")
    print(f"Storage: {base_args.storage}")
    print(f"Remaining Trials: {base_args.n_trials - len(study.trials)} (Total requested: {base_args.n_trials})")
    
    # >> 최적화를 수행한다.
    try:
        study.optimize(lambda trial: objective(trial, base_args), n_trials=base_args.n_trials)
    except KeyboardInterrupt:
        print("\n[Optuna] Optimization interrupted. Saving current progress...")
        
    # >> 하나라도 완료된 Trial이 있을 경우 결과를 출력한다.
    if len(study.trials) > 0:
        print("\n[Best Trial]")
        try:
            trial = study.best_trial
            print(f"  Value (Acc): {trial.value}")
            print("  Params: ")
            for key, value in trial.params.items():
                print(f"    {key}: {value}")

            # >> 베스트 파라미터를 텍스트 파일로 저장한다.
            with open(f"{base_args.study_name}_best_params.txt", "w") as f:
                f.write(f"Best Accuracy: {trial.value}\n")
                f.write(str(trial.params))
        except ValueError:
            print("No completed trials yet.")
    else:
        print("No trials completed.")
