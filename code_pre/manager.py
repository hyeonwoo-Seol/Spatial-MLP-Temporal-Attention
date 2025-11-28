# manager.py

import optuna
import argparse
import config # config.py의 기본값을 참조하기 위해 임포트 (필수는 아님)
import sys

def main():
    parser = argparse.ArgumentParser(description="Optuna Study Manager (Ask-and-Tell)")
    
    # --- 공통 인자 ---
    parser.add_argument('--study-name', type=str, default="slowfast_tuning",
                        help="Name for the Optuna study DB file (e.g., 'study.db')")
    
    subparsers = parser.add_subparsers(dest='command', required=True,
                                       help="Command to execute: 'ask' or 'tell'")

    # --- 1. "ask" 커맨드 파서 ---
    # 'ask' 명령: Optuna에게 다음 하이퍼파라미터를 요청
    ask_parser = subparsers.add_parser('ask', help="Ask Optuna for new trial parameters.")
    
    # --- 2. "tell" 커맨드 파서 ---
    # 'tell' 명령: 완료된 트라이얼의 결과를 Optuna에 보고
    tell_parser = subparsers.add_parser('tell', help="Tell Optuna the result of a completed trial.")
    tell_parser.add_argument('--trial-number', type=int, required=True,
                             help="The trial number that was executed (e.g., 5)")
    tell_parser.add_argument('--value', type=float, required=True, 
                             help="The final best validation accuracy achieved (e.g., 0.915)")
    tell_parser.add_argument('--state', type=str, default='complete', choices=['complete', 'fail'],
                             help="Set the trial state ('complete' or 'fail'). Default: 'complete'")

    args = parser.parse_args()

    # --- 스터디(DB) 로드 또는 생성 ---
    # SQLite DB 파일 이름을 지정하여 스터디 기록을 영구 저장합니다.
    storage_name = f"sqlite:///{args.study_name}.db"
    
    try:
        study = optuna.create_study(
            study_name=args.study_name,
            storage=storage_name,
            direction='maximize',   # 우리는 정확도(Accuracy)를 '최대화'하는 것이 목표
            load_if_exists=True # DB 파일이 존재하면 새로 만들지 않고 불러옵니다.
        )
    except Exception as e:
        print(f"Error connecting to Optuna storage: {e}")
        print("Please ensure you have necessary permissions and 'optuna' is installed correctly.")
        sys.exit(1)


    # --- 1. ASK 로직 실행 ---
    if args.command == 'ask':
        print(f"Asking study '{args.study_name}' for next parameters...")
        
        # 이전에 'RUNNING' 상태로 중단된 트라이얼이 있는지 확인
        running_trials = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.RUNNING,))
        if running_trials:
            print(f"Warning: Found {len(running_trials)} running trial(s).")
            print("This might happen if 'manager.py --tell' was not called after a trial.")
            print(f"Consider manually setting their state (e.g., 'fail') or deleting them.")
            print(f"Running trial numbers: {[t.number for t in running_trials]}")

        # Optuna에게 하이퍼파라미터 제안을 요청 (TPE 알고리즘 기반)
        # 이 정의는 train.py의 suggest_... 와 동일해야 합니다.
        trial = study.ask(
            {
                "LEARNING_RATE": optuna.distributions.FloatDistribution(4e-4, 6e-4, log=True),
                "DROPOUT": optuna.distributions.FloatDistribution(0.2, 0.5),
                "ADVERSARIAL_ALPHA": optuna.distributions.FloatDistribution(0.05, 0.35),
                "PROB": optuna.distributions.FloatDistribution(0.5, 0.8),
                "ADAMW_WEIGHT_DECAY": optuna.distributions.FloatDistribution(0.001, 0.1, log=True),
                "LABEL_SMOOTHING": optuna.distributions.FloatDistribution(0.0, 0.2)
            }
        )
        
        params = trial.params
        print("\n" + "="*50)
        print(f"--- Trial {trial.number} Parameters ---")
        
        # train.py를 실행할 명령어를 자동으로 생성해줍니다.
        # 사용자는 이 명령어를 복사/붙여넣기만 하면 됩니다.
        cmd_parts = [
            "python train.py",
            f"--study-name {args.study_name}",
            "--protocol xsub",
            "--scheduler cosine_decay", # (필요시 변경)
            f"--trial-number {trial.number}",
            f"--lr {params['LEARNING_RATE']}",
            f"--dropout {params['DROPOUT']}",
            f"--alpha {params['ADVERSARIAL_ALPHA']}",
            f"--prob {params['PROB']}",
            f"--weight-decay {params['ADAMW_WEIGHT_DECAY']}",
            f"--smoothing {params['LABEL_SMOOTHING']}"
        ]
        
        # 터미널에 복사하기 편하도록 한 줄로 출력
        run_command = " \\\n    ".join(cmd_parts)
        
        print("\n[Action Required] Copy and paste the following command to run the trial:\n")
        print(run_command)
        print("="*50)

    # --- 2. TELL 로직 실행 ---
    elif args.command == 'tell':
        print(f"Reporting result for Trial {args.trial_number}...")
        
        trial_state = optuna.trial.TrialState.COMPLETE if args.state == 'complete' else optuna.trial.TrialState.FAIL

        try:
            # Optuna에게 "이 트라이얼은 이 값으로 완료되었다"고 보고
            study.tell(
                args.trial_number, 
                args.value if trial_state == optuna.trial.TrialState.COMPLETE else None, 
                trial_state
            )
            
            print(f"Successfully reported Trial {args.trial_number} as {args.state.upper()}.")
            if args.state == 'complete':
                print(f"  Value: {args.value:.4f}")
            
            print(f"\nRun 'python manager.py --study-name {args.study_name} ask' to get the next trial.")
            
        except Exception as e:
            print(f"\nError reporting trial: {e}")
            print(f"  > Did you already report Trial {args.trial_number}?")
            print(f"  > Was the trial number correct?")
            print(f"  > Check study status with: optuna-dashboard sqlite:///{args.study_name}.db")

if __name__ == '__main__':
    main()
