# 실행 방법

## 데이터 준비하기

python preprocess_ntu_data.py

## 1번 실행하기

python train.py --protocol xview --study-name XView1 --trial-number 1


python train.py --protocol xsub --study-name XSub1 --trial-number 1 --auto-resume


## Optuna로 실행하기

python optuna_search.py --study-name ntuexperiment_01 --n-trials 50 --protocol xsub
python optuna_search.py --study-name ntuexperiment_01 --n-trials 50 --protocol xview

## 학습 로그 남기기

python train.py --protocol xsub | tee log.txt
python train.py --protocol xview | tee log.txt

## optuna

ptuna-dashboard sqlite:///optuna_study.db

## 검증

python evaluate.py --checkpoint checkpoints/XView1/trial_1/best_model.pth.tar --protocol xview