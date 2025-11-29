# 실행 방법

## 데이터 준비하기

python preprocess_ntu_data.py

## 1번 실행하기

python train.py --protocol xview --auto-resume --trial-number 1

## Optuna로 실행하기

python optuna_search.py --study-name ntuexperiment_01 --n-trials 50 --protocol xsub
python optuna_search.py --study-name ntuexperiment_01 --n-trials 50 --protocol xview

## 학습 로그 남기기

python train.py --protocol xsub | tee log.txt
python train.py --protocol xview | tee log.txt