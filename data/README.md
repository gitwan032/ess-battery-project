# Data

원본 `.mat` 파일은 용량(약 8GB) 때문에 저장소에 포함하지 않습니다.

1. Kaggle "MIT-Stanford Dataset"에서 아래 3개 파일을 받아 `data/raw/`에 둡니다.
   - `2017-05-12_batchdata_updated_struct_errorcorrect.mat` (Batch 1, 학습)
   - `2018-02-20_batchdata_updated_struct_errorcorrect.mat` (Batch 2, 1차 평가)
   - `2018-04-12_batchdata_updated_struct_errorcorrect.mat` (Batch 3, 2차 평가)
   - `2018-04-03_varcharge...` 파일은 다른 논문(Attia 2020) 데이터라 사용하지 않습니다.
2. `python src/preprocess.py --data_dir data/raw` → 필요한 필드만 추출해 `data/cache/*.pkl` 생성
3. `python src/features.py` → `data/features.csv` 생성 (저장소에 포함, 115셀)

참고 : 과제의 Batch 2(2018-02-20)는 원논문 평가셋(2017-06-30)과 다른 파일입니다.
