# ESS 배터리 수명 예측

ESS 비용의 30~40%를 차지하는 배터리 교체를 계획하려면, 용량이 떨어지기 전에 수명을 알아야 한다.
**셀별 초기 100사이클 데이터로 총 수명(cycle_life)을 예측**한다.

## 프로젝트 개요
- 데이터셋 : MIT-Stanford Battery Dataset (Severson et al., Nature Energy 2019)
- 학습 데이터 : Batch 1 (2017-05-12), 36셀
- 평가 데이터 : Batch 2 (2018-02-20), 39셀 / Batch 3 (2018-04-12), 40셀
- 태스크 : **Regression (Cycle Life 예측)**
- Target : cycle_life = 방전 용량이 0.88Ah(공칭 1.1Ah의 80%)에 도달한 사이클 수, 학습은 log10(cycle_life)

## 파일 구조
```
├── data/
│   ├── README.md              # 원본 데이터 준비 방법
│   └── features.csv           # 셀별 피처 (115셀)
├── notebooks/
│   ├── 01_EDA.ipynb           # DAY1 EDA (Q1~Q5)
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── preprocess.py          # .mat 추출 · 제외 셀 · 정제
│   ├── features.py            # ΔQ(V) 등 피처 계산
│   └── train.py               # 학습 · 평가 · 결과 저장
├── results/
│   ├── model_performance.csv
│   ├── extrapolation.csv      # 학습 범위 안 / 밖 오차
│   ├── predictions.csv
│   ├── pred_vs_true.png
│   └── relation_extended.png
├── requirements.txt
└── README.md
```

## 환경 설정
```bash
git clone https://github.com/<계정>/ess-battery-project
cd ess-battery-project
pip install -r requirements.txt

python src/preprocess.py --data_dir data/raw   # 원본이 있을 때만 (data/README.md 참고)
python src/features.py                          # data/features.csv 생성
python src/train.py                             # results/ 생성
```

## EDA

분석 대상 : 139셀 중 115셀 (원저자 로딩 코드 기준 제외 · 수명 없음 셀 제외)

- **Cycle Life 분포**
  - 전체 392 ~ 1,935 사이클 (약 5배). B1 534 ~ 1,074 / B2 392 ~ 1,186 / B3 541 ~ 1,935
  - 550 미만 셀 : B1 1개, B2 30개, B3 1개
  - 핵심 발견 : 평가 배치가 학습 배치의 수명 범위를 양쪽으로 벗어나 외삽이 필요하고, B1만으로는 분류를 학습할 수 없다

- **열화 곡선 분석**
  - 수명 40%까지 용량 감소가 거의 없고 이후 급가속
  - knee는 수명의 약 73% (B1 71 · B2 70 · B3 76%)
  - 핵심 발견 : 초기 100사이클의 용량만으로는 장 · 단수명을 구분할 수 없다

- **ΔQ(V) 곡선 분석**
  - ΔQ(V) = Q₁₀₀(V) − Q₁₀(V), 단수명 셀은 3.0V 부근에서 ΔQ가 깊음
  - log10 Var(ΔQ)와 log 수명 : B1 r −0.84, B2 −0.92, B3 −0.80
  - 핵심 발견 : 용량이 줄기 전에도 방전 곡선 모양 변화로 수명을 설명할 수 있다

- **충전 속도(C-rate)와 수명의 관계**
  - B1에서만 C-rate가 높을수록 수명이 짧음 (ρ −0.43). B2 · B3는 모두 4.8C(10분 충전)인데 수명 392 ~ 1,935
  - 핵심 발견 : 수명 차이는 충전 조건보다 셀 상태가 더 크게 설명한다

- **배치 효과**
  - 초기 용량 · 초기 저항은 전체 ρ −0.55지만 B1 안에서는 0.16 · 0.18
  - 핵심 발견 : 배치를 합친 상관은 측정 조건 차이가 만든 가짜 상관일 수 있어, 피처는 B1 기준으로 선택한다

## Modeling

### 피처 엔지니어링 전략
| 세트 | 피처 | 근거 |
|---|---|---|
| A (기본) | log10 Var(ΔQ) | B1 최상위 피처, 세 배치 모두 같은 방향 |
| B (비교) | A + 초기 용량 기울기 (cycle 10~100) | B1 ρ 0.50 |
| C (대체) | log10 \|min ΔQ\| | A와 중복 (r 0.996) |
| D (대조군) | A + B + 초기 용량 · 초기 저항 · 평균 온도 · 충전시간 | EDA에서 제외한 피처를 넣으면 실제로 나빠지는지 확인 |

- 모든 피처는 cycle 100 이전 데이터만 사용 (knee 등 이후 정보는 누수로 제외)
- 결측 대체 · 표준화는 Pipeline 안에서 수행 (CV fold마다 학습 fold로만 적합)

### 모델 선택 및 근거
- 후보 모델 : Dummy(평균, 기준선), ElasticNet, Ridge, Random Forest, GBM
- 최종 모델 : **ElasticNet (피처 세트 A)**
- 선택 이유 : log 공간에서 ΔQ 분산과 수명이 선형 관계이고, 선형 모델은 학습 범위 밖으로 관계를 연장할 수 있다. 트리 모델은 학습한 수명 범위 밖을 예측하지 못한다.
- 튜닝 : B1 내부 충전 정책 단위 GroupKFold (5-fold). B2 · B3는 튜닝에 사용하지 않음

## 성능 결과

피처 세트 A, MAPE(%) / MAE / RMSE (사이클)

| 모델 | B1 CV | B2 (1차 평가) | B3 (2차 평가) |
|---|---|---|---|
| Dummy (평균) | 16.9 / 131 / 153 | 59.1 / 286 / 303 | 22.7 / 277 / 399 |
| **ElasticNet** | **8.5** / 69 / 86 | **28.6** / 144 / 163 | **11.8** / 133 / 209 |
| Ridge | 8.5 / 69 / 86 | 28.6 / 144 / 163 | 11.7 / 133 / 209 |
| Random Forest | 8.8 / 70 / 90 | 30.2 / 150 / 157 | 15.5 / 190 / 302 |
| GBM | 8.9 / 73 / 98 | 30.8 / 151 / 160 | 14.2 / 175 / 286 |

- 원논문 테스트 오차 : 9.1%
- **GAP (B2 MAPE − 논문) = 28.6 − 9.1 = +19.5%p**
- 참고 : B1 CV − 논문 = −0.6%p, B3 − 논문 = +2.7%p

피처 세트 비교 (ElasticNet, MAPE %)

| 세트 | B1 CV | B2 | B3 |
|---|---|---|---|
| A 기본 | 8.5 | **28.6** | 11.8 |
| B + 기울기 | 8.9 | 31.5 | 11.6 |
| C 대체 | 9.0 | 30.6 | 11.8 |
| D 제외 피처 포함 | **6.8** | 43.6 | 12.8 |

외삽 점검 (피처 세트 A, MAPE %) : 학습 수명 범위 534 ~ 1,074 기준

| 구간 | ElasticNet | Random Forest | GBM |
|---|---|---|---|
| B2 범위 안 (7셀) | 21.0 | 13.1 | 12.7 |
| B2 범위 아래 (30셀) | 31.9 | 35.0 | 36.1 |
| B3 범위 안 (27셀) | 10.4 | 10.0 | 9.2 |
| B3 범위 위 (13셀) | **14.6** | 26.9 | 24.5 |

## 오류 분석
- 모델이 가장 크게 틀린 셀의 공통점 :
  - ElasticNet은 B2를 평균 +139사이클 **과대예측** (B3는 −28사이클로 거의 치우침 없음)
  - 오차가 큰 셀은 대부분 B2의 단수명 셀 (예 : b2c6 실제 393 → 예측 648, b2c18 449 → 732)
  - `relation_extended.png` : B2 점들이 B1에서 배운 직선보다 **아래쪽**에 모여 있음
  - Random Forest · GBM은 예측값이 약 600 ~ 1,000 사이에 갇힘
- 원인 가설 및 개선 방향 : (작성)

## ESS 도메인 해석
- 이 모델을 실제 BESS에 적용한다면 어떤 의사결정에 활용 가능한가? (작성)
- 어떤 한계가 있으며, 실 배포를 위해 추가로 필요한 것은 무엇인가? (작성)

## 참고문헌
- Severson et al. (2019). Data-driven prediction of battery cycle life before capacity degradation. *Nature Energy*, 4, 383–391.

## 팀 구성
- 홍동완 (울산_1반) : EDA, 피처 엔지니어링, 모델 개발, 성능 평가
