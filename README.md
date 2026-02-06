# 불균형 데이터 처리 및 리스크 조기 탐지 시스템


[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> 도메인 지식 기반 머신러닝을 활용한 축구 선수 부상 예측 시스템

## Problem Statement

### Background
축구 선수 부상은 팀 성적에 직접적인 영향을 미치며, 프리미어리그 기준 **시즌당 평균 50건 이상**의 부상이 발생합니다. 부상으로 인한 경제적 손실은 **팀당 연간 수십억 원**에 달하며, 선수 개인의 커리어에도 치명적입니다.

### Problem
- **사후 대응의 한계**: 부상 발생 후 치료보다 예방이 더 효과적이나 예측 도구 부재
- **훈련 부하 관리 어려움**: 과도한 훈련량과 부족한 회복시간 균형 찾기 어려움
- **개인별 차이 무시**: 획일적인 훈련 프로그램으로 개인별 위험도 고려 불가
- **데이터 미활용**: GPS, 웨어러블 데이터는 수집되나 예측 모델 부재

### Objective
훈련 부하, 회복 지표, 개인 이력 데이터를 활용해 **3일/7일/14일 이내 부상 위험도를 예측**하는 머신러닝 시스템
- **높은 재현율(Recall)**: 부상 위험 선수를 놓치지 않는 것이 최우선 (F2 Score 최적화)
- **해석 가능성**: Feature Importance로 부상 원인 분석 가능
- **실시간 예측**: 일일 데이터 입력으로 즉시 위험도 산출

### Value Proposition
- **부상 예방**: 고위험 선수 조기 식별로 훈련량 조정 → 부상률 감소
- **경제적 효과**: 핵심 선수 이탈 방지로 **팀 가치 손실 최소화**
- **선수 수명 연장**: 개인 맞춤형 부하 관리로 커리어 보호
- **과학적 의사결정**: 경험 대신 데이터 기반 훈련 계획 수립

### Approach
- **Tim Gabbett ACWR 이론** 기반 피처 엔지니어링 (Acute:Chronic Workload Ratio)
- **SMOTE** 불균형 데이터 처리 + **F2 Score** 최적화
- **앙상블 전략**: 9개 모델 훈련 → Top 5 가중 투표
- **AUC 0.92+** 달성

---

## 주요 성과

- **AUC**: 3일/7일/14일 예측 윈도우에서 0.92+ 달성
- **과적합 제어**: Train/Test 성능 차이 < 0.03
- **피처 효율성**: 단 20개 피처로 최적 성능 달성
- **앙상블 전략**: F2 Score 기반 상위 5개 모델 가중 투표

---

## 빠른 시작

```bash
# 1. 환경 설정
conda create -n soccermon python=3.9
conda activate soccermon
pip install -r requirements.txt

# 2. 데이터 전처리
python scripts/preprocessing/fix_injury_data.py
python scripts/preprocessing/fix_data_split.py

# 3. 모델 훈련
python scripts/train.py

# 4. 시각화 생성
python scripts/visualization/paper_figures.py
python scripts/visualization/feature_importance.py
python scripts/visualization/ensemble_analysis.py
```

---

## 프로젝트 구조

```
SoccerMon_Injury_Prediction/
├── README.md                       # 프로젝트 개요
├── requirements.txt                # Python 의존성
├── .gitignore                      # Git 제외 파일
│
├── soccermon/                      # 메인 패키지
│   ├── config.yaml                 # 모델 설정
│   ├── models/
│   │   └── ml_models.py            # ML 모델 클래스
│   └── data/
│       └── loader.py               # 데이터 로딩
│
├── scripts/                        # 실행 스크립트
│   ├── train.py                    # 메인 훈련 파이프라인
│   ├── preprocessing/
│   │   ├── fix_data_split.py       # Train/Test 분할 생성
│   │   └── fix_injury_data.py      # 부상 데이터 검증
│   └── visualization/
│       ├── paper_figures.py        # 논문 품질 플롯
│       ├── feature_importance.py   # 피처 분석
│       └── ensemble_analysis.py    # 앙상블 시각화
│
├── data/
│   ├── raw/                        # 원본 데이터 (Git 제외)
│   └── processed/                  # 전처리 데이터
│       ├── master_dataset.csv
│       └── split_indices.json
│
├── results/                        # 훈련 결과 (Git 제외)
│   └── ml_optimized/
│       ├── ml_results_summary.json
│       └── ml_report.md
│
├── models/                         # 훈련된 모델 (Git 제외)
├── figures/                        # 시각화 (Git 제외)
│
└── docs/                           # 문서
    ├── SETUP.md                    # GitHub 설정 가이드
    └── ARCHITECTURE.md             # 코드 아키텍처 상세
```

---

## 작업 흐름

### 1. 데이터 전처리
```bash
python scripts/preprocessing/fix_injury_data.py
python scripts/preprocessing/fix_data_split.py
```

### 2. 모델 훈련
```bash
python scripts/train.py
```

**이 명령어 하나로:**
- 데이터 로드 및 전처리
- 도메인 특화 피처 엔지니어링
- 9개 ML 모델 훈련
- 상위 5개 앙상블 생성
- 모든 결과를 `results/ml_optimized/`에 저장

### 3. 시각화
```bash
# 저장된 결과로부터 논문 품질 그림 생성
python scripts/visualization/paper_figures.py

# 피처 중요도 분석
python scripts/visualization/feature_importance.py

# 앙상블 분석
python scripts/visualization/ensemble_analysis.py
```

---

## 핵심 기능

### 1. 도메인 지식 기반 피처 엔지니어링
- **Tim Gabbett 훈련 부하 이론**: ACWR 위험 구간 및 최적 구간
- **누적 피로도**: 7일 이동 평균
- **회복 지수**: 준비도, 수면 질, 근육통, 스트레스의 복합 지표

### 2. 불균형 데이터 처리
- **SMOTE**: 소수 클래스(부상) 오버샘플링
- **비용 민감 학습**: 부상 클래스에 10배 가중치

### 3. F2 Score 최적화
- Recall에 2배 가중치 (놓친 부상 최소화)
- 자동 임계값 최적화

### 4. 앙상블 전략
- 9개 모델 훈련: Logistic Regression, Random Forest, XGBoost, LightGBM, MLP 등
- F2 Score 기준 상위 5개 선택
- 가중 확률로 소프트 보팅

---

## 결과

`scripts/train.py` 실행 후 확인:
- **성능 메트릭**: `results/ml_optimized/ml_results_summary.json`
- **상세 리포트**: `results/ml_optimized/ml_report.md`
- **모델 비교**: `results/ml_optimized/9_models_f2_comparison.csv`
- **앙상블 가중치**: `results/ml_optimized/top5_ensemble_weights.csv`

---

## 기술 스택

- **핵심**: Python 3.9+
- **ML**: scikit-learn, XGBoost, LightGBM
- **데이터**: pandas, numpy
- **불균형 처리**: imbalanced-learn (SMOTE)
- **시각화**: matplotlib, seaborn

---

## 문서

- [설정 가이드](docs/SETUP.md) - Git 설정 및 프로젝트 재현
- [아키텍처](docs/ARCHITECTURE.md) - 코드 구조 및 설계 결정
- [시각화 가이드](scripts/visualization/README_VISUALIZATION.md) - 플롯 생성 방법

---

## 라이선스

MIT License - 자세한 내용은 LICENSE 파일 참조

---

## 인용

이 작업을 사용하시는 경우 다음과 같이 인용해주세요:
```
@software{soccermon_injury_prediction,
  title={SoccerMon Injury Prediction System},
  author={SoccerMon Team},
  year={2024},
  url={https://github.com/namgilwoo999/SoccerMon-Injury-Prediction}
}
```
