# SoccerMon 부상 예측 - 아키텍처 가이드

> 코드 구조와 설계 결정에 대한 상세 문서

---

## 패키지 구성

### `soccermon/` - 핵심 패키지

재사용 가능한 ML 컴포넌트를 포함하는 메인 Python 패키지입니다.

#### `soccermon/models/ml_models.py`
**목적**: 불균형 데이터를 위한 최적화된 ML 모델 클래스

**핵심 클래스**: `OptimizedMLModels`
- 9개의 서로 다른 ML 알고리즘 처리
- SMOTE/ADASYN 오버샘플링
- 비용 민감 학습
- F2 Score 최적화
- 자동 임계값 튜닝
- 앙상블 가중치 계산

**사용법**:
```python
from soccermon.models import OptimizedMLModels

model_trainer = OptimizedMLModels(
    use_smote=True,
    smote_ratio=0.7,
    sampling_strategy='combined',
    random_state=42
)

trained_models = model_trainer.train_all_models(
    X_train, y_train, X_test, y_test, cv_folds=3
)
```

#### `soccermon/data/loader.py`
**목적**: SoccerMon 원시 데이터 로드 및 통합

**핵심 클래스**: `SoccerMonDataProcessor`
- 훈련 부하 데이터 로드 (ACWR, CTL, ATL)
- 웰니스 데이터 로드 (피로도, 기분, 수면)
- 부상 기록 로드
- 모든 데이터 소스를 마스터 데이터셋으로 병합

**사용법**:
```python
from soccermon.data import SoccerMonDataProcessor

processor = SoccerMonDataProcessor('data/raw')
master_data = processor.load_all_data()
```

**참고**: `data/processed/master_dataset.csv`가 없을 때만 사용됩니다. 대부분의 실행은 전처리된 데이터를 직접 사용합니다.

#### `soccermon/config.yaml`
**목적**: 모델 훈련을 위한 중앙화된 설정

**주요 설정**:
- 데이터 경로
- Train/Test 분할 비율
- 피처 선택 파라미터
- 모델 하이퍼파라미터
- SMOTE 비율
- 교차 검증 폴드 수

---

### `scripts/` - 실행 가능한 스크립트

#### `scripts/train.py`
**목적**: 메인 훈련 파이프라인

**작업 흐름**:
1. 데이터 로드 및 전처리
2. 도메인 피처 엔지니어링 (Tim Gabbett 이론)
3. 피처 선택 (상위 20개)
4. 9개 ML 모델 훈련
5. F2 Score 기반 상위 5개 모델 앙상블
6. 결과 저장

**의존성**:
- `soccermon.models.ml_models.OptimizedMLModels`
- `soccermon.data.loader.SoccerMonDataProcessor`

**실행**: `python scripts/train.py`

#### `scripts/preprocessing/fix_data_split.py`
**목적**: 층화 추출 Train/Test 분할 생성

- 선수 ID별 층화 (데이터 누출 방지)
- 80/20 분할
- 인덱스를 `data/processed/split_indices.json`에 저장

**실행**: `python scripts/preprocessing/fix_data_split.py`

#### `scripts/preprocessing/fix_injury_data.py`
**목적**: 부상 데이터 검증 및 정제

- 결측치 확인
- 날짜 형식 검증
- 부상 레이블이 이진값인지 확인
- `data/processed/master_dataset_fixed.csv`에 저장

**실행**: `python scripts/preprocessing/fix_injury_data.py`

#### `scripts/visualization/paper_figures.py`
**목적**: 논문 품질의 그림 생성

- `results/ml_optimized/ml_results_summary.json` 로드
- AUC 비교 플롯 생성
- F2 Score 비교 플롯 생성
- `figures/ml_optimized_paper/*.png`에 저장

**실행**: `python scripts/visualization/paper_figures.py`

#### `scripts/visualization/feature_importance.py`
**목적**: 피처 중요도 분석 및 시각화

- Random Forest 및 Logistic Regression 사용
- 상호 정보량 점수
- 상관관계 분석
- 플롯을 `figures/`에 저장

**실행**: `python scripts/visualization/feature_importance.py`

#### `scripts/visualization/ensemble_analysis.py`
**목적**: 앙상블 선택 과정 시각화

- 9개 모델의 F2 Score 표시
- 앙상블에 선택된 상위 5개 강조
- 개별 vs 앙상블 성능 비교

**의존성**:
- `soccermon.models.ml_models.OptimizedMLModels`

**실행**: `python scripts/visualization/ensemble_analysis.py`

---

## 실행 흐름

### 최초 설정
```
1. fix_injury_data.py → 데이터 검증
2. fix_data_split.py  → 분할 생성
3. train.py           → 모델 훈련
4. paper_figures.py   → 플롯 생성
5. feature_importance.py → 피처 분석
6. ensemble_analysis.py  → 앙상블 시각화
```

### 이후 실행
```
1. train.py (단독) → 기존 전처리 데이터 및 분할 사용
```

---

## 핵심 알고리즘

### 1. Tim Gabbett ACWR 이론
```python
# 위험 구역 감지
acwr_danger_zone = (acwr > 1.5) | (acwr < 0.8)

# 최적 구간 (최적 훈련 부하)
acwr_sweet_spot = (acwr >= 0.8) & (acwr <= 1.3)

# 만성 부하 안정성
chronic_stability = CTL_28day / (CTL_42day + epsilon)
```

### 2. SMOTE 오버샘플링
```python
smote = SMOTE(
    sampling_strategy=0.7,  # 다수 클래스의 70%
    random_state=42,
    k_neighbors=5
)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

### 3. F2 Score 최적화
```python
# F2 Score: 정밀도보다 재현율에 2배 가중치
f2_score = (1 + 2^2) * (precision * recall) / (2^2 * precision + recall)

# 임계값 최적화
best_threshold = max(thresholds, key=lambda t: f2_score_at_threshold(t))
```

### 4. 앙상블 가중치
```python
# F2 Score 기준 상위 5개 모델 선택
top_5_models = sorted(models, key=lambda m: m.f2_score, reverse=True)[:5]

# F2 기반 가중치로 소프트 보팅
weights = [model.f2_score for model in top_5_models]
ensemble_pred = sum(w * model.predict_proba() for w, model in zip(weights, top_5_models))
ensemble_pred /= sum(weights)
```

---

## 데이터 흐름

```
data/raw/
├── subjective/training-load/
│   ├── acwr.csv
│   ├── ctl28.csv
│   └── daily_load.csv
└── subjective/wellness/
    ├── fatigue.csv
    ├── readiness.csv
    └── sleep_quality.csv

    ↓ (SoccerMonDataProcessor)

data/processed/
├── master_dataset.csv       # 통합 데이터
└── split_indices.json       # Train/Test 분할

    ↓ (scripts/train.py)

results/ml_optimized/
├── ml_results_summary.json  # 성능 메트릭
└── ml_report.md             # 상세 리포트

models/ml_optimized/
└── *.pkl                    # 훈련된 모델

figures/ml_optimized/
└── *.png                    # 시각화
```


## 설계 결정

### 왜 F1 대신 F2 Score인가?
- **재현율이 중요**: 부상을 놓치는 것(거짓 음성)이 거짓 경보보다 나쁨
- F2는 재현율에 정밀도보다 2배 가중치
- 의료/스포츠 과학 우선순위와 일치

### 왜 상위 5개 앙상블인가?
- 5개 이상에서는 경험적으로 수익 체감
- 약한 모델의 과적합 위험 감소
- 성능과 복잡성의 균형

### 왜 SMOTE 비율 0.7인가?
- 0.5는 너무 보수적 (낮은 재현율)
- 0.8+ 는 과적합 발생
- 0.7이 최고의 F2 Score / 과적합 균형 제공

### 왜 scripts/와 soccermon/을 분리했는가?
- **soccermon/**: 재사용 가능한 라이브러리 코드 (import 가능)
- **scripts/**: 실행 진입점 (실행 가능)
- 표준 Python 프로젝트 구조를 따름
- 코드 구성과 발견성 향상
