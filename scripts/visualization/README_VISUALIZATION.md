# 시각화 스크립트 사용 가이드

## 사용 순서

### 1. 먼저 모델 학습
```bash
cd H:/SoccerMon_Injury_Prediction
python scripts/train.py
```
이 명령이 생성하는 것:
- `results/ml_optimized/ml_results_summary.json` - 모든 모델 성능 지표
- `results/ml_optimized/ml_report.md` - 상세 리포트
- `results/ml_optimized/9_models_f2_comparison.csv` - 9개 모델 F2 Score 비교
- `results/ml_optimized/top5_ensemble_weights.csv` - 앙상블 가중치

### 2. 시각화 생성

#### paper_figures.py
```bash
python scripts/visualization/paper_figures.py
```
- train.py 결과를 읽어서 논문용 고품질 그래프 생성
- 모델 재학습 안 함 (빠름!)
- **출력**: AUC, F1, F2, Precision, Recall, Confusion Matrix

#### ensemble_analysis.py (앙상블 시각화)
```bash
python scripts/visualization/ensemble_analysis.py
```
- train.py 결과를 읽어서 앙상블 보팅 과정 시각화
- 모델 재학습 안 함 (빠름!)
- **출력**: 
  - 9개 모델 F2 Score 비교
  - Top 5 앙상블 가중치
  - 앙상블 최종 성능
  - 모델 순위표

#### feature_importance.py (피처 분석)
```bash
python scripts/visualization/feature_importance.py
```
- 피처 중요도 분석
- **주의**: 이 스크립트만 모델을 학습합니다 (RandomForest, Logistic)
- **이유**: 피처 중요도를 계산하려면 직접 모델을 학습해야 함
- train.py와 독립적으로 실행 가능
- **출력**: 피처 중요도 순위, Mutual Information, 상관관계 분석

---

## 출력 위치

모든 시각화는 `figures/ml_optimized/` 폴더에 저장됩니다:

```
figures/ml_optimized/
├── performance_summary.png              (paper_figures.py)
├── overfitting_analysis.png             (paper_figures.py)
├── confusion_matrices.png               (paper_figures.py)
├── threshold_analysis.png               (paper_figures.py)
├── ensemble_01_9models_f2_comparison.png  (ensemble_analysis.py)
├── ensemble_02_top5_weights.png           (ensemble_analysis.py)
├── ensemble_03_final_performance.png      (ensemble_analysis.py)
├── ensemble_04_ranking_table.png          (ensemble_analysis.py)
├── feature_importance_*.png               (feature_importance.py)
└── ...
```

---

## 워크플로우 요약

```bash
# 전체 워크플로우 (처음부터 끝까지)

# 1. 전처리 (최초 1회)
python scripts/preprocessing/fix_injury_data.py
python scripts/preprocessing/fix_data_split.py

# 2. 모델 학습 (메인)
python scripts/train.py

# 3. 시각화 (선택 - 원하는 것만 실행)
python scripts/visualization/paper_figures.py      # 논문용 성능 그래프
python scripts/visualization/ensemble_analysis.py  # 앙상블 보팅 시각화
python scripts/visualization/feature_importance.py # 피처 중요도 분석
```

---

## 중요 사항

### paper_figures.py와 ensemble_analysis.py
- train.py 결과만 읽음 (모델 학습 ❌)
- 빠르게 실행됨
- train.py 없이 실행하면 에러 발생

### feature_importance.py
- 모델을 직접 학습함 (RandomForest, Logistic)
- train.py의 피처 엔지니어링 코드를 의도적으로 복사
- train.py와 독립적으로 실행 가능
- **중복 이유**: 피처 중요도를 계산하려면 실제 모델 학습이 필요

---

## 각 스크립트의 목적

| 스크립트 | 목적 | 모델 학습 | train.py 의존 |
|---------|------|----------|-------------|
| `paper_figures.py` | 논문용 성능 그래프 | ❌ | ✅ 필수 |
| `ensemble_analysis.py` | 앙상블 보팅 과정 시각화 | ❌ | ✅ 필수 |
| `feature_importance.py` | 피처 중요도 분석 | ✅ (2개 모델) | ❌ 독립 |

---

**작성일**: 2024-12-31  
**프로젝트**: SoccerMon Injury Prediction System
