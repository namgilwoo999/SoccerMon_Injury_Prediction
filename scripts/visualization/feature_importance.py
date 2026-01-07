"""
피처 중요도 분석 및 시각화

이 스크립트는 의도적으로 train.py의 피처 엔지니어링 코드를 복사합니다.

1. 피처 중요도를 계산하려면 실제로 모델을 학습해야 함
2. RandomForest와 LogisticRegression을 직접 학습하여 중요도 추출
3. train.py와 동일한 피처를 사용해야 결과가 일치함

train.py와의 차이점:
- train.py: 9개 모델 학습 + 앙상블 (메인 파이프라인)
- 이 스크립트: 2개 모델만 학습 (피처 분석 전용)
"""

# 실제 파이프라인과 동일하게 파생 변수 생성 후 피처 중요도 계산
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
os.chdir(Path(__file__).parent.parent.parent)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
# 주의: 아래 피처 엔지니어링 코드는 train.py와 동일합니다.
# 이유: 피처 중요도 분석을 위해 동일한 피처 세트가 필요하기 때문입니다.
# train.py의 피처 엔지니어링이 변경되면 여기도 함께 업데이트해야 합니다.
from sklearn.linear_model import LogisticRegression
import sys
sys.path.append('.')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print('파생 변수 포함 피처 중요도 계산')

# Step 1: 데이터 로드
print('\n[Step 1] 원본 데이터 로드...')
data = pd.read_csv('data/processed/master_dataset.csv')
data['date'] = pd.to_datetime(data['date'])

print(f'  원본 데이터: {data.shape}')
print(f'  원본 컬럼 수: {len(data.columns)}개')

# Step 2: 파생 변수 생성 (main_ml_optimized.py의 create_domain_features 로직)
print('\n[Step 2] 파생 변수 생성 중...')

df = data.copy()

# 필수 컬럼 확인
required_cols = ['acwr', 'ctl28', 'ctl42', 'daily_load', 'monotony',
                'fatigue', 'readiness', 'sleep_quality', 'soreness', 'stress']

# 1. Tim Gabbett의 훈련 부하 이론 적용
print('  [2-1] Tim Gabbett 이론 기반 피처...')
df['acwr_danger_zone'] = ((df['acwr'] > 1.5) | (df['acwr'] < 0.8)).astype(int)
df['acwr_sweet_spot'] = ((df['acwr'] >= 0.8) & (df['acwr'] <= 1.3)).astype(int)
df['chronic_load_stability'] = df['ctl28'] / (df['ctl42'] + 1e-5)
df['training_monotony_risk'] = (df['monotony'] > df['monotony'].quantile(0.75)).astype(int)

# 2. 급격한 부하 변화 감지
print('  [2-2] 부하 변화 감지 피처...')
if 'player_id' in df.columns:
    df['load_spike_3d'] = df.groupby('player_id')['daily_load'].transform(
        lambda x: x.diff(3) > x.std() * 2
    ).fillna(0).astype(int)

    df['load_spike_7d'] = df.groupby('player_id')['daily_load'].transform(
        lambda x: x.diff(7) > x.std() * 1.5
    ).fillna(0).astype(int)

    # 3. 누적 피로도 지표
    print('  [2-3] 누적 피로도 피처...')
    df['cumulative_fatigue'] = df.groupby('player_id')['fatigue'].transform(
        lambda x: x.rolling(7, min_periods=3).mean()
    ).fillna(df['fatigue'].mean())

    df['fatigue_trend'] = df.groupby('player_id')['fatigue'].transform(
        lambda x: x.diff(7)
    ).fillna(0)

    # 4. 부하 변동성
    df['load_variability'] = df.groupby('player_id')['daily_load'].transform(
        lambda x: x.rolling(7, min_periods=3).std()
    ).fillna(0)

# 5. 회복 지표
print('  [2-4] 회복 지표 피처...')
df['recovery_index'] = (
    df['readiness'] * 0.3 +
    df['sleep_quality'] * 0.3 +
    (10 - df['soreness']) * 0.2 +
    (10 - df['stress']) * 0.2
) / 10

# 6. 웰니스 일관성 지표
print('[2-5] 웰니스 피처...')
wellness_cols = ['fatigue', 'readiness', 'soreness', 'stress']
wellness_cols = [col for col in wellness_cols if col in df.columns]

if wellness_cols:
    df['wellness_consistency'] = df[wellness_cols].std(axis=1)
    df['wellness_decline'] = (df[wellness_cols].mean(axis=1) < 5).astype(int)
    df['wellness_score'] = df[wellness_cols].mean(axis=1)

# 7. 복합 리스크 지표
print('  [2-6] 복합 리스크 피처...')
df['composite_risk'] = (
    df['acwr_danger_zone'] * 0.3 +
    df['training_monotony_risk'] * 0.2 +
    df.get('load_spike_3d', 0) * 0.2 +
    df['wellness_decline'] * 0.3
)

# 8. 포지션별 리스크 (시뮬레이션)
df['position_risk_factor'] = np.random.choice([0.8, 1.0, 1.2], size=len(df))

# 9. 시즌 단계별 리스크
print('  [2-7] 시즌 단계 피처...')
if 'date' in df.columns:
    df['day_of_season'] = (df['date'] - df['date'].min()).dt.days
    df['early_season_risk'] = (df['day_of_season'] < 30).astype(int)
    df['late_season_risk'] = (df['day_of_season'] > 300).astype(int)
    df['mid_season'] = ((df['day_of_season'] >= 30) & (df['day_of_season'] <= 300)).astype(int)

print(f'\n  파생 변수 생성 완료!')
print(f'총 컬럼 수: {len(df.columns)}개 (원본: {len(data.columns)}개)')
print(f'추가된 파생 변수: {len(df.columns) - len(data.columns)}개')

# Step 3: 피처 준비
print('\n[Step 3] 피처 선택 준비...')

target = 'injury_within_3d'

# 숫자형 피처만 선택
numeric_features = df.select_dtypes(include=[np.number]).columns
exclude_cols = ['injury', 'injury_within_3d', 'injury_within_7d',
               'injury_within_14d', 'player_id', 'day_of_season']
feature_cols = [col for col in numeric_features if col not in exclude_cols]

X = df[feature_cols].fillna(df[feature_cols].median())
y = df[target].fillna(0)

print(f'피처 개수: {len(feature_cols)}개 ← 원본 + 파생')
print(f'양성 샘플: {y.sum()}개 ({y.mean():.2%})')

# Step 4: 3가지 방법으로 중요도 계산
print('\n[Step 4] 피처 중요도 계산...')

# Method 1: MI
print('[4-1] Mutual Information...')
mi_selector = SelectKBest(mutual_info_classif, k=min(30, len(feature_cols)))
mi_selector.fit(X, y)
mi_scores = pd.DataFrame({
    'feature': feature_cols,
    'mi_score': mi_selector.scores_
}).sort_values('mi_score', ascending=False)

# Method 2: RF
print('  [4-2] Random Forest...')
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42,
    class_weight='balanced'
)
rf.fit(X, y)
rf_importance = pd.DataFrame({
    'feature': feature_cols,
    'rf_importance': rf.feature_importances_
}).sort_values('rf_importance', ascending=False)

# Method 3: L1
print('  [4-3] L1 Lasso (C=0.05)...')
lasso = LogisticRegression(
    penalty='l1',
    solver='liblinear',
    C=0.05,
    class_weight='balanced',
    random_state=42,
    max_iter=200
)
lasso.fit(X, y)
lasso_importance = pd.DataFrame({
    'feature': feature_cols,
    'lasso_coef': np.abs(lasso.coef_[0])
}).sort_values('lasso_coef', ascending=False)

# Step 5: 하이브리드 점수 계산
print('\n[Step 5] 하이브리드 점수 계산...')

# 병합
feature_scores = pd.merge(mi_scores, rf_importance, on='feature')
feature_scores = pd.merge(feature_scores, lasso_importance, on='feature')

# 정규화
for col in ['mi_score', 'rf_importance', 'lasso_coef']:
    max_val = feature_scores[col].max()
    if max_val > 0:
        feature_scores[f'{col}_norm'] = feature_scores[col] / max_val
    else:
        feature_scores[f'{col}_norm'] = 0

# 가중 평균
feature_scores['combined_score'] = (
    feature_scores['mi_score_norm'] * 0.3 +
    feature_scores['rf_importance_norm'] * 0.4 +
    feature_scores['lasso_coef_norm'] * 0.3
)

feature_scores = feature_scores.sort_values('combined_score', ascending=False)

# 상위 20개
top_20 = feature_scores.head(20)

print(f'\n상위 20개 피처 (Combined Score):')
for i, (idx, row) in enumerate(top_20.iterrows(), 1):
    # 원본 vs 파생 구분
    is_original = row['feature'] in ['acwr', 'atl', 'ctl28', 'ctl42', 'daily_load',
                                     'monotony', 'strain', 'fatigue', 'mood', 'readiness',
                                     'sleep_duration', 'sleep_quality', 'soreness', 'stress']
    marker = '⭕' if is_original else '⚙️'
    print(f'  {i:2d}. {marker} {row["feature"]:30s} {row["combined_score"]:.4f}')

# 통계
original_count = sum(1 for idx, row in top_20.iterrows()
                    if row['feature'] in ['acwr', 'atl', 'ctl28', 'ctl42', 'daily_load',
                                          'monotony', 'strain', 'fatigue', 'mood', 'readiness',
                                          'sleep_duration', 'sleep_quality', 'soreness', 'stress'])
derived_count = 20 - original_count

print(f'\n통계:')
print(f'원본 변수: {original_count}개 ({original_count/20*100:.0f}%)')
print(f'파생 변수: {derived_count}개 ({derived_count/20*100:.0f}%)')


# CSV 저장
output_dir = Path('results/ml_optimized')
output_dir.mkdir(parents=True, exist_ok=True)
csv_path = output_dir / 'feature_importance_full_top20.csv'
top_20.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f'저장: {csv_path}')

# 전체 피처 중요도도 저장
all_csv_path = output_dir / 'feature_importance_all.csv'
feature_scores.to_csv(all_csv_path, index=False, encoding='utf-8-sig')
print(f'저장: {all_csv_path}')

# 시각화
print('[Step 6] 시각화 생성...')
fig, ax = plt.subplots(figsize=(10, 10))
top_20_sorted = top_20.sort_values('combined_score')

colors = []
for idx, row in top_20_sorted.iterrows():
    if row['feature'] in ['acwr', 'atl', 'ctl28', 'ctl42', 'daily_load',
                         'monotony', 'strain', 'fatigue', 'mood', 'readiness',
                         'sleep_duration', 'sleep_quality', 'soreness', 'stress']:
        colors.append('#3498db')  # 파랑 = 원본
    else:
        colors.append('#2ecc71')  # 초록 = 파생

bars = ax.barh(range(len(top_20_sorted)), top_20_sorted['combined_score'], color=colors)

ax.set_yticks(range(len(top_20_sorted)))
ax.set_yticklabels(top_20_sorted['feature'], fontsize=10)
ax.set_xlabel('Combined Score', fontsize=12, fontweight='bold')
ax.set_title(f'하이브리드 피처 선택 결과 (Top 20)\n파랑=원본({original_count}개), 초록=파생({derived_count}개)',
             fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

# 값 표시
for i, (idx, row) in enumerate(top_20_sorted.iterrows()):
    ax.text(row['combined_score'] + 0.01, i, f'{row["combined_score"]:.3f}',
            va='center', fontsize=8)

plt.tight_layout()
plt.savefig('figures/feature_importance_full_combined.png', dpi=300, bbox_inches='tight')
print('  저장: figures/feature_importance_full_combined.png')

print('\n' + '=' * 70)
print(f'완료! 총 {len(feature_cols)}개 피처에서 상위 20개 선택')
print(f'-원본 변수: {original_count}개')
print(f'-파생 변수: {derived_count}개')
print('=' * 70)
