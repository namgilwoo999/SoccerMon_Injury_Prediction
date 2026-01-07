"""
앙상블 보팅 시각화 - train.py 결과 활용

이 스크립트는 train.py가 생성한 결과 파일을 읽어서 시각화합니다.
모델을 재학습하지 않습니다.

필요한 파일:
- results/ml_optimized/9_models_f2_comparison.csv
- results/ml_optimized/top5_ensemble_weights.csv
- results/ml_optimized/ml_results_summary.json
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
os.chdir(Path(__file__).parent.parent.parent)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print('앙상블 보팅 시각화 (train.py 결과 기반)')

# Step 1: 결과 파일 로드
print('\n[Step 1] train.py 결과 파일 로드...')

results_dir = Path('results/ml_optimized')

# 1-1. 9개 모델 F2 Score 비교 파일
f2_comparison_file = results_dir / '9_models_f2_comparison.csv'
if not f2_comparison_file.exists():
    print(f' 오류: {f2_comparison_file} 파일이 없습니다.')
    print('먼저 python scripts/train.py 를 실행하세요.')
    sys.exit(1)

df_f2 = pd.read_csv(f2_comparison_file)
print(f'9개 모델 F2 Score 데이터 로드: {df_f2.shape}')
print(f'컬럼: {list(df_f2.columns)}')

# 1-2. Top 5 앙상블 가중치 파일
weights_file = results_dir / 'top5_ensemble_weights.csv'
if not weights_file.exists():
    print(f' 오류: {weights_file} 파일이 없습니다.')
    print('먼저 python scripts/train.py 를 실행하세요.')
    sys.exit(1)

df_weights = pd.read_csv(weights_file)
print(f'Top 5 앙상블 가중치 데이터 로드: {df_weights.shape}')

# 1-3. 전체 결과 요약
summary_file = results_dir / 'ml_results_summary.json'
if summary_file.exists():
    with open(summary_file, 'r', encoding='utf-8') as f:
        summary = json.load(f)
    print(f'전체 결과 요약 로드')
else:
    summary = None
    print('ml_results_summary.json 없음 (선택 파일)')

# Step 2: 시각화 생성
print('\n[Step 2] 시각화 생성 중...')

# 출력 폴더 생성
output_dir = Path('figures/ml_optimized')
output_dir.mkdir(parents=True, exist_ok=True)

# 그래프 1: 9개 모델 F2 Score 비교
print('  [2-1] 9개 모델 F2 Score 비교 그래프...')

fig, ax = plt.subplots(figsize=(12, 8))

# 데이터 정렬 (F2 Score 기준)
model_scores = df_f2[['model_name', 'f2_score']].sort_values('f2_score', ascending=True)

# Top 5 모델 강조
colors = ['#d62728' if i >= len(model_scores) - 5 else '#1f77b4'
          for i in range(len(model_scores))]

# 수평 막대 그래프
bars = ax.barh(model_scores['model_name'], model_scores['f2_score'], color=colors)

# 값 표시
for i, (name, score) in enumerate(zip(model_scores['model_name'], model_scores['f2_score'])):
    ax.text(score + 0.01, i, f'{score:.3f}',
            va='center', fontweight='bold', fontsize=10)

ax.set_xlabel('F2 Score', fontsize=14, fontweight='bold')
ax.set_title('9개 모델 F2 Score 비교\n(빨간색 = Top 5 앙상블)', fontsize=16, fontweight='bold')
ax.set_xlim([0, max(model_scores['f2_score']) * 1.15])
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(output_dir / 'ensemble_01_9models_f2_comparison.png', dpi=300, bbox_inches='tight')
print(f'    ✓ 저장: {output_dir / "ensemble_01_9models_f2_comparison.png"}')
plt.close()

# 그래프 2: Top 5 앙상블 가중치
print('  [2-2] Top 5 앙상블 가중치 그래프...')

fig, ax = plt.subplots(figsize=(10, 8))

# 가중치 정규화 (0-1 범위로)
total_weight = df_weights['weight'].sum()
df_weights['normalized_weight'] = df_weights['weight'] / total_weight

# 파이 차트
colors_palette = plt.cm.Set3(np.linspace(0, 1, len(df_weights)))
wedges, texts, autotexts = ax.pie(
    df_weights['normalized_weight'],
    labels=df_weights['model_name'],
    autopct='%1.1f%%',
    startangle=90,
    colors=colors_palette,
    textprops={'fontsize': 11, 'fontweight': 'bold'}
)

ax.set_title('Top 5 앙상블 가중치 분포', fontsize=16, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'ensemble_02_top5_weights.png', dpi=300, bbox_inches='tight')
print(f'저장: {output_dir / "ensemble_02_top5_weights.png"}')
plt.close()

# 그래프 3: 모델별 성능 지표 비교 (F2, AUC, Precision, Recall)
print('  [2-3] 모델별 성능 지표 비교 그래프...')

metrics = ['f2_score', 'auc', 'precision', 'recall']
metric_labels = ['F2 Score', 'AUC', 'Precision', 'Recall']

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
    ax = axes[idx]

    # 데이터 정렬
    sorted_data = df_f2[['model_name', metric]].sort_values(metric, ascending=True)

    # Top 5 모델 강조
    colors = ['#d62728' if i >= len(sorted_data) - 5 else '#1f77b4'
              for i in range(len(sorted_data))]

    # 수평 막대 그래프
    bars = ax.barh(sorted_data['model_name'], sorted_data[metric], color=colors)

    # 값 표시
    for i, (name, score) in enumerate(zip(sorted_data['model_name'], sorted_data[metric])):
        ax.text(score + 0.01, i, f'{score:.3f}',
                va='center', fontweight='bold', fontsize=9)

    ax.set_xlabel(label, fontsize=12, fontweight='bold')
    ax.set_title(f'{label} 비교', fontsize=14, fontweight='bold')
    ax.set_xlim([0, max(sorted_data[metric]) * 1.15])
    ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(output_dir / 'ensemble_03_all_metrics_comparison.png', dpi=300, bbox_inches='tight')
print(f'    ✓ 저장: {output_dir / "ensemble_03_all_metrics_comparison.png"}')
plt.close()

# 그래프 4: 모델 순위표
print('  [2-4] 모델 순위표 생성...')

fig, ax = plt.subplots(figsize=(14, 10))
ax.axis('off')

# F2 Score 기준으로 순위 계산
sorted_models = df_f2.sort_values('f2_score', ascending=False).reset_index(drop=True)

# 테이블 데이터 생성
cell_text = []
for rank, (_, row) in enumerate(sorted_models.iterrows(), 1):
    cell_text.append([
        rank,
        row['model_name'],
        f"{row['f2_score']:.4f}",
        f"{row['auc']:.4f}",
        f"{row['precision']:.4f}",
        f"{row['recall']:.4f}",
        '★' if rank <= 5 else ''
    ])

table = ax.table(cellText=cell_text,
                colLabels=['Rank', 'Model', 'F2 Score', 'AUC', 'Precision', 'Recall', 'Top 5'],
                cellLoc='center',
                loc='center',
                colWidths=[0.08, 0.3, 0.12, 0.12, 0.12, 0.12, 0.08])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# 헤더 스타일
for i in range(7):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white', fontsize=12)

# Top 5 행 강조
for i, row in enumerate(cell_text, 1):
    if row[6] == '★':  # Top 5
        for j in range(7):
            table[(i, j)].set_facecolor('#FFE5E5')

ax.set_title('모델 순위표 (F2 Score 기준)', fontsize=18, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(output_dir / 'ensemble_04_ranking_table.png', dpi=300, bbox_inches='tight')
print(f'저장: {output_dir / "ensemble_04_ranking_table.png"}')
plt.close()

# 그래프 5: Top 5 모델 가중치 막대 그래프
print('  [2-5] Top 5 모델 가중치 막대 그래프...')

fig, ax = plt.subplots(figsize=(12, 8))

# 가중치 정규화된 데이터 정렬
weights_sorted = df_weights.sort_values('normalized_weight', ascending=True)

# 막대 그래프
bars = ax.barh(weights_sorted['model_name'], weights_sorted['normalized_weight'],
               color='#2ecc71')

# 값 표시
for i, (name, weight) in enumerate(zip(weights_sorted['model_name'],
                                       weights_sorted['normalized_weight'])):
    ax.text(weight + 0.01, i, f'{weight*100:.1f}%',
            va='center', fontweight='bold', fontsize=11)

ax.set_xlabel('앙상블 가중치 (%)', fontsize=14, fontweight='bold')
ax.set_title('Top 5 앙상블 모델 가중치', fontsize=16, fontweight='bold')
ax.set_xlim([0, max(weights_sorted['normalized_weight']) * 1.15])
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(output_dir / 'ensemble_05_weights_bar.png', dpi=300, bbox_inches='tight')
print(f'    ✓ 저장: {output_dir / "ensemble_05_weights_bar.png"}')
plt.close()

# 완료
print('\n' + '=' * 70)
print(' 앙상블 시각화 완료!')
print(f' 저장 위치: {output_dir}/')
print('\n생성된 파일:')
print('  - ensemble_01_9models_f2_comparison.png   (9개 모델 F2 Score 비교)')
print('  - ensemble_02_top5_weights.png            (Top 5 앙상블 가중치 - 파이)')
print('  - ensemble_03_all_metrics_comparison.png  (모든 성능 지표 비교)')
print('  - ensemble_04_ranking_table.png           (모델 순위표)')
print('  - ensemble_05_weights_bar.png             (Top 5 앙상블 가중치 - 막대)')
print('=' * 70)
