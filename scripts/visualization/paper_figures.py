import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
os.chdir(Path(__file__).parent.parent.parent)


import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 폰트 크기 설정
plt.rcParams.update({
    'font.size': 16,                # 기본 폰트 크기
    'axes.titlesize': 20,            # 축 제목 크기
    'axes.labelsize': 18,            # 축 라벨 크기
    'xtick.labelsize': 16,           # x축 틱 라벨 크기
    'ytick.labelsize': 16,           # y축 틱 라벨 크기
    'legend.fontsize': 16,           # 범례 폰트 크기
    'figure.titlesize': 24,          # 그림 제목 크기
    'font.weight': 'bold'            # 굵은 글씨
})

def create_visualizations():
    #결과 파일을 읽어서 시각화 생성
    
    logger.info("="*60)
    logger.info("Starting visualization generation...")
    logger.info("="*60)
    
    # 결과 파일 로드
    results_file = Path('r' \
    'esults/ml_optimized/ml_results_summary.json')
    
    if not results_file.exists():
        logger.error(f"Results file not found: {results_file}")
        logger.info("Please run main_ml_optimized.py first to generate results")
        return
    
    # JSON 로드
    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
        logger.info(f"Results loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load results: {e}")
        return
    
    # 출력 디렉토리 생성
    output_dir = Path('figures/ml_optimized_paper')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 스타일 설정
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # 1. 전체 성능 요약 그래프 (개선된 버전)
    logger.info("Creating enhanced performance summary plot...")
    create_enhanced_performance_summary(data, output_dir)
    
    # 2. 과적합 분석 그래프
    logger.info("Creating overfitting analysis plot...")
    create_overfitting_analysis(data, output_dir)
    
    # 3. Confusion Matrix 히트맵
    logger.info("Creating confusion matrix heatmap...")
    create_confusion_matrix_plot(data, output_dir)
    
    # 4. 임계값 분석 그래프
    logger.info("Creating threshold analysis plot...")
    create_threshold_analysis(data, output_dir)
    
    logger.info(f"All visualizations saved to {output_dir}")
    logger.info("="*60)
    logger.info("Visualization completed successfully!")
    logger.info(f"Check the '{output_dir}' folder for results")
    logger.info("="*60)

def create_enhanced_performance_summary(data, output_dir):
    #개선된 전체 성능 요약 그래프 - F2 score 포함
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))  # 크기 증가
    fig.suptitle('SoccerMon ML Pipeline - Enhanced Performance Summary', 
                 fontsize=26, fontweight='bold', y=0.98)
    
    # 데이터 추출
    performance = data.get('ensemble_performance', {})
    
    targets = []
    auc_scores = []
    f1_scores = []
    f2_scores = []
    precision_scores = []
    recall_scores = []
    
    for target, metrics in performance.items():
        targets.append(target.replace('injury_within_', '').replace('d', ' days'))
        auc_scores.append(metrics.get('auc', 0))
        f1_scores.append(metrics.get('f1', 0))
        f2_scores.append(metrics.get('f2', 0))
        precision_scores.append(metrics.get('precision', 0))
        recall_scores.append(metrics.get('recall', 0))
    
    # 색상 설정
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # 1. AUC Score
    ax = axes[0, 0]
    bars = ax.bar(targets, auc_scores, color=colors, width=0.6)
    ax.set_ylabel('ROC AUC Score', fontsize=20, fontweight='bold')
    ax.set_title('AUC Score by Prediction Window', fontsize=22, fontweight='bold', pad=20)
    ax.set_ylim([0.8, 1.0])
    ax.grid(True, alpha=0.3, axis='y')
    for bar, score in zip(bars, auc_scores):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
               f'{score:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=18)
    
    # 2. F1 Score
    ax = axes[0, 1]
    bars = ax.bar(targets, f1_scores, color=colors, width=0.6)
    ax.set_ylabel('F1 Score', fontsize=20, fontweight='bold')
    ax.set_title('F1 Score by Prediction Window', fontsize=22, fontweight='bold', pad=20)
    ax.set_ylim([0, 1.0])
    ax.grid(True, alpha=0.3, axis='y')
    for bar, score in zip(bars, f1_scores):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
               f'{score:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=18)
    
    # 3. F2 Score (새로 추가)
    ax = axes[0, 2]
    bars = ax.bar(targets, f2_scores, color=['#d62728', '#9467bd', '#8c564b'], width=0.6)
    ax.set_ylabel('F2 Score (Recall-focused)', fontsize=20, fontweight='bold')
    ax.set_title('F2 Score by Prediction Window', fontsize=22, fontweight='bold', pad=20)
    ax.set_ylim([0, 1.0])
    ax.grid(True, alpha=0.3, axis='y')
    for bar, score in zip(bars, f2_scores):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
               f'{score:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=18)
    
    # 4. Precision
    ax = axes[1, 0]
    bars = ax.bar(targets, precision_scores, color=colors, width=0.6)
    ax.set_ylabel('Precision', fontsize=20, fontweight='bold')
    ax.set_title('Precision by Prediction Window', fontsize=22, fontweight='bold', pad=20)
    ax.set_ylim([0, 1.0])
    ax.grid(True, alpha=0.3, axis='y')
    for bar, score in zip(bars, precision_scores):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
               f'{score:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=18)
    
    # 5. Recall
    ax = axes[1, 1]
    bars = ax.bar(targets, recall_scores, color=colors, width=0.6)
    ax.set_ylabel('Recall', fontsize=20, fontweight='bold')
    ax.set_title('Recall by Prediction Window', fontsize=22, fontweight='bold', pad=20)
    ax.set_ylim([0, 1.0])
    ax.grid(True, alpha=0.3, axis='y')
    for bar, score in zip(bars, recall_scores):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
               f'{score:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=18)
    
    # 6. 종합 비교 (Table)
    ax = axes[1, 2]
    ax.axis('off')
    
    # 표 생성
    table_data = []
    for i, target in enumerate(targets):
        table_data.append([
            target,
            f'{auc_scores[i]:.4f}',
            f'{f1_scores[i]:.3f}',
            f'{f2_scores[i]:.3f}',
            f'{precision_scores[i]:.3f}',
            f'{recall_scores[i]:.3f}'
        ])
    
    table = ax.table(cellText=table_data,
                    colLabels=['Window', 'AUC', 'F1', 'F2', 'Precision', 'Recall'],
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.18, 0.14, 0.14, 0.14, 0.16, 0.14])
    table.auto_set_font_size(False)
    table.set_fontsize(16)
    table.scale(1.2, 3)
    
    # 헤더 스타일
    for i in range(6):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white', size=18)
    
    # 데이터 셀 폰트 크기
    for i in range(1, 4):
        for j in range(6):
            table[(i, j)].set_text_props(size=16)
    
    ax.set_title('Performance Summary Table', fontsize=22, fontweight='bold', pad=30)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'enhanced_performance_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Enhanced performance summary plot saved")

def create_overfitting_analysis(data, output_dir):
    #과적합 분석 그래프
    
    # 실제 데이터에서 추출
    performance = data.get('ensemble_performance', {})
    
    targets = []
    train_aucs = []
    test_aucs = []
    gaps = []
    
    for target, metrics in performance.items():
        targets.append(target.replace('injury_within_', '').replace('d', ' days'))
        train_auc = metrics.get('train_auc', 0.95)
        test_auc = metrics.get('test_auc', metrics.get('auc', 0.90))
        train_aucs.append(train_auc)
        test_aucs.append(test_auc)
        gaps.append(metrics.get('overfitting_gap', train_auc - test_auc))
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))  # 크기 증가
    fig.suptitle('Overfitting Analysis', fontsize=24, fontweight='bold')
    
    # Train vs Test 비교
    ax = axes[0]
    x = np.arange(len(targets))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, train_aucs, width, label='Train AUC', color='lightblue')
    bars2 = ax.bar(x + width/2, test_aucs, width, label='Test AUC', color='orange')
    
    ax.set_xlabel('Prediction Window', fontsize=20, fontweight='bold')
    ax.set_ylabel('AUC Score', fontsize=20, fontweight='bold')
    ax.set_title('Train vs Test Performance', fontsize=22, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(targets, fontsize=18)
    ax.legend(fontsize=18)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0.85, 1.0])
    
    # 값 표시
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
               f'{height:.3f}', ha='center', va='bottom', fontsize=16)
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
               f'{height:.3f}', ha='center', va='bottom', fontsize=16)
    
    # Overfitting Gap
    ax = axes[1]
    bars = ax.bar(targets, gaps, color=['green' if g < 0.05 else 'orange' if g < 0.1 else 'red' for g in gaps])
    ax.set_xlabel('Prediction Window', fontsize=20, fontweight='bold')
    ax.set_ylabel('Overfitting Gap (Train - Test)', fontsize=20, fontweight='bold')
    ax.set_title('Overfitting Gap Analysis', fontsize=22, fontweight='bold')
    ax.axhline(y=0.1, color='r', linestyle='--', linewidth=2, label='Danger Threshold')
    ax.axhline(y=0.05, color='orange', linestyle='--', linewidth=2, label='Warning Threshold')
    ax.legend(fontsize=18)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, gap in zip(bars, gaps):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.002,
               f'{gap:.4f}', ha='center', va='bottom', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'overfitting_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Overfitting analysis plot saved")

def create_confusion_matrix_plot(data, output_dir):
    #Confusion Matrix 히트맵
    
    # 실제 데이터에서 추출
    performance = data.get('ensemble_performance', {})
    
    matrices = {}
    for target, metrics in performance.items():
        cm = metrics.get('confusion_matrix', {})
        if cm:
            label = target.replace('injury_within_', '').replace('d', ' days')
            matrices[label] = [
                [cm.get('tn', 0), cm.get('fp', 0)],
                [cm.get('fn', 0), cm.get('tp', 0)]
            ]
    
    if not matrices:
        logger.warning("No confusion matrix data found")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))  # 크기 증가
    fig.suptitle('Confusion Matrices by Prediction Window', fontsize=24, fontweight='bold')
    
    for idx, (title, cm) in enumerate(matrices.items()):
        ax = axes[idx]
        
        # 정규화된 confusion matrix
        cm_array = np.array(cm)
        cm_normalized = cm_array / cm_array.sum(axis=1, keepdims=True)
        
        # 히트맵
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                   square=True, cbar_kws={'label': 'Percentage'},
                   xticklabels=['No Injury', 'Injury'],
                   yticklabels=['No Injury', 'Injury'],
                   ax=ax, annot_kws={'size': 18, 'weight': 'bold'})  # 폰트 크기 증가
        
        ax.set_title(title, fontsize=20, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=18, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=18, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=16)
        
        # 실제 값도 표시
        for i in range(2):
            for j in range(2):
                ax.text(j + 0.5, i + 0.7, f'n={cm[i][j]}',
                       ha='center', va='center', fontsize=14, color='gray')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Confusion matrix plot saved")

def create_threshold_analysis(data, output_dir):
    #임계값 분석 그래프 - 실제 데이터 사용
    
    # 실제 threshold curves 데이터 확인
    threshold_curves_data = data.get('threshold_curves', {})
    performance = data.get('ensemble_performance', {})
    
    if threshold_curves_data:
        # 실제 데이터가 있으면 사용
        create_threshold_analysis_real(data, output_dir, threshold_curves_data)
    else:
        # 없으면 시뮬레이션 사용
        logger.warning("No real threshold curves data found - using simulation")
        create_threshold_analysis_simulation(data, output_dir)

def create_threshold_analysis_real(data, output_dir, threshold_curves_data):
    #실제 threshold curves 데이터를 사용한 분석
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))  # 3개 subplot
    fig.suptitle('Threshold Analysis (Real Data)', fontsize=24, fontweight='bold')
    
    # 각 예측 기간별 임계값 곡선
    for idx, (target, curves) in enumerate(threshold_curves_data.items()):
        ax = axes[idx]
        
        thresholds = curves.get('thresholds', [])
        f1_scores = curves.get('f1_scores', [])
        f2_scores = curves.get('f2_scores', [])
        precision_scores = curves.get('precision_scores', [])
        recall_scores = curves.get('recall_scores', [])
    
        # 곡선 그리기
        ax.plot(thresholds, f1_scores, 'b-', label='F1 Score', linewidth=3)
        ax.plot(thresholds, f2_scores, 'r-', label='F2 Score', linewidth=3)
        ax.plot(thresholds, precision_scores, 'g--', label='Precision', linewidth=2.5)
        ax.plot(thresholds, recall_scores, 'm--', label='Recall', linewidth=2.5)
    
        # 최적 F2 임계값 표시
        if f2_scores:
            optimal_f2_idx = np.argmax(f2_scores)
            optimal_threshold = thresholds[optimal_f2_idx]
            optimal_f2 = f2_scores[optimal_f2_idx]
            
            ax.plot(optimal_threshold, optimal_f2, 'ro', markersize=15)
            ax.axvline(x=optimal_threshold, color='red', linestyle=':', alpha=0.5, linewidth=2)
            ax.text(optimal_threshold, 0.1, f'Optimal\n{optimal_threshold:.3f}',
                   ha='center', fontweight='bold', fontsize=14)
    
        ax.set_xlabel('Threshold', fontsize=18, fontweight='bold')
        ax.set_ylabel('Score', fontsize=18, fontweight='bold')
        ax.set_title(target.replace('injury_within_', '').replace('d', ' days'), 
                    fontsize=20, fontweight='bold')
        ax.legend(loc='best', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.tick_params(axis='both', which='major', labelsize=14)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'threshold_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Threshold analysis plot with real data saved")
    
def create_threshold_analysis_simulation(data, output_dir):
    #시뮬레이션 데이터를 사용한 임계값 분석
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle('Threshold Analysis (Simulation)', fontsize=24, fontweight='bold')
    
    # 시뮬레이션 데이터
    thresholds = np.linspace(0.1, 0.9, 20)
    f1_scores = [0.3 + 0.3 * np.exp(-(t-0.4)**2/0.1) for t in thresholds]
    f2_scores = [0.35 + 0.35 * np.exp(-(t-0.35)**2/0.12) for t in thresholds]
    precision = [0.2 + 0.6 * t for t in thresholds]
    recall = [0.9 - 0.7 * t for t in thresholds]
    
    ax = axes[0]
    ax.plot(thresholds, f1_scores, 'b-', label='F1 Score', linewidth=3)
    ax.plot(thresholds, f2_scores, 'r-', label='F2 Score', linewidth=3)
    ax.plot(thresholds, precision, 'g--', label='Precision', linewidth=2.5)
    ax.plot(thresholds, recall, 'm--', label='Recall', linewidth=2.5)
    
    ax.set_xlabel('Threshold', fontsize=20, fontweight='bold')
    ax.set_ylabel('Score', fontsize=20, fontweight='bold')
    ax.set_title('Metrics vs Threshold (Simulation)', fontsize=22, fontweight='bold')
    ax.legend(loc='best', fontsize=18)
    ax.grid(True, alpha=0.3)
    
    # 실제 사용된 임계값
    ax = axes[1]
    performance = data.get('ensemble_performance', {})
    
    windows = []
    actual_thresholds = []
    f2_at_threshold = []
    
    for target, metrics in performance.items():
        windows.append(target.replace('injury_within_', '').replace('d', ' days'))
        actual_thresholds.append(metrics.get('threshold', 0.5))
        f2_at_threshold.append(metrics.get('f2', 0))
    
    bars = ax.bar(windows, actual_thresholds, color='steelblue', alpha=0.7)
    ax2 = ax.twinx()
    line = ax2.plot(windows, f2_at_threshold, 'ro-', linewidth=3, markersize=12, label='F2 Score')
    
    ax.set_ylabel('Optimal Threshold', color='steelblue', fontsize=20, fontweight='bold')
    ax2.set_ylabel('F2 Score at Threshold', color='red', fontsize=20, fontweight='bold')
    ax.set_title('Optimal Thresholds by Prediction Window', fontsize=22, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax2.tick_params(axis='both', which='major', labelsize=16)
    
    for bar, thresh in zip(bars, actual_thresholds):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
               f'{thresh:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=16)
    
    for i, (x, y) in enumerate(zip(range(len(windows)), f2_at_threshold)):
        ax2.text(x, y + 0.02, f'{y:.3f}', ha='center', va='bottom', 
                fontweight='bold', fontsize=16, color='red')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'threshold_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Threshold analysis plot saved")

if __name__ == "__main__":
    create_visualizations()
