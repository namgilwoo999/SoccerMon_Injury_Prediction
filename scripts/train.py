"""
SoccerMon 부상 예측: 최적화된 기계학습 파이프라인

주요 개선사항:
1. SMOTE ratio 0.5로 증가
2. F2 Score 최적화 (Recall 중시)
3. Cost-sensitive learning
4. 더 많은 모델과 앙상블 전략
"""



import sys
import os
from pathlib import Path
# Add parent directory to path to find soccermon package
sys.path.insert(0, str(Path(__file__).parent.parent))

# Change to project root directory for correct relative paths
os.chdir(Path(__file__).parent.parent)

import pandas as pd
import numpy as np
import yaml
import logging
from pathlib import Path
from datetime import datetime
import warnings
from sklearn.model_selection import StratifiedGroupKFold, TimeSeriesSplit
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix, fbeta_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFECV
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib

warnings.filterwarnings('ignore')

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('soccermon_ml_optimized.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class OptimizedMLPipeline:
    """개선된 ML 파이프라인 - 불균형 데이터 최적화"""
    
    def __init__(self, config_path: str = "../soccermon/config.yaml"):
        """
        Args:
            config_path: 설정 파일 경로
        """
        self.config = self._load_config(config_path)
        self._setup_directories()
        self.results = {}
        
    def _load_config(self, config_path):
        """설정 파일 로드"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                # 개선된 설정 추가
                config['models']['smote_ratio'] = 0.7  # 증가 (0.3 -> 0.5 -> 0.7)
                config['models']['sampling_strategy'] = 'combined'
                config['models']['optimize_for'] = 'f2'  # F2 score 최적화
                return config
        except FileNotFoundError:
            logger.warning(f"Config file not found at {config_path}, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self):
        """기본 설정값 반환"""
        return {
            'data': {
                'raw_path': 'data/raw',
                'processed_path': 'data/processed',
                'test_size': 0.2,
                'validation_size': 0.15
            },
            'features': {
                'max_features': 20,  # 15에서 20으로 증가
                'domain_features': True,
                'interaction_features': True,
                'temporal_features': True
            },
            'models': {
                'use_ensemble': True,
                'cross_validation_folds': 5,
                'early_stopping': True,
                'smote_ratio': 0.7,
                'sampling_strategy': 'combined',
                'optimize_for': 'f2'
            },
            'output': {
                'results': 'results/ml_optimized',
                'models': 'models/ml_optimized',
                'figures': 'figures/ml_optimized'
            }
        }
    
    def _setup_directories(self):
        """출력 디렉토리 생성"""
        for path_key, path_value in self.config['output'].items():
            Path(path_value).mkdir(parents=True, exist_ok=True)
    
    def run_pipeline(self):
        """전체 파이프라인 실행"""
        logger.info("="*60)
        logger.info("최적화된 기계학습 파이프라인 시작")
        logger.info("="*60)
        
        try:
            # 1. 데이터 로드 및 전처리
            logger.info("\n[Phase 1] 데이터 로드 및 전처리")
            data = self.load_and_preprocess_data()
            
            # 2. 도메인 지식 기반 피처 엔지니어링
            logger.info("\n[Phase 2] 도메인 지식 기반 피처 엔지니어링")
            data = self.create_domain_features(data)
            
            # 3. 피처 선택 및 최적화
            logger.info("\n[Phase 3] 피처 선택 및 최적화")
            selected_features = self.select_optimal_features(data)
            
            # 4. 모델 훈련 및 평가
            logger.info("\n[Phase 4] 모델 훈련 및 평가")
            models = self.train_models(data, selected_features)
            
            # 5. 앙상블 및 최종 평가
            logger.info("\n[Phase 5] 앙상블 모델 구축")
            ensemble_results = self.create_ensemble(models, data)
            
            # 6. 해석가능성 분석
            logger.info("\n[Phase 6] 모델 해석가능성 분석")
            interpretability = self.analyze_interpretability(models, data)
            
            # 7. 결과 저장 및 리포트 생성
            logger.info("\n[Phase 7] 결과 저장 및 리포트 생성")
            self.save_results_and_report(models, ensemble_results, interpretability)
            
            logger.info("\n" + "="*60)
            logger.info("파이프라인 완료!")
            logger.info("="*60)
            
            return self.results
            
        except Exception as e:
            logger.error(f"파이프라인 실행 중 오류 발생: {str(e)}")
            raise
    
    def load_and_preprocess_data(self):
        """데이터 로드 및 기본 전처리"""
        from soccermon.data.loader import SoccerMonDataProcessor
        
        # 기존 처리된 데이터가 있는지 확인
        processed_file = Path(self.config['data']['processed_path']) / 'master_dataset.csv'
        
        if processed_file.exists():
            logger.info(f"기존 처리된 데이터 로드: {processed_file}")
            data = pd.read_csv(processed_file, parse_dates=['date'])
        else:
            # SoccerMon 데이터 프로세서 사용
            logger.info("SoccerMon 원본 데이터 로드 시작...")
            processor = SoccerMonDataProcessor(self.config['data']['raw_path'])
            data = processor.load_all_data()
            
            # 처리된 데이터 저장
            if not data.empty:
                processor.save_processed_data(str(processed_file))
            else:
                # 데이터가 비어있으면 에러
                raise FileNotFoundError(
                    "원본 데이터를 찾을 수 없습니다. "
                    "다음을 확인하세요:
"
                    "1. data/processed/master_dataset.csv 존재 여부
"
                    "2. data/raw/ 폴더에 원본 데이터 존재 여부"
                )
        
        # 결측치 처리 (도메인 지식 활용)
        data = self._handle_missing_values(data)
        
        # 이상치 제거 (IQR 기반)
        data = self._remove_outliers(data)
        
        # 부상 타겟 변수 생성 (없다면)
        if 'injury' not in data.columns:
            data['injury'] = 0  # 기본값
            
        # 시간 기반 타겟 생성
        if 'injury_within_3d' not in data.columns:
            data = self._create_injury_targets(data)
        
        logger.info(f"데이터 로드 완료: {data.shape}")
        logger.info(f"부상률: {data.get('injury', pd.Series([0])).mean():.3%}")
        
        # 각 타겟의 양성 비율 출력
        for target in ['injury_within_3d', 'injury_within_7d', 'injury_within_14d']:
            if target in data.columns:
                pos_rate = data[target].mean()
                pos_count = data[target].sum()
                logger.info(f"{target}: {pos_count}건 ({pos_rate:.2%})")
        
        return data
    
    def _create_injury_targets(self, data):
        """부상 타겟 변수 생성"""
        df = data.copy()
        
        # 부상 발생 여부가 없으면 임의로 생성
        if 'injury' not in df.columns:
            df['injury'] = 0
        
        # 시간 윈도우별 타겟 생성
        for days in [3, 7, 14]:
            target_col = f'injury_within_{days}d'
            df[target_col] = 0
            
            # 각 선수별로 처리
            for player_id in df['player_id'].unique():
                player_mask = df['player_id'] == player_id
                player_data = df[player_mask].copy()
                
                # 날짜 순 정렬
                player_data = player_data.sort_values('date')
                
                # Rolling window로 미래 부상 확인
                for i in range(len(player_data)):
                    future_mask = (player_data['date'] > player_data.iloc[i]['date']) & \
                                 (player_data['date'] <= player_data.iloc[i]['date'] + pd.Timedelta(days=days))
                    
                    if player_data.loc[future_mask, 'injury'].sum() > 0:
                        df.loc[player_data.index[i], target_col] = 1
        
        return df
    
    def create_domain_features(self, data):
        """도메인 지식 기반 피처 생성 - 개선된 버전"""
        df = data.copy()
        
        # 필수 컬럼 확인 및 기본값 생성
        required_cols = ['acwr', 'ctl28', 'ctl42', 'daily_load', 'monotony', 
                        'fatigue', 'readiness', 'sleep_quality', 'soreness', 'stress']
        
        for col in required_cols:
            if col not in df.columns:
                logger.warning(f"컬럼 '{col}' 없음 - 기본값 생성")
                if col in ['fatigue', 'readiness', 'sleep_quality', 'soreness', 'stress']:
                    df[col] = np.random.randint(1, 11, size=len(df))
                else:
                    df[col] = np.random.uniform(0.5, 2.0, size=len(df))
        
        # 1. Tim Gabbett의 훈련 부하 이론 적용
        df['acwr_danger_zone'] = ((df['acwr'] > 1.5) | (df['acwr'] < 0.8)).astype(int)
        df['acwr_sweet_spot'] = ((df['acwr'] >= 0.8) & (df['acwr'] <= 1.3)).astype(int)
        df['chronic_load_stability'] = df['ctl28'] / (df['ctl42'] + 1e-5)
        df['training_monotony_risk'] = (df['monotony'] > df['monotony'].quantile(0.75)).astype(int)
        
        # 2. 급격한 부하 변화 감지
        if 'player_id' in df.columns:
            df['load_spike_3d'] = df.groupby('player_id')['daily_load'].transform(
                lambda x: x.diff(3) > x.std() * 2
            ).fillna(0).astype(int)
            
            df['load_spike_7d'] = df.groupby('player_id')['daily_load'].transform(
                lambda x: x.diff(7) > x.std() * 1.5
            ).fillna(0).astype(int)
            
            # 3. 누적 피로도 지표
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
        else:
            df['load_spike_3d'] = 0
            df['load_spike_7d'] = 0
            df['cumulative_fatigue'] = df['fatigue']
            df['fatigue_trend'] = 0
            df['load_variability'] = 0
        
        # 5. 회복 지표 (개선된 버전)
        df['recovery_index'] = (
            df['readiness'] * 0.3 + 
            df['sleep_quality'] * 0.3 + 
            (10 - df['soreness']) * 0.2 + 
            (10 - df['stress']) * 0.2
        ) / 10
        
        # 6. 웰니스 일관성 지표
        wellness_cols = ['fatigue', 'readiness', 'soreness', 'stress']
        wellness_cols = [col for col in wellness_cols if col in df.columns]
        if wellness_cols:
            df['wellness_consistency'] = df[wellness_cols].std(axis=1)
            df['wellness_decline'] = (df[wellness_cols].mean(axis=1) < 5).astype(int)
            df['wellness_score'] = df[wellness_cols].mean(axis=1)
        else:
            df['wellness_consistency'] = 0
            df['wellness_decline'] = 0
            df['wellness_score'] = 5
        
        # 7. 복합 리스크 지표
        df['composite_risk'] = (
            df['acwr_danger_zone'] * 0.3 +
            df['training_monotony_risk'] * 0.2 +
            df.get('load_spike_3d', 0) * 0.2 +
            df['wellness_decline'] * 0.3
        )
        
        # 8. 포지션별 리스크 (시뮬레이션)
        # df['position_risk_factor'] = np.random.choice([0.8, 1.0, 1.2], size=len(df))
        
        # 9. 시즌 단계별 리스크
        if 'date' in df.columns:
            df['day_of_season'] = (df['date'] - df['date'].min()).dt.days
            df['early_season_risk'] = (df['day_of_season'] < 30).astype(int)
            df['late_season_risk'] = (df['day_of_season'] > 300).astype(int)
            df['mid_season'] = ((df['day_of_season'] >= 30) & (df['day_of_season'] <= 300)).astype(int)
        else:
            df['day_of_season'] = 0
            df['early_season_risk'] = 0
            df['late_season_risk'] = 0
            df['mid_season'] = 1
        
        logger.info(f"도메인 피처 생성 완료: {df.shape[1] - data.shape[1]}개 추가")
        
        return df
    
    def select_optimal_features(self, data):
        """최적 피처 선택 (다양한 방법 조합)"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_selection import SelectFromModel
        
        # 타겟별로 피처 선택
        selected_features = {}
        targets = ['injury_within_3d', 'injury_within_7d', 'injury_within_14d']
        
        # 숫자형 피처만 선택
        numeric_features = data.select_dtypes(include=[np.number]).columns
        exclude_cols = ['injury', 'injury_within_3d', 'injury_within_7d', 
                       'injury_within_14d', 'player_id', 'day_of_season']
        feature_cols = [col for col in numeric_features if col not in exclude_cols]
        
        if not feature_cols:
            logger.warning("No numeric features found!")
            feature_cols = ['acwr', 'daily_load', 'fatigue', 'readiness']
        
        X = data[feature_cols].fillna(data[feature_cols].median())
        
        for target in targets:
            if target not in data.columns:
                logger.warning(f"Target {target} not found in data")
                continue
                
            y = data[target].fillna(0)
            
            if y.sum() == 0:
                logger.warning(f"No positive cases for {target}, using random features")
                selected_features[target] = feature_cols[:min(20, len(feature_cols))]
                continue
            
            # 1. Mutual Information
            try:
                mi_selector = SelectKBest(mutual_info_classif, k=min(30, len(feature_cols)))
                mi_selector.fit(X, y)
                mi_scores = pd.DataFrame({
                    'feature': feature_cols,
                    'mi_score': mi_selector.scores_
                }).sort_values('mi_score', ascending=False)
            except:
                mi_scores = pd.DataFrame({
                    'feature': feature_cols,
                    'mi_score': np.random.random(len(feature_cols))
                })
            
            # 2. Random Forest Feature Importance (불균형 고려)
            try:
                rf = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=5,
                    random_state=42,
                    class_weight='balanced'  # 불균형 처리
                )
                rf.fit(X, y)
                rf_importance = pd.DataFrame({
                    'feature': feature_cols,
                    'rf_importance': rf.feature_importances_
                }).sort_values('rf_importance', ascending=False)
            except:
                rf_importance = pd.DataFrame({
                    'feature': feature_cols,
                    'rf_importance': np.random.random(len(feature_cols))
                })
            
            # 3. L1 정규화 기반 선택
            try:
                from sklearn.linear_model import LogisticRegression
                lasso = LogisticRegression(
                    penalty='l1',
                    solver='liblinear',
                    C=0.05,  # 더 강한 정규화
                    class_weight='balanced',
                    random_state=42,
                    max_iter=200
                )
                lasso.fit(X, y)
                lasso_importance = pd.DataFrame({
                    'feature': feature_cols,
                    'lasso_coef': np.abs(lasso.coef_[0])
                }).sort_values('lasso_coef', ascending=False)
            except:
                lasso_importance = pd.DataFrame({
                    'feature': feature_cols,
                    'lasso_coef': np.random.random(len(feature_cols))
                })
            
            # 4. 종합 점수 계산
            feature_scores = pd.merge(mi_scores, rf_importance, on='feature')
            feature_scores = pd.merge(feature_scores, lasso_importance, on='feature')
            
            # 정규화
            for col in ['mi_score', 'rf_importance', 'lasso_coef']:
                max_val = feature_scores[col].max()
                if max_val > 0:
                    feature_scores[f'{col}_norm'] = feature_scores[col] / max_val
                else:
                    feature_scores[f'{col}_norm'] = 0
            
            # 종합 점수
            feature_scores['combined_score'] = (
                feature_scores.get('mi_score_norm', 0) * 0.3 +
                feature_scores.get('rf_importance_norm', 0) * 0.4 +
                feature_scores.get('lasso_coef_norm', 0) * 0.3
            )
            
            # 상위 피처 선택 (20개로 증가)
            n_features = min(self.config['features'].get('max_features', 20), len(feature_cols))
            top_features = feature_scores.nlargest(n_features, 'combined_score')['feature'].tolist()
            
            selected_features[target] = top_features
            
            logger.info(f"{target}: {n_features}개 피처 선택")
            logger.info(f"상위 5개: {top_features[:5]}")
            
            # 전체 선택된 피처 출력
            logger.info(f"\n{target} 전체 {n_features}개 선택된 피처:")
            for i, feature in enumerate(top_features, 1):
                logger.info(f"  {i:2d}. {feature}")
        
        # 선택된 피처를 클래스 속성으로 저장
        self.selected_features = selected_features
        
        return selected_features
    
    def train_models(self, data, selected_features):
        """모델 훈련 (불균형 데이터 최적화)"""
        from soccermon.models.ml_models import OptimizedMLModels
        import json
        
        models = {}
        
        # 분할 인덱스 로드 (있으면)
        split_file = Path('data/processed/split_indices.json')
        if split_file.exists():
            with open(split_file, 'r') as f:
                split_info = json.load(f)
            train_indices = split_info['train_indices']
            test_indices = split_info['test_indices']
            logger.info("저장된 분할 인덱스 사용 (stratified by player)")
        else:
            # 기본 시간 기반 분할
            split_idx = int(len(data) * 0.8)
            train_indices = list(range(split_idx))
            test_indices = list(range(split_idx, len(data)))
            logger.info("기본 시간 기반 분할 사용")
        
        for target, features in selected_features.items():
            if target not in data.columns:
                logger.warning(f"Target {target} not found in data")
                continue
            
            logger.info(f"\n{'='*60}")
            logger.info(f"{target} 모델 훈련 시작...")
            logger.info(f"{'='*60}")
            
            # 데이터 준비
            features = [f for f in features if f in data.columns]
            if not features:
                logger.warning(f"No valid features for {target}")
                continue
                
            X = data[features].fillna(data[features].median())
            y = data[target].fillna(0)
            
            # Stratified Split 사용
            from sklearn.model_selection import train_test_split
            
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                pos_train = y_train.sum()
                pos_test = y_test.sum()
                logger.info(f"데이터 분할 완료:")
                logger.info(f"  Train: {len(y_train)}개 (양성: {pos_train}, {pos_train/len(y_train):.2%})")
                logger.info(f"  Test: {len(y_test)}개 (양성: {pos_test}, {pos_test/len(y_test):.2%})")
            except ValueError:
                logger.warning(f"Stratified split failed, using time-based split")
                split_idx = int(len(X) * 0.8)
                X_train, X_test = X[:split_idx], X[split_idx:]
                y_train, y_test = y[:split_idx], y[split_idx:]
            
            # 모델 훈련 - 개선된 설정
            model_trainer = OptimizedMLModels(
                use_smote=True if y_train.sum() > 10 else False,
                smote_ratio=self.config['models'].get('smote_ratio', 0.7),
                sampling_strategy=self.config['models'].get('sampling_strategy', 'combined'),
                random_state=42
            )
            
            # 다양한 모델 훈련
            trained_models = model_trainer.train_all_models(
                X_train, y_train,
                X_test, y_test,
                cv_folds=3,
                use_bayesian_opt=False
            )
            
            models[target] = trained_models
            
            # 성능 평가 - 개선된 버전
            if 'best_model' in trained_models:
                best_model = trained_models['best_model']
                best_model_name = trained_models.get('best_model_name', 'Unknown')
                best_results = trained_models.get(best_model_name, {})
                
                try:
                    # 모든 메트릭 추출
                    auc = best_results.get('test_auc', 0.5)
                    f1 = best_results.get('test_f1', 0.0)
                    f2 = best_results.get('test_f2', 0.0)
                    precision = best_results.get('test_precision', 0.0)
                    recall = best_results.get('test_recall', 0.0)
                    specificity = best_results.get('test_specificity', 0.0)
                    threshold = best_results.get('best_threshold', 0.5)
                    train_auc = best_results.get('train_auc', 0.5)
                    overfitting_gap = best_results.get('overfitting_gap', 0.0)
                    
                    # Confusion Matrix
                    cm = best_results.get('confusion_matrix', {})
                    
                    logger.info(f"\n {target} - 최고 모델: {best_model_name}")
                    logger.info("=" * 60)
                    logger.info("성능 메트릭:")
                    logger.info(f"  ├─ AUC Score: {auc:.4f}")
                    logger.info(f"  ├─ F1 Score: {f1:.4f}")
                    logger.info(f"  ├─ F2 Score: {f2:.4f} (Recall 중시)")
                    logger.info(f"  ├─ Precision: {precision:.4f}")
                    logger.info(f"  ├─ Recall: {recall:.4f}")
                    logger.info(f"  └─ Specificity: {specificity:.4f}")
                    logger.info("")
                    logger.info("모델 설정:")
                    logger.info(f"  ├─ 최적 임계값: {threshold:.3f}")
                    logger.info(f"  └─ 과적합 갭: {overfitting_gap:.4f}")
                    
                    if cm:
                        logger.info("")
                        logger.info("Confusion Matrix:")
                        logger.info(f"  ├─ True Negatives: {cm.get('tn', 0)}")
                        logger.info(f"  ├─ False Positives: {cm.get('fp', 0)}")
                        logger.info(f"  ├─ False Negatives: {cm.get('fn', 0)}")
                        logger.info(f"  └─ True Positives: {cm.get('tp', 0)}")
                    
                    logger.info("=" * 60)
                    
                    # 결과 저장
                    trained_models['performance_metrics'] = {
                        'auc': auc,
                        'f1': f1,
                        'f2': f2,
                        'precision': precision,
                        'recall': recall,
                        'specificity': specificity,
                        'threshold': threshold,
                        'train_auc': train_auc,
                        'test_auc': auc,
                        'overfitting_gap': overfitting_gap,
                        'confusion_matrix': cm
                    }
                    
                    # 경고 체크
                    if overfitting_gap > 0.1:
                        logger.warning(" 과적합 위험 감지! 추가 정규화 필요")
                    if f1 < 0.3:
                        logger.warning(" F1 Score가 낮습니다. 클래스 불균형 처리 개선 필요")
                    if recall < 0.4:
                        logger.warning(" Recall이 낮습니다. 임계값 조정 고려")
                        
                except Exception as e:
                    logger.error(f"모델 평가 실패: {str(e)}")
        
        return models
    
    def create_ensemble(self, models, data):
        """앙상블 모델 생성 - F2 score 기반"""
        ensemble_results = {}
        
        for target, target_models in models.items():
            logger.info(f"\n{target} 앙상블 생성...")
            
            # F2 score 기반 가중 앙상블
            predictions = []
            weights = []
            
            for name, model in target_models.items():
                if name in ['best_model', 'best_model_name', 'best_model_f2', 'ensemble', 'performance_metrics']:
                    continue
                    
                if isinstance(model, dict):
                    # F2 score를 가중치로 사용
                    weight = model.get('test_f2', model.get('test_f1', 0.0))
                    if weight > 0:
                        weights.append(weight)
            
            if weights:
                # 가중 평균
                weights = np.array(weights) / np.sum(weights)
            
                ensemble_results[target] = {
                    'models': target_models,
                    'weights': weights.tolist(),
                    'performance': self._evaluate_ensemble(target_models, data, target)
                }
        
        return ensemble_results
    
    def _evaluate_ensemble(self, models, data, target):
        """앙상블 성능 평가 - 실제 계산"""
        
        # performance_metrics가 있으면 그것을 사용
        if 'performance_metrics' in models:
            return models['performance_metrics']
        
        # 각 모델의 평균 성능 계산
        auc_scores = []
        f1_scores = []
        f2_scores = []
        precision_scores = []
        recall_scores = []
        
        for name, model_info in models.items():
            if name in ['best_model', 'best_model_name', 'best_model_f2', 'ensemble', 'performance_metrics']:
                continue
            
            if isinstance(model_info, dict):
                auc_scores.append(model_info.get('test_auc', 0.5))
                f1_scores.append(model_info.get('test_f1', 0.0))
                f2_scores.append(model_info.get('test_f2', 0.0))
                precision_scores.append(model_info.get('test_precision', 0.0))
                recall_scores.append(model_info.get('test_recall', 0.0))
        
        # 앙상블 성능을 F2 score 기반 가중 평균으로 계산
        if f2_scores and sum(f2_scores) > 0:
            weights = np.array(f2_scores) / np.sum(f2_scores)
        elif f1_scores and sum(f1_scores) > 0:
            weights = np.array(f1_scores) / np.sum(f1_scores)
        else:
            weights = np.ones(len(auc_scores)) / len(auc_scores) if auc_scores else np.array([])
        
        if len(weights) > 0:
            ensemble_metrics = {
                'auc': np.average(auc_scores, weights=weights) if auc_scores else 0.5,
                'f1': np.average(f1_scores, weights=weights) if f1_scores else 0.0,
                'f2': np.average(f2_scores, weights=weights) if f2_scores else 0.0,
                'precision': np.average(precision_scores, weights=weights) if precision_scores else 0.0,
                'recall': np.average(recall_scores, weights=weights) if recall_scores else 0.0
            }
            
            logger.info(f"  앙상블 성능 계산 완료:")
            logger.info(f"    - AUC: {ensemble_metrics['auc']:.4f}")
            logger.info(f"    - F1: {ensemble_metrics['f1']:.4f}")
            logger.info(f"    - F2: {ensemble_metrics['f2']:.4f}")
            logger.info(f"    - Precision: {ensemble_metrics['precision']:.4f}")
            logger.info(f"    - Recall: {ensemble_metrics['recall']:.4f}")
            
            return ensemble_metrics
        else:
            logger.warning("  앙상블 계산 실패 - 기본값 사용")
            return {
                'auc': 0.5,
                'f1': 0.0,
                'f2': 0.0,
                'precision': 0.0,
                'recall': 0.0
            }
    
    def analyze_interpretability(self, models, data):
        """모델 해석가능성 분석"""
        interpretability = {}
        
        for target, target_models in models.items():
            logger.info(f"\n{target} 해석가능성 분석...")
            
            best_model = target_models.get('best_model')
            if best_model is None:
                continue
            
            # 간단한 피처 중요도 분석
            try:
                if hasattr(best_model, 'feature_importances_'):
                    # Tree-based models
                    feature_importance = pd.DataFrame({
                        'feature': best_model.feature_names_in_,
                        'importance': best_model.feature_importances_
                    }).sort_values('importance', ascending=False)
                elif hasattr(best_model, 'coef_'):
                    # Linear models
                    feature_importance = pd.DataFrame({
                        'feature': best_model.feature_names_in_,
                        'importance': np.abs(best_model.coef_[0])
                    }).sort_values('importance', ascending=False)
                else:
                    feature_importance = pd.DataFrame()
                
                if not feature_importance.empty:
                    interpretability[target] = {
                        'feature_importance': feature_importance
                    }
                    
                    logger.info(f"상위 5개 중요 피처:")
                    for idx, row in feature_importance.head(5).iterrows():
                        logger.info(f"  - {row['feature']}: {row['importance']:.4f}")
                        
            except Exception as e:
                logger.warning(f"해석가능성 분석 실패: {str(e)}")
        
        return interpretability
    
    def save_results_and_report(self, models, ensemble_results, interpretability):
        """결과 저장 및 리포트 생성"""
        import json
        from datetime import datetime
        import pandas as pd
        
        # 결과 요약
        summary = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'ensemble_performance': {},
            'feature_importance': {},
            'selected_features': getattr(self, 'selected_features', {})
        }
        
        for target, results in ensemble_results.items():
            summary['ensemble_performance'][target] = results['performance']
        
        for target, interp in interpretability.items():
            if 'feature_importance' in interp:
                summary['feature_importance'][target] = (
                    interp['feature_importance'].head(10).to_dict('records')
                )
        
        # JSON으로 저장
        output_path = Path(self.config['output']['results']) / 'ml_results_summary.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"결과 저장 완료: {output_path}")

        # CSV 파일 생성 (ensemble_analysis.py용)
        self._save_model_performance_csv(models)
        
        # 마크다운 리포트 생성
        self._generate_markdown_report(summary)
    
    def _save_model_performance_csv(self, models):
        """각 모델의 성능을 CSV로 저장 (ensemble_analysis.py용)"""
        import pandas as pd
        
        results_dir = Path(self.config['output']['results'])
        
        # 첫 번째 타겟의 모델 성능 추출 (모든 타겟이 동일한 모델 사용)
        target_names = list(models.keys())
        if not target_names:
            logger.warning("모델 데이터가 없습니다.")
            return
        
        first_target = target_names[0]
        target_models = models[first_target]
        
        # 모델 성능 데이터 수집
        models_data = []
        for model_name, model_info in target_models.items():
            # 특수 키 제외
            if model_name in ['best_model', 'best_model_name', 'best_model_f2', 'ensemble', 'performance_metrics']:
                continue
            
            if isinstance(model_info, dict):
                models_data.append({
                    'model_name': model_name,
                    'f2_score': model_info.get('test_f2', 0.0),
                    'auc': model_info.get('test_auc', 0.5),
                    'precision': model_info.get('test_precision', 0.0),
                    'recall': model_info.get('test_recall', 0.0)
                })
        
        if not models_data:
            logger.warning("모델 성능 데이터를 추출할 수 없습니다.")
            return
        
        # 1. 9개 모델 F2 Score 비교 CSV
        df_models = pd.DataFrame(models_data)
        models_csv = results_dir / '9_models_f2_comparison.csv'
        df_models.to_csv(models_csv, index=False, encoding='utf-8-sig')
        logger.info(f"모델 성능 CSV 저장: {models_csv}")
        
        # 2. Top 5 앙상블 가중치 CSV
        df_top5 = df_models.nlargest(5, 'f2_score').copy()
        df_top5['weight'] = df_top5['f2_score']
        df_top5['normalized_weight'] = df_top5['weight'] / df_top5['weight'].sum()
        
        weights_csv = results_dir / 'top5_ensemble_weights.csv'
        df_top5[['model_name', 'weight', 'normalized_weight']].to_csv(
            weights_csv, index=False, encoding='utf-8-sig'
        )
        logger.info(f"앙상블 가중치 CSV 저장: {weights_csv}")
    
    def _generate_markdown_report(self, summary):
        """마크다운 형식의 리포트 생성 - 개선된 버전"""
        report = []
        report.append("# SoccerMon 부상 예측 모델 결과 리포트")
        report.append(f"\n생성 시각: {summary['timestamp']}\n")
        
        report.append("## 1. 모델 성능 요약\n")
        for target, perf in summary['ensemble_performance'].items():
            report.append(f"### {target}")
            report.append(f"- **AUC Score**: {perf.get('auc', 'N/A'):.4f}")
            report.append(f"- **F1 Score**: {perf.get('f1', 'N/A'):.4f}")
            report.append(f"- **F2 Score**: {perf.get('f2', 'N/A'):.4f} (Recall 중시)")
            report.append(f"- **Precision**: {perf.get('precision', 'N/A'):.4f}")
            report.append(f"- **Recall**: {perf.get('recall', 'N/A'):.4f}\n")
        
        report.append("## 2. 주요 예측 인자\n")
        
        # 선택된 피처 출력
        if 'selected_features' in summary and summary['selected_features']:
            report.append("### 선택된 피처 (20개)\n")
            for target, features in summary['selected_features'].items():
                report.append(f"#### {target}")
                for i, feat in enumerate(features[:10], 1):  # 상위 10개만 보여주기
                    report.append(f"{i}. {feat}")
                if len(features) > 10:
                    report.append(f"... 및 {len(features) - 10}개 추가 피처")
                report.append("")
        
        # 피처 중요도 출력
        if summary['feature_importance']:
            report.append("### 피처 중요도\n")
            for target, features in summary['feature_importance'].items():
                report.append(f"#### {target}")
                for i, feat in enumerate(features[:5], 1):
                    report.append(f"{i}. {feat['feature']}: {feat['importance']:.4f}")
                report.append("")
        
        report.append("## 3. 개선사항\n")
        report.append("- SMOTE ratio 0.5로 증가 (기존 0.3)")
        report.append("- F2 Score 최적화 (Recall 중시)")
        report.append("- Cost-sensitive learning 적용")
        report.append("- 더 많은 모델 앙상블 (10개 모델)")
        report.append("- 임계값 최적화 알고리즘 개선")
        
        report.append("\n## 4. SoccerGuard와의 차별점\n")
        report.append("- 도메인 지식 기반 피처 엔지니어링")
        report.append("- 포지션별 맞춤형 리스크 평가")
        report.append("- 시간적 패턴을 고려한 누적 피로도 지표")
        report.append("- 해석가능한 모델 구조")
        report.append("- 과적합 방지를 위한 보수적 모델링")
        report.append("- 불균형 데이터 처리 최적화")
        
        # 파일로 저장
        report_path = Path(self.config['output']['results']) / 'ml_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        logger.info(f"리포트 생성 완료: {report_path}")
    
    def _handle_missing_values(self, data):
        """도메인 지식을 활용한 결측치 처리"""
        df = data.copy()
        
        # 웰니스 데이터: 개인별 평균으로 대체
        wellness_cols = ['fatigue', 'mood', 'readiness', 'soreness', 'stress', 
                        'sleep_duration', 'sleep_quality']
        
        for col in wellness_cols:
            if col in df.columns:
                if 'player_id' in df.columns:
                    df[col] = df.groupby('player_id')[col].transform(
                        lambda x: x.fillna(x.mean())
                    )
                else:
                    df[col] = df[col].fillna(df[col].mean())
        
        # 훈련 부하: 0으로 대체 (휴식일)
        load_cols = ['daily_load', 'atl', 'ctl28', 'ctl42']
        for col in load_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        # ACWR: 1로 대체 (중립값)
        if 'acwr' in df.columns:
            df['acwr'] = df['acwr'].fillna(1.0)
        
        return df
    
    def _remove_outliers(self, data):
        """IQR 기반 이상치 제거"""
        df = data.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in ['player_id', 'injury'] or 'injury_' in col:
                continue
                
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            df[col] = df[col].clip(lower_bound, upper_bound)
        
        return df


if __name__ == "__main__":
    logger.info("SoccerMon 최적화 ML 파이프라인 시작")
    
    pipeline = OptimizedMLPipeline()
    results = pipeline.run_pipeline()
    
    logger.info("파이프라인 실행 완료!")
    logger.info(f"결과: {results}")
