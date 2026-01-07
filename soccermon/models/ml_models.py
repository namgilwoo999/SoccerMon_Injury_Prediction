import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (roc_auc_score, f1_score, precision_score, 
                           recall_score, fbeta_score, make_scorer,
                           precision_recall_curve, confusion_matrix)
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')
import joblib
from pathlib import Path

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                            GradientBoostingClassifier, VotingClassifier,
                            HistGradientBoostingClassifier)
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

logger = logging.getLogger(__name__)


class OptimizedMLModels:
    def __init__(self, use_smote: bool = True, smote_ratio: float = 0.7,
                 sampling_strategy: str = 'combined', random_state: int = 42):
        """
        Args:
            use_smote: SMOTE 사용 여부
            smote_ratio: SMOTE 비율 (기본 0.5로 증가)
            sampling_strategy: 'smote', 'adasyn', 'borderline', 'combined'
            random_state: 난수 시드
        """
        self.use_smote = use_smote
        self.smote_ratio = smote_ratio
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.models = {}
        self.results = {}
        
    def get_class_weights(self, y_train):
        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        
        # 부상 클래스에 추가 가중치
        if len(classes) == 2:
            # 소수 클래스(1)에 더 많은 가중치
            weights[1] = weights[1] * 2.0  # 2배 가중치
        
        return dict(zip(classes, weights))
        
    def get_regularized_models(self) -> Dict:
        models = {
            # 1. 로지스틱 회귀 (class_weight='balanced' + L2)
            'logistic_balanced': LogisticRegression(
                penalty='l2',
                C=0.05,  # 더 강한 정규화
                class_weight='balanced',
                max_iter=2000,
                solver='lbfgs',
                random_state=self.random_state
            ),
            
            # 2. 로지스틱 회귀 (커스텀 class_weight)
            'logistic_weighted': LogisticRegression(
                penalty='l2',
                C=0.1,
                class_weight={0: 1, 1: 10},  # 부상에 10배 가중치
                max_iter=2000,
                solver='liblinear',
                random_state=self.random_state
            ),
            
            # 3. 랜덤 포레스트 (불균형 최적화)
            'random_forest_balanced': RandomForestClassifier(
                n_estimators=500,
                max_depth=7,  # 약간 더 깊게
                min_samples_split=15,
                min_samples_leaf=7,
                max_features='sqrt',
                max_samples=0.8,
                class_weight='balanced_subsample',  # 서브샘플별 균형
                random_state=self.random_state,
                n_jobs=-1
            ),
            
            # 4. Extra Trees (불균형 최적화)
            'extra_trees_balanced': ExtraTreesClassifier(
                n_estimators=500,
                max_depth=7,
                min_samples_split=15,
                min_samples_leaf=7,
                max_features='sqrt',
                max_samples=0.8,
                class_weight='balanced',
                bootstrap=True,
                random_state=self.random_state,
                n_jobs=-1
            ),
            
            # 5. LightGBM (불균형 최적화)
            'lightgbm_balanced': LGBMClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.03,
                num_leaves=31,
                min_child_samples=10,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                # is_unbalance=True 제거 (scale_pos_weight와 충돌)
                scale_pos_weight=15,  # 양성 클래스 가중치 증가 (10->15)
                random_state=self.random_state,
                verbose=-1
            ),
            
            # 6. XGBoost (불균형 최적화)
            'xgboost_balanced': XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                scale_pos_weight=10,  # 양성 클래스 가중치
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=self.random_state
            ),
            
            # 7. Gradient Boosting (불균형 최적화)
            'gradient_boosting_balanced': GradientBoostingClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                min_samples_split=15,
                min_samples_leaf=7,
                subsample=0.8,
                max_features='sqrt',
                random_state=self.random_state
            ),
            
            # 8. Histogram Gradient Boosting (빠르고 효율적)
            'hist_gradient_balanced': HistGradientBoostingClassifier(
                max_iter=300,
                max_depth=6,
                learning_rate=0.05,
                min_samples_leaf=10,
                l2_regularization=0.1,
                class_weight='balanced',
                random_state=self.random_state
            ),
            
            # 9. Neural Network (불균형 최적화)
            'mlp_balanced': MLPClassifier(
                hidden_layer_sizes=(64, 32, 16),  # 더 큰 네트워크
                activation='relu',
                solver='adam',
                alpha=0.5,
                learning_rate='adaptive',
                learning_rate_init=0.001,
                max_iter=1000,
                early_stopping=True,
                validation_fraction=0.2,
                n_iter_no_change=20,
                random_state=self.random_state
            )
        }
        
        return models
    
    def get_sampler(self):
        """샘플링 전략에 따른 sampler 반환"""
        if self.sampling_strategy == 'smote':
            return SMOTE(
                sampling_strategy=self.smote_ratio,
                k_neighbors=5,
                random_state=self.random_state
            )
        elif self.sampling_strategy == 'adasyn':
            try:
                return ADASYN(
                    sampling_strategy=self.smote_ratio,
                    n_neighbors=5,
                    random_state=self.random_state
                )
            except:
                return SMOTE(
                    sampling_strategy=self.smote_ratio,
                    k_neighbors=5,
                    random_state=self.random_state
                )
        elif self.sampling_strategy == 'borderline':
            return BorderlineSMOTE(
                sampling_strategy=self.smote_ratio,
                k_neighbors=5,
                kind='borderline-1',
                random_state=self.random_state
            )
        elif self.sampling_strategy == 'combined':
            return SMOTETomek(
                smote=SMOTE(
                    sampling_strategy=self.smote_ratio,
                    k_neighbors=5,
                    random_state=self.random_state
                ),
                random_state=self.random_state
            )
        else:
            return SMOTE(
                sampling_strategy=self.smote_ratio,
                random_state=self.random_state
            )
    
    def create_pipeline(self, model, use_scaling: bool = True) -> Pipeline:
        #모델 파이프라인 생성
        steps = []
        
        # 스케일링
        if use_scaling:
            steps.append(('scaler', RobustScaler()))
        
        # 샘플링
        if self.use_smote:
            steps.append(('sampler', self.get_sampler()))
            
        # 모델
        steps.append(('classifier', model))
        
        # 파이프라인 생성
        if self.use_smote:
            return ImbPipeline(steps)
        else:
            return Pipeline(steps)
    
    def find_optimal_threshold_f2(self, y_true, y_scores) -> float:
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
        
        # F2 score 계산 (Recall에 2배 가중치)
        beta = 2.0
        f2_scores = ((1 + beta**2) * precisions * recalls) / (beta**2 * precisions + recalls)
        
        # NaN 제거
        f2_scores = np.nan_to_num(f2_scores)
        
        # 최고 F2 score의 임계값
        best_idx = np.argmax(f2_scores[:-1])  # 마지막 값은 threshold=1.0
        best_threshold = thresholds[best_idx]
        
        # 너무 낮은 임계값 방지
        best_threshold = max(0.2, min(0.8, best_threshold))
        
        return best_threshold
    
    def _find_optimal_threshold(self, y_true, y_scores) -> float:
        """F1 score 기준 최적 임계값 찾기 (기존 메서드와 호환성)"""
        return self.find_optimal_threshold_f2(y_true, y_scores)
    
    def train_all_models(self, X_train, y_train, X_test, y_test,
                        cv_folds: int = 5, 
                        use_bayesian_opt: bool = False) -> Dict:
        """모든 모델 훈련 및 평가 - 개선된 버전"""
        
        models = self.get_regularized_models()
        results = {}
        
        # 양성 클래스 비율 확인
        n_positive_train = y_train.sum()
        n_positive_test = y_test.sum()
        pos_ratio_train = n_positive_train / len(y_train)
        pos_ratio_test = n_positive_test / len(y_test)
        
        logger.info(f"Train set - Positive: {n_positive_train} ({pos_ratio_train:.2%})")
        logger.info(f"Test set - Positive: {n_positive_test} ({pos_ratio_test:.2%})")
        
        # CV 설정
        max_folds = min(cv_folds, max(2, n_positive_train // 10))
        skf = StratifiedKFold(n_splits=max_folds, shuffle=True, 
                             random_state=self.random_state)
        
        logger.info(f"총 {len(models)}개 모델 훈련 시작...")
        
        best_f2_score = -np.inf
        best_model = None
        best_name = None
        
        # F2 scorer 생성
        f2_scorer = make_scorer(fbeta_score, beta=2)
        
        for name, model in models.items():
            logger.info(f"  훈련 중: {name}")
            
            try:
                # 파이프라인 생성
                pipeline = self.create_pipeline(
                    model, 
                    use_scaling=(name not in ['random_forest_balanced', 'extra_trees_balanced'])
                )
                
                # Cross-validation (F2 score 기준)
                if n_positive_train > max_folds:
                    try:
                        cv_scores = cross_val_score(
                            pipeline, X_train, y_train,
                            cv=skf, scoring=f2_scorer,
                            n_jobs=-1
                        )
                    except:
                        cv_scores = np.array([0.5])
                else:
                    cv_scores = np.array([0.5])
                
                # 모델 훈련
                pipeline.fit(X_train, y_train)
                
                # 예측
                y_train_pred_proba = pipeline.predict_proba(X_train)[:, 1]
                y_test_pred_proba = pipeline.predict_proba(X_test)[:, 1]
                
                # 평가 메트릭 (확률 기반)
                train_auc = roc_auc_score(y_train, y_train_pred_proba)
                test_auc = roc_auc_score(y_test, y_test_pred_proba)
                
                # F2 기준 최적 임계값 찾기
                best_threshold = self.find_optimal_threshold_f2(y_test, y_test_pred_proba)
                
                # 최적 임계값으로 이진 예측
                y_test_pred = (y_test_pred_proba > best_threshold).astype(int)
                
                # 메트릭 계산
                test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
                test_f2 = fbeta_score(y_test, y_test_pred, beta=2, zero_division=0)
                test_precision = precision_score(y_test, y_test_pred, zero_division=0)
                test_recall = recall_score(y_test, y_test_pred, zero_division=0)
                
                # Confusion Matrix
                tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                
                # 과적합 체크
                overfitting_gap = train_auc - test_auc
                
                # 결과 저장
                results[name] = {
                    'model': pipeline,
                    'cv_scores': cv_scores,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'train_auc': train_auc,
                    'test_auc': test_auc,
                    'test_f1': test_f1,
                    'test_f2': test_f2,
                    'test_precision': test_precision,
                    'test_recall': test_recall,
                    'test_specificity': specificity,
                    'overfitting_gap': overfitting_gap,
                    'best_threshold': best_threshold,
                    'confusion_matrix': {
                        'tn': int(tn), 'fp': int(fp),
                        'fn': int(fn), 'tp': int(tp)
                    }
                }
                
                # 로깅
                logger.info(f"    AUC: {test_auc:.4f}, F1: {test_f1:.4f}, "
                          f"F2: {test_f2:.4f}, Precision: {test_precision:.4f}, "
                          f"Recall: {test_recall:.4f}, Threshold: {best_threshold:.3f}")
                
                # 최고 모델 업데이트 (F2 score 기준)
                if test_f2 > best_f2_score and overfitting_gap < 0.15:
                    best_f2_score = test_f2
                    best_model = pipeline
                    best_name = name
                    
            except Exception as e:
                logger.error(f"{name} 훈련 실패: {str(e)}")
                results[name] = {
                    'model': None,
                    'cv_scores': np.array([0.0]),
                    'cv_mean': 0.0,
                    'cv_std': 0.0,
                    'train_auc': 0.5,
                    'test_auc': 0.5,
                    'test_f1': 0.0,
                    'test_f2': 0.0,
                    'test_precision': 0.0,
                    'test_recall': 0.0,
                    'test_specificity': 0.0,
                    'overfitting_gap': 0.0,
                    'best_threshold': 0.5
                }
                continue
        
        # 최고 모델 저장
        if best_model is not None:
            results['best_model'] = best_model
            results['best_model_name'] = best_name
            results['best_model_f2'] = best_f2_score
            logger.info(f"\n최고 모델 (F2 기준): {best_name} (F2={best_f2_score:.4f})")
        
        # 앙상블 모델 생성
        ensemble_model = self.create_weighted_ensemble(results)
        if ensemble_model is not None:
            results['ensemble'] = ensemble_model
        
        self.results = results
        return results
    
    def create_ensemble(self, model_results: Dict) -> Optional[VotingClassifier]:
        return self.create_weighted_ensemble(model_results)
    
    def create_weighted_ensemble(self, model_results: Dict) -> Optional[VotingClassifier]:
        try:
            # F2 score 기준 상위 5개 모델 선택
            sorted_models = sorted(
                [(k, v) for k, v in model_results.items() 
                 if k not in ['best_model', 'best_model_name', 'best_model_f2', 'ensemble']
                 and v['model'] is not None],
                key=lambda x: x[1].get('test_f2', x[1].get('test_f1', 0)),
                reverse=True
            )[:5]
            
            if len(sorted_models) < 3:
                logger.warning("앙상블을 위한 충분한 모델이 없습니다.")
                return None
            
            # Voting Classifier 생성 (F2 score로 가중)
            estimators = [(name, result['model']) for name, result in sorted_models]
            weights = [result.get('test_f2', result.get('test_f1', 0)) for _, result in sorted_models]
            
            # 가중치 정규화
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w/total_weight for w in weights]
            else:
                weights = [1/len(weights) for _ in weights]
            
            ensemble = VotingClassifier(
                estimators=estimators,
                voting='soft',
                weights=weights
            )
            
            logger.info(f"가중 앙상블 모델 생성 완료: {len(estimators)}개 모델")
            logger.info(f"모델 가중치: {dict(zip([n for n, _ in estimators], weights))}")
            
            return ensemble
            
        except Exception as e:
            logger.error(f"앙상블 생성 실패: {str(e)}")
            return None
    
    def get_model_comparison(self) -> pd.DataFrame:
        """모델 비교 테이블 생성 - 개선된 버전"""
        if not self.results:
            return pd.DataFrame()
        
        comparison = []
        for name, result in self.results.items():
            if name in ['best_model', 'best_model_name', 'best_model_f2', 'ensemble']:
                continue
                
            comparison.append({
                'Model': name,
                'CV F2': f"{result['cv_mean']:.4f} ± {result['cv_std']:.4f}",
                'Train AUC': f"{result['train_auc']:.4f}",
                'Test AUC': f"{result['test_auc']:.4f}",
                'F1 Score': f"{result['test_f1']:.4f}",
                'F2 Score': f"{result.get('test_f2', 0):.4f}",
                'Precision': f"{result['test_precision']:.4f}",
                'Recall': f"{result['test_recall']:.4f}",
                'Specificity': f"{result.get('test_specificity', 0):.4f}",
                'Threshold': f"{result['best_threshold']:.3f}",
                'Overfitting Gap': f"{result['overfitting_gap']:.4f}"
            })
        
        df = pd.DataFrame(comparison)
        df = df.sort_values('F2 Score', ascending=False)
        
        return df
    
    def save_models(self, save_path: str):
        
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for name, result in self.results.items():
            if name in ['best_model', 'best_model_name', 'best_model_f2']:
                continue
                
            if result.get('model') is not None:
                model_path = save_dir / f"{name}.joblib"
                joblib.dump(result['model'], model_path)
                logger.info(f"모델 저장: {model_path}")
        
        # 앙상블 모델 저장
        if 'ensemble' in self.results and self.results['ensemble'] is not None:
            ensemble_path = save_dir / "ensemble_model.joblib"
            joblib.dump(self.results['ensemble'], ensemble_path)
            logger.info(f"앙상블 모델 저장: {ensemble_path}")
        
        # 베스트 모델 저장
        if 'best_model' in self.results and self.results['best_model'] is not None:
            best_path = save_dir / "best_model.joblib"
            joblib.dump(self.results['best_model'], best_path)
            logger.info(f"베스트 모델 저장: {best_path}")
        
        logger.info(f"\n모델 저장 완료: {save_dir}")
        
        return save_dir
