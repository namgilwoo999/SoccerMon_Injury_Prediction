import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SoccerMonDataProcessor:
    """실제 SoccerMon 데이터 구조를 처리하는 클래스"""
    
    def __init__(self, base_path: str = "data/raw"):
        self.base_path = Path(base_path)
        self.subjective_path = self.base_path / "subjective"
        self.processed_data = None
        
    def load_all_data(self) -> pd.DataFrame:
        """모든 데이터를 통합하여 로드"""
        logger.info("="*60)
        logger.info("SoccerMon 데이터 로드 시작")
        logger.info("="*60)
        
        # 1. 훈련 부하 데이터 로드
        training_load = self.load_training_load_data()
        logger.info(f"훈련 부하 데이터: {training_load.shape}")
        
        # 2. 웰니스 데이터 로드
        wellness = self.load_wellness_data()
        logger.info(f"웰니스 데이터: {wellness.shape}")
        
        # 3. 부상 데이터 로드
        injuries = self.load_injury_data()
        logger.info(f"부상 데이터: {len(injuries)} 건")
        
        # 4. 데이터 통합
        master_data = self.merge_all_data(training_load, wellness, injuries)
        logger.info(f"통합 데이터: {master_data.shape}")
        
        # 5. 부상 타겟 생성
        master_data = self.create_injury_targets(master_data, injuries)
        
        self.processed_data = master_data
        return master_data
    
    def load_training_load_data(self) -> pd.DataFrame:
        """훈련 부하 데이터 로드 (Wide to Long format 변환)"""
        training_path = self.subjective_path / "training-load"
        
        metrics = ['acwr', 'atl', 'ctl28', 'ctl42', 'daily_load', 'monotony', 'strain']
        all_data = []
        
        for metric in metrics:
            file_path = training_path / f"{metric}.csv"
            if file_path.exists():
                df = self.load_wide_csv_as_long(file_path, metric)
                all_data.append(df)
                logger.info(f"  {metric}.csv 로드: {len(df)} records")
        
        # 모든 메트릭을 병합
        if all_data:
            merged = all_data[0]
            for df in all_data[1:]:
                # team, player_uuid 컬럼은 첫 번째 데이터프레임에서만 유지
                cols_to_merge = ['date', 'player_id', metric] 
                # metric 컬럼명 찾기
                metric_col = [col for col in df.columns if col not in ['date', 'player_id', 'team', 'player_uuid']]
                if metric_col:
                    cols_to_merge = ['date', 'player_id'] + metric_col
                    df_to_merge = df[cols_to_merge]
                    merged = pd.merge(merged, df_to_merge, on=['date', 'player_id'], how='outer')
            return merged
        else:
            return pd.DataFrame()
    
    def load_wellness_data(self) -> pd.DataFrame:
        wellness_path = self.subjective_path / "wellness"
        
        metrics = ['fatigue', 'mood', 'readiness', 'sleep_duration', 
                  'sleep_quality', 'soreness', 'stress']
        all_data = []
        
        for metric in metrics:
            file_path = wellness_path / f"{metric}.csv"
            if file_path.exists():
                df = self.load_wide_csv_as_long(file_path, metric)
                all_data.append(df)
                logger.info(f"  {metric}.csv 로드: {len(df)} records")
        
        # 모든 메트릭을 병합
        if all_data:
            merged = all_data[0]
            for df in all_data[1:]:
                # team, player_uuid 컬럼은 첫 번째 데이터프레임에서만 유지
                cols_to_merge = ['date', 'player_id', metric] 
                # metric 컬럼명 찾기
                metric_col = [col for col in df.columns if col not in ['date', 'player_id', 'team', 'player_uuid']]
                if metric_col:
                    cols_to_merge = ['date', 'player_id'] + metric_col
                    df_to_merge = df[cols_to_merge]
                    merged = pd.merge(merged, df_to_merge, on=['date', 'player_id'], how='outer')
            return merged
        else:
            return pd.DataFrame()
    
    def load_injury_data(self) -> pd.DataFrame:
        injury_path = self.subjective_path / "injury"
        
        # injury 폴더의 CSV 파일들 확인
        injury_files = list(injury_path.glob("*.csv"))
        
        all_injuries = []
        
        for injury_file in injury_files:
            try:
                # CSV 파일 로드
                df = pd.read_csv(injury_file)
                
                # Wide format인 경우 Long format으로 변환
                if any(col.startswith('Team') for col in df.columns):
                    df = self.load_wide_csv_as_long(injury_file, 'injury_reported')
                    
                all_injuries.append(df)
                logger.info(f"  부상 파일 로드: {injury_file.name} - {len(df)} records")
            except Exception as e:
                logger.warning(f"  부상 파일 로드 실패 {injury_file.name}: {str(e)}")
        
        if all_injuries:
            # 모든 부상 데이터 병합
            injury_df = pd.concat(all_injuries, ignore_index=True)
            
            # injury_reported가 있는 행만 필터링 (실제 부상)
            if 'injury_reported' in injury_df.columns:
                injury_df = injury_df[injury_df['injury_reported'].notna()]
            
            return injury_df
        else:
            logger.warning("부상 데이터 파일이 없습니다")
            return pd.DataFrame()
    
    def load_wide_csv_as_long(self, file_path: Path, metric_name: str) -> pd.DataFrame:
        #Wide format CSV를 Long format으로 변환
        df = pd.read_csv(file_path)
        
        # Date 컬럼 확인
        date_col = 'Date' if 'Date' in df.columns else df.columns[0]
        
        # Player 컬럼들 찾기 (TeamA-, TeamB- 로 시작하는 컬럼들)
        player_cols = [col for col in df.columns 
                      if col.startswith('TeamA-') or col.startswith('TeamB-')]
        
        if not player_cols:
            logger.warning(f"{file_path.name}에 선수 컬럼이 없습니다")
            return pd.DataFrame()
        
        # Wide to Long 변환
        df_long = pd.melt(df,
                         id_vars=[date_col],
                         value_vars=player_cols,
                         var_name='player_id',
                         value_name=metric_name)
        
        # date 컬럼명 표준화
        df_long.rename(columns={date_col: 'date'}, inplace=True)
        
        # date 형식 변환 (DD.MM.YYYY -> datetime)
        df_long['date'] = pd.to_datetime(df_long['date'], format='%d.%m.%Y', errors='coerce')
        
        # 결측치 제거 (빈 값이나 0이 아닌 실제 결측치만)
        if metric_name in ['fatigue', 'mood', 'readiness', 'soreness', 'stress', 
                          'sleep_quality', 'sleep_duration']:
            # 웰니스 데이터는 빈 문자열을 NaN으로
            df_long[metric_name] = pd.to_numeric(df_long[metric_name], errors='coerce')
        else:
            # 훈련 부하 데이터는 0도 유효한 값
            df_long[metric_name] = pd.to_numeric(df_long[metric_name], errors='coerce')
        
        # player_id 간소화 (팀명은 유지)
        df_long['team'] = df_long['player_id'].str.split('-').str[0]
        df_long['player_uuid'] = df_long['player_id']
        
        # 간단한 player_id 생성
        unique_players = df_long['player_id'].unique()
        player_mapping = {player: f"{player.split('-')[0]}_{i+1}" 
                         for i, player in enumerate(unique_players)}
        df_long['player_id'] = df_long['player_id'].map(player_mapping)
        
        return df_long
    
    def merge_all_data(self, training_load: pd.DataFrame, 
                      wellness: pd.DataFrame, 
                      injuries: pd.DataFrame) -> pd.DataFrame:
        #모든 데이터를 통합
        
        # 훈련 부하와 웰니스 데이터 병합
        if not training_load.empty and not wellness.empty:
            # 공통 컬럼 확인
            common_cols = ['date', 'player_id']
            
            # team과 player_uuid는 한쪽에만 유지
            if 'team' in training_load.columns:
                if 'team' in wellness.columns:
                    wellness = wellness.drop(columns=['team'], errors='ignore')
            if 'player_uuid' in training_load.columns:
                if 'player_uuid' in wellness.columns:
                    wellness = wellness.drop(columns=['player_uuid'], errors='ignore')
            
            master = pd.merge(training_load, wellness, 
                            on=common_cols, 
                            how='outer')
        elif not training_load.empty:
            master = training_load
        elif not wellness.empty:
            master = wellness
        else:
            # 둘 다 비어있으면 빈 DataFrame
            return pd.DataFrame()
        
        # 날짜 순으로 정렬
        master = master.sort_values(['player_id', 'date'])
        
        # injury 플래그 초기화
        master['injury'] = 0
        
        # 부상 데이터 매칭
        if not injuries.empty:
            # 부상 날짜와 선수 매칭
            for _, injury_row in injuries.iterrows():
                if 'date' in injury_row and 'player_id' in injury_row:
                    mask = (master['player_id'] == injury_row['player_id']) & \
                          (master['date'] == injury_row['date'])
                    master.loc[mask, 'injury'] = 1
                elif 'date' in injury_row and 'player_uuid' in injury_row:
                    # player_uuid로 매칭
                    mask = (master['player_uuid'] == injury_row['player_uuid']) & \
                          (master['date'] == injury_row['date'])
                    master.loc[mask, 'injury'] = 1
        
        return master
    
    def create_injury_targets(self, master_data: pd.DataFrame, 
                            injuries: pd.DataFrame) -> pd.DataFrame:
        #부상 타겟 변수 생성
        
        df = master_data.copy()
        
        # 시간 윈도우별 타겟 생성 (3일, 7일, 14일 이내 부상)
        for days in [3, 7, 14]:
            target_col = f'injury_within_{days}d'
            df[target_col] = 0
            
            # 각 선수별로 처리
            for player_id in df['player_id'].unique():
                player_mask = df['player_id'] == player_id
                player_df = df[player_mask].copy()
                player_df = player_df.sort_values('date')
                
                # 각 행에 대해 미래 부상 확인
                for idx in player_df.index:
                    current_date = df.loc[idx, 'date']
                    
                    # NaT 체크
                    if pd.isna(current_date):
                        continue
                        
                    future_mask = (df['player_id'] == player_id) & \
                                 (df['date'] > current_date) & \
                                 (df['date'] <= current_date + pd.Timedelta(days=days))
                    
                    if df.loc[future_mask, 'injury'].sum() > 0:
                        df.loc[idx, target_col] = 1
        
        # 부상 통계
        logger.info(f"부상 통계:")
        logger.info(f"  - 전체 부상: {df['injury'].sum()}건 ({df['injury'].mean():.2%})")
        logger.info(f"  - 3일 이내: {df['injury_within_3d'].sum()}건 ({df['injury_within_3d'].mean():.2%})")
        logger.info(f"  - 7일 이내: {df['injury_within_7d'].sum()}건 ({df['injury_within_7d'].mean():.2%})")
        logger.info(f"  - 14일 이내: {df['injury_within_14d'].sum()}건 ({df['injury_within_14d'].mean():.2%})")
        
        return df
    
    def save_processed_data(self, output_path: str = "data/processed/master_dataset.csv"):
        """처리된 데이터 저장"""
        if self.processed_data is not None:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 데이터 정리
            df_to_save = self.processed_data.copy()
            
            # player_uuid 컬럼 제거 (용량 절약)
            if 'player_uuid' in df_to_save.columns:
                df_to_save = df_to_save.drop(columns=['player_uuid'])
            
            df_to_save.to_csv(output_file, index=False)
            logger.info(f"데이터 저장 완료: {output_file}")
            logger.info(f"  저장된 데이터: {df_to_save.shape}")
        else:
            logger.warning("저장할 데이터가 없습니다")


# 독립 실행 가능
if __name__ == "__main__":
    processor = SoccerMonDataProcessor("data/raw")
    master_data = processor.load_all_data()
    
    if not master_data.empty:
        print(f"\n데이터 로드 성공!")
        print(f"Shape: {master_data.shape}")
        print(f"컬럼: {master_data.columns.tolist()}")
        print(f"\n데이터 샘플:")
        print(master_data.head())
        
        # 기본 통계
        print(f"\n기본 통계:")
        print(f"- 선수 수: {master_data['player_id'].nunique()}")
        print(f"- 날짜 범위: {master_data['date'].min()} ~ {master_data['date'].max()}")
        print(f"- 총 레코드: {len(master_data)}")
        
        # 처리된 데이터 저장
        processor.save_processed_data()
    else:
        print("데이터 로드 실패")
