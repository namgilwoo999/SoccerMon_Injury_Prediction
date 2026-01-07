#부상 데이터 검증 및 수정 스크립트

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_and_fix_injury_data():
    #부상 데이터 확인 및 수정
    
    # 1. 부상 데이터 로드
    injury_file = Path("data/raw/subjective/injury/injury.csv")
    injury_df = pd.read_csv(injury_file)
    
    logger.info(f"부상 데이터 로드: {len(injury_df)} 건")
    logger.info(f"컬럼: {injury_df.columns.tolist()}")
    logger.info(f"샘플:\n{injury_df.head()}")
    
    # 2. master_dataset 로드
    master_file = Path("data/processed/master_dataset.csv")
    master_df = pd.read_csv(master_file, parse_dates=['date'])
    
    logger.info(f"\n마스터 데이터 로드: {len(master_df)} 건")
    logger.info(f"기존 부상률: {master_df['injury'].mean():.3%}")
    
    # 3. 부상 데이터 매칭
    master_df['injury'] = 0  # 초기화
    
    # 부상 데이터의 날짜 형식 변환 (DD.MM.YYYY)
    injury_df['date'] = pd.to_datetime(injury_df['timestamp'], format='%d.%m.%Y', errors='coerce')
    injury_df['player_id'] = injury_df['player_name']
    
    # 선수 ID 매핑 (UUID를 간단한 ID로)
    unique_players = master_df['player_id'].unique()
    uuid_to_simple = {}
    
    for simple_id in unique_players:
        # TeamA_1, TeamB_1 형태에서 팀명 추출
        team = simple_id.split('_')[0]
        
        # 해당 팀의 UUID 매핑
        for uuid in injury_df['player_id'].unique():
            if uuid.startswith(team):
                if uuid not in uuid_to_simple:
                    uuid_to_simple[uuid] = f"{team}_{len([k for k in uuid_to_simple.keys() if k.startswith(team)]) + 1}"
    
    injury_df['simple_id'] = injury_df['player_id'].map(uuid_to_simple)
    
    # 부상 매칭
    matched = 0
    for _, injury_row in injury_df.iterrows():
        if pd.isna(injury_row['date']) or pd.isna(injury_row['simple_id']):
            continue
            
        # 날짜와 선수 ID로 매칭
        mask = (master_df['date'] == injury_row['date']) & \
               (master_df['player_id'] == injury_row['simple_id'])
        
        if mask.sum() > 0:
            master_df.loc[mask, 'injury'] = 1
            matched += 1
    
    logger.info(f"\n부상 매칭 완료: {matched}/{len(injury_df)} 건")
    logger.info(f"새로운 부상률: {master_df['injury'].mean():.3%}")
    
    # 4. 시간 윈도우별 타겟 생성
    for days in [3, 7, 14]:
        target_col = f'injury_within_{days}d'
        master_df[target_col] = 0
        
        # 각 선수별로 처리
        for player_id in master_df['player_id'].unique():
            player_mask = master_df['player_id'] == player_id
            player_df = master_df[player_mask].copy()
            player_df = player_df.sort_values('date')
            
            # 미래 부상 확인
            for idx in player_df.index:
                current_date = master_df.loc[idx, 'date']
                
                # 미래 days일 이내 부상 확인
                future_mask = (master_df['player_id'] == player_id) & \
                             (master_df['date'] > current_date) & \
                             (master_df['date'] <= current_date + pd.Timedelta(days=days)) & \
                             (master_df['injury'] == 1)
                
                if future_mask.sum() > 0:
                    master_df.loc[idx, target_col] = 1
        
        logger.info(f"{days}일 이내 부상률: {master_df[target_col].mean():.3%} ({master_df[target_col].sum()}건)")
    
    # 5. 저장
    output_file = Path("data/processed/master_dataset_fixed.csv")
    master_df.to_csv(output_file, index=False)
    logger.info(f"\n수정된 데이터 저장: {output_file}")
    
    # 원본 파일 백업 후 교체
    import shutil
    shutil.copy2(master_file, master_file.with_suffix('.csv.bak'))
    shutil.copy2(output_file, master_file)
    logger.info(f"원본 파일 교체 완료")
    
    return master_df

if __name__ == "__main__":
    logger.info("="*60)
    logger.info("부상 데이터 검증 및 수정 시작")
    logger.info("="*60)
    
    fixed_df = check_and_fix_injury_data()
    
    # 검증
    logger.info("\n="*60)
    logger.info("최종 검증")
    logger.info("="*60)
    logger.info(f"총 레코드: {len(fixed_df)}")
    logger.info(f"부상 케이스: {fixed_df['injury'].sum()}")
    logger.info(f"3일 타겟: {fixed_df['injury_within_3d'].sum()}")
    logger.info(f"7일 타겟: {fixed_df['injury_within_7d'].sum()}")
    logger.info(f"14일 타겟: {fixed_df['injury_within_14d'].sum()}")
