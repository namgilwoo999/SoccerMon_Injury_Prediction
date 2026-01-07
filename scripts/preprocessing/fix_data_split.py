#부상 데이터 시간 분포 확인 및 재분할 스크립트

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_and_fix_injury_distribution():
    #부상 데이터 시간 분포 분석 및 수정
    
    # 데이터 로드
    df = pd.read_csv("data/processed/master_dataset.csv", parse_dates=['date'])
    df = df.sort_values(['player_id', 'date'])
    
    logger.info("="*60)
    logger.info("부상 데이터 시간 분포 분석")
    logger.info("="*60)
    
    # 부상 시간 분포 확인
    injury_dates = df[df['injury'] == 1]['date']
    logger.info(f"\n부상 발생 날짜 범위:")
    logger.info(f"첫 부상: {injury_dates.min()}")
    logger.info(f"마지막 부상: {injury_dates.max()}")
    logger.info(f"전체 데이터 범위: {df['date'].min()} ~ {df['date'].max()}")
    
    # 월별 부상 통계
    df['year_month'] = df['date'].dt.to_period('M')
    monthly_injuries = df.groupby('year_month')['injury'].sum()
    logger.info(f"\n월별 부상 건수:")
    for month, count in monthly_injuries[monthly_injuries > 0].items():
        logger.info(f"  {month}: {count}건")
    
    # 선수별 층화 추출로 재분할
    logger.info("\n="*60)
    logger.info("선수별 층화 추출로 데이터 재분할")
    logger.info("="*60)
    
    # 각 선수별로 80/20 분할
    train_indices = []
    test_indices = []
    
    for player_id in df['player_id'].unique():
        player_data = df[df['player_id'] == player_id]
        player_indices = player_data.index.tolist()
        
        # 랜덤 셔플 후 80/20 분할
        np.random.seed(42)
        np.random.shuffle(player_indices)
        
        split_idx = int(len(player_indices) * 0.8)
        train_indices.extend(player_indices[:split_idx])
        test_indices.extend(player_indices[split_idx:])
    
    # 새로운 train/test 세트 생성
    train_df = df.loc[train_indices].sort_index()
    test_df = df.loc[test_indices].sort_index()
    
    logger.info(f"Train set: {len(train_df)} rows")
    logger.info(f"Test set: {len(test_df)} rows")
    
    # 부상 분포 확인
    targets = ['injury', 'injury_within_3d', 'injury_within_7d', 'injury_within_14d']
    for target in targets:
        if target in df.columns:
            train_pos = train_df[target].sum()
            test_pos = test_df[target].sum()
            logger.info(f"\n{target}:")
            logger.info(f"  Train: {train_pos} ({train_pos/len(train_df):.3%})")
            logger.info(f"  Test: {test_pos} ({test_pos/len(test_df):.3%})")
    
    # 분할 인덱스 저장
    split_info = {
        'train_indices': train_indices,
        'test_indices': test_indices,
        'split_type': 'stratified_by_player',
        'train_size': len(train_indices),
        'test_size': len(test_indices)
    }
    
    import json
    with open('data/processed/split_indices.json', 'w') as f:
        json.dump(split_info, f)
    
    logger.info("\n분할 인덱스 저장: data/processed/split_indices.json")
    
    return train_df, test_df

if __name__ == "__main__":
    train_df, test_df = analyze_and_fix_injury_distribution()
