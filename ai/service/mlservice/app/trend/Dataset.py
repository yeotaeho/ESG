"""데이터 관리 계층 - 데이터 로딩, 전처리, 분할"""

import pandas as pd
import numpy as np
import os
import json
import redis
import ssl
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging

from .exceptions import DataLoadError

logger = logging.getLogger(__name__)

# Redis 클라이언트 설정
redis_host = os.getenv("UPSTASH_REDIS_HOST", "unified-gibbon-39731.upstash.io")
redis_port = int(os.getenv("UPSTASH_REDIS_PORT", "6379"))
redis_token = os.getenv("UPSTASH_REDIS_TOKEN", "")

redis_client = None
if redis_host:
    try:
        redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            password=redis_token,
            ssl=True,
            ssl_cert_reqs=ssl.CERT_NONE,
            decode_responses=True
        )
    except Exception as e:
        logger.warning(f"Redis 연결 실패: {e}")


def load_data(source: str) -> pd.DataFrame:
    """
    다양한 소스에서 데이터 로딩
    
    Args:
        source: 데이터 소스 경로 (CSV, JSON, Redis 키)
        
    Returns:
        pd.DataFrame: 로딩된 데이터
        
    Raises:
        DataLoadError: 데이터 로딩 실패 시
    """
    try:
        # CSV 파일
        if source.endswith('.csv'):
            df = pd.read_csv(source)
            logger.info(f"CSV 파일 로딩 완료: {source}, Shape: {df.shape}")
            return df
        
        # JSON 파일
        elif source.endswith('.json'):
            df = pd.read_json(source)
            logger.info(f"JSON 파일 로딩 완료: {source}, Shape: {df.shape}")
            return df
        
        # Redis에서 로딩
        elif source.startswith('redis:'):
            return load_from_redis(source.replace('redis:', ''))
        
        else:
            raise DataLoadError(f"지원하지 않는 데이터 형식: {source}")
            
    except Exception as e:
        logger.error(f"데이터 로딩 실패: {e}")
        raise DataLoadError(f"데이터 로딩 실패: {e}")


def load_from_redis(key: str) -> pd.DataFrame:
    """
    Redis에서 데이터 로딩
    
    Args:
        key: Redis 키
        
    Returns:
        pd.DataFrame: 로딩된 데이터
        
    Raises:
        DataLoadError: Redis 연결 실패 또는 데이터 없음
    """
    if not redis_client:
        raise DataLoadError("Redis 클라이언트가 초기화되지 않았습니다")
    
    try:
        data = redis_client.get(key)
        if data is None:
            raise DataLoadError(f"Redis에서 키를 찾을 수 없습니다: {key}")
        
        # JSON 형식으로 가정
        data_dict = json.loads(data)
        df = pd.DataFrame(data_dict)
        logger.info(f"Redis에서 데이터 로딩 완료: {key}, Shape: {df.shape}")
        return df
        
    except Exception as e:
        logger.error(f"Redis 데이터 로딩 실패: {e}")
        raise DataLoadError(f"Redis 데이터 로딩 실패: {e}")


def preprocess_data(df: pd.DataFrame, target_column: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    데이터 전처리
    
    Args:
        df: 원본 데이터프레임
        target_column: 타겟 컬럼명 (선택사항)
        
    Returns:
        Tuple[pd.DataFrame, Optional[pd.Series]]: 전처리된 X와 y
    """
    df = df.copy()
    
    # 타겟 분리
    y = None
    if target_column and target_column in df.columns:
        y = df.pop(target_column)
    
    # 결측치 처리
    df = handle_missing_values(df)
    
    # 범주형 데이터 인코딩
    df = encode_categorical(df)
    
    logger.info(f"데이터 전처리 완료. Shape: {df.shape}")
    
    return df, y


def handle_missing_values(df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
    """
    결측치 처리
    
    Args:
        df: 데이터프레임
        strategy: 처리 전략 ('mean', 'median', 'mode', 'drop')
        
    Returns:
        pd.DataFrame: 결측치가 처리된 데이터프레임
    """
    df = df.copy()
    
    for col in df.columns:
        if df[col].isna().any():
            if df[col].dtype in ['int64', 'float64']:
                if strategy == 'mean':
                    df[col].fillna(df[col].mean(), inplace=True)
                elif strategy == 'median':
                    df[col].fillna(df[col].median(), inplace=True)
                elif strategy == 'mode':
                    df[col].fillna(df[col].mode()[0], inplace=True)
                elif strategy == 'drop':
                    df.dropna(subset=[col], inplace=True)
            else:
                # 범주형 데이터는 최빈값으로 채우기
                df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else '', inplace=True)
    
    logger.info(f"결측치 처리 완료 (전략: {strategy})")
    return df


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    범주형 데이터 인코딩
    
    Args:
        df: 데이터프레임
        
    Returns:
        pd.DataFrame: 인코딩된 데이터프레임
    """
    df = df.copy()
    label_encoders = {}
    
    for col in df.columns:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    
    if label_encoders:
        logger.info(f"범주형 데이터 인코딩 완료: {list(label_encoders.keys())}")
    
    return df


def normalize_data(X_train: pd.DataFrame, X_val: Optional[pd.DataFrame] = None, 
                   X_test: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame], StandardScaler]:
    """
    데이터 정규화
    
    Args:
        X_train: 훈련 데이터
        X_val: 검증 데이터 (선택사항)
        X_test: 테스트 데이터 (선택사항)
        
    Returns:
        Tuple: 정규화된 X_train, X_val, X_test, scaler
    """
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    
    X_val_scaled = None
    if X_val is not None:
        X_val_scaled = pd.DataFrame(
            scaler.transform(X_val),
            columns=X_val.columns,
            index=X_val.index
        )
    
    X_test_scaled = None
    if X_test is not None:
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
    
    logger.info("데이터 정규화 완료")
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def split_data(df: pd.DataFrame, target: pd.Series, test_size: float = 0.2, 
               val_size: float = 0.1, random_state: int = 42) -> Tuple:
    """
    데이터를 train/validation/test로 분할
    
    Args:
        df: 특성 데이터프레임
        target: 타겟 시리즈
        test_size: 테스트 데이터 비율
        val_size: 검증 데이터 비율 (전체 대비)
        random_state: 랜덤 시드
        
    Returns:
        Tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # 먼저 train+val과 test로 분할
    X_temp, X_test, y_temp, y_test = train_test_split(
        df, target, test_size=test_size, random_state=random_state, stratify=target if target.dtype == 'object' else None
    )
    
    # train+val을 train과 val로 분할
    # val_size는 전체 대비 비율이므로, temp에 대한 비율로 조정
    adjusted_val_size = val_size / (1 - test_size)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=adjusted_val_size, random_state=random_state,
        stratify=y_temp if y_temp.dtype == 'object' else None
    )
    
    logger.info(f"데이터 분할 완료 - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def transform_for_model(X: pd.DataFrame, to_tensor: bool = False):
    """
    모델 입력을 위한 데이터 변환
    
    Args:
        X: 데이터프레임
        to_tensor: PyTorch 텐서로 변환할지 여부
        
    Returns:
        변환된 데이터 (numpy array 또는 torch.Tensor)
    """
    if to_tensor:
        import torch
        return torch.FloatTensor(X.values)
    else:
        return X.values


def save_to_redis(key: str, df: pd.DataFrame, ttl: int = 3600):
    """
    데이터프레임을 Redis에 저장
    
    Args:
        key: Redis 키
        df: 저장할 데이터프레임
        ttl: Time to live (초)
    """
    if not redis_client:
        logger.warning("Redis 클라이언트가 없어 저장하지 않습니다")
        return
    
    try:
        data_json = df.to_json(orient='records')
        redis_client.setex(key, ttl, data_json)
        logger.info(f"Redis에 데이터 저장 완료: {key}, TTL: {ttl}초")
    except Exception as e:
        logger.error(f"Redis 저장 실패: {e}")

