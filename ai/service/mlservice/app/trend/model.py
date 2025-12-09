"""모델 정의 계층 - 모델 아키텍처, 저장/로드"""

import torch
import torch.nn as nn
import os
import json
import logging
from typing import Dict, List, Optional
from datetime import datetime

from .exceptions import ModelNotFoundError, ModelSaveError

logger = logging.getLogger(__name__)

# 모델 저장 디렉토리
MODELS_DIR = os.getenv("MODELS_DIR", "./models")
os.makedirs(MODELS_DIR, exist_ok=True)


class ESGModel(nn.Module):
    """ESG 데이터 분석을 위한 신경망 모델"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, 
                 dropout_rate: float = 0.2, activation: str = 'relu'):
        """
        Args:
            input_dim: 입력 차원
            hidden_dims: 은닉층 차원 리스트
            output_dim: 출력 차원
            dropout_rate: 드롭아웃 비율
            activation: 활성화 함수 ('relu', 'tanh', 'sigmoid')
        """
        super(ESGModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        
        # 활성화 함수 선택
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.ReLU()
        
        # 레이어 구성
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self.activation)
            layers.append(nn.Dropout(dropout_rate))
            layers.append(nn.BatchNorm1d(hidden_dim))
            prev_dim = hidden_dim
        
        # 출력층
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
        
        # 가중치 초기화
        self._initialize_weights()
    
    def _initialize_weights(self):
        """가중치 초기화"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        순전파
        
        Args:
            x: 입력 텐서 (batch_size, input_dim)
            
        Returns:
            출력 텐서 (batch_size, output_dim)
        """
        return self.model(x)


def initialize_model(config: Dict) -> ESGModel:
    """
    설정에 따라 모델 초기화
    
    Args:
        config: 모델 설정 딕셔너리
            - input_dim: 입력 차원
            - hidden_dims: 은닉층 차원 리스트
            - output_dim: 출력 차원
            - dropout_rate: 드롭아웃 비율 (기본값: 0.2)
            - activation: 활성화 함수 (기본값: 'relu')
            
    Returns:
        초기화된 모델
    """
    model = ESGModel(
        input_dim=config['input_dim'],
        hidden_dims=config.get('hidden_dims', [128, 64, 32]),
        output_dim=config['output_dim'],
        dropout_rate=config.get('dropout_rate', 0.2),
        activation=config.get('activation', 'relu')
    )
    
    logger.info(f"모델 초기화 완료: {config}")
    return model


def save_model(model: nn.Module, model_id: str, metadata: Optional[Dict] = None):
    """
    모델 저장
    
    Args:
        model: 저장할 모델
        model_id: 모델 ID (버전 포함 가능)
        metadata: 모델 메타데이터 (하이퍼파라미터, 성능 등)
        
    Raises:
        ModelSaveError: 모델 저장 실패 시
    """
    try:
        model_dir = os.path.join(MODELS_DIR, model_id)
        os.makedirs(model_dir, exist_ok=True)
        
        # 모델 가중치 저장
        model_path = os.path.join(model_dir, 'model.pt')
        torch.save(model.state_dict(), model_path)
        
        # 메타데이터 저장
        if metadata is None:
            metadata = {}
        
        metadata['model_id'] = model_id
        metadata['saved_at'] = datetime.now().isoformat()
        metadata['model_type'] = model.__class__.__name__
        
        # 모델 아키텍처 정보
        if isinstance(model, ESGModel):
            metadata['architecture'] = {
                'input_dim': model.input_dim,
                'hidden_dims': model.hidden_dims,
                'output_dim': model.output_dim,
                'dropout_rate': model.dropout_rate
            }
        
        metadata_path = os.path.join(model_dir, 'metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"모델 저장 완료: {model_id}, 경로: {model_dir}")
        
    except Exception as e:
        logger.error(f"모델 저장 실패: {e}")
        raise ModelSaveError(f"모델 저장 실패: {e}")


def load_model(model_id: str, device: str = 'cpu') -> nn.Module:
    """
    모델 로드
    
    Args:
        model_id: 모델 ID
        device: 디바이스 ('cpu' 또는 'cuda')
        
    Returns:
        로드된 모델
        
    Raises:
        ModelNotFoundError: 모델을 찾을 수 없을 때
    """
    try:
        model_dir = os.path.join(MODELS_DIR, model_id)
        model_path = os.path.join(model_dir, 'model.pt')
        metadata_path = os.path.join(model_dir, 'metadata.json')
        
        if not os.path.exists(model_path):
            raise ModelNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
        
        # 메타데이터 로드
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        # 모델 초기화
        if 'architecture' in metadata:
            arch = metadata['architecture']
            model = ESGModel(
                input_dim=arch['input_dim'],
                hidden_dims=arch['hidden_dims'],
                output_dim=arch['output_dim'],
                dropout_rate=arch.get('dropout_rate', 0.2),
                activation=metadata.get('activation', 'relu')
            )
        else:
            raise ModelNotFoundError("모델 아키텍처 정보가 없습니다")
        
        # 가중치 로드
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        logger.info(f"모델 로드 완료: {model_id}, 디바이스: {device}")
        return model
        
    except ModelNotFoundError:
        raise
    except Exception as e:
        logger.error(f"모델 로드 실패: {e}")
        raise ModelNotFoundError(f"모델 로드 실패: {e}")


def get_model_info(model_id: str) -> Dict:
    """
    모델 정보 조회
    
    Args:
        model_id: 모델 ID
        
    Returns:
        모델 메타데이터 딕셔너리
        
    Raises:
        ModelNotFoundError: 모델을 찾을 수 없을 때
    """
    try:
        model_dir = os.path.join(MODELS_DIR, model_id)
        metadata_path = os.path.join(model_dir, 'metadata.json')
        
        if not os.path.exists(metadata_path):
            raise ModelNotFoundError(f"모델 메타데이터를 찾을 수 없습니다: {model_id}")
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # 파일 크기 추가
        model_path = os.path.join(model_dir, 'model.pt')
        if os.path.exists(model_path):
            metadata['file_size'] = os.path.getsize(model_path)
        
        return metadata
        
    except ModelNotFoundError:
        raise
    except Exception as e:
        logger.error(f"모델 정보 조회 실패: {e}")
        raise ModelNotFoundError(f"모델 정보 조회 실패: {e}")


def get_model_architecture(model: nn.Module) -> Dict:
    """
    모델 아키텍처 정보 추출
    
    Args:
        model: 모델 객체
        
    Returns:
        아키텍처 정보 딕셔너리
    """
    if isinstance(model, ESGModel):
        return {
            'input_dim': model.input_dim,
            'hidden_dims': model.hidden_dims,
            'output_dim': model.output_dim,
            'dropout_rate': model.dropout_rate,
            'total_params': sum(p.numel() for p in model.parameters()),
            'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad)
        }
    else:
        return {
            'model_type': model.__class__.__name__,
            'total_params': sum(p.numel() for p in model.parameters()),
            'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad)
        }


def list_available_models() -> List[str]:
    """
    사용 가능한 모델 목록 조회
    
    Returns:
        모델 ID 리스트
    """
    if not os.path.exists(MODELS_DIR):
        return []
    
    models = []
    for item in os.listdir(MODELS_DIR):
        model_dir = os.path.join(MODELS_DIR, item)
        model_path = os.path.join(model_dir, 'model.pt')
        if os.path.isdir(model_dir) and os.path.exists(model_path):
            models.append(item)
    
    return sorted(models)

