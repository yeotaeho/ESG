"""비즈니스 로직 계층 - ML 파이프라인 오케스트레이션"""

import os
import logging
from typing import Dict, Optional, List
import torch
import numpy as np

from .Dataset import (
    load_data, preprocess_data, split_data, normalize_data,
    transform_for_model, save_to_redis, load_from_redis
)
from .model import (
    initialize_model, save_model, load_model, get_model_info,
    list_available_models, get_model_architecture
)
from .method import (
    train_model, evaluate_model, create_dataloader, calculate_metrics
)
from .exceptions import (
    DataLoadError, TrainingError, PredictionError, ModelNotFoundError
)

logger = logging.getLogger(__name__)


class MLService:
    """머신러닝 서비스 클래스"""
    
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"MLService 초기화 완료, 디바이스: {self.device}")
    
    def train_pipeline(self, config: Dict) -> Dict:
        """
        전체 학습 파이프라인 실행
        
        Args:
            config: 학습 설정
                - dataset_path: 데이터셋 경로   
                - target_column: 타겟 컬럼명
                - model_config: 모델 설정
                - train_config: 학습 설정
                - test_size: 테스트 데이터 비율
                - val_size: 검증 데이터 비율
                - model_id: 모델 ID (저장용)
                
        Returns:
            학습 결과 딕셔너리
        """
        try:
            logger.info("학습 파이프라인 시작")
            
            # 1. 데이터 로딩
            dataset_path = config['dataset_path']
            logger.info(f"데이터 로딩: {dataset_path}")
            df = load_data(dataset_path)
            
            # 2. 데이터 전처리
            target_column = config.get('target_column')
            logger.info("데이터 전처리 시작")
            X, y = preprocess_data(df, target_column)
            
            if y is None:
                raise DataLoadError("타겟 컬럼을 찾을 수 없습니다")
            
            # 3. 데이터 분할
            test_size = config.get('test_size', 0.2)
            val_size = config.get('val_size', 0.1)
            logger.info(f"데이터 분할: test={test_size}, val={val_size}")
            X_train, X_val, X_test, y_train, y_val, y_test = split_data(
                X, y, test_size=test_size, val_size=val_size
            )
            
            # 4. 데이터 정규화
            logger.info("데이터 정규화")
            X_train_scaled, X_val_scaled, X_test_scaled, scaler = normalize_data(
                X_train, X_val, X_test
            )
            
            # 5. 모델 초기화
            model_config = config['model_config']
            model_config['input_dim'] = X_train_scaled.shape[1]
            if config.get('task_type') == 'classification':
                model_config['output_dim'] = len(y.unique()) if hasattr(y, 'unique') else len(np.unique(y))
            else:
                model_config['output_dim'] = 1
            
            logger.info("모델 초기화")
            model = initialize_model(model_config)
            
            # 6. 데이터 로더 생성
            task_type = config.get('task_type', 'classification')
            train_loader = create_dataloader(
                transform_for_model(X_train_scaled, to_tensor=True),
                torch.LongTensor(y_train.values) if task_type == 'classification' 
                else torch.FloatTensor(y_train.values),
                batch_size=config.get('train_config', {}).get('batch_size', 32)
            )
            val_loader = create_dataloader(
                transform_for_model(X_val_scaled, to_tensor=True),
                torch.LongTensor(y_val.values) if task_type == 'classification'
                else torch.FloatTensor(y_val.values),
                batch_size=config.get('train_config', {}).get('batch_size', 32),
                shuffle=False
            )
            
            # 7. 학습 실행
            train_config = config.get('train_config', {})
            train_config['device'] = self.device
            train_config['task_type'] = task_type
            
            logger.info("모델 학습 시작")
            train_result = train_model(model, train_loader, val_loader, train_config)
            
            # 8. 테스트 평가
            logger.info("테스트 데이터 평가")
            test_loader = create_dataloader(
                transform_for_model(X_test_scaled, to_tensor=True),
                torch.LongTensor(y_test.values) if task_type == 'classification'
                else torch.FloatTensor(y_test.values),
                batch_size=32,
                shuffle=False
            )
            
            import torch.nn as nn
            criterion = nn.CrossEntropyLoss() if task_type == 'classification' else nn.MSELoss()
            test_metrics = evaluate_model(model, test_loader, criterion, self.device, task_type)
            
            # 9. 모델 저장
            model_id = config.get('model_id', f"model_{len(list_available_models()) + 1}")
            metadata = {
                'train_config': train_config,
                'model_config': model_config,
                'train_result': train_result,
                'test_metrics': test_metrics,
                'dataset_info': {
                    'shape': df.shape,
                    'features': list(X.columns) if hasattr(X, 'columns') else None
                }
            }
            
            logger.info(f"모델 저장: {model_id}")
            save_model(model, model_id, metadata)
            
            logger.info("학습 파이프라인 완료")
            
            return {
                'model_id': model_id,
                'train_result': train_result,
                'test_metrics': test_metrics,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"학습 파이프라인 실패: {e}")
            raise TrainingError(f"학습 파이프라인 실패: {e}")
    
    def predict(self, data: Dict, model_id: Optional[str] = None) -> Dict:
        """
        예측 수행
        
        Args:
            data: 예측할 데이터
                - features: 특성 데이터 (리스트 또는 딕셔너리)
                - 또는 직접 특성 배열
            model_id: 모델 ID (None이면 최신 모델 사용)
            
        Returns:
            예측 결과 딕셔너리
        """
        try:
            # 모델 로드
            if model_id is None:
                models = list_available_models()
                if not models:
                    raise ModelNotFoundError("사용 가능한 모델이 없습니다")
                model_id = models[-1]  # 최신 모델 사용
            
            logger.info(f"모델 로드: {model_id}")
            model = load_model(model_id, device=self.device)
            model_info = get_model_info(model_id)
            
            # 입력 데이터 처리
            if 'features' in data:
                features = data['features']
            else:
                features = data
            
            if isinstance(features, dict):
                # 딕셔너리를 배열로 변환
                features = [features.get(f'feature_{i}', 0.0) for i in range(len(features))]
            
            if not isinstance(features, (list, np.ndarray)):
                raise ValueError("입력 데이터 형식이 올바르지 않습니다")
            
            # 텐서로 변환
            input_tensor = torch.FloatTensor([features]).to(self.device)
            
            # 예측 수행
            model.eval()
            with torch.no_grad():
                output = model(input_tensor)
                
                # 결과 처리
                if output.dim() > 1 and output.size(1) > 1:
                    # 분류
                    probabilities = torch.softmax(output, dim=1)
                    predicted_class = output.argmax(dim=1).item()
                    confidence = probabilities[0][predicted_class].item()
                    
                    result = {
                        'predicted_class': int(predicted_class),
                        'confidence': float(confidence),
                        'probabilities': probabilities[0].cpu().numpy().tolist()
                    }
                else:
                    # 회귀
                    predicted_value = output.item()
                    result = {
                        'predicted_value': float(predicted_value)
                    }
            
            result['model_id'] = model_id
            logger.info(f"예측 완료: {result}")
            
            return result
            
        except ModelNotFoundError:
            raise
        except Exception as e:
            logger.error(f"예측 실패: {e}")
            raise PredictionError(f"예측 실패: {e}")
    
    def batch_predict(self, data_list: List[Dict], model_id: Optional[str] = None) -> List[Dict]:
        """
        배치 예측
        
        Args:
            data_list: 예측할 데이터 리스트
            model_id: 모델 ID
            
        Returns:
            예측 결과 리스트
        """
        results = []
        for data in data_list:
            try:
                result = self.predict(data, model_id)
                results.append(result)
            except Exception as e:
                logger.error(f"배치 예측 중 오류: {e}")
                results.append({'error': str(e)})
        
        return results
    
    def get_model_info(self, model_id: str) -> Dict:
        """모델 정보 조회"""
        try:
            return get_model_info(model_id)
        except Exception as e:
            logger.error(f"모델 정보 조회 실패: {e}")
            raise
    
    def list_available_models(self) -> List[str]:
        """사용 가능한 모델 목록"""
        return list_available_models()
    
    def delete_model(self, model_id: str) -> Dict:
        """
        모델 삭제
        
        Args:
            model_id: 삭제할 모델 ID
            
        Returns:
            삭제 결과
        """
        try:
            import shutil
            models_dir = os.getenv("MODELS_DIR", "./models")
            model_dir = os.path.join(models_dir, model_id)
            
            if not os.path.exists(model_dir):
                raise ModelNotFoundError(f"모델을 찾을 수 없습니다: {model_id}")
            
            shutil.rmtree(model_dir)
            logger.info(f"모델 삭제 완료: {model_id}")
            
            return {'status': 'success', 'model_id': model_id}
            
        except Exception as e:
            logger.error(f"모델 삭제 실패: {e}")
            raise


# 싱글톤 인스턴스
_ml_service = None

def get_ml_service() -> MLService:
    """MLService 싱글톤 인스턴스 반환"""
    global _ml_service
    if _ml_service is None:
        _ml_service = MLService()
    return _ml_service

