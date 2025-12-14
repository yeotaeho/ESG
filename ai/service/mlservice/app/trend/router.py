"""API 엔드포인트 계층"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import logging
import uuid

from .service import get_ml_service
from .exceptions import (
    ModelNotFoundError, DataLoadError, TrainingError, 
    PredictionError, InvalidInputError
)

logger = logging.getLogger(__name__)

# 라우터 생성
ml_router = APIRouter(prefix="/ml", tags=["Machine Learning"])


# 요청/응답 모델 정의
class TrainRequest(BaseModel):
    """학습 요청 모델"""
    dataset_path: str = Field(..., description="데이터셋 경로 (CSV, JSON, redis:key)")
    target_column: str = Field(..., description="타겟 컬럼명")
    model_config: Dict[str, Any] = Field(
        default={
            "hidden_dims": [128, 64, 32],
            "dropout_rate": 0.2,
            "activation": "relu"
        },
        description="모델 설정"
    )
    train_config: Dict[str, Any] = Field(
        default={
            "epochs": 10,
            "batch_size": 32,
            "lr": 0.001,
            "optimizer": "adam",
            "criterion": "crossentropy"
        },
        description="학습 설정"
    )
    test_size: float = Field(default=0.2, ge=0.0, le=1.0, description="테스트 데이터 비율")
    val_size: float = Field(default=0.1, ge=0.0, le=1.0, description="검증 데이터 비율")
    task_type: str = Field(default="classification", description="작업 타입 (classification/regression)")
    model_id: Optional[str] = Field(default=None, description="모델 ID (자동 생성되지 지정)")


class PredictRequest(BaseModel):
    """예측 요청 모델"""
    data: Dict[str, Any] = Field(..., description="예측할 데이터 (features 또는 직접 특성 배열)")
    model_id: Optional[str] = Field(default=None, description="모델 ID (None이면 최신 모델 사용)")


class BatchPredictRequest(BaseModel):
    """배치 예측 요청 모델"""
    data_list: List[Dict[str, Any]] = Field(..., description="예측할 데이터 리스트")
    model_id: Optional[str] = Field(default=None, description="모델 ID")


class TrainingTaskResponse(BaseModel):
    """학습 작업 응답 모델"""
    task_id: str
    status: str
    message: str


class PredictionResponse(BaseModel):
    """예측 응답 모델"""
    model_id: str
    result: Dict[str, Any]


class ModelInfoResponse(BaseModel):
    """모델 정보 응답 모델"""
    model_id: str
    info: Dict[str, Any]


# 학습 작업 저장 (실제 프로덕션에서는 Redis 등 사용)
training_tasks: Dict[str, Dict] = {}


@ml_router.get("/")
async def ml_root():
    """ML 서비스 상태 확인"""
    return {
        "message": "Machine Learning Service",
        "status": "running",
        "available_models": len(get_ml_service().list_available_models())
    }


@ml_router.post("/train", response_model=TrainingTaskResponse)
async def train_model(request: TrainRequest, background_tasks: BackgroundTasks):
    """
    모델 학습 시작 (비동기)
    
    학습은 백그라운드에서 실행되며, task_id를 반환합니다.
    학습 진행 상황은 /train/status/{task_id} 엔드포인트로 확인할 수 있습니다.
    """
    try:
        task_id = str(uuid.uuid4())
        
        # 학습 설정 구성
        config = {
            'dataset_path': request.dataset_path,
            'target_column': request.target_column,
            'model_config': request.model_config,
            'train_config': request.train_config,
            'test_size': request.test_size,
            'val_size': request.val_size,
            'task_type': request.task_type,
            'model_id': request.model_id
        }
        
        # 작업 상태 초기화
        training_tasks[task_id] = {
            'status': 'pending',
            'progress': 0,
            'message': '학습 대기 중...',
            'result': None,
            'error': None
        }
        
        # 백그라운드 작업으로 학습 실행
        async def train_task():
            try:
                training_tasks[task_id]['status'] = 'running'
                training_tasks[task_id]['message'] = '학습 시작...'
                
                service = get_ml_service()
                result = service.train_pipeline(config)
                
                training_tasks[task_id]['status'] = 'completed'
                training_tasks[task_id]['progress'] = 100
                training_tasks[task_id]['message'] = '학습 완료'
                training_tasks[task_id]['result'] = result
                
            except Exception as e:
                training_tasks[task_id]['status'] = 'failed'
                training_tasks[task_id]['message'] = f'학습 실패: {str(e)}'
                training_tasks[task_id]['error'] = str(e)
                logger.error(f"학습 작업 실패 (task_id: {task_id}): {e}")
        
        background_tasks.add_task(train_task)
        
        return TrainingTaskResponse(
            task_id=task_id,
            status="pending",
            message="학습 작업이 시작되었습니다"
        )
        
    except Exception as e:
        logger.error(f"학습 요청 처리 실패: {e}")
        raise HTTPException(status_code=500, detail=f"학습 요청 처리 실패: {str(e)}")


@ml_router.get("/train/status/{task_id}")
async def get_training_status(task_id: str):
    """학습 진행 상태 조회"""
    if task_id not in training_tasks:
        raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다")
    
    return training_tasks[task_id]


@ml_router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictRequest):
    """
    예측 수행
    
    단일 데이터에 대한 예측을 수행합니다.
    """
    try:
        service = get_ml_service()
        result = service.predict(request.data, request.model_id)
        
        return PredictionResponse(
            model_id=result.pop('model_id'),
            result=result
        )
        
    except ModelNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except InvalidInputError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except PredictionError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"예측 요청 처리 실패: {e}")
        raise HTTPException(status_code=500, detail=f"예측 실패: {str(e)}")


@ml_router.post("/predict/batch")
async def batch_predict(request: BatchPredictRequest):
    """
    배치 예측
    
    여러 데이터에 대한 배치 예측을 수행합니다.
    """
    try:
        service = get_ml_service()
        results = service.batch_predict(request.data_list, request.model_id)
        
        return {
            "count": len(results),
            "results": results
        }
        
    except ModelNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"배치 예측 요청 처리 실패: {e}")
        raise HTTPException(status_code=500, detail=f"배치 예측 실패: {str(e)}")


@ml_router.get("/models")
async def list_models():
    """
    사용 가능한 모델 목록 조회
    
    저장된 모든 모델의 목록을 반환합니다.
    """
    try:
        service = get_ml_service()
        models = service.list_available_models()
        
        # 각 모델의 기본 정보 포함
        models_info = []
        for model_id in models:
            try:
                info = service.get_model_info(model_id)
                models_info.append({
                    "model_id": model_id,
                    "created_at": info.get("saved_at"),
                    "model_type": info.get("model_type"),
                    "test_metrics": info.get("test_metrics", {})
                })
            except Exception as e:
                logger.warning(f"모델 정보 조회 실패 ({model_id}): {e}")
                models_info.append({
                    "model_id": model_id,
                    "error": "정보 조회 실패"
                })
        
        return {
            "count": len(models),
            "models": models_info
        }
        
    except Exception as e:
        logger.error(f"모델 목록 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"모델 목록 조회 실패: {str(e)}")


@ml_router.get("/models/{model_id}", response_model=ModelInfoResponse)
async def get_model_info(model_id: str):
    """
    특정 모델 정보 조회
    
    모델의 상세 정보를 반환합니다.
    """
    try:
        service = get_ml_service()
        info = service.get_model_info(model_id)
        
        return ModelInfoResponse(
            model_id=model_id,
            info=info
        )
        
    except ModelNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"모델 정보 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"모델 정보 조회 실패: {str(e)}")


@ml_router.delete("/models/{model_id}")
async def delete_model(model_id: str):
    """
    모델 삭제
    
    지정된 모델을 삭제합니다.
    """
    try:
        service = get_ml_service()
        result = service.delete_model(model_id)
        
        return {
            "status": "success",
            "message": f"모델 {model_id}이(가) 삭제되었습니다",
            **result
        }
        
    except ModelNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"모델 삭제 실패: {e}")
        raise HTTPException(status_code=500, detail=f"모델 삭제 실패: {str(e)}")


@ml_router.get("/health")
async def health_check():
    """헬스 체크"""
    try:
        service = get_ml_service()
        models = service.list_available_models()
        
        return {
            "status": "healthy",
            "device": service.device,
            "available_models": len(models),
            "models": models
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

