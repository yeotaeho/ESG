# 머신러닝 코드 작성 전략

## 📋 개요

이 문서는 `testlearning` 폴더에 머신러닝 기능을 구현하기 위한 전체적인 전략과 구조를 제시합니다.

---

## 🏗️ 파일별 역할 정의

### 1. **Dataset.py** - 데이터 관리 계층

**책임:**
- 데이터 로딩 및 저장
- 데이터 전처리 및 변환
- 데이터 분할 (train/validation/test)
- 데이터 캐싱

**주요 기능:**
```python
# 데이터 로딩
def load_data(source: str) -> pd.DataFrame
def load_from_redis(key: str) -> pd.DataFrame
def load_from_database(query: str) -> pd.DataFrame

# 데이터 전처리
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame
def normalize_data(df: pd.DataFrame) -> pd.DataFrame
def encode_categorical(df: pd.DataFrame) -> pd.DataFrame

# 데이터 분할
def split_data(df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1) -> tuple
def create_dataloader(data, batch_size: int, shuffle: bool = True)

# 데이터 변환
def transform_for_model(data) -> torch.Tensor / np.ndarray
def inverse_transform(data) -> pd.DataFrame
```

**고려사항:**
- ESG 데이터 특성에 맞는 전처리 로직
- Redis 캐싱을 통한 성능 최적화
- 데이터 버전 관리

---

### 2. **model.py** - 모델 정의 계층

**책임:**
- 모델 아키텍처 정의
- 모델 초기화 및 하이퍼파라미터 관리
- 모델 저장 및 로드
- 모델 버전 관리

**주요 기능:**
```python
# 모델 클래스 정의
class ESGModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int)
    def forward(self, x: torch.Tensor) -> torch.Tensor
    
# 모델 관리
def save_model(model: nn.Module, path: str, version: str = None)
def load_model(path: str, device: str = 'cpu') -> nn.Module
def get_model_info(model_path: str) -> dict

# 모델 초기화
def initialize_model(config: dict) -> nn.Module
def get_model_architecture(model: nn.Module) -> dict
```

**모델 타입 예시:**
- 분류 모델: ESG 점수 분류 (A, B, C 등급)
- 회귀 모델: ESG 점수 예측
- 시계열 모델: ESG 트렌드 분석
- NLP 모델: ESG 리포트 분석

---

### 3. **method.py** - 학습 및 평가 메서드 계층

**책임:**
- 학습 함수 구현
- 평가 및 메트릭 계산
- 최적화 전략
- 조기 종료 및 콜백

**주요 기능:**
```python
# 학습 함수
def train_epoch(model, dataloader, criterion, optimizer, device)
def train_model(model, train_loader, val_loader, epochs, config) -> dict
def train_loop(model, train_data, val_data, config: dict) -> dict

# 평가 함수
def evaluate_model(model, dataloader, device) -> dict
def calculate_metrics(y_true, y_pred, task_type: str) -> dict
def calculate_confusion_matrix(y_true, y_pred)

# 최적화
def create_optimizer(model, optimizer_type: str, lr: float)
def create_scheduler(optimizer, scheduler_type: str, **kwargs)
def apply_early_stopping(patience: int, min_delta: float)

# 하이퍼파라미터 튜닝
def optimize_hyperparameters(model_class, train_data, val_data, search_space) -> dict
def grid_search(configs: list) -> dict
def random_search(configs: list, n_iter: int) -> dict
```

**학습 전략:**
- 교차 검증 (K-Fold)
- 학습률 스케줄링
- 정규화 (Dropout, BatchNorm, L2)
- 데이터 증강 (필요시)

---

### 4. **service.py** - 비즈니스 로직 계층

**책임:**
- 전체 ML 파이프라인 오케스트레이션
- 모델 추론 서비스
- 모델 관리 및 버전 관리
- 결과 후처리

**주요 기능:**
```python
# 학습 파이프라인
def train_pipeline(config: dict) -> dict:
    """
    1. 데이터 로딩 (Dataset.py)
    2. 데이터 전처리 (Dataset.py)
    3. 모델 초기화 (model.py)
    4. 학습 실행 (method.py)
    5. 모델 평가 (method.py)
    6. 모델 저장 (model.py)
    """
    
# 추론 서비스
def predict(data: dict, model_id: str = None) -> dict
def batch_predict(data_list: list, model_id: str) -> list
def predict_with_confidence(data: dict) -> dict

# 모델 관리
def get_model_info(model_id: str) -> dict
def list_available_models() -> list
def update_model(model_path: str, model_id: str)
def delete_model(model_id: str)

# 결과 후처리
def format_prediction_result(raw_result) -> dict
def add_explainability(model, data, prediction) -> dict
```

**비즈니스 로직:**
- 모델 선택 (A/B 테스트)
- 예측 결과 검증
- 에러 핸들링 및 재시도 로직
- 로깅 및 모니터링

---

### 5. **router.py** - API 엔드포인트 계층

**책임:**
- FastAPI 라우터 정의
- 요청/응답 스키마 정의
- 에러 핸들링
- API 문서화

**주요 기능:**
```python
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

# 라우터 생성
ml_router = APIRouter(prefix="/ml", tags=["Machine Learning"])

# 요청/응답 모델
class TrainRequest(BaseModel):
    dataset_path: str
    model_type: str
    hyperparameters: dict
    validation_split: float = 0.2

class PredictRequest(BaseModel):
    data: dict
    model_id: str = None

class ModelInfo(BaseModel):
    model_id: str
    version: str
    accuracy: float
    created_at: str

# API 엔드포인트
@ml_router.get("/")
async def ml_root():
    """ML 서비스 상태 확인"""
    
@ml_router.post("/train")
async def train_model(request: TrainRequest, background_tasks: BackgroundTasks):
    """모델 학습 시작 (비동기)"""
    
@ml_router.post("/predict")
async def predict(request: PredictRequest):
    """예측 수행"""
    
@ml_router.get("/models")
async def list_models():
    """사용 가능한 모델 목록 조회"""
    
@ml_router.get("/models/{model_id}")
async def get_model_info(model_id: str):
    """특정 모델 정보 조회"""
    
@ml_router.delete("/models/{model_id}")
async def delete_model(model_id: str):
    """모델 삭제"""
    
@ml_router.get("/train/status/{task_id}")
async def get_training_status(task_id: str):
    """학습 진행 상태 조회"""
```

---

## 🔄 전체 워크플로우

```
[API Request]
    ↓
[router.py] - 요청 검증 및 라우팅
    ↓
[service.py] - 비즈니스 로직 처리
    ↓
┌─────────────────────────────────────┐
│  학습 워크플로우                     │
│  1. Dataset.py - 데이터 로딩         │
│  2. Dataset.py - 데이터 전처리       │
│  3. model.py - 모델 초기화           │
│  4. method.py - 학습 실행            │
│  5. method.py - 모델 평가            │
│  6. model.py - 모델 저장             │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  추론 워크플로우                     │
│  1. Dataset.py - 입력 데이터 전처리  │
│  2. model.py - 모델 로드             │
│  3. model.py - 추론 수행             │
│  4. service.py - 결과 후처리         │
└─────────────────────────────────────┘
    ↓
[router.py] - 응답 반환
```

---

## 📦 의존성 구조

```
┌─────────────┐
│  router.py  │ ← API 계층 (FastAPI)
└──────┬──────┘
       │ 의존
┌──────▼──────┐
│ service.py  │ ← 비즈니스 로직 계층
└──────┬──────┘
       │ 의존
┌──────▼──────┐
│ method.py   │ ← 학습/평가 계층
└──────┬──────┘
       │ 의존
┌──────▼──────┐
│  model.py   │ ← 모델 계층 (PyTorch/TensorFlow)
└──────┬──────┘
       │ 의존
┌──────▼──────┐
│ Dataset.py  │ ← 데이터 계층 (Pandas/NumPy)
└─────────────┘
```

**의존성 방향:**
- 상위 계층은 하위 계층을 import 가능
- 하위 계층은 상위 계층을 import 불가 (순환 참조 방지)
- 각 계층은 독립적으로 테스트 가능

---

## 🎯 구현 단계별 전략

### Phase 1: 기본 구조 (1주)
**목표:** 기본 프레임워크 구축

**작업 내용:**
- [ ] `Dataset.py`: 기본 데이터 로딩 함수 (CSV, JSON)
- [ ] `model.py`: 간단한 모델 클래스 정의 (Linear Model)
- [ ] `router.py`: 기본 API 엔드포인트 (GET /, GET /health)
- [ ] `main.py`: 라우터 등록 및 서버 실행 확인

**체크리스트:**
- ✅ FastAPI 서버가 정상 실행됨
- ✅ 기본 엔드포인트 응답 확인
- ✅ 데이터 로딩 테스트 통과

---

### Phase 2: 학습 기능 (2주)
**목표:** 모델 학습 파이프라인 구축

**작업 내용:**
- [ ] `Dataset.py`: 데이터 전처리 함수 추가
- [ ] `Dataset.py`: 데이터 분할 함수 (train/val/test)
- [ ] `method.py`: 학습 함수 구현 (train_epoch, train_loop)
- [ ] `method.py`: 평가 함수 구현 (evaluate, calculate_metrics)
- [ ] `service.py`: 학습 파이프라인 오케스트레이션
- [ ] `router.py`: POST /train 엔드포인트 구현
- [ ] `model.py`: 모델 저장/로드 기능

**체크리스트:**
- ✅ 모델이 정상적으로 학습됨
- ✅ 학습 결과가 저장됨
- ✅ 학습 진행상황 모니터링 가능

---

### Phase 3: 추론 기능 (1주)
**목표:** 모델 추론 서비스 구현

**작업 내용:**
- [ ] `service.py`: 예측 함수 구현
- [ ] `service.py`: 배치 예측 함수 구현
- [ ] `router.py`: POST /predict 엔드포인트 구현
- [ ] `router.py`: GET /models 엔드포인트 구현
- [ ] 에러 핸들링 추가

**체크리스트:**
- ✅ 단일 예측 요청 처리
- ✅ 배치 예측 요청 처리
- ✅ 모델 목록 조회 가능

---

### Phase 4: 고급 기능 (2주)
**목표:** 프로덕션 수준 기능 추가

**작업 내용:**
- [ ] 모델 버전 관리 시스템
- [ ] 하이퍼파라미터 튜닝 기능
- [ ] A/B 테스트 기능
- [ ] 모델 성능 모니터링
- [ ] Redis 캐싱 통합
- [ ] 로깅 및 에러 추적
- [ ] API 문서화 완성

**체크리스트:**
- ✅ 여러 버전의 모델 관리 가능
- ✅ 모델 성능 비교 가능
- ✅ 캐싱으로 응답 시간 개선
- ✅ 완전한 API 문서 제공

---

## 🛠️ 기술 스택

### 필수 라이브러리
```python
# 머신러닝 프레임워크
torch>=2.0.0          # PyTorch
# 또는
tensorflow>=2.13.0    # TensorFlow
keras>=2.13.0         # Keras

# 데이터 처리
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0

# API 프레임워크
fastapi>=0.100.0
uvicorn>=0.23.0
pydantic>=2.0.0

# 유틸리티
python-dotenv>=1.0.0
redis>=5.0.0
```

### 선택적 라이브러리
```python
# 모델 관리
mlflow>=2.5.0         # ML 모델 라이프사이클 관리
tensorboard>=2.13.0   # 학습 시각화

# 하이퍼파라미터 튜닝
optuna>=3.3.0         # 하이퍼파라미터 최적화

# 데이터 검증
great-expectations    # 데이터 품질 검증

# 모니터링
prometheus-client     # 메트릭 수집
```

---

## 📝 코드 패턴 가이드

### 1. 레포지토리 패턴
각 계층은 명확한 인터페이스를 제공하며, 상위 계층은 인터페이스에만 의존합니다.

```python
# service.py
from app.testlearning.model import ModelRepository
from app.testlearning.Dataset import DataRepository

class MLService:
    def __init__(self):
        self.model_repo = ModelRepository()
        self.data_repo = DataRepository()
```

### 2. 의존성 주입
테스트 용이성을 위해 의존성을 외부에서 주입받습니다.

```python
# service.py
class MLService:
    def __init__(self, model_repo=None, data_repo=None):
        self.model_repo = model_repo or ModelRepository()
        self.data_repo = data_repo or DataRepository()
```

### 3. 설정 관리
환경 변수 또는 설정 파일로 하이퍼파라미터를 관리합니다.

```python
# config.yaml 또는 .env
MODEL_TYPE: "neural_network"
HIDDEN_LAYERS: [128, 64, 32]
LEARNING_RATE: 0.001
BATCH_SIZE: 32
EPOCHS: 100
```

### 4. 에러 핸들링
계층별로 적절한 예외를 정의하고 처리합니다.

```python
# custom_exceptions.py
class ModelNotFoundError(Exception):
    pass

class DataLoadError(Exception):
    pass

class TrainingError(Exception):
    pass
```

### 5. 로깅
각 계층에서 적절한 로깅을 수행합니다.

```python
import logging

logger = logging.getLogger(__name__)

def train_model(...):
    logger.info("Training started")
    try:
        # 학습 로직
        logger.info("Training completed successfully")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
```

---

## 🔍 ESG 데이터 특화 고려사항

### 데이터 특성
- **다양한 데이터 소스**: 재무, 환경, 사회, 지배구조 데이터
- **시계열 데이터**: ESG 점수 변화 추적
- **텍스트 데이터**: ESG 리포트 분석
- **불균형 데이터**: ESG 등급 분포 불균형 가능

### 모델 선택 전략
- **분류 태스크**: ESG 등급 분류 → Random Forest, XGBoost, Neural Network
- **회귀 태스크**: ESG 점수 예측 → Linear Regression, Gradient Boosting
- **시계열 태스크**: ESG 트렌드 예측 → LSTM, Transformer
- **NLP 태스크**: 리포트 분석 → BERT, GPT (Fine-tuning)

### 평가 메트릭
- **분류**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **회귀**: MAE, RMSE, R² Score
- **불균형 데이터**: F1-Score (macro), ROC-AUC, Precision-Recall Curve

---

## 📊 모니터링 및 로깅 전략

### 학습 모니터링
- 실시간 손실 추적
- 검증 메트릭 추적
- 학습 시간 측정
- GPU/CPU 사용률 모니터링

### 추론 모니터링
- 예측 응답 시간
- 요청 처리량 (Throughput)
- 에러율
- 모델 성능 드리프트 감지

### 로깅 포인트
- 학습 시작/종료
- 에포크별 메트릭
- 모델 저장/로드
- 예측 요청/응답 (개인정보 제외)

---

## 🚀 배포 전략

### 모델 버전 관리
- Semantic Versioning (v1.0.0, v1.1.0 등)
- Git 태그와 연동
- 모델 메타데이터 저장 (학습 날짜, 하이퍼파라미터 등)

### 모델 서빙
- 온라인 추론: FastAPI를 통한 RESTful API
- 배치 추론: 스케줄링된 작업으로 대량 데이터 처리
- A/B 테스트: 여러 모델 버전 동시 서빙

### 성능 최적화
- 모델 양자화 (Quantization)
- 모델 압축 (Pruning)
- 배치 처리 최적화
- Redis 캐싱 활용

---

## ✅ 체크리스트

### 코드 품질
- [ ] 각 파일이 단일 책임 원칙을 따름
- [ ] 타입 힌팅 적용
- [ ] Docstring 작성
- [ ] 에러 핸들링 구현
- [ ] 로깅 구현

### 테스트
- [ ] 단위 테스트 작성
- [ ] 통합 테스트 작성
- [ ] API 엔드포인트 테스트
- [ ] 모델 학습/추론 테스트

### 문서화
- [ ] API 문서 (OpenAPI/Swagger)
- [ ] 코드 주석
- [ ] README 작성
- [ ] 사용 예제 제공

### 보안
- [ ] 입력 데이터 검증
- [ ] 모델 파일 접근 제어
- [ ] API 인증/인가 (필요시)
- [ ] 환경 변수 보안 관리

---

## 📚 참고 자료

- [FastAPI 공식 문서](https://fastapi.tiangolo.com/)
- [PyTorch 공식 문서](https://pytorch.org/docs/stable/index.html)
- [MLflow 공식 문서](https://www.mlflow.org/docs/latest/index.html)
- [scikit-learn 공식 문서](https://scikit-learn.org/stable/)

---

## 🔄 업데이트 이력

- 2024-XX-XX: 초기 문서 작성

