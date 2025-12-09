# 머신러닝 워크플로우 상세 설명

## 📋 개요

이 문서는 구현된 머신러닝 서비스의 전체 워크플로우와 각 컴포넌트가 어떻게 상호작용하는지 설명합니다.

---

## 🏗️ 전체 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Router Layer                      │
│              (router.py - API 엔드포인트)                     │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────────┐
│                  Business Logic Layer                        │
│              (service.py - MLService 클래스)                  │
└──────┬──────────────────┬──────────────────┬────────────────┘
       │                  │                  │
       ↓                  ↓                  ↓
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│ Dataset.py  │  │  model.py   │  │ method.py   │
│ 데이터 관리  │  │ 모델 정의   │  │ 학습/평가   │
└─────────────┘  └─────────────┘  └─────────────┘
```

---

## 🔄 전체 워크플로우

### 1️⃣ 학습 워크플로우 (Training Pipeline)

#### Step 1: API 요청 수신
```python
# router.py - POST /ml/train
@ml_router.post("/train")
async def train_model(request: TrainRequest, background_tasks: BackgroundTasks):
```

**과정:**
1. 클라이언트가 학습 요청을 `/api/ml/train` 엔드포인트로 전송
2. `TrainRequest` 모델로 요청 데이터 검증
3. 고유한 `task_id` 생성 (UUID)
4. 백그라운드 작업으로 학습 시작 (비동기 처리)
5. `task_id`를 즉시 반환하여 클라이언트가 진행 상황 추적 가능

**요청 예시:**
```json
{
  "dataset_path": "data/esg_data.csv",
  "target_column": "esg_rating",
  "model_config": {
    "hidden_dims": [128, 64, 32],
    "dropout_rate": 0.2,
    "activation": "relu"
  },
  "train_config": {
    "epochs": 10,
    "batch_size": 32,
    "lr": 0.001,
    "optimizer": "adam"
  },
  "test_size": 0.2,
  "val_size": 0.1,
  "task_type": "classification"
}
```

---

#### Step 2: 서비스 계층으로 전달
```python
# service.py - MLService.train_pipeline()
def train_pipeline(self, config: Dict) -> Dict:
```

**역할:**
- 전체 학습 파이프라인의 오케스트레이션
- 각 단계를 순차적으로 실행하고 결과를 다음 단계로 전달

---

#### Step 3: 데이터 로딩
```python
# Dataset.py - load_data()
df = load_data(dataset_path)
```

**지원 형식:**
- **CSV 파일**: `.csv` 확장자
- **JSON 파일**: `.json` 확장자
- **Redis**: `redis:key` 형식으로 시작하는 키

**처리 과정:**
```python
def load_data(source: str) -> pd.DataFrame:
    # 1. 파일 확장자 확인
    if source.endswith('.csv'):
        df = pd.read_csv(source)
    elif source.endswith('.json'):
        df = pd.read_json(source)
    elif source.startswith('redis:'):
        df = load_from_redis(source.replace('redis:', ''))
    
    # 2. 데이터프레임 반환
    return df
```

**예시:**
- CSV: `"data/train.csv"` → `pd.read_csv()` 사용
- Redis: `"redis:esg_data_2024"` → Redis에서 JSON 로드 후 DataFrame 변환

---

#### Step 4: 데이터 전처리
```python
# Dataset.py - preprocess_data()
X, y = preprocess_data(df, target_column)
```

**전처리 단계:**

##### 4-1. 타겟 분리
```python
# 타겟 컬럼 추출 및 제거
if target_column and target_column in df.columns:
    y = df.pop(target_column)  # y는 타겟, df는 특성 데이터
```

##### 4-2. 결측치 처리
```python
def handle_missing_values(df: pd.DataFrame, strategy: str = 'mean'):
    for col in df.columns:
        if df[col].isna().any():
            if df[col].dtype in ['int64', 'float64']:
                # 수치형: 평균/중앙값/최빈값으로 채우기
                df[col].fillna(df[col].mean(), inplace=True)
            else:
                # 범주형: 최빈값으로 채우기
                df[col].fillna(df[col].mode()[0], inplace=True)
```

**처리 전략:**
- `mean`: 평균값으로 채우기
- `median`: 중앙값으로 채우기
- `mode`: 최빈값으로 채우기
- `drop`: 결측치가 있는 행 삭제

##### 4-3. 범주형 데이터 인코딩
```python
def encode_categorical(df: pd.DataFrame):
    for col in df.columns:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
```

**변환 예시:**
```
원본: ['A', 'B', 'A', 'C'] 
→ 인코딩: [0, 1, 0, 2]
```

---

#### Step 5: 데이터 분할
```python
# Dataset.py - split_data()
X_train, X_val, X_test, y_train, y_val, y_test = split_data(
    X, y, test_size=0.2, val_size=0.1
)
```

**분할 과정:**
```python
# 1단계: 전체 데이터를 (Train+Val)과 Test로 분할
X_temp, X_test, y_temp, y_test = train_test_split(
    df, target, test_size=0.2, random_state=42
)

# 2단계: (Train+Val)을 Train과 Val로 분할
# val_size는 전체 대비 비율이므로 조정 필요
adjusted_val_size = 0.1 / (1 - 0.2)  # = 0.125

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.125, random_state=42
)
```

**최종 분할 결과:**
- **Train**: 70% (학습용)
- **Validation**: 10% (검증용 - 하이퍼파라미터 튜닝)
- **Test**: 20% (최종 평가용 - 모델 성능 평가)

**왜 3개로 나누나?**
- **Train**: 모델이 학습하는 데이터
- **Validation**: 학습 중 모델 성능을 평가하여 과적합 방지
- **Test**: 모델이 한 번도 보지 못한 데이터로 최종 성능 평가

---

#### Step 6: 데이터 정규화
```python
# Dataset.py - normalize_data()
X_train_scaled, X_val_scaled, X_test_scaled, scaler = normalize_data(
    X_train, X_val, X_test
)
```

**정규화 과정:**
```python
scaler = StandardScaler()

# 1. Train 데이터로 fit (평균과 표준편차 계산)
X_train_scaled = scaler.fit_transform(X_train)
# → 각 특성의 평균을 0, 표준편차를 1로 변환

# 2. Val과 Test는 같은 scaler로 transform (중요!)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
```

**정규화 공식:**
```
z = (x - μ) / σ
```
- `μ`: 평균
- `σ`: 표준편차

**예시:**
```
원본: [100, 200, 300, 400, 500]
평균: 300, 표준편차: 141.42

정규화 후: [-1.41, -0.71, 0, 0.71, 1.41]
```

**왜 정규화가 필요한가?**
- 서로 다른 스케일의 특성들이 있을 때 (예: 나이 0-100, 수입 0-1000000)
- 정규화를 통해 모든 특성이 동일한 스케일을 가지게 되어 학습이 안정적임
- 신경망 모델은 정규화된 데이터에서 더 빠르게 수렴함

---

#### Step 7: 모델 초기화
```python
# model.py - initialize_model()
model_config['input_dim'] = X_train_scaled.shape[1]  # 특성 개수
model_config['output_dim'] = len(y.unique())  # 클래스 개수 (분류)
model = initialize_model(model_config)
```

**모델 아키텍처:**
```python
class ESGModel(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        # 예: input_dim=10, hidden_dims=[128, 64, 32], output_dim=5
        
        # 레이어 구성
        layers = []
        prev_dim = input_dim  # 10
        
        for hidden_dim in hidden_dims:
            # Linear(10, 128) → ReLU → Dropout → BatchNorm
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            layers.append(nn.BatchNorm1d(hidden_dim))
            prev_dim = hidden_dim  # 128, 64, 32 순서로
        
        # 출력층: Linear(32, 5)
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
```

**네트워크 구조:**
```
Input (10 features)
    ↓
Linear(10 → 128)
    ↓
ReLU
    ↓
Dropout(0.2)  # 과적합 방지
    ↓
BatchNorm1d  # 학습 안정화
    ↓
Linear(128 → 64)
    ↓
ReLU → Dropout → BatchNorm
    ↓
Linear(64 → 32)
    ↓
ReLU → Dropout → BatchNorm
    ↓
Linear(32 → 5)  # 출력층 (5개 클래스)
    ↓
Output (5 logits)
```

**각 구성 요소 설명:**
- **Linear**: 선형 변환 (y = Wx + b)
- **ReLU**: 활성화 함수 (음수는 0으로, 양수는 그대로)
- **Dropout**: 무작위로 일부 뉴런을 비활성화하여 과적합 방지
- **BatchNorm**: 배치 단위로 정규화하여 학습 안정화

---

#### Step 8: 데이터 로더 생성
```python
# method.py - create_dataloader()
train_loader = create_dataloader(
    transform_for_model(X_train_scaled, to_tensor=True),
    torch.LongTensor(y_train.values),
    batch_size=32
)
```

**변환 과정:**
```python
# 1. NumPy 배열 → PyTorch Tensor
X_tensor = torch.FloatTensor(X_train_scaled.values)  # (700, 10)
y_tensor = torch.LongTensor(y_train.values)  # (700,)

# 2. TensorDataset 생성
dataset = TensorDataset(X_tensor, y_tensor)

# 3. DataLoader 생성 (배치 단위로 제공)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

**배치 처리 예시:**
```
전체 데이터: 700개
배치 크기: 32

→ 총 배치 수: 22개 (700 / 32 = 21.875)
- 배치 1: 샘플 0-31
- 배치 2: 샘플 32-63
- ...
- 배치 22: 샘플 672-699
```

**왜 배치로 나누나?**
- 메모리 효율성: 전체 데이터를 한 번에 로드하지 않음
- 학습 속도: GPU 병렬 처리 효율 향상
- 일반화: 각 배치마다 다른 샘플 순서 (shuffle=True)

---

#### Step 9: 학습 실행
```python
# method.py - train_model()
train_result = train_model(model, train_loader, val_loader, train_config)
```

##### 9-1. 손실 함수 설정
```python
# 분류: CrossEntropyLoss
criterion = nn.CrossEntropyLoss()

# 회귀: MSELoss
criterion = nn.MSELoss()
```

**CrossEntropyLoss 예시:**
```
예측: [0.1, 0.8, 0.05, 0.03, 0.02]  # 5개 클래스에 대한 확률
실제: 1  # 클래스 1이 정답

손실 = -log(0.8) = 0.223  # 정답 클래스의 확률이 높을수록 손실 낮음
```

##### 9-2. 옵티마이저 설정
```python
# Adam 옵티마이저 (가장 많이 사용)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습률(learning rate): 한 스텝마다 얼마나 큰 보폭으로 업데이트할지
# 너무 크면: 수렴하지 못하고 발산
# 너무 작으면: 학습이 느림
```

##### 9-3. 에포크별 학습
```python
for epoch in range(epochs):  # 예: 10 에포크
    # === 학습 단계 ===
    for batch_idx, (data, target) in enumerate(train_loader):
        # 1. Forward pass (순전파)
        output = model(data)  # 예측
        
        # 2. Loss 계산
        loss = criterion(output, target)
        
        # 3. Backward pass (역전파)
        optimizer.zero_grad()  # 이전 그래디언트 초기화
        loss.backward()  # 그래디언트 계산
        optimizer.step()  # 가중치 업데이트
    
    # === 검증 단계 ===
    val_metrics = evaluate_model(model, val_loader, ...)
```

**학습 과정 상세:**

**Forward Pass (순전파):**
```
입력 데이터 (32, 10)  # 배치 크기 32, 특성 10개
    ↓
모델 통과
    ↓
예측 출력 (32, 5)  # 배치 크기 32, 클래스 5개
    ↓
손실 계산
```

**Backward Pass (역전파):**
```
손실 값
    ↓
loss.backward()  # 각 가중치에 대한 그래디언트 계산
    ↓
optimizer.step()  # 그래디언트를 반영하여 가중치 업데이트
```

**가중치 업데이트 공식 (Adam):**
```
w_new = w_old - learning_rate * gradient
```

##### 9-4. 검증 및 메트릭 계산
```python
def evaluate_model(model, dataloader, criterion, device):
    model.eval()  # 평가 모드 (Dropout 비활성화)
    
    with torch.no_grad():  # 그래디언트 계산 안 함 (메모리 절약)
        for data, target in dataloader:
            output = model(data)
            
            # 분류 작업
            if task_type == 'classification':
                pred = output.argmax(dim=1)  # 가장 높은 확률의 클래스
                # 정확도, 정밀도, 재현율, F1 점수 계산
```

**분류 메트릭:**
- **Accuracy**: 전체 예측 중 정확한 예측의 비율
- **Precision**: 양성으로 예측한 것 중 실제 양성인 비율
- **Recall**: 실제 양성 중 양성으로 예측한 비율
- **F1-Score**: Precision과 Recall의 조화평균

**회귀 메트릭:**
- **MAE (Mean Absolute Error)**: 평균 절대 오차
- **RMSE (Root Mean Squared Error)**: 평균 제곱근 오차
- **R² Score**: 결정계수 (1에 가까울수록 좋음)

##### 9-5. 조기 종료 (Early Stopping)
```python
early_stopping = EarlyStopping(patience=10, min_delta=0.0)

for epoch in range(epochs):
    val_loss = evaluate_model(...)
    
    if early_stopping(val_loss):
        print("조기 종료: 검증 손실이 개선되지 않음")
        break
```

**동작 방식:**
- 검증 손실이 개선되지 않으면 카운터 증가
- `patience` 번 연속 개선 없으면 학습 중단
- 과적합 방지 및 학습 시간 단축

---

#### Step 10: 테스트 평가
```python
# 학습 완료 후 테스트 데이터로 최종 평가
test_metrics = evaluate_model(model, test_loader, criterion, device, task_type)
```

**중요:** 
- 테스트 데이터는 학습 중 한 번도 사용하지 않음
- 모델의 **실제 성능**을 평가하는 유일한 방법
- 검증 데이터와 달리 모델 선택에 사용되지 않음

---

#### Step 11: 모델 저장
```python
# model.py - save_model()
save_model(model, model_id, metadata)
```

**저장 구조:**
```
models/
  └── model_2024_01_15_001/
      ├── model.pt          # 모델 가중치
      └── metadata.json     # 메타데이터
```

**metadata.json 내용:**
```json
{
  "model_id": "model_2024_01_15_001",
  "saved_at": "2024-01-15T10:30:00",
  "model_type": "ESGModel",
  "architecture": {
    "input_dim": 10,
    "hidden_dims": [128, 64, 32],
    "output_dim": 5,
    "dropout_rate": 0.2
  },
  "train_config": {...},
  "test_metrics": {
    "accuracy": 0.85,
    "precision": 0.84,
    "recall": 0.85,
    "f1_score": 0.84
  }
}
```

---

### 2️⃣ 예측 워크플로우 (Prediction Pipeline)

#### Step 1: API 요청 수신
```python
# router.py - POST /ml/predict
@ml_router.post("/predict")
async def predict(request: PredictRequest):
```

**요청 예시:**
```json
{
  "data": {
    "features": [0.5, 0.3, 0.8, 0.2, 0.9, 0.1, 0.7, 0.4, 0.6, 0.3]
  },
  "model_id": "model_2024_01_15_001"  // 선택사항 (없으면 최신 모델 사용)
}
```

---

#### Step 2: 모델 로드
```python
# service.py - MLService.predict()
model = load_model(model_id, device=self.device)
```

**로드 과정:**
```python
# 1. 메타데이터 로드
metadata = json.load('models/model_id/metadata.json')

# 2. 모델 아키텍처 재구성
model = ESGModel(
    input_dim=metadata['architecture']['input_dim'],
    hidden_dims=metadata['architecture']['hidden_dims'],
    output_dim=metadata['architecture']['output_dim']
)

# 3. 가중치 로드
model.load_state_dict(torch.load('models/model_id/model.pt'))

# 4. 평가 모드로 설정
model.eval()
```

---

#### Step 3: 입력 데이터 전처리
```python
# 입력 데이터를 텐서로 변환
input_tensor = torch.FloatTensor([features]).to(device)
# 형태: (1, 10) - 배치 크기 1, 특성 10개
```

**주의사항:**
- 학습 시 사용한 **동일한 전처리** 적용 필요
- 정규화 스케일러도 함께 저장/로드해야 함 (현재 구현에서는 메타데이터에 포함 가능)

---

#### Step 4: 예측 수행
```python
model.eval()
with torch.no_grad():
    output = model(input_tensor)
    
    # 분류 작업
    if output.dim() > 1 and output.size(1) > 1:
        probabilities = torch.softmax(output, dim=1)
        predicted_class = output.argmax(dim=1).item()
        confidence = probabilities[0][predicted_class].item()
```

**예측 과정:**
```
입력: [0.5, 0.3, 0.8, ...]  # 정규화된 특성 값
    ↓
모델 통과
    ↓
로짓 출력: [-0.2, 2.5, 0.1, -1.3, 0.8]  # 5개 클래스에 대한 점수
    ↓
Softmax 적용
    ↓
확률: [0.12, 0.68, 0.18, 0.05, 0.25]  # 합계 = 1.0
    ↓
예측 클래스: 1 (가장 높은 확률)
신뢰도: 0.68 (68%)
```

**Softmax 함수:**
```
softmax(x_i) = exp(x_i) / Σ exp(x_j)
```
- 모든 출력을 0~1 사이의 확률로 변환
- 모든 확률의 합은 1

---

#### Step 5: 결과 반환
```python
return {
    "model_id": "model_2024_01_15_001",
    "predicted_class": 1,
    "confidence": 0.68,
    "probabilities": [0.12, 0.68, 0.18, 0.05, 0.25]
}
```

**회귀 작업인 경우:**
```python
return {
    "model_id": "model_2024_01_15_001",
    "predicted_value": 75.5  # 연속값
}
```

---

## 📊 데이터 흐름도

### 학습 시
```
CSV/JSON/Redis 데이터
    ↓
DataFrame (Pandas)
    ↓
전처리 (결측치, 인코딩)
    ↓
분할 (Train/Val/Test)
    ↓
정규화 (StandardScaler)
    ↓
Tensor 변환 (PyTorch)
    ↓
DataLoader (배치 단위)
    ↓
모델 학습
    ↓
평가 메트릭
    ↓
모델 저장
```

### 예측 시
```
입력 데이터 (JSON)
    ↓
특성 배열 추출
    ↓
Tensor 변환
    ↓
모델 로드
    ↓
Forward Pass
    ↓
Softmax (분류) 또는 직접 출력 (회귀)
    ↓
결과 반환 (JSON)
```

---

## 🔍 주요 개념 설명

### 1. 배치 처리 (Batch Processing)
- 전체 데이터를 한 번에 처리하지 않고 작은 단위로 나눠서 처리
- **배치 크기**: 한 번에 처리하는 샘플 수 (예: 32)
- **장점**: 메모리 효율, GPU 병렬 처리, 학습 안정성

### 2. 에포크 (Epoch)
- 전체 학습 데이터를 한 번 모두 사용하는 것
- 10 에포크 = 전체 데이터를 10번 반복 학습

### 3. 과적합 (Overfitting)
- 모델이 학습 데이터에만 너무 잘 맞춰져서 새로운 데이터에서는 성능이 떨어지는 현상
- **해결 방법**: Dropout, 정규화, 더 많은 데이터, 조기 종료

### 4. 그래디언트 (Gradient)
- 손실 함수의 기울기
- 가중치를 어느 방향으로 얼마나 조정해야 손실이 줄어드는지 알려줌
- `loss.backward()`로 계산

### 5. 학습률 (Learning Rate)
- 가중치를 업데이트할 때 얼마나 큰 보폭으로 이동할지 결정
- 너무 크면: 수렴하지 못함
- 너무 작으면: 학습이 느림

---

## 🚀 사용 예시

### 학습 시작
```bash
curl -X POST "http://localhost:9003/api/ml/train" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_path": "data/esg_train.csv",
    "target_column": "rating",
    "model_config": {
      "hidden_dims": [128, 64],
      "dropout_rate": 0.2
    },
    "train_config": {
      "epochs": 20,
      "batch_size": 32,
      "lr": 0.001
    },
    "task_type": "classification"
  }'
```

**응답:**
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "message": "학습 작업이 시작되었습니다"
}
```

### 학습 상태 확인
```bash
curl "http://localhost:9003/api/ml/train/status/550e8400-e29b-41d4-a716-446655440000"
```

**응답:**
```json
{
  "status": "running",
  "progress": 50,
  "message": "학습 중... (Epoch 5/10)"
}
```

### 예측 수행
```bash
curl -X POST "http://localhost:9003/api/ml/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "features": [0.5, 0.3, 0.8, 0.2, 0.9, 0.1, 0.7, 0.4, 0.6, 0.3]
    }
  }'
```

**응답:**
```json
{
  "model_id": "model_2024_01_15_001",
  "predicted_class": 1,
  "confidence": 0.85,
  "probabilities": [0.05, 0.85, 0.08, 0.01, 0.01]
}
```

---

## ⚠️ 주의사항

### 1. 데이터 일관성
- 예측 시에도 학습 시와 **동일한 전처리** 적용 필요
- 정규화 스케일러 저장/로드 필요 (현재는 재학습 시마다 재계산)

### 2. 메모리 관리
- 큰 데이터셋의 경우 배치 크기 조정 필요
- GPU 메모리 부족 시 배치 크기 감소 또는 모델 크기 축소

### 3. 모델 버전 관리
- 모델 ID로 버전 관리
- 메타데이터에 학습 날짜, 하이퍼파라미터 저장

### 4. 에러 핸들링
- 데이터 로딩 실패: `DataLoadError`
- 모델 없음: `ModelNotFoundError`
- 학습 실패: `TrainingError`
- 예측 실패: `PredictionError`

---

## 📈 성능 최적화 팁

### 1. 학습 속도 향상
- GPU 사용 (CUDA)
- 배치 크기 증가 (메모리 허용 범위 내)
- 데이터 로더의 `num_workers` 증가

### 2. 모델 성능 향상
- 하이퍼파라미터 튜닝
- 더 깊은 네트워크
- 데이터 증강 (Data Augmentation)

### 3. 메모리 최적화
- 배치 크기 감소
- 모델 양자화 (Quantization)
- 그래디언트 누적 (Gradient Accumulation)

---

## 🔗 관련 파일

- `Dataset.py`: 데이터 관리
- `model.py`: 모델 정의
- `method.py`: 학습 및 평가 메서드
- `service.py`: 비즈니스 로직
- `router.py`: API 엔드포인트
- `exceptions.py`: 커스텀 예외

---

## 📚 참고 자료

- [PyTorch 공식 문서](https://pytorch.org/docs/stable/index.html)
- [FastAPI 공식 문서](https://fastapi.tiangolo.com/)
- [scikit-learn 공식 문서](https://scikit-learn.org/stable/)

