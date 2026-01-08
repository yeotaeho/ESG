# GPT 기반 챗봇 AI 구현 전략

## 전략 개요
프론트엔드에서 받은 메시지를 OpenAI GPT API로 전달하여 자연스러운 대화형 AI를 구현합니다.

---

## 1. OpenAI API 연동 방식

### 기본 구조
```python
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@chatbot_router.post("/chat")
async def chat(request: ChatRequest):
    # OpenAI API 호출
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # 또는 "gpt-4", "gpt-3.5-turbo"
        messages=[
            {"role": "system", "content": "당신은 친절한 AI 어시스턴트입니다."},
            {"role": "user", "content": request.message}
        ]
    )
    
    ai_response = response.choices[0].message.content
    return {"response": ai_response}
```

### 모델 선택
- **gpt-4o**: 최신, 빠르고 성능 좋음 (권장)
- **gpt-4o-mini**: 가볍고 저렴함 (권장)
- **gpt-4**: 고성능, 비쌈
- **gpt-3.5-turbo**: 저렴, 빠름

---

## 2. 대화 히스토리 관리 (선택사항)

### 왜 필요한가?
- 이전 대화를 기억하여 더 자연스러운 대화
- 문맥을 이해하고 연속된 질문에 답변

### 구현 방법

#### Pydantic 모델 수정
```python
class ChatMessage(BaseModel):
    role: str  # "user" 또는 "assistant"
    content: str

class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []  # 이전 대화 내역
```

#### 엔드포인트 수정
```python
@chatbot_router.post("/chat")
async def chat(request: ChatRequest):
    messages = [
        {"role": "system", "content": "당신은 친절한 AI 어시스턴트입니다."}
    ]
    
    # 이전 대화 추가
    for msg in request.history:
        messages.append({"role": msg.role, "content": msg.content})
    
    # 현재 질문 추가
    messages.append({"role": "user", "content": request.message})
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    
    ai_message = response.choices[0].message.content
    
    return {
        "response": ai_message,
        "updated_history": messages + [{"role": "assistant", "content": ai_message}]
    }
```

#### 프론트엔드 연동
프론트에서는:
1. 첫 메시지: `{ message: "안녕", history: [] }`
2. 두 번째 메시지: `{ message: "이름 뭐야?", history: [...이전대화] }`

---

## 3. 시스템 프롬프트 커스터마이징

### 기본 프롬프트
```python
SYSTEM_PROMPT = """
당신은 전문적이고 친절한 AI 어시스턴트입니다.
- 질문에 명확하고 간결하게 답변합니다
- 한국어로 자연스럽게 대화합니다
- 모르는 것은 솔직히 모른다고 말합니다
"""
```

### 전문 도메인 예시
```python
# 고객 서비스 챗봇
SYSTEM_PROMPT = """
당신은 우리 회사의 고객 지원 AI입니다.
- 항상 정중하고 친절하게 응대합니다
- 제품 관련 질문에 정확히 답변합니다
- 복잡한 문제는 상담원 연결을 안내합니다
"""

# 기술 지원 챗봇
SYSTEM_PROMPT = """
당신은 개발자를 돕는 기술 지원 AI입니다.
- 코드 예제를 제공할 때는 마크다운 형식을 사용합니다
- 기술적으로 정확한 답변을 제공합니다
- 필요시 공식 문서 링크를 안내합니다
"""
```

---

## 4. 에러 핸들링

### 기본 에러 처리
```python
from openai import OpenAIError

@chatbot_router.post("/chat")
async def chat(request: ChatRequest):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": request.message}
            ]
        )
        return {"response": response.choices[0].message.content}
    
    except OpenAIError as e:
        print(f"[OpenAI API 오류] {e}")
        return {"response": "죄송합니다. AI 서비스에 일시적인 문제가 발생했습니다."}
    
    except Exception as e:
        print(f"[예상치 못한 오류] {e}")
        return {"response": "오류가 발생했습니다. 잠시 후 다시 시도해주세요."}
```

### 타임아웃 처리
```python
import asyncio
from openai import AsyncOpenAI

async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@chatbot_router.post("/chat")
async def chat(request: ChatRequest):
    try:
        response = await asyncio.wait_for(
            async_client.chat.completions.create(...),
            timeout=30.0  # 30초 타임아웃
        )
        return {"response": response.choices[0].message.content}
    
    except asyncio.TimeoutError:
        return {"response": "응답 시간이 초과되었습니다. 다시 시도해주세요."}
```

---

## 5. API 파라미터 최적화

### 주요 파라미터
```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    
    # 창의성 조절 (0~2)
    # 0: 일관되고 예측 가능 (사실 기반 답변)
    # 1: 균형 (일반 대화)
    # 2: 매우 창의적 (스토리텔링)
    temperature=0.7,
    
    # 최대 토큰 수 (응답 길이 제한)
    max_tokens=1000,
    
    # 핵 샘플링 (0~1, 보통 1.0 사용)
    top_p=1.0,
    
    # 반복 줄이기 (0~2)
    frequency_penalty=0.0,
    
    # 주제 다양성 (0~2)
    presence_penalty=0.0,
)
```

### 용도별 설정

#### 정확한 정보 제공 (FAQ, 기술 지원)
```python
temperature=0.3
max_tokens=500
frequency_penalty=0.0
presence_penalty=0.0
```

#### 일반 대화
```python
temperature=0.7
max_tokens=1000
frequency_penalty=0.0
presence_penalty=0.0
```

#### 창의적 작문
```python
temperature=1.2
max_tokens=2000
frequency_penalty=0.5
presence_penalty=0.6
```

---

## 6. 구현 순서

### Step 1: 환경 설정 ✅
- [x] `requirements.txt`에 `openai`, `python-dotenv` 추가
- [x] `.env` 파일에 `OPENAI_API_KEY` 설정
- [x] `docker-compose.yaml`에 환경변수 추가

### Step 2: 기본 구현
```python
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = "당신은 친절한 AI 어시스턴트입니다."

@chatbot_router.post("/chat")
async def chat(request: ChatRequest):
    print(f"[수신된 메시지] {request.message}")
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": request.message}
        ],
        temperature=0.7,
        max_tokens=1000
    )
    
    ai_response = response.choices[0].message.content
    print(f"[AI 응답] {ai_response}")
    
    return {"response": ai_response}
```

### Step 3: 에러 핸들링 추가
```python
from openai import OpenAIError

@chatbot_router.post("/chat")
async def chat(request: ChatRequest):
    try:
        # ... API 호출
        return {"response": ai_response}
    except OpenAIError as e:
        print(f"[오류] {e}")
        return {"response": "일시적인 오류가 발생했습니다."}
```

### Step 4: 대화 히스토리 (선택)
- Pydantic 모델에 `history` 필드 추가
- messages 배열에 이전 대화 포함
- 프론트엔드에서 대화 기록 관리

### Step 5: 테스트 및 최적화
- 다양한 질문으로 테스트
- 프롬프트 조정
- 파라미터 튜닝

---

## 7. 비용 최적화 팁

### 모델 선택
- 개발/테스트: `gpt-4o-mini` (저렴)
- 프로덕션: 용도에 따라 `gpt-4o` 또는 `gpt-4o-mini`

### 토큰 사용 줄이기
```python
# 히스토리 제한 (최근 10개 대화만)
history = request.history[-10:]

# 시스템 프롬프트 간결하게
SYSTEM_PROMPT = "친절한 AI 어시스턴트"

# max_tokens 적절히 설정
max_tokens=500  # 너무 크지 않게
```

### 캐싱 (고급)
- 자주 묻는 질문은 캐싱
- Redis 등 사용

---

## 8. 보안 고려사항

### API 키 보호
- ✅ 환경변수 사용 (하드코딩 금지)
- ✅ `.env` 파일 `.gitignore`에 추가
- Docker secrets 사용 (프로덕션)

### 입력 검증
```python
class ChatRequest(BaseModel):
    message: str
    
    @validator('message')
    def validate_message(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('메시지는 비어있을 수 없습니다')
        if len(v) > 10000:
            raise ValueError('메시지가 너무 깁니다')
        return v
```

### Rate Limiting
```python
from fastapi import HTTPException
from collections import defaultdict
import time

# 간단한 rate limiter
rate_limit = defaultdict(list)

@chatbot_router.post("/chat")
async def chat(request: Request):
    client_ip = request.client.host
    now = time.time()
    
    # 최근 1분간 요청 확인
    rate_limit[client_ip] = [t for t in rate_limit[client_ip] if now - t < 60]
    
    if len(rate_limit[client_ip]) >= 10:  # 분당 10회 제한
        raise HTTPException(status_code=429, detail="Too many requests")
    
    rate_limit[client_ip].append(now)
    
    # ... 정상 처리
```

---

## 9. 완성된 예제 코드

```python
from fastapi import FastAPI, APIRouter, HTTPException
from pydantic import BaseModel, validator
from openai import OpenAI, OpenAIError
import uvicorn
import os
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# OpenAI 클라이언트
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 시스템 프롬프트
SYSTEM_PROMPT = """
당신은 친절하고 전문적인 AI 어시스턴트입니다.
- 질문에 명확하고 간결하게 답변합니다
- 한국어로 자연스럽게 대화합니다
- 모르는 것은 솔직히 말합니다
"""

# Pydantic 모델
class ChatRequest(BaseModel):
    message: str
    
    @validator('message')
    def validate_message(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('메시지를 입력해주세요')
        if len(v) > 5000:
            raise ValueError('메시지가 너무 깁니다')
        return v

# 라우터
chatbot_router = APIRouter(prefix="/chatbot", tags=["chatbot"])

@chatbot_router.get("/")
async def chatbot_root():
    return {"message": "Chatbot Service", "status": "running"}

@chatbot_router.post("/chat")
async def chat(request: ChatRequest):
    print(f"[수신된 메시지] {request.message}")
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": request.message}
            ],
            temperature=0.7,
            max_tokens=1000,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        
        ai_response = response.choices[0].message.content
        print(f"[AI 응답] {ai_response}")
        
        return {"response": ai_response}
    
    except OpenAIError as e:
        print(f"[OpenAI API 오류] {e}")
        return {"response": "죄송합니다. AI 서비스에 일시적인 문제가 발생했습니다."}
    
    except Exception as e:
        print(f"[예상치 못한 오류] {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# FastAPI 앱
app = FastAPI(
    title="Chatbot Service API",
    description="GPT 기반 챗봇 서비스",
    version="1.0.0"
)

app.include_router(chatbot_router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9001)
```

---

## 10. 테스트 방법

### curl 테스트
```bash
curl -X POST http://localhost:9000/chatbot/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "안녕하세요!"}'
```

### Python 테스트
```python
import requests

response = requests.post(
    "http://localhost:9000/chatbot/chat",
    json={"message": "파이썬으로 Hello World 출력하는 방법 알려줘"}
)

print(response.json())
```

---

## 참고 자료
- [OpenAI API 문서](https://platform.openai.com/docs/api-reference)
- [FastAPI 문서](https://fastapi.tiangolo.com/)
- [OpenAI Python 라이브러리](https://github.com/openai/openai-python)

