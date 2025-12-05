from fastapi import FastAPI, APIRouter
from pydantic import BaseModel
from openai import OpenAI, OpenAIError
import uvicorn
import os
import redis
import ssl

# OpenAI 클라이언트
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Upstash Redis 클라이언트
redis_host = os.getenv("UPSTASH_REDIS_HOST", "unified-gibbon-39731.upstash.io")
redis_port = int(os.getenv("UPSTASH_REDIS_PORT", "6379"))
redis_token = os.getenv("UPSTASH_REDIS_TOKEN", "")

redis_client = redis.Redis(
    host=redis_host,
    port=redis_port,
    password=redis_token,
    ssl=True,
    ssl_cert_reqs=ssl.CERT_NONE,
    decode_responses=True
)

# 시스템 프롬프트
SYSTEM_PROMPT = """
당신은 친절하고 전문적인 AI 어시스턴트입니다.
- 질문에 명확하고 간결하게 답변합니다
- 한국어로 자연스럽게 대화합니다
- 모르는 것은 솔직히 말합니다
- 모든 질문을 esg 와 연관지어 답변합니다
"""

class ChatRequest(BaseModel):
    message: str

# 서브라우터 생성
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
            max_tokens=1000
        )
        
        ai_response = response.choices[0].message.content
        print(f"[AI 응답] {ai_response}")
        
        return {"response": ai_response}
    
    except OpenAIError as e:
        print(f"[OpenAI API 오류] {e}")
        return {"response": "죄송합니다. AI 서비스에 일시적인 문제가 발생했습니다."}
    
    except Exception as e:
        print(f"[예상치 못한 오류] {e}")
        return {"response": "오류가 발생했습니다. 잠시 후 다시 시도해주세요."}

app = FastAPI(
    title="Chatbot Service API",
    description="GPT 기반 챗봇 서비스",
    version="1.0.0"
)

app.include_router(chatbot_router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9001)
