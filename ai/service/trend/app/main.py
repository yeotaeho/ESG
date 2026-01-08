from fastapi import FastAPI
import uvicorn
import logging
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# News 라우터 import
from app.news.router import news_router

app = FastAPI(
    title="Trend Service API",
    description="트렌드 분석 및 뉴스 서비스",
    version="1.0.0"
)

# 라우터 등록
app.include_router(news_router)

@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "Trend Service",
        "status": "running",
        "version": "1.0.0",
        "docs": "/docs"
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 9004))
    uvicorn.run(app, host="0.0.0.0", port=port)

