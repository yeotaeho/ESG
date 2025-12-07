from fastapi import FastAPI
import uvicorn
import logging
import os

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# ML 라우터 import
from app.testlearning.router import ml_router

app = FastAPI(
    title="ML Service API",
    description="머신러닝 모델 학습 및 예측 서비스",
    version="1.0.0"
)

# 라우터 등록
app.include_router(ml_router)

@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "ML Service",
        "status": "running",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 9003))
    uvicorn.run(app, host="0.0.0.0", port=port)

