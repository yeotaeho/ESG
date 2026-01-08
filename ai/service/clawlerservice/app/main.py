from fastapi import FastAPI, APIRouter
import uvicorn
import os
import redis
import ssl
from .bs_demo.bugsmusic import crawl_bugs_chart
from .sel_demo.danawa import crawl_danawa_products

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

# 서브라우터 생성
clawler_router = APIRouter(prefix="/clawler", tags=["clawler"])

@clawler_router.get("/")
async def clawler_root():
    return {"message": "Clawler Service", "status": "running"}

@clawler_router.get("/bugsmusic")
async def get_bugs_music_chart():
    """벅스 뮤직 실시간 차트 크롤링"""
    try:
        songs = crawl_bugs_chart()
        return {
            "status": "success",
            "count": len(songs),
            "data": songs
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@clawler_router.get("/danawa")
async def get_danawa_products():
    """다나와 제품 리스트 크롤링"""
    try:
        products = crawl_danawa_products()
        return {
            "status": "success",
            "count": len(products),
            "data": products
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

app = FastAPI(
    title="Clawler Service API",
    description="크롤링 서비스 API 문서",
    version="1.0.0"
)

app.include_router(clawler_router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9002)

