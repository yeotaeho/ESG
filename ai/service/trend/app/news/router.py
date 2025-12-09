"""News API 엔드포인트"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import logging

from .news import NewsAPI

logger = logging.getLogger(__name__)

# 라우터 생성
news_router = APIRouter(prefix="/news", tags=["News"])


# 응답 모델 정의
class ArticleResponse(BaseModel):
    """기사 응답 모델"""
    title: str
    description: str
    url: str
    urlToImage: Optional[str]
    publishedAt: str
    source: str
    author: Optional[str]


class NewsResponse(BaseModel):
    """뉴스 응답 모델"""
    status: str
    totalResults: int
    articles: List[ArticleResponse]


@news_router.get("/")
async def news_root():
    """News 서비스 상태 확인"""
    return {
        "message": "News Service",
        "status": "running",
        "endpoints": {
            "bitcoin": "/news/bitcoin",
            "bitcoin_with_params": "/news/bitcoin?page_size=20&sort_by=publishedAt"
        }
    }


@news_router.get("/bitcoin", response_model=NewsResponse)
async def get_bitcoin_news(
    page_size: int = Query(default=20, ge=1, le=100, description="가져올 기사 개수 (1-100)"),
    sort_by: str = Query(default="publishedAt", description="정렬 기준 (publishedAt, popularity, relevancy)")
):
    """
    비트코인에 관한 최신 기사를 가져옵니다.
    
    - **page_size**: 가져올 기사 개수 (기본값: 20, 최대: 100)
    - **sort_by**: 정렬 기준
        - `publishedAt`: 최신순 (기본값)
        - `popularity`: 인기순
        - `relevancy`: 관련도순
    
    Returns:
        비트코인 관련 뉴스 기사 리스트
    """
    try:
        news_api = NewsAPI()
        response = news_api.get_bitcoin_news(page_size=page_size, sort_by=sort_by)
        articles = news_api.format_news_response(response)
        
        return {
            "status": "ok",
            "totalResults": len(articles),
            "articles": articles
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"NewsAPI 요청 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=f"뉴스 가져오기 실패: {str(e)}")

