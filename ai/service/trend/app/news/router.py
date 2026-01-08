"""News API 엔드포인트"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import logging

from .news import NewsAPI
from .growth_service import GrowthRateService

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
            "bitcoin_with_params": "/news/bitcoin?page_size=20&sort_by=publishedAt",
            "growth_rate": "/news/growth-rate/{keyword}",
            "prediction": "/news/prediction/{keyword}"
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


@news_router.get("/growth-rate/{keyword}")
async def get_growth_rate(keyword: str):
    """
    특정 키워드에 대한 뉴스 빈도 성장률을 분석합니다.
    
    - **keyword**: 분석할 키워드 (예: "bitcoin", "ethereum")
    
    Returns:
        성장률 분석 결과 (4주전 대비 현재주 기사 건수 변화율)
    """
    try:
        growth_service = GrowthRateService()
        result = growth_service.analyze_growth_rate(keyword=keyword)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"성장률 분석 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=f"성장률 분석 실패: {str(e)}")


@news_router.get("/prediction/{keyword}")
async def get_prediction(keyword: str):
    """
    특정 키워드에 대한 뉴스 빈도 예측 결과를 조회합니다.
    
    - **keyword**: 예측할 키워드 (예: "bitcoin", "ethereum")
    
    Returns:
        ML 예측 결과 (현재는 성장률 분석 결과와 동일, 향후 ML 모델 연동 예정)
    """
    try:
        growth_service = GrowthRateService()
        result = growth_service.analyze_growth_rate(keyword=keyword)
        
        # ML 예측 결과 구조 추가 (현재는 기본값, 향후 실제 ML 모델 연동 필요)
        result["ml_prediction"] = {
            "next_week_predicted_count": int(result["growth_metrics"]["current_count"] * 1.05),
            "next_week_predicted_growth_rate": 5.0,
            "confidence": 0.75,
            "model_version": "v1.0",
            "note": "현재는 기본 예측값입니다. 실제 ML 모델 연동 필요."
        }
        
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"예측 분석 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=f"예측 분석 실패: {str(e)}")

