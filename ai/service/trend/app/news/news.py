import os
import requests
from typing import List, Dict, Optional
from datetime import datetime, timedelta


class NewsAPI:
    """NewsAPI를 사용하여 뉴스 기사를 가져오는 클래스"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        NewsAPI 초기화
        
        Args:
            api_key: NewsAPI 키 (없으면 환경변수 NEWSAPI_KEY에서 가져옴)
        """
        self.api_key = api_key or os.getenv("NEWSAPI_KEY")
        if not self.api_key:
            raise ValueError("NewsAPI 키가 필요합니다. 환경변수 NEWSAPI_KEY를 설정하거나 api_key를 전달하세요.")
        
        self.base_url = "https://newsapi.org/v2"
        self.headers = {
            "X-Api-Key": self.api_key
        }
    
    def get_bitcoin_news(self, page_size: int = 20, sort_by: str = "publishedAt") -> Dict:
        """
        비트코인에 관한 최신 기사를 가져옵니다.
        
        Args:
            page_size: 가져올 기사 개수 (기본값: 20)
            sort_by: 정렬 기준 (publishedAt: 최신순, popularity: 인기순, relevancy: 관련도순)
        
        Returns:
            API 응답 딕셔너리
        """
        url = f"{self.base_url}/everything"
        
        # 최근 7일 이내의 기사만 가져오기
        from_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        
        params = {
            "q": "bitcoin",
            "language": "en",
            "sortBy": sort_by,
            "pageSize": page_size,
            "from": from_date
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"NewsAPI 요청 실패: {str(e)}")
    
    def format_news_response(self, api_response: Dict) -> List[Dict]:
        """
        API 응답을 포맷팅합니다.
        
        Args:
            api_response: NewsAPI 응답 딕셔너리
        
        Returns:
            포맷팅된 기사 리스트
        """
        if api_response.get("status") != "ok":
            raise Exception(f"API 오류: {api_response.get('message', '알 수 없는 오류')}")
        
        articles = api_response.get("articles", [])
        formatted_articles = []
        
        for article in articles:
            formatted_article = {
                "title": article.get("title", ""),
                "description": article.get("description", ""),
                "url": article.get("url", ""),
                "urlToImage": article.get("urlToImage", ""),
                "publishedAt": article.get("publishedAt", ""),
                "source": article.get("source", {}).get("name", ""),
                "author": article.get("author", "")
            }
            formatted_articles.append(formatted_article)
        
        return formatted_articles


def get_bitcoin_news_simple(api_key: str, count: int = 20) -> List[Dict]:
    """
    간단한 함수로 비트코인 기사를 가져옵니다.
    
    Args:
        api_key: NewsAPI 키
        count: 가져올 기사 개수 (기본값: 20)
    
    Returns:
        포맷팅된 기사 리스트
    """
    news_api = NewsAPI(api_key=api_key)
    response = news_api.get_bitcoin_news(page_size=count)
    return news_api.format_news_response(response)


# 사용 예시
if __name__ == "__main__":
    # 환경변수에서 API 키 가져오기 또는 직접 입력
    API_KEY = os.getenv("NEWSAPI_KEY", "YOUR_API_KEY_HERE")
    
    try:
        # 방법 1: 클래스 사용
        news_api = NewsAPI(api_key=API_KEY)
        response = news_api.get_bitcoin_news(page_size=20)
        articles = news_api.format_news_response(response)
        
        print(f"총 {len(articles)}개의 기사를 가져왔습니다.\n")
        print("=" * 80)
        
        for i, article in enumerate(articles, 1):
            print(f"\n[{i}] {article['title']}")
            print(f"출처: {article['source']}")
            print(f"작성자: {article['author'] or 'N/A'}")
            print(f"발행일: {article['publishedAt']}")
            print(f"설명: {article['description'] or 'N/A'}")
            print(f"URL: {article['url']}")
            print("-" * 80)
        
        # 방법 2: 간단한 함수 사용
        # articles = get_bitcoin_news_simple(API_KEY, count=20)
        # print(f"총 {len(articles)}개의 기사를 가져왔습니다.")
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")

