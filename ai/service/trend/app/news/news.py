import os
import requests
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import time


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
        # API 요청 간 최소 대기 시간 (초) - Rate Limit 방지
        self.request_delay = 0.1
    
    def get_news_by_date_range(
        self,
        keyword: str,
        from_date: str,
        to_date: str,
        language: str = "en",
        sort_by: str = "publishedAt",
        page_size: int = 100,
        fetch_all_pages: bool = True
    ) -> Dict:
        """
        특정 날짜 범위의 뉴스 기사를 가져옵니다 (Pagination 처리 포함).
        
        Args:
            keyword: 검색 키워드 (예: "bitcoin")
            from_date: 시작 날짜 (YYYY-MM-DD 형식)
            to_date: 종료 날짜 (YYYY-MM-DD 형식)
            language: 언어 코드 (기본값: "en")
            sort_by: 정렬 기준 (publishedAt: 최신순, popularity: 인기순, relevancy: 관련도순)
            page_size: 페이지당 기사 개수 (최대 100, 기본값: 100)
            fetch_all_pages: 모든 페이지를 가져올지 여부 (기본값: True)
        
        Returns:
            모든 페이지의 기사를 합친 API 응답 딕셔너리
        """
        url = f"{self.base_url}/everything"
        all_articles = []
        total_results = 0
        current_page = 1
        max_page_size = min(page_size, 100)  # NewsAPI 최대 제한
        
        while True:
            params = {
                "q": keyword,
                "language": language,
                "sortBy": sort_by,
                "pageSize": max_page_size,
                "page": current_page,
                "from": from_date,
                "to": to_date
            }
            
            try:
                response = requests.get(url, headers=self.headers, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                if data.get("status") != "ok":
                    error_msg = data.get("message", "알 수 없는 오류")
                    raise Exception(f"NewsAPI 오류: {error_msg}")
                
                # 첫 페이지에서 totalResults 확인
                if current_page == 1:
                    total_results = data.get("totalResults", 0)
                    print(f"📊 전체 기사 수: {total_results}개 (키워드: {keyword}, 기간: {from_date} ~ {to_date})")
                
                articles = data.get("articles", [])
                if not articles:
                    break
                
                all_articles.extend(articles)
                
                # 모든 페이지를 가져왔는지 확인
                if not fetch_all_pages:
                    break
                
                # 현재 페이지의 기사 수가 page_size보다 적으면 마지막 페이지
                if len(articles) < max_page_size:
                    break
                
                # 이미 수집한 기사 수가 totalResults와 같거나 크면 종료
                if total_results > 0 and len(all_articles) >= total_results:
                    break
                
                current_page += 1
                
                # Rate Limit 방지를 위한 대기
                time.sleep(self.request_delay)
                
            except requests.exceptions.RequestException as e:
                raise Exception(f"NewsAPI 요청 실패 (페이지 {current_page}): {str(e)}")
        
        # 수집된 기사 수 검증
        collected_count = len(all_articles)
        if total_results > 0 and collected_count < total_results * 0.9:
            print(f"⚠️ 경고: 전체 기사 수({total_results}개) 대비 수집된 기사 수({collected_count}개)가 90% 미만입니다.")
            print(f"   데이터 수집이 불완전할 수 있습니다.")
        
        print(f"✅ 총 {collected_count}개의 기사를 수집했습니다. (페이지: {current_page}개)")
        
        return {
            "status": "ok",
            "totalResults": total_results,
            "articles": all_articles,
            "collectedCount": collected_count,
            "pagesFetched": current_page
        }
    
    def get_bitcoin_news(
        self,
        page_size: int = 20,
        sort_by: str = "publishedAt",
        fetch_all_pages: bool = False
    ) -> Dict:
        """
        비트코인에 관한 최신 기사를 가져옵니다.
        
        Args:
            page_size: 가져올 기사 개수 (기본값: 20, 최대: 100)
            sort_by: 정렬 기준 (publishedAt: 최신순, popularity: 인기순, relevancy: 관련도순)
            fetch_all_pages: 모든 페이지를 가져올지 여부 (기본값: False, True일 경우 Pagination 처리)
        
        Returns:
            API 응답 딕셔너리
        """
        # 최근 7일 이내의 기사만 가져오기
        to_date = datetime.now().strftime("%Y-%m-%d")
        from_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        
        # Pagination 처리가 필요한 경우 get_news_by_date_range 사용
        if fetch_all_pages:
            return self.get_news_by_date_range(
                keyword="bitcoin",
                from_date=from_date,
                to_date=to_date,
                language="en",
                sort_by=sort_by,
                page_size=min(page_size, 100),
                fetch_all_pages=True
            )
        
        # 기존 로직 (하위 호환성 유지)
        url = f"{self.base_url}/everything"
        
        params = {
            "q": "bitcoin",
            "language": "en",
            "sortBy": sort_by,
            "pageSize": min(page_size, 100),
            "from": from_date,
            "to": to_date
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
    
    def collect_weekly_articles(
        self,
        keyword: str,
        week_start_date: str,
        week_end_date: str,
        language: str = "en",
        sort_by: str = "publishedAt"
    ) -> Dict:
        """
        특정 주(week)의 기사 건수를 수집합니다 (Pagination 처리 포함).
        
        Args:
            keyword: 검색 키워드 (예: "bitcoin")
            week_start_date: 주의 시작 날짜 (YYYY-MM-DD 형식, 예: "2025-12-01")
            week_end_date: 주의 종료 날짜 (YYYY-MM-DD 형식, 예: "2025-12-07")
            language: 언어 코드 (기본값: "en")
            sort_by: 정렬 기준 (기본값: "publishedAt")
        
        Returns:
            주간 기사 수집 결과 딕셔너리
        """
        print(f"📅 주간 기사 수집 시작: {week_start_date} ~ {week_end_date}")
        
        response = self.get_news_by_date_range(
            keyword=keyword,
            from_date=week_start_date,
            to_date=week_end_date,
            language=language,
            sort_by=sort_by,
            page_size=100,
            fetch_all_pages=True
        )
        
        article_count = response.get("collectedCount", len(response.get("articles", [])))
        total_results = response.get("totalResults", article_count)
        
        return {
            "keyword": keyword,
            "week_start_date": week_start_date,
            "week_end_date": week_end_date,
            "article_count": article_count,
            "total_results": total_results,
            "pages_fetched": response.get("pagesFetched", 1),
            "collection_complete": article_count >= total_results * 0.9 if total_results > 0 else True
        }


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

