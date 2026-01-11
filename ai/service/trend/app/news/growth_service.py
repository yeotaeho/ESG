"""뉴스 빈도 성장률 분석 서비스"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
from .news import NewsAPI
from .growth_rate import GrowthRateCalculator
from .feature_engineering import FeatureEngineer


class GrowthRateService:
    """뉴스 빈도 성장률 분석 서비스 클래스"""
    
    def __init__(self, news_api: Optional[NewsAPI] = None):
        """
        GrowthRateService 초기화
        
        Args:
            news_api: NewsAPI 인스턴스 (없으면 새로 생성)
        """
        self.news_api = news_api or NewsAPI()
        self.growth_calculator = GrowthRateCalculator()
        self.feature_engineer = FeatureEngineer()
    
    def get_week_dates(self, weeks_ago: int = 0) -> Dict[str, str]:
        """
        특정 주차의 시작일과 종료일을 계산합니다.
        
        Args:
            weeks_ago: 몇 주 전인지 (0: 이번주, 1: 1주전, ...)
        
        Returns:
            {"start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD"}
        """
        today = datetime.now()
        
        # 이번 주의 시작일 계산 (월요일 기준)
        days_since_monday = today.weekday()
        this_monday = today - timedelta(days=days_since_monday)
        
        # 목표 주차의 시작일과 종료일
        target_monday = this_monday - timedelta(weeks=weeks_ago)
        target_sunday = target_monday + timedelta(days=6)
        
        return {
            "start_date": target_monday.strftime("%Y-%m-%d"),
            "end_date": target_sunday.strftime("%Y-%m-%d")
        }
    
    def collect_weekly_data(
        self,
        keyword: str,
        weeks: int = 5
    ) -> List[Dict]:
        """
        여러 주차의 기사 건수를 수집합니다.
        
        Zero-value imputation: 0값이 수집된 경우 4주 이동 평균으로 대체합니다.
        
        Args:
            keyword: 검색 키워드
            weeks: 수집할 주차 수 (기본값: 5, 현재주 + 과거 4주)
        
        Returns:
            주차별 기사 수집 결과 리스트 (0값이 imputation된 데이터)
        """
        weekly_data = []
        
        for week_offset in range(weeks):
            week_dates = self.get_week_dates(weeks_ago=week_offset)
            
            try:
                result = self.news_api.collect_weekly_articles(
                    keyword=keyword,
                    week_start_date=week_dates["start_date"],
                    week_end_date=week_dates["end_date"]
                )
                
                weekly_data.append({
                    "week": f"{week_dates['start_date']} ~ {week_dates['end_date']}",
                    "start_date": week_dates["start_date"],
                    "end_date": week_dates["end_date"],
                    "article_count": result["article_count"],
                    "week_number": week_offset,
                    "collection_complete": result["collection_complete"],
                    "is_imputed": False  # 원본 데이터임을 표시
                })
                
            except Exception as e:
                print(f"⚠️ 주차 {week_offset} 데이터 수집 실패: {str(e)}")
                # 실패한 경우에도 빈 데이터 추가 (나중에 imputation)
                weekly_data.append({
                    "week": f"{week_dates['start_date']} ~ {week_dates['end_date']}",
                    "start_date": week_dates["start_date"],
                    "end_date": week_dates["end_date"],
                    "article_count": 0,
                    "week_number": week_offset,
                    "collection_complete": False,
                    "is_imputed": False  # 수집 실패로 인한 0값
                })
        
        # Zero-value imputation: 0값을 4주 이동 평균으로 대체
        weekly_data = self._impute_zero_values(weekly_data)
        
        return weekly_data
    
    def _impute_zero_values(self, weekly_data: List[Dict]) -> List[Dict]:
        """
        Zero-value imputation: 0값을 4주 이동 평균으로 대체합니다.
        
        Args:
            weekly_data: 주차별 기사 수집 결과 리스트
        
        Returns:
            Imputation이 적용된 주차별 데이터 리스트
        """
        # 0이 아닌 값들만 추출하여 평균 계산
        non_zero_counts = [
            item["article_count"] 
            for item in weekly_data 
            if item["article_count"] > 0
        ]
        
        if not non_zero_counts:
            # 모든 값이 0인 경우 (매우 드문 경우)
            print("⚠️ 경고: 모든 주차의 기사 건수가 0입니다. Imputation 불가능.")
            return weekly_data
        
        # 4주 이동 평균 계산 (0이 아닌 값들의 평균)
        moving_average = sum(non_zero_counts) / len(non_zero_counts)
        
        # 0값을 평균으로 대체
        imputed_count = 0
        for item in weekly_data:
            if item["article_count"] == 0:
                original_count = item["article_count"]
                item["article_count"] = int(round(moving_average))
                item["is_imputed"] = True
                imputed_count += 1
                print(f"⚠️ Zero-value imputation: 주차 {item['week_number']} ({item['week']}) "
                      f"기사 건수 {original_count} → {item['article_count']} (4주 평균: {moving_average:.2f})")
        
        if imputed_count > 0:
            print(f"✅ Zero-value imputation 완료: {imputed_count}개 주차 데이터 대체됨")
        
        return weekly_data
    
    def analyze_growth_rate(
        self,
        keyword: str
    ) -> Dict:
        """
        키워드에 대한 뉴스 빈도 성장률을 분석합니다.
        
        Args:
            keyword: 검색 키워드
        
        Returns:
            성장률 분석 결과 딕셔너리
        """
        print(f"🔍 키워드 '{keyword}'에 대한 성장률 분석 시작...")
        
        # 5주간 데이터 수집 (현재주 + 과거 4주)
        historical_data = self.collect_weekly_data(keyword, weeks=5)
        
        if len(historical_data) < 5:
            raise ValueError("충분한 데이터를 수집하지 못했습니다. 최소 5주간의 데이터가 필요합니다.")
        
        # 현재주와 4주전 데이터 추출
        current_week = {
            "start_date": historical_data[0]["start_date"],
            "end_date": historical_data[0]["end_date"],
            "article_count": historical_data[0]["article_count"]
        }
        
        baseline_week = {
            "start_date": historical_data[4]["start_date"],
            "end_date": historical_data[4]["end_date"],
            "article_count": historical_data[4]["article_count"]
        }
        
        # 성장률 메트릭 계산
        growth_metrics = self.growth_calculator.calculate_growth_metrics(
            baseline_week=baseline_week,
            current_week=current_week
        )
        
        # 시계열 분석
        trend_analysis = self.growth_calculator.analyze_historical_data(historical_data)
        
        # Feature Engineering (ML 입력 데이터 생성)
        feature_vector = self.feature_engineer.create_feature_vector(
            current_week_count=growth_metrics["current_count"],
            baseline_week_count=growth_metrics["baseline_count"],
            growth_rate_percentage=growth_metrics["growth_rate_percentage"],
            absolute_change=growth_metrics["absolute_change"],
            historical_data=historical_data
        )
        
        # 최종 결과 구성
        result = {
            "status": "success",
            "keyword": keyword,
            "analysis_period": {
                "current_week": f"{current_week['start_date']} ~ {current_week['end_date']}",
                "baseline_week": f"{baseline_week['start_date']} ~ {baseline_week['end_date']}"
            },
            "growth_metrics": growth_metrics,
            "historical_data": historical_data,
            "trend_analysis": trend_analysis,
            "ml_features": feature_vector
        }
        
        print(f"✅ 분석 완료: 성장률 {growth_metrics['growth_rate_percentage']}% ({growth_metrics['direction']})")
        
        return result

