"""뉴스 빈도 성장률 계산 모듈"""

from typing import List, Dict, Optional
from datetime import datetime, timedelta
import statistics


class GrowthRateCalculator:
    """주간 기사 건수 비교 및 성장률 계산 클래스"""
    
    def __init__(self):
        """GrowthRateCalculator 초기화"""
        pass
    
    def calculate_growth_rate(
        self,
        baseline_count: int,
        current_count: int
    ) -> Dict:
        """
        기본 성장률을 계산합니다.
        
        Zero-division handling: baseline_count가 0일 때 epsilon을 사용하여 계산하고,
        결과값이 지나치게 크면 capping을 적용합니다.
        
        Args:
            baseline_count: 기준 주차(4주전) 기사 건수
            current_count: 현재 주차 기사 건수
        
        Returns:
            성장률 메트릭 딕셔너리
        """
        # Zero-division handling: epsilon 사용
        EPSILON = 1e-6
        MAX_GROWTH_RATE = 9999.0  # 최대 성장률 캡 (10,000% 초과 시)
        
        absolute_change = current_count - baseline_count
        
        # baseline_count가 0인 경우 epsilon 처리
        if baseline_count == 0:
            if current_count == 0:
                # 둘 다 0인 경우
                percentage = 0.0
                relative_change = 0.0
                direction = "stable"
            else:
                # 0에서 증가한 경우: epsilon 사용
                adjusted_baseline = EPSILON
                percentage = (absolute_change / adjusted_baseline) * 100
                relative_change = (current_count / adjusted_baseline) - 1
                
                # Capping: 지나치게 큰 값은 최대값으로 제한
                if percentage > MAX_GROWTH_RATE:
                    percentage = MAX_GROWTH_RATE
                    relative_change = (MAX_GROWTH_RATE / 100)
                
                direction = "increase"
        else:
            # 정상적인 경우
            percentage = (absolute_change / baseline_count) * 100
            relative_change = (current_count / baseline_count) - 1
            
            # Capping: 지나치게 큰 값은 최대값으로 제한
            if percentage > MAX_GROWTH_RATE:
                percentage = MAX_GROWTH_RATE
                relative_change = (MAX_GROWTH_RATE / 100)
        
        # 증감 방향 결정
        if percentage > 0:
            direction = "increase"
        elif percentage < 0:
            direction = "decrease"
        else:
            direction = "stable"
        
        return {
            "percentage": round(percentage, 2),
            "absolute_change": absolute_change,
            "relative_change": round(relative_change, 4),
            "direction": direction
        }
    
    def calculate_weekly_average(self, article_count: int, days: int = 7) -> float:
        """
        주간 평균 기사 건수를 계산합니다.
        
        Args:
            article_count: 주간 기사 건수
            days: 기간 일수 (기본값: 7일)
        
        Returns:
            일평균 기사 건수
        """
        if days == 0:
            return 0.0
        return round(article_count / days, 2)
    
    def calculate_growth_metrics(
        self,
        baseline_week: Dict,
        current_week: Dict
    ) -> Dict:
        """
        주간 기사 건수 비교 및 성장률 메트릭을 계산합니다.
        
        Args:
            baseline_week: 기준 주차 정보 {
                "start_date": "YYYY-MM-DD",
                "end_date": "YYYY-MM-DD",
                "article_count": int
            }
            current_week: 현재 주차 정보 {
                "start_date": "YYYY-MM-DD",
                "end_date": "YYYY-MM-DD",
                "article_count": int
            }
        
        Returns:
            성장률 메트릭 딕셔너리
        """
        baseline_count = baseline_week.get("article_count", 0)
        current_count = current_week.get("article_count", 0)
        
        # 기본 성장률 계산
        growth_rate = self.calculate_growth_rate(baseline_count, current_count)
        
        # 주간 평균 계산
        baseline_start = datetime.strptime(baseline_week["start_date"], "%Y-%m-%d")
        baseline_end = datetime.strptime(baseline_week["end_date"], "%Y-%m-%d")
        baseline_days = (baseline_end - baseline_start).days + 1
        
        current_start = datetime.strptime(current_week["start_date"], "%Y-%m-%d")
        current_end = datetime.strptime(current_week["end_date"], "%Y-%m-%d")
        current_days = (current_end - current_start).days + 1
        
        weekly_average_baseline = self.calculate_weekly_average(baseline_count, baseline_days)
        weekly_average_current = self.calculate_weekly_average(current_count, current_days)
        
        return {
            "baseline_count": baseline_count,
            "current_count": current_count,
            "absolute_change": growth_rate["absolute_change"],
            "growth_rate_percentage": growth_rate["percentage"],
            "direction": growth_rate["direction"],
            "weekly_average_baseline": weekly_average_baseline,
            "weekly_average_current": weekly_average_current
        }
    
    def analyze_historical_data(
        self,
        historical_data: List[Dict]
    ) -> Dict:
        """
        과거 4주간 데이터를 분석하여 시계열 특징을 추출합니다.
        
        Args:
            historical_data: 과거 주차별 기사 건수 리스트 [
                {
                    "week": "YYYY-MM-DD ~ YYYY-MM-DD",
                    "article_count": int,
                    "week_number": int  # 0: 현재, 1: 1주전, ...
                },
                ...
            ]
        
        Returns:
            시계열 분석 결과 딕셔너리
        """
        if not historical_data or len(historical_data) < 2:
            return {
                "moving_average_4weeks": 0.0,
                "trend_direction": "unknown",
                "volatility": 0.0,
                "momentum": 0.0
            }
        
        # 주차별 기사 건수 추출 (최신순으로 정렬)
        counts = [item["article_count"] for item in sorted(historical_data, key=lambda x: x["week_number"])]
        
        # 4주 이동 평균
        if len(counts) >= 4:
            moving_average = statistics.mean(counts[:4])
        else:
            moving_average = statistics.mean(counts)
        
        # 추세 방향 결정
        if len(counts) >= 2:
            recent_trend = counts[0] - counts[1]  # 현재 vs 1주전
            if recent_trend > 0:
                trend_direction = "increasing"
            elif recent_trend < 0:
                trend_direction = "decreasing"
            else:
                trend_direction = "stable"
        else:
            trend_direction = "unknown"
        
        # 변동성 (표준편차)
        if len(counts) >= 2:
            volatility = statistics.stdev(counts) if len(counts) > 1 else 0.0
        else:
            volatility = 0.0
        
        # 모멘텀 (최근 2주 평균 vs 이전 2주 평균)
        if len(counts) >= 4:
            recent_2weeks_avg = statistics.mean(counts[:2])
            previous_2weeks_avg = statistics.mean(counts[2:4])
            momentum = recent_2weeks_avg - previous_2weeks_avg
        elif len(counts) >= 2:
            momentum = counts[0] - counts[1]
        else:
            momentum = 0.0
        
        # 변동성 레벨 분류
        if volatility == 0:
            volatility_level = "none"
        elif volatility < statistics.mean(counts) * 0.1:
            volatility_level = "low"
        elif volatility < statistics.mean(counts) * 0.3:
            volatility_level = "medium"
        else:
            volatility_level = "high"
        
        # 모멘텀 강도 분류
        if abs(momentum) < statistics.mean(counts) * 0.05:
            momentum_level = "weak"
        elif abs(momentum) < statistics.mean(counts) * 0.15:
            momentum_level = "moderate"
        else:
            momentum_level = "strong"
        
        return {
            "moving_average_4weeks": round(moving_average, 2),
            "trend_direction": trend_direction,
            "volatility": round(volatility, 2),
            "volatility_level": volatility_level,
            "momentum": round(momentum, 2),
            "momentum_level": momentum_level
        }

