"""Feature Engineering 모듈 - ML 모델 입력 데이터 생성"""

from typing import List, Dict, Optional
import statistics


class FeatureEngineer:
    """머신러닝 모델을 위한 특징 추출 클래스"""
    
    def __init__(self):
        """FeatureEngineer 초기화"""
        pass
    
    def extract_basic_features(
        self,
        current_week_count: int,
        baseline_week_count: int,
        growth_rate_percentage: float,
        absolute_change: int
    ) -> Dict:
        """
        기본 특징을 추출합니다.
        
        Args:
            current_week_count: 이번주 기사 건수
            baseline_week_count: 4주전 기사 건수
            growth_rate_percentage: 성장률 (%)
            absolute_change: 절대 변화량
        
        Returns:
            기본 특징 딕셔너리
        """
        return {
            "current_week_count": current_week_count,
            "baseline_week_count": baseline_week_count,
            "growth_rate": growth_rate_percentage,
            "absolute_change": absolute_change
        }
    
    def extract_time_series_features(
        self,
        historical_data: List[Dict]
    ) -> Dict:
        """
        시계열 특징을 추출합니다.
        
        ML Features 매핑 일관성 확보:
        - count_W0_current: 이번 주 (T, week_number: 0)
        - count_W1_t_minus_1: 1주 전 (T-1, week_number: 1)
        - count_W2_t_minus_2: 2주 전 (T-2, week_number: 2)
        - count_W3_t_minus_3: 3주 전 (T-3, week_number: 3)
        - count_W4_baseline: 4주 전 (T-4, week_number: 4)
        
        Args:
            historical_data: 과거 주차별 기사 건수 리스트 [
                {
                    "week_number": int,  # 0: 현재, 1: 1주전, 2: 2주전, ...
                    "article_count": int
                },
                ...
            ]
        
        Returns:
            시계열 특징 딕셔너리
        """
        if not historical_data:
            return {
                "count_W0_current": 0,
                "count_W1_t_minus_1": 0,
                "count_W2_t_minus_2": 0,
                "count_W3_t_minus_3": 0,
                "count_W4_baseline": 0,
                "moving_average_4weeks": 0.0,
                "trend_direction": 0  # 0: unknown, 1: increase, -1: decrease, 0: stable
            }
        
        # 주차별로 정렬 (week_number 기준, 오름차순: 0, 1, 2, 3, 4)
        sorted_data = sorted(historical_data, key=lambda x: x.get("week_number", 0))
        
        # week_number를 기준으로 명시적으로 매핑
        # week_number 0 = 현재주, 1 = 1주전, 2 = 2주전, 3 = 3주전, 4 = 4주전
        count_W0_current = 0
        count_W1_t_minus_1 = 0
        count_W2_t_minus_2 = 0
        count_W3_t_minus_3 = 0
        count_W4_baseline = 0
        
        for item in sorted_data:
            week_number = item.get("week_number", -1)
            article_count = item.get("article_count", 0)
            
            if week_number == 0:
                count_W0_current = article_count
            elif week_number == 1:
                count_W1_t_minus_1 = article_count
            elif week_number == 2:
                count_W2_t_minus_2 = article_count
            elif week_number == 3:
                count_W3_t_minus_3 = article_count
            elif week_number == 4:
                count_W4_baseline = article_count
        
        # 4주 이동 평균 (week_number 1, 2, 3, 4의 평균, baseline 제외)
        baseline_counts = [
            count_W1_t_minus_1,
            count_W2_t_minus_2,
            count_W3_t_minus_3,
            count_W4_baseline
        ]
        moving_average = statistics.mean(baseline_counts) if baseline_counts else 0.0
        
        # 추세 방향 (현재주 vs 1주전)
        if count_W0_current > 0 and count_W1_t_minus_1 > 0:
            recent_change = count_W0_current - count_W1_t_minus_1
            if recent_change > 0:
                trend_direction = 1  # 증가
            elif recent_change < 0:
                trend_direction = -1  # 감소
            else:
                trend_direction = 0  # 유지
        else:
            trend_direction = 0  # 알 수 없음
        
        return {
            "count_W0_current": count_W0_current,
            "count_W1_t_minus_1": count_W1_t_minus_1,
            "count_W2_t_minus_2": count_W2_t_minus_2,
            "count_W3_t_minus_3": count_W3_t_minus_3,
            "count_W4_baseline": count_W4_baseline,
            "moving_average_4weeks": round(moving_average, 2),
            "trend_direction": trend_direction
        }
    
    def extract_derived_features(
        self,
        historical_data: List[Dict]
    ) -> Dict:
        """
        파생 특징을 추출합니다.
        
        Args:
            historical_data: 과거 주차별 기사 건수 리스트
        
        Returns:
            파생 특징 딕셔너리
        """
        if not historical_data or len(historical_data) < 2:
            return {
                "week_over_week_change": 0.0,
                "volatility": 0.0,
                "momentum": 0.0
            }
        
        # 주차별로 정렬
        sorted_data = sorted(historical_data, key=lambda x: x.get("week_number", 0))
        counts = [item["article_count"] for item in sorted_data]
        
        # 주간 변화율 (이번주 vs 1주전)
        # Zero-division handling: epsilon 사용
        EPSILON = 1e-6
        MAX_CHANGE_RATE = 9999.0
        
        if len(counts) >= 2:
            current_count = counts[0]
            previous_count = counts[1]
            if previous_count > 0:
                week_over_week_change = ((current_count - previous_count) / previous_count) * 100
            else:
                # 0으로 나누기 방지: epsilon 사용
                adjusted_previous = EPSILON
                week_over_week_change = ((current_count - adjusted_previous) / adjusted_previous) * 100
                if week_over_week_change > MAX_CHANGE_RATE:
                    week_over_week_change = MAX_CHANGE_RATE
        else:
            week_over_week_change = 0.0
        
        # 변동성 (표준편차)
        if len(counts) >= 2:
            volatility = statistics.stdev(counts) if len(counts) > 1 else 0.0
        else:
            volatility = 0.0
        
        # 모멘텀 (최근 2주 평균 vs 이전 2주 평균)
        # Zero-division handling: epsilon 사용
        EPSILON = 1e-6
        MAX_MOMENTUM_RATE = 9999.0
        
        if len(counts) >= 4:
            recent_2weeks_avg = statistics.mean(counts[:2])
            previous_2weeks_avg = statistics.mean(counts[2:4])
            if previous_2weeks_avg > 0:
                momentum = ((recent_2weeks_avg - previous_2weeks_avg) / previous_2weeks_avg * 100)
            else:
                # 0으로 나누기 방지: epsilon 사용
                adjusted_previous = EPSILON
                momentum = ((recent_2weeks_avg - adjusted_previous) / adjusted_previous * 100)
                if momentum > MAX_MOMENTUM_RATE:
                    momentum = MAX_MOMENTUM_RATE
        elif len(counts) >= 2:
            momentum = week_over_week_change
        else:
            momentum = 0.0
        
        return {
            "week_over_week_change": round(week_over_week_change, 2),
            "volatility": round(volatility, 2),
            "momentum": round(momentum, 2)
        }
    
    def create_feature_vector(
        self,
        current_week_count: int,
        baseline_week_count: int,
        growth_rate_percentage: float,
        absolute_change: int,
        historical_data: List[Dict]
    ) -> Dict:
        """
        전체 특징 벡터를 생성합니다.
        
        Args:
            current_week_count: 이번주 기사 건수
            baseline_week_count: 4주전 기사 건수
            growth_rate_percentage: 성장률 (%)
            absolute_change: 절대 변화량
            historical_data: 과거 주차별 기사 건수 리스트
        
        Returns:
            전체 특징 벡터 딕셔너리
        """
        basic_features = self.extract_basic_features(
            current_week_count,
            baseline_week_count,
            growth_rate_percentage,
            absolute_change
        )
        
        time_series_features = self.extract_time_series_features(historical_data)
        
        derived_features = self.extract_derived_features(historical_data)
        
        return {
            **basic_features,
            **time_series_features,
            **derived_features
        }

