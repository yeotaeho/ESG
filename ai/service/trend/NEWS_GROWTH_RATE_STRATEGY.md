# 뉴스 빈도 성장률 계산 및 머신러닝 전략

## 📋 개요

NewsAPI에서 받은 뉴스 데이터를 활용하여 **4주전 기사 건수**와 **이번주 기사 건수**를 비교하여 **빈도 성장률**을 계산하고, 이를 머신러닝 모델의 입력 데이터로 활용하는 전략입니다.

---

## 🎯 목표

1. **빈도 성장률 계산**: 주간 기사 건수 변화율 측정
2. **시계열 데이터 구축**: 과거 데이터를 통한 패턴 분석
3. **머신러닝 입력 데이터 생성**: 성장률을 특징(feature)으로 활용
4. **최종 출력**: 성장률 데이터 및 예측 결과 제공

---

## 📊 데이터 수집 전략

### 1단계: 시간대별 기사 수집

```
현재 시점: T (이번주)
- T-1주: 1주 전
- T-2주: 2주 전  
- T-3주: 3주 전
- T-4주: 4주 전 (기준점)
```

**구현 방식:**
- `get_bitcoin_news()` 메서드를 확장하여 `from_date`, `to_date` 파라미터 추가
- 각 주(week)별로 API 호출하여 기사 건수 집계
- 날짜 범위: `(T-4주 시작일) ~ (T-4주 종료일)`, `(T주 시작일) ~ (T주 종료일)`

---

## 📈 빈도 성장률 계산 공식

### 기본 성장률 계산

```python
성장률(%) = ((이번주_기사건수 - 4주전_기사건수) / 4주전_기사건수) * 100

# 예시:
# 4주전 기사 건수: 100개
# 이번주 기사 건수: 150개
# 성장률 = ((150 - 100) / 100) * 100 = 50% 증가
```

### 추가 메트릭

1. **절대 변화량**: `이번주_기사건수 - 4주전_기사건수`
2. **상대 변화율**: `(이번주_기사건수 / 4주전_기사건수) - 1`
3. **주간 평균**: `이번주_기사건수 / 7일`
4. **증감 방향**: `증가(+1)`, `감소(-1)`, `동일(0)`

---

## 🏗️ 데이터 파이프라인 구조

### Phase 1: 데이터 수집 및 집계

```
[NewsAPI] 
  ↓
[주간 기사 수집] (4주전 ~ 이번주)
  ↓
[기사 건수 집계] (주별 카운트)
  ↓
[성장률 계산]
```

**데이터 구조:**
```json
{
  "keyword": "bitcoin",
  "period": {
    "current_week": {
      "start_date": "2025-12-01",
      "end_date": "2025-12-07",
      "article_count": 150
    },
    "baseline_week": {
      "start_date": "2025-11-03",
      "end_date": "2025-11-09",
      "article_count": 100
    }
  },
  "growth_rate": {
    "percentage": 50.0,
    "absolute_change": 50,
    "direction": "increase"
  }
}
```

---

## 🤖 머신러닝 입력 데이터 설계

### Feature Engineering (특징 추출)

#### 1. 기본 특징 (Basic Features)
- `current_week_count`: 이번주 기사 건수
- `baseline_week_count`: 4주전 기사 건수
- `growth_rate`: 성장률 (%)
- `absolute_change`: 절대 변화량

#### 2. 시계열 특징 (Time Series Features)
- `week_1_count`: 1주 전 기사 건수
- `week_2_count`: 2주 전 기사 건수
- `week_3_count`: 3주 전 기사 건수
- `week_4_count`: 4주 전 기사 건수 (기준)
- `moving_average_4weeks`: 4주 이동 평균
- `trend_direction`: 추세 방향 (증가/감소/유지)

#### 3. 파생 특징 (Derived Features)
- `week_over_week_change`: 주간 변화율 (이번주 vs 1주전)
- `volatility`: 변동성 (표준편차)
- `momentum`: 모멘텀 (최근 2주 평균 vs 이전 2주 평균)

---

## 🎯 머신러닝 모델 전략

### 모델 선택 옵션

#### Option 1: 회귀 모델 (Regression)
**목표**: 성장률 예측
- **입력**: 과거 4주간 기사 건수, 현재 성장률
- **출력**: 다음 주 예상 성장률 (연속값)
- **모델**: Linear Regression, Random Forest, LSTM

#### Option 2: 분류 모델 (Classification)
**목표**: 성장 방향 예측
- **입력**: 과거 4주간 기사 건수
- **출력**: 증가/감소/유지 (카테고리)
- **모델**: Logistic Regression, Random Forest, XGBoost

#### Option 3: 시계열 모델 (Time Series)
**목표**: 미래 기사 건수 예측
- **입력**: 과거 N주간 기사 건수 시계열
- **출력**: 다음 주 예상 기사 건수
- **모델**: ARIMA, Prophet, LSTM

---

## 📦 최종 출력 데이터 구조

### API 응답 형식

```json
{
  "status": "success",
  "keyword": "bitcoin",
  "analysis_period": {
    "current_week": "2025-12-01 ~ 2025-12-07",
    "baseline_week": "2025-11-03 ~ 2025-11-09"
  },
  "growth_metrics": {
    "baseline_count": 100,
    "current_count": 150,
    "absolute_change": 50,
    "growth_rate_percentage": 50.0,
    "direction": "increase",
    "weekly_average_current": 21.43,
    "weekly_average_baseline": 14.29
  },
  "historical_data": [
    {
      "week": "2025-11-03 ~ 2025-11-09",
      "article_count": 100,
      "week_number": 4
    },
    {
      "week": "2025-11-10 ~ 2025-11-16",
      "article_count": 120,
      "week_number": 3
    },
    {
      "week": "2025-11-17 ~ 2025-11-23",
      "article_count": 130,
      "week_number": 2
    },
    {
      "week": "2025-11-24 ~ 2025-11-30",
      "article_count": 140,
      "week_number": 1
    },
    {
      "week": "2025-12-01 ~ 2025-12-07",
      "article_count": 150,
      "week_number": 0
    }
  ],
  "ml_prediction": {
    "next_week_predicted_count": 155,
    "next_week_predicted_growth_rate": 3.33,
    "confidence": 0.85,
    "model_version": "v1.0"
  },
  "trend_analysis": {
    "trend": "increasing",
    "volatility": "low",
    "momentum": "strong"
  }
}
```

---

## 🔄 구현 단계별 전략

### Step 1: 데이터 수집 모듈 확장
- `NewsAPI` 클래스에 `get_news_by_date_range()` 메서드 추가
- 주간 단위로 데이터 수집하는 `collect_weekly_articles()` 메서드 추가

### Step 2: 성장률 계산 모듈
- `GrowthRateCalculator` 클래스 생성
- 주간 기사 건수 비교 및 성장률 계산 로직 구현

### Step 3: 데이터 저장 및 관리
- Redis 또는 데이터베이스에 주간 데이터 저장
- 시계열 데이터 누적 관리

### Step 4: Feature Engineering 모듈
- 과거 데이터를 특징 벡터로 변환
- 정규화 및 전처리 로직

### Step 5: 머신러닝 통합
- 기존 ML 서비스와 연동
- 학습 및 예측 파이프라인 구축

### Step 6: API 엔드포인트 추가
- `/news/growth-rate/{keyword}`: 성장률 조회
- `/news/prediction/{keyword}`: ML 예측 결과 조회

---

## 📝 주요 고려사항

### 1. 데이터 품질
- NewsAPI의 무료 플랜 제한 고려 (하루 100 요청)
- 캐싱 전략 필요 (Redis 활용)
- 데이터 누락 시 처리 방안

### 2. 시간대 처리
- UTC 기준 통일
- 주(week) 정의: 월요일 ~ 일요일 또는 일요일 ~ 토요일

### 3. 성능 최적화
- 배치 처리로 여러 주 데이터 한번에 수집
- 비동기 처리 고려

### 4. 확장성
- 다른 키워드로 확장 가능하도록 설계
- 다중 키워드 비교 기능

---

## 🎯 최종 목표

**입력**: NewsAPI 뉴스 데이터  
**처리**: 빈도 성장률 계산 + 머신러닝 예측  
**출력**: 성장률 메트릭 + 예측 결과 + 트렌드 분석

이 전략을 통해 뉴스 빈도 변화를 정량화하고, 머신러닝을 통해 미래 트렌드를 예측할 수 있습니다.

