# Gateway 전환 완료 가이드

## ✅ 완료된 작업

### 1. Discovery → Gateway 전환
- ✅ `build.gradle`: Spring Cloud Gateway 의존성 추가
- ✅ `GatewayApplication.java`: 메인 클래스 변경
- ✅ `application.yaml`: Gateway 라우팅 설정 추가
- ✅ `docker-compose.yaml`: discovery → gateway로 변경
- ✅ 서비스 포트 조정

### 2. 포트 구조 변경

| 서비스 | 이전 포트 | 현재 포트 | 역할 |
|--------|----------|----------|------|
| **Gateway** | 8762 | **8080** | API Gateway (WebFlux) |
| **Common** | 8080 | **8082** | 공통 서비스 |
| **User** | 8081 | **8083** | 사용자 서비스 |
| **Eureka** | 8761 | 8761 | 서비스 레지스트리 (변경 없음) |

## 🚀 사용 방법

### 1. 서비스 시작

```bash
# 전체 스택 시작
docker compose up -d

# 또는 단계별로
docker compose up -d eureka
docker compose up -d config
docker compose up -d gateway
docker compose up -d common user
```

### 2. API 호출 방법

#### 이전 (직접 호출)
```
http://localhost:8082/api/common/...
http://localhost:8083/api/users/...
```

#### 현재 (Gateway를 통한 호출) ⭐
```
http://localhost:8080/api/common/...
http://localhost:8080/api/users/...
```

### 3. 라우팅 규칙

Gateway는 다음 규칙으로 요청을 라우팅합니다:

```
클라이언트 요청: http://localhost:8080/api/users/123
                ↓
Gateway 라우팅: /api/users/** → user-service
                ↓
실제 서비스: http://user-service:8083/users/123
```

## 📋 Gateway 라우팅 설정

### application.yaml 설정

```yaml
spring:
  cloud:
    gateway:
      routes:
        # User Service
        - id: user-service
          uri: lb://user-service
          predicates:
            - Path=/api/users/**
          filters:
            - StripPrefix=1
        
        # Common Service
        - id: common-service
          uri: lb://common-service
          predicates:
            - Path=/api/common/**
          filters:
            - StripPrefix=1
```

### 자동 라우팅 (Discovery Locator)

Eureka에 등록된 모든 서비스는 자동으로 라우팅됩니다:

```
http://localhost:8080/user-service/...
http://localhost:8080/common-service/...
```

## 🔍 확인 방법

### 1. Gateway 상태 확인

```bash
# Gateway 로그 확인
docker logs gateway-service

# Gateway 헬스 체크
curl http://localhost:8080/actuator/health
```

### 2. Eureka 대시보드 확인

```
http://localhost:8761
```

등록된 서비스:
- gateway-service
- common-service
- user-service

### 3. 라우팅 테스트

```bash
# User Service 테스트
curl http://localhost:8080/api/users/test

# Common Service 테스트
curl http://localhost:8080/api/common/test
```

## ⚠️ 주의사항

### 1. 파일 이름 변경 필요

현재 `DiscoveryApplication.java` 파일 이름이 남아있습니다.
IDE에서 파일 이름을 `GatewayApplication.java`로 변경하거나,
클래스 이름과 일치하도록 리팩토링하세요.

### 2. CORS 설정

Gateway에서 CORS가 설정되어 있습니다.
프론트엔드에서 API 호출 시 CORS 에러가 발생하지 않습니다.

### 3. 서비스 직접 접근

서비스는 여전히 직접 접근 가능합니다:
- Common: http://localhost:8082
- User: http://localhost:8083

하지만 **Gateway를 통한 접근을 권장**합니다.

## 🎯 다음 단계 (선택사항)

### 1. 인증/인가 필터 추가

```java
@Component
public class AuthenticationFilter implements GatewayFilter {
    // JWT 토큰 검증 등
}
```

### 2. Rate Limiting

```yaml
spring:
  cloud:
    gateway:
      routes:
        - id: user-service
          filters:
            - name: RequestRateLimiter
              args:
                redis-rate-limiter.replenishRate: 10
                redis-rate-limiter.burstCapacity: 20
```

### 3. 로깅

```yaml
spring:
  cloud:
    gateway:
      routes:
        - id: user-service
          filters:
            - name: Logging
              args:
                level: INFO
```

## 📚 참고 자료

- [Spring Cloud Gateway 공식 문서](https://spring.io/projects/spring-cloud-gateway)
- [WebFlux 공식 문서](https://docs.spring.io/spring-framework/reference/web/webflux.html)

---

**전환 완료!** 이제 Gateway를 통해 모든 API 요청을 라우팅할 수 있습니다. 🎉

