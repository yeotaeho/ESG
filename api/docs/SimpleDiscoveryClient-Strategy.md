# SimpleDiscoveryClient 전략 가이드

## 📋 목차
1. [개요](#개요)
2. [전략 비교](#전략-비교)
3. [구현 방법](#구현-방법)
4. [Docker 환경 설정](#docker-환경-설정)
5. [프로필 기반 전략](#프로필-기반-전략)
6. [장단점 분석](#장단점-분석)
7. [실행 계획](#실행-계획)

---

## 개요

**SimpleDiscoveryClient**는 Spring Cloud LoadBalancer의 기능으로, Eureka 없이도 서비스 디스커버리와 로드밸런싱을 사용할 수 있게 해줍니다.

### 현재 구조
```
┌─────────────┐
│ Eureka      │ (8761) - 서비스 레지스트리
└──────┬──────┘
       │
       ├─── Gateway (8080) - lb://user-service
       ├─── User Service (8083)
       └─── Common Service (8082)
```

### 변경 후 구조
```
Gateway (8080) ──lb://──→ User Service (8083)
              └──lb://──→ Common Service (8082)
              
(Eureka 없이 SimpleDiscoveryClient 사용)
```

---

## 전략 비교

| 기능 | Eureka | SimpleDiscovery | 직접 URL |
|-----|--------|----------------|----------|
| **로드밸런싱** | ✅ 자동 | ✅ 가능 | ❌ 불가 |
| **동적 등록** | ✅ 자동 | ❌ 수동 설정 | ❌ 수동 설정 |
| **헬스체크** | ✅ 자동 | ⚠️ 제한적 | ❌ 없음 |
| **설정 복잡도** | 높음 | 중간 | 낮음 |
| **외부 서버** | 필요 (8761) | 불필요 | 불필요 |
| **확장성** | 높음 | 중간 | 낮음 |
| **리소스 사용** | 높음 | 낮음 | 낮음 |
| **lb:// 사용** | ✅ | ✅ | ❌ |

### 시나리오별 적합도

| 시나리오 | Eureka | SimpleDiscovery | Direct URL |
|---------|--------|----------------|------------|
| **단일 인스턴스** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **다중 인스턴스** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ❌ |
| **동적 스케일** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ❌ |
| **설정 간편성** | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **리소스 사용** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **헬스체크** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ❌ |
| **개발 환경** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **프로덕션** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |

---

## 구현 방법

### STEP 1: 의존성 확인

#### pom.xml (Gateway)

```xml
<!-- Eureka 제거 또는 주석 처리 -->
<!-- 
<dependency>
    <groupId>org.springframework.cloud</groupId>
    <artifactId>spring-cloud-starter-netflix-eureka-client</artifactId>
</dependency>
-->

<!-- LoadBalancer 추가 (보통 Gateway에 이미 포함) -->
<dependency>
    <groupId>org.springframework.cloud</groupId>
    <artifactId>spring-cloud-starter-loadbalancer</artifactId>
</dependency>
```

---

### STEP 2: Gateway application.yaml 수정

#### 기본 설정 (단일 인스턴스)

```yaml
spring:
  application:
    name: gateway-server
  
  cloud:
    # Eureka Discovery 비활성화
    discovery:
      enabled: false
    
    gateway:
      # Eureka 자동 라우팅 비활성화
      discovery:
        locator:
          enabled: false
      
      routes:
        # User Service 라우팅
        - id: user-service
          uri: lb://user-service  # lb:// 프로토콜 유지!
          predicates:
            - Path=/api/user/**
          filters:
            - StripPrefix=1
        
        # Common Service 라우팅
        - id: common-service
          uri: lb://common-service
          predicates:
            - Path=/api/common/**
          filters:
            - StripPrefix=1
      
      # CORS 설정
      globalcors:
        cors-configurations:
          '[/**]':
            allowedOrigins:
              - "http://localhost:3000"
            allowedMethods:
              - GET
              - POST
              - PUT
              - DELETE
              - PATCH
              - OPTIONS
            allowedHeaders: "*"
            allowCredentials: true
            maxAge: 3600

  # ⭐ SimpleDiscoveryClient 설정
  cloud:
    discovery:
      client:
        simple:
          instances:
            # User Service 인스턴스 정의
            user-service:
              - uri: http://localhost:8083
                instance-id: user-service-1
            
            # Common Service 인스턴스 정의
            common-service:
              - uri: http://localhost:8082
                instance-id: common-service-1

server:
  port: 8080

# Eureka 설정 완전 제거
```

#### 로드밸런싱 설정 (다중 인스턴스)

```yaml
spring:
  cloud:
    discovery:
      client:
        simple:
          instances:
            user-service:
              # 인스턴스 1
              - uri: http://localhost:8083
                instance-id: user-service-1
              # 인스턴스 2 (로드밸런싱)
              - uri: http://localhost:8084
                instance-id: user-service-2
              # 인스턴스 3
              - uri: http://localhost:8085
                instance-id: user-service-3
```

---

### STEP 3: User Service application.yaml 수정

```yaml
spring:
  application:
    name: user-service

server:
  port: 8083
  servlet:
    context-path: /api

# Eureka 설정 완전 제거
# eureka 섹션 삭제
```

#### Common Service도 동일하게 수정

```yaml
spring:
  application:
    name: common-service

server:
  port: 8082

# Eureka 설정 완전 제거
```

---

## Docker 환경 설정

### STEP 4: docker-compose.yaml 수정

```yaml
services:
  # ❌ Eureka 서비스 제거
  # eureka:
  #   ...

  # ❌ Config 서비스도 불필요하면 제거
  # config:
  #   ...

  gateway:
    build:
      context: .
      dockerfile: server/discovery/Dockerfile
    container_name: gateway-server
    ports:
      - "8080:8080"
    networks:
      - api-network
    depends_on:
      - user-service
      - common-service
    restart: unless-stopped
    environment:
      - SPRING_PROFILES_ACTIVE=docker
      - USER_SERVICE_HOST=user-service
      - USER_SERVICE_PORT=8083
      - COMMON_SERVICE_HOST=common-service
      - COMMON_SERVICE_PORT=8082

  common:
    build:
      context: .
      dockerfile: service/common/Dockerfile
    container_name: common-service
    ports:
      - "8082:8082"
    networks:
      - api-network
    restart: unless-stopped
    environment:
      - SPRING_PROFILES_ACTIVE=docker

  user:
    build:
      context: .
      dockerfile: service/user/Dockerfile
    container_name: user-service
    ports:
      - "8083:8083"
    networks:
      - api-network
    depends_on:
      - common-service
    restart: unless-stopped
    environment:
      - SPRING_PROFILES_ACTIVE=docker

  postgres:
    image: postgres:16-alpine
    container_name: postgres-db
    ports:
      - "5432:5432"
    networks:
      - api-network
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=postgres
      - POSTGRES_DB=api_db
    volumes:
      - postgres-data:/var/lib/postgresql/data
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    container_name: redis-cache
    ports:
      - "6379:6379"
    networks:
      - api-network
    volumes:
      - redis-data:/data
    command: redis-server --appendonly yes
    restart: unless-stopped

networks:
  api-network:
    driver: bridge

volumes:
  postgres-data:
  redis-data:
```

---

## 프로필 기반 전략

### 권장: 환경별 프로필 분리

#### application.yaml (Gateway)

```yaml
spring:
  profiles:
    active: ${SPRING_PROFILES_ACTIVE:local}
  application:
    name: gateway-server

---
# ============================================
# 로컬 개발 환경 프로필
# ============================================
spring:
  config:
    activate:
      on-profile: local
  
  cloud:
    discovery:
      client:
        simple:
          instances:
            user-service:
              - uri: http://localhost:8083
                instance-id: user-local-1
            common-service:
              - uri: http://localhost:8082
                instance-id: common-local-1

---
# ============================================
# Docker 환경 프로필
# ============================================
spring:
  config:
    activate:
      on-profile: docker
  
  cloud:
    discovery:
      client:
        simple:
          instances:
            user-service:
              - uri: http://user-service:8083
                instance-id: user-docker-1
              # 스케일 아웃 시 추가
              # - uri: http://user-service-2:8083
              #   instance-id: user-docker-2
            common-service:
              - uri: http://common-service:8082
                instance-id: common-docker-1

---
# ============================================
# 프로덕션 환경 프로필
# ============================================
spring:
  config:
    activate:
      on-profile: prod
  
  cloud:
    discovery:
      client:
        simple:
          instances:
            user-service:
              - uri: http://user-service-1.prod.internal:8083
                instance-id: user-prod-1
              - uri: http://user-service-2.prod.internal:8083
                instance-id: user-prod-2
            common-service:
              - uri: http://common-service.prod.internal:8082
                instance-id: common-prod-1
```

### 환경변수 활용 (Docker)

```yaml
spring:
  cloud:
    discovery:
      client:
        simple:
          instances:
            user-service:
              - uri: http://${USER_SERVICE_HOST:user-service}:${USER_SERVICE_PORT:8083}
                instance-id: user-service-1
            common-service:
              - uri: http://${COMMON_SERVICE_HOST:common-service}:${COMMON_SERVICE_PORT:8082}
                instance-id: common-service-1
```

---

## 장단점 분석

### ✅ SimpleDiscoveryClient 장점

```
✅ Eureka 서버 불필요
   - 메모리 절약 (보통 512MB~1GB)
   - 포트 절약 (8761)
   - 관리 포인트 감소

✅ lb:// 프로토콜 유지
   - 기존 코드 변경 최소화
   - Gateway 라우팅 로직 유지

✅ 로드밸런싱 지원
   - Round Robin (기본)
   - Random
   - Weighted Response

✅ 여러 인스턴스 지원
   - 같은 서비스 여러 개 실행 가능
   - 수평 확장 가능

✅ 설정 간단
   - YAML 파일로 모든 설정 가능
   - 별도 서버 관리 불필요
```

### ⚠️ SimpleDiscoveryClient 단점

```
⚠️ 정적 설정
   - 서비스 목록을 수동으로 관리
   - 서비스 추가/제거 시 재배포 필요

⚠️ 제한적 헬스체크
   - Eureka만큼 강력하지 않음
   - 자동 장애 감지 제한적

⚠️ 동적 스케일링 어려움
   - Kubernetes 등과 연동 필요
   - Auto-scaling 지원 제한적

⚠️ 서비스 메타데이터 제한
   - Eureka 대비 기능 제한적
```

### 📊 적합한 사용 사례

#### ✅ SimpleDiscovery가 적합한 경우

```
✅ 마이크로서비스 5개 이하
✅ 서비스 위치가 자주 변경되지 않음
✅ 개발/테스트/스테이징 환경
✅ Docker Compose 환경
✅ 고정 IP/Port 사용
✅ 간단한 로드밸런싱만 필요
```

#### ❌ Eureka가 더 적합한 경우

```
❌ 마이크로서비스 10개 이상
❌ 동적 스케일링 필요 (Auto-scaling)
❌ 프로덕션 환경 (대규모)
❌ 클라우드 환경 (AWS, Azure, GCP)
❌ 서비스 인스턴스가 자주 변경됨
❌ 강력한 헬스체크 필요
```

---

## 실행 계획

### 1️⃣ 준비 단계

```bash
# 1. 현재 브랜치 백업
git checkout -b feature/remove-eureka

# 2. 프로젝트 구조 확인
tree -L 3
```

### 2️⃣ 파일 수정 순서

```
순서 1: Gateway pom.xml 확인
       └─ LoadBalancer 의존성 확인

순서 2: Gateway application.yaml 수정
       └─ SimpleDiscoveryClient 설정 추가
       └─ Eureka 설정 제거

순서 3: User Service application.yaml 수정
       └─ Eureka 설정 제거

순서 4: Common Service application.yaml 수정
       └─ Eureka 설정 제거

순서 5: docker-compose.yaml 수정
       └─ Eureka 서비스 제거
       └─ depends_on 수정

순서 6: 테스트
       └─ 로컬 환경 테스트
       └─ Docker 환경 테스트
```

### 3️⃣ 테스트 체크리스트

#### 로컬 환경 테스트

```bash
# Gateway 실행
cd server/discovery
./mvnw spring-boot:run

# User Service 실행
cd service/user
./mvnw spring-boot:run

# API 테스트
curl -X POST http://localhost:8080/api/user/kakao/login
curl -X POST http://localhost:8080/api/user/naver/login
curl -X POST http://localhost:8080/api/user/google/login
```

#### Docker 환경 테스트

```bash
# Docker Compose 빌드 및 실행
docker-compose build
docker-compose up -d

# 로그 확인
docker-compose logs gateway
docker-compose logs user-service

# API 테스트
curl -X POST http://localhost:8080/api/user/kakao/login

# 종료
docker-compose down
```

### 4️⃣ 검증 포인트

```
✅ Gateway가 정상 시작되는가?
✅ User Service가 정상 시작되는가?
✅ Gateway → User Service 통신이 되는가?
✅ 로드밸런싱이 동작하는가? (여러 인스턴스 시)
✅ CORS가 정상 동작하는가?
✅ 에러 핸들링이 정상인가?
```

---

## 고급 설정

### LoadBalancer 전략 커스터마이징

```java
@Configuration
public class LoadBalancerConfig {
    
    @Bean
    public ReactorLoadBalancer<ServiceInstance> customLoadBalancer(
            Environment environment,
            LoadBalancerClientFactory loadBalancerClientFactory) {
        
        String name = environment.getProperty(LoadBalancerClientFactory.PROPERTY_NAME);
        
        // Round Robin (기본값)
        return new RoundRobinLoadBalancer(
            loadBalancerClientFactory
                .getLazyProvider(name, ServiceInstanceListSupplier.class),
            name
        );
    }
}
```

### 헬스체크 추가 (선택사항)

```java
@Configuration
public class HealthCheckConfig {
    
    @Bean
    public HealthCheckServiceInstanceListSupplier healthCheckSupplier(
            ConfigurableApplicationContext context) {
        
        return new HealthCheckServiceInstanceListSupplier(
            ServiceInstanceListSupplier.builder()
                .withDiscoveryClient()
                .build(context),
            context.getBean(LoadBalancerProperties.class)
        );
    }
}
```

---

## 마이그레이션 가이드

### Before (Eureka 사용)

```yaml
# Gateway
spring:
  cloud:
    gateway:
      routes:
        - id: user-service
          uri: lb://user-service
          
eureka:
  client:
    service-url:
      defaultZone: http://eureka-server:8761/eureka/
```

### After (SimpleDiscovery 사용)

```yaml
# Gateway
spring:
  cloud:
    gateway:
      routes:
        - id: user-service
          uri: lb://user-service  # 동일
    
    discovery:
      client:
        simple:
          instances:
            user-service:
              - uri: http://user-service:8083
                instance-id: user-1

# eureka 설정 제거
```

---

## 트러블슈팅

### 문제 1: 503 Service Unavailable

```
원인: SimpleDiscoveryClient가 서비스를 찾지 못함

해결:
1. application.yaml의 인스턴스 설정 확인
2. 서비스 이름(user-service) 일치 확인
3. URI 형식 확인 (http:// 포함)
4. 포트 번호 확인
```

### 문제 2: 로드밸런싱이 동작하지 않음

```
원인: LoadBalancer 의존성 누락

해결:
<dependency>
    <groupId>org.springframework.cloud</groupId>
    <artifactId>spring-cloud-starter-loadbalancer</artifactId>
</dependency>
```

### 문제 3: Docker 환경에서 연결 실패

```
원인: 잘못된 호스트 이름

해결:
- localhost → service-name (Docker 내부 네트워크)
- 예: http://user-service:8083
```

---

## 결론

### 권장 전략: SimpleDiscoveryClient

현재 프로젝트는 다음 이유로 **SimpleDiscoveryClient**가 최적입니다:

```
✅ 서비스 개수 적음 (Gateway, User, Common)
✅ 개발/테스트 환경
✅ 고정 IP/Port 사용
✅ 간단한 로드밸런싱으로 충분
✅ Eureka 서버 리소스 절약 가능
```

### 향후 확장 시

프로젝트가 커지면 다음을 고려:

```
- Kubernetes 환경 → Kubernetes Service Discovery
- 대규모 마이크로서비스 → Consul 또는 Eureka
- 클라우드 환경 → AWS Cloud Map, Azure Service Fabric
```

---

## 참고 자료

- [Spring Cloud LoadBalancer 공식 문서](https://docs.spring.io/spring-cloud-commons/docs/current/reference/html/#spring-cloud-loadbalancer)
- [SimpleDiscoveryClient JavaDoc](https://docs.spring.io/spring-cloud-commons/docs/current/api/org/springframework/cloud/client/discovery/simple/SimpleDiscoveryClient.html)
- [Spring Cloud Gateway 공식 문서](https://docs.spring.io/spring-cloud-gateway/docs/current/reference/html/)

---

**작성일**: 2025-11-25  
**버전**: 1.0  
**작성자**: AI Assistant

