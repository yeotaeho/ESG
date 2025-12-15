# ---- Build Stage ----
FROM gradle:8.5-jdk21 AS build

# 루트 코드 복사
WORKDIR /app
COPY . .

# Gradle 래퍼가 있는 api 루트로 이동해서 빌드
WORKDIR /app/api
RUN chmod +x gradlew || true
RUN ./gradlew :discovery:clean :discovery:build -x test --no-daemon

# ---- Runtime Stage ----
FROM eclipse-temurin:21-jre

WORKDIR /app

COPY --from=build /app/api/discovery/build/libs/*.jar ./
RUN find . -name "*.jar" ! -name "*-plain.jar" -exec mv {} app.jar \;

EXPOSE 8080

ENTRYPOINT ["java", "-jar", "app.jar"]

