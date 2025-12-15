package kr.yeotaeho.api.service;

import kr.yeotaeho.api.config.KakaoOAuthConfig;
import kr.yeotaeho.api.dto.KakaoTokenResponse;
import kr.yeotaeho.api.dto.KakaoUserInfo;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpMethod;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestTemplate;

/**
 * 카카오 OAuth 서비스 (RestTemplate 사용)
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class KakaoOAuthService {

    private final KakaoOAuthConfig kakaoConfig;
    private final RestTemplate restTemplate;

    /**
     * 카카오 로그인 URL 생성
     *
     * @return 카카오 인증 URL
     */
    public String getAuthorizationUrl() {
        return String.format(
                "%s?client_id=%s&redirect_uri=%s&response_type=code",
                KakaoOAuthConfig.KAKAO_AUTH_URL,
                kakaoConfig.getClientId(),
                kakaoConfig.getRedirectUri());
    }

    /**
     * 인가 코드로 액세스 토큰 발급
     *
     * @param code 인가 코드
     * @return 카카오 토큰 응답
     */
    public KakaoTokenResponse getAccessToken(String code) {
        log.info("카카오 액세스 토큰 요청: code={}", code);

        // HTTP 헤더 설정
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_FORM_URLENCODED);

        // HTTP 바디 파라미터 설정
        MultiValueMap<String, String> params = new LinkedMultiValueMap<>();
        params.add("grant_type", "authorization_code");
        params.add("client_id", kakaoConfig.getClientId());
        params.add("redirect_uri", kakaoConfig.getRedirectUri());
        params.add("code", code);

        // client_secret이 있으면 추가
        if (kakaoConfig.getClientSecret() != null && !kakaoConfig.getClientSecret().isEmpty()) {
            params.add("client_secret", kakaoConfig.getClientSecret());
        }

        // HTTP 요청 엔티티 생성
        HttpEntity<MultiValueMap<String, String>> request = new HttpEntity<>(params, headers);

        try {
            // 카카오 토큰 API 호출
            ResponseEntity<KakaoTokenResponse> response = restTemplate.postForEntity(
                    KakaoOAuthConfig.KAKAO_TOKEN_URL,
                    request,
                    KakaoTokenResponse.class);

            log.info("카카오 액세스 토큰 발급 성공");
            return response.getBody();

        } catch (Exception e) {
            log.error("카카오 액세스 토큰 발급 실패", e);
            throw new RuntimeException("카카오 토큰 발급 실패: " + e.getMessage());
        }
    }

    /**
     * 액세스 토큰으로 사용자 정보 조회
     *
     * @param accessToken 액세스 토큰
     * @return 카카오 사용자 정보
     */
    public KakaoUserInfo getUserInfo(String accessToken) {
        log.info("카카오 사용자 정보 조회");

        // HTTP 헤더 설정
        HttpHeaders headers = new HttpHeaders();
        headers.set("Authorization", "Bearer " + accessToken);
        headers.setContentType(MediaType.APPLICATION_FORM_URLENCODED);

        // HTTP 요청 엔티티 생성
        HttpEntity<Void> request = new HttpEntity<>(headers);

        try {
            // 카카오 사용자 정보 API 호출
            ResponseEntity<KakaoUserInfo> response = restTemplate.exchange(
                    KakaoOAuthConfig.KAKAO_USER_INFO_URL,
                    HttpMethod.GET,
                    request,
                    KakaoUserInfo.class);

            KakaoUserInfo userInfo = response.getBody();
            log.info("카카오 사용자 정보 조회 성공: id={}", userInfo.getId());
            return userInfo;

        } catch (Exception e) {
            log.error("카카오 사용자 정보 조회 실패", e);
            throw new RuntimeException("카카오 사용자 정보 조회 실패: " + e.getMessage());
        }
    }

    /**
     * 전체 OAuth 플로우 실행
     *
     * @param code 인가 코드
     * @return 카카오 사용자 정보
     */
    public KakaoUserInfo processOAuth(String code) {
        // 1. 토큰 발급
        KakaoTokenResponse tokenResponse = getAccessToken(code);

        // 2. 사용자 정보 조회
        return getUserInfo(tokenResponse.getAccessToken());
    }
}
