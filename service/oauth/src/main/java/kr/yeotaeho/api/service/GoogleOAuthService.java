package kr.yeotaeho.api.service;

import kr.yeotaeho.api.config.GoogleOAuthConfig;
import kr.yeotaeho.api.dto.GoogleTokenResponse;
import kr.yeotaeho.api.dto.GoogleUserInfo;
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
import org.springframework.web.util.UriComponentsBuilder;

/**
 * 구글 OAuth 서비스 (RestTemplate 사용)
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class GoogleOAuthService {

    private final GoogleOAuthConfig googleConfig;
    private final RestTemplate restTemplate;

    /**
     * 구글 로그인 URL 생성
     *
     * @return 구글 인증 URL
     */
    public String getAuthorizationUrl() {
        String authUrl = UriComponentsBuilder
                .fromHttpUrl(GoogleOAuthConfig.GOOGLE_AUTH_URL)
                .queryParam("client_id", googleConfig.getClientId())
                .queryParam("redirect_uri", googleConfig.getRedirectUri())
                .queryParam("response_type", "code")
                .queryParam("scope", "email profile")
                .queryParam("access_type", "offline")
                .queryParam("prompt", "consent")
                .build()
                .toUriString();

        log.info("생성된 구글 인증 URL: {}", authUrl);
        return authUrl;
    }

    /**
     * 인가 코드로 액세스 토큰 발급
     *
     * @param code 인가 코드
     * @return 구글 토큰 응답
     */
    public GoogleTokenResponse getAccessToken(String code) {
        log.info("구글 액세스 토큰 요청: code={}", code);

        // HTTP 헤더 설정
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_FORM_URLENCODED);

        // HTTP 바디 파라미터 설정
        MultiValueMap<String, String> params = new LinkedMultiValueMap<>();
        params.add("grant_type", "authorization_code");
        params.add("client_id", googleConfig.getClientId());
        params.add("client_secret", googleConfig.getClientSecret());
        params.add("redirect_uri", googleConfig.getRedirectUri());
        params.add("code", code);

        // HTTP 요청 엔티티 생성
        HttpEntity<MultiValueMap<String, String>> request = new HttpEntity<>(params, headers);

        try {
            // 구글 토큰 API 호출
            ResponseEntity<GoogleTokenResponse> response = restTemplate.postForEntity(
                    GoogleOAuthConfig.GOOGLE_TOKEN_URL,
                    request,
                    GoogleTokenResponse.class);

            log.info("구글 액세스 토큰 발급 성공");
            return response.getBody();

        } catch (Exception e) {
            log.error("구글 액세스 토큰 발급 실패: {}", e.getMessage());
            throw new RuntimeException("구글 토큰 발급 실패: " + e.getMessage());
        }
    }

    /**
     * 액세스 토큰으로 사용자 정보 조회
     *
     * @param accessToken 액세스 토큰
     * @return 구글 사용자 정보
     */
    public GoogleUserInfo getUserInfo(String accessToken) {
        log.info("구글 사용자 정보 조회");

        // HTTP 헤더 설정
        HttpHeaders headers = new HttpHeaders();
        headers.set("Authorization", "Bearer " + accessToken);

        // HTTP 요청 엔티티 생성
        HttpEntity<Void> request = new HttpEntity<>(headers);

        try {
            // 구글 사용자 정보 API 호출
            ResponseEntity<GoogleUserInfo> response = restTemplate.exchange(
                    GoogleOAuthConfig.GOOGLE_USER_INFO_URL,
                    HttpMethod.GET,
                    request,
                    GoogleUserInfo.class);

            GoogleUserInfo userInfo = response.getBody();
            log.info("구글 사용자 정보 조회 성공: id={}", userInfo.getId());
            return userInfo;

        } catch (Exception e) {
            log.error("구글 사용자 정보 조회 실패: {}", e.getMessage());
            throw new RuntimeException("구글 사용자 정보 조회 실패: " + e.getMessage());
        }
    }

    /**
     * 전체 OAuth 플로우 실행
     *
     * @param code 인가 코드
     * @return 구글 사용자 정보
     */
    public GoogleUserInfo processOAuth(String code) {
        // 1. 토큰 발급
        GoogleTokenResponse tokenResponse = getAccessToken(code);

        // 2. 사용자 정보 조회
        return getUserInfo(tokenResponse.getAccessToken());
    }
}

