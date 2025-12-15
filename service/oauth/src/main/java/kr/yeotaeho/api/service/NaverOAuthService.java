package kr.yeotaeho.api.service;

import kr.yeotaeho.api.config.NaverOAuthConfig;
import kr.yeotaeho.api.dto.NaverTokenResponse;
import kr.yeotaeho.api.dto.NaverUserInfo;
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

import java.io.UnsupportedEncodingException;
import java.net.URLEncoder;
import java.util.UUID;

/**
 * 네이버 OAuth 서비스 (RestTemplate 사용)
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class NaverOAuthService {

    private final NaverOAuthConfig naverConfig;
    private final RestTemplate restTemplate;

    /**
     * 네이버 로그인 URL 생성
     * URL 파라미터를 안전하게 인코딩하여 생성
     *
     * @return 네이버 인증 URL
     */
    public String getAuthorizationUrl() {
        String state = UUID.randomUUID().toString();
        
        // TODO: State 값을 세션이나 Redis에 저장하여 콜백 시 검증 필요
        // 현재는 간단한 구현을 위해 state 검증을 생략하지만
        // 프로덕션 환경에서는 반드시 state 검증을 구현해야 함
        log.info("네이버 로그인 URL 생성: state={}", state);
        
        try {
            // UriComponentsBuilder를 사용하여 안전하게 URL 생성
            String authUrl = UriComponentsBuilder
                    .fromHttpUrl(NaverOAuthConfig.NAVER_AUTH_URL)
                    .queryParam("client_id", naverConfig.getClientId())
                    .queryParam("redirect_uri", naverConfig.getRedirectUri())
                    .queryParam("response_type", "code")
                    .queryParam("state", state)
                    .build()
                    .toUriString();
            
            log.info("생성된 네이버 인증 URL: {}", authUrl);
            return authUrl;
            
        } catch (Exception e) {
            log.error("네이버 인증 URL 생성 실패", e);
            throw new RuntimeException("네이버 인증 URL 생성 실패: " + e.getMessage());
        }
    }

    /**
     * 인가 코드로 액세스 토큰 발급
     *
     * @param code 인가 코드
     * @param state 상태 토큰
     * @return 네이버 토큰 응답
     */
    public NaverTokenResponse getAccessToken(String code, String state) {
        log.info("네이버 액세스 토큰 요청: code={}, state={}", code, state);
        
        // TODO: 여기서 state 값을 검증해야 함
        // 저장된 state와 콜백으로 받은 state가 일치하는지 확인
        
        log.debug("사용중인 Client ID: {}", naverConfig.getClientId());
        log.debug("사용중인 Redirect URI: {}", naverConfig.getRedirectUri());

        // HTTP 헤더 설정
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_FORM_URLENCODED);

        // HTTP 바디 파라미터 설정
        MultiValueMap<String, String> params = new LinkedMultiValueMap<>();
        params.add("grant_type", "authorization_code");
        params.add("client_id", naverConfig.getClientId());
        params.add("client_secret", naverConfig.getClientSecret());
        params.add("code", code);
        params.add("state", state);

        // HTTP 요청 엔티티 생성
        HttpEntity<MultiValueMap<String, String>> request = new HttpEntity<>(params, headers);

        try {
            // 네이버 토큰 API 호출
            ResponseEntity<NaverTokenResponse> response = restTemplate.postForEntity(
                    NaverOAuthConfig.NAVER_TOKEN_URL,
                    request,
                    NaverTokenResponse.class);

            log.info("네이버 액세스 토큰 발급 성공");
            return response.getBody();

        } catch (Exception e) {
            log.error("네이버 액세스 토큰 발급 실패: {}", e.getMessage());
            log.error("상세 에러: ", e);
            throw new RuntimeException("네이버 토큰 발급 실패: " + e.getMessage());
        }
    }

    /**
     * 액세스 토큰으로 사용자 정보 조회
     *
     * @param accessToken 액세스 토큰
     * @return 네이버 사용자 정보
     */
    public NaverUserInfo getUserInfo(String accessToken) {
        log.info("네이버 사용자 정보 조회");

        // HTTP 헤더 설정
        HttpHeaders headers = new HttpHeaders();
        headers.set("Authorization", "Bearer " + accessToken);

        // HTTP 요청 엔티티 생성
        HttpEntity<Void> request = new HttpEntity<>(headers);

        try {
            // 네이버 사용자 정보 API 호출
            ResponseEntity<NaverUserInfo> response = restTemplate.exchange(
                    NaverOAuthConfig.NAVER_USER_INFO_URL,
                    HttpMethod.GET,
                    request,
                    NaverUserInfo.class);

            NaverUserInfo userInfo = response.getBody();
            log.info("네이버 사용자 정보 조회 성공: id={}", userInfo.getResponse().getId());
            return userInfo;

        } catch (Exception e) {
            log.error("네이버 사용자 정보 조회 실패", e);
            throw new RuntimeException("네이버 사용자 정보 조회 실패: " + e.getMessage());
        }
    }

    /**
     * 전체 OAuth 플로우 실행
     *
     * @param code 인가 코드
     * @param state 상태 토큰
     * @return 네이버 사용자 정보
     */
    public NaverUserInfo processOAuth(String code, String state) {
        // 1. 토큰 발급
        NaverTokenResponse tokenResponse = getAccessToken(code, state);

        // 2. 사용자 정보 조회
        return getUserInfo(tokenResponse.getAccessToken());
    }
}

