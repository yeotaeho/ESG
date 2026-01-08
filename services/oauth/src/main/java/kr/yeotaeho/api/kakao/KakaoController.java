package kr.yeotaeho.api.kakao;

import kr.yeotaeho.api.dto.KakaoUserInfo;
import kr.yeotaeho.api.entity.User;
import kr.yeotaeho.api.service.KakaoOAuthService;
import kr.yeotaeho.api.service.RefreshTokenService;
import kr.yeotaeho.api.service.UserService;
import kr.yeotaeho.api.util.CookieUtil;
import kr.yeotaeho.api.util.JwtTokenUtil;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.Map;

/**
 * 카카오 로그인 관련 컨트롤러
 * 
 * 주의: CORS는 Gateway에서 처리하므로 여기서는 설정하지 않습니다.
 * Gateway의 application.yaml에서 globalcors 설정을 확인하세요.
 */
@Slf4j
@RestController
@RequestMapping("/kakao")
@RequiredArgsConstructor
public class KakaoController {

    private final KakaoOAuthService kakaoOAuthService;
    private final UserService userService;
    private final JwtTokenUtil jwtTokenUtil;
    private final RefreshTokenService refreshTokenService;
    private final CookieUtil cookieUtil;

    /**
     * 카카오 로그인 URL 요청
     * 
     * @return 카카오 인증 URL
     */
    @GetMapping("/login")
    public ResponseEntity<Map<String, String>> getKakaoLoginUrl() {
        log.info("카카오 로그인 URL 요청");

        String authUrl = kakaoOAuthService.getAuthorizationUrl();

        Map<String, String> response = new HashMap<>();
        response.put("authUrl", authUrl);
        response.put("message", "카카오 로그인 페이지로 이동하세요");

        return ResponseEntity.ok(response);
    }

    /**
     * 카카오 로그인 콜백 처리 (프론트에서 code를 POST로 전송)
     * 
     * @param body code를 포함한 요청 바디
     * @return 사용자 정보 및 JWT 토큰
     */
    @PostMapping("/callback")
    public ResponseEntity<Map<String, Object>> kakaoCallback(@RequestBody Map<String, String> body) {
        String code = body.get("code");
        log.info("카카오 로그인 콜백 처리 시작: code={}", code);

        try {
            // OAuth 플로우 실행 (토큰 발급 + 사용자 정보 조회)
            KakaoUserInfo userInfo = kakaoOAuthService.processOAuth(code);

            // 사용자 정보 추출
            String providerId = String.valueOf(userInfo.getId());
            String email = userInfo.getKakaoAccount() != null ? userInfo.getKakaoAccount().getEmail() : null;
            String nickname = null;
            String profileImage = null;

            if (userInfo.getKakaoAccount() != null && userInfo.getKakaoAccount().getProfile() != null) {
                nickname = userInfo.getKakaoAccount().getProfile().getNickname();
                profileImage = userInfo.getKakaoAccount().getProfile().getProfileImageUrl();
            }

            // 1. DB에 사용자 정보 저장/업데이트
            User user = userService.findOrCreateUser("kakao", providerId, email, null, nickname, profileImage);

            // 2. JWT 토큰 생성
            String accessToken = jwtTokenUtil.generateToken(user.getId(), "kakao", email);
            String refreshToken = jwtTokenUtil.generateRefreshToken(user.getId(), "kakao", email);

            // 3. 리프레시 토큰을 Redis에 저장
            refreshTokenService.saveRefreshToken(user.getId(), refreshToken);

            // 4. 응답 생성
            Map<String, Object> response = new HashMap<>();
            response.put("success", true);
            response.put("message", "카카오 로그인 성공");
            response.put("userId", user.getId());
            response.put("kakaoId", providerId);
            response.put("email", email);
            response.put("nickname", nickname);
            response.put("profileImage", profileImage);
            response.put("accessToken", accessToken);
            response.put("tokenType", "Bearer");

            log.info("카카오 로그인 성공: userId={}, kakaoId={}", user.getId(), providerId);
            return ResponseEntity.ok()
                    .header(HttpHeaders.SET_COOKIE, cookieUtil.createRefreshTokenCookie(refreshToken).toString())
                    .body(response);

        } catch (Exception e) {
            log.error("카카오 로그인 실패: {}", e.getMessage(), e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body(Map.of(
                            "success", false,
                            "message", "카카오 로그인 실패: " + e.getMessage(),
                            "error", e.getClass().getSimpleName()
                    ));
        }
    }
}
