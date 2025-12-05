package kr.yeotaeho.api.google;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.Map;

/**
 * 구글 로그인 관련 컨트롤러
 * 
 * 주의: CORS는 Gateway에서 처리하므로 여기서는 설정하지 않습니다.
 * Gateway의 application.yaml에서 globalcors 설정을 확인하세요.
 */
@RestController
@RequestMapping("/google")
public class GoogleController {

    /**
     * 구글 로그인 요청 처리
     * 
     * @return 로그인 토큰 (테스트용)
     */
    @PostMapping("/login")
    public ResponseEntity<Map<String, String>> googleLogin() {
        System.out.println("구글 로그인 요청");
        // TODO: 실제 구글 OAuth 로그인 URL로 리다이렉트
        // 현재는 테스트용으로 토큰을 바로 반환

        Map<String, String> response = new HashMap<>();
        response.put("token", "google-auth-token-" + System.currentTimeMillis());
        response.put("message", "구글 로그인 성공");

        return ResponseEntity.ok(response);
    }

    /**
     * 구글 로그인 콜백 처리
     */
    @GetMapping("/callback")
    public String googleCallback(@RequestParam(required = false) String code) {
        // TODO: 구글 OAuth 콜백 처리
        return "구글 로그인 콜백 처리: code=" + code;
    }

    /**
     * 구글 로그인 테스트
     */
    @PostMapping("/test")
    public String googleTest(@RequestBody(required = false) String body) {
        return "구글 로그인 테스트 성공";
    }
}
