package kr.yeotaeho.api.naver;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.Map;

/**
 * 네이버 로그인 관련 컨트롤러
 * 
 * 주의: CORS는 Gateway에서 처리하므로 여기서는 설정하지 않습니다.
 * Gateway의 application.yaml에서 globalcors 설정을 확인하세요.
 */
@RestController
@RequestMapping("/naver")
public class NaverController {

    /**
     * 네이버 로그인 요청 처리
     * 
     * @return 로그인 토큰 (테스트용)
     */
    @PostMapping("/login")
    public ResponseEntity<Map<String, String>> naverLogin() {

        System.out.println("네이버 로그인 요청");
        // TODO: 실제 네이버 OAuth 로그인 URL로 리다이렉트
        // 현재는 테스트용으로 토큰을 바로 반환

        Map<String, String> response = new HashMap<>();
        response.put("token", "naver-auth-token-" + System.currentTimeMillis());
        response.put("message", "네이버 로그인 성공");

        return ResponseEntity.ok(response);
    }

    /**
     * 네이버 로그인 콜백 처리
     */
    @GetMapping("/callback")
    public String naverCallback(@RequestParam(required = false) String code) {
        // TODO: 네이버 OAuth 콜백 처리
        return "네이버 로그인 콜백 처리: code=" + code;
    }

    /**
     * 네이버 로그인 테스트
     */
    @PostMapping("/test")
    public String naverTest(@RequestBody(required = false) String body) {
        return "네이버 로그인 테스트 성공";
    }
}
