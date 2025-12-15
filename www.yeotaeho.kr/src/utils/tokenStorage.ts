/**
 * JWT 토큰 저장 및 관리 유틸리티
 */

const TOKEN_KEY = 'accessToken';

/**
 * JWT 토큰 저장
 * @param token JWT 토큰 문자열
 */
export const saveToken = (token: string): void => {
    if (typeof window !== 'undefined') {
        localStorage.setItem(TOKEN_KEY, token);
        console.log('토큰이 저장되었습니다.');
    }
};

/**
 * 저장된 JWT 토큰 조회
 * @returns JWT 토큰 문자열 또는 null
 */
export const getToken = (): string | null => {
    if (typeof window !== 'undefined') {
        return localStorage.getItem(TOKEN_KEY);
    }
    return null;
};

/**
 * 저장된 JWT 토큰 삭제
 */
export const removeToken = (): void => {
    if (typeof window !== 'undefined') {
        localStorage.removeItem(TOKEN_KEY);
        console.log('토큰이 삭제되었습니다.');
    }
};

/**
 * 토큰 존재 여부 확인
 * @returns 토큰이 존재하면 true, 없으면 false
 */
export const hasToken = (): boolean => {
    return getToken() !== null;
};

/**
 * JWT 토큰 디코딩 (Base64 디코딩)
 * @param token JWT 토큰 문자열
 * @returns 디코딩된 페이로드 객체 또는 null
 */
export const decodeToken = (token: string): any | null => {
    try {
        const parts = token.split('.');
        if (parts.length !== 3) {
            console.error('Invalid JWT token format');
            return null;
        }

        // JWT 페이로드 (두 번째 부분) 디코딩
        const payload = parts[1];
        const decoded = atob(payload.replace(/-/g, '+').replace(/_/g, '/'));
        return JSON.parse(decoded);
    } catch (error) {
        console.error('Failed to decode token:', error);
        return null;
    }
};

/**
 * JWT 토큰에서 사용자 이메일 추출
 * @returns 사용자 이메일 또는 null
 */
export const getUserEmail = (): string | null => {
    const token = getToken();
    if (!token) return null;

    const decoded = decodeToken(token);
    return decoded?.email || null;
};

/**
 * JWT 토큰에서 사용자 이름 추출 (name이 없으면 email 사용)
 * @returns 사용자 이름 또는 null
 */
export const getUserName = (): string | null => {
    const token = getToken();
    if (!token) return null;

    const decoded = decodeToken(token);
    // name이 있으면 name을, 없으면 email을 사용
    return decoded?.name || decoded?.email || null;
};

/**
 * JWT 토큰에서 사용자 ID 추출
 * @returns 사용자 ID 또는 null
 */
export const getUserId = (): string | null => {
    const token = getToken();
    if (!token) return null;

    const decoded = decodeToken(token);
    return decoded?.userId?.toString() || decoded?.sub || null;
};



