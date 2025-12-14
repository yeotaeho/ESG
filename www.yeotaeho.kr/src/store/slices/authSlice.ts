import { SliceCreator, AuthSlice } from '../types';

/**
 * Auth Slice
 * 인증 관련 상태 및 액션을 관리하는 Slice
 */
export const createAuthSlice: SliceCreator<AuthSlice> = (set, get) => ({
  token: null,
  isAuthenticated: false,

  /**
   * 로그인
   * 토큰을 저장하고 인증 상태를 true로 설정
   */
  login: (token: string) => {
    set(
      {
        token,
        isAuthenticated: true,
      },
      false, // replace 옵션
      'auth/login' // DevTools 액션 이름
    );
  },

  /**
   * 로그아웃
   * 토큰을 제거하고 인증 상태를 false로 설정
   * 유저 프로필도 함께 초기화
   */
  logout: () => {
    set(
      {
        token: null,
        isAuthenticated: false,
      },
      false,
      'auth/logout'
    );
    // 다른 Slice 상태도 리셋
    get().clearProfile();
  },

  /**
   * 토큰 설정
   * 토큰이 있으면 인증 상태를 true로, 없으면 false로 설정
   */
  setToken: (token: string | null) => {
    set(
      {
        token,
        isAuthenticated: !!token,
      },
      false,
      'auth/setToken'
    );
  },
});

