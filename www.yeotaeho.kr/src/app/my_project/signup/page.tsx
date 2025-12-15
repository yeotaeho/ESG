"use client";

import React from 'react';

export default function SignupPage() {
    // 소셜 로그인 핸들러
    const handleSocialLogin = async (provider: 'kakao' | 'naver' | 'google') => {
        try {
            const response = await fetch(`http://localhost:8080/api/oauth/${provider}/login`, {
                method: 'GET',
                credentials: 'include',
                headers: {
                    'Content-Type': 'application/json',
                },
            });

            if (response.ok) {
                const data = await response.json();
                // authUrl 또는 redirectUrl이 있으면 이동
                const redirectUrl = data.authUrl || data.redirectUrl;
                if (redirectUrl) {
                    window.location.href = redirectUrl;
                } else {
                    console.log('Login response:', data);
                    alert('로그인 URL을 받지 못했습니다.');
                }
            } else {
                // 에러 응답의 상세 정보 확인
                let errorMessage = `Login failed: ${response.status} ${response.statusText}`;
                try {
                    const errorData = await response.json();
                    errorMessage = errorData.message || errorData.error || errorMessage;
                } catch (e) {
                    // JSON 파싱 실패 시 텍스트로 읽기 시도
                    const text = await response.text();
                    if (text) {
                        errorMessage = text;
                    }
                }
                console.error('Login failed:', errorMessage);
                alert(`로그인 실패: ${errorMessage}`);
            }
        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : '알 수 없는 오류가 발생했습니다.';
            console.error('Error during social login:', error);
            alert(`연결 오류: ${errorMessage}\n서버가 실행 중인지 확인해주세요.`);
        }
    };

    return (
        <div className="min-h-screen bg-white flex items-center justify-center px-4 py-12">
            <div className="w-full max-w-md">
                {/* WELCOME Header */}
                <h1 className="text-4xl font-bold text-black text-center mb-6">WELCOME!</h1>

                {/* Introductory Text */}
                <div className="text-center mb-8 text-gray-700 leading-relaxed">
                    <p className="mb-1">데상트코리아 통합 회원이 되시면 온라인 스토어 및</p>
                    <p>오프라인 매장에서 다양한 혜택이 제공됩니다.</p>
                </div>

                {/* Primary Sign Up Button */}
                <button className="w-full bg-black text-white py-4 rounded-lg font-medium hover:bg-gray-800 transition mb-8 text-lg">
                    회원가입
                </button>

                {/* Divider */}
                <div className="border-t border-gray-300 mb-6"></div>

                {/* Social Login Section */}
                <div className="text-center mb-6">
                    <p className="text-black text-sm">SNS로 시작하기</p>
                </div>

                {/* Social Login Buttons */}
                <div className="flex justify-center items-center space-x-4">
                    {/* Kakao */}
                    <button 
                        onClick={() => handleSocialLogin('kakao')}
                        className="w-16 h-16 bg-yellow-400 rounded-full flex items-center justify-center hover:bg-yellow-500 transition shadow-sm cursor-pointer"
                    >
                        <svg className="w-8 h-8 text-black" viewBox="0 0 24 24" fill="currentColor">
                            <path d="M12 3c5.799 0 10.5 3.664 10.5 8.185 0 4.52-4.701 8.184-10.5 8.184a13.5 13.5 0 0 1-1.727-.11l-4.408 2.883c-.501.265-.678.236-.472-.413l.892-3.678c-2.88-1.46-4.785-3.99-4.785-6.866C1.5 6.665 6.201 3 12 3z"/>
                        </svg>
                    </button>

                    {/* Naver */}
                    <button 
                        onClick={() => handleSocialLogin('naver')}
                        className="w-16 h-16 bg-green-500 rounded-full flex items-center justify-center hover:bg-green-600 transition shadow-sm cursor-pointer"
                    >
                        <span className="text-white font-bold text-xl">N</span>
                    </button>

                    {/* Google */}
                    <button 
                        onClick={() => handleSocialLogin('google')}
                        className="w-16 h-16 bg-white border border-gray-300 rounded-full flex items-center justify-center hover:bg-gray-50 transition shadow-sm cursor-pointer"
                    >
                        <svg className="w-8 h-8" viewBox="0 0 24 24">
                            <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                            <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                            <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
                            <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
                        </svg>
                    </button>
                </div>
            </div>
        </div>
    );
}

