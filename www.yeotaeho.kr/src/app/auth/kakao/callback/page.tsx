"use client";

import React, { useEffect, useState } from 'react';
import { useSearchParams, useRouter } from 'next/navigation';
import { saveToken } from '@/utils/tokenStorage';
import { useAuth } from '@/hooks/useStore';

export default function KakaoCallbackPage() {
    const searchParams = useSearchParams();
    const router = useRouter();
    const { login } = useAuth();
    const [status, setStatus] = useState<'loading' | 'success' | 'error'>('loading');
    const [message, setMessage] = useState<string>('');

    useEffect(() => {
        const handleCallback = async () => {
            try {
                // URL에서 code 파라미터 추출
                const code = searchParams.get('code');
                const error = searchParams.get('error');

                // 에러가 있는 경우
                if (error) {
                    setStatus('error');
                    setMessage(`인증 오류: ${error}`);
                    console.error('OAuth error:', error);
                    return;
                }

                // code가 없는 경우
                if (!code) {
                    setStatus('error');
                    setMessage('인증 코드를 받지 못했습니다.');
                    console.error('No authorization code received');
                    return;
                }

                // 백엔드로 code 전송하여 토큰 교환
                const response = await fetch('http://localhost:8080/api/oauth/kakao/callback', {
                    method: 'POST',
                    credentials: 'include',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ code }),
                });

                if (response.ok) {
                    const data = await response.json();
                    setStatus('success');
                    setMessage('로그인 성공!');
                    console.log('Login successful:', data);

                    // JWT 토큰 저장 (localStorage + Zustand store)
                    if (data.accessToken) {
                        saveToken(data.accessToken);
                        login(data.accessToken); // Zustand store에 토큰 저장
                    }

                    // 로그인 성공 후 메인 페이지나 대시보드로 리디렉션
                    setTimeout(() => {
                        router.push('/my_project');
                    }, 2000);
                } else {
                    const errorData = await response.json().catch(() => ({ message: '알 수 없는 오류' }));
                    setStatus('error');
                    setMessage(`로그인 실패: ${errorData.message || response.statusText}`);
                    console.error('Login failed:', errorData);
                }
            } catch (error) {
                setStatus('error');
                const errorMessage = error instanceof Error ? error.message : '알 수 없는 오류가 발생했습니다.';
                setMessage(`연결 오류: ${errorMessage}`);
                console.error('Error during callback:', error);
            }
        };

        handleCallback();
    }, [searchParams, router]);

    return (
        <div className="min-h-screen bg-white flex items-center justify-center px-4">
            <div className="w-full max-w-md text-center">
                {status === 'loading' && (
                    <>
                        <div className="mb-4">
                            <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-yellow-400"></div>
                        </div>
                        <h2 className="text-2xl font-bold text-black mb-2">로그인 처리 중...</h2>
                        <p className="text-gray-600">잠시만 기다려주세요.</p>
                    </>
                )}

                {status === 'success' && (
                    <>
                        <div className="mb-4">
                            <svg className="mx-auto h-12 w-12 text-green-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                            </svg>
                        </div>
                        <h2 className="text-2xl font-bold text-black mb-2">로그인 성공!</h2>
                        <p className="text-gray-600 mb-4">{message}</p>
                        <p className="text-sm text-gray-500">잠시 후 메인 페이지로 이동합니다...</p>
                    </>
                )}

                {status === 'error' && (
                    <>
                        <div className="mb-4">
                            <svg className="mx-auto h-12 w-12 text-red-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                            </svg>
                        </div>
                        <h2 className="text-2xl font-bold text-black mb-2">로그인 실패</h2>
                        <p className="text-gray-600 mb-4">{message}</p>
                        <button
                            onClick={() => router.push('/login')}
                            className="px-6 py-2 bg-black text-white rounded-md hover:bg-gray-800 transition"
                        >
                            로그인 페이지로 돌아가기
                        </button>
                    </>
                )}
            </div>
        </div>
    );
}

