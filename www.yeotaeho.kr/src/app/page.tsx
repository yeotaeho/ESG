"use client";

import React from 'react';
import Link from 'next/link';

// 메뉴 항목 정의
const menuItems: string[] = ["개인프로젝트", "팀프로젝트", "내 블로그", "더보기"];

// 메뉴 항목과 경로 매핑
const menuItemPaths: Record<string, string> = {
    "개인프로젝트": "/my_project",
    "팀프로젝트": "#",
    "내 블로그": "#",
    "더보기": "#"
};

/**
 * 상단 헤더 컴포넌트
 */
const Header: React.FC = () => (
    <header className="flex justify-between items-center px-10 py-5 h-20 border-b border-gray-100 shadow-sm sticky top-0 bg-white z-50">
        {/* 로고 (Osstem Vascular) */}
        <div className="text-xl font-bold tracking-tight">
            OSSTEM<span className="text-orange-600">VASCULAR</span>
        </div>

        {/* 메인 메뉴 */}
        <nav className="hidden lg:block">
            <ul className="flex space-x-8">
                {menuItems.map((item) => {
                    const path = menuItemPaths[item] || "#";
                    return (
                        <li key={item}>
                            {path === "#" ? (
                                <a href="#" className="text-sm text-gray-800 hover:text-orange-600 transition duration-150 font-medium">
                                    {item}
                                </a>
                            ) : (
                                <Link href={path} className="text-sm text-gray-800 hover:text-orange-600 transition duration-150 font-medium">
                                    {item}
                                </Link>
                            )}
                        </li>
                    );
                })}
            </ul>
        </nav>

        {/* 오른쪽 섹션 (문의 버튼 및 언어 스위치) */}
        <div className="flex items-center space-x-5">
            <a href="#" className="px-4 py-2 text-xs border border-gray-900 rounded-md hover:bg-gray-100 transition duration-150 font-medium shadow-sm">
                제품문의/견적
            </a>
            <div className="text-sm font-medium">
                <span className="text-black font-semibold">KR</span>
                <span className="text-gray-400"> / </span>
                <a href="#" className="text-gray-400 hover:text-black">EN</a>
            </div>
        </div>
    </header>
);

/**
 * 왼쪽 텍스트 콘텐츠 섹션 컴포넌트
 */
const TextSection: React.FC = () => {
    return (
        <div className="flex flex-col w-full lg:w-1/2 min-h-screen lg:min-h-0 px-8 py-10 lg:px-16 lg:py-24 justify-between bg-white">
            {/* 상단 여백 및 한글 메시지 */}
            <div className="flex-grow pt-10">
                {/* 한글 메시지 */}
                <div className="text-2xl sm:text-3xl lg:text-4xl font-light leading-relaxed tracking-tight">
                    국내외 의료진, <br />
                    <span className="font-semibold text-orange-600">글로벌 파트너</span>와 함께 <br />
                    <span className="font-semibold text-orange-600">지속적</span>으로 <span className="font-semibold text-orange-600">성장</span>
                </div>
            </div>

            {/* 하단 콘텐츠 (페이지네이션 및 영문 헤드라인) */}
            <div className="mt-16 lg:mt-auto pt-16">
                {/* 페이지네이션 */}
                <div className="flex items-center space-x-2 mb-3 text-lg font-bold text-gray-400">
                    <span className="cursor-pointer text-xl hover:text-black transition">&lt;</span>
                    <span className="text-lg text-black font-extrabold">3</span>
                    <span className="cursor-pointer text-xl hover:text-black transition">&gt;</span>
                </div>

                {/* 서브타이틀 */}
                <p className="text-xs font-semibold tracking-[0.3em] mb-4 text-gray-700">
                    TOTAL SOLUTION PROVIDER
                </p>

                {/* 영문 헤드라인 */}
                <h1 className="text-6xl sm:text-7xl md:text-8xl lg:text-9xl font-extrabold leading-none tracking-tighter">
                    GROWING <br />
                    TOGETHER
                </h1>
            </div>
        </div>
    );
};

/**
 * 오른쪽 이미지 섹션 컴포넌트 (Placeholder)
 * 원본 이미지의 블러 처리된 배경과 수직 텍스트 오버레이를 흉내냅니다.
 */
const ImageSection: React.FC = () => {
    return (
        // min-h-[calc(100vh-80px)]는 헤더 높이를 뺀 나머지 높이
        <div className="hidden lg:flex w-1/2 relative overflow-hidden items-center justify-center bg-gray-50 min-h-[calc(100vh-80px)]">
            
            {/* 배경 시뮬레이션: 흰색 테이블과 블러 처리된 인물 (텍스트/박스로 대체) */}
            <div 
                className="absolute inset-0 bg-white p-20 flex flex-col space-y-4" 
                aria-label="블러 처리된 배경 이미지 시뮬레이션"
            >
                {/* 블러 처리된 인물 시뮬레이션: 상단에 희미한 그림자처럼 배치 */}
                <div className="flex justify-around opacity-30 pt-10">
                    <div className="w-1/4 h-8 bg-gray-300 rounded-full filter blur-md"></div>
                    <div className="w-1/4 h-8 bg-gray-300 rounded-full filter blur-md"></div>
                    <div className="w-1/4 h-8 bg-gray-300 rounded-full filter blur-md"></div>
                </div>
                {/* 테이블 시뮬레이션 (손과 컵) */}
                <div className="flex justify-center items-center h-full">
                    {/* 원본 이미지의 블러 처리된 손과 테이블의 느낌을 나타내는 Placeholder 텍스트 */}
                    <div className="text-5xl text-gray-400 opacity-20 filter blur-sm">👋☕️🤝</div> 
                </div>
                {/* 테이블 위의 부드러운 빛 효과 */}
                <div className="absolute inset-0 bg-gradient-to-t from-white/30 via-white/70 to-white/10"></div>
            </div>

            {/* 수직 오버레이 텍스트 (희미하게 표시) - 크기 줄이고 위치 조정 */}
            <div 
                className="absolute text-gray-900/10 font-extrabold tracking-widest pointer-events-none z-10"
                style={{ 
                    fontSize: '8rem', // 폰트 크기
                    writingMode: 'vertical-rl', 
                    textOrientation: 'mixed',
                    transform: 'translateY(10%)', // 중앙에서 약간 아래로 이동
                }}
            >
                OSSTEM VASCULAR
            </div>

            {/* 밝은 소프트 포커스 영역을 흉내내는 오버레이 (이미지 섹션의 전경) */}
            <div className="absolute inset-0 bg-black/5 z-20"></div>
            
                            </div>
    );
};

/**
 * 메인 애플리케이션 컴포넌트
 */
const App: React.FC = () => {
    return (
        <div className="min-h-screen antialiased">
            <Header />
            <main className="flex flex-col lg:flex-row min-h-[calc(100vh-80px)]">
                <TextSection />
                <ImageSection />
            </main>
        </div>
    );
};

export default App;