'use client';

import { useState } from 'react';

interface Message {
  type: 'ai' | 'user';
  content: string;
}

export default function EsgPage() {
  const [activeTab, setActiveTab] = useState<'consult' | 'report' | 'chart' | 'company'>('consult');
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [messages, setMessages] = useState<Message[]>([
    { type: 'ai', content: '2단계: 문단 분석 및 부족정보 파악이 완료되었습니다. IFRS S2 기준에 맞춰 몇 가지 질문을 시작하겠습니다. (흐름 4.1)' },
    { type: 'ai', content: '질문 (S2-5 거버넌스): 경영진의 기후 리스크 감독 주체가 누구인가요? (예: 이사회, ESG 위원회) (흐름 4.2)' },
    { type: 'user', content: '이사회 산하의 \'지속가능경영위원회\'가 분기별로 감독합니다.' },
    { type: 'ai', content: '좋습니다. S2-5 문단이 우측에 생성되었습니다. 확인해주세요.\n다음 질문 (S2-15 관련): Scope 1·2 데이터의 기준연도는 어떻게 되나요? (흐름 4.2)' },
  ]);

  const getAiResponse = async (userMessage: string): Promise<string> => {
    try {
      const response = await fetch('http://localhost:8080/api/chatbot/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: userMessage }),
      });

      if (response.ok) {
        const data = await response.json();
        return data.response || data.message || '응답을 받았습니다.';
      } else {
        throw new Error('API 요청 실패');
      }
    } catch (error) {
      console.error('API 호출 에러:', error);
      // 폴백: 로컬 응답
      const lowerMsg = userMessage.toLowerCase();

      if (lowerMsg.includes('기준연도') || lowerMsg.includes('2023') || lowerMsg.includes('2022')) {
        return `감사합니다. 기준연도 정보가 확인되었습니다. S2-15 문단에 반영하겠습니다.\n\n다음 질문: Scope 3 배출량 산정 범위는 어떻게 되나요? (카테고리 1~15 중 해당 항목)`;
      }
      if (lowerMsg.includes('scope') || lowerMsg.includes('배출')) {
        return `Scope 관련 정보 감사합니다. 해당 내용을 Metrics & Targets 섹션에 반영하겠습니다.\n\n추가로 확인이 필요한 사항이 있으시면 말씀해주세요.`;
      }
      if (lowerMsg.includes('위원회') || lowerMsg.includes('이사회') || lowerMsg.includes('감독')) {
        return `거버넌스 관련 정보가 업데이트되었습니다. 우측 프리뷰에서 확인해주세요.\n\n다음으로 리스크 관리 프로세스에 대해 설명해주시겠어요?`;
      }

      return `입력하신 내용을 분석했습니다: "${userMessage}"\n\n해당 정보를 보고서에 반영하겠습니다. 추가 질문이 있으시면 말씀해주세요.`;
    }
  };

  const handleSend = async () => {
    if (inputValue.trim() && !isLoading) {
      const userMsg = inputValue;
      setMessages(prev => [...prev, { type: 'user', content: userMsg }]);
      setInputValue('');
      setIsLoading(true);

      try {
        const aiResponse = await getAiResponse(userMsg);
        setMessages(prev => [...prev, { type: 'ai', content: aiResponse }]);
      } catch (error) {
        setMessages(prev => [...prev, { type: 'ai', content: '죄송합니다. 오류가 발생했습니다. 다시 시도해주세요.' }]);
      } finally {
        setIsLoading(false);
      }
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      handleSend();
    }
  };

  return (
    <div className="h-screen flex flex-col bg-gray-100 font-sans">
      {/* 1. 헤더 */}
      <header className="bg-white border-b border-gray-200 shadow-sm w-full z-10">
        <div className="max-w-screen-2xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            {/* 로고 및 타이틀 */}
            <div className="flex items-center space-x-2">
              <div className="bg-blue-600 p-2 rounded-lg">
                <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path d="M12 10v4M12 18v-2M15 13h-3M10 21h4c4.477 0 8-3.523 8-8V7c0-4.477-3.523-8-8-8H7C2.523-1 2 2.523 2 7v4c0 4.477 3.523 8 8 8z" />
                </svg>
              </div>
              <h1 className="text-xl font-bold text-gray-800">AI ESG Consultant</h1>
              <span className="bg-gray-100 text-gray-600 text-xs font-semibold px-2 py-0.5 rounded-full">Prototype</span>
            </div>
            {/* 버전 및 내보내기 버튼 */}
            <div className="flex items-center space-x-4">
              <span className="text-sm font-medium text-gray-500">보고서 버전: v1.0 (초안)</span>
              {activeTab === 'report' && (
                <>
                  <button className="text-sm bg-red-600 text-white px-4 py-2 rounded-lg font-semibold hover:bg-red-700 transition-colors flex items-center space-x-1.5">
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path d="M4 6V4a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2v2M4 12h16M4 18v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2M9 15v-4H7l5-5 5 5h-2v4H9Z" />
                    </svg>
                    <span>PDF 내보내기</span>
                  </button>
                  <button className="text-sm bg-blue-600 text-white px-4 py-2 rounded-lg font-semibold hover:bg-blue-700 transition-colors flex items-center space-x-1.5">
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path d="M4 6V4a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2v2M4 12h16M4 18v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2M9 15v-4H7l5-5 5 5h-2v4H9Z" />
                    </svg>
                    <span>Word 내보내기</span>
                  </button>
                </>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* 2. 메인 컨텐츠 영역 */}
      <main className="flex-1 flex flex-col overflow-hidden">
        {/* 탭 네비게이션 */}
        <div className="bg-white border-b border-gray-200">
          <nav className="-mb-px flex max-w-screen-2xl mx-auto px-4 sm:px-6 lg:px-8">
            <button
              onClick={() => setActiveTab('consult')}
              className={`whitespace-nowrap py-3 px-4 border-b-2 font-medium text-sm ${activeTab === 'consult'
                ? 'border-blue-500 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
            >
              전문가 상담 및 문단 생성
            </button>
            <button
              onClick={() => setActiveTab('report')}
              className={`whitespace-nowrap py-3 px-4 border-b-2 font-medium text-sm ${activeTab === 'report'
                ? 'border-blue-500 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
            >
              최종 보고서 조합
            </button>
            <button
              onClick={() => setActiveTab('chart')}
              className={`whitespace-nowrap py-3 px-4 border-b-2 font-medium text-sm ${activeTab === 'chart'
                ? 'border-blue-500 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
            >
              도표
            </button>
            <button
              onClick={() => setActiveTab('company')}
              className={`whitespace-nowrap py-3 px-4 border-b-2 font-medium text-sm ${activeTab === 'company'
                ? 'border-blue-500 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
            >
              회사정보
            </button>
          </nav>
        </div>

        {/* 탭 컨텐츠 */}
        <div className="flex-1 overflow-hidden">
          {/* 전문가 상담 UI + 문단 생성 프리뷰 */}
          {activeTab === 'consult' && (
            <div className="flex h-full">
              {/* 왼쪽: 필수정보 체크리스트 */}
              <div className="w-64 flex-shrink-0 bg-white border-r border-gray-200 p-4">
                <h2 className="text-lg font-semibold text-gray-900 mb-1">IFRS S2 필수 정보 체크리스트</h2>
                <p className="text-xs text-gray-400 mb-3">정확도 0%</p>
                <ul className="space-y-2">
                  <ChecklistItem status="complete" text="S2-5: 거버넌스 (감독 주체)" />
                  <ChecklistItem status="warning" text="S2-7: 리스크 및 기회 (전략 우선순위)" highlight />
                  <ChecklistItem status="error" text="S2-15: 시나리오 분석 (기준 연도)" />
                  <ChecklistItem status="error" text="Scope 1, 2, 3 배출량" />
                </ul>
              </div>

              {/* 중앙: 전문가 상담 UI */}
              <div className="flex-1 flex flex-col h-full bg-white border-r border-gray-200">
                {/* 대화 영역 */}
                <div className="flex-1 flex flex-col overflow-hidden">
                  {/* 대화 히스토리 */}
                  <div className="flex-1 overflow-y-auto p-6 space-y-4">
                    {/* 1단계: PDF 업로드 */}
                    <div className="flex justify-center">
                      <button className="bg-blue-50 text-blue-700 font-semibold px-4 py-2 rounded-lg text-sm">
                        1단계: TCFD 보고서.pdf (7.8MB) 업로드됨
                      </button>
                    </div>

                    {messages.map((msg, index) => (
                      msg.type === 'ai' ? (
                        <AiMessage key={index}>{msg.content}</AiMessage>
                      ) : (
                        <UserMessage key={index}>{msg.content}</UserMessage>
                      )
                    ))}
                    {isLoading && (
                      <div className="flex items-start space-x-3">
                        <div className="flex-shrink-0 bg-blue-600 p-2 rounded-full">
                          <svg className="w-5 h-5 text-white animate-pulse" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path d="M12 10v4M12 18v-2M15 13h-3M10 21h4c4.477 0 8-3.523 8-8V7c0-4.477-3.523-8-8-8H7C2.523-1 2 2.523 2 7v4c0 4.477 3.523 8 8 8z" />
                          </svg>
                        </div>
                        <div className="bg-white border border-gray-200 p-4 rounded-lg rounded-tl-none">
                          <p className="font-semibold text-blue-800">AI ESG Consultant</p>
                          <div className="flex items-center space-x-1 text-gray-500">
                            <span className="animate-bounce">●</span>
                            <span className="animate-bounce" style={{ animationDelay: '0.1s' }}>●</span>
                            <span className="animate-bounce" style={{ animationDelay: '0.2s' }}>●</span>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* 대화 입력창 */}
                  <div className="p-4 bg-gray-50 border-t border-gray-200">
                    <div className="flex items-center space-x-3">
                      <input
                        type="text"
                        value={inputValue}
                        onChange={(e) => setInputValue(e.target.value)}
                        onKeyDown={handleKeyDown}
                        className="flex-1 p-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                        placeholder="AI의 질문에 답변하거나, 궁금한 점을 입력하세요..."
                      />
                      <button
                        onClick={handleSend}
                        disabled={isLoading}
                        className="bg-blue-600 text-white p-3 rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed">
                        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <line x1="22" y1="2" x2="11" y2="13" />
                          <polygon points="22 2 15 22 11 13 2 9 22 2" />
                        </svg>
                      </button>
                    </div>
                  </div>
                </div>
              </div>

              {/* 오른쪽: 문단 생성 프리뷰 */}
              <div className="w-1/2 flex flex-col h-full bg-gray-50 overflow-y-auto">
                <div className="p-6">
                  <h2 className="text-xl font-semibold text-gray-900 mb-4">실시간 보고서 문단 프리뷰 (흐름 6.2)</h2>

                  {/* 생성된 문단 예시 1 */}
                  <PreviewCard title="IFRS S2-5: Governance" status="success">
                    <p>
                      당사는 기후 관련 리스크 및 기회에 대한 효과적인 감독을 위해{' '}
                      <HighlightSource tooltip="근거: 사용자 입력 (Q:감독 주체)">
                        이사회 산하 '지속가능경영위원회'
                      </HighlightSource>
                      를 설치하여 운영하고 있습니다.{' '}
                      <HighlightSource tooltip="근거: TCFD 보고서 p.5 (수정됨)">
                        위원회는 분기별
                      </HighlightSource>
                      로 기후 관련 주요 안건을 보고받고, 관련 전략 및 성과를 감독합니다.
                    </p>
                    <AiComment type="success">
                      기준서 충족. '위원회'의 구체적인 역할(예: 성과 측정, 보상 연계)을 추가하면 더 좋습니다.
                    </AiComment>
                  </PreviewCard>

                  {/* 생성된 문단 예시 2 */}
                  <PreviewCard title="IFRS S2-15: Scenario Analysis" status="error">
                    <p>
                      당사는 NZE 2050, 2도 시나리오 등을 활용하여 기후 관련 전환 리스크를 분석합니다.{' '}
                      <HighlightSource tooltip="근거: 정량요소 부족 (AI가 수정 요청)" error>
                        [기준연도 데이터 입력 필요]
                      </HighlightSource>
                      를 기준으로 분석을 수행하였으며...
                    </p>
                    <AiComment type="error">
                      정량요소 부족. 'Scope 1·2 데이터의 기준연도'를 왼쪽 채팅창에 입력해주세요. (흐름 4.1-6)
                    </AiComment>
                  </PreviewCard>
                </div>
              </div>
            </div>
          )}

          {/* 최종 보고서 화면 */}
          {activeTab === 'report' && (
            <div className="h-full overflow-y-auto bg-white">
              <div className="max-w-4xl mx-auto p-8 lg:p-12">
                <h1 className="text-3xl font-bold text-gray-900 mb-4">연간 기후 공시 보고서 (IFRS S2 기반)</h1>
                <p className="text-gray-600 mb-8">
                  본 보고서는 AI ESG Consultant와의 대화형 정보 수집 및 검증을 통해 생성되었습니다. (버전: v1.0)
                </p>

                <ReportSection title="1. Governance (S2-5)">
                  <p>당사는 기후 관련 리스크 및 기회에 대한 효과적인 감독을 위해 이사회 산하 '지속가능경영위원회'를 설치하여 운영하고 있습니다. 위원회는 분기별로 기후 관련 주요 안건을 보고받고, 관련 전략 및 성과를 감독합니다.</p>
                  <p>경영진은 위원회에서 승인된 기후 전략을 이행하며, 기후 리스크 식별 및 평가에 대한 책임을 집니다.</p>
                </ReportSection>

                <ReportSection title="2. Risks & Opportunities (S2-7)">
                  <p>당사는 단기, 중기, 장기에 걸쳐 식별된 주요 기후 관련 리스크와 기회를 관리합니다. 주요 전환 리스크로는 탄소 배출 규제 강화가 있으며, 물리적 리스크로는 극한 기후 현상 증가를 식별하였습니다.</p>
                  <p>기회 요인으로는 저탄소 제품 및 서비스 시장 확대를 식별하고, 관련 기술 개발에 R&D 투자를 집중하고 있습니다.</p>
                </ReportSection>

                <ReportSection title="3. Metrics & Targets (S2-15)">
                  <p>당사의 온실가스 배출량은 2023년(기준연도) 대비 2030년까지 Scope 1, 2 배출량을 40% 감축하는 것을 목표로 합니다.</p>
                  <p className="font-semibold text-gray-800">주요 지표:</p>
                  <ul className="list-disc pl-5">
                    <li>Scope 1 배출량 (tCO2e)</li>
                    <li>Scope 2 배출량 (Market-based, tCO2e)</li>
                    <li>기후 관련 전환 리스크에 노출된 자산 (금액)</li>
                  </ul>
                </ReportSection>
              </div>
            </div>
          )}

          {/* 도표 화면 */}
          {activeTab === 'chart' && (
            <div className="h-full overflow-y-auto bg-white">
              <div className="max-w-4xl mx-auto p-8 lg:p-12">
                <h1 className="text-3xl font-bold text-gray-900 mb-4">도표</h1>
                <p className="text-gray-600 mb-8">
                  여기에 도표 관련 컨텐츠가 표시됩니다.
                </p>
              </div>
            </div>
          )}

          {/* 회사정보 화면 */}
          {activeTab === 'company' && (
            <div className="h-full overflow-y-auto bg-white">
              <div className="max-w-4xl mx-auto p-8 lg:p-12">
                <h1 className="text-3xl font-bold text-gray-900 mb-4">회사정보</h1>
                <p className="text-gray-600 mb-8">
                  여기에 회사정보 관련 컨텐츠가 표시됩니다.
                </p>
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

// 컴포넌트들
function ChecklistItem({ status, text, highlight }: { status: 'complete' | 'warning' | 'error'; text: string; highlight?: boolean }) {
  const icons = {
    complete: <svg className="w-4 h-4 text-green-500 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" /><polyline points="22 4 12 14.01 9 11.01" /></svg>,
    warning: <svg className="w-4 h-4 text-yellow-500 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10" /><line x1="12" y1="8" x2="12" y2="12" /><line x1="12" y1="16" x2="12.01" y2="16" /></svg>,
    error: <svg className="w-4 h-4 text-red-500 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10" /><line x1="15" y1="9" x2="9" y2="15" /><line x1="9" y1="9" x2="15" y2="15" /></svg>,
  };

  return (
    <li className="flex items-center text-sm">
      {icons[status]}
      <span className={highlight ? 'text-gray-900 font-medium' : status === 'complete' ? 'text-gray-600' : 'text-gray-500'}>
        {text}
      </span>
    </li>
  );
}

function AiMessage({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex items-start space-x-3">
      <div className="flex-shrink-0 bg-blue-600 p-2 rounded-full">
        <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path d="M12 10v4M12 18v-2M15 13h-3M10 21h4c4.477 0 8-3.523 8-8V7c0-4.477-3.523-8-8-8H7C2.523-1 2 2.523 2 7v4c0 4.477 3.523 8 8 8z" />
        </svg>
      </div>
      <div className="bg-white border border-gray-200 p-4 rounded-lg rounded-tl-none max-w-lg">
        <p className="font-semibold text-blue-800">AI ESG Consultant</p>
        <div className="text-gray-700">{children}</div>
      </div>
    </div>
  );
}

function UserMessage({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex justify-end">
      <div className="bg-blue-100 p-4 rounded-lg rounded-br-none max-w-lg">
        <p className="font-semibold text-gray-800">사용자</p>
        <p className="text-gray-700">{children}</p>
      </div>
    </div>
  );
}

function HighlightSource({ children, tooltip, error }: { children: React.ReactNode; tooltip: string; error?: boolean }) {
  return (
    <span className={`relative cursor-pointer border-b-2 group ${error ? 'bg-red-100 border-red-400' : 'bg-yellow-100 border-amber-500'}`}>
      {children}
      <span className="invisible group-hover:visible opacity-0 group-hover:opacity-100 transition-opacity absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-2 py-1 bg-gray-700 text-white text-xs rounded whitespace-nowrap z-10">
        {tooltip}
      </span>
    </span>
  );
}

function PreviewCard({ title, status, children }: { title: string; status: 'success' | 'error'; children: React.ReactNode }) {
  return (
    <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200 mb-6">
      <h3 className="text-lg font-semibold text-gray-800 mb-3">{title}</h3>
      <div className="prose prose-sm max-w-none text-gray-700 space-y-3">{children}</div>
    </div>
  );
}

function AiComment({ type, children }: { type: 'success' | 'error'; children: React.ReactNode }) {
  const styles = {
    success: 'bg-blue-50 border-blue-200 text-blue-700',
    error: 'bg-red-50 border-red-200 text-red-700',
  };

  return (
    <div className={`border p-3 rounded-md text-sm mt-3 ${styles[type]}`}>
      <strong>AI 코멘트 (흐름 5.1 검증):</strong> <span>{children}</span>
    </div>
  );
}

function ReportSection({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="mb-10">
      <h2 className="text-2xl font-semibold text-gray-800 border-b-2 border-blue-500 pb-2 mb-4">{title}</h2>
      <div className="prose prose-lg max-w-none text-gray-700 space-y-4">{children}</div>
    </section>
  );
}
