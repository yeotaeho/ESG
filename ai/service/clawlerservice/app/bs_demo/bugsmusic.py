"""
벅스 뮤직 실시간 차트 크롤링 전략

1. 페이지 구조 분석
   - URL: https://music.bugs.co.kr/chart/track/realtime/total
   - 대상 테이블: class="list trackList byChart"
   - 각 행(tr)에서 추출할 정보:
     * title: p.title > a 태그
     * artist: p.artist > a 태그  
     * album: a.album 또는 td > a[href*='album'] 태그

2. 크롤링 단계
   Step 1: requests로 HTML 가져오기
   Step 2: BeautifulSoup로 파싱
   Step 3: CSS 선택자로 테이블 행 찾기
   Step 4: 각 행에서 title, artist, album 추출
   Step 5: JSON 형태로 변환하여 출력

3. 주의사항
   - User-Agent 헤더 설정 필요 (403 에러 방지)
   - robots.txt 확인 및 이용약관 준수
   - 요청 간 딜레이 권장 (서버 부하 방지)
"""

import requests
from bs4 import BeautifulSoup
import json

def crawl_bugs_chart():
    """벅스 뮤직 실시간 차트 크롤링"""
    
    # URL 설정
    url = 'https://music.bugs.co.kr/chart/track/realtime/total'
    
    # User-Agent 설정 (403 에러 방지)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    try:
        # HTTP 요청
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # HTML 파싱
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 차트 데이터 추출
        songs = []
        
        # 테이블 내의 각 행 선택
        # 실제 HTML 구조에 따라 선택자 조정 필요
        table = soup.find('table', class_='list')
        
        if table:
            rows = table.find('tbody').find_all('tr')
            
            for row in rows:
                try:
                    # title 추출
                    title_elem = row.find('p', class_='title')
                    title = title_elem.find('a').text.strip() if title_elem else 'N/A'
                    
                    # artist 추출
                    artist_elem = row.find('p', class_='artist')
                    artist = artist_elem.find('a').text.strip() if artist_elem else 'N/A'
                    
                    # album 추출
                    album_elem = row.find('a', class_='album')
                    if not album_elem:
                        # 대체 방법: album 링크 찾기
                        album_elem = row.find('a', href=lambda x: x and '/album/' in x)
                    album = album_elem.text.strip() if album_elem else 'N/A'
                    
                    songs.append({
                        'title': title,
                        'artist': artist,
                        'album': album
                    })
                    
                except Exception as e:
                    print(f"행 파싱 중 오류: {e}")
                    continue
        
        # JSON 형태로 출력
        print(json.dumps(songs, ensure_ascii=False, indent=2))
        return songs
        
    except requests.exceptions.RequestException as e:
        print(f"HTTP 요청 오류: {e}")
        return []
    except Exception as e:
        print(f"크롤링 오류: {e}")
        return []

if __name__ == "__main__":
    print("=== 벅스 뮤직 실시간 차트 크롤링 시작 ===\n")
    crawl_bugs_chart()
    print("\n=== 크롤링 완료 ===")

