"""
다나와 제품 리스트 크롤링 전략

1. 페이지 구조 분석
   - URL: https://prod.danawa.com/list/?cate=10354413
   - 대상 데이터:
     * prod_name: 제품명
     * view_dic: 제품 설명/스펙
     * cm_mark: 마크/배지 정보

2. 크롤링 단계
   Step 1: requests로 HTML 가져오기
   Step 2: BeautifulSoup로 파싱
   Step 3: 제품 리스트 컨테이너 찾기
   Step 4: 각 제품에서 prod_name, view_dic, cm_mark 추출
   Step 5: JSON 형태로 변환하여 출력

3. 주의사항
   - User-Agent 헤더 설정 필요
   - 다나와는 동적 로딩을 사용할 수 있으므로 Selenium 필요할 수 있음
   - robots.txt 확인 및 이용약관 준수
   - 요청 간 딜레이 권장 (서버 부하 방지)

4. 기술적 고려사항
   - 다나와는 JavaScript로 동적 렌더링하는 경우가 많음
   - 정적 크롤링(BS4)으로 안될 경우 Selenium 사용 권장
   - Ajax 요청 분석하여 API 직접 호출 방법도 고려
"""

import requests
from bs4 import BeautifulSoup
import json
import time

def crawl_danawa_products():
    """다나와 제품 리스트 크롤링"""
    
    # URL 설정
    url = 'https://prod.danawa.com/list/?cate=10354413'
    
    # User-Agent 설정
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
        'Referer': 'https://www.danawa.com/'
    }
    
    try:
        # HTTP 요청
        print("페이지 요청 중...")
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # HTML 파싱
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 제품 데이터 추출
        products = []
        
        # 방법 1: 제품 리스트 컨테이너 찾기 (실제 HTML 구조에 따라 선택자 조정 필요)
        # 다나와의 경우 보통 ul.product_list 또는 div.main_prodlist 같은 구조 사용
        
        # 제품 아이템 찾기 (여러 패턴 시도)
        product_items = []
        
        # 패턴 1: product_list
        product_list = soup.find('ul', class_='product_list')
        if product_list:
            product_items = product_list.find_all('li', class_='prod_item')
        
        # 패턴 2: main_prodlist
        if not product_items:
            main_list = soup.find('div', class_='main_prodlist')
            if main_list:
                product_items = main_list.find_all('div', class_='prod_item')
        
        # 패턴 3: 일반 리스트 구조
        if not product_items:
            product_items = soup.find_all('div', attrs={'class': lambda x: x and 'prod' in x.lower()})
        
        print(f"찾은 제품 수: {len(product_items)}")
        
        for idx, item in enumerate(product_items, 1):
            try:
                # prod_name 추출 (여러 패턴 시도)
                prod_name = 'N/A'
                name_elem = item.find('p', class_='prod_name') or \
                           item.find('a', class_='prod_name') or \
                           item.find(attrs={'class': lambda x: x and 'prod_name' in x if x else False})
                
                if name_elem:
                    # a 태그가 있으면 그 안의 텍스트 추출
                    name_link = name_elem.find('a') if name_elem.name != 'a' else name_elem
                    prod_name = name_link.get_text(strip=True) if name_link else name_elem.get_text(strip=True)
                
                # view_dic 추출
                view_dic = 'N/A'
                dic_elem = item.find('div', class_='view_dic') or \
                          item.find('p', class_='view_dic') or \
                          item.find(attrs={'class': lambda x: x and 'view_dic' in x if x else False})
                
                if dic_elem:
                    view_dic = dic_elem.get_text(strip=True)
                
                # cm_mark 추출
                cm_mark = 'N/A'
                mark_elem = item.find('span', class_='cm_mark') or \
                           item.find('div', class_='cm_mark') or \
                           item.find(attrs={'class': lambda x: x and 'cm_mark' in x if x else False})
                
                if mark_elem:
                    cm_mark = mark_elem.get_text(strip=True)
                
                # 데이터가 하나라도 있으면 추가
                if prod_name != 'N/A' or view_dic != 'N/A' or cm_mark != 'N/A':
                    products.append({
                        'index': idx,
                        'prod_name': prod_name,
                        'view_dic': view_dic,
                        'cm_mark': cm_mark
                    })
                    
            except Exception as e:
                print(f"제품 {idx} 파싱 중 오류: {e}")
                continue
        
        # JSON 형태로 출력
        if products:
            print("\n=== 크롤링 결과 ===")
            print(json.dumps(products, ensure_ascii=False, indent=2))
        else:
            print("\n경고: 제품 데이터를 찾지 못했습니다.")
            print("페이지가 JavaScript로 동적 렌더링될 수 있습니다.")
            print("Selenium을 사용한 동적 크롤링을 고려하세요.")
            
            # 디버깅: HTML 구조 일부 출력
            print("\n=== HTML 구조 샘플 ===")
            body = soup.find('body')
            if body:
                print(body.prettify()[:1000])  # 처음 1000자만 출력
        
        return products
        
    except requests.exceptions.RequestException as e:
        print(f"HTTP 요청 오류: {e}")
        return []
    except Exception as e:
        print(f"크롤링 오류: {e}")
        import traceback
        traceback.print_exc()
        return []

if __name__ == "__main__":
    print("=== 다나와 제품 리스트 크롤링 시작 ===\n")
    crawl_danawa_products()
    print("\n=== 크롤링 완료 ===")

