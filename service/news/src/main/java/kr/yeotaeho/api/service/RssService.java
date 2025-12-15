package kr.yeotaeho.api.service;

import com.rometools.rome.feed.synd.SyndContent;
import com.rometools.rome.feed.synd.SyndEntry;
import com.rometools.rome.feed.synd.SyndEnclosure;
import com.rometools.rome.feed.synd.SyndFeed;
import com.rometools.rome.io.SyndFeedInput;
import com.rometools.rome.io.XmlReader;
import kr.yeotaeho.api.dto.NewsArticle;
import lombok.extern.slf4j.Slf4j;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;
import org.springframework.stereotype.Service;

import java.net.URL;
import java.time.*;
import java.time.format.DateTimeFormatter;
import java.util.Date;
import java.util.List;
import java.util.stream.Collectors;

@Slf4j
@Service
public class RssService {
    
    /**
     * RSS 피드 주소를 입력받아 뉴스 기사 목록을 반환하는 메서드
     */
    public List<NewsArticle> fetchNewsFromRss(String rssUrl) {
        try {
            // 1. URL 객체 생성
            URL feedUrl = new URL(rssUrl);
            
            // 2. SyndFeedInput을 사용하여 피드 읽기
            SyndFeedInput input = new SyndFeedInput();
            
            // 3. 피드 파싱 (XmlReader 사용)
            SyndFeed feed = input.build(new XmlReader(feedUrl));
            
            log.info("RSS 피드 수집 성공: URL={}, 기사 수={}", rssUrl, feed.getEntries().size());
            
            // 4. 피드 엔트리(기사)를 NewsArticle DTO로 변환
            List<NewsArticle> articles = feed.getEntries().stream()
                    .map(entry -> convertToNewsArticle(entry, rssUrl))
                    .collect(Collectors.toList());
            
            // 5. 날짜순 정렬 (최신 기사가 먼저)
            articles.sort((a, b) -> {
                try {
                    DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy.MM.dd");
                    LocalDate dateA = LocalDate.parse(a.getDate(), formatter);
                    LocalDate dateB = LocalDate.parse(b.getDate(), formatter);
                    return dateB.compareTo(dateA); // 내림차순
                } catch (Exception e) {
                    log.warn("날짜 정렬 실패: dateA={}, dateB={}", a.getDate(), b.getDate());
                    return 0;
                }
            });
            
            return articles;

        } catch (Exception e) {
            log.error("RSS 피드 읽기 실패: URL={}, 에러={}", rssUrl, e.getMessage(), e);
            return List.of();
        }
    }

    /**
     * SyndEntry 객체를 NewsArticle DTO로 변환하는 헬퍼 메서드
     */
    private NewsArticle convertToNewsArticle(SyndEntry entry, String rssUrl) {
        // HTML 태그 제거
        String cleanTitle = cleanHtml(entry.getTitle());
        String cleanDescription = entry.getDescription() != null 
                ? cleanHtml(entry.getDescription().getValue()) 
                : "";
        
        // 날짜 추출 및 포맷팅
        Date publishedDate = extractPublishedDate(entry);
        String formattedDate = formatDate(publishedDate);
        
        // 이미지 URL 추출
        String imageUrl = extractImageUrl(entry);
        
        // type은 RSS URL에서 카테고리 추출 (또는 기본값 "RSS")
        String type = extractCategoryFromUrl(rssUrl);
        
        return NewsArticle.builder()
                .type(type)
                .title(cleanTitle)
                .link(entry.getLink())
                .date(formattedDate)
                .description(cleanDescription)
                .image(imageUrl)
                .build();
    }
    
    /**
     * HTML 태그 제거
     */
    private String cleanHtml(String html) {
        if (html == null || html.isEmpty()) {
            return "";
        }
        
        // Jsoup을 사용하여 HTML 태그 제거
        Document doc = Jsoup.parse(html);
        String text = doc.text();
        
        // HTML 엔티티 디코딩
        return text.replaceAll("&quot;", "\"")
                .replaceAll("&amp;", "&")
                .replaceAll("&lt;", "<")
                .replaceAll("&gt;", ">")
                .replaceAll("&nbsp;", " ")
                .trim();
    }
    
    /**
     * 날짜 추출 (다중 소스 시도)
     */
    private Date extractPublishedDate(SyndEntry entry) {
        // 1. publishedDate 시도
        if (entry.getPublishedDate() != null) {
            return entry.getPublishedDate();
        }
        
        // 2. updatedDate 시도
        if (entry.getUpdatedDate() != null) {
            return entry.getUpdatedDate();
        }
        
        // 3. 모두 실패 시 현재 날짜 반환
        log.warn("날짜 추출 실패: entry={}", entry.getTitle());
        return new Date();
    }
    
    /**
     * 날짜 포맷팅 (java.util.Date -> yyyy.MM.dd)
     * java.time API 활용
     */
    private String formatDate(Date date) {
        if (date == null) {
            log.warn("날짜가 null입니다. 현재 날짜 반환");
            return LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy.MM.dd"));
        }
        
        try {
            // java.util.Date를 LocalDateTime으로 변환
            Instant instant = date.toInstant();
            LocalDateTime localDateTime = LocalDateTime.ofInstant(instant, ZoneId.systemDefault());
            
            // 표준 포맷으로 변환
            return localDateTime.format(DateTimeFormatter.ofPattern("yyyy.MM.dd"));
        } catch (Exception e) {
            log.warn("날짜 포맷팅 실패: date={}, 현재 날짜 반환", date, e);
            return LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy.MM.dd"));
        }
    }
    
    /**
     * 이미지 URL 추출 (Jsoup 활용)
     * 
     * 3단계 폴백 전략:
     * 1. Enclosure/Media 태그 확인
     * 2. Description/Content HTML 파싱
     * 3. 기본 이미지 반환
     */
    private String extractImageUrl(SyndEntry entry) {
        log.debug("이미지 추출 시작: title={}", entry.getTitle());
        
        // ============================================
        // 1단계: Enclosure/Media 태그 확인
        // ============================================
        String imageUrl = extractImageFromEnclosure(entry);
        if (imageUrl != null) {
            log.debug("✓ 이미지 추출 성공 (Enclosure): url={}", imageUrl);
            return imageUrl;
        }
        
        // ============================================
        // 2단계: Description HTML 파싱
        // ============================================
        imageUrl = extractImageFromDescription(entry);
        if (imageUrl != null) {
            log.debug("✓ 이미지 추출 성공 (Description): url={}", imageUrl);
            return imageUrl;
        }
        
        // ============================================
        // 2-1단계: Content:Encoded 파싱
        // ============================================
        imageUrl = extractImageFromContent(entry);
        if (imageUrl != null) {
            log.debug("✓ 이미지 추출 성공 (Content:Encoded): url={}", imageUrl);
            return imageUrl;
        }
        
        // ============================================
        // 3단계: 기본 이미지 반환
        // ============================================
        log.debug("✗ 이미지 추출 실패, 기본 이미지 반환: title={}", entry.getTitle());
        return "https://placehold.co/400x250/000000/FFFFFF?text=RSS";
    }
    
    /**
     * 1단계: Enclosure/Media 태그에서 이미지 추출
     * RSS 표준 <enclosure> 또는 Media RSS <media:content> 태그 확인
     */
    private String extractImageFromEnclosure(SyndEntry entry) {
        List<SyndEnclosure> enclosures = entry.getEnclosures();
        if (enclosures == null || enclosures.isEmpty()) {
            return null;
        }
        
        for (SyndEnclosure enclosure : enclosures) {
            // MIME 타입이 image/로 시작하는지 확인
            if (enclosure.getType() != null && enclosure.getType().startsWith("image/")) {
                String url = enclosure.getUrl();
                if (url != null && !url.isEmpty()) {
                    log.debug("  → Enclosure 이미지 발견: type={}, url={}", enclosure.getType(), url);
                    return url;
                }
            }
        }
        
        return null;
    }
    
    /**
     * 2단계: Description HTML에서 이미지 추출
     * <description> 태그의 HTML을 Jsoup으로 파싱
     */
    private String extractImageFromDescription(SyndEntry entry) {
        if (entry.getDescription() == null) {
            return null;
        }
        
        String html = entry.getDescription().getValue();
        if (html == null || html.isEmpty()) {
            return null;
        }
        
        return extractImageFromHtml(html, "Description");
    }
    
    /**
     * 2-1단계: Content:Encoded HTML에서 이미지 추출
     * <content:encoded> 태그의 HTML을 Jsoup으로 파싱 (더 풍부한 콘텐츠)
     */
    private String extractImageFromContent(SyndEntry entry) {
        List<SyndContent> contents = entry.getContents();
        if (contents == null || contents.isEmpty()) {
            return null;
        }
        
        // 첫 번째 content 사용
        SyndContent content = contents.get(0);
        String html = content.getValue();
        if (html == null || html.isEmpty()) {
            return null;
        }
        
        return extractImageFromHtml(html, "Content:Encoded");
    }
    
    /**
     * HTML에서 이미지 URL 추출 (Jsoup 활용)
     * - src 속성 우선
     * - data-src 속성 (Lazy Loading 대응)
     * - 여러 이미지가 있을 경우 첫 번째 선택
     */
    private String extractImageFromHtml(String html, String source) {
        try {
            Document doc = Jsoup.parse(html);
            Elements images = doc.select("img");
            
            if (images.isEmpty()) {
                log.debug("  → {} HTML에 <img> 태그 없음", source);
                return null;
            }
            
            log.debug("  → {} HTML에서 <img> 태그 {}개 발견", source, images.size());
            
            // 모든 img 태그를 순회하며 유효한 이미지 URL 찾기
            for (Element img : images) {
                // 1. src 속성 확인
                String src = img.attr("src");
                if (isValidImageUrl(src)) {
                    log.debug("    ✓ 유효한 src 발견: {}", src);
                    return src;
                }
                
                // 2. data-src 속성 확인 (Lazy Loading)
                String dataSrc = img.attr("data-src");
                if (isValidImageUrl(dataSrc)) {
                    log.debug("    ✓ 유효한 data-src 발견: {}", dataSrc);
                    return dataSrc;
                }
                
                // 3. data-lazy-src 속성 확인 (일부 사이트)
                String dataLazySrc = img.attr("data-lazy-src");
                if (isValidImageUrl(dataLazySrc)) {
                    log.debug("    ✓ 유효한 data-lazy-src 발견: {}", dataLazySrc);
                    return dataLazySrc;
                }
            }
            
            log.debug("  → {} HTML의 <img> 태그에서 유효한 URL을 찾지 못함", source);
            
        } catch (Exception e) {
            log.warn("HTML 파싱 중 오류 발생: source={}, error={}", source, e.getMessage());
        }
        
        return null;
    }
    
    /**
     * 이미지 URL 유효성 검증
     * - null이 아닌지
     * - 빈 문자열이 아닌지
     * - http:// 또는 https://로 시작하는지
     */
    private boolean isValidImageUrl(String url) {
        if (url == null || url.trim().isEmpty()) {
            return false;
        }
        
        // 상대 경로 제거 (http/https로 시작하는 절대 경로만 허용)
        String trimmedUrl = url.trim().toLowerCase();
        return trimmedUrl.startsWith("http://") || trimmedUrl.startsWith("https://");
    }
    
    /**
     * RSS URL에서 카테고리 추출 (간단한 추출 로직)
     */
    private String extractCategoryFromUrl(String rssUrl) {
        if (rssUrl == null) {
            return "RSS";
        }
        
        // URL에서 카테고리 키워드 추출 시도
        if (rssUrl.contains("economy") || rssUrl.contains("경제")) {
            return "경제";
        } else if (rssUrl.contains("politics") || rssUrl.contains("정치")) {
            return "정치";
        } else if (rssUrl.contains("society") || rssUrl.contains("사회")) {
            return "사회";
        } else if (rssUrl.contains("culture") || rssUrl.contains("문화")) {
            return "문화";
        } else if (rssUrl.contains("world") || rssUrl.contains("국제") || rssUrl.contains("세계")) {
            return "세계";
        } else if (rssUrl.contains("technology") || rssUrl.contains("it") || rssUrl.contains("과학")) {
            return "IT/과학";
        } else if (rssUrl.contains("sports") || rssUrl.contains("스포츠")) {
            return "스포츠";
        } else if (rssUrl.contains("entertainment") || rssUrl.contains("연예")) {
            return "연예";
        }
        
        return "RSS";
    }
}