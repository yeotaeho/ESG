package kr.yeotaeho.api.service;

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
import org.springframework.stereotype.Service;

import java.net.URL;
import java.time.*;
import java.time.format.DateTimeFormatter;
import java.util.Arrays;
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
     */
    private String extractImageUrl(SyndEntry entry) {
        // 1. enclosure 태그 확인 (media:content)
        List<SyndEnclosure> enclosures = entry.getEnclosures();
        if (!enclosures.isEmpty()) {
            SyndEnclosure enclosure = enclosures.get(0);
            if (enclosure.getType() != null && enclosure.getType().startsWith("image/")) {
                return enclosure.getUrl();
            }
        }
        
        // 2. description의 HTML에서 img 태그 추출 (Jsoup 사용)
        if (entry.getDescription() != null) {
            String html = entry.getDescription().getValue();
            if (html != null && !html.isEmpty()) {
                try {
                    Document doc = Jsoup.parse(html);
                    Element img = doc.selectFirst("img");
                    if (img != null) {
                        String src = img.attr("src");
                        if (src != null && !src.isEmpty()) {
                            return src;
                        }
                    }
                } catch (Exception e) {
                    log.debug("이미지 추출 실패: entry={}", entry.getTitle());
                }
            }
        }
        
        // 3. 기본 이미지 반환
        return "https://placehold.co/400x250/000000/FFFFFF?text=RSS";
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