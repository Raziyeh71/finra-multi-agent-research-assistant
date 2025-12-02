"""
Scraper tools using Playwright - ALL FREE, NO API KEYS!

WebGPT-style browsing: LLM controls what to scrape and extract.
"""

import asyncio
import re
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from urllib.parse import quote_plus, urljoin

from bs4 import BeautifulSoup


async def get_browser():
    """Get a Playwright browser instance."""
    from playwright.async_api import async_playwright
    
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(headless=True)
    return playwright, browser


async def read_webpage(
    url: str,
    extract_text: bool = True,
    extract_links: bool = False,
    timeout: int = 30000,
) -> Dict[str, Any]:
    """
    Read and extract content from any webpage - WebGPT style!
    
    Args:
        url: URL to read
        extract_text: Whether to extract main text content
        extract_links: Whether to extract links
        timeout: Page load timeout in milliseconds
        
    Returns:
        Dictionary with title, text, links, etc.
    """
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        return {"error": "Playwright not installed"}
    
    result = {
        "url": url,
        "title": "",
        "text": "",
        "links": [],
        "error": None,
    }
    
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            
            await page.set_extra_http_headers({
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
            })
            
            await page.goto(url, timeout=timeout, wait_until="domcontentloaded")
            
            # Get title
            result["title"] = await page.title()
            
            # Get HTML content
            html = await page.content()
            await browser.close()
        
        soup = BeautifulSoup(html, "html.parser")
        
        # Remove script and style elements
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()
        
        if extract_text:
            # Extract main content
            main_content = (
                soup.select_one("article") or
                soup.select_one("main") or
                soup.select_one(".content") or
                soup.select_one("#content") or
                soup.body
            )
            
            if main_content:
                result["text"] = main_content.get_text(separator="\n", strip=True)[:5000]
        
        if extract_links:
            for link in soup.find_all("a", href=True)[:50]:
                href = link.get("href", "")
                if href.startswith("http"):
                    result["links"].append({
                        "text": link.get_text(strip=True)[:100],
                        "url": href,
                    })
        
        return result
        
    except Exception as e:
        result["error"] = str(e)
        return result


async def scrape_nature(
    query: str,
    journal: str = "natmachintell",
    max_results: int = 10,
    max_age_days: int = 365,
) -> List[Dict[str, Any]]:
    """
    Scrape Nature journals using Playwright - FREE!
    
    Args:
        query: Search query
        journal: Nature journal code (natmachintell, ncomms, srep)
        max_results: Maximum results to return
        max_age_days: Recency filter
        
    Returns:
        List of paper dictionaries
    """
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        print("❌ Playwright not installed")
        return []
    
    papers = []
    encoded_query = quote_plus(query)
    
    # Map journal codes to names
    journal_names = {
        "natmachintell": "Nature Machine Intelligence",
        "ncomms": "Nature Communications",
        "srep": "Scientific Reports",
    }
    journal_name = journal_names.get(journal, "Nature")
    
    url = f"https://www.nature.com/search?q={encoded_query}&journal={journal}&date_range=last_year&order=date_desc"
    
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            
            await page.set_extra_http_headers({
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
            })
            
            await page.goto(url, timeout=60000)
            
            # Wait for results
            try:
                await page.wait_for_selector(".c-card__body, .c-no-results", timeout=30000)
            except:
                pass
            
            html = await page.content()
            await browser.close()
        
        soup = BeautifulSoup(html, "html.parser")
        
        # Check for no results
        if soup.select_one(".c-no-results"):
            print(f"ℹ️ Nature ({journal_name}): No results for '{query}'")
            return []
        
        articles = soup.select(".c-card__body")[:max_results]
        
        for article in articles:
            title_elem = article.select_one(".c-card__title a")
            if not title_elem:
                continue
            
            title = title_elem.text.strip()
            link = "https://www.nature.com" + title_elem.get("href", "")
            
            # Get date
            date_elem = article.select_one(".c-meta__item time")
            date_str = "Unknown"
            if date_elem:
                date_str = date_elem.get("datetime") or date_elem.text.strip()
            
            # Get abstract snippet
            abstract_elem = article.select_one(".c-card__summary")
            abstract = abstract_elem.text.strip() if abstract_elem else ""
            
            papers.append({
                "title": title,
                "link": link,
                "abstract": abstract,
                "date": date_str,
                "authors": [],
                "source": journal_name,
            })
        
        print(f"✅ Nature ({journal_name}): Found {len(papers)} papers for '{query}'")
        return papers
        
    except Exception as e:
        print(f"❌ Nature scraping error: {e}")
        return []


async def scrape_ieee(
    query: str,
    max_results: int = 10,
    max_age_days: int = 365,
) -> List[Dict[str, Any]]:
    """
    Scrape IEEE Xplore using Playwright - FREE!
    
    Args:
        query: Search query
        max_results: Maximum results to return
        max_age_days: Recency filter
        
    Returns:
        List of paper dictionaries
    """
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        print("❌ Playwright not installed")
        return []
    
    papers = []
    encoded_query = quote_plus(query)
    url = f"https://ieeexplore.ieee.org/search/searchresult.jsp?queryText={encoded_query}&sortType=newest"
    
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            
            await page.set_extra_http_headers({
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
            })
            
            await page.goto(url, timeout=60000)
            
            # Wait for results to load
            try:
                await page.wait_for_selector(".List-results-items", timeout=30000)
            except:
                pass
            
            # Scroll to load more results
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(2)
            
            html = await page.content()
            await browser.close()
        
        soup = BeautifulSoup(html, "html.parser")
        
        # Try different selectors for IEEE's varying layouts
        articles = (
            soup.select(".List-results-items .result-item")[:max_results] or
            soup.select(".result-item")[:max_results]
        )
        
        for article in articles:
            title_elem = article.select_one("h2 a, h3 a, .result-item-title a")
            if not title_elem:
                continue
            
            title = title_elem.text.strip()
            link = title_elem.get("href", "")
            if link and not link.startswith("http"):
                link = "https://ieeexplore.ieee.org" + link
            
            # Get abstract
            abstract_elem = article.select_one(".result-item-abstract, .description")
            abstract = abstract_elem.text.strip() if abstract_elem else ""
            
            # Get date
            date_elem = article.select_one(".publisher-info-container, .publication-date")
            date_str = date_elem.text.strip() if date_elem else "Unknown"
            
            papers.append({
                "title": title,
                "link": link,
                "abstract": abstract,
                "date": date_str,
                "authors": [],
                "source": "IEEE Xplore",
            })
        
        print(f"✅ IEEE: Found {len(papers)} papers for '{query}'")
        return papers
        
    except Exception as e:
        print(f"❌ IEEE scraping error: {e}")
        return []


async def scrape_youtube(
    query: str,
    max_results: int = 20,
    sort_by: str = "relevance",  # relevance, view_count, date
) -> List[Dict[str, Any]]:
    """
    Scrape YouTube search results using Playwright - FREE!
    
    Extracts view counts and engagement metrics for ranking.
    
    Args:
        query: Search query
        max_results: Maximum results to return
        sort_by: Sort order (relevance, view_count, date)
        
    Returns:
        List of video dictionaries with views/likes for ranking
    """
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        print("❌ Playwright not installed")
        return []
    
    videos = []
    encoded_query = quote_plus(query)
    
    # Add sort parameter
    sort_param = ""
    if sort_by == "view_count":
        sort_param = "&sp=CAM%253D"  # Sort by view count
    elif sort_by == "date":
        sort_param = "&sp=CAI%253D"  # Sort by upload date
    
    url = f"https://www.youtube.com/results?search_query={encoded_query}{sort_param}"
    
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            
            await page.goto(url, timeout=60000)
            
            # Wait for video results
            try:
                await page.wait_for_selector("ytd-video-renderer", timeout=15000)
            except:
                pass
            
            # Scroll to load more results
            for _ in range(3):
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                await asyncio.sleep(1)
            
            html = await page.content()
            await browser.close()
        
        soup = BeautifulSoup(html, "html.parser")
        
        # Parse video results
        video_elements = soup.select("ytd-video-renderer")[:max_results]
        
        for video in video_elements:
            # Title and link
            title_elem = video.select_one("#video-title")
            if not title_elem:
                continue
            
            title = title_elem.get("title", "") or title_elem.text.strip()
            link = title_elem.get("href", "")
            if link and not link.startswith("http"):
                link = "https://www.youtube.com" + link
            
            # Channel name
            channel_elem = video.select_one("#channel-name a, .ytd-channel-name a")
            channel = channel_elem.text.strip() if channel_elem else "Unknown"
            
            # Description snippet
            desc_elem = video.select_one("#description-text, .metadata-snippet-text")
            description = desc_elem.text.strip() if desc_elem else ""
            
            # Get metadata line (views and date)
            metadata_items = video.select("#metadata-line span")
            views = 0
            date_str = "Unknown"
            
            for item in metadata_items:
                text = item.text.strip().lower()
                # Parse view count
                if "view" in text:
                    view_match = re.search(r"([\d,.]+)\s*[kmb]?\s*view", text)
                    if view_match:
                        view_str = view_match.group(1).replace(",", "")
                        try:
                            views = int(float(view_str))
                            # Handle K, M, B suffixes
                            if "k" in text:
                                views *= 1000
                            elif "m" in text:
                                views *= 1000000
                            elif "b" in text:
                                views *= 1000000000
                        except:
                            pass
                # Parse date
                elif any(x in text for x in ["ago", "year", "month", "week", "day", "hour"]):
                    date_str = item.text.strip()
            
            # Calculate recency score from relative date
            recency_score = _parse_youtube_recency(date_str)
            
            videos.append({
                "title": title,
                "link": link,
                "channel": channel,
                "description": description,
                "date": date_str,
                "views": views,
                "recency_score": recency_score,
                "source": "YouTube",
            })
        
        print(f"✅ YouTube: Found {len(videos)} videos for '{query}'")
        return videos
        
    except Exception as e:
        print(f"❌ YouTube scraping error: {e}")
        return []


def _parse_youtube_recency(date_str: str) -> int:
    """
    Parse YouTube relative date string to recency score (0-10).
    Higher score = more recent.
    """
    if not date_str or date_str == "Unknown":
        return 0
    
    text = date_str.lower()
    
    # Extract number
    num_match = re.search(r"(\d+)", text)
    num = int(num_match.group(1)) if num_match else 1
    
    if "hour" in text or "minute" in text:
        return 10  # Very recent
    elif "day" in text:
        if num <= 7:
            return 9
        else:
            return 8
    elif "week" in text:
        if num <= 2:
            return 7
        else:
            return 6
    elif "month" in text:
        if num <= 3:
            return 5
        elif num <= 6:
            return 4
        else:
            return 3
    elif "year" in text:
        if num == 1:
            return 2
        else:
            return 1
    
    return 0


async def get_paper_abstract(url: str) -> str:
    """
    Fetch the full abstract from a paper URL.
    
    Works with arXiv, Nature, IEEE, and other academic sites.
    """
    result = await read_webpage(url, extract_text=True)
    
    if result.get("error"):
        return "Abstract not available"
    
    text = result.get("text", "")
    
    # Try to find abstract section
    abstract_patterns = [
        r"Abstract[:\s]*(.{100,1500}?)(?=\n\n|Introduction|Keywords|1\.|Background)",
        r"Summary[:\s]*(.{100,1500}?)(?=\n\n|Introduction|Keywords)",
    ]
    
    for pattern in abstract_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
    
    # Fallback: return first 500 chars
    return text[:500] if text else "Abstract not available"
