"""
Search tools - ALL FREE, NO API KEYS REQUIRED!

- arXiv API: Free, no key needed
- Google Scholar: Free via Playwright scraping
- Google Search: Free via Playwright scraping
"""

import asyncio
import re
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from urllib.parse import quote_plus

import httpx
from bs4 import BeautifulSoup

# Optional: scholarly for Google Scholar (can fallback to Playwright)
try:
    from scholarly import scholarly
    HAS_SCHOLARLY = True
except ImportError:
    HAS_SCHOLARLY = False


async def search_arxiv(
    query: str,
    max_results: int = 20,
    max_age_days: int = 365,
) -> List[Dict[str, Any]]:
    """
    Search arXiv using their FREE API (no key required!).
    
    Args:
        query: Search query
        max_results: Maximum number of results
        max_age_days: Only return papers from last N days
        
    Returns:
        List of paper dictionaries
    """
    # Build arXiv API URL (HTTPS required)
    encoded_query = quote_plus(query)
    url = (
        f"https://export.arxiv.org/api/query?"
        f"search_query=all:{encoded_query}"
        f"&start=0"
        f"&max_results={max_results}"
        f"&sortBy=submittedDate"
        f"&sortOrder=descending"
    )
    
    papers = []
    cutoff_date = datetime.utcnow() - timedelta(days=max_age_days)
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=30)
            response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "xml")
        entries = soup.find_all("entry")
        
        for entry in entries:
            # Parse date
            published = entry.find("published")
            if published:
                date_str = published.text.strip()
                try:
                    pub_date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                    if pub_date.replace(tzinfo=None) < cutoff_date:
                        continue  # Skip old papers
                except:
                    pass
            else:
                date_str = "Unknown"
            
            # Extract fields
            title_elem = entry.find("title")
            title = title_elem.text.strip().replace("\n", " ") if title_elem else ""
            
            summary_elem = entry.find("summary")
            abstract = summary_elem.text.strip() if summary_elem else ""
            
            link_elem = entry.find("id")
            link = link_elem.text.strip() if link_elem else ""
            
            # Get authors
            authors = []
            for author in entry.find_all("author"):
                name = author.find("name")
                if name:
                    authors.append(name.text.strip())
            
            papers.append({
                "title": title,
                "link": link,
                "abstract": abstract,
                "date": date_str,
                "authors": authors,
                "source": "arXiv",
            })
        
        print(f"✅ arXiv: Found {len(papers)} papers for '{query}'")
        return papers
        
    except Exception as e:
        print(f"❌ arXiv search error: {e}")
        return []


async def search_google_scholar(
    query: str,
    max_results: int = 20,
    max_age_days: int = 365,
) -> List[Dict[str, Any]]:
    """
    Search Google Scholar - FREE, no API key!
    
    Returns papers with citation counts for ranking.
    Uses the 'scholarly' library or falls back to Playwright scraping.
    
    Args:
        query: Search query
        max_results: Maximum number of results
        max_age_days: Only return papers from last N days
        
    Returns:
        List of paper dictionaries with citations for ranking
    """
    papers = []
    
    if HAS_SCHOLARLY:
        try:
            # Use scholarly library (simpler, but may hit rate limits)
            search_query = scholarly.search_pubs(query)
            
            for i, result in enumerate(search_query):
                if i >= max_results:
                    break
                
                # Extract year and filter by recency
                year = result.get("bib", {}).get("pub_year", "")
                pub_year = None
                if year:
                    try:
                        pub_year = int(year)
                        current_year = datetime.now().year
                        if current_year - pub_year > (max_age_days // 365 + 1):
                            continue
                    except:
                        pass
                
                # Get citation count for ranking
                citations = result.get("num_citations", 0)
                
                # Calculate recency score (newer = higher)
                recency_score = 0
                if pub_year:
                    years_old = datetime.now().year - pub_year
                    recency_score = max(0, 10 - years_old)  # 0-10 scale
                
                papers.append({
                    "title": result.get("bib", {}).get("title", ""),
                    "link": result.get("pub_url", result.get("eprint_url", "")),
                    "abstract": result.get("bib", {}).get("abstract", ""),
                    "date": str(year) if year else "Unknown",
                    "year": pub_year,
                    "authors": result.get("bib", {}).get("author", []),
                    "source": "Google Scholar",
                    "citations": citations,
                    "recency_score": recency_score,
                    "venue": result.get("bib", {}).get("venue", ""),
                })
            
            print(f"✅ Google Scholar: Found {len(papers)} papers for '{query}'")
            return papers
            
        except Exception as e:
            print(f"⚠️ Scholarly library error: {e}, falling back to Playwright")
    
    # Fallback: Use Playwright to scrape Google Scholar directly
    return await _search_scholar_playwright(query, max_results)


async def _search_scholar_playwright(
    query: str,
    max_results: int = 20,
) -> List[Dict[str, Any]]:
    """
    Scrape Google Scholar using Playwright (fallback method).
    Extracts citation counts for ranking.
    """
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        print("❌ Playwright not installed. Run: pip install playwright && playwright install")
        return []
    
    papers = []
    encoded_query = quote_plus(query)
    url = f"https://scholar.google.com/scholar?q={encoded_query}&hl=en&as_ylo={datetime.now().year - 5}"
    
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            
            # Set user agent to avoid blocking
            await page.set_extra_http_headers({
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
            })
            
            await page.goto(url, timeout=30000)
            await page.wait_for_selector(".gs_ri", timeout=15000)
            
            html = await page.content()
            await browser.close()
        
        soup = BeautifulSoup(html, "html.parser")
        results = soup.select(".gs_ri")[:max_results]
        
        for result in results:
            title_elem = result.select_one(".gs_rt a")
            title = title_elem.text.strip() if title_elem else ""
            link = title_elem.get("href", "") if title_elem else ""
            
            snippet_elem = result.select_one(".gs_rs")
            abstract = snippet_elem.text.strip() if snippet_elem else ""
            
            # Extract year from citation info
            info_elem = result.select_one(".gs_a")
            info_text = info_elem.text if info_elem else ""
            year_match = re.search(r"\b(19|20)\d{2}\b", info_text)
            year_str = year_match.group() if year_match else "Unknown"
            pub_year = int(year_str) if year_str != "Unknown" else None
            
            # Extract citation count
            citations = 0
            cite_elem = result.select_one(".gs_fl a")
            if cite_elem:
                cite_text = cite_elem.text
                cite_match = re.search(r"Cited by (\d+)", cite_text)
                if cite_match:
                    citations = int(cite_match.group(1))
            
            # Calculate recency score
            recency_score = 0
            if pub_year:
                years_old = datetime.now().year - pub_year
                recency_score = max(0, 10 - years_old)
            
            papers.append({
                "title": title,
                "link": link,
                "abstract": abstract,
                "date": year_str,
                "year": pub_year,
                "authors": [],
                "source": "Google Scholar",
                "citations": citations,
                "recency_score": recency_score,
            })
        
        print(f"✅ Google Scholar (Playwright): Found {len(papers)} papers for '{query}'")
        return papers
        
    except Exception as e:
        print(f"❌ Google Scholar scraping error: {e}")
        return []


async def search_google(
    query: str,
    max_results: int = 10,
    site_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Search Google using Playwright - FREE, no API key!
    
    Args:
        query: Search query
        max_results: Maximum number of results
        site_filter: Optional site filter (e.g., "github.com")
        
    Returns:
        List of web result dictionaries
    """
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        print("❌ Playwright not installed")
        return []
    
    results = []
    
    # Build search query
    search_query = query
    if site_filter:
        search_query = f"site:{site_filter} {query}"
    
    encoded_query = quote_plus(search_query)
    url = f"https://www.google.com/search?q={encoded_query}&num={max_results}"
    
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            
            await page.set_extra_http_headers({
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
            })
            
            await page.goto(url, timeout=30000)
            
            # Wait for results
            try:
                await page.wait_for_selector("#search", timeout=10000)
            except:
                pass
            
            html = await page.content()
            await browser.close()
        
        soup = BeautifulSoup(html, "html.parser")
        
        # Parse search results
        for div in soup.select("div.g")[:max_results]:
            title_elem = div.select_one("h3")
            link_elem = div.select_one("a")
            snippet_elem = div.select_one("div.VwiC3b, span.aCOpRe")
            
            if title_elem and link_elem:
                results.append({
                    "title": title_elem.text.strip(),
                    "link": link_elem.get("href", ""),
                    "snippet": snippet_elem.text.strip() if snippet_elem else "",
                    "source": "Google",
                })
        
        print(f"✅ Google: Found {len(results)} results for '{query}'")
        return results
        
    except Exception as e:
        print(f"❌ Google search error: {e}")
        return []


# Convenience function to search all sources
async def search_all(
    query: str,
    sources: List[str] = None,
    max_results_per_source: int = 10,
    max_age_days: int = 90,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Search multiple sources in parallel.
    
    Args:
        query: Search query
        sources: List of sources to search (default: arxiv, scholar)
        max_results_per_source: Max results per source
        max_age_days: Recency filter
        
    Returns:
        Dictionary with results from each source
    """
    if sources is None:
        sources = ["arxiv", "scholar"]
    
    tasks = []
    source_names = []
    
    for source in sources:
        if source == "arxiv":
            tasks.append(search_arxiv(query, max_results_per_source, max_age_days))
            source_names.append("arxiv")
        elif source == "scholar":
            tasks.append(search_google_scholar(query, max_results_per_source, max_age_days))
            source_names.append("scholar")
        elif source == "google":
            tasks.append(search_google(query, max_results_per_source))
            source_names.append("google")
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    output = {}
    for name, result in zip(source_names, results):
        if isinstance(result, Exception):
            print(f"❌ {name} search failed: {result}")
            output[name] = []
        else:
            output[name] = result
    
    return output
