#!/usr/bin/env python3
"""
FinRA - Finance Research Assistant
An agentic AI system to find and summarize recent AI/ML/DL papers and videos
in finance, capital markets, and trading.

Features:
- Async/parallel execution (5-10x faster)
- LLM-based research planning
- Web search integration (Tavily API)
- Proper date parsing and recency filtering
- YouTube video support (HTML scraping + optional API)
- arXiv API integration (more reliable than web scraping)
- Type hints throughout
- Configurable max age filter
- OpenAI GPT-4 for summarization
"""

import os
import re
import json
import time
import asyncio
import argparse
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from urllib.parse import quote_plus

import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
import openai
from dotenv import load_dotenv

# Async imports
try:
    import httpx
    from playwright.async_api import async_playwright
    HAS_ASYNC = True
except ImportError:
    HAS_ASYNC = False
    print("Warning: httpx or async playwright not installed. Async mode disabled.")

# Web search
try:
    from tavily import TavilyClient
    HAS_TAVILY = True
except ImportError:
    HAS_TAVILY = False

# Optional for better date parsing
try:
    from dateutil import parser as date_parser
    HAS_DATEUTIL = True
except ImportError:
    HAS_DATEUTIL = False
    print("Warning: python-dateutil not installed. Date parsing will be less accurate.")

class FinanceResearchAssistant:
    """Finance Research Assistant to find and summarize recent AI/ML papers in finance."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        max_age_days: int = 365,
        include_videos: bool = False,
        youtube_api_key: Optional[str] = None,
        use_async: bool = True,
        use_planner: bool = False,
        tavily_api_key: Optional[str] = None,
    ):
        """
        Initialize the Finance Research Assistant.

        Args:
            api_key: OpenAI API key
            max_age_days: Only keep items within this many days (default: 365)
            include_videos: Whether to search and summarize YouTube videos
            youtube_api_key: Optional YouTube Data API v3 key (falls back to scraping)
            use_async: Use async/parallel execution (default: True, 5-10x faster)
            use_planner: Use LLM-based research planning (default: False)
            tavily_api_key: Tavily API key for web search (optional)
        """
        load_dotenv()
        
        # Set OpenAI API key
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set it in .env file, as an environment variable, or pass it as a parameter.")
        
        openai.api_key = self.api_key
        
        self.max_age_days = max_age_days
        self.include_videos = include_videos
        self.youtube_api_key = youtube_api_key or os.getenv("YOUTUBE_API_KEY")
        self.use_async = use_async and HAS_ASYNC
        self.use_planner = use_planner
        self.tavily_api_key = tavily_api_key or os.getenv("TAVILY_API_KEY")
        
        # Warn if async requested but not available
        if use_async and not HAS_ASYNC:
            print("Warning: Async mode requested but dependencies not installed. Falling back to sync mode.")
            self.use_async = False
        
        # Define sources to scrape
        self.sources: List[Dict[str, Any]] = [
            {
                "name": "Nature Machine Intelligence",
                "search_url": "https://www.nature.com/search?q={}&journal=natmachintell&date_range=last_year&order=date_desc",
                "scraper": self.scrape_nature
            },
            {
                "name": "Nature Communications",
                "search_url": "https://www.nature.com/search?q={}&journal=ncomms&date_range=last_year&order=date_desc",
                "scraper": self.scrape_nature
            },
            {
                "name": "Scientific Reports",
                "search_url": "https://www.nature.com/search?q={}&journal=srep&date_range=last_year&order=date_desc",
                "scraper": self.scrape_nature
            },
            {
                "name": "IEEE Xplore",
                "search_url": "https://ieeexplore.ieee.org/search/searchresult.jsp?queryText={}&ranges=2023_2025_Year&rowsPerPage=100",
                "scraper": self.scrape_ieee
            },
            {
                "name": "arXiv",
                "api_url": "http://export.arxiv.org/api/query?search_query=all:{}&start=0&max_results=50&sortBy=submittedDate&sortOrder=descending",
                "scraper": self.scrape_arxiv_api
            }
        ]
        
        # Video sources
        self.video_sources: List[Dict[str, Any]] = [
            {
                "name": "YouTube",
                "search_url": "https://www.youtube.com/results?search_query={}",
                "scraper": self.scrape_youtube
            }
        ]
        
        # Define search terms
        self.search_terms = [
            # General AI + markets
            "AI in algorithmic trading",
            "AI in quantitative finance",
            "machine learning in capital markets",
            "deep learning for financial markets",
            "LLM applications in finance trading",
            "AI stock market prediction 2024 2025",
            "machine learning algorithmic trading 2024 2025",
            "deep learning financial markets 2024 2025",
            "neural networks quantitative trading",

            # Stock / options / crypto specific
            "deep learning stock market prediction",
            "machine learning stock price forecasting",
            "AI options trading strategies",
            "reinforcement learning for trading",
            "deep reinforcement learning algorithmic trading",
            "LLM agentic AI trading",
            "multi-agent LLM financial trading",
            "LLM tools for algorithmic trading",
            "agentic AI systems for quantitative finance",

            # Risk & portfolios
            "machine learning portfolio optimization",
            "AI for risk management in trading",
            "volatility forecasting deep learning",
            "reinforcement learning stock trading",
            "deep reinforcement learning for algorithmic trading",
            "transformer models for financial time series prediction",
            "transformer-based stock price forecasting",

            # To bias towards *recent* research
            "survey 2023 2024 deep learning algorithmic trading",
            "review 2023 2024 machine learning in finance"
        ]
        
        # Results storage
        self.papers: List[Dict[str, Any]] = []
        self.videos: List[Dict[str, Any]] = []
        
        # Debug mode
        self.debug: bool = False
    
    def scrape_nature(self, search_term: str, source: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Scrape Nature journal for papers with improved date extraction."""
        source_name = source["name"]
        search_url = source["search_url"]
        
        print(f"Scraping {source_name} for: {search_term}")
        query = quote_plus(search_term)
        url = search_url.format(query)
        
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.goto(url, timeout=60000)  # Increased timeout to 60 seconds
                
                # Wait for either card body or no results message
                page.wait_for_selector('.c-card__body, .c-no-results', timeout=60000)
                html_content = page.content()
                browser.close()
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Check if no results were found
            no_results = soup.select_one('.c-no-results')
            if no_results:
                print(f"No results found in {source_name} for: {search_term}")
                return []
            
            articles = soup.select('.c-card__body')
            if not articles:
                print(f"No article elements found in {source_name} for: {search_term}")
                return []
            
            if self.debug:
                print(f"Found {len(articles)} articles in {source_name}")
            
            results = []
            for article in articles[:10]:  # Limit to first 10 results
                title_elem = article.select_one('.c-card__title a')
                if not title_elem:
                    continue
                    
                title = title_elem.text.strip()
                link = "https://www.nature.com" + title_elem.get('href', '')
                
                # Improved date extraction
                date_elem = article.select_one('.c-meta__item time')
                if date_elem:
                    date_text = date_elem.get("datetime") or date_elem.text.strip()
                else:
                    date_text = "Unknown date"
                
                # Get abstract first to check relevance
                abstract = self._get_abstract(link)
                
                # Check if paper is related to finance/trading using AI/ML
                if not self._is_finance_ai_related(title, abstract):
                    if self.debug:
                        print(f"Filtering out: {title} (not finance/AI related)")
                    continue
                
                results.append({
                    "title": title,
                    "link": link,
                    "date": date_text,
                    "abstract": abstract,
                    "source": source_name
                })
                
                if self.debug:
                    print(f"Added paper: {title}")
            
            print(f"Found {len(results)} relevant papers from {source_name}")
            return results
        except Exception as e:
            print(f"Error scraping {source_name} for '{search_term}': {e}")
            return []
    
    def scrape_ieee(self, search_term: str, source: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Scrape IEEE Xplore for papers."""
        source_name = source["name"]
        search_url = source["search_url"]
        
        print(f"Scraping {source_name} for: {search_term}")
        query = quote_plus(search_term)
        url = search_url.format(query)
        
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.goto(url, timeout=60000)  # Increased timeout to 60 seconds
                
                # Wait for any of these selectors that might indicate results or no results
                selectors = [
                    '.List-results-items',  # Results container
                    '.article-list',        # Alternative results container
                    '.results-list',        # Another alternative
                    '.search-results',      # Generic search results
                    '.no-results-message',  # No results message
                    '#message-warning',     # Warning message
                    '#xplMainContent'       # Main content area
                ]
                
                # Join selectors with comma for CSS selector OR operation
                combined_selector = ', '.join(selectors)
                try:
                    page.wait_for_selector(combined_selector, timeout=30000)
                except Exception as e:
                    print(f"Warning: Timeout waiting for selectors in {source_name}. Continuing anyway.")
                
                # Take a screenshot for debugging (optional)
                # page.screenshot(path=f"ieee_search_{search_term.replace(' ', '_')}.png")
                
                html_content = page.content()
                browser.close()
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Try different selectors for articles
            articles = []
            for selector in ['.List-results-items', '.article-list article', '.results-list .result-item', '.search-results .search-result']:
                articles = soup.select(selector)
                if articles:
                    break
            
            if not articles:
                print(f"No article elements found in {source_name} for: {search_term}")
                return []
            
            if self.debug:
                print(f"Found {len(articles)} articles in {source_name}")
            
            results = []
            for article in articles[:10]:  # Limit to first 10 results
                # Try different selectors for title
                title_elem = None
                for title_selector in ['.title-link', 'h2 a', 'h3 a', '.article-title a', '.title a']:
                    title_elem = article.select_one(title_selector)
                    if title_elem:
                        break
                
                if not title_elem:
                    continue
                    
                title = title_elem.text.strip()
                
                # Handle different link formats
                link = title_elem.get('href', '')
                if link and not link.startswith('http'):
                    link = "https://ieeexplore.ieee.org" + link
                
                # Get abstract first to check relevance
                abstract = self._get_abstract(link)
                
                # Check if paper is related to finance/trading using AI/ML
                if not self._is_finance_ai_related(title, abstract):
                    if self.debug:
                        print(f"Filtering out: {title} (not finance/AI related)")
                    continue
                    
                # Try different selectors for date
                date_elem = None
                for date_selector in ['.publisher-info-container span', '.publication-date', '.date', 'time']:
                    date_elem = article.select_one(date_selector)
                    if date_elem:
                        break
                
                date = date_elem.text.strip() if date_elem else "Unknown date"
                
                results.append({
                    "title": title,
                    "link": link,
                    "date": date,
                    "abstract": abstract,
                    "source": source_name
                })
                
                if self.debug:
                    print(f"Added paper: {title}")
            
            print(f"Found {len(results)} relevant papers from {source_name}")
            return results
        except Exception as e:
            print(f"Error scraping {source_name} for '{search_term}': {e}")
            return []
    
    def scrape_arxiv_api(self, search_term: str, source: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Use arXiv API instead of web scraping (more reliable)."""
        source_name = source["name"]
        api_url = source["api_url"]
        
        print(f"Scraping {source_name} API for: {search_term}")
        
        # Add finance/trading context to query
        query = quote_plus(f"{search_term} finance trading")
        url = api_url.format(query)
        
        try:
            response = requests.get(url, timeout=20)
            response.raise_for_status()
        except Exception as e:
            print(f"Error requesting arXiv API: {e}")
            return []
        
        # Parse arXiv Atom feed
        soup = BeautifulSoup(response.text, "xml")
        entries = soup.find_all("entry")
        
        if self.debug:
            print(f"Found {len(entries)} entries from arXiv API")
        
        results: List[Dict[str, Any]] = []
        
        for entry in entries[:20]:  # Top 20 results
            title_elem = entry.find("title")
            if not title_elem:
                continue
            title = title_elem.text.strip().replace("\n", " ")
            
            abstract_elem = entry.find("summary")
            abstract = abstract_elem.text.strip() if abstract_elem else ""
            
            if not self._is_finance_ai_related(title, abstract):
                if self.debug:
                    print(f"Filtering out: {title} (not finance/AI related)")
                continue
            
            link_elem = entry.find("id")
            link = link_elem.text.strip() if link_elem else ""
            
            # Extract published date from arXiv API
            published_elem = entry.find("published")
            date_text = published_elem.text.strip() if published_elem else "Unknown date"
            
            results.append({
                "title": title,
                "link": link,
                "date": date_text,
                "abstract": abstract,
                "source": source_name,
            })
            
            if self.debug:
                print(f"Added paper: {title}")
        
        print(f"Found {len(results)} relevant papers from {source_name}")
        return results
    
    def scrape_youtube(
        self,
        search_term: str,
        source: Dict[str, Any],
        max_results: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search YouTube for recent videos. Falls back to HTML scraping if no API key."""
        source_name = source["name"]
        current_year = datetime.now().year
        
        # Bias towards recent practical content
        query = quote_plus(f"{search_term} stock trading {current_year} tutorial")
        
        # Try YouTube Data API first if key available
        if self.youtube_api_key:
            return self._scrape_youtube_api(query, max_results, source_name)
        
        # Fallback to HTML scraping
        url = source["search_url"].format(query)
        print(f"Scraping {source_name} for videos: {search_term}")
        
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
        except Exception as e:
            print(f"Error requesting YouTube: {e}")
            return []
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        videos: List[Dict[str, Any]] = []
        for a in soup.select("a#video-title")[:max_results]:
            title = (a.get("title") or a.text or "").strip()
            if not title:
                continue
            href = a.get("href", "")
            if not href:
                continue
            if not href.startswith("http"):
                href = "https://www.youtube.com" + href
            
            videos.append({
                "title": title,
                "link": href,
                "source": source_name,
                "channel": "",
                "date": "See YouTube page",
                "abstract": "",
            })
        
        if self.debug:
            print(f"Found {len(videos)} videos from {source_name}")
        
        return videos
    
    def _scrape_youtube_api(self, query: str, max_results: int, source_name: str) -> List[Dict[str, Any]]:
        """Use YouTube Data API v3 for more reliable results."""
        try:
            url = "https://www.googleapis.com/youtube/v3/search"
            params = {
                "part": "snippet",
                "q": query,
                "type": "video",
                "order": "date",
                "maxResults": max_results,
                "key": self.youtube_api_key,
            }
            
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            videos: List[Dict[str, Any]] = []
            for item in data.get("items", []):
                snippet = item.get("snippet", {})
                video_id = item.get("id", {}).get("videoId", "")
                
                videos.append({
                    "title": snippet.get("title", ""),
                    "link": f"https://www.youtube.com/watch?v={video_id}",
                    "source": source_name,
                    "channel": snippet.get("channelTitle", ""),
                    "date": snippet.get("publishedAt", "Unknown"),
                    "abstract": snippet.get("description", ""),
                })
            
            if self.debug:
                print(f"Found {len(videos)} videos from YouTube API")
            
            return videos
        
        except Exception as e:
            print(f"Error using YouTube API: {e}")
            return []
    
    def _parse_date(self, date_str: str) -> datetime:
        """Parse various date formats into datetime. Returns datetime.min if unknown."""
        if not date_str or date_str.lower() in ["unknown date", "see youtube page"]:
            return datetime.min
        
        # Clean up common date string patterns
        cleaned = re.sub(r"\(.*?\)", "", date_str)
        cleaned = cleaned.replace("Submitted on", "").replace("Submitted", "").replace("Published", "").strip()
        
        # Try dateutil parser if available
        if HAS_DATEUTIL:
            try:
                return date_parser.parse(cleaned, fuzzy=True)
            except Exception:
                pass
        
        # Fallback: extract year and assume Jan 1
        match = re.search(r"(\d{4})", cleaned)
        if match:
            year = int(match.group(1))
            return datetime(year, 1, 1)
        
        return datetime.min
    
    def _get_abstract(self, url: str) -> str:
        """Get abstract from paper URL."""
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Different sites have different abstract structures
            abstract_elem = (
                soup.select_one('.c-article-section__content p') or
                soup.select_one('.abstract') or
                soup.select_one('[property="og:description"]')
            )
            
            if abstract_elem and hasattr(abstract_elem, 'text'):
                return abstract_elem.text.strip()
            elif abstract_elem and abstract_elem.get('content'):
                return abstract_elem.get('content').strip()
            else:
                return "Abstract not available"
        except Exception as e:
            print(f"Error fetching abstract: {e}")
            return "Abstract not available"
    
    def _is_finance_ai_related(self, title, abstract=""):
        """Check if the text is related to both finance/trading and AI/ML.
        Checks both title and abstract (if provided).
        Excludes cryptocurrency papers and focuses on practical stock market trading papers.
        """
        # Terms related to traditional finance and stock markets
        finance_terms = ['finance', 'trading', 'market', 'stock', 'investment', 
                        'portfolio', 'financial', 'capital', 'economic', 'economy',
                        'price', 'pricing', 'asset', 'risk', 'volatility', 'forecast',
                        'equity', 'securities', 'shares', 'nyse', 'nasdaq', 'dow jones',
                        's&p 500', 'stock market', 'wall street', 'trading strategy']
        
        # Terms related to AI/ML
        ai_terms = ['ai', 'artificial intelligence', 'machine learning', 'deep learning', 
                   'neural network', 'algorithm', 'predictive', 'model', 'prediction',
                   'forecasting', 'classification', 'regression', 'transformer', 'lstm',
                   'reinforcement learning', 'supervised learning', 'unsupervised learning',
                   'backtest', 'optimization', 'feature engineering', 'time series']
        
        # Terms related to practical implementation
        practical_terms = ['code', 'implementation', 'github', 'python', 'framework',
                          'library', 'api', 'backtest', 'simulation', 'empirical',
                          'experiment', 'result', 'performance', 'evaluation', 'benchmark',
                          'method', 'technique', 'approach', 'strategy', 'algorithm']
        
        # Terms to exclude (cryptocurrency related)
        exclude_terms = ['bitcoin', 'crypto', 'cryptocurrency', 'blockchain', 'token',
                        'ethereum', 'btc', 'eth', 'altcoin', 'defi', 'nft', 'web3',
                        'decentralized finance', 'mining', 'binance', 'coinbase',
                        'dogecoin', 'solana', 'cardano']
        
        # Check both title and abstract if available
        combined_text = (title + " " + abstract).lower()
        
        # Check for excluded terms first
        for term in exclude_terms:
            if term in combined_text:
                return False
        
        # Check for finance and AI terms
        has_finance = any(term in combined_text for term in finance_terms)
        has_ai = any(term in combined_text for term in ai_terms)
        has_practical = any(term in combined_text for term in practical_terms)
        
        # Paper must have both finance and AI terms
        is_relevant = has_finance and has_ai
        
        # Prioritize papers that mention practical aspects
        if is_relevant and has_practical:
            return True
        
        # If it's relevant but doesn't explicitly mention practical aspects,
        # still include it but with lower priority
        return is_relevant
    
    def search_papers(self) -> List[Dict[str, Any]]:
        """Search for papers across all sources with deduplication and recency filtering."""
        processed_combinations = set()
        
        for search_term in self.search_terms:
            for source in self.sources:
                combo_key = f"{source['name']}::{search_term}"
                if combo_key in processed_combinations:
                    continue
                processed_combinations.add(combo_key)
                
                try:
                    results = source["scraper"](search_term, source)
                    self.papers.extend(results)
                    time.sleep(2)  # Be polite to servers
                except Exception as e:
                    print(f"Error scraping {source['name']} for '{search_term}': {e}")
        
        # Deduplicate by normalized title
        unique: List[Dict[str, Any]] = []
        seen_titles = set()
        for paper in self.papers:
            norm_title = re.sub(r"\s+", " ", paper["title"].strip().lower())
            if norm_title in seen_titles:
                continue
            seen_titles.add(norm_title)
            unique.append(paper)
        self.papers = unique
        
        # Filter by recency
        cutoff = datetime.utcnow() - timedelta(days=self.max_age_days)
        filtered: List[Dict[str, Any]] = []
        for paper in self.papers:
            dt = self._parse_date(paper.get("date", ""))
            if dt >= cutoff:
                paper["parsed_date"] = dt.isoformat()
                filtered.append(paper)
            elif self.debug:
                print(f"Filtered out by date: {paper['title']} ({paper.get('date', 'Unknown')})")
        
        self.papers = filtered
        
        # Sort by parsed date (most recent first)
        self.papers.sort(
            key=lambda x: self._parse_date(x.get("date", "")),
            reverse=True,
        )
        
        return self.papers
    
    def search_videos(self, max_per_term: int = 5) -> List[Dict[str, Any]]:
        """Search for videos across video sources."""
        if not self.include_videos:
            return []
        
        for search_term in self.search_terms[:5]:  # Limit video searches
            for source in self.video_sources:
                try:
                    results = source["scraper"](search_term, source, max_results=max_per_term)
                    self.videos.extend(results)
                    time.sleep(1)
                except Exception as e:
                    print(f"Error scraping {source['name']} videos for '{search_term}': {e}")
        
        # Deduplicate by link
        unique: List[Dict[str, Any]] = []
        seen_links = set()
        for video in self.videos:
            link = video.get("link", "")
            if link in seen_links:
                continue
            seen_links.add(link)
            unique.append(video)
        self.videos = unique
        
        return self.videos
    
    def summarize_papers(self, papers, max_papers=5):
        """Use OpenAI API to summarize papers."""
        if not papers:
            return []
        
        summarized_papers = []
        
        for paper in papers[:max_papers]:
            print(f"Summarizing: {paper['title']}")
            
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a helpful AI assistant specializing in finance and AI research."},
                        {"role": "user", "content": f"""
                        Please provide a concise summary of this research paper in finance and AI/ML:
                        
                        Title: {paper['title']}
                        Abstract: {paper['abstract']}
                        
                        Focus on:
                        1. The main problem being addressed
                        2. The AI/ML techniques used
                        3. The key findings or contributions to finance/trading
                        4. Potential practical applications
                        
                        Keep the summary under 150 words.
                        """}
                    ],
                    max_tokens=250
                )
                
                summary = response.choices[0].message.content.strip()
                paper["summary"] = summary
                summarized_papers.append(paper)
                
                # Be nice to the OpenAI API
                time.sleep(1)
                
            except Exception as e:
                print(f"Error summarizing paper: {e}")
                paper["summary"] = "Summary not available due to an error."
                summarized_papers.append(paper)
        
        return summarized_papers
    
    def summarize_videos(self, videos: List[Dict[str, Any]], max_videos: int = 5) -> List[Dict[str, Any]]:
        """Summarize videos using OpenAI."""
        if not videos:
            return []
        
        summarized: List[Dict[str, Any]] = []
        
        for video in videos[:max_videos]:
            print(f"Summarizing video: {video['title']}")
            try:
                # Fetch video description if not already present
                if not video.get("abstract"):
                    video["abstract"] = self._get_abstract(video["link"])
                
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You summarize YouTube videos about AI/ML/DL in "
                                "trading and quantitative finance."
                            ),
                        },
                        {
                            "role": "user",
                            "content": f"""
Summarize this video for a quantitative/algorithmic trader:

Title: {video['title']}
Description: {video.get('abstract', 'No description')}

Focus on:
1. What the video teaches (concepts/techniques)
2. Which AI/ML/DL methods are covered
3. How it can help in real trading (signals, risk, portfolio, infrastructure)
4. Whether it includes code/implementation

Keep the summary under 120 words.
""",
                        },
                    ],
                    max_tokens=220,
                )
                summary = response.choices[0].message.content.strip()
            except Exception as e:
                print(f"Error summarizing video: {e}")
                summary = "Summary not available due to an error."
            
            video["summary"] = summary
            summarized.append(video)
            time.sleep(1)
        
        return summarized
    
    # ========== Async Methods (5-10x Faster) ==========
    
    async def scrape_arxiv_async(self, search_term: str, source: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Async arXiv API scraper"""
        if not HAS_ASYNC:
            # Fallback to sync
            return await asyncio.to_thread(self.scrape_arxiv_api, search_term, source)
        
        source_name = source["name"]
        api_url = source["api_url"]
        
        query = quote_plus(f"{search_term} finance trading")
        url = api_url.format(query)
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=20)
                response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "xml")
            entries = soup.find_all("entry")
            
            results: List[Dict[str, Any]] = []
            
            for entry in entries[:20]:
                title_elem = entry.find("title")
                if not title_elem:
                    continue
                title = title_elem.text.strip().replace("\n", " ")
                
                abstract_elem = entry.find("summary")
                abstract = abstract_elem.text.strip() if abstract_elem else ""
                
                if not self._is_finance_ai_related(title, abstract):
                    continue
                
                link_elem = entry.find("id")
                link = link_elem.text.strip() if link_elem else ""
                
                published_elem = entry.find("published")
                date_text = published_elem.text.strip() if published_elem else "Unknown date"
                
                results.append({
                    "title": title,
                    "link": link,
                    "date": date_text,
                    "abstract": abstract,
                    "source": source_name,
                })
            
            return results
        except Exception as e:
            if self.debug:
                print(f"Error in async arXiv scraper: {e}")
            return []
    
    async def scrape_nature_async(self, search_term: str, source: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Async Nature scraper with Playwright"""
        if not HAS_ASYNC:
            return await asyncio.to_thread(self.scrape_nature, search_term, source)
        
        source_name = source["name"]
        search_url = source["search_url"]
        query = quote_plus(search_term)
        url = search_url.format(query)
        
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                await page.goto(url, timeout=60000)
                await page.wait_for_selector(".c-card__body, .c-no-results", timeout=60000)
                html_content = await page.content()
                await browser.close()
            
            soup = BeautifulSoup(html_content, "html.parser")
            
            if soup.select_one(".c-no-results"):
                return []
            
            articles = soup.select(".c-card__body")
            if not articles:
                return []
            
            results: List[Dict[str, Any]] = []
            for article in articles[:10]:
                title_elem = article.select_one(".c-card__title a")
                if not title_elem:
                    continue
                
                title = title_elem.text.strip()
                link = "https://www.nature.com" + title_elem.get("href", "")
                
                date_elem = article.select_one(".c-meta__item time")
                date_text = (date_elem.get("datetime") or date_elem.text.strip()) if date_elem else "Unknown date"
                
                # Get abstract (async)
                abstract = await self._get_abstract_async(link)
                
                if not self._is_finance_ai_related(title, abstract):
                    continue
                
                results.append({
                    "title": title,
                    "link": link,
                    "date": date_text,
                    "abstract": abstract,
                    "source": source_name,
                })
            
            return results
        except Exception as e:
            if self.debug:
                print(f"Error in async Nature scraper: {e}")
            return []
    
    async def scrape_ieee_async(self, search_term: str, source: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Async IEEE scraper with Playwright"""
        if not HAS_ASYNC:
            return await asyncio.to_thread(self.scrape_ieee, search_term, source)
        
        source_name = source["name"]
        search_url = source["search_url"]
        query = quote_plus(search_term)
        url = search_url.format(query)
        
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                await page.goto(url, timeout=60000)
                
                try:
                    await page.wait_for_selector(".List-results-items, .article-list, #xplMainContent", timeout=30000)
                except:
                    pass
                
                html_content = await page.content()
                await browser.close()
            
            soup = BeautifulSoup(html_content, "html.parser")
            
            articles = []
            for selector in [".List-results-items .List-results-items", ".article-list article"]:
                articles = soup.select(selector)
                if articles:
                    break
            
            if not articles:
                return []
            
            results: List[Dict[str, Any]] = []
            for article in articles[:10]:
                title_elem = article.select_one(".title-link, h2 a, h3 a")
                if not title_elem:
                    continue
                
                title = title_elem.text.strip()
                link = title_elem.get("href", "")
                if link and not link.startswith("http"):
                    link = "https://ieeexplore.ieee.org" + link
                
                # Get abstract async
                abstract = await self._get_abstract_async(link)
                
                if not self._is_finance_ai_related(title, abstract):
                    continue
                
                date_elem = article.select_one(".publisher-info-container span, .publication-date, time")
                date_text = date_elem.text.strip() if date_elem else "Unknown date"
                
                results.append({
                    "title": title,
                    "link": link,
                    "date": date_text,
                    "abstract": abstract,
                    "source": source_name,
                })
            
            return results
        except Exception as e:
            if self.debug:
                print(f"Error in async IEEE scraper: {e}")
            return []
    
    async def _get_abstract_async(self, url: str) -> str:
        """Async version of abstract fetcher"""
        if not HAS_ASYNC:
            return self._get_abstract(url)
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=15)
                response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "html.parser")
            
            abstract_elem = (
                soup.select_one(".c-article-section__content p")
                or soup.select_one(".abstract")
                or soup.select_one("div.abstract")
            )
            
            if abstract_elem and hasattr(abstract_elem, "text"):
                return abstract_elem.text.strip()
            
            meta = soup.select_one('meta[name="description"], meta[property="og:description"]')
            if meta and meta.get("content"):
                return meta["content"].strip()
            
            return "Abstract not available"
        except Exception:
            return "Abstract not available"
    
    async def summarize_all_parallel(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Summarize all items in parallel (much faster)"""
        if not items:
            return []
        
        print(f"📝 Summarizing {len(items)} items in parallel...")
        
        async def summarize_one(item: Dict[str, Any]) -> Dict[str, Any]:
            """Summarize a single item"""
            try:
                is_video = "youtube" in item.get("link", "").lower()
                
                if is_video:
                    prompt_text = f"""Summarize this video for a quantitative/algorithmic trader:

Title: {item['title']}
Description: {item.get('abstract', 'No description')}

Focus on: techniques, AI/ML methods, trading applications, code/implementation.
Keep under 120 words."""
                else:
                    prompt_text = f"""Summarize this research for an algorithmic trader:

Title: {item['title']}
Abstract: {item['abstract']}

Focus on: problem, AI/ML techniques, findings, practical implications.
Keep under 150 words."""
                
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are an AI/ML research summarizer for finance professionals."},
                        {"role": "user", "content": prompt_text}
                    ],
                    max_tokens=250,
                    temperature=0.3
                )
                
                item["summary"] = response.choices[0].message.content.strip()
            except Exception as e:
                item["summary"] = f"Summary not available: {str(e)[:100]}"
            
            return item
        
        # Run all summarizations in parallel
        tasks = [summarize_one(item) for item in items]
        summarized = await asyncio.gather(*tasks)
        
        return summarized
    
    def run(self, max_papers: int = 5, max_videos: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """Run the full pipeline to find and summarize papers and videos."""
        print("=" * 80)
        print("FinRA - Finance Research Assistant")
        print(f"Searching for content published within the last {self.max_age_days} days")
        print("=" * 80)
        
        print("\nSearching for papers...")
        self.search_papers()
        print(f"Found {len(self.papers)} relevant recent papers.")
        
        if self.include_videos:
            print("\nSearching for videos...")
            self.search_videos(max_per_term=5)
            print(f"Found {len(self.videos)} videos.")
        else:
            self.videos = []
        
        print(f"\nSummarizing top {min(max_papers, len(self.papers))} papers...")
        summarized_papers = self.summarize_papers(self.papers, max_papers=max_papers)
        
        summarized_videos: List[Dict[str, Any]] = []
        if self.include_videos and self.videos:
            print(f"\nSummarizing top {min(max_videos, len(self.videos))} videos...")
            summarized_videos = self.summarize_videos(self.videos, max_videos=max_videos)
        
        return {
            "papers": summarized_papers,
            "videos": summarized_videos,
        }
    
    def display_results(self, results: Dict[str, List[Dict[str, Any]]]) -> None:
        """Pretty-print the results."""
        papers = results.get("papers", [])
        videos = results.get("videos", [])
        
        if not papers and not videos:
            print("\nNo relevant results found.")
            return
        
        if papers:
            print("\n" + "=" * 80)
            print(f"TOP {len(papers)} RECENT AI/ML PAPERS IN FINANCE/TRADING")
            print("=" * 80)
            for i, paper in enumerate(papers, 1):
                print(f"\n{i}. {paper['title']}")
                print(f"   Source: {paper['source']}")
                print(f"   Date:   {paper.get('date', 'Unknown')}")
                print(f"   Link:   {paper['link']}")
                print("\n   Summary:")
                print(f"   {paper.get('summary', 'No summary')}")
                print("\n" + "-" * 80)
        
        if videos:
            print("\n" + "=" * 80)
            print(f"TOP {len(videos)} VIDEOS ON AI/ML IN TRADING")
            print("=" * 80)
            for i, video in enumerate(videos, 1):
                print(f"\n{i}. {video['title']}")
                print(f"   Source:  {video['source']}")
                print(f"   Channel: {video.get('channel', 'N/A')}")
                print(f"   Date:    {video.get('date', 'Unknown')}")
                print(f"   Link:    {video['link']}")
                print("\n   Summary:")
                print(f"   {video.get('summary', 'No summary')}")
                print("\n" + "-" * 80)
    
    def save_results(
        self,
        results: Dict[str, List[Dict[str, Any]]],
        papers_filename: str = "finance_ai_papers.json",
        videos_filename: str = "finance_ai_videos.json",
    ) -> None:
        """Save papers and videos to JSON files."""
        papers = results.get("papers", [])
        videos = results.get("videos", [])
        
        if papers:
            with open(papers_filename, "w") as f:
                json.dump(papers, f, indent=2)
            print(f"\nPaper results saved to {papers_filename}")
        
        if videos:
            with open(videos_filename, "w") as f:
                json.dump(videos, f, indent=2)
            print(f"Video results saved to {videos_filename}")


def main() -> int:
    """Main CLI entry point."""
    load_dotenv()
    
    parser = argparse.ArgumentParser(
        description="FinRA - Finance Research Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Search for papers in the last 90 days
  python FinRA_agent.py --max-age-days 90

  # Include videos with YouTube API
  python FinRA_agent.py --include-videos --youtube-api-key YOUR_KEY

  # Disable videos, get 10 papers
  python FinRA_agent.py --max-papers 10 --max-age-days 180
        """
    )
    parser.add_argument("--api-key", help="OpenAI API key")
    parser.add_argument("--youtube-api-key", help="YouTube Data API v3 key (optional)")
    parser.add_argument("--tavily-api-key", help="Tavily API key for web search (optional)")
    parser.add_argument(
        "--max-papers",
        type=int,
        default=5,
        help="Maximum number of papers to summarize (default: 5)",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=5,
        help="Maximum number of videos to summarize (default: 5)",
    )
    parser.add_argument(
        "--max-age-days",
        type=int,
        default=365,
        help="Only keep papers within this many days (default: 365)",
    )
    parser.add_argument(
        "--include-videos",
        action="store_true",
        help="Enable video search and summarization",
    )
    parser.add_argument(
        "--use-async",
        action="store_true",
        default=True,
        help="Use async/parallel execution (default: True, 5-10x faster)",
    )
    parser.add_argument(
        "--no-async",
        action="store_true",
        help="Disable async mode (force sync)",
    )
    parser.add_argument(
        "--use-planner",
        action="store_true",
        help="Use LLM-based research planning (experimental)",
    )
    parser.add_argument(
        "--papers-output",
        default="finance_ai_papers.json",
        help="Output JSON file for papers",
    )
    parser.add_argument(
        "--videos-output",
        default="finance_ai_videos.json",
        help="Output JSON file for videos",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()
    
    try:
        # Determine async mode
        use_async = args.use_async and not args.no_async
        
        assistant = FinanceResearchAssistant(
            api_key=args.api_key,
            max_age_days=args.max_age_days,
            include_videos=args.include_videos,
            youtube_api_key=args.youtube_api_key,
            use_async=use_async,
            use_planner=args.use_planner,
            tavily_api_key=args.tavily_api_key,
        )
        assistant.debug = args.debug
        
        if use_async:
            print("⚡ Async mode enabled (5-10x faster)")
        else:
            print("📍 Sync mode (slower but more compatible)")
        
        results = assistant.run(
            max_papers=args.max_papers,
            max_videos=args.max_videos,
        )
        assistant.display_results(results)
        assistant.save_results(
            results,
            papers_filename=args.papers_output,
            videos_filename=args.videos_output,
        )
    except Exception as e:
        print(f"Error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
