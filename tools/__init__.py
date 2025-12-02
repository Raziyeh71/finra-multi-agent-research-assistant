"""
FinRA Tools
===========

Free tools for web scraping and search.
NO API KEYS REQUIRED (except OpenAI for LLM).

Tools:
- search_tools: Google Scholar, arXiv API, web search
- scraper_tools: Playwright-based scrapers for Nature, IEEE, YouTube
- summarizer_tools: GPT-3.5 powered summarization
"""

from .search_tools import (
    search_arxiv,
    search_google_scholar,
    search_google,
)
from .scraper_tools import (
    scrape_nature,
    scrape_ieee,
    scrape_youtube,
    read_webpage,
)
from .ranking import (
    rank_papers,
    rank_videos,
    calculate_paper_score,
    calculate_video_score,
    format_paper_for_display,
    format_video_for_display,
)

__all__ = [
    # Search (FREE - no API keys)
    "search_arxiv",
    "search_google_scholar", 
    "search_google",
    # Scrapers (FREE - Playwright)
    "scrape_nature",
    "scrape_ieee",
    "scrape_youtube",
    "read_webpage",
    # Ranking
    "rank_papers",
    "rank_videos",
    "calculate_paper_score",
    "calculate_video_score",
    "format_paper_for_display",
    "format_video_for_display",
]
