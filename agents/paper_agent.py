"""
Paper Research Agent
Specialized agent for searching and filtering academic papers
"""

import asyncio
from typing import List, Dict, Any
import sys
import os

# Add parent directory to path to import FinRA_agent
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class PaperResearchAgent:
    """
    Autonomous agent for academic paper research.
    
    Searches multiple sources (arXiv, Nature, IEEE) in parallel
    and applies intelligent filtering.
    """
    
    def __init__(self, assistant):
        """
        Initialize the paper research agent.
        
        Args:
            assistant: FinanceResearchAssistant instance with scrapers
        """
        self.assistant = assistant
        self.papers: List[Dict[str, Any]] = []
    
    async def search_papers_parallel(
        self,
        search_terms: List[str],
        max_age_days: int = 90
    ) -> List[Dict[str, Any]]:
        """
        Search papers across all sources in parallel.
        
        Args:
            search_terms: List of search queries
            max_age_days: Recency filter in days
            
        Returns:
            List of paper dictionaries
        """
        if not self.assistant.use_async:
            # Fall back to sync mode
            return self._search_papers_sync(search_terms, max_age_days)
        
        tasks = []
        
        # Create tasks for each source/term combination
        for search_term in search_terms:
            for source in self.assistant.sources:
                if hasattr(self.assistant, 'scrape_arxiv_async'):
                    # Use async scrapers if available
                    if source["name"] == "arXiv":
                        tasks.append(
                            self.assistant.scrape_arxiv_async(search_term, source)
                        )
                    elif "Nature" in source["name"]:
                        tasks.append(
                            self.assistant.scrape_nature_async(search_term, source)
                        )
                    elif "IEEE" in source["name"]:
                        tasks.append(
                            self.assistant.scrape_ieee_async(search_term, source)
                        )
                else:
                    # Fallback to sync in thread
                    tasks.append(
                        asyncio.to_thread(source["scraper"], search_term, source)
                    )
        
        print(f"🔍 Running {len(tasks)} paper searches in parallel...")
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flatten results and filter exceptions
        all_papers = []
        for result in results:
            if isinstance(result, list):
                all_papers.extend(result)
            elif isinstance(result, Exception):
                print(f"Warning: Search task failed: {result}")
        
        # Deduplicate and filter by date
        unique_papers = self._deduplicate_papers(all_papers)
        recent_papers = self._filter_by_recency(unique_papers, max_age_days)
        
        self.papers = recent_papers
        return recent_papers
    
    def _search_papers_sync(
        self,
        search_terms: List[str],
        max_age_days: int
    ) -> List[Dict[str, Any]]:
        """Synchronous fallback for paper search"""
        all_papers = []
        
        for search_term in search_terms:
            for source in self.assistant.sources:
                try:
                    results = source["scraper"](search_term, source)
                    all_papers.extend(results)
                except Exception as e:
                    print(f"Error: {e}")
        
        unique_papers = self._deduplicate_papers(all_papers)
        recent_papers = self._filter_by_recency(unique_papers, max_age_days)
        
        return recent_papers
    
    def _deduplicate_papers(self, papers: List[Dict]) -> List[Dict]:
        """Remove duplicate papers by normalized title"""
        import re
        
        unique = []
        seen_titles = set()
        
        for paper in papers:
            # Normalize title
            title = paper.get("title", "")
            norm_title = re.sub(r"\s+", " ", title.strip().lower())
            
            if norm_title not in seen_titles and norm_title:
                seen_titles.add(norm_title)
                unique.append(paper)
        
        return unique
    
    def _filter_by_recency(
        self,
        papers: List[Dict],
        max_age_days: int
    ) -> List[Dict]:
        """Filter papers by publication date"""
        from datetime import datetime, timedelta
        
        cutoff = datetime.utcnow() - timedelta(days=max_age_days)
        filtered = []
        
        for paper in papers:
            date_str = paper.get("date", "")
            parsed_date = self.assistant._parse_date(date_str)
            
            if parsed_date >= cutoff:
                paper["parsed_date"] = parsed_date.isoformat()
                filtered.append(paper)
        
        # Sort by date (most recent first)
        filtered.sort(key=lambda x: x.get("parsed_date", ""), reverse=True)
        
        return filtered
