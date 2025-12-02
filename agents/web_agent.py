"""
Web Research Agent
Searches the web for latest code, blogs, and GitHub repos
"""

from typing import List, Dict, Any, Optional


class WebResearchAgent:
    """
    Agent for searching the web using Tavily API.
    
    Finds latest blog posts, GitHub repos, and tutorials
    related to AI/ML in finance/trading.
    """
    
    def __init__(self, tavily_api_key: Optional[str] = None):
        """
        Initialize the web research agent.
        
        Args:
            tavily_api_key: Tavily API key (optional)
        """
        self.tavily_api_key = tavily_api_key
        self.web_results: List[Dict[str, Any]] = []
        
        if tavily_api_key:
            try:
                from tavily import TavilyClient
                self.client = TavilyClient(api_key=tavily_api_key)
                self.enabled = True
            except ImportError:
                print("Warning: tavily package not installed. Web search disabled.")
                self.enabled = False
        else:
            self.enabled = False
    
    async def search_web(
        self,
        search_terms: List[str],
        max_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search the web for recent content.
        
        Args:
            search_terms: List of search queries
            max_results: Max results per query
            
        Returns:
            List of web result dictionaries
        """
        if not self.enabled:
            print("ℹ️  Web search disabled (no Tavily API key)")
            return []
        
        all_results = []
        
        for term in search_terms[:3]:  # Limit to first 3 terms to avoid rate limits
            try:
                print(f"🌐 Searching web for: {term}")
                
                # Search with focus on code and implementation
                query = f"{term} implementation code github"
                
                response = self.client.search(
                    query=query,
                    max_results=max_results,
                    search_depth="advanced",
                    include_domains=[
                        "github.com",
                        "medium.com",
                        "towardsdatascience.com",
                        "arxiv.org",
                        "paperswithcode.com"
                    ]
                )
                
                # Format results
                for result in response.get("results", []):
                    all_results.append({
                        "title": result.get("title", ""),
                        "link": result.get("url", ""),
                        "snippet": result.get("content", "")[:500],
                        "source": "Web (Tavily)",
                        "score": result.get("score", 0.0)
                    })
                
            except Exception as e:
                print(f"Warning: Web search failed for '{term}': {e}")
        
        # Sort by relevance score
        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        self.web_results = all_results
        print(f"✅ Found {len(all_results)} web results")
        
        return all_results
    
    def search_github(self, search_terms: List[str]) -> List[Dict[str, Any]]:
        """
        Search specifically for GitHub repositories.
        
        Args:
            search_terms: Search queries
            
        Returns:
            List of GitHub repo results
        """
        if not self.enabled:
            return []
        
        results = []
        
        for term in search_terms[:2]:  # Limit queries
            try:
                query = f"{term} github repo stars:>10"
                
                response = self.client.search(
                    query=query,
                    max_results=3,
                    include_domains=["github.com"]
                )
                
                for result in response.get("results", []):
                    results.append({
                        "title": result.get("title", ""),
                        "link": result.get("url", ""),
                        "snippet": result.get("content", "")[:300],
                        "source": "GitHub",
                        "score": result.get("score", 0.0)
                    })
                
            except Exception as e:
                print(f"Warning: GitHub search failed: {e}")
        
        return results
