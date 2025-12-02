#!/usr/bin/env python3
"""
FinRA - Finance Research Assistant (Hybrid Version)
A tool to find and summarize recent AI/ML papers in finance, capital markets, and trading.
Uses web scraping and local Ollama LLM for summarization.
"""

import os
import re
import json
import time
import random
import argparse
import requests
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from dotenv import load_dotenv

# List of user agents to rotate through
USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0"
]

class FinanceResearchAssistant:
    """Finance Research Assistant to find and summarize recent AI/ML papers in finance."""
    
    def __init__(self, ollama_model="llama3"):
        """Initialize the Finance Research Assistant."""
        # Load environment variables from .env file
        load_dotenv()
        
        # Set Ollama model
        self.ollama_model = ollama_model
        
        # Define sources to scrape
        self.sources = [
            {
                "name": "arXiv",
                "search_url": "https://arxiv.org/search/?query={}&searchtype=all&abstracts=show&order=-announced_date_first&size=50",
                "scraper": self.scrape_arxiv
            },
            {
                "name": "Semantic Scholar",
                "search_url": "https://www.semanticscholar.org/search?q={}&sort=newest",
                "scraper": self.scrape_semantic_scholar
            }
        ]
        
        # Define search terms
        self.search_terms = [
            "AI finance trading",
            "machine learning capital markets",
            "deep learning financial markets",
            "AI algorithmic trading",
            "machine learning stock prediction"
        ]
        
        # Results storage
        self.papers = []
        
        # Debug mode
        self.debug = True
    
    def get_random_user_agent(self):
        """Get a random user agent from the list."""
        return random.choice(USER_AGENTS)
    
    def random_sleep(self, min_sec=1, max_sec=3):
        """Sleep for a random amount of time between min_sec and max_sec seconds."""
        sleep_time = random.uniform(min_sec, max_sec)
        if self.debug:
            print(f"Sleeping for {sleep_time:.2f} seconds...")
        time.sleep(sleep_time)
    
    def scrape_semantic_scholar(self, search_term):
        """Scrape Semantic Scholar for papers."""
        print(f"Scraping Semantic Scholar for: {search_term}")
        url = self.sources[1]["search_url"].format(search_term.replace(' ', '%20'))
        
        headers = {"User-Agent": self.get_random_user_agent()}
        
        try:
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Try different selectors for articles
            articles = soup.select("article")
            
            if not articles:
                print(f"No article elements found in Semantic Scholar for: {search_term}")
                if self.debug:
                    # Save the HTML for debugging in a debug folder
                    os.makedirs("debug", exist_ok=True)
                    debug_file = f"debug/semantic_debug_{search_term.replace(' ', '_')}.html"
                    with open(debug_file, "w") as f:
                        f.write(response.text)
                    print(f"Saved HTML to {debug_file}")
                return []
            
            if self.debug:
                print(f"Found {len(articles)} articles in Semantic Scholar")
            
            results = []
            for article in articles[:10]:  # Limit to first 10 results
                title_elem = article.select_one("h2 a")
                if not title_elem:
                    continue
                    
                title = title_elem.text.strip()
                link = "https://www.semanticscholar.org" + title_elem.get('href', '')
                
                # Try to get abstract
                abstract = self._get_abstract(link)
                
                # Check if paper is related to finance/trading using AI/ML
                if not self._is_finance_ai_related(title, abstract):
                    if self.debug:
                        print(f"Filtering out: {title} (not finance/AI related)")
                    continue
                
                # Try to get date
                date_elem = article.select_one(".cl-paper-pubdates")
                date = date_elem.text.strip() if date_elem else "Unknown date"
                
                results.append({
                    "title": title,
                    "link": link,
                    "date": date,
                    "abstract": abstract,
                    "source": "Semantic Scholar"
                })
                
                if self.debug:
                    print(f"Added paper: {title}")
                
                # Be nice to the server
                self.random_sleep()
            
            print(f"Found {len(results)} relevant papers from Semantic Scholar")
            return results
        except Exception as e:
            print(f"Error scraping Semantic Scholar for '{search_term}': {e}")
            return []
    
    def scrape_arxiv(self, search_term):
        """Scrape arXiv for papers."""
        print(f"Scraping arXiv for: {search_term}")
        url = self.sources[0]["search_url"].format(search_term.replace(" ", "+"))
        
        headers = {"User-Agent": self.get_random_user_agent()}
        
        try:
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            articles = soup.select('.arxiv-result')
            
            if self.debug:
                print(f"Found {len(articles)} articles in arXiv")
            
            results = []
            for article in articles[:10]:  # Limit to first 10 results
                title_elem = article.select_one('.title')
                if not title_elem:
                    continue
                    
                title = title_elem.text.strip()
                
                abstract_elem = article.select_one('.abstract-full')
                abstract = abstract_elem.text.strip() if abstract_elem else ""
                
                # Check if paper is related to finance/trading using AI/ML
                if not self._is_finance_ai_related(title, abstract):
                    if self.debug:
                        print(f"Filtering out: {title} (not finance/AI related)")
                    continue
                    
                link_elem = article.select_one('.list-title a')
                link = link_elem.get('href', '') if link_elem else ""
                
                date_elem = article.select_one('.submitted-date')
                date = date_elem.text.replace('Submitted', '').strip() if date_elem else "Unknown date"
                
                results.append({
                    "title": title,
                    "link": link,
                    "date": date,
                    "abstract": abstract,
                    "source": "arXiv"
                })
                
                if self.debug:
                    print(f"Added paper: {title}")
                
                # Be nice to the server
                self.random_sleep()
            
            print(f"Found {len(results)} relevant papers from arXiv")
            return results
        except Exception as e:
            print(f"Error scraping arXiv for '{search_term}': {e}")
            return []
    
    def _get_abstract(self, url):
        """Get abstract from paper URL."""
        try:
            headers = {"User-Agent": self.get_random_user_agent()}
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Different sites have different abstract structures
            abstract_elem = (
                soup.select_one('.abstract__text') or  # Semantic Scholar
                soup.select_one('.abstract-full') or   # arXiv
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
            if self.debug:
                print(f"Error fetching abstract: {e}")
            return "Abstract not available"
    
    def _is_finance_ai_related(self, title, abstract=""):
        """Check if the text is related to trading and AI/Machine Learning.
        Checks both title and abstract (if provided).
        """
        finance_terms = ['finance', 'trading', 'option trading', 'market', 'stock', 'investment', 
                        'portfolio', 'financial', 'capital market', 'economic', 'economy',
                        'price', 'pricing', 'asset', 'risk', 'volatility', 'forecast']
        ai_terms = ['ai', 'artificial intelligence', 'machine learning', 'deep learning', 
                   'neural network', 'algorithm', 'predictive', 'model', 'prediction',
                   'forecasting', 'classification', 'regression', 'transformer', 'lstm']
        
        # Check both title and abstract if available
        combined_text = (title + " " + abstract).lower()
        
        has_finance = any(term in combined_text for term in finance_terms)
        has_ai = any(term in combined_text for term in ai_terms)
        
        # If we have both terms, it's definitely relevant
        if has_finance and has_ai:
            return True
            
        # If we only have the title and it has one of the terms, consider it potentially relevant
        if not abstract and (has_finance or has_ai):
            return True
            
        return False
    
    def search_papers(self):
        """Search for papers across all sources and search terms."""
        # Track which source/term combinations we've already processed
        processed_combinations = set()
        
        for search_term in self.search_terms:
            for source in self.sources:
                # Create a unique key for this combination
                combo_key = f"{source['name']}:{search_term}"
                
                # Skip if we've already processed this combination
                if combo_key in processed_combinations:
                    if self.debug:
                        print(f"Skipping duplicate: {combo_key}")
                    continue
                
                processed_combinations.add(combo_key)
                
                try:
                    results = source["scraper"](search_term)
                    self.papers.extend(results)
                    # Be nice to the servers
                    self.random_sleep(2, 4)
                except Exception as e:
                    print(f"Error scraping {source['name']} for '{search_term}': {e}")
        
        # Remove duplicates based on title
        unique_papers = []
        seen_titles = set()
        
        for paper in self.papers:
            if paper["title"] not in seen_titles:
                seen_titles.add(paper["title"])
                unique_papers.append(paper)
        
        self.papers = unique_papers
        
        # Sort by date (most recent first)
        # This is approximate since date formats vary
        self.papers.sort(key=lambda x: x.get("date", ""), reverse=True)
        
        return self.papers
    
    def summarize_with_ollama(self, paper):
        """Use local Ollama LLM to summarize papers."""
        print(f"Summarizing with Ollama ({self.ollama_model}): {paper['title']}")
        
        try:
            prompt = f"""
            Please provide a concise summary of this research paper in finance and AI/ML:
            
            Title: {paper['title']}
            Abstract: {paper['abstract']}
            
            Focus on:
            1. The main problem being addressed
            2. The AI/ML techniques used
            3. The key findings or contributions to finance/trading
            4. Potential practical applications
            
            Keep the summary under 150 words.
            """
            
            response = requests.post('http://localhost:11434/api/generate', 
                                   json={
                                       "model": self.ollama_model,
                                       "prompt": prompt,
                                       "stream": False
                                   })
            
            if response.status_code == 200:
                summary = response.json()['response'].strip()
                return summary
            else:
                print(f"Error from Ollama API: {response.status_code}")
                return "Summary not available due to an error with the local LLM."
                
        except Exception as e:
            print(f"Error summarizing paper with Ollama: {e}")
            return "Summary not available due to an error with the local LLM."
    
    def summarize_papers(self, papers, max_papers=5):
        """Summarize papers using local Ollama LLM."""
        if not papers:
            return []
        
        summarized_papers = []
        
        for paper in papers[:max_papers]:
            summary = self.summarize_with_ollama(paper)
            paper["summary"] = summary
            summarized_papers.append(paper)
            
            # Be nice to the local LLM
            time.sleep(1)
        
        return summarized_papers
    
    def run(self, max_papers=5):
        """Run the full pipeline to find and summarize papers."""
        print("Searching for papers...")
        self.search_papers()
        
        print(f"\nFound {len(self.papers)} relevant papers.")
        
        if not self.papers:
            return []
        
        print(f"\nSummarizing top {min(max_papers, len(self.papers))} papers...")
        summarized_papers = self.summarize_papers(self.papers, max_papers)
        
        return summarized_papers
    
    def display_results(self, papers):
        """Display the results in a readable format."""
        if not papers:
            print("\nNo relevant papers found.")
            return
        
        print("\n" + "="*80)
        print(f"TOP {len(papers)} RECENT AI/ML PAPERS IN FINANCE")
        print("="*80)
        
        for i, paper in enumerate(papers, 1):
            print(f"\n{i}. {paper['title']}")
            print(f"   Source: {paper['source']}")
            print(f"   Date: {paper['date']}")
            print(f"   Link: {paper['link']}")
            print("\n   Summary:")
            print(f"   {paper['summary']}")
            print("\n" + "-"*80)
    
    def save_results(self, papers, filename="finance_ai_papers.json"):
        """Save results to a JSON file."""
        if not papers:
            return
            
        with open(filename, 'w') as f:
            json.dump(papers, f, indent=2)
        
        print(f"\nResults saved to {filename}")


def check_ollama_available():
    """Check if Ollama is available and running."""
    try:
        response = requests.get('http://localhost:11434/api/tags')
        if response.status_code == 200:
            models = response.json().get('models', [])
            if models:
                print(f"Available Ollama models: {', '.join([m['name'] for m in models])}")
                return True
            else:
                print("Ollama is running but no models are available. Please pull a model first.")
                print("Example: ollama pull llama3")
                return False
        else:
            print("Ollama API responded with an error.")
            return False
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        print("Please make sure Ollama is installed and running.")
        print("Install: brew install ollama")
        print("Run: ollama serve")
        return False


def main():
    """Main function to run the Finance Research Assistant."""
    parser = argparse.ArgumentParser(description="Finance Research Assistant (Hybrid Version)")
    parser.add_argument("--model", default="llama3", help="Ollama model to use for summarization")
    parser.add_argument("--max-papers", type=int, default=5, help="Maximum number of papers to summarize")
    parser.add_argument("--output", default="finance_ai_papers.json", help="Output JSON file")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    
    # Check if Ollama is available
    if not check_ollama_available():
        return 1
    
    try:
        assistant = FinanceResearchAssistant(ollama_model=args.model)
        assistant.debug = args.debug
        papers = assistant.run(max_papers=args.max_papers)
        assistant.display_results(papers)
        assistant.save_results(papers, filename=args.output)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
