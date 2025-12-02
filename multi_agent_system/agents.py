"""
LangGraph Agent Nodes
=====================

Each function is a node in the LangGraph state machine.
Agents use LLM-controlled browsing (WebGPT-style prompt pattern).

Agent Flow:
1. Memory Agent (ChromaDB) — Queried first; recalls similar past queries and results
2. Planner Agent (GPT-4o-mini) — Uses user query + memory context to decide sources
3. Paper/Video Retrievers (Playwright) — Fetch new results in parallel; merge with recalled
4. Evaluator Agent (GPT-4o-mini) — Ranks by credibility (citations, views, recency)
5. Summarizer Agent (GPT-4o-mini) — Summarizes and ranks; updates memory for future runs
6. Reporter Agent (GPT-4o-mini) — Creates final executive summary

All LLM agents use gpt-4o-mini by default (configurable via LLM_MODEL in .env).

Production Features:
- Retry with exponential backoff
- Configurable timeouts
- Rate limiting
- Structured logging
"""

import os
import asyncio
import time
from typing import Dict, Any, List
from datetime import datetime

from .state import ResearchState, ResearchPlan
from .config import logger, MAX_RETRIES, REQUEST_TIMEOUT, RATE_LIMIT_DELAY
from .production import (
    log_agent_start,
    log_agent_complete,
    log_agent_error,
    safe_request,
    rate_limit,
    metrics_collector,
)

# Default model (configurable via LLM_MODEL env var)
DEFAULT_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")


async def planner_node(
    state: ResearchState,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Planner Agent (GPT-4o-mini): Creates research strategy from user goal.
    
    Uses GPT-4o-mini by default (configurable via LLM_MODEL env var).
    
    Generates:
    - Search terms
    - Sources to search
    - Recency filter
    - Focus areas
    """
    from tools.summarizer_tools import generate_research_plan
    
    start_time = time.time()
    log_agent_start("PLANNER", goal=state["user_goal"][:100])
    
    api_key = config.get("openai_api_key")
    user_goal = state["user_goal"]
    
    print("\n" + "="*60)
    print(f"🧠 PLANNER AGENT ({DEFAULT_MODEL})")
    print("="*60)
    print(f"Goal: {user_goal}")
    
    # Check for relevant past research and build memory context
    recalled = state.get("recalled_memories", [])
    memory_context = ""
    if recalled:
        print(f"📚 Found {len(recalled)} relevant past searches")
        logger.info(f"Recalled {len(recalled)} memories for context")
        # Format memory context for planner
        memory_context = "\n".join([
            f"- Query: '{m.get('query', 'Unknown')}' → Found {m.get('result_count', '?')} results"
            for m in recalled[:5]
        ])
    
    # Generate plan with memory context (prompt engineering: context-aware planning)
    try:
        plan = await generate_research_plan(user_goal, api_key, memory_context=memory_context)
    except Exception as e:
        log_agent_error("PLANNER", e)
        # Fallback plan
        plan = {
            "search_terms": ["AI trading", "machine learning finance"],
            "sources": ["arxiv", "scholar"],
            "max_age_days": 90,
            "focus_areas": [],
            "reasoning": f"Fallback plan due to error: {str(e)[:50]}",
        }
    
    print(f"\n📋 Research Plan:")
    print(f"   Search Terms: {plan['search_terms']}")
    print(f"   Sources: {plan['sources']}")
    print(f"   Max Age: {plan['max_age_days']} days")
    print(f"   Reasoning: {plan['reasoning']}")
    
    duration_ms = (time.time() - start_time) * 1000
    log_agent_complete("PLANNER", duration_ms, terms=len(plan['search_terms']))
    
    return {
        "plan": plan,
        "current_step": 1,
        "total_steps": len(plan["sources"]) + 2,  # sources + summarize + report
        "next_agent": "researcher",
        "logs": [f"Plan created: {len(plan['search_terms'])} search terms, {len(plan['sources'])} sources"],
    }


async def paper_researcher_node(
    state: ResearchState,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Paper Retriever Agent: Searches academic sources with production safeguards.
    
    Uses FREE tools:
    - arXiv API (no key needed)
    - Google Scholar (Playwright, rate-limited)
    - Nature journals (Playwright, rate-limited)
    - IEEE Xplore (Playwright, rate-limited)
    
    Production: Rate limiting, retries, timeouts, structured logging.
    """
    from tools.search_tools import search_arxiv, search_google_scholar
    from tools.scraper_tools import scrape_nature, scrape_ieee
    
    start_time = time.time()
    log_agent_start("PAPER_RETRIEVER")
    
    plan = state.get("plan", {})
    search_terms = plan.get("search_terms", ["AI trading"])
    sources = plan.get("sources", ["arxiv"])
    max_age_days = plan.get("max_age_days", 90)
    max_results = config.get("max_results_per_source", 10)
    
    print("\n" + "="*60)
    print("📄 PAPER RETRIEVER AGENT")
    print("="*60)
    
    all_papers = []
    tasks = []
    task_names = []
    
    # Create search tasks for each source and term
    for term in search_terms[:3]:  # Limit to 3 terms
        if "arxiv" in sources:
            tasks.append(search_arxiv(term, max_results, max_age_days))
            task_names.append(f"arXiv: {term}")
        
        if "scholar" in sources:
            tasks.append(search_google_scholar(term, max_results, max_age_days))
            task_names.append(f"Scholar: {term}")
        
        if "nature" in sources:
            tasks.append(scrape_nature(term, "natmachintell", max_results, max_age_days))
            task_names.append(f"Nature: {term}")
        
        if "ieee" in sources:
            tasks.append(scrape_ieee(term, max_results, max_age_days))
            task_names.append(f"IEEE: {term}")
    
    print(f"🔍 Running {len(tasks)} searches in parallel...")
    
    # Run all searches in parallel
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Collect results
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"❌ {task_names[i]} failed: {result}")
        elif isinstance(result, list):
            all_papers.extend(result)
    
    # Deduplicate by title
    seen_titles = set()
    unique_papers = []
    for paper in all_papers:
        title_key = paper.get("title", "").lower().strip()
        if title_key and title_key not in seen_titles:
            seen_titles.add(title_key)
            unique_papers.append(paper)
    
    # Filter by relevance
    from tools.summarizer_tools import is_finance_ai_related
    relevant_papers = [
        p for p in unique_papers
        if is_finance_ai_related(p.get("title", ""), p.get("abstract", ""))
    ]
    
    print(f"\n✅ Found {len(all_papers)} total → {len(unique_papers)} unique → {len(relevant_papers)} relevant")
    
    duration_ms = (time.time() - start_time) * 1000
    log_agent_complete("PAPER_RETRIEVER", duration_ms, papers=len(relevant_papers))
    
    return {
        "papers": relevant_papers,
        "current_step": state["current_step"] + 1,
        "logs": [f"Paper search: {len(relevant_papers)} relevant papers found"],
    }


async def video_researcher_node(
    state: ResearchState,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Video Retriever Agent: Searches YouTube for tutorials.
    
    Uses FREE Playwright scraping (no API key needed).
    Production: Rate limiting for YouTube requests.
    """
    from tools.scraper_tools import scrape_youtube
    
    plan = state.get("plan", {})
    search_terms = plan.get("search_terms", ["AI trading tutorial"])
    
    # Only search videos if requested
    if "youtube" not in plan.get("sources", []):
        return {"videos": [], "logs": ["Video search skipped (not in plan)"]}
    
    print("\n" + "="*60)
    print("🎥 VIDEO RESEARCH AGENT")
    print("="*60)
    
    all_videos = []
    
    for term in search_terms[:2]:  # Limit to 2 terms for videos
        videos = await scrape_youtube(f"{term} tutorial", max_results=5)
        all_videos.extend(videos)
    
    # Deduplicate
    seen_links = set()
    unique_videos = []
    for video in all_videos:
        link = video.get("link", "")
        if link and link not in seen_links:
            seen_links.add(link)
            unique_videos.append(video)
    
    print(f"✅ Found {len(unique_videos)} unique videos")
    
    return {
        "videos": unique_videos,
        "logs": [f"Video search: {len(unique_videos)} videos found"],
    }


async def web_researcher_node(
    state: ResearchState,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Web Research Agent: LLM-controlled browsing (WebGPT-style).
    
    Searches Google for blogs, GitHub repos, tutorials.
    Uses FREE Playwright scraping.
    """
    from tools.search_tools import search_google
    from tools.scraper_tools import read_webpage
    
    plan = state.get("plan", {})
    search_terms = plan.get("search_terms", [])
    
    # Only search web if requested
    if "web" not in plan.get("sources", []) and "google" not in plan.get("sources", []):
        return {"web_results": [], "logs": ["Web search skipped (not in plan)"]}
    
    print("\n" + "="*60)
    print("🌐 WEB RESEARCH AGENT (LLM-controlled browsing)")
    print("="*60)
    
    all_results = []
    
    for term in search_terms[:2]:
        # Search for code/implementations
        query = f"{term} python code github implementation"
        results = await search_google(query, max_results=5)
        all_results.extend(results)
    
    # Optionally read top results for more context
    # (WebGPT-style pattern: follow links and extract content)
    enriched_results = []
    for result in all_results[:5]:
        link = result.get("link", "")
        if link and "github.com" in link:
            # Read GitHub page for more details
            page_content = await read_webpage(link, extract_text=True)
            if not page_content.get("error"):
                result["full_text"] = page_content.get("text", "")[:1000]
        enriched_results.append(result)
    
    print(f"✅ Found {len(enriched_results)} web results")
    
    return {
        "web_results": enriched_results,
        "logs": [f"Web search: {len(enriched_results)} results found"],
    }


async def summarizer_node(
    state: ResearchState,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Summarizer Agent (Step 5): Summarizes and ranks results.
    
    Responsibilities:
    - Generates trader-focused summaries using GPT-4o-mini
    - Ranks results by relevance and quality
    - **Updates memory for future runs** (key for learning)
    
    Uses GPT-4o-mini by default (configurable via LLM_MODEL env var).
    """
    from tools.summarizer_tools import summarize_batch
    
    start_time = time.time()
    log_agent_start("SUMMARIZER")
    
    api_key = config.get("openai_api_key")
    max_papers = config.get("max_papers_to_summarize", 10)
    max_videos = config.get("max_videos_to_summarize", 5)
    
    print("\n" + "="*60)
    print(f"📝 SUMMARIZER AGENT ({DEFAULT_MODEL})")
    print("="*60)
    
    papers = state.get("papers", [])[:max_papers]
    videos = state.get("videos", [])[:max_videos]
    
    all_items = papers + videos
    
    if not all_items:
        print("⚠️ No items to summarize")
        return {"summaries": [], "logs": ["No items to summarize"]}
    
    print(f"Summarizing {len(papers)} papers and {len(videos)} videos...")
    
    # Summarize in parallel
    summarized = await summarize_batch(all_items, api_key, max_concurrent=5)
    
    print(f"✅ Generated {len(summarized)} summaries")
    
    # Update memory with results for future recall
    try:
        from memory import MemoryManager
        memory_path = config.get("memory_path", "./finra_memory")
        memory = MemoryManager(persist_directory=memory_path)
        user_goal = state.get("user_goal", "")
        memory.archive_research(user_goal, summarized)
        print(f"💾 Updated memory with {len(summarized)} results for future recall")
    except Exception as e:
        logger.warning(f"Memory update failed: {e}")
    
    duration_ms = (time.time() - start_time) * 1000
    log_agent_complete("SUMMARIZER", duration_ms, summaries=len(summarized))
    
    return {
        "summaries": summarized,
        "current_step": state["current_step"] + 1,
        "logs": [f"Summarization: {len(summarized)} items summarized"],
    }


async def evaluator_node(
    state: ResearchState,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Evaluator Agent (GPT-4o-mini): Assesses source credibility and relevance.
    
    Responsibilities:
    - Ranks papers by citations + relevance + recency
    - Ranks videos by views + recency
    - Filters low-quality or irrelevant sources
    - Ensures only credible sources reach the Summarizer
    
    Uses GPT-4o-mini by default (configurable via LLM_MODEL env var).
    """
    from tools.ranking import rank_papers, rank_videos
    from tools.summarizer_tools import is_finance_ai_related
    import openai
    
    api_key = config.get("openai_api_key")
    max_papers = config.get("max_papers_to_summarize", 10)
    max_videos = config.get("max_videos_to_summarize", 10)
    
    print("\n" + "="*60)
    print(f"🔍 EVALUATOR AGENT ({DEFAULT_MODEL})")
    print("="*60)
    
    papers = state.get("papers", [])
    videos = state.get("videos", [])
    
    print(f"📄 Evaluating {len(papers)} papers and {len(videos)} videos...")
    
    # Step 1: Filter by relevance (AI/ML + Finance)
    relevant_papers = [
        p for p in papers
        if is_finance_ai_related(p.get("title", ""), p.get("abstract", ""))
    ]
    print(f"   → {len(relevant_papers)} papers passed relevance filter")
    
    # Step 2: Rank papers by credibility (citations + recency + relevance)
    ranked_papers = rank_papers(relevant_papers, top_n=max_papers)
    print(f"   → Top {len(ranked_papers)} papers by credibility score")
    
    # Step 3: Rank videos by engagement (views + recency)
    ranked_videos = rank_videos(videos, top_n=max_videos)
    print(f"   → Top {len(ranked_videos)} videos by engagement score")
    
    # Step 4: Optional LLM-based quality assessment for top results
    if api_key and len(ranked_papers) > 0:
        # Use LLM to assess quality of top 3 papers
        openai.api_key = api_key
        try:
            top_titles = [p.get("title", "") for p in ranked_papers[:3]]
            prompt = f"""Rate the academic credibility of these paper titles for algorithmic trading research (1-10):
            
{chr(10).join(f'{i+1}. {t}' for i, t in enumerate(top_titles))}

Output only numbers separated by commas, e.g., "8,7,9" """
            
            response = await asyncio.to_thread(
                openai.ChatCompletion.create,
                model=DEFAULT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20,
                temperature=0.1,
            )
            
            scores_str = response.choices[0].message.content.strip()
            # Parse scores and boost rankings
            try:
                scores = [int(s.strip()) for s in scores_str.split(",")]
                for i, score in enumerate(scores[:3]):
                    if i < len(ranked_papers):
                        ranked_papers[i]["llm_quality_score"] = score
                print(f"   → LLM quality scores: {scores_str}")
            except:
                pass
                
        except Exception as e:
            print(f"   ⚠️ LLM quality check skipped: {e}")
    
    print(f"✅ Evaluation complete: {len(ranked_papers)} papers, {len(ranked_videos)} videos")
    
    return {
        "evaluated_papers": ranked_papers,
        "evaluated_videos": ranked_videos,
        "papers": ranked_papers,  # Update state for downstream agents
        "videos": ranked_videos,
        "logs": [f"Evaluation: {len(ranked_papers)} papers, {len(ranked_videos)} videos passed"],
    }


async def reporter_node(
    state: ResearchState,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Reporter Agent (GPT-4o-mini): Generates final research report.
    
    Uses GPT-4o-mini by default (configurable via LLM_MODEL env var).
    Combines all findings into a structured report.
    """
    import openai
    
    api_key = config.get("openai_api_key")
    openai.api_key = api_key
    
    print("\n" + "="*60)
    print(f"📊 REPORTER AGENT ({DEFAULT_MODEL})")
    print("="*60)
    
    summaries = state.get("summaries", [])
    user_goal = state.get("user_goal", "")
    
    if not summaries:
        return {
            "final_report": "No results found for your research query.",
            "should_continue": False,
            "end_time": datetime.utcnow().isoformat(),
        }
    
    # Build summary list for report
    summary_text = ""
    for i, item in enumerate(summaries[:10], 1):
        summary_text += f"\n{i}. {item.get('title', 'Untitled')}\n"
        summary_text += f"   Source: {item.get('source', 'Unknown')}\n"
        summary_text += f"   Summary: {item.get('summary', 'No summary')}\n"
    
    # Generate final report
    prompt = f"""Create a brief research report based on these findings.

User Goal: {user_goal}

Findings:
{summary_text}

Write a 2-3 paragraph executive summary highlighting:
1. Key themes and trends found
2. Most promising papers/resources
3. Recommended next steps for the researcher

Be concise and actionable."""

    try:
        response = await asyncio.to_thread(
            openai.ChatCompletion.create,
            model=DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": "You are a research analyst creating executive summaries."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.3,
        )
        
        report = response.choices[0].message.content.strip()
        
    except Exception as e:
        report = f"Report generation error: {e}"
    
    print("✅ Final report generated")
    
    return {
        "final_report": report,
        "should_continue": False,
        "end_time": datetime.utcnow().isoformat(),
        "logs": ["Final report generated"],
    }


async def memory_node(
    state: ResearchState,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Memory Agent (Step 1): Queried FIRST in the pipeline.
    
    Long-term retrieval memory (MemGPT-inspired):
    - Recalls similar past queries and results from ChromaDB
    - Passes context to Planner for informed decision-making
    - Archives new findings after research completes
    
    This enables the system to learn from past searches and avoid redundant work.
    """
    from memory import MemoryManager
    
    start_time = time.time()
    log_agent_start("MEMORY")
    
    memory_path = config.get("memory_path", "./finra_memory")
    memory = MemoryManager(persist_directory=memory_path)
    
    user_goal = state.get("user_goal", "")
    
    print("\n" + "="*60)
    print("🧠 MEMORY AGENT (ChromaDB)")
    print("="*60)
    print(f"Query: {user_goal[:80]}...")
    
    # Recall relevant past research
    recalled = memory.recall(user_goal, max_results=5)
    
    if recalled:
        print(f"📚 Found {len(recalled)} similar past searches")
        for i, item in enumerate(recalled[:3], 1):
            print(f"   {i}. {item.get('query', 'Unknown')[:50]}...")
    else:
        print("📭 No relevant past research found")
    
    # Add current search to working memory
    memory.add_to_working_memory({
        "query": user_goal,
        "timestamp": datetime.utcnow().isoformat(),
    })
    
    # Archive results if we have them (called again at end of pipeline)
    papers = state.get("papers", [])
    if papers:
        memory.archive_research(user_goal, papers)
        print(f"💾 Archived {len(papers)} papers for future recall")
    
    duration_ms = (time.time() - start_time) * 1000
    log_agent_complete("MEMORY", duration_ms, recalled=len(recalled))
    
    return {
        "recalled_memories": recalled,
        "logs": [f"Memory: recalled {len(recalled)} relevant items"],
    }
