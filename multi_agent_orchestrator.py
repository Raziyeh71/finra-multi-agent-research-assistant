#!/usr/bin/env python3
"""
FinRA Multi-Agent Orchestrator with LangGraph
True multi-agent system with parallel execution and LLM planning
"""

import asyncio
import os
from typing import TypedDict, List, Dict, Any, Annotated
from datetime import datetime
import operator

from dotenv import load_dotenv

# Agent imports
from agents.planner_agent import PlannerAgent, ResearchPlan
from agents.paper_agent import PaperResearchAgent
from agents.web_agent import WebResearchAgent

# FinRA import
from FinRA_agent import FinanceResearchAssistant

# LangGraph imports (optional, graceful degradation)
try:
    from langgraph.graph import StateGraph, END
    HAS_LANGGRAPH = True
except ImportError:
    print("Warning: langgraph not installed. Using simple orchestration.")
    HAS_LANGGRAPH = False


# Define shared state across agents
class ResearchState(TypedDict):
    """Shared state for multi-agent research"""
    user_goal: str
    research_plan: Dict[str, Any]
    papers: Annotated[List[Dict], operator.add]  # Accumulates results
    videos: Annotated[List[Dict], operator.add]
    web_results: Annotated[List[Dict], operator.add]
    summaries: List[Dict]
    status: str


class MultiAgentOrchestrator:
    """
    Orchestrates multiple specialized agents using LangGraph.
    
    Flow:
    1. PlannerAgent creates research strategy
    2. PaperAgent, VideoAgent, WebAgent search in parallel
    3. Results aggregated and summarized
    """
    
    def __init__(
        self,
        openai_api_key: str = None,
        tavily_api_key: str = None,
        youtube_api_key: str = None
    ):
        """
        Initialize the multi-agent orchestrator.
        
        Args:
            openai_api_key: OpenAI API key
            tavily_api_key: Tavily API key for web search
            youtube_api_key: YouTube Data API key
        """
        load_dotenv()
        
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.tavily_api_key = tavily_api_key or os.getenv("TAVILY_API_KEY")
        self.youtube_api_key = youtube_api_key or os.getenv("YOUTUBE_API_KEY")
        
        # Initialize main assistant
        self.assistant = FinanceResearchAssistant(
            api_key=self.openai_api_key,
            youtube_api_key=self.youtube_api_key,
            tavily_api_key=self.tavily_api_key,
            use_async=True,
            use_planner=True
        )
        
        # Initialize agents
        self.planner = PlannerAgent(api_key=self.openai_api_key)
        self.paper_agent = PaperResearchAgent(self.assistant)
        self.web_agent = WebResearchAgent(tavily_api_key=self.tavily_api_key)
        
        if HAS_LANGGRAPH:
            self.graph = self._build_graph()
        else:
            self.graph = None
    
    def _build_graph(self):
        """Build the LangGraph orchestration graph"""
        
        # Define node functions
        async def planner_node(state: ResearchState) -> ResearchState:
            """Planner creates research strategy"""
            print("\n" + "="*60)
            print("🧠 PLANNING PHASE")
            print("="*60)
            
            plan = self.planner.create_plan(
                state["user_goal"],
                current_year=datetime.now().year
            )
            
            state["research_plan"] = plan.dict()
            state["status"] = "planned"
            return state
        
        async def paper_search_node(state: ResearchState) -> ResearchState:
            """Paper agent searches academic sources"""
            print("\n" + "="*60)
            print("📄 PAPER RESEARCH PHASE")
            print("="*60)
            
            plan = state["research_plan"]
            papers = await self.paper_agent.search_papers_parallel(
                search_terms=plan["search_terms"],
                max_age_days=plan["max_age_days"]
            )
            
            print(f"✅ Found {len(papers)} papers")
            state["papers"] = papers
            return state
        
        async def video_search_node(state: ResearchState) -> ResearchState:
            """Video search (if enabled)"""
            plan = state["research_plan"]
            
            if "youtube" in plan.get("sources", []):
                print("\n" + "="*60)
                print("🎥 VIDEO RESEARCH PHASE")
                print("="*60)
                
                videos = self.assistant.search_videos(max_per_term=5)
                print(f"✅ Found {len(videos)} videos")
                state["videos"] = videos
            
            return state
        
        async def web_search_node(state: ResearchState) -> ResearchState:
            """Web agent searches for code/blogs"""
            plan = state["research_plan"]
            
            if "web" in plan.get("sources", []):
                print("\n" + "="*60)
                print("🌐 WEB RESEARCH PHASE")
                print("="*60)
                
                web_results = await self.web_agent.search_web(
                    search_terms=plan["search_terms"],
                    max_results=5
                )
                
                state["web_results"] = web_results
            
            return state
        
        async def summarize_node(state: ResearchState) -> ResearchState:
            """Summarize all findings"""
            print("\n" + "="*60)
            print("📝 SUMMARIZATION PHASE")
            print("="*60)
            
            all_items = (
                state.get("papers", [])[:10] +
                state.get("videos", [])[:5] +
                state.get("web_results", [])[:5]
            )
            
            # Summarize in parallel
            summarized = await self.assistant.summarize_all_parallel(all_items)
            
            state["summaries"] = summarized
            state["status"] = "completed"
            
            print(f"✅ Summarized {len(summarized)} items")
            return state
        
        # Build graph
        workflow = StateGraph(ResearchState)
        
        # Add nodes
        workflow.add_node("planner", planner_node)
        workflow.add_node("paper_search", paper_search_node)
        workflow.add_node("video_search", video_search_node)
        workflow.add_node("web_search", web_search_node)
        workflow.add_node("summarize", summarize_node)
        
        # Define flow
        workflow.set_entry_point("planner")
        
        # Planner → parallel search agents
        workflow.add_edge("planner", "paper_search")
        workflow.add_edge("planner", "video_search")
        workflow.add_edge("planner", "web_search")
        
        # All searches → summarizer
        workflow.add_edge("paper_search", "summarize")
        workflow.add_edge("video_search", "summarize")
        workflow.add_edge("web_search", "summarize")
        
        # Summarizer → end
        workflow.add_edge("summarize", END)
        
        return workflow.compile()
    
    async def run(self, user_goal: str) -> Dict[str, Any]:
        """
        Run the multi-agent research system.
        
        Args:
            user_goal: User's research objective
            
        Returns:
            Dictionary with papers, videos, web results, and summaries
        """
        if HAS_LANGGRAPH and self.graph:
            return await self._run_with_langgraph(user_goal)
        else:
            return await self._run_simple(user_goal)
    
    async def _run_with_langgraph(self, user_goal: str) -> Dict[str, Any]:
        """Run using LangGraph orchestration"""
        print("\n🚀 Starting Multi-Agent Research System (LangGraph)")
        print(f"📋 Goal: {user_goal}\n")
        
        initial_state = {
            "user_goal": user_goal,
            "research_plan": {},
            "papers": [],
            "videos": [],
            "web_results": [],
            "summaries": [],
            "status": "started"
        }
        
        # Run the graph
        result = await self.graph.ainvoke(initial_state)
        
        return {
            "plan": result.get("research_plan", {}),
            "papers": result.get("papers", []),
            "videos": result.get("videos", []),
            "web_results": result.get("web_results", []),
            "summaries": result.get("summaries", []),
            "status": result.get("status", "unknown")
        }
    
    async def _run_simple(self, user_goal: str) -> Dict[str, Any]:
        """Fallback: simple sequential execution"""
        print("\n🚀 Starting Multi-Agent Research System (Simple Mode)")
        print(f"📋 Goal: {user_goal}\n")
        
        # Step 1: Plan
        print("🧠 Planning...")
        plan = self.planner.create_plan(user_goal, datetime.now().year)
        
        # Step 2: Search (parallel)
        print("🔍 Searching...")
        papers_task = self.paper_agent.search_papers_parallel(
            plan.search_terms,
            plan.max_age_days
        )
        
        web_task = self.web_agent.search_web(plan.search_terms, max_results=5)
        
        papers, web_results = await asyncio.gather(papers_task, web_task)
        
        # Step 3: Summarize
        print("📝 Summarizing...")
        all_items = papers[:10] + web_results[:5]
        summaries = await self.assistant.summarize_all_parallel(all_items)
        
        return {
            "plan": plan.dict(),
            "papers": papers,
            "videos": [],
            "web_results": web_results,
            "summaries": summaries,
            "status": "completed"
        }
    
    def display_results(self, results: Dict[str, Any]):
        """Pretty print results"""
        print("\n" + "="*80)
        print("📊 FINAL RESULTS")
        print("="*80)
        
        plan = results.get("plan", {})
        print(f"\n✅ Research Plan:")
        print(f"   Terms: {plan.get('search_terms', [])}")
        print(f"   Sources: {plan.get('sources', [])}")
        print(f"   Timeframe: Last {plan.get('max_age_days', 'N/A')} days")
        
        papers = results.get("papers", [])
        videos = results.get("videos", [])
        web = results.get("web_results", [])
        summaries = results.get("summaries", [])
        
        print(f"\n📈 Content Found:")
        print(f"   Papers: {len(papers)}")
        print(f"   Videos: {len(videos)}")
        print(f"   Web Results: {len(web)}")
        print(f"   Summaries: {len(summaries)}")
        
        # Display top summaries
        print(f"\n📝 Top Summaries:")
        for i, item in enumerate(summaries[:5], 1):
            print(f"\n{i}. {item.get('title', 'N/A')}")
            print(f"   Source: {item.get('source', 'N/A')}")
            print(f"   Link: {item.get('link', 'N/A')}")
            if 'summary' in item:
                print(f"   Summary: {item['summary'][:200]}...")


async def main():
    """Example usage"""
    orchestrator = MultiAgentOrchestrator()
    
    user_goal = """
    Find the latest research and tutorials on using LLMs for algorithmic trading,
    focusing on practical implementations with code from the last 60 days.
    """
    
    results = await orchestrator.run(user_goal)
    orchestrator.display_results(results)


if __name__ == "__main__":
    asyncio.run(main())
