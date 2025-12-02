"""
LangGraph 0.2.x Multi-Agent Orchestration
==========================================

Uses the NEW LangGraph 0.2.x API with:
- Better streaming support
- Checkpointing for state persistence
- Improved async execution
- GPT-4o-mini for reasoning

Main graph orchestrates:
1. Memory Agent (recall past research)
2. Planner Agent (create strategy)
3. Research Agents (parallel: papers, videos, web)
4. Summarizer Agent (GPT-4o-mini summaries)
5. Reporter Agent (final report)

Only requires OpenAI API key - all other tools are FREE!
"""

import asyncio
from typing import Dict, Any, Optional, AsyncIterator, Annotated, Sequence
from datetime import datetime
import operator

# LangGraph 0.2.x imports
try:
    from langgraph.graph import StateGraph, END, START
    from langgraph.checkpoint.memory import MemorySaver
    HAS_LANGGRAPH = True
except ImportError:
    try:
        # Fallback for older versions
        from langgraph.graph import StateGraph, END
        START = None
        MemorySaver = None
        HAS_LANGGRAPH = True
    except ImportError:
        HAS_LANGGRAPH = False
        START = None
        MemorySaver = None
        print("⚠️ LangGraph not installed. Run: pip install langgraph>=0.2.0")

from .state import ResearchState, create_initial_state
from .config import FinRAConfig, get_config
from .agents import (
    planner_node,
    paper_researcher_node,
    video_researcher_node,
    web_researcher_node,
    evaluator_node,
    summarizer_node,
    reporter_node,
    memory_node,
)


class FinRAMultiAgent:
    """
    FinRA Multi-Agent Research System (LangGraph 0.2.x).
    
    Uses LangGraph 0.2.x for orchestration with:
    - LLM-controlled browsing (WebGPT-style) via Playwright
    - Long-term retrieval memory (MemGPT-inspired) via ChromaDB
    - GPT-4o-mini for reasoning (only paid API)
    - Checkpointing for state persistence
    - Streaming support for real-time updates
    
    All other tools are FREE - no API keys required!
    
    Usage:
        agent = FinRAMultiAgent(openai_api_key="your-key")
        results = await agent.research("Find latest LLM trading papers")
        
        # With streaming:
        async for event in agent.stream("Find LLM papers"):
            print(event)
    """
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        memory_path: str = "./finra_memory",
        max_results_per_source: int = 10,
        max_papers_to_summarize: int = 10,
        max_videos_to_summarize: int = 5,
        include_videos: bool = False,
        include_web: bool = True,
        debug: bool = False,
        enable_checkpointing: bool = True,
    ):
        """
        Initialize the multi-agent system.
        
        Args:
            openai_api_key: OpenAI API key (ONLY required API key!)
            memory_path: Path for ChromaDB memory storage
            max_results_per_source: Max results per search source
            max_papers_to_summarize: Max papers to summarize
            max_videos_to_summarize: Max videos to summarize
            include_videos: Whether to search YouTube
            include_web: Whether to search web/Google
            debug: Enable debug logging
            enable_checkpointing: Enable state checkpointing (LangGraph 0.2.x)
        """
        # Get config with API key
        self.config = get_config(openai_api_key=openai_api_key)
        
        # Store settings
        self.settings = {
            "openai_api_key": self.config.openai_api_key,
            "memory_path": memory_path,
            "max_results_per_source": max_results_per_source,
            "max_papers_to_summarize": max_papers_to_summarize,
            "max_videos_to_summarize": max_videos_to_summarize,
            "include_videos": include_videos,
            "include_web": include_web,
            "debug": debug,
        }
        
        # Setup checkpointing (LangGraph 0.2.x feature)
        self.checkpointer = None
        if enable_checkpointing and MemorySaver is not None:
            self.checkpointer = MemorySaver()
        
        # Build the graph
        if HAS_LANGGRAPH:
            self.graph = self._build_graph()
        else:
            self.graph = None
            print("⚠️ Running in fallback mode (no LangGraph)")
    
    def _build_graph(self):
        """Build the LangGraph 0.2.x state machine with checkpointing."""
        
        # Create graph with state schema
        workflow = StateGraph(ResearchState)
        
        # Wrap nodes to pass config
        async def _memory(state):
            return await memory_node(state, self.settings)
        
        async def _planner(state):
            return await planner_node(state, self.settings)
        
        async def _paper_researcher(state):
            return await paper_researcher_node(state, self.settings)
        
        async def _video_researcher(state):
            return await video_researcher_node(state, self.settings)
        
        async def _web_researcher(state):
            return await web_researcher_node(state, self.settings)
        
        async def _evaluator(state):
            return await evaluator_node(state, self.settings)
        
        async def _summarizer(state):
            return await summarizer_node(state, self.settings)
        
        async def _reporter(state):
            return await reporter_node(state, self.settings)
        
        # Add nodes
        workflow.add_node("memory", _memory)
        workflow.add_node("planner", _planner)
        workflow.add_node("paper_researcher", _paper_researcher)
        workflow.add_node("video_researcher", _video_researcher)
        workflow.add_node("web_researcher", _web_researcher)
        workflow.add_node("evaluator", _evaluator)
        workflow.add_node("summarizer", _summarizer)
        workflow.add_node("reporter", _reporter)
        
        # Define edges using LangGraph 0.2.x API
        if START is not None:
            # LangGraph 0.2.x: Use START constant
            workflow.add_edge(START, "memory")
        else:
            # Fallback for older versions
            workflow.set_entry_point("memory")
        
        # Memory → Planner
        workflow.add_edge("memory", "planner")
        
        # Planner → Paper researcher
        workflow.add_edge("planner", "paper_researcher")
        
        # Conditional routing based on plan
        def should_search_videos(state):
            plan = state.get("plan", {})
            if "youtube" in plan.get("sources", []):
                return "video_researcher"
            return "web_researcher"
        
        workflow.add_conditional_edges(
            "paper_researcher",
            should_search_videos,
            {
                "video_researcher": "video_researcher",
                "web_researcher": "web_researcher",
            }
        )
        
        # Video → Web or Evaluator
        def should_search_web(state):
            plan = state.get("plan", {})
            if "web" in plan.get("sources", []) or "google" in plan.get("sources", []):
                return "web_researcher"
            return "evaluator"
        
        workflow.add_conditional_edges(
            "video_researcher",
            should_search_web,
            {
                "web_researcher": "web_researcher",
                "evaluator": "evaluator",
            }
        )
        
        # Web → Evaluator (NEW: Evaluator assesses before Summarizer)
        workflow.add_edge("web_researcher", "evaluator")
        
        # Evaluator → Summarizer (credibility assessment before summarization)
        workflow.add_edge("evaluator", "summarizer")
        
        # Summarizer → Reporter
        workflow.add_edge("summarizer", "reporter")
        
        # Reporter → END
        workflow.add_edge("reporter", END)
        
        # Compile with checkpointing if available (LangGraph 0.2.x)
        if self.checkpointer is not None:
            return workflow.compile(checkpointer=self.checkpointer)
        else:
            return workflow.compile()
    
    async def research(
        self,
        goal: str,
        max_age_days: Optional[int] = None,
        thread_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run research for a given goal.
        
        Args:
            goal: Research objective (e.g., "Find latest LLM trading papers")
            max_age_days: Override recency filter
            thread_id: Thread ID for checkpointing (LangGraph 0.2.x)
            
        Returns:
            Dictionary with papers, videos, summaries, and report
        """
        print("\n" + "="*70)
        print("🚀 FinRA MULTI-AGENT RESEARCH SYSTEM (LangGraph 0.2.x)")
        print("="*70)
        print(f"📋 Goal: {goal}")
        print(f"⚡ Mode: {'LangGraph 0.2.x' if self.graph else 'Fallback'}")
        if self.checkpointer:
            print(f"💾 Checkpointing: Enabled")
        print("="*70)
        
        # Create initial state
        state = create_initial_state(goal)
        
        if self.graph:
            # Run with LangGraph 0.2.x
            config = {}
            if thread_id and self.checkpointer:
                config["configurable"] = {"thread_id": thread_id}
            
            result = await self.graph.ainvoke(state, config=config if config else None)
        else:
            # Fallback: run agents sequentially
            result = await self._run_fallback(state)
        
        # Print summary
        self._print_summary(result)
        
        return {
            "goal": goal,
            "plan": result.get("plan"),
            "papers": result.get("papers", []),
            "videos": result.get("videos", []),
            "web_results": result.get("web_results", []),
            "summaries": result.get("summaries", []),
            "report": result.get("final_report", ""),
            "logs": result.get("logs", []),
            "errors": result.get("errors", []),
        }
    
    async def stream(
        self,
        goal: str,
        thread_id: Optional[str] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream research results in real-time (LangGraph 0.2.x feature).
        
        Yields events as each agent completes its work.
        
        Args:
            goal: Research objective
            thread_id: Thread ID for checkpointing
            
        Yields:
            Dictionary with event type and data
        """
        state = create_initial_state(goal)
        
        if not self.graph:
            # Fallback without streaming
            result = await self._run_fallback(state)
            yield {"event": "complete", "data": result}
            return
        
        config = {}
        if thread_id and self.checkpointer:
            config["configurable"] = {"thread_id": thread_id}
        
        # LangGraph 0.2.x streaming
        try:
            async for event in self.graph.astream(state, config=config if config else None):
                # Extract node name and output
                for node_name, node_output in event.items():
                    yield {
                        "event": "node_complete",
                        "node": node_name,
                        "data": node_output,
                    }
        except Exception as e:
            yield {"event": "error", "message": str(e)}
    
    async def stream_events(
        self,
        goal: str,
        thread_id: Optional[str] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream detailed events (LangGraph 0.2.x astream_events).
        
        Provides more granular events including:
        - on_chain_start
        - on_chain_end
        - on_tool_start
        - on_tool_end
        
        Args:
            goal: Research objective
            thread_id: Thread ID for checkpointing
            
        Yields:
            Detailed event dictionaries
        """
        state = create_initial_state(goal)
        
        if not self.graph:
            yield {"event": "error", "message": "LangGraph not available"}
            return
        
        config = {}
        if thread_id and self.checkpointer:
            config["configurable"] = {"thread_id": thread_id}
        
        # LangGraph 0.2.x detailed event streaming
        try:
            async for event in self.graph.astream_events(
                state, 
                config=config if config else None,
                version="v2"
            ):
                yield event
        except AttributeError:
            # Fallback if astream_events not available
            async for event in self.stream(goal, thread_id):
                yield event
        except Exception as e:
            yield {"event": "error", "message": str(e)}
    
    async def _run_fallback(self, state: ResearchState) -> Dict[str, Any]:
        """Fallback execution without LangGraph."""
        
        # Memory
        state.update(await memory_node(state, self.settings))
        
        # Planner
        state.update(await planner_node(state, self.settings))
        
        # Retrievers (parallel)
        paper_task = paper_researcher_node(state, self.settings)
        video_task = video_researcher_node(state, self.settings)
        web_task = web_researcher_node(state, self.settings)
        
        results = await asyncio.gather(paper_task, video_task, web_task)
        
        for result in results:
            state.update(result)
        
        # Evaluator (NEW: assess credibility before summarization)
        state.update(await evaluator_node(state, self.settings))
        
        # Summarizer
        state.update(await summarizer_node(state, self.settings))
        
        # Reporter
        state.update(await reporter_node(state, self.settings))
        
        return state
    
    def _print_summary(self, result: Dict[str, Any]):
        """Print a summary of results."""
        print("\n" + "="*70)
        print("📊 RESEARCH COMPLETE")
        print("="*70)
        
        papers = result.get("papers", [])
        videos = result.get("videos", [])
        summaries = result.get("summaries", [])
        
        print(f"\n📄 Papers found: {len(papers)}")
        print(f"🎥 Videos found: {len(videos)}")
        print(f"📝 Summaries generated: {len(summaries)}")
        
        # Print top papers
        if summaries:
            print("\n🏆 Top Results:")
            for i, item in enumerate(summaries[:5], 1):
                print(f"\n{i}. {item.get('title', 'Untitled')[:60]}...")
                print(f"   Source: {item.get('source', 'Unknown')}")
                if item.get("summary"):
                    print(f"   Summary: {item['summary'][:100]}...")
        
        # Print report
        report = result.get("final_report", "")
        if report:
            print("\n" + "="*70)
            print("📋 EXECUTIVE SUMMARY")
            print("="*70)
            print(report)
        
        print("\n" + "="*70)
    
    def display_results(self, results: Dict[str, Any]):
        """Display results in a formatted way."""
        self._print_summary(results)
    
    def save_results(
        self,
        results: Dict[str, Any],
        filename: str = "finra_research_results.json",
    ):
        """Save results to JSON file."""
        import json
        
        # Make results JSON-serializable
        output = {
            "goal": results.get("goal", ""),
            "plan": results.get("plan", {}),
            "papers": results.get("papers", []),
            "videos": results.get("videos", []),
            "summaries": results.get("summaries", []),
            "report": results.get("report", ""),
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        with open(filename, "w") as f:
            json.dump(output, f, indent=2, default=str)
        
        print(f"💾 Results saved to {filename}")


async def main():
    """Example usage."""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Create agent (only needs OpenAI key!)
    agent = FinRAMultiAgent(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        include_videos=False,  # Set to True to include YouTube
        include_web=True,
    )
    
    # Run research
    results = await agent.research(
        "Find the latest papers on using LLMs and transformers for algorithmic trading"
    )
    
    # Save results
    agent.save_results(results)


if __name__ == "__main__":
    asyncio.run(main())
