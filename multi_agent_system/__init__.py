"""
FinRA Multi-Agent System (LangGraph 0.2.x)
==========================================

A production-oriented multi-agent research system using:
- LangGraph 0.2.x for orchestration (with checkpointing & streaming)
- LLM-controlled browsing (WebGPT-style prompt pattern) via Playwright
- Long-term retrieval memory (MemGPT-inspired) backed by ChromaDB
- GPT-4o-mini for reasoning (only paid API)

All other tools are FREE - no API keys required!

Usage:
    from multi_agent_system import FinRAMultiAgent
    
    agent = FinRAMultiAgent(openai_api_key="your-key")
    
    # Standard execution
    results = await agent.research("Find latest LLM trading papers")
    
    # With streaming (LangGraph 0.2.x)
    async for event in agent.stream("Find LLM papers"):
        print(event)
    
    # With checkpointing
    results = await agent.research("query", thread_id="session-123")
"""

from .graph import FinRAMultiAgent
from .state import ResearchState, ResearchPlan

__all__ = ["FinRAMultiAgent", "ResearchState", "ResearchPlan"]
__version__ = "2.1.0"
