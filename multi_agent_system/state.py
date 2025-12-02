"""
State definitions for the multi-agent system.
Uses TypedDict for type safety and LangGraph compatibility.
"""

from typing import TypedDict, List, Dict, Any, Optional, Annotated
from datetime import datetime
import operator
from pydantic import BaseModel, Field


class ResearchPlan(BaseModel):
    """Structured research plan created by the planner agent."""
    
    search_terms: List[str] = Field(
        default_factory=list,
        description="Optimized search terms for papers/videos"
    )
    sources: List[str] = Field(
        default_factory=lambda: ["arxiv", "scholar", "nature"],
        description="Sources to search: arxiv, scholar, nature, ieee, youtube"
    )
    max_results_per_source: int = Field(
        default=10,
        description="Maximum results to fetch per source"
    )
    max_age_days: int = Field(
        default=90,
        description="Only include content from last N days"
    )
    focus_areas: List[str] = Field(
        default_factory=list,
        description="Specific focus areas (e.g., 'transformers', 'reinforcement learning')"
    )
    reasoning: str = Field(
        default="",
        description="Explanation of the research strategy"
    )


class Paper(TypedDict):
    """Structure for a research paper."""
    title: str
    link: str
    abstract: str
    date: str
    source: str
    authors: Optional[List[str]]
    relevance_score: Optional[float]
    summary: Optional[str]


class Video(TypedDict):
    """Structure for a video result."""
    title: str
    link: str
    channel: str
    date: str
    description: str
    summary: Optional[str]


class WebResult(TypedDict):
    """Structure for a web search result."""
    title: str
    link: str
    snippet: str
    source: str


class MemoryEntry(TypedDict):
    """Structure for a memory entry."""
    id: str
    content: str
    metadata: Dict[str, Any]
    timestamp: str
    relevance_score: Optional[float]


class ResearchState(TypedDict):
    """
    Shared state across all agents in the LangGraph.
    
    Uses Annotated with operator.add for lists that accumulate results
    from multiple agents running in parallel.
    """
    
    # === Input ===
    user_goal: str
    
    # === Planning ===
    plan: Optional[Dict[str, Any]]  # ResearchPlan as dict
    current_step: int
    total_steps: int
    
    # === Results (accumulated from parallel Retriever agents) ===
    papers: Annotated[List[Dict[str, Any]], operator.add]
    videos: Annotated[List[Dict[str, Any]], operator.add]
    web_results: Annotated[List[Dict[str, Any]], operator.add]
    
    # === Evaluated Results (from Evaluator agent) ===
    evaluated_papers: Optional[List[Dict[str, Any]]]
    evaluated_videos: Optional[List[Dict[str, Any]]]
    
    # === Summaries ===
    summaries: List[Dict[str, Any]]
    final_report: Optional[str]
    
    # === Memory (MemGPT-inspired long-term retrieval) ===
    working_memory: List[str]  # Current session context
    recalled_memories: List[Dict[str, Any]]  # Retrieved from long-term
    
    # === Control Flow ===
    next_agent: str
    should_continue: bool
    iteration: int
    max_iterations: int
    
    # === Errors & Logging ===
    errors: Annotated[List[str], operator.add]
    logs: Annotated[List[str], operator.add]
    
    # === Metadata ===
    start_time: str
    end_time: Optional[str]


def create_initial_state(user_goal: str, max_iterations: int = 3) -> ResearchState:
    """Create initial state for a new research session."""
    return ResearchState(
        # Input
        user_goal=user_goal,
        
        # Planning
        plan=None,
        current_step=0,
        total_steps=0,
        
        # Results (from Retrievers)
        papers=[],
        videos=[],
        web_results=[],
        
        # Evaluated Results (from Evaluator)
        evaluated_papers=None,
        evaluated_videos=None,
        
        # Summaries
        summaries=[],
        final_report=None,
        
        # Memory
        working_memory=[],
        recalled_memories=[],
        
        # Control
        next_agent="planner",
        should_continue=True,
        iteration=0,
        max_iterations=max_iterations,
        
        # Errors
        errors=[],
        logs=[],
        
        # Metadata
        start_time=datetime.utcnow().isoformat(),
        end_time=None,
    )
