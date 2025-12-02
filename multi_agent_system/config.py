"""
Configuration for the multi-agent system.
All settings in one place for easy customization.

Environment Variables:
    OPENAI_API_KEY: Required - Your OpenAI API key
    LLM_MODEL: Optional - Model name (default: gpt-4o-mini)
    LLM_PROVIDER: Optional - Provider name (default: openai)
    
Production Environment Variables:
    MAX_CONCURRENT: Max parallel requests (default: 5)
    RATE_LIMIT_DELAY: Seconds between requests (default: 1.0)
    REQUEST_TIMEOUT: Request timeout in seconds (default: 30)
    MAX_RETRIES: Max retry attempts (default: 3)
    LOG_LEVEL: Logging level (default: INFO)
    ENV: Environment (development/production)
"""

import os
import logging
from dataclasses import dataclass, field
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()

# Default model configuration (configurable via .env)
DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
DEFAULT_LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")

# Production settings (configurable via .env)
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", "5"))
RATE_LIMIT_DELAY = float(os.getenv("RATE_LIMIT_DELAY", "1.0"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
ENV = os.getenv("ENV", "development")

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("finra")


@dataclass
class ModelConfig:
    """
    LLM model configuration.
    
    Planner and Summarizer use gpt-4o-mini by default (configurable).
    Set LLM_MODEL in .env to change the default model.
    """
    
    # Default model for all agents (configurable via LLM_MODEL env var)
    model_name: str = field(default_factory=lambda: DEFAULT_LLM_MODEL)
    provider: str = field(default_factory=lambda: DEFAULT_LLM_PROVIDER)
    temperature: float = 0.3
    max_tokens: int = 1000
    
    # Agent-specific models (default to same as model_name)
    planning_model: str = field(default_factory=lambda: DEFAULT_LLM_MODEL)
    summarization_model: str = field(default_factory=lambda: DEFAULT_LLM_MODEL)


@dataclass
class SearchConfig:
    """Search and scraping configuration with production-oriented settings."""
    
    # Results limits
    max_results_per_source: int = 10
    max_total_results: int = 50
    
    # Timeouts (seconds) - configurable via env
    page_timeout: int = field(default_factory=lambda: REQUEST_TIMEOUT)
    request_timeout: int = field(default_factory=lambda: REQUEST_TIMEOUT)
    
    # Rate limiting - configurable via env
    rate_limit_delay: float = field(default_factory=lambda: RATE_LIMIT_DELAY)
    max_concurrent: int = field(default_factory=lambda: MAX_CONCURRENT)
    
    # Retry settings - configurable via env
    max_retries: int = field(default_factory=lambda: MAX_RETRIES)
    retry_backoff_base: float = 2.0  # Exponential backoff base
    
    # Default sources to search
    default_sources: List[str] = field(
        default_factory=lambda: ["arxiv", "scholar", "nature"]
    )
    
    # Recency filter
    default_max_age_days: int = 90
    
    # Ethical scraping
    respect_robots_txt: bool = True
    user_agent: str = "FinRA-Research-Agent/2.0 (Academic Research Tool)"


@dataclass
class MemoryConfig:
    """Memory system configuration."""
    
    # ChromaDB settings
    persist_directory: str = "./finra_memory"
    collection_name: str = "finra_research"
    
    # Working memory limits
    max_working_memory_items: int = 10
    max_context_tokens: int = 3000
    
    # Retrieval settings
    max_recall_results: int = 5
    similarity_threshold: float = 0.7


@dataclass
class AgentConfig:
    """Agent behavior configuration."""
    
    # Iteration limits
    max_iterations: int = 3
    max_retries: int = 2
    
    # Parallel execution
    max_parallel_agents: int = 3
    
    # Summarization
    max_papers_to_summarize: int = 10
    max_videos_to_summarize: int = 5
    summary_max_words: int = 150


@dataclass
class FinRAConfig:
    """Main configuration class combining all settings."""
    
    # API Keys (only OpenAI required!)
    openai_api_key: Optional[str] = None
    
    # Sub-configs
    model: ModelConfig = field(default_factory=ModelConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    
    # Debug mode
    debug: bool = False
    verbose: bool = True
    
    def __post_init__(self):
        """Load API key from environment if not provided."""
        if self.openai_api_key is None:
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.openai_api_key:
            raise ValueError(
                "OpenAI API key is required. "
                "Set OPENAI_API_KEY environment variable or pass openai_api_key parameter."
            )


# Default configuration instance
DEFAULT_CONFIG = None  # Will be created when needed with API key


def get_config(openai_api_key: Optional[str] = None, **kwargs) -> FinRAConfig:
    """Get configuration with optional overrides."""
    return FinRAConfig(openai_api_key=openai_api_key, **kwargs)
