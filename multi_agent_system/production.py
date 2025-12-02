"""
Production utilities for FinRA multi-agent system.

Provides:
- Retry with exponential backoff
- Rate limiting with semaphores
- Request timeout handling
- Structured logging
- Metrics collection (optional)
"""

import asyncio
import time
import logging
from functools import wraps
from typing import Any, Callable, Optional, TypeVar, Dict
from dataclasses import dataclass, field
from datetime import datetime

from .config import (
    MAX_CONCURRENT,
    RATE_LIMIT_DELAY,
    REQUEST_TIMEOUT,
    MAX_RETRIES,
    ENV,
    logger,
)

T = TypeVar("T")


# Global semaphore for concurrency control
_semaphore: Optional[asyncio.Semaphore] = None
_last_request_time: Dict[str, float] = {}


def get_semaphore() -> asyncio.Semaphore:
    """Get or create the global semaphore for concurrency control."""
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    return _semaphore


@dataclass
class RequestMetrics:
    """Metrics for a single request."""
    url: str
    start_time: float
    end_time: Optional[float] = None
    success: bool = False
    retries: int = 0
    error: Optional[str] = None
    
    @property
    def duration_ms(self) -> float:
        if self.end_time is None:
            return 0
        return (self.end_time - self.start_time) * 1000


@dataclass
class MetricsCollector:
    """Collects and reports metrics for monitoring."""
    
    requests: list = field(default_factory=list)
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_retries: int = 0
    
    def record(self, metrics: RequestMetrics):
        """Record a request's metrics."""
        self.requests.append(metrics)
        self.total_requests += 1
        self.total_retries += metrics.retries
        
        if metrics.success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
    
    def summary(self) -> Dict[str, Any]:
        """Get a summary of collected metrics."""
        if not self.requests:
            return {"total": 0}
        
        durations = [r.duration_ms for r in self.requests if r.end_time]
        
        return {
            "total_requests": self.total_requests,
            "successful": self.successful_requests,
            "failed": self.failed_requests,
            "total_retries": self.total_retries,
            "avg_duration_ms": sum(durations) / len(durations) if durations else 0,
            "max_duration_ms": max(durations) if durations else 0,
            "success_rate": self.successful_requests / self.total_requests if self.total_requests else 0,
        }


# Global metrics collector
metrics_collector = MetricsCollector()


async def rate_limit(source: str = "default"):
    """
    Apply rate limiting delay for a specific source.
    
    Ensures minimum delay between requests to the same source.
    """
    global _last_request_time
    
    now = time.time()
    last_time = _last_request_time.get(source, 0)
    elapsed = now - last_time
    
    if elapsed < RATE_LIMIT_DELAY:
        delay = RATE_LIMIT_DELAY - elapsed
        logger.debug(f"Rate limiting: waiting {delay:.2f}s for {source}")
        await asyncio.sleep(delay)
    
    _last_request_time[source] = time.time()


def with_retry(
    max_retries: int = MAX_RETRIES,
    backoff_base: float = 2.0,
    exceptions: tuple = (Exception,),
):
    """
    Decorator for async functions with exponential backoff retry.
    
    Args:
        max_retries: Maximum number of retry attempts
        backoff_base: Base for exponential backoff (delay = base^attempt)
        exceptions: Tuple of exceptions to catch and retry
    
    Example:
        @with_retry(max_retries=3)
        async def fetch_data(url):
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        delay = backoff_base ** attempt
                        logger.warning(
                            f"Retry {attempt + 1}/{max_retries} for {func.__name__}: "
                            f"{str(e)[:100]}. Waiting {delay:.1f}s"
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            f"All {max_retries} retries failed for {func.__name__}: {e}"
                        )
            
            raise last_exception
        
        return wrapper
    return decorator


def with_timeout(timeout: int = REQUEST_TIMEOUT):
    """
    Decorator for async functions with timeout.
    
    Args:
        timeout: Timeout in seconds
    
    Example:
        @with_timeout(30)
        async def slow_operation():
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                logger.error(f"Timeout ({timeout}s) for {func.__name__}")
                raise
        
        return wrapper
    return decorator


def with_concurrency_limit(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to limit concurrent executions using a semaphore.
    
    Example:
        @with_concurrency_limit
        async def fetch_page(url):
            ...
    """
    @wraps(func)
    async def wrapper(*args, **kwargs) -> T:
        semaphore = get_semaphore()
        async with semaphore:
            return await func(*args, **kwargs)
    
    return wrapper


async def safe_request(
    func: Callable[..., T],
    *args,
    source: str = "default",
    timeout: int = REQUEST_TIMEOUT,
    max_retries: int = MAX_RETRIES,
    **kwargs,
) -> Optional[T]:
    """
    Execute a request with all production safeguards.
    
    Applies:
    - Rate limiting
    - Concurrency control
    - Timeout
    - Retry with backoff
    - Metrics collection
    
    Args:
        func: Async function to execute
        source: Source identifier for rate limiting
        timeout: Request timeout in seconds
        max_retries: Maximum retry attempts
        *args, **kwargs: Arguments to pass to func
    
    Returns:
        Result of func or None if all retries fail
    """
    metrics = RequestMetrics(url=source, start_time=time.time())
    semaphore = get_semaphore()
    
    try:
        # Rate limit
        await rate_limit(source)
        
        # Concurrency limit
        async with semaphore:
            # Retry with backoff
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    # Timeout
                    result = await asyncio.wait_for(
                        func(*args, **kwargs),
                        timeout=timeout
                    )
                    
                    metrics.success = True
                    metrics.end_time = time.time()
                    metrics.retries = attempt
                    metrics_collector.record(metrics)
                    
                    logger.debug(
                        f"Request to {source} succeeded in {metrics.duration_ms:.0f}ms "
                        f"(retries: {attempt})"
                    )
                    
                    return result
                    
                except asyncio.TimeoutError as e:
                    last_exception = e
                    logger.warning(f"Timeout for {source} (attempt {attempt + 1})")
                    
                except Exception as e:
                    last_exception = e
                    metrics.retries = attempt
                    
                    if attempt < max_retries:
                        delay = 2 ** attempt
                        logger.warning(
                            f"Retry {attempt + 1}/{max_retries} for {source}: "
                            f"{str(e)[:100]}. Waiting {delay:.1f}s"
                        )
                        await asyncio.sleep(delay)
            
            # All retries failed
            metrics.success = False
            metrics.end_time = time.time()
            metrics.error = str(last_exception)
            metrics_collector.record(metrics)
            
            logger.error(f"All retries failed for {source}: {last_exception}")
            return None
            
    except Exception as e:
        metrics.success = False
        metrics.end_time = time.time()
        metrics.error = str(e)
        metrics_collector.record(metrics)
        
        logger.error(f"Request to {source} failed: {e}")
        return None


def log_agent_start(agent_name: str, **context):
    """Log the start of an agent execution."""
    logger.info(f"🚀 {agent_name} started", extra={"agent": agent_name, **context})


def log_agent_complete(agent_name: str, duration_ms: float, **context):
    """Log the completion of an agent execution."""
    logger.info(
        f"✅ {agent_name} completed in {duration_ms:.0f}ms",
        extra={"agent": agent_name, "duration_ms": duration_ms, **context}
    )


def log_agent_error(agent_name: str, error: Exception, **context):
    """Log an agent error."""
    logger.error(
        f"❌ {agent_name} failed: {error}",
        extra={"agent": agent_name, "error": str(error), **context}
    )


# Prometheus-style metrics endpoint data
def get_prometheus_metrics() -> str:
    """
    Generate Prometheus-compatible metrics output.
    
    Can be exposed via FastAPI endpoint for monitoring.
    """
    summary = metrics_collector.summary()
    
    lines = [
        "# HELP finra_requests_total Total number of requests",
        "# TYPE finra_requests_total counter",
        f'finra_requests_total {summary.get("total_requests", 0)}',
        "",
        "# HELP finra_requests_successful Successful requests",
        "# TYPE finra_requests_successful counter",
        f'finra_requests_successful {summary.get("successful", 0)}',
        "",
        "# HELP finra_requests_failed Failed requests",
        "# TYPE finra_requests_failed counter",
        f'finra_requests_failed {summary.get("failed", 0)}',
        "",
        "# HELP finra_retries_total Total retry attempts",
        "# TYPE finra_retries_total counter",
        f'finra_retries_total {summary.get("total_retries", 0)}',
        "",
        "# HELP finra_request_duration_ms Average request duration in milliseconds",
        "# TYPE finra_request_duration_ms gauge",
        f'finra_request_duration_ms {summary.get("avg_duration_ms", 0):.2f}',
        "",
        "# HELP finra_success_rate Request success rate",
        "# TYPE finra_success_rate gauge",
        f'finra_success_rate {summary.get("success_rate", 0):.4f}',
    ]
    
    return "\n".join(lines)
