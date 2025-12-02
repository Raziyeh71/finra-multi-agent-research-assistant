"""
Summarization tools using GPT-4o-mini with domain-specific prompt engineering.

Planner and Summarizer use gpt-4o-mini by default (configurable).
This is the ONLY component that requires an API key (OpenAI).

Prompt Engineering Techniques:
- Structured prompts with clear sections (Problem, Method, Results, Applicability)
- Few-shot examples for consistent output format
- Domain-specific system prompts for finance/trading context
- Temperature tuning for factual accuracy (0.2-0.3)

Configuration via .env:
    LLM_MODEL=gpt-4o-mini
    LLM_PROVIDER=openai

Future PEFT Options:
- LoRA adapters for domain fine-tuning
- OpenAI fine-tuning API for custom models
- Prompt caching for high-volume production
"""

import os
import asyncio
import json
from typing import List, Dict, Any, Optional
from openai import OpenAI

# Default model (configurable via LLM_MODEL env var)
DEFAULT_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

def get_openai_client(api_key: str) -> OpenAI:
    """Get OpenAI client (v1.0+ API)."""
    return OpenAI(api_key=api_key)

# =============================================================================
# DOMAIN-SPECIFIC PROMPTS (Prompt Engineering Best Practices)
# =============================================================================

FINANCE_EXPERT_SYSTEM_PROMPT = """You are a senior quantitative researcher at a hedge fund, specializing in AI/ML applications for algorithmic trading.

Your expertise includes:
- Machine learning for price prediction and signal generation
- Deep learning architectures (Transformers, LSTMs, CNNs) for time-series
- Reinforcement learning for portfolio optimization
- NLP for sentiment analysis and news trading
- Risk management and backtesting methodologies

When summarizing research:
- Focus on practical implementation details
- Highlight quantitative results (accuracy, Sharpe ratio, returns)
- Assess real-world applicability for trading systems
- Use precise financial and ML terminology"""

FEW_SHOT_PAPER_EXAMPLES = """
EXAMPLE 1:
Title: "Transformer-based Stock Price Prediction using Multi-source Sentiment"
Abstract: We propose a BERT-based model that combines financial news sentiment with technical indicators for intraday S&P 500 prediction. Our approach achieves 62.3% directional accuracy and 1.82 Sharpe ratio in backtesting.

Summary:
**Problem**: Intraday directional prediction for S&P 500 constituents.
**Method**: BERT fine-tuned on financial news + attention fusion with technical indicators (RSI, MACD, volume).
**Results**: 62.3% directional accuracy, 1.82 Sharpe ratio, tested on 2019-2022 data.
**Applicability**: Use as confirmation signal for momentum strategies; sentiment scores can filter trade entries.

EXAMPLE 2:
Title: "Deep Reinforcement Learning for Dynamic Portfolio Allocation"
Abstract: This paper presents a PPO-based agent for multi-asset portfolio optimization with transaction costs. Compared to mean-variance optimization, our method achieves 15.2% annual return vs 11.1% baseline with 30% lower maximum drawdown.

Summary:
**Problem**: Dynamic asset allocation with realistic transaction costs.
**Method**: PPO agent with state = [returns, volatility, correlations]; action = portfolio weights; reward = risk-adjusted returns minus costs.
**Results**: 15.2% annual return (vs 11.1% baseline), 30% lower max drawdown, 0.95 Sharpe ratio.
**Applicability**: Suitable for daily/weekly rebalancing in multi-asset portfolios; requires careful hyperparameter tuning for live trading.
"""

FEW_SHOT_VIDEO_EXAMPLES = """
EXAMPLE 1:
Title: "Building a Trading Bot with Python and Machine Learning"
Channel: QuantPy
Description: Step-by-step tutorial on creating an ML-based trading system using scikit-learn and backtrader.

Summary:
**Content**: Hands-on tutorial covering data collection (yfinance), feature engineering (technical indicators), model training (Random Forest), and backtesting integration.
**Key Techniques**: Feature engineering with TA-Lib, walk-forward validation, position sizing.
**Skill Level**: Intermediate Python, basic ML knowledge required.
**Applicability**: Good starting point for prototyping; production systems need more robust infrastructure.

EXAMPLE 2:
Title: "Attention Mechanisms for Financial Time Series - Paper Explained"
Channel: AI in Finance
Description: Deep dive into using transformer attention for stock prediction, with code walkthrough.

Summary:
**Content**: Explains self-attention mechanism adapted for irregular time-series, positional encoding for trading days, and multi-head attention for multi-asset modeling.
**Key Techniques**: Temporal attention, cross-asset attention, handling missing data.
**Skill Level**: Advanced; requires deep learning background.
**Applicability**: State-of-the-art approach; implementation complexity high but results promising for institutional use.
"""

PAPER_SUMMARY_TEMPLATE = """You are analyzing a research paper for algorithmic trading applications.

PAPER:
Title: {title}
Abstract: {abstract}
Source: {source}

TASK: Provide a structured summary using EXACTLY this format:

**Problem**: [What trading/finance problem does this solve? 1-2 sentences]
**Method**: [What AI/ML technique is used? Include architecture details. 1-2 sentences]
**Results**: [Key quantitative findings - accuracy, returns, Sharpe ratio, etc. 1-2 sentences]
**Applicability**: [How could a quant trader implement this? Practical considerations. 1-2 sentences]

GUIDELINES:
- Be specific about ML architectures (e.g., "3-layer LSTM with attention" not just "deep learning")
- Include numbers when available (accuracy %, Sharpe ratio, return %)
- Mention data requirements and limitations
- Keep total response under {max_words} words

{few_shot_context}"""

PLANNER_PROMPT_TEMPLATE = """You are a research planning expert for quantitative finance.

USER GOAL: "{user_goal}"

MEMORY CONTEXT (similar past searches):
{memory_context}

TASK: Create an optimal research plan as JSON:

{{
    "search_terms": ["term1", "term2", "term3"],
    "sources": ["arxiv", "scholar", "nature", "ieee", "youtube"],
    "max_age_days": 90,
    "focus_areas": ["specific_area1", "specific_area2"],
    "reasoning": "Brief explanation of strategy"
}}

GUIDELINES:
- Generate 3-5 SPECIFIC search terms (e.g., "transformer stock prediction" not "AI trading")
- Choose sources based on goal: arxiv for cutting-edge, scholar for citations, youtube for tutorials
- Use shorter max_age_days (30-60) for "latest" or "recent" requests
- Exclude cryptocurrency unless explicitly requested
- Focus on implementable, practical research
- Consider what worked in similar past searches (memory context)

Output ONLY valid JSON, no markdown or explanation."""


async def summarize_paper(
    paper: Dict[str, Any],
    api_key: str,
    model: str = None,
    max_words: int = 150,
    use_few_shot: bool = True,
) -> str:
    """Summarize a research paper for algorithmic traders."""
    model = model or DEFAULT_MODEL
    client = get_openai_client(api_key)
    
    title = paper.get("title", "")
    abstract = paper.get("abstract", "")
    source = paper.get("source", "Unknown")
    
    if not abstract or abstract == "Abstract not available":
        return "**Problem**: Unknown (no abstract). **Method**: N/A. **Results**: N/A. **Applicability**: Cannot assess without abstract."
    
    few_shot_context = ""
    if use_few_shot:
        few_shot_context = f"\nREFERENCE EXAMPLES (follow this format):\n{FEW_SHOT_PAPER_EXAMPLES}"
    
    prompt = PAPER_SUMMARY_TEMPLATE.format(
        title=title,
        abstract=abstract[:2000],
        source=source,
        max_words=max_words,
        few_shot_context=few_shot_context,
    )

    try:
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=model,
            messages=[
                {"role": "system", "content": FINANCE_EXPERT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            max_tokens=400,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Summary error: {str(e)[:100]}"


async def summarize_video(
    video: Dict[str, Any],
    api_key: str,
    model: str = None,
    max_words: int = 120,
    use_few_shot: bool = True,
) -> str:
    """Summarize a video for algorithmic traders."""
    model = model or DEFAULT_MODEL
    client = get_openai_client(api_key)
    
    title = video.get("title", "")
    description = video.get("description", "")
    channel = video.get("channel", "")
    
    few_shot_context = ""
    if use_few_shot:
        few_shot_context = f"\nREFERENCE EXAMPLES (follow this format):\n{FEW_SHOT_VIDEO_EXAMPLES}"
    
    prompt = f"""You are analyzing a video for algorithmic trading education.

VIDEO:
Title: {title}
Channel: {channel}
Description: {description[:1000]}

TASK: Provide a structured summary using EXACTLY this format:

**Content**: [What topics/techniques are covered? 1-2 sentences]
**Key Techniques**: [Specific ML/trading methods mentioned. 1 sentence]
**Skill Level**: [Beginner/Intermediate/Advanced + prerequisites. 1 sentence]
**Applicability**: [How useful for building trading systems? 1 sentence]

Keep total response under {max_words} words.
{few_shot_context}"""

    try:
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=model,
            messages=[
                {"role": "system", "content": FINANCE_EXPERT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            max_tokens=250,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Summary error: {str(e)[:100]}"


async def summarize_batch(
    items: List[Dict[str, Any]],
    api_key: str,
    model: str = None,
    max_concurrent: int = 5,
) -> List[Dict[str, Any]]:
    """Summarize multiple items in parallel."""
    model = model or DEFAULT_MODEL
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def summarize_one(item):
        async with semaphore:
            item_type = item.get("type", "paper")
            if item_type == "video" or "channel" in item:
                summary = await summarize_video(item, api_key, model)
            else:
                summary = await summarize_paper(item, api_key, model)
            item["summary"] = summary
            return item
    
    tasks = [summarize_one(item) for item in items]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    output = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            items[i]["summary"] = f"Error: {str(result)[:100]}"
            output.append(items[i])
        else:
            output.append(result)
    
    return output


async def generate_research_plan(
    user_goal: str,
    api_key: str,
    model: str = None,
    memory_context: str = "",
) -> Dict[str, Any]:
    """Generate a research plan from user goal using LLM."""
    model = model or DEFAULT_MODEL
    client = get_openai_client(api_key)
    
    if not memory_context:
        memory_context = "No similar past searches found."
    
    prompt = PLANNER_PROMPT_TEMPLATE.format(
        user_goal=user_goal,
        memory_context=memory_context,
    )

    try:
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=model,
            messages=[
                {"role": "system", "content": "You are a quantitative finance research planning expert. Output only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.2,
        )
        
        content = response.choices[0].message.content.strip()
        
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        
        plan = json.loads(content)
        
        plan.setdefault("search_terms", ["AI trading", "machine learning finance"])
        plan.setdefault("sources", ["arxiv", "scholar"])
        plan.setdefault("max_age_days", 90)
        plan.setdefault("focus_areas", [])
        plan.setdefault("reasoning", "")
        
        return plan
        
    except Exception as e:
        return {
            "search_terms": ["AI algorithmic trading", "machine learning stock prediction"],
            "sources": ["arxiv", "scholar"],
            "max_age_days": 90,
            "focus_areas": ["machine learning", "trading"],
            "reasoning": f"Fallback plan due to error: {str(e)[:50]}",
        }


def is_finance_ai_related(title: str, abstract: str = "") -> bool:
    """Check if content is related to both finance/trading AND AI/ML."""
    text = (title + " " + abstract).lower()
    
    finance_terms = [
        "trading", "stock", "market", "portfolio", "finance", "financial",
        "investment", "asset", "price", "return", "risk", "hedge",
        "forex", "equity", "bond", "derivative", "option", "futures",
        "algorithmic", "quant", "alpha", "sharpe", "volatility"
    ]
    
    ai_terms = [
        "machine learning", "deep learning", "neural network", "ai",
        "artificial intelligence", "lstm", "transformer", "bert", "gpt",
        "reinforcement learning", "prediction", "forecast", "nlp",
        "sentiment", "classification", "regression", "model"
    ]
    
    crypto_terms = ["crypto", "bitcoin", "ethereum", "blockchain", "defi", "nft"]
    
    has_finance = any(term in text for term in finance_terms)
    has_ai = any(term in text for term in ai_terms)
    has_crypto = any(term in text for term in crypto_terms)
    
    return has_finance and has_ai and not has_crypto
