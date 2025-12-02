"""
Ranking algorithms for papers and videos.

Papers: Recency (PRIMARY) + Venue Prestige + Citations + Relevance
Videos: Relevance + Views + Recency

Priority: 2025 papers > 2024 papers > older
Top venues: AAAI, ICML, NeurIPS, ICLR, CVPR, ACL, EMNLP, KDD, WWW, SIGIR, etc.
"""

from typing import List, Dict, Any
from datetime import datetime
import math

# Top-tier AI/ML/DL conferences and journals (for venue bonus)
TOP_VENUES = {
    # Tier 1 - Premier AI/ML conferences (1.0)
    "neurips": 1.0, "nips": 1.0, "icml": 1.0, "iclr": 1.0,
    "neural information processing": 1.0,  # Full name
    "international conference on machine learning": 1.0,
    "international conference on learning representations": 1.0,
    
    # Tier 1 - Top AI conferences (0.95)
    "aaai": 0.95, "cvpr": 0.95, "acl": 0.95,
    "association for the advancement of artificial intelligence": 0.95,
    "computer vision and pattern recognition": 0.95,
    "association for computational linguistics": 0.95,
    
    # Tier 1 - Vision/NLP (0.9)
    "iccv": 0.9, "eccv": 0.9, "emnlp": 0.9, "naacl": 0.9,
    "international conference on computer vision": 0.9,
    "empirical methods in natural language processing": 0.9,
    
    # Tier 1 - Data Mining/IR (0.9)
    "kdd": 0.9, "sigkdd": 0.9, "www": 0.9, "sigir": 0.9, "wsdm": 0.85,
    "knowledge discovery and data mining": 0.9,
    "world wide web": 0.9,
    
    # Tier 1 - Top Journals (1.0)
    "nature": 1.0, "science": 1.0, "cell": 1.0,
    "nature machine intelligence": 1.0,
    "nature communications": 0.9,
    
    # Tier 1 - ML/AI Journals (0.9)
    "jmlr": 0.9, "journal of machine learning research": 0.9,
    "tpami": 0.9, "ieee transactions on pattern analysis": 0.9,
    "tacl": 0.9, "transactions of the association for computational linguistics": 0.9,
    "artificial intelligence": 0.9,
    "machine learning": 0.85,
    
    # Tier 2 - Good venues (0.8)
    "ijcai": 0.8, "aistats": 0.8, "uai": 0.8, "colt": 0.8, "coling": 0.75,
    "international joint conference on artificial intelligence": 0.8,
    "uncertainty in artificial intelligence": 0.8,
    
    # Finance-specific (0.9)
    "journal of finance": 0.9, "review of financial studies": 0.9, 
    "journal of financial economics": 0.9, "quantitative finance": 0.85,
    "journal of banking": 0.8, "financial analysts journal": 0.8,
    "journal of portfolio management": 0.8,
    
    # arXiv categories (0.7) - preprints but often good
    "arxiv": 0.7, "cs.lg": 0.7, "cs.ai": 0.7, "cs.cl": 0.7, "q-fin": 0.7,
    "stat.ml": 0.7,
}

# Innovation keywords (boost papers with these)
INNOVATION_KEYWORDS = [
    "transformer", "attention", "llm", "large language model", "gpt", "bert",
    "diffusion", "generative", "foundation model", "self-supervised",
    "reinforcement learning", "rl", "ppo", "dpo", "rlhf",
    "graph neural", "gnn", "contrastive learning", "multimodal",
    "state-of-the-art", "sota", "novel", "breakthrough", "outperform",
]


def get_venue_score(paper: Dict[str, Any]) -> float:
    """Get venue prestige score (0-1)."""
    venue = paper.get("venue", "").lower()
    source = paper.get("source", "").lower()
    link = paper.get("link", "").lower()
    abstract = paper.get("abstract", "").lower()
    
    # Combine all text to search for venue mentions
    all_text = f"{venue} {source} {link} {abstract}"
    
    # Check for venue mentions
    best_score = 0.0
    for venue_name, score in TOP_VENUES.items():
        if venue_name in all_text:
            best_score = max(best_score, score)
    
    if best_score > 0:
        return best_score
    
    # arXiv papers get moderate score (preprints but often good)
    if "arxiv" in source or "arxiv.org" in link:
        return 0.7
    
    # Google Scholar without venue info
    if "scholar" in source:
        return 0.5
    
    return 0.4


def get_innovation_score(paper: Dict[str, Any]) -> float:
    """Score based on innovation keywords in title/abstract."""
    text = (paper.get("title", "") + " " + paper.get("abstract", "")).lower()
    
    matches = sum(1 for kw in INNOVATION_KEYWORDS if kw in text)
    # Normalize: 0 matches = 0, 3+ matches = 1.0
    return min(matches / 3, 1.0)


def get_recency_score(paper: Dict[str, Any]) -> float:
    """
    Calculate recency score with STRONG preference for 2025.
    
    2025: 1.0 (maximum)
    2024: 0.7-0.9 (high)
    2023: 0.4-0.6 (medium)
    2022 and older: 0.1-0.3 (low)
    """
    year = paper.get("year")
    date_str = paper.get("date", "")
    
    # Try to extract year
    if not year:
        try:
            if "2025" in str(date_str):
                year = 2025
            elif "2024" in str(date_str):
                year = 2024
            elif "2023" in str(date_str):
                year = 2023
            else:
                # Try parsing
                for fmt in ["%Y-%m-%d", "%Y/%m/%d", "%B %Y", "%Y"]:
                    try:
                        parsed = datetime.strptime(str(date_str)[:10], fmt)
                        year = parsed.year
                        break
                    except:
                        continue
        except:
            pass
    
    current_year = datetime.now().year
    
    if not year:
        return 0.3  # Unknown date gets low score
    
    if year >= 2025:
        return 1.0  # 2025 = maximum priority
    elif year == 2024:
        return 0.8  # 2024 = high priority
    elif year == 2023:
        return 0.5  # 2023 = medium
    elif year == 2022:
        return 0.3  # 2022 = low
    else:
        return 0.1  # Older = very low


def calculate_paper_score(
    paper: Dict[str, Any],
    relevance_score: float = 1.0,
    weights: Dict[str, float] = None,
) -> float:
    """
    Calculate ranking score for a paper.
    
    BALANCED PRIORITY (all important):
    1. Recency (2025 > 2024 > older) - 30%
    2. Citations (highly cited = impactful) - 30%
    3. Venue prestige (top conferences/journals) - 25%
    4. Innovation keywords - 15%
    
    Args:
        paper: Paper dictionary
        relevance_score: Relevance score from 0-1
        weights: Custom weights
        
    Returns:
        Combined score (higher = better)
    """
    if weights is None:
        weights = {
            "recency": 0.30,      # 30% - Recent papers
            "citations": 0.30,    # 30% - Citations EQUALLY important
            "venue": 0.25,        # 25% - Top conferences/journals
            "innovation": 0.15,   # 15% - Innovation keywords
        }
    
    # Calculate component scores
    recency = get_recency_score(paper)
    venue = get_venue_score(paper)
    innovation = get_innovation_score(paper)
    
    # Citations - use better normalization
    # 0 citations = 0, 10 = 0.33, 50 = 0.57, 100 = 0.67, 500 = 0.90, 1000+ = 1.0
    citations = paper.get("citations", 0)
    if citations > 0:
        citations_normalized = math.log10(citations + 1) / 3  # /3 instead of /4 for higher scores
        citations_normalized = min(citations_normalized, 1.0)
    else:
        # For papers with 0 citations (new papers or arXiv), give partial credit if recent
        citations_normalized = 0.2 if recency >= 0.8 else 0.0
    
    # Store component scores for display
    paper["_recency_score"] = round(recency, 2)
    paper["_venue_score"] = round(venue, 2)
    paper["_innovation_score"] = round(innovation, 2)
    paper["_citations_score"] = round(citations_normalized, 2)
    
    # Calculate weighted score
    score = (
        recency * weights["recency"] +
        citations_normalized * weights["citations"] +
        venue * weights["venue"] +
        innovation * weights["innovation"]
    )
    
    return round(score, 4)


# AI/ML/DL keywords for video relevance scoring
VIDEO_RELEVANCE_KEYWORDS = {
    # High relevance (1.0)
    "machine learning": 1.0, "deep learning": 1.0, "neural network": 1.0,
    "transformer": 1.0, "lstm": 1.0, "reinforcement learning": 1.0,
    "pytorch": 1.0, "tensorflow": 1.0, "keras": 1.0,
    "trading bot": 1.0, "algorithmic trading": 1.0, "quant": 1.0,
    "stock prediction": 1.0, "price prediction": 1.0,
    # Medium relevance (0.7)
    "python": 0.7, "tutorial": 0.7, "course": 0.7, "lecture": 0.7,
    "ai": 0.7, "artificial intelligence": 0.7, "data science": 0.7,
    "backtest": 0.7, "strategy": 0.7, "model": 0.7,
    # Low relevance (0.4)
    "finance": 0.4, "trading": 0.4, "stock": 0.4, "market": 0.4,
}

# Keywords that indicate low-quality/irrelevant content
VIDEO_NEGATIVE_KEYWORDS = [
    "shorts", "#shorts", "tiktok", "reels", "meme", "funny",
    "reaction", "vlog", "unboxing", "asmr", "prank",
    "get rich quick", "guaranteed profit", "100% win",
    "millionaire", "lambo", "lifestyle",
]

# Trusted educational channels (bonus score)
TRUSTED_CHANNELS = [
    "sentdex", "3blue1brown", "statquest", "two minute papers",
    "yannic kilcher", "andrej karpathy", "lex fridman",
    "stanford", "mit", "coursera", "udacity", "deepmind",
    "google", "microsoft", "nvidia", "hugging face",
    "freecodecamp", "tech with tim", "codebasics",
]


def get_video_relevance_score(video: Dict[str, Any]) -> float:
    """
    Calculate AI/ML relevance score for a video.
    
    Returns:
        Score from 0-1 (1 = highly relevant to AI/ML/DL)
    """
    title = video.get("title", "").lower()
    description = video.get("description", "").lower()
    channel = video.get("channel", "").lower()
    text = f"{title} {description} {channel}"
    
    # Check for negative keywords (disqualify)
    for neg_kw in VIDEO_NEGATIVE_KEYWORDS:
        if neg_kw in text:
            return 0.0  # Disqualify
    
    # Check for Shorts URL
    link = video.get("link", "").lower()
    if "/shorts/" in link:
        return 0.0  # Disqualify shorts
    
    # Calculate positive relevance
    max_score = 0.0
    keyword_matches = 0
    
    for keyword, score in VIDEO_RELEVANCE_KEYWORDS.items():
        if keyword in text:
            max_score = max(max_score, score)
            keyword_matches += 1
    
    # Bonus for multiple keyword matches
    if keyword_matches >= 3:
        max_score = min(max_score + 0.2, 1.0)
    
    # Bonus for trusted channels
    for trusted in TRUSTED_CHANNELS:
        if trusted in channel:
            max_score = min(max_score + 0.3, 1.0)
            break
    
    return max_score


def is_quality_video(video: Dict[str, Any]) -> bool:
    """
    Filter out low-quality videos (shorts, reels, etc.)
    
    Returns:
        True if video passes quality check
    """
    title = video.get("title", "").lower()
    link = video.get("link", "").lower()
    views = video.get("views", 0)
    
    # Filter out Shorts
    if "/shorts/" in link:
        return False
    
    # Filter out videos with negative keywords in title
    for neg_kw in VIDEO_NEGATIVE_KEYWORDS:
        if neg_kw in title:
            return False
    
    # Filter out very low view count (likely spam)
    if views < 100:
        return False
    
    # Title too short (likely clickbait)
    if len(title) < 15:
        return False
    
    return True


def calculate_video_score(
    video: Dict[str, Any],
    weights: Dict[str, float] = None,
) -> float:
    """
    Calculate ranking score for a video.
    
    PRIORITY ORDER:
    1. AI/ML/DL Relevance (50%) - MOST IMPORTANT
    2. Recency (30%) - Recent content
    3. Views/Engagement (20%) - Popular content
    
    Args:
        video: Video dictionary with views, recency_score
        weights: Custom weights for each factor
        
    Returns:
        Combined score (higher = better)
    """
    if weights is None:
        weights = {
            "relevance": 0.50,  # 50% - AI/ML relevance is TOP priority
            "recency": 0.30,    # 30% - Recent content
            "views": 0.20,      # 20% - Popular content
        }
    
    # Get relevance score (AI/ML keywords)
    relevance = get_video_relevance_score(video)
    
    # If relevance is 0, video is disqualified
    if relevance == 0:
        return 0.0
    
    # Get raw values
    views = video.get("views", 0)
    recency = video.get("recency_score", 5)
    
    # Normalize views using log scale
    views_normalized = math.log10(views + 1) / 6
    views_normalized = min(views_normalized, 1)
    
    # Normalize recency (already 0-10, convert to 0-1)
    recency_normalized = recency / 10
    
    # Store component scores
    video["_relevance_score"] = relevance
    video["_recency_score"] = recency_normalized
    video["_views_score"] = views_normalized
    
    # Calculate weighted score
    score = (
        relevance * weights["relevance"] +
        recency_normalized * weights["recency"] +
        views_normalized * weights["views"]
    )
    
    return round(score, 4)


def rank_papers(
    papers: List[Dict[str, Any]],
    top_n: int = 10,
    relevance_scores: Dict[str, float] = None,
) -> List[Dict[str, Any]]:
    """
    Rank papers by combined score and return top N.
    
    Args:
        papers: List of paper dictionaries
        top_n: Number of top papers to return
        relevance_scores: Optional dict mapping title to relevance score
        
    Returns:
        Top N papers sorted by score (highest first)
    """
    if relevance_scores is None:
        relevance_scores = {}
    
    # Calculate scores
    for paper in papers:
        title = paper.get("title", "")
        relevance = relevance_scores.get(title, 0.7)  # Default relevance
        paper["ranking_score"] = calculate_paper_score(paper, relevance)
    
    # Sort by score (descending)
    sorted_papers = sorted(papers, key=lambda x: x.get("ranking_score", 0), reverse=True)
    
    # Return top N
    return sorted_papers[:top_n]


def rank_videos(
    videos: List[Dict[str, Any]],
    top_n: int = 10,
) -> List[Dict[str, Any]]:
    """
    Rank videos by combined score and return top N.
    
    Filters out:
    - YouTube Shorts
    - Low-quality/spam content
    - Videos without AI/ML relevance
    
    Priority:
    1. AI/ML/DL relevance (50%)
    2. Recency (30%)
    3. Views (20%)
    
    Args:
        videos: List of video dictionaries
        top_n: Number of top videos to return
        
    Returns:
        Top N quality videos sorted by score (highest first)
    """
    # First, filter out low-quality videos
    quality_videos = [v for v in videos if is_quality_video(v)]
    
    # Calculate scores
    for video in quality_videos:
        video["ranking_score"] = calculate_video_score(video)
    
    # Filter out videos with 0 score (no AI/ML relevance)
    relevant_videos = [v for v in quality_videos if v.get("ranking_score", 0) > 0]
    
    # Sort by score (descending)
    sorted_videos = sorted(relevant_videos, key=lambda x: x.get("ranking_score", 0), reverse=True)
    
    print(f"📹 Video filtering: {len(videos)} → {len(quality_videos)} quality → {len(relevant_videos)} relevant")
    
    # Return top N
    return sorted_videos[:top_n]


def format_paper_for_display(paper: Dict[str, Any], rank: int) -> Dict[str, Any]:
    """Format a paper for display in the UI."""
    # Extract year for prominent display
    year = paper.get("year")
    date_str = paper.get("date", "")
    if not year:
        for y in ["2025", "2024", "2023", "2022"]:
            if y in str(date_str):
                year = int(y)
                break
    
    return {
        "rank": rank,
        "title": paper.get("title", ""),
        "link": paper.get("link", ""),
        "source": paper.get("source", ""),
        "date": paper.get("date", ""),
        "year": year,
        "citations": paper.get("citations", 0),
        "venue": paper.get("venue", ""),
        "abstract": paper.get("abstract", "")[:300] + "..." if len(paper.get("abstract", "")) > 300 else paper.get("abstract", ""),
        "summary": paper.get("summary", ""),
        "score": paper.get("ranking_score", 0),
    }


def format_video_for_display(video: Dict[str, Any], rank: int) -> Dict[str, Any]:
    """Format a video for display in the UI."""
    views = video.get("views", 0)
    views_str = f"{views:,}" if views < 1000000 else f"{views/1000000:.1f}M"
    
    return {
        "rank": rank,
        "title": video.get("title", ""),
        "link": video.get("link", ""),
        "channel": video.get("channel", ""),
        "date": video.get("date", ""),
        "views": views,
        "views_formatted": views_str,
        "description": video.get("description", "")[:200] + "..." if len(video.get("description", "")) > 200 else video.get("description", ""),
        "summary": video.get("summary", ""),
        "score": video.get("ranking_score", 0),
    }
