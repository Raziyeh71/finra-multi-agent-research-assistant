"""
FinRA FastAPI Application with Streaming UI

Features:
- Real-time streaming updates via Server-Sent Events (SSE)
- Beautiful, responsive UI with Tailwind CSS
- Search papers and videos with ranking
- Click to view papers/videos directly
- Prometheus-compatible metrics endpoint

Only requires OpenAI API key - all other tools are FREE!
"""

import asyncio
import json
import os
from datetime import datetime
from typing import Optional, List, Dict, Any, AsyncGenerator

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, StreamingResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="FinRA - Finance Research Assistant",
    description="Multi-agent AI system for finding top papers and videos",
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ResearchRequest(BaseModel):
    topic: str = Field(..., description="Research topic or keywords")
    max_papers: int = Field(default=10, ge=1, le=50)
    max_videos: int = Field(default=10, ge=1, le=50)
    max_age_days: int = Field(default=365, ge=1)
    include_videos: bool = Field(default=True)


@app.get("/", response_class=HTMLResponse)
async def home():
    return get_html_ui()


@app.get("/api/research/stream")
async def research_stream(
    topic: str = Query(..., description="Research topic or keywords"),
    max_papers: int = Query(default=10, ge=1, le=50, description="Max papers to return"),
    max_videos: int = Query(default=10, ge=0, le=50, description="Max videos to return"),
    max_age_days: int = Query(default=730, ge=1, description="Max age of content in days (default 2 years, prioritizes 2025/2024)"),
    include_videos: bool = Query(default=True, description="Include YouTube search"),
):
    """
    Stream research results via Server-Sent Events (SSE).
    
    Runs the same multi-agent pipeline as `run_multi_agent.py`:
    Planner → Retrievers → Evaluator → Summarizer → Reporter
    
    **SSE Event Format:**
    
    Each event has `event: <type>` and `data: <json>`:
    
    - `status`: Progress updates `{"message": "...", "progress": 0-100}`
    - `paper`: Paper found `{"rank": 1, "title": "...", "link": "...", "score": 8.5}`
    - `video`: Video found `{"rank": 1, "title": "...", "link": "...", "score": 7.2}`
    - `done`: Research complete `{"papers": [...], "videos": [...], "total_time": 12.5}`
    
    **Example:**
    ```
    event: status
    data: {"message": "Planning research...", "progress": 10}
    
    event: paper
    data: {"rank": 1, "title": "Deep Learning for Trading", "link": "https://arxiv.org/...", "score": 8.5}
    ```
    """
    return StreamingResponse(
        stream_research(topic, max_papers, max_videos if include_videos else 0, max_age_days),
        media_type="text/event-stream",
    )


async def stream_research(
    topic: str,
    max_papers: int = 10,
    max_videos: int = 10,
    max_age_days: int = 365,
) -> AsyncGenerator[str, None]:
    from tools.search_tools import search_arxiv, search_google_scholar
    from tools.scraper_tools import scrape_youtube
    from tools.ranking import rank_papers, rank_videos, format_paper_for_display, format_video_for_display
    from tools.summarizer_tools import summarize_paper, summarize_video, is_finance_ai_related
    
    api_key = os.getenv("OPENAI_API_KEY")
    
    def sse(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"
    
    try:
        yield sse("status", {"message": f"Starting research: {topic}", "progress": 0})
        
        # Enhance query with AI/ML/DL innovation keywords for better results
        # Priority: Latest innovations (transformers, LLMs, RL, etc.)
        innovation_terms = [
            "transformer",
            "deep learning",
            "machine learning",
            "neural network",
            "reinforcement learning",
            "LLM",
        ]
        enhanced_queries = [
            f"{topic} {innovation_terms[0]}",  # transformer (most innovative)
            f"{topic} {innovation_terms[1]}",  # deep learning
            f"{topic} {innovation_terms[2]}",  # machine learning
            f"{topic} prediction neural network",
        ]
        
        yield sse("status", {"message": "Searching arXiv...", "progress": 10})
        arxiv_papers = []
        for eq in enhanced_queries[:2]:
            papers = await search_arxiv(eq, max_results=15, max_age_days=max_age_days)
            arxiv_papers.extend(papers)
        # Also search original query
        arxiv_papers.extend(await search_arxiv(topic, max_results=15, max_age_days=max_age_days))
        yield sse("status", {"message": f"Found {len(arxiv_papers)} arXiv papers", "progress": 25})
        
        yield sse("status", {"message": "Searching Google Scholar...", "progress": 30})
        scholar_papers = []
        for eq in enhanced_queries[:2]:
            papers = await search_google_scholar(eq, max_results=10, max_age_days=max_age_days)
            scholar_papers.extend(papers)
        yield sse("status", {"message": f"Found {len(scholar_papers)} Scholar papers", "progress": 45})
        
        all_papers = arxiv_papers + scholar_papers
        
        # Deduplicate by title (case-insensitive)
        seen_titles = set()
        unique_papers = []
        for p in all_papers:
            title_lower = p.get("title", "").lower().strip()
            if title_lower and title_lower not in seen_titles:
                seen_titles.add(title_lower)
                unique_papers.append(p)
        
        relevant_papers = [p for p in unique_papers if is_finance_ai_related(p.get("title", ""), p.get("abstract", ""))]
        yield sse("status", {"message": f"{len(relevant_papers)} relevant papers (from {len(unique_papers)} unique)", "progress": 50})
        
        top_papers = rank_papers(relevant_papers, top_n=max_papers)
        
        for i, paper in enumerate(top_papers):
            yield sse("paper", format_paper_for_display(paper, i + 1))
            await asyncio.sleep(0.05)
        
        if max_videos > 0:
            yield sse("status", {"message": "Searching YouTube...", "progress": 60})
            # Search with specific AI/ML/DL keywords for high-quality educational videos
            video_queries = [
                f"{topic} machine learning python tutorial",
                f"{topic} deep learning neural network",
                f"{topic} transformer model pytorch",
                f"{topic} reinforcement learning trading",
            ]
            all_videos = []
            for vq in video_queries:
                vids = await scrape_youtube(vq, max_results=20)
                all_videos.extend(vids)
            
            # Deduplicate videos by title
            seen_titles = set()
            unique_videos = []
            for v in all_videos:
                title_lower = v.get("title", "").lower().strip()
                if title_lower and title_lower not in seen_titles:
                    seen_titles.add(title_lower)
                    unique_videos.append(v)
            
            top_videos = rank_videos(unique_videos, top_n=max_videos)
            
            for i, video in enumerate(top_videos):
                yield sse("video", format_video_for_display(video, i + 1))
                await asyncio.sleep(0.05)
        else:
            top_videos = []
        
        yield sse("status", {"message": "Generating summaries...", "progress": 80})
        
        # Summarize ALL papers (not just 5)
        total_items = len(top_papers) + len(top_videos)
        for i, paper in enumerate(top_papers):
            if api_key:
                summary = await summarize_paper(paper, api_key)
                yield sse("summary", {"type": "paper", "rank": i + 1, "summary": summary})
                # Update progress
                progress = 80 + int((i + 1) / total_items * 18)
                yield sse("status", {"message": f"Summarized paper {i + 1}/{len(top_papers)}", "progress": progress})
        
        # Summarize ALL videos (not just 5)
        for i, video in enumerate(top_videos):
            if api_key:
                summary = await summarize_video(video, api_key)
                yield sse("summary", {"type": "video", "rank": i + 1, "summary": summary})
                # Update progress
                progress = 80 + int((len(top_papers) + i + 1) / total_items * 18)
                yield sse("status", {"message": f"Summarized video {i + 1}/{len(top_videos)}", "progress": progress})
        
        yield sse("complete", {"papers": len(top_papers), "videos": len(top_videos)})
        
    except Exception as e:
        yield sse("error", {"message": str(e)})


def get_html_ui() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FinRA - Finance Research Assistant</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .gradient-bg { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
        .card-hover:hover { transform: translateY(-2px); box-shadow: 0 10px 40px rgba(0,0,0,0.15); }
        .fade-in { animation: fadeIn 0.3s ease-in; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
    </style>
</head>
<body class="bg-gray-50 min-h-screen">
    <div class="gradient-bg text-white py-12 px-4">
        <div class="max-w-4xl mx-auto text-center">
            <h1 class="text-4xl font-bold mb-2">FinRA</h1>
            <p class="text-xl opacity-90">Finance Research Assistant</p>
            <p class="mt-2 opacity-75">Find top AI/ML papers and videos for trading & finance</p>
        </div>
    </div>
    
    <div class="max-w-4xl mx-auto px-4 -mt-8">
        <div class="bg-white rounded-xl shadow-lg p-6 mb-8">
            <div class="flex gap-4 flex-wrap">
                <input type="text" id="topic" placeholder="Enter research topic (e.g., LLM trading, stock prediction)"
                    class="flex-1 min-w-64 px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent">
                <select id="maxPapers" class="px-4 py-3 border border-gray-300 rounded-lg">
                    <option value="5">5 Papers</option>
                    <option value="10" selected>10 Papers</option>
                    <option value="20">20 Papers</option>
                </select>
                <select id="maxVideos" class="px-4 py-3 border border-gray-300 rounded-lg">
                    <option value="0">No Videos</option>
                    <option value="5">5 Videos</option>
                    <option value="10" selected>10 Videos</option>
                </select>
                <button onclick="startSearch()" id="searchBtn"
                    class="px-6 py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition font-medium">
                    Search
                </button>
            </div>
            <div id="progress" class="mt-4 hidden">
                <div class="flex items-center gap-2 text-gray-600">
                    <div class="animate-spin h-5 w-5 border-2 border-purple-600 border-t-transparent rounded-full"></div>
                    <span id="statusText">Starting...</span>
                </div>
                <div class="mt-2 h-2 bg-gray-200 rounded-full overflow-hidden">
                    <div id="progressBar" class="h-full bg-purple-600 transition-all duration-300" style="width: 0%"></div>
                </div>
            </div>
        </div>
        
        <div id="results" class="space-y-8 pb-12"></div>
    </div>

    <script>
        let papers = [];
        let videos = [];
        
        function startSearch() {
            const topic = document.getElementById('topic').value.trim();
            if (!topic) { alert('Please enter a topic'); return; }
            
            papers = [];
            videos = [];
            document.getElementById('results').innerHTML = '';
            document.getElementById('progress').classList.remove('hidden');
            document.getElementById('searchBtn').disabled = true;
            
            const params = new URLSearchParams({
                topic: topic,
                max_papers: document.getElementById('maxPapers').value,
                max_videos: document.getElementById('maxVideos').value,
                include_videos: document.getElementById('maxVideos').value !== '0'
            });
            
            const eventSource = new EventSource('/api/research/stream?' + params);
            
            eventSource.addEventListener('status', (e) => {
                const data = JSON.parse(e.data);
                document.getElementById('statusText').textContent = data.message;
                document.getElementById('progressBar').style.width = data.progress + '%';
            });
            
            eventSource.addEventListener('paper', (e) => {
                const paper = JSON.parse(e.data);
                papers.push(paper);
                renderPapers();
            });
            
            eventSource.addEventListener('video', (e) => {
                const video = JSON.parse(e.data);
                videos.push(video);
                renderVideos();
            });
            
            eventSource.addEventListener('summary', (e) => {
                const data = JSON.parse(e.data);
                if (data.type === 'paper' && papers[data.rank - 1]) {
                    papers[data.rank - 1].summary = data.summary;
                    renderPapers();
                } else if (data.type === 'video' && videos[data.rank - 1]) {
                    videos[data.rank - 1].summary = data.summary;
                    renderVideos();
                }
            });
            
            eventSource.addEventListener('complete', (e) => {
                eventSource.close();
                document.getElementById('progress').classList.add('hidden');
                document.getElementById('searchBtn').disabled = false;
            });
            
            eventSource.addEventListener('error', (e) => {
                eventSource.close();
                document.getElementById('progress').classList.add('hidden');
                document.getElementById('searchBtn').disabled = false;
                if (e.data) {
                    const data = JSON.parse(e.data);
                    alert('Error: ' + data.message);
                }
            });
        }
        
        function renderPapers() {
            let html = '<div class="bg-white rounded-xl shadow-lg p-6"><h2 class="text-2xl font-bold mb-4 text-gray-800">📚 Top Papers</h2><div class="space-y-4">';
            papers.forEach(p => {
                html += `
                <div class="border border-gray-200 rounded-lg p-4 card-hover transition fade-in">
                    <div class="flex items-start gap-3">
                        <span class="bg-purple-100 text-purple-700 font-bold px-3 py-1 rounded-full text-sm">#${p.rank}</span>
                        <div class="flex-1">
                            <a href="${p.link}" target="_blank" class="text-lg font-semibold text-blue-600 hover:underline">${p.title}</a>
                            <div class="flex gap-4 mt-1 text-sm text-gray-500">
                                <span>📖 ${p.source}</span>
                                <span>📅 ${p.date}</span>
                                <span>📊 ${p.citations} citations</span>
                                <span>⭐ Score: ${p.score.toFixed(2)}</span>
                            </div>
                            <p class="mt-2 text-gray-600 text-sm">${p.abstract}</p>
                            ${p.summary ? `<div class="mt-2 p-3 bg-green-50 rounded-lg text-sm"><strong>AI Summary:</strong> ${p.summary}</div>` : ''}
                        </div>
                    </div>
                </div>`;
            });
            html += '</div></div>';
            
            const resultsDiv = document.getElementById('results');
            const videosSection = resultsDiv.querySelector('.videos-section');
            const papersSection = resultsDiv.querySelector('.papers-section');
            
            if (papersSection) {
                papersSection.outerHTML = '<div class="papers-section">' + html + '</div>';
            } else {
                resultsDiv.insertAdjacentHTML('afterbegin', '<div class="papers-section">' + html + '</div>');
            }
        }
        
        function renderVideos() {
            let html = '<div class="bg-white rounded-xl shadow-lg p-6"><h2 class="text-2xl font-bold mb-4 text-gray-800">🎥 Top Videos</h2><div class="space-y-4">';
            videos.forEach(v => {
                html += `
                <div class="border border-gray-200 rounded-lg p-4 card-hover transition fade-in">
                    <div class="flex items-start gap-3">
                        <span class="bg-red-100 text-red-700 font-bold px-3 py-1 rounded-full text-sm">#${v.rank}</span>
                        <div class="flex-1">
                            <a href="${v.link}" target="_blank" class="text-lg font-semibold text-blue-600 hover:underline">${v.title}</a>
                            <div class="flex gap-4 mt-1 text-sm text-gray-500">
                                <span>📺 ${v.channel}</span>
                                <span>👁 ${v.views_formatted} views</span>
                                <span>📅 ${v.date}</span>
                                <span>⭐ Score: ${v.score.toFixed(2)}</span>
                            </div>
                            <p class="mt-2 text-gray-600 text-sm">${v.description}</p>
                            ${v.summary ? `<div class="mt-2 p-3 bg-blue-50 rounded-lg text-sm"><strong>AI Summary:</strong> ${v.summary}</div>` : ''}
                        </div>
                    </div>
                </div>`;
            });
            html += '</div></div>';
            
            const resultsDiv = document.getElementById('results');
            const videosSection = resultsDiv.querySelector('.videos-section');
            
            if (videosSection) {
                videosSection.outerHTML = '<div class="videos-section">' + html + '</div>';
            } else {
                resultsDiv.insertAdjacentHTML('beforeend', '<div class="videos-section">' + html + '</div>');
            }
        }
        
        document.getElementById('topic').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') startSearch();
        });
    </script>
</body>
</html>"""


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    """
    Prometheus-compatible metrics endpoint.
    
    Returns metrics in Prometheus text format for monitoring.
    """
    try:
        from multi_agent_system.production import get_prometheus_metrics
        return get_prometheus_metrics()
    except ImportError:
        return """# HELP finra_requests_total Total requests
# TYPE finra_requests_total counter
finra_requests_total 0
"""


@app.get("/health")
async def health():
    """Health check endpoint for load balancers."""
    return {
        "status": "healthy",
        "version": "2.1.0",
        "timestamp": datetime.utcnow().isoformat(),
    }


if __name__ == "__main__":
    import uvicorn
    # Cloud Run uses PORT env var, default to 8000 for local development
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
