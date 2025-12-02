# 🔬 FinRA - Multi-Agent Finance Research Assistant

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.104+-green.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/LangGraph-0.2+-purple.svg" alt="LangGraph">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
  <img src="https://img.shields.io/badge/Deployed-Cloud%20Run-blue.svg" alt="Cloud Run">
</p>

**FinRA** is an intelligent multi-agent research assistant that automates the discovery, ranking, and summarization of cutting-edge AI/ML research papers and educational videos for quantitative finance and algorithmic trading.

**Key Highlights:**

- 🤖 **6-Agent LangGraph Pipeline** — Memory → Planner → Retrievers → Evaluator → Summarizer → Reporter
- 🔍 **Multi-Source Scraping** — arXiv, Google Scholar, Nature, IEEE, YouTube (all FREE via Playwright)
- 🧠 **GPT-4o-mini Powered** — Intelligent planning, domain-specific summaries, quality filtering
- 💾 **Semantic Memory** — ChromaDB-backed long-term recall (MemGPT-inspired)
- 🌊 **Real-time Streaming** — Server-Sent Events with beautiful Tailwind UI
- ☁️ **Cloud-Ready** — One-command deploy to Google Cloud Run

> **💡 Only OpenAI API key required** — all scraping tools are FREE!

---

## 🎯 Quick Start

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/FinRA.git
cd FinRA

# Install dependencies
pip install -r requirements.txt
playwright install

# Set your OpenAI API key
export OPENAI_API_KEY=your_key_here

# Run the web UI
python api.py
# Open http://localhost:8000
```

## 🚀 Features

### Multi-Agent Architecture (NEW!)

```text
┌─────────────────────────────────────────────────────────────┐
│                 FinRA Multi-Agent System                    │
└─────────────────────────────────────────────────────────────┘

  ┌──────────┐    ┌──────────┐    ┌──────────────┐    ┌────────────┐
  │  MEMORY  │───▶│ PLANNER  │───▶│  RETRIEVERS  │───▶│ SUMMARIZER │
  │ (recall) │    │  (plan)  │    │   (fetch)    │    │  (rank)    │
  └──────────┘    └──────────┘    └──────────────┘    └─────┬──────┘
       ▲                                                    │
       └────────────────── updates memory ──────────────────┘
```

**Agent Flow:**

1. **Memory Agent** (ChromaDB) — Queried first; recalls similar past queries and results
2. **Planner Agent** (GPT-4o-mini) — Uses user query + memory context to decide sources and search strategy
3. **Paper/Video Retrievers** (Playwright) — Scrape and fetch new results in parallel; merge with recalled results
4. **Evaluator Agent** (GPT-4o-mini) — Ranks by credibility (citations, views, recency)
5. **Summarizer Agent** (GPT-4o-mini) — Summarizes and ranks results; **updates memory for future runs**
6. **Reporter Agent** (GPT-4o-mini) — Creates final executive summary

> **Note:** All LLM agents use `gpt-4o-mini` by default (configurable via `.env`).

### Core Capabilities

- **Paper Sources**: arXiv API, Google Scholar, Nature, IEEE (all FREE!)
- **Video Sources**: YouTube (Playwright scraping - FREE!)
- **Web Search**: Google search via Playwright (FREE!)
- **Smart Filtering**: AI/ML + finance relevance detection, excludes crypto
- **Date Filtering**: Configurable recency filter (e.g., last 30/60/90 days)
- **AI Summaries**: GPT-4o-mini generates concise, trader-focused summaries

### Advanced Features

- **WebGPT-Style Browsing**: LLM-controlled browsing (WebGPT-style prompt pattern) using Playwright
- **MemGPT-Style Memory**: Long-term retrieval memory inspired by MemGPT, backed by ChromaDB (local vector store)
- **LangGraph Orchestration**: True multi-agent coordination
- **Parallel Execution**: 5-10x faster with async scrapers
- **LLM-Based Planning**: Intelligent research strategy generation

### Prompt Engineering (Domain-Optimized)

The system uses advanced prompt engineering techniques for finance/trading domain:

| Technique | Implementation | Benefit |
|-----------|----------------|---------|
| **Structured Prompts** | Problem → Method → Results → Applicability format | Consistent, actionable summaries |
| **Few-Shot Examples** | 2 domain-specific examples per prompt | Better output quality and formatting |
| **Domain System Prompt** | "Senior quant researcher" persona | Finance-specific terminology and focus |
| **Low Temperature** | 0.2 for summaries, 0.2 for planning | Factual accuracy over creativity |
| **Memory Context** | Past searches inform planning | Learns from previous research |

**Example Output Format (Papers):**

```text
**Problem**: Intraday directional prediction for S&P 500 constituents.
**Method**: BERT fine-tuned on financial news + attention fusion with technical indicators.
**Results**: 62.3% directional accuracy, 1.82 Sharpe ratio.
**Applicability**: Use as confirmation signal for momentum strategies.
```

**Future PEFT Options:**

- LoRA adapters for domain fine-tuning (OpenAI fine-tuning API)
- Prompt caching for high-volume production
- Custom embedding models for better semantic recall

## Requirements

- Python 3.8+ (3.10+ recommended)
- **OpenAI API key** (ONLY required API key!)
- No other API keys needed - everything else is FREE!

## Installation

```bash
# 1. Clone and install dependencies
pip install -r requirements.txt

# 2. Install Playwright browsers
playwright install

# 3. Create .env file with your configuration
cat > .env << EOF
OPENAI_API_KEY=your_key_here
LLM_MODEL=gpt-4o-mini
LLM_PROVIDER=openai
EOF
```

## Usage

### 🆕 Multi-Agent System (Recommended)

```bash
# Set your OpenAI API key
export OPENAI_API_KEY=your_key_here

# Run multi-agent research
python run_multi_agent.py "Find latest LLM trading papers"

# With options
python run_multi_agent.py --goal "AI stock prediction" --max-age 60 --include-videos

# Save to custom file
python run_multi_agent.py "transformer models finance" --output my_research.json
```

### Multi-Agent CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `goal` | str | - | Research goal (positional or --goal) |
| `--api-key` | str | env | OpenAI API key |
| `--max-age` | int | 90 | Max age of papers in days |
| `--max-papers` | int | 10 | Max papers to summarize |
| `--include-videos` | flag | False | Include YouTube search |
| `--include-web` | flag | True | Include web/Google search |
| `--output` | str | finra_research_results.json | Output file |
| `--memory-path` | str | ./finra_memory | ChromaDB memory path |
| `--debug` | flag | False | Enable debug mode |

### Legacy Mode (Single-Agent)

```bash
# Original FinRA agent (kept for compatibility)
python FinRA_agent.py --max-age-days 90

# With options
python FinRA_agent.py --max-papers 10 --max-age-days 60 --debug
```

### 🌐 FastAPI Web Interface (Recommended for Deployment)

```bash
# Start the web server
python api.py

# Or with uvicorn directly
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Then open <http://localhost:8000> in your browser.

**Features:**

- Beautiful, responsive UI with Tailwind CSS
- Real-time streaming updates (Server-Sent Events)
- Click to view papers/videos directly
- Ranking by: Citations + Relevance + Recency (papers), Views + Recency (videos)

**API Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/api/research/stream` | GET | Streaming research (SSE) |
| `/metrics` | GET | Prometheus metrics |
| `/health` | GET | Health check |

**Query Parameters for `/api/research/stream`:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `topic` | str | required | Research topic |
| `max_papers` | int | 10 | Max papers to return |
| `max_videos` | int | 10 | Max videos to return |
| `include_videos` | bool | true | Include YouTube search |

**SSE Event Format:**

Each Server-Sent Event has `event: update` with a JSON payload:

```json
{ "type": "paper" | "video" | "status", "data": { ... } }
```

**Event Types:**

| Type | Description | Data Fields |
|------|-------------|-------------|
| `status` | Progress update | `message`, `step`, `total_steps` |
| `paper` | Paper found | `rank`, `title`, `link`, `source`, `citations`, `score`, `summary` |
| `video` | Video found | `rank`, `title`, `link`, `channel`, `views`, `score`, `summary` |
| `done` | Research complete | `papers[]`, `videos[]`, `total_time` |

**Example SSE Stream:**

```text
event: update
data: {"type": "status", "data": {"message": "Planning research...", "step": 1, "total_steps": 5}}

event: update
data: {"type": "paper", "data": {"rank": 1, "title": "Deep Learning for Stock Prediction", "link": "https://arxiv.org/...", "source": "arXiv", "citations": 142, "score": 8.5, "summary": "..."}}

event: update
data: {"type": "video", "data": {"rank": 1, "title": "ML Trading Tutorial", "link": "https://youtube.com/...", "channel": "QuantPy", "views": 50000, "score": 7.2}}

event: update
data: {"type": "done", "data": {"papers": [...], "videos": [...], "total_time": 12.5}}
```

> **Note:** The SSE stream corresponds to `run_multi_agent.py` execution. The same multi-agent pipeline (Planner → Retrievers → Evaluator → Summarizer → Reporter) runs behind the API.

**Interactive API Docs:**

FastAPI automatically exposes interactive docs at `/docs` (Swagger UI) and `/redoc` (ReDoc) — test the API without the UI!

## Output

Both CLI (`run_multi_agent.py`) and API (`api.py`) generate:

1. **Terminal/UI Output**: Formatted display of papers and videos with summaries
2. **JSON Files**:
   - `finance_ai_papers.json` - Papers with metadata and summaries
   - `finance_ai_videos.json` - Videos with metadata and summaries (if enabled)

## How It Works

The system uses the same multi-agent pipeline for both CLI and API:

| Step | Agent | What Happens |
|------|-------|--------------|
| 1 | **Memory** | Recalls similar past queries from ChromaDB |
| 2 | **Planner** | Uses query + memory to decide sources and strategy |
| 3 | **Retrievers** | Fetch papers (arXiv, Scholar, Nature, IEEE) and videos (YouTube) in parallel |
| 4 | **Evaluator** | Ranks by credibility (citations, views, recency) |
| 5 | **Summarizer** | Generates summaries; updates memory for future runs |
| 6 | **Reporter** | Creates final executive summary |

**Sources:**

- **Papers**: arXiv API, Google Scholar, Nature, IEEE (all via Playwright)
- **Videos**: YouTube (Playwright scraping)
- **Web**: Google search (Playwright)

**Filtering:**

- AI/ML + finance relevance detection
- Excludes cryptocurrency content
- Configurable date filter (`--max-age-days`)

**Entry Points:**

- `run_multi_agent.py` — CLI for multi-agent workflow
- `api.py` — FastAPI with streaming UI (same pipeline)
- `FinRA_agent.py` — Legacy single-agent (kept for compatibility)

### Multi-Agent Architecture (LangGraph 0.2.x)

```text
┌─────────────────────────────────────────────────────────────────────┐
│                    FinRA Multi-Agent Pipeline                       │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────┐
│ 1. MEMORY   │ ← Queried FIRST: recalls similar past queries & results
│   Agent     │   (ChromaDB vector store)
└──────┬──────┘
       │ passes context
┌──────▼──────┐
│ 2. PLANNER  │ ← Uses user query + memory context
│   Agent     │   Decides sources, search terms, aggressiveness
│(GPT-4o-mini)│
└──────┬──────┘
       │ dispatches
       ├──────────────────────────────────────┐
       │                                      │
┌──────▼──────┐                        ┌──────▼──────┐
│ 3. PAPER    │  ← Parallel fetch      │ 3. VIDEO    │
│  Retriever  │    arXiv, Scholar,     │  Retriever  │
│ (Playwright)│    Nature, IEEE        │  (YouTube)  │
└──────┬──────┘                        └──────┬──────┘
       │ merges new + recalled                │
       └──────────────┬───────────────────────┘
                      │
               ┌──────▼──────┐
               │ 4. EVALUATOR│ ← Ranks by credibility
               │   Agent     │   Citations, views, recency
               │(GPT-4o-mini)│   Filters low-quality sources
               └──────┬──────┘
                      │
               ┌──────▼──────┐
               │ 5. SUMMARIZER ← Summarizes & ranks results
               │   Agent     │   Updates memory for future runs
               │(GPT-4o-mini)│
               └──────┬──────┘
                      │
               ┌──────▼──────┐
               │ 6. REPORTER │ ← Creates executive summary
               │   Agent     │
               └──────┬──────┘
                      │ writes back
                      ▼
               ┌─────────────┐
               │   MEMORY    │ ← Stores results for future recall
               └─────────────┘
```

> **Note:** All LLM agents use `gpt-4o-mini` by default (configurable via `LLM_MODEL` in `.env`).

**Agent Responsibilities:**

| Step | Agent | What It Does |
|------|-------|--------------|
| 1 | **Memory** | Queried first; recalls similar past queries and results from ChromaDB |
| 2 | **Planner** | Uses user query + memory context to decide sources and search strategy |
| 3 | **Retrievers** | Scrape and fetch new results in parallel; merge with recalled results |
| 4 | **Evaluator** | Ranks by credibility (citations, views, recency); filters low-quality |
| 5 | **Summarizer** | Summarizes and ranks results; **updates memory for future runs** |
| 6 | **Reporter** | Creates final executive summary |

**Key Design Principles:**

- **Memory-First**: Past research informs current searches (MemGPT-inspired retrieval)
- **Context-Aware Planning**: Planner sees both query and memory
- **Parallel Fetching**: Retrievers run concurrently for speed
- **Memory Loop**: Results stored back for future recall

## Key Improvements

- ✅ **Async/Parallel Execution**: 5-10x speed improvement
- ✅ **LLM-Based Planning**: Intelligent research strategy
- ✅ **Multi-Agent System**: Specialized agents with LangGraph
- ✅ **Web Search**: Google via Playwright (FREE, no API key)
- ✅ **arXiv API Integration**: More reliable than web scraping
- ✅ **Date Parsing**: Uses `python-dateutil` for accurate date extraction
- ✅ **Recency Filter**: Configurable age limit (e.g., last 30/60/90 days)
- ✅ **Video Support**: YouTube search and summarization
- ✅ **Type Safety**: Full type hints throughout
- ✅ **Fixed Bugs**: Nature source detection, IEEE selectors

## File Structure

```text
FinRA/
├── api.py                      # FastAPI with streaming UI
├── run_multi_agent.py          # CLI entry point (recommended)
├── FinRA_agent.py              # Legacy single-agent (backup)
├── multi_agent_system/
│   ├── graph.py                # LangGraph orchestration
│   ├── agents.py               # Agent nodes (Memory, Planner, etc.)
│   ├── state.py                # Typed state management
│   ├── config.py               # Configuration & env vars
│   └── production.py           # Retries, rate limiting, metrics
├── tools/
│   ├── search_tools.py         # arXiv API, Google Scholar
│   ├── scraper_tools.py        # Playwright scrapers
│   ├── summarizer_tools.py     # GPT-4o-mini summaries
│   └── ranking.py              # Citation/view-based ranking
├── memory/
│   └── memory_manager.py       # ChromaDB memory
├── requirements.txt
├── README.md
└── .env                        # OPENAI_API_KEY only!
```

## Performance Comparison

| Metric | Before Upgrade | After Upgrade |
|--------|----------------|---------------|
| **Speed** | 45-60s | 8-12s |
| **Execution** | Sequential | Parallel |
| **Planning** | Hard-coded | LLM-based (optional) |
| **Sources** | 3 (papers only) | 6+ (papers + videos + web) |
| **Architecture** | Single script | Multi-agent capable |
| **Scalability** | Limited | High (async) |

**5-10x faster with async mode!** ⚡

## Limitations

- Web scraping (Nature, IEEE) depends on website structure
- YouTube HTML scraping may be less reliable than API
- Summary quality depends on abstract availability
- Rate limits apply to OpenAI and YouTube APIs
- Multi-agent mode requires LangGraph (optional dependency)

## What's Implemented ✅

- ✅ **FastAPI Deployment** - Web UI with streaming
- ✅ **Streaming Responses** - Real-time SSE updates
- ✅ **Vector Database** - ChromaDB for memory
- ✅ **Google Scholar** - Citation-based ranking
- ✅ **YouTube Views** - View-based video ranking

## Production Considerations

### Reliability & Rate Limiting

| Feature | Implementation | Config |
|---------|----------------|--------|
| **Retries** | Exponential backoff for Playwright/httpx | `MAX_RETRIES=3` |
| **Timeouts** | Configurable per-request timeouts | `REQUEST_TIMEOUT=30` |
| **Concurrency** | Semaphore-limited parallel requests | `MAX_CONCURRENT=5` |
| **Rate Limiting** | Delays between requests to avoid blocks | `RATE_LIMIT_DELAY=1.0` |

### Ethical Scraping

⚠️ **Important**: Respect site Terms of Service and `robots.txt`:

- **Google Scholar** - Rate-limited, may block aggressive scraping
- **Google Search** - Use delays, consider SerpAPI for production
- **arXiv** - Official API is preferred (no scraping needed)
- **Nature/IEEE** - Playwright with respectful delays

```python
# Example: Configure in .env
MAX_CONCURRENT=3          # Limit parallel requests
RATE_LIMIT_DELAY=2.0      # Seconds between requests
REQUEST_TIMEOUT=30        # Timeout per request
MAX_RETRIES=3             # Retry failed requests
RESPECT_ROBOTS_TXT=true   # Check robots.txt before scraping
```

### Logging & Monitoring

- **Structured Logging** - Uses Python `logging` with configurable levels
- **Request Tracing** - Logs all HTTP requests with timing
- **Error Tracking** - Captures and reports scraping failures
- **Metrics Endpoint** - `/metrics` for Prometheus (optional)

### Deployment Checklist

```bash
# 1. Set production environment
export ENV=production
export LOG_LEVEL=INFO

# 2. Configure rate limits
export MAX_CONCURRENT=3
export RATE_LIMIT_DELAY=2.0

# 3. Run with uvicorn workers
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4
```

## ☁️ Cloud Deployment (Google Cloud Run)

Deploy FinRA to the cloud with one command:

```bash
# 1. Install gcloud CLI
brew install google-cloud-sdk  # or download from cloud.google.com

# 2. Login and set project
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# 3. Deploy to Cloud Run
gcloud run deploy finra \
    --source . \
    --region us-central1 \
    --allow-unauthenticated \
    --memory 2Gi \
    --timeout 300

# 4. Set your OpenAI API key
gcloud run services update finra \
    --region us-central1 \
    --set-env-vars=OPENAI_API_KEY=your-key-here

# 5. Get your live URL
gcloud run services describe finra --region us-central1 --format="value(status.url)"
```

**Estimated Cost:** $0-5/month for low traffic (Cloud Run free tier).

## 🔮 Future Enhancements

- **Embeddings Search** — Semantic paper similarity
- **PapersWithCode** — Code availability tracking
- **Citation Graph** — Track paper influence
- **Automated Alerts** — Email/Slack notifications
- **Fine-tuned Models** — LoRA adapters for domain specialization

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🤝 Contributing

Contributions welcome! Please read the contributing guidelines before submitting PRs.

---

**Built with ❤️ for the quant finance community**
