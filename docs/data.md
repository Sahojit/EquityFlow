# AlphaAgents — Data Sources and Schemas

## 1. HuggingFace Inference API

**Purpose:** LLM inference for all agent reasoning steps.  
**Endpoint:** `https://api-inference.huggingface.co/v1/` (OpenAI-compatible)  
**Auth:** `HF_TOKEN` (free, no credit card required)  
**Primary model:** `meta-llama/Llama-3.1-8B-Instruct`  
**Fallback model:** `mistralai/Mistral-7B-Instruct-v0.3`  
**Rate limit:** ~10 requests/minute on free tier  
**Input format:** OpenAI chat completions API with `response_format={"type": "json_object"}`  
**Output schema:** JSON validated against agent-specific Pydantic models  

---

## 2. Tavily Search API

**Purpose:** Web search (web researcher agent) and news search (news agent).  
**Auth:** `TAVILY_API_KEY` (free tier, ~1000 searches/month)  
**Web search call:**
```python
client.search(query=..., max_results=5, include_answer=False)
```
**News search call:**
```python
client.search(query=..., max_results=10, search_depth="advanced", topic="news")
```
**Output schema (per result):**
```json
{
  "url": "string",
  "title": "string",
  "content": "string (snippet)",
  "source": "string",
  "published_date": "ISO 8601 string (news only)"
}
```

---

## 3. yfinance (Yahoo Finance)

**Purpose:** Financial metrics for the target company.  
**Auth:** None required (non-commercial public data).  
**Call:** `yf.Ticker(ticker).info` — returns a dict of ~100 fields.  
**Fields used:**

| Field | yfinance key | Type |
|-------|-------------|------|
| Company name | `longName` | str |
| Current price | `currentPrice` / `regularMarketPrice` | float |
| P/E ratio (trailing) | `trailingPE` | float |
| P/B ratio | `priceToBook` | float |
| Return on equity | `returnOnEquity` | float (decimal) |
| Revenue growth YoY | `revenueGrowth` | float (decimal) |
| Debt-to-equity | `debtToEquity` | float |
| Market cap | `marketCap` | float (USD) |

**Fallback:** If `info` is empty or `currentPrice` is None, `data_available=False` is set.

---

## 4. ChromaDB (Local Vector Store)

**Purpose:** Persistent memory of approved research notes.  
**Collection name:** `alpha_agents_memory`  
**Embedding model:** `sentence-transformers/all-MiniLM-L6-v2` (local, 384 dimensions)  
**Storage path:** `CHROMADB_PATH` env var (default `./chroma_db`)  
**Document schema (per stored note):**
```json
{
  "id": "job_id UUID",
  "document": "full markdown text of the research note",
  "metadata": {
    "job_id": "string",
    "ticker": "string",
    "company_name": "string",
    "date_generated": "ISO 8601 string"
  }
}
```

---

## 5. SQLite (Job Metadata)

**Purpose:** Job tracking, status polling, and HITL decision persistence.  
**File path:** `DATABASE_URL` env var (default `./alpha_agents.db`)  
**Table:** `jobs`

| Column | Type | Description |
|--------|------|-------------|
| `job_id` | TEXT PRIMARY KEY | UUID |
| `status` | TEXT | `running`, `awaiting_hitl`, `done`, `error`, `rejected` |
| `created_at` | TEXT | ISO 8601 UTC timestamp |
| `updated_at` | TEXT | ISO 8601 UTC timestamp |
| `state_json` | TEXT | Full serialised `ResearchState` as JSON |

---

## 6. Pydantic Data Models (Internal)

All inter-agent data contracts are defined in `graph/state.py`. Key models:

- **`WebResult`** — url, title, snippet, summary, retrieved_at
- **`FinancialData`** — ticker, current_price, pe_ratio, pb_ratio, roe, revenue_growth_yoy, debt_to_equity, market_cap, data_available
- **`NewsResult`** — headline, source_name, url, published_at, sentiment, summary
- **`CritiqueItem`** — claim, verdict (`supported`|`unsupported`|`missing_citation`), reason
- **`ResearchNote`** — company_name, ticker, analyst, investment_thesis, key_risks, valuation_summary, comparable_companies, recommendation, confidence, citations, full_text
