# ADR-005: Deployment Strategy — HuggingFace Inference API + Render Free Tier

**Status:** Accepted  
**Date:** 2026-06-22

## Context

The project requires a live public URL for demo purposes, free hosting, no Docker
build pipeline, and no paid LLM API. The internship timeline is 5 weeks; ongoing
hosting cost must be $0.

## Decision

- **LLM:** HuggingFace Inference API free tier (`https://api-inference.huggingface.co/v1/`)
  - Primary: `meta-llama/Llama-3.1-8B-Instruct`
  - Fallback: `mistralai/Mistral-7B-Instruct-v0.3`
  - Free tier: ~10 req/min; sufficient for demo/eval workloads

- **Hosting:** Render free tier — two web services:
  - `alpha-agents-api` → FastAPI (uvicorn), auto-deploys from `main` on push
  - `alpha-agents-ui` → Streamlit, auto-deploys from `main` on push
  - Both configured via `render.yaml`

- **Persistent storage:** SQLite (local file on Render ephemeral disk) + ChromaDB (same)

## Consequences

**Positive:**
- Zero cost during internship period
- GitHub push → Render auto-deploy; no manual CI/CD wiring beyond `render.yaml`
- HuggingFace free tier covers ~100–200 full pipeline runs (sufficient for eval)
- No Docker required — Render builds from `pyproject.toml` + uv

**Negative:**
- **Cold starts:** Render free tier spins down after 15 min inactivity; first request takes ~30s
- **Rate limits:** HuggingFace free tier ~10 req/min; parallel agent calls can hit this
- **Ephemeral disk:** SQLite and ChromaDB are lost on Render service redeploy; use Render Persistent Disk (free tier available) to persist
- **Model quality:** Llama-3.1-8B is significantly below GPT-4 class; JSON compliance is less reliable (handled by fallback + Pydantic validation)

## Alternatives Considered

| Option | Reason rejected |
|--------|----------------|
| **Ollama (local)** | Not publicly deployable; requires user hardware |
| **Anthropic Claude API** | Costs money; violates $0 constraint |
| **Together AI free tier** | HuggingFace has broader model selection and better uptime |
| **Railway / Fly.io** | Similar free tier; Render has simpler `render.yaml` config |
| **Vercel** | Python FastAPI support is limited; not designed for long-running async tasks |
