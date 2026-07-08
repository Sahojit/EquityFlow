# AlphaAgents — Test Report

> **TODO:** Run `make test` after `uv sync` and paste the full pytest output here.
> Update pass/fail counts, coverage percentage, and any skipped tests.

## Test Suite Summary

| Module | Test File | Tests | Status |
|--------|-----------|-------|--------|
| `llm/client.py` | `tests/test_llm_client.py` | 7 | TODO |
| `agents/orchestrator.py` | `tests/test_orchestrator.py` | 2 | TODO |
| `agents/web_researcher.py` | `tests/test_web_researcher.py` | 3 | TODO |
| `agents/financial_data.py` | `tests/test_financial_data.py` | 3 | TODO |
| `agents/news.py` | `tests/test_news.py` | 3 | TODO |
| `agents/writer.py` | `tests/test_writer.py` | 3 | TODO |
| `agents/critic.py` | `tests/test_critic.py` | 3 | TODO |
| `graph/pipeline.py` | `tests/test_pipeline.py` | 1 | TODO |
| **Total** | | **25** | TODO |

## Coverage Target

Target: ≥ 80% coverage on `llm/`, `agents/`, `graph/`.

## How to Run

```bash
uv sync --all-extras
make test
```
