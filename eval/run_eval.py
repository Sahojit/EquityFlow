"""Evaluation runner for AlphaAgents pipeline.

Loops over test_cases.json, runs each query through the full pipeline
(real API calls — not mocked), saves output to eval/results/, and prints
a summary table.

Usage:
    uv run python eval/run_eval.py
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

RESULTS_DIR = Path(__file__).parent / "results"
TEST_CASES_PATH = Path(__file__).parent / "test_cases.json"


def _load_test_cases() -> list[dict]:
    """Load evaluation test cases from test_cases.json.

    Returns:
        List of test case dicts with query, expected_ticker, expected_sections.

    Raises:
        FileNotFoundError: If test_cases.json does not exist.
    """
    with open(TEST_CASES_PATH) as f:
        return json.load(f)


def _citation_score(note_dict: dict) -> float:
    """Compute citation precision for a serialised ResearchNote dict.

    Args:
        note_dict: Deserialised ResearchNote.

    Returns:
        Float in [0.0, 1.0].
    """
    from eval.citation_precision import citation_precision
    from graph.state import ResearchNote

    try:
        note = ResearchNote.model_validate(note_dict)
        return citation_precision(note)
    except Exception:
        return 0.0


def _check_sections(note_dict: dict, expected_sections: list[str]) -> bool:
    """Check that all expected section keys are non-empty in the note.

    Args:
        note_dict: Deserialised ResearchNote.
        expected_sections: List of field names that must be non-empty.

    Returns:
        True if all sections are present and non-empty.
    """
    for section in expected_sections:
        value = note_dict.get(section)
        if not value or (isinstance(value, list) and len(value) == 0):
            return False
    return True


async def _run_single(query: str, job_id: str) -> dict:
    """Run the pipeline for one query and return the final state.

    Args:
        query: Research query string.
        job_id: UUID for this eval run.

    Returns:
        Final ResearchState dict.
    """
    from graph.pipeline import build_graph
    from graph.state import ResearchState

    pipeline = build_graph()
    initial_state: ResearchState = {
        "query": query,
        "job_id": job_id,
        "revision_count": 0,
        "hitl_decision": None,
        "final_note": None,
        "error": None,
        "langfuse_trace_id": None,
    }
    final_state: ResearchState = await pipeline.ainvoke(initial_state)  # type: ignore[attr-defined]
    return dict(final_state)


def _save_result(job_id: str, state: dict) -> None:
    """Persist a pipeline result to eval/results/{job_id}.json.

    Args:
        job_id: UUID for this eval run.
        state: Final state dict to serialise.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{job_id}.json"

    def _default(obj: object) -> object:
        if hasattr(obj, "model_dump"):
            return obj.model_dump(mode="json")
        if hasattr(obj, "isoformat"):
            return obj.isoformat()  # type: ignore[union-attr]
        raise TypeError(f"Not serialisable: {type(obj)}")

    with open(out_path, "w") as f:
        json.dump(state, f, default=_default, indent=2)


def _print_summary(rows: list[dict]) -> None:
    """Print a formatted summary table to stdout.

    Args:
        rows: List of result dicts with query, ticker, citation_precision,
            sections_ok, time_s, error fields.
    """
    header = f"{'Query':<45} {'Ticker':<12} {'CitPrec':>7} {'Sects':>6} {'Time(s)':>8} {'Error'}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for r in rows:
        query_short = r["query"][:43] + ".." if len(r["query"]) > 45 else r["query"]
        error_short = (r.get("error") or "")[:30]
        print(
            f"{query_short:<45} {r['ticker']:<12} "
            f"{r['citation_precision']:>7.2f} {str(r['sections_ok']):>6} "
            f"{r['time_s']:>8.1f} {error_short}"
        )
    print("=" * len(header))

    avg_cit = sum(r["citation_precision"] for r in rows) / len(rows) if rows else 0
    avg_time = sum(r["time_s"] for r in rows) / len(rows) if rows else 0
    sects_ok = sum(1 for r in rows if r["sections_ok"])
    print(
        f"\nSummary: {len(rows)} queries | "
        f"Avg citation precision: {avg_cit:.2f} | "
        f"Sections OK: {sects_ok}/{len(rows)} | "
        f"Avg time: {avg_time:.1f}s"
    )


async def main() -> None:
    """Run the full evaluation suite and print results."""
    test_cases = _load_test_cases()
    print(f"Loaded {len(test_cases)} test cases.")

    rows: list[dict] = []

    for i, tc in enumerate(test_cases, 1):
        query = tc["query"]
        expected_ticker = tc.get("expected_ticker", "")
        expected_sections = tc.get("expected_sections", [])
        job_id = str(uuid.uuid4())

        print(f"\n[{i}/{len(test_cases)}] Running: {query}")
        start = time.time()

        try:
            state = await _run_single(query, job_id)
            elapsed = time.time() - start

            error = state.get("error")
            note_dict = state.get("draft_note") or state.get("final_note")

            if note_dict and hasattr(note_dict, "model_dump"):
                note_dict = note_dict.model_dump(mode="json")

            cit_prec = _citation_score(note_dict) if note_dict else 0.0
            sects_ok = _check_sections(note_dict, expected_sections) if note_dict else False
            actual_ticker = (note_dict or {}).get("ticker", "")

            _save_result(job_id, state)

            row = {
                "query": query,
                "ticker": actual_ticker or expected_ticker,
                "citation_precision": cit_prec,
                "sections_ok": sects_ok,
                "time_s": elapsed,
                "error": error or "",
            }

        except Exception as exc:
            elapsed = time.time() - start
            print(f"  ERROR: {exc}")
            row = {
                "query": query,
                "ticker": expected_ticker,
                "citation_precision": 0.0,
                "sections_ok": False,
                "time_s": elapsed,
                "error": str(exc)[:80],
            }

        rows.append(row)
        print(
            f"  Done in {row['time_s']:.1f}s | "
            f"CitPrec={row['citation_precision']:.2f} | "
            f"Sections OK={row['sections_ok']}"
        )

    _print_summary(rows)


if __name__ == "__main__":
    asyncio.run(main())
