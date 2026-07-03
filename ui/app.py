"""Streamlit frontend for AlphaAgents — interactive multi-agent equity research."""

from __future__ import annotations

import os
import time

import requests
import streamlit as st

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="AlphaAgents",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    .big-metric {font-size:2rem; font-weight:700; color:#00d4aa;}
    .rec-buy    {background:#1a4731;color:#00d4aa;padding:4px 14px;border-radius:20px;font-weight:700;}
    .rec-hold   {background:#3d3a1a;color:#f5c842;padding:4px 14px;border-radius:20px;font-weight:700;}
    .rec-sell   {background:#4a1a1a;color:#ff6b6b;padding:4px 14px;border-radius:20px;font-weight:700;}
    .rec-nr     {background:#2a2a2a;color:#aaa;padding:4px 14px;border-radius:20px;font-weight:700;}
    .verdict-supported      {color:#00d4aa;font-weight:600;}
    .verdict-unsupported    {color:#ff6b6b;font-weight:600;}
    .verdict-missing_citation{color:#f5c842;font-weight:600;}
    .stage-done  {color:#00d4aa;}
    .stage-run   {color:#f5c842;}
    .stage-wait  {color:#555;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("📈 AlphaAgents")
    st.caption("Agentic Equity Research Analyst")
    st.divider()

    api_ok = False
    try:
        r = requests.get(f"{API_BASE}/research", timeout=3)
        api_ok = r.status_code == 200
    except Exception:
        pass

    status_icon = "🟢" if api_ok else "🔴"
    st.markdown(f"{status_icon} **API** {'connected' if api_ok else 'unreachable'}")
    st.caption(f"`{API_BASE}`")

    st.divider()

    if "active_job_id" in st.session_state:
        st.markdown("**Active Job**")
        job_id = st.session_state["active_job_id"]
        st.code(job_id, language=None)
        try:
            s = requests.get(f"{API_BASE}/research/{job_id}", timeout=3).json()
            status = s.get("status", "unknown")
            colour = {"running": "🟡", "awaiting_hitl": "🟢", "done": "✅", "error": "🔴"}.get(status, "⚪")
            st.markdown(f"{colour} `{status}`")
        except Exception:
            pass

    st.divider()
    st.caption("Made with LangGraph + Groq + Streamlit")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_new, tab_review, tab_history = st.tabs(["🔍 New Research", "📝 Review & Approve", "📚 History"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _post_research(query: str) -> str | None:
    """Submit a research query and return the job_id, or None on failure."""
    try:
        r = requests.post(f"{API_BASE}/research", json={"query": query}, timeout=10)
        r.raise_for_status()
        return r.json()["job_id"]
    except requests.RequestException as exc:
        st.error(f"Failed to submit: {exc}")
        return None


def _get_status(job_id: str) -> dict | None:
    """Poll job status endpoint."""
    try:
        r = requests.get(f"{API_BASE}/research/{job_id}", timeout=5)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None


def _get_state(job_id: str) -> dict | None:
    """Fetch full raw state (draft_note + critique)."""
    try:
        r = requests.get(f"{API_BASE}/research/{job_id}/state", timeout=5)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None


def _post_hitl(job_id: str, decision: str, edited_note: dict | None = None) -> bool:
    """Submit a HITL decision."""
    payload: dict = {"decision": decision}
    if edited_note:
        payload["edited_note"] = edited_note
    try:
        r = requests.post(f"{API_BASE}/research/{job_id}/hitl", json=payload, timeout=10)
        r.raise_for_status()
        return True
    except requests.RequestException as exc:
        st.error(f"HITL failed: {exc}")
        return False


def _rec_badge(rec: str) -> str:
    """Return an HTML badge for a recommendation."""
    css = {
        "Buy": "rec-buy", "Hold": "rec-hold",
        "Sell": "rec-sell", "Not Rated": "rec-nr",
    }.get(rec, "rec-nr")
    return f'<span class="{css}">{rec}</span>'


def _render_pipeline_stages(status_data: dict) -> None:
    """Show animated pipeline stage tracker."""
    status = status_data.get("status", "running")
    rev = status_data.get("revision_count", 0)
    has_draft = status_data.get("has_draft_note", False)
    has_critique = status_data.get("has_critique", False)
    is_done = status in ("awaiting_hitl", "done", "error")

    stages = [
        ("🔭", "Orchestrator",    True),
        ("🌐", "Web Research",    True),
        ("📰", "News Analysis",   True),
        ("💹", "Financial Data",  True),
        ("✍️", "Writing Draft",   has_draft or is_done),
        ("🔍", "Critic Review",   has_critique or is_done),
        ("👤", "Awaiting Review", status in ("awaiting_hitl", "done")),
    ]

    cols = st.columns(len(stages))
    for col, (icon, label, done) in zip(cols, stages):
        with col:
            if done:
                st.markdown(f"<div class='stage-done'>{icon}<br><b>{label}</b><br>✓</div>",
                            unsafe_allow_html=True)
            elif status == "running":
                st.markdown(f"<div class='stage-run'>{icon}<br><b>{label}</b><br>⏳</div>",
                            unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='stage-wait'>{icon}<br>{label}<br>–</div>",
                            unsafe_allow_html=True)

    if rev > 0:
        st.caption(f"Revision passes: {rev}")


# ---------------------------------------------------------------------------
# Tab 1 — New Research
# ---------------------------------------------------------------------------

with tab_new:
    st.header("Submit a Research Query")

    col_input, col_btn = st.columns([5, 1])
    with col_input:
        query = st.text_input(
            "Company or sector",
            placeholder="e.g. Infosys Q1 FY27 outlook, Apple AI strategy 2026",
            label_visibility="collapsed",
        )
    with col_btn:
        run_clicked = st.button("Run ▶", type="primary", use_container_width=True)

    # Quick example chips
    st.caption("Try:")
    chip_cols = st.columns(4)
    examples = ["Infosys FY27", "TCS Q1 2026", "HDFC Bank outlook", "Wipro AI strategy"]
    for col, ex in zip(chip_cols, examples):
        with col:
            if st.button(ex, key=f"chip_{ex}"):
                st.session_state["prefill_query"] = ex
                st.rerun()

    # Apply prefill if set
    if "prefill_query" in st.session_state:
        query = st.session_state.pop("prefill_query")
        st.session_state["_last_query"] = query

    if run_clicked and query.strip():
        with st.spinner("Submitting…"):
            job_id = _post_research(query.strip())
        if job_id:
            st.session_state["active_job_id"] = job_id
            st.session_state["active_query"] = query.strip()
            st.rerun()

    # Live polling block
    if "active_job_id" in st.session_state:
        job_id = st.session_state["active_job_id"]
        query_label = st.session_state.get("active_query", "")

        st.divider()
        st.markdown(f"**Query:** _{query_label}_")
        st.caption(f"Job ID: `{job_id}`")

        status_data = _get_status(job_id)
        if status_data:
            status = status_data.get("status", "unknown")

            _render_pipeline_stages(status_data)
            st.divider()

            if status == "running":
                st.info("⏳ Pipeline running… refreshing automatically.")
                time.sleep(2)
                st.rerun()

            elif status == "awaiting_hitl":
                st.success("✅ Research complete! Review and approve the draft below.")
                st.session_state["review_job_id"] = job_id
                if st.button("→ Go to Review tab", type="primary"):
                    st.session_state["auto_load_review"] = True
                    st.rerun()

            elif status == "error":
                st.error(f"❌ Pipeline error: {status_data.get('error')}")
                if st.button("🔄 Try again"):
                    del st.session_state["active_job_id"]
                    st.rerun()

            elif status in ("done", "rejected"):
                label = "✅ Approved and stored!" if status == "done" else "🚫 Rejected."
                st.info(label)
                if st.button("Start new research"):
                    del st.session_state["active_job_id"]
                    st.rerun()


# ---------------------------------------------------------------------------
# Tab 2 — Review & Approve
# ---------------------------------------------------------------------------

with tab_review:
    st.header("Review Draft Research Note")

    col_jid, col_load = st.columns([5, 1])
    with col_jid:
        default_jid = st.session_state.get("review_job_id", "")
        review_id = st.text_input(
            "Job ID", value=default_jid,
            placeholder="Paste a job ID or complete a query in New Research",
            label_visibility="collapsed",
            key="review_id_input",
        )
    with col_load:
        load_clicked = st.button("Load", type="primary", use_container_width=True)

    # Auto-load when redirected from tab 1
    if st.session_state.pop("auto_load_review", False) and default_jid:
        load_clicked = True
        review_id = default_jid

    if load_clicked and review_id.strip():
        st.session_state["loaded_review_id"] = review_id.strip()

    if "loaded_review_id" in st.session_state:
        rjid = st.session_state["loaded_review_id"]
        status_data = _get_status(rjid)

        if not status_data:
            st.error("Job not found.")
            st.stop()

        status = status_data.get("status", "unknown")

        # Status badge row
        col_s1, col_s2, col_s3 = st.columns(3)
        col_s1.metric("Status", status.upper())
        col_s2.metric("Revisions", status_data.get("revision_count", 0))
        col_s3.metric("Has Critique", "Yes" if status_data.get("has_critique") else "No")

        if status == "running":
            st.info("Pipeline still running. Refresh in a few seconds.")
            st.stop()

        if not status_data.get("has_draft_note"):
            st.warning("No draft note yet.")
            st.stop()

        raw = _get_state(rjid)
        if not raw or not raw.get("draft_note"):
            st.warning("Could not load draft note.")
            st.stop()

        draft = raw["draft_note"]
        critique = raw.get("critique", [])
        rec = draft.get("recommendation", "Not Rated")
        confidence = draft.get("confidence", "–")

        st.divider()

        # Header row
        hcol1, hcol2, hcol3 = st.columns([3, 1, 1])
        with hcol1:
            st.markdown(
                f"## {draft.get('company_name', '')} "
                f"<span style='color:#888;font-size:1.1rem'>({draft.get('ticker', '')})</span>",
                unsafe_allow_html=True,
            )
            st.caption(f"Analyst: {draft.get('analyst', '')} · {draft.get('date_generated', '')[:10]}")
        with hcol2:
            st.markdown(f"### {_rec_badge(rec)}", unsafe_allow_html=True)
        with hcol3:
            st.markdown(f"**Confidence**<br><span style='font-size:1.4rem'>{confidence}</span>",
                        unsafe_allow_html=True)

        st.divider()

        # Main content columns
        note_col, critique_col = st.columns([3, 2])

        with note_col:
            st.subheader("📄 Draft Note")
            with st.container(border=True):
                st.markdown(draft.get("full_text", "_No text._"))

            # Quick facts expander
            with st.expander("📊 Key Facts"):
                st.markdown(f"**Investment Thesis:** {draft.get('investment_thesis','')}")
                st.markdown("**Key Risks:**")
                for r in draft.get("key_risks", []):
                    st.markdown(f"- {r}")
                st.markdown(f"**Valuation:** {draft.get('valuation_summary','')}")
                comps = draft.get("comparable_companies", [])
                if comps:
                    st.markdown(f"**Comparables:** {', '.join(comps)}")

            with st.expander(f"🔗 Citations ({len(draft.get('citations',[]))})"):
                for url in draft.get("citations", []):
                    st.markdown(f"- [{url}]({url})")

        with critique_col:
            st.subheader("🔍 Critic Review")
            if critique:
                verdicts = [c.get("verdict", "") for c in critique]
                n_supported = verdicts.count("supported")
                n_unsupported = verdicts.count("unsupported")
                n_missing = verdicts.count("missing_citation")

                m1, m2, m3 = st.columns(3)
                m1.metric("✅ Supported", n_supported)
                m2.metric("❌ Unsupported", n_unsupported)
                m3.metric("⚠️ Missing Cite", n_missing)

                st.divider()
                for item in critique:
                    v = item.get("verdict", "")
                    css = {
                        "supported": "verdict-supported",
                        "unsupported": "verdict-unsupported",
                        "missing_citation": "verdict-missing_citation",
                    }.get(v, "")
                    icon = {"supported": "✅", "unsupported": "❌", "missing_citation": "⚠️"}.get(v, "•")
                    with st.container(border=True):
                        st.markdown(
                            f"<span class='{css}'>{icon} {v.upper()}</span>",
                            unsafe_allow_html=True,
                        )
                        st.markdown(f"_{item.get('claim', '')}_")
                        st.caption(item.get("reason", ""))
            else:
                st.info("No critique available.")

        # HITL action bar
        st.divider()
        st.subheader("👤 Your Decision")

        if status in ("done", "rejected"):
            already = "✅ Already approved." if status == "done" else "🚫 Already rejected."
            st.info(already)
        else:
            action_cols = st.columns([2, 2, 4])
            with action_cols[0]:
                if st.button("✅ Approve", type="primary", use_container_width=True):
                    if _post_hitl(rjid, "approved"):
                        st.toast("Note approved and stored in memory!", icon="✅")
                        del st.session_state["loaded_review_id"]
                        if "active_job_id" in st.session_state:
                            del st.session_state["active_job_id"]
                        st.rerun()

            with action_cols[1]:
                if st.button("🚫 Reject", use_container_width=True):
                    if _post_hitl(rjid, "rejected"):
                        st.toast("Note rejected.", icon="🚫")
                        del st.session_state["loaded_review_id"]
                        st.rerun()

            with action_cols[2]:
                with st.expander("✏️ Edit then submit"):
                    edited_text = st.text_area(
                        "Edit full_text",
                        value=draft.get("full_text", ""),
                        height=300,
                        key="edit_area",
                    )
                    if st.button("Submit Edit", type="secondary"):
                        edited = {**draft, "full_text": edited_text}
                        if _post_hitl(rjid, "edited", edited):
                            st.toast("Edited note submitted!", icon="✏️")
                            del st.session_state["loaded_review_id"]
                            st.rerun()


# ---------------------------------------------------------------------------
# Tab 3 — History
# ---------------------------------------------------------------------------

with tab_history:
    st.header("📚 Approved Research Notes")

    col_ref, col_blank = st.columns([1, 5])
    with col_ref:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()

    try:
        jobs = requests.get(f"{API_BASE}/research", timeout=5).json()
    except Exception:
        jobs = []

    if not jobs:
        st.info("No approved notes yet. Approve a research note to see it here.")
    else:
        # Build a richer table by fetching states in parallel
        st.caption(f"{len(jobs)} approved note(s)")
        for job in jobs:
            jid = job.get("job_id", "")
            created = job.get("created_at", "")[:10]

            # Try to fetch the note for rich display
            raw = _get_state(jid)
            note = (raw or {}).get("final_note") or (raw or {}).get("draft_note")

            with st.container(border=True):
                if note:
                    rec = note.get("recommendation", "–")
                    h1, h2, h3, h4 = st.columns([3, 1, 1, 1])
                    h1.markdown(
                        f"**{note.get('company_name', jid)}** "
                        f"<span style='color:#888'>({note.get('ticker','')})</span>",
                        unsafe_allow_html=True,
                    )
                    h2.markdown(_rec_badge(rec), unsafe_allow_html=True)
                    h3.markdown(f"<span style='color:#888'>{note.get('confidence','')}</span>",
                                unsafe_allow_html=True)
                    h4.caption(created)

                    detail_cols = st.columns([4, 1])
                    with detail_cols[0]:
                        st.caption(note.get("investment_thesis", "")[:200])
                    with detail_cols[1]:
                        st.link_button("View Note →", f"{API_BASE}/research/{jid}/note",
                                       use_container_width=True)
                else:
                    c1, c2 = st.columns([4, 1])
                    c1.caption(f"Job `{jid}` · {created}")
                    c2.link_button("View →", f"{API_BASE}/research/{jid}/note",
                                   use_container_width=True)
