"""
app.py — Streamlit UI for the Resume Ranker.
"""
from __future__ import annotations
import streamlit as st
import pandas as pd
from io import BytesIO

import config
from utils import LLM, Embedder
from parsers import parse_job_description, parse_resume
from scorer import rank_candidates


# ── Page Config ──────────────────────────────────────────────────────────

st.set_page_config(page_title="Resume Ranker", page_icon="📄", layout="wide")


# ── Cached Resources ────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading embedding model…")
def get_embedder() -> Embedder:
    return Embedder()


@st.cache_resource(show_spinner="Connecting to Groq…")
def get_llm() -> LLM:
    return LLM()


# ── Header ───────────────────────────────────────────────────────────────

st.title("📄 Resume Ranker")
st.caption("Paste a job description, upload resumes, and get a ranked shortlist.")


# ── Sidebar: Settings ───────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Settings")

    st.subheader("Scoring Weights")
    w_skills = st.slider("Skills", 0.0, 1.0, config.DEFAULT_WEIGHTS["skills"], 0.05)
    w_exp = st.slider("Experience", 0.0, 1.0, config.DEFAULT_WEIGHTS["experience"], 0.05)
    w_sen = st.slider("Seniority", 0.0, 1.0, config.DEFAULT_WEIGHTS["seniority"], 0.05)
    w_edu = st.slider("Education", 0.0, 1.0, config.DEFAULT_WEIGHTS["education"], 0.05)
    w_pref = st.slider("Preferred", 0.0, 1.0, config.DEFAULT_WEIGHTS["preferred"], 0.05)

    weights = {
        "skills": w_skills,
        "experience": w_exp,
        "seniority": w_sen,
        "education": w_edu,
        "preferred": w_pref,
    }

    # Validate weights sum > 0
    if sum(weights.values()) == 0:
        st.warning("All weights are zero — set at least one above 0.")


# ── Main Area: Inputs ───────────────────────────────────────────────────

col_jd, col_resumes = st.columns([1, 1])

with col_jd:
    st.subheader("Job Description")
    jd_text = st.text_area(
        "Paste the full JD here",
        height=300,
        placeholder="e.g. We are looking for a Senior Backend Engineer with 5+ years of experience in Python…",
    )

with col_resumes:
    st.subheader("Resumes")
    uploaded_files = st.file_uploader(
        "Upload resume PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        # Validate count
        if len(uploaded_files) > config.MAX_RESUMES:
            st.error(f"Maximum {config.MAX_RESUMES} resumes allowed. You uploaded {len(uploaded_files)}.")
            uploaded_files = uploaded_files[: config.MAX_RESUMES]

        # Validate sizes
        oversized = [f.name for f in uploaded_files if f.size > config.MAX_FILE_SIZE_MB * 1024 * 1024]
        if oversized:
            st.warning(f"Skipping oversized files (>{config.MAX_FILE_SIZE_MB}MB): {', '.join(oversized)}")
            uploaded_files = [f for f in uploaded_files if f.size <= config.MAX_FILE_SIZE_MB * 1024 * 1024]

        st.info(f"{len(uploaded_files)} resume(s) ready to process.")


# ── Run Button ───────────────────────────────────────────────────────────

if st.button("🚀 Rank Candidates", type="primary", use_container_width=True):
    # Validations
    if not jd_text.strip():
        st.error("Please paste a job description.")
        st.stop()
    if not uploaded_files:
        st.error("Please upload at least one resume.")
        st.stop()
    if sum(weights.values()) == 0:
        st.error("All scoring weights are zero. Adjust in the sidebar.")
        st.stop()

    # Initialize resources
    try:
        llm = get_llm()
    except ValueError as e:
        st.error(str(e))
        st.stop()

    embedder = get_embedder()

    # Step 1: Parse JD
    with st.spinner("Parsing job description…"):
        job = parse_job_description(jd_text, llm)

    st.success(f"**JD Parsed:** {job.title or 'Untitled'} ({job.seniority_level or 'unspecified'} level)")

    # Show parsed JD in an expander
    with st.expander("View parsed JD", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Hard Requirements**")
            st.write(f"Required skills: {', '.join(job.hard.required_skills) or '—'}")
            st.write(f"Min experience: {job.hard.min_experience_years} years")
            st.write(f"Required degree: {job.hard.required_degree or '—'}")
            if job.hard.custom_musts:
                st.write(f"Must-haves: {', '.join(job.hard.custom_musts)}")
        with col2:
            st.markdown("**Preferred Criteria**")
            st.write(f"Preferred skills: {', '.join(job.preferred.preferred_skills) or '—'}")
            st.write(f"Certifications: {', '.join(job.preferred.preferred_certifications) or '—'}")
            if job.preferred.custom_preferences:
                st.write(f"Nice-to-haves: {', '.join(job.preferred.custom_preferences)}")

    # Step 2: Parse resumes
    candidates = []
    parse_progress = st.progress(0, text="Parsing resumes…")
    for i, file in enumerate(uploaded_files):
        parse_progress.progress((i + 1) / len(uploaded_files), text=f"Parsing {file.name}…")
        candidate = parse_resume(file, llm)
        candidates.append(candidate)
    parse_progress.empty()

    unparseable = [c for c in candidates if c.name in ("UNPARSEABLE", "PARSE_ERROR")]
    if unparseable:
        st.warning(f"{len(unparseable)} resume(s) could not be parsed: {', '.join(c.source_file for c in unparseable)}")

    # Step 3: Rank
    with st.spinner("Scoring and ranking candidates…"):
        results = rank_candidates(candidates, job, embedder, weights)

    # ── Results ──────────────────────────────────────────────────────────

    st.divider()
    st.subheader("🏆 Rankings")

    # Summary table
    table_data = []
    for ev in results:
        row = {
            "Rank": ev.rank,
            "Candidate": ev.candidate.name,
            "File": ev.candidate.source_file,
            "Score": round(ev.composite_score, 1),
            "Status": "✅ Passed" if ev.filter_result.passed else "❌ Filtered",
        }
        # Add dimension scores
        for dim in ("skills", "experience", "seniority", "education", "preferred"):
            if dim in ev.scores:
                row[dim.capitalize()] = round(ev.scores[dim].score, 1)
            else:
                row[dim.capitalize()] = "—"
        table_data.append(row)

    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Export to Excel
    excel_buffer = BytesIO()
    df.to_excel(excel_buffer, index=False, engine="openpyxl")
    st.download_button(
        "📥 Download Results (Excel)",
        data=excel_buffer.getvalue(),
        file_name="resume_rankings.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    # Detailed cards
    st.subheader("📋 Detailed Results")
    for ev in results:
        status_icon = "✅" if ev.filter_result.passed else "❌"
        with st.expander(f"{status_icon} #{ev.rank} — {ev.candidate.name} ({ev.candidate.source_file}) — {ev.composite_score:.1f}/100"):
            if not ev.filter_result.passed:
                st.error(f"**Filtered out:** {'; '.join(ev.filter_result.failures)}")
                continue

            # Score bars
            for dim, ds in ev.scores.items():
                st.markdown(f"**{dim.capitalize()}** — {ds.evidence}")
                st.progress(ds.score / 100)
                if ds.matched:
                    st.caption(f"  ✅ Matched: {', '.join(ds.matched)}")
                if ds.missing:
                    st.caption(f"  ❌ Missing: {', '.join(ds.missing)}")

            st.divider()
            col_s, col_g = st.columns(2)
            with col_s:
                st.markdown("**Strengths**")
                for s in ev.strengths:
                    st.write(f"• {s}")
            with col_g:
                st.markdown("**Gaps**")
                for g in ev.gaps:
                    st.write(f"• {g}")
