"""
Sinhala Answer Scorer — Streamlit Entry Point
Run: streamlit run app.py
"""
import sys
import os
import time
import pandas as pd
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

# ── Page config (MUST be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="ශ්‍රී ලංකා ඉතිහාසය Scorer",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from ui.components import (
    inject_custom_css, render_header, render_score_card,
    render_criterion_breakdown, render_ontology_concepts,
    render_rag_sources, render_explanation,
)
from questions import QUESTIONS
from agents.orchestrator import OrchestratorAgent

# ── Inject CSS ────────────────────────────────────────────────────────────────
inject_custom_css()

# ── Cached orchestrator ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_orchestrator():
    return OrchestratorAgent()

# ── Era flag helper ───────────────────────────────────────────────────────────
ERA_FLAGS = {
    "portuguese": "🇵🇹",
    "dutch":      "🇳🇱",
    "british":    "🇬🇧",
}

def era_flag(era: str) -> str:
    return ERA_FLAGS.get(era.lower(), "🏛️")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:16px 0;">
        <div style="font-size:48px;">🏛️</div>
        <div style="font-weight:600; color:#e8e8f0; font-size:16px; margin-top:8px;">
            Answer Scorer
        </div>
        <div style="color:#8080a0; font-size:12px;">Colonial Sri Lanka</div>
    </div>
    <hr style="border-color:#0f3460; margin:12px 0;">
    """, unsafe_allow_html=True)

    st.markdown("**⚙️ Model Settings**")
    st.markdown("""
    <div style="background:#0f3460; border-radius:8px; padding:12px; font-size:13px; color:#a0a0c0;">
        🤖 LLM: <b style="color:#e8e8f0;">SinGemma via Ollama</b><br>
        📦 Embeddings: <b style="color:#e8e8f0;">nomic-embed-text</b><br>
        🗄️ Vector Store: <b style="color:#e8e8f0;">JSON + cosine similarity</b><br>
        🔗 Ontology: <b style="color:#e8e8f0;">OWL 2.0 (owlready2)</b>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#0f3460;'>", unsafe_allow_html=True)
    st.markdown("**📖 Colonial Periods**")
    st.markdown("""
    <div style="color:#a0a0c0; font-size:13px; line-height:1.8;">
        🇵🇹 Portuguese (1505–1658)<br>
        🇳🇱 Dutch (1658–1796)<br>
        🇬🇧 British (1796–1948)
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#0f3460;'>", unsafe_allow_html=True)

    if st.button("🔄 Rebuild Knowledge Base Index"):
        with st.spinner("Rebuilding vector store…"):
            from rag.rag_pipeline import build_vector_store
            build_vector_store(force_rebuild=True)
        st.success("Index rebuilt!")

    if st.button("🧱 Rebuild Ontology"):
        with st.spinner("Building ontology…"):
            from ontology.ontology_builder import build_ontology
            build_ontology()
        st.success("Ontology rebuilt!")

# ── Main Panel ────────────────────────────────────────────────────────────────
render_header()

# ── Question Selector ─────────────────────────────────────────────────────────
q_options = {
    f"Q{qid}: [{era_flag(q['era'])} {q['era'].capitalize()}] {q['text'][:65]}…": (qid, q)
    for qid, q in QUESTIONS.items()
}

selected_label = st.selectbox(
    "**📝 Select a Question**",
    list(q_options.keys()),
    help="Choose one of the exam questions",
)
selected_id, selected_q = q_options[selected_label]

# Inject id into the question dict so agents can reference it
selected_q = {**selected_q, "id": selected_id}

# ── Question Display ──────────────────────────────────────────────────────────
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown(f"""
    <div style="background:#16213e; border-radius:10px; padding:16px 20px;
                border:1px solid #0f3460; min-height:80px;">
        <div style="color:#8080a0; font-size:12px; margin-bottom:6px;">
            📝 Question {selected_id}
        </div>
        <div style="color:#e8e8f0; font-size:15px; line-height:1.7;">
            {selected_q['text']}
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div style="background:#16213e; border-radius:10px; padding:16px 20px;
                border:1px solid #0f3460; min-height:80px; text-align:center;">
        <div style="color:#8080a0; font-size:12px; margin-bottom:8px;">🏛️ Colonial Era</div>
        <div style="font-size:32px;">{era_flag(selected_q['era'])}</div>
        <div style="color:#e8e8f0; font-size:14px; font-weight:600; margin-top:4px;">
            {selected_q['era'].capitalize()}
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Marking Guide ─────────────────────────────────────────────────────────────
with st.expander("📋 View Marking Guide", expanded=False):
    total = 0
    for i, (criterion, marks) in enumerate(selected_q["guide"].items(), 1):
        st.markdown(f"""
        <div class="criterion-card">
            <b style="color:#a0c4ff;">{i}. {criterion}</b>
            <span style="float:right; color:#4ade80; font-weight:600;">{marks} marks</span>
        </div>
        """, unsafe_allow_html=True)
        total += marks
    st.markdown(
        f"<div style='color:#a0a0c0; margin-top:8px; font-size:13px;'>"
        f"Total allocatable marks: <b style='color:#e8e8f0;'>{total}/20</b></div>",
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── Answer Input ──────────────────────────────────────────────────────────────
student_answer = st.text_area(
    label="✍️ ඔබේ පිළිතුර ලියන්න (Write your answer in Sinhala)",
    label_visibility="visible",
    placeholder="මෙහි ඔබේ සිංහල පිළිතුර ටයිප් කරන්න… (Type your Sinhala answer here…)",
    height=220,
    key="student_answer_input",
)

st.markdown("<br>", unsafe_allow_html=True)
submit = st.button("🎯 Score My Answer")

result = None

if submit:
    if not student_answer.strip():
        st.warning("⚠️ Please enter your answer before submitting.")
    else:
        orchestrator = get_orchestrator()

        st.markdown("---")
        st.markdown("**⚙️ Scoring Pipeline — Running…**")
        st.info("⏳ Scoring your answer offline. Please wait and do NOT refresh.")

        progress_bar = st.progress(20)
        status_text = st.empty()
        status_text.markdown("🔎 **Agent 1:** Retrieving RAG + ontology context…")

        try:
            with st.spinner("Please wait — scoring in progress…"):
                progress_bar.progress(55)
                status_text.markdown("⚖️ **Agents 2–3:** Checking coverage and scoring…")

                result = orchestrator.score_answer(selected_q, student_answer)

                progress_bar.progress(90)
                status_text.markdown("📝 **Agent 4:** Generating explanation…")

                st.session_state["last_result"] = result
                st.session_state["last_question_id"] = selected_id
                st.session_state["last_question_text"] = selected_q["text"]
                st.session_state["last_era"] = selected_q["era"]

                progress_bar.progress(100)

        except Exception as e:
            st.error(f"❌ Scoring failed: {e}")
            st.stop()

if "last_result" in st.session_state:
    result = st.session_state["last_result"]

    st.markdown("---")
    render_score_card(result["final_score"], result["max_score"])
    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([3, 2])

    with left:
        render_criterion_breakdown(result["scoring"].get("criterion_scores", []))

    with right:
        coverage = result.get("coverage", {})
        ratio = coverage.get("coverage_ratio", 0)
        color = "#4ade80" if ratio > 0.6 else "#fbbf24" if ratio > 0.3 else "#f87171"

        st.markdown("### 📊 Coverage Analysis")
        st.markdown(f"""
        <div style="background:#16213e; border-radius:10px; padding:16px 20px;
                    border:1px solid #0f3460;">
            <div style="color:#a0a0c0; font-size:13px;">Surface criterion coverage</div>
            <div style="font-size:32px; font-weight:700; color:{color};">{int(ratio*100)}%</div>
            <div style="color:#8080a0; font-size:12px; margin-top:4px;">
                Keywords found: {', '.join(coverage.get('topic_keywords_found', [])) or 'none'}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.progress(ratio)
        st.markdown("<br>", unsafe_allow_html=True)
        render_ontology_concepts(result["retrieval"].get("ontology_concepts", []))

    render_rag_sources(result["retrieval"].get("rag_passages", []))
    render_explanation(result.get("explanation", ""))

    overall = result["scoring"].get("overall_comment", "")
    if overall:
        st.markdown("### 🗒️ Examiner's Note")
        st.info(overall)

# ── Session Score History ─────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

if submit and result:
    st.session_state.history.append({
        "Question": f"Q{selected_id}: {selected_q['text'][:45]}…",
        "Era":      selected_q["era"].capitalize(),
        "Score /20": result["final_score"],
    })

if st.session_state.history:
    st.markdown("---")
    st.markdown("### 📈 Session Score History")
    df = pd.DataFrame(st.session_state.history)
    df.index += 1
    st.dataframe(df, use_container_width=True)