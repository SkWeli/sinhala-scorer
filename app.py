"""
Sinhala Answer Scorer — Streamlit Entry Point
Run: streamlit run app.py
"""
import sys, os
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
    model_info = st.empty()
    model_info.markdown("""
    <div style="background:#0f3460; border-radius:8px; padding:12px; font-size:13px; color:#a0a0c0;">
        🤖 LLM: <b style="color:#e8e8f0;">llama3.1:8b</b><br>
        📦 Embeddings: <b style="color:#e8e8f0;">nomic-embed-text</b><br>
        🗄️ Vector DB: <b style="color:#e8e8f0;">ChromaDB (local)</b><br>
        🔗 Ontology: <b style="color:#e8e8f0;">OWL 2.0 (owlready2)</b>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#0f3460;'>", unsafe_allow_html=True)
    st.markdown("**📖 Topic**")
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

# Question selector
q_options = {
    f"Q{q['id']}: [{q['topic']}] {q['question_en'][:60]}…": q
    for q in QUESTIONS
}
selected_label = st.selectbox(
    "**📝 Select a Question**",
    list(q_options.keys()),
    help="Choose one of the 5 exam questions"
)
selected_q = q_options[selected_label]

# Show question in both languages
col1, col2 = st.columns([1, 1])
with col1:
    st.markdown(f"""
    <div style="background:#16213e; border-radius:10px; padding:16px 20px;
                border: 1px solid #0f3460; min-height:80px;">
        <div style="color:#8080a0; font-size:12px; margin-bottom:6px;">🇬🇧 English</div>
        <div style="color:#e8e8f0; font-size:15px; line-height:1.6;">{selected_q['question_en']}</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div style="background:#16213e; border-radius:10px; padding:16px 20px;
                border: 1px solid #0f3460; min-height:80px;">
        <div style="color:#8080a0; font-size:12px; margin-bottom:6px;">🇱🇰 සිංහල</div>
        <div style="color:#e8e8f0; font-size:16px; line-height:1.8;">{selected_q['question_si']}</div>
    </div>
    """, unsafe_allow_html=True)

# Show marking guide
with st.expander("📋 View Marking Guide", expanded=False):
    total = 0
    for i, c in enumerate(selected_q["marking_guide"], 1):
        st.markdown(f"""
        <div class="criterion-card">
            <b style="color:#a0c4ff;">{i}. {c['criterion']}</b>
            <span style="float:right; color:#4ade80; font-weight:600;">{c['marks']} marks</span>
        </div>
        """, unsafe_allow_html=True)
        total += c["marks"]
    st.markdown(f"**Total: {total}/20**")

st.markdown("<br>", unsafe_allow_html=True)

# Answer input
st.markdown("**✍️ ඔබේ පිළිතුර ලියන්න (Write your answer in Sinhala)**")
student_answer = st.text_area(
    label="",
    placeholder="මෙහි ඔබේ සිංහල පිළිතුර ටයිප් කරන්න… (Type your Sinhala answer here…)",
    height=220,
    key="student_answer_input",
)

st.markdown("<br>", unsafe_allow_html=True)
submit = st.button("🎯 Score My Answer")

# ── Scoring Pipeline ──────────────────────────────────────────────────────────
if submit:
    if not student_answer.strip():
        st.warning("⚠️ Please enter your answer before submitting.")
    else:
        orchestrator = get_orchestrator()

        # Show live pipeline progress
        st.markdown("---")
        st.markdown("**⚙️ Scoring Pipeline**")
        progress_bar = st.progress(0)
        status_text  = st.empty()

        status_text.markdown("🔍 **Agent 1:** Retrieving relevant passages + ontology concepts…")
        progress_bar.progress(15)

        result = None
        with st.spinner("Running multi-agent scoring pipeline…"):
            try:
                # Patch orchestrator to report progress via session state
                # (full pipeline runs inside score_answer)
                import time

                status_text.markdown("🧠 **Agent 2:** Checking criterion coverage…")
                progress_bar.progress(35)
                time.sleep(0.3)

                status_text.markdown("⚖️ **Agent 3:** LLM scoring each criterion via OLLAMA…")
                progress_bar.progress(55)

                result = orchestrator.score_answer(selected_q, student_answer)

                status_text.markdown("💡 **Agent 4:** Generating explainable feedback…")
                progress_bar.progress(85)
                time.sleep(0.2)

                progress_bar.progress(100)
                status_text.markdown("✅ **Scoring complete!**")

            except Exception as e:
                st.error(f"❌ Error during scoring: {e}")
                st.stop()

        if result:
            st.markdown("---")

            # Score card
            render_score_card(result["final_score"], result["max_score"])

            st.markdown("<br>", unsafe_allow_html=True)

            # Three columns: breakdown | ontology | sources
            left, right = st.columns([3, 2])

            with left:
                render_criterion_breakdown(
                    result["scoring"].get("criterion_scores", [])
                )

            with right:
                # Coverage summary
                coverage = result.get("coverage", {})
                ratio = coverage.get("coverage_ratio", 0)
                st.markdown("### 📊 Coverage Analysis")
                st.markdown(f"""
                <div style="background:#16213e; border-radius:10px; padding:16px 20px;
                            border:1px solid #0f3460;">
                    <div style="color:#a0a0c0; font-size:13px;">Surface criterion coverage</div>
                    <div style="font-size:32px; font-weight:700;
                                color:{'#4ade80' if ratio > 0.6 else '#fbbf24' if ratio > 0.3 else '#f87171'};">
                        {int(ratio*100)}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.progress(ratio)

                st.markdown("<br>", unsafe_allow_html=True)
                render_ontology_concepts(
                    result["retrieval"].get("ontology_concepts", [])
                )

            # RAG evidence
            render_rag_sources(result["retrieval"].get("rag_passages", []))

            # Explanation
            render_explanation(result.get("explanation", ""))

            # Overall comment from LLM
            overall = result["scoring"].get("overall_comment", "")
            if overall:
                st.markdown("### 🗒️ Examiner's Note")
                st.info(overall)

# ── History (session) ─────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

if submit and result:
    st.session_state.history.append({
        "q": selected_q["question_en"][:50],
        "score": result["final_score"],
    })

if st.session_state.history:
    st.markdown("---")
    st.markdown("### 📈 Session Score History")
    import pandas as pd
    df = pd.DataFrame(st.session_state.history)
    df.index += 1
    df.columns = ["Question", "Score /20"]
    st.dataframe(df, use_container_width=True)