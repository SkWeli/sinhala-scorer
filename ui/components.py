"""Reusable Streamlit UI components with Claude-like styling."""
import streamlit as st

def inject_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main background */
    .stApp {
        background-color: #1a1a2e;
        color: #e8e8f0;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #16213e;
        border-right: 1px solid #0f3460;
    }

    /* Score card */
    .score-card {
        background: linear-gradient(135deg, #0f3460 0%, #533483 100%);
        border-radius: 16px;
        padding: 28px 32px;
        margin: 16px 0;
        text-align: center;
        box-shadow: 0 8px 32px rgba(83,52,131,0.4);
    }
    .score-big {
        font-size: 72px;
        font-weight: 700;
        color: #e94560;
        line-height: 1;
    }
    .score-label {
        font-size: 16px;
        color: #a0a0c0;
        margin-top: 8px;
    }

    /* Criterion card */
    .criterion-card {
        background: #16213e;
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 16px 20px;
        margin: 8px 0;
        border-left: 4px solid #533483;
    }
    .criterion-awarded { color: #4ade80; font-weight: 600; }
    .criterion-missed  { color: #f87171; font-weight: 600; }
    .criterion-partial { color: #fbbf24; font-weight: 600; }

    /* Chat bubble for explanation */
    .explanation-box {
        background: #0f3460;
        border-radius: 12px 12px 12px 2px;
        padding: 20px 24px;
        margin: 12px 0;
        border: 1px solid #533483;
        line-height: 1.7;
        color: #d0d0e8;
    }

    /* Ontology pill */
    .ontology-pill {
        display: inline-block;
        background: #533483;
        color: #e8e8f0;
        border-radius: 20px;
        padding: 4px 14px;
        margin: 4px;
        font-size: 13px;
    }

    /* Progress bar override */
    .stProgress > div > div {
        background: linear-gradient(90deg, #533483, #e94560);
        border-radius: 8px;
    }

    /* Text area */
    .stTextArea textarea {
        background-color: #16213e !important;
        color: #e8e8f0 !important;
        border: 1px solid #0f3460 !important;
        border-radius: 10px !important;
        font-size: 16px !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #533483, #e94560);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 32px;
        font-size: 16px;
        font-weight: 600;
        width: 100%;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }

    /* Select box */
    .stSelectbox > div > div {
        background: #16213e !important;
        border: 1px solid #0f3460 !important;
        color: #e8e8f0 !important;
    }

    /* Header */
    .app-header {
        background: linear-gradient(135deg, #0f3460 0%, #533483 100%);
        padding: 24px 32px;
        border-radius: 16px;
        margin-bottom: 24px;
        border: 1px solid #533483;
    }
    </style>
    """, unsafe_allow_html=True)


def render_header():
    st.markdown("""
    <div class="app-header">
        <h1 style="color:#e8e8f0; margin:0; font-size:26px;">
            🏛️ ශ්‍රී ලංකා ඉතිහාසය — Intelligent Answer Scorer
        </h1>
        <p style="color:#a0a0c0; margin:6px 0 0;">
            Colonial Sri Lanka (Portuguese → Dutch → British) &nbsp;|&nbsp;
            Powered by OLLAMA + RAG + Ontology + Agent Architecture
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_score_card(score: int, max_score: int = 20):
    pct = int((score / max_score) * 100)
    grade = "A" if pct >= 80 else "B" if pct >= 65 else "C" if pct >= 50 else "D" if pct >= 35 else "F"
    color = "#4ade80" if pct >= 65 else "#fbbf24" if pct >= 50 else "#f87171"
    st.markdown(f"""
    <div class="score-card">
        <div class="score-big" style="color:{color};">{score}</div>
        <div class="score-label">out of {max_score} marks &nbsp;|&nbsp; Grade: <b>{grade}</b> ({pct}%)</div>
    </div>
    """, unsafe_allow_html=True)
    st.progress(score / max_score)


def render_criterion_breakdown(criterion_scores: list):
    st.markdown("### 📋 Mark Breakdown")
    for c in criterion_scores:
        awarded = c.get("awarded", 0)
        max_m   = c.get("max", 0)
        reason  = c.get("reason", "")
        ratio   = awarded / max_m if max_m else 0

        if ratio >= 0.75:
            badge_class = "criterion-awarded"
            icon = "✅"
        elif ratio >= 0.4:
            badge_class = "criterion-partial"
            icon = "⚠️"
        else:
            badge_class = "criterion-missed"
            icon = "❌"

        st.markdown(f"""
        <div class="criterion-card">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div style="font-weight:500; color:#e8e8f0;">{icon} {c['criterion']}</div>
                <div class="{badge_class}">{awarded}/{max_m}</div>
            </div>
            <div style="color:#8080a0; margin-top:8px; font-size:14px;">{reason}</div>
        </div>
        """, unsafe_allow_html=True)


def render_ontology_concepts(concepts: list):
    if not concepts:
        return
    st.markdown("### 🔗 Ontology Concepts Used")
    pills = "".join(f'<span class="ontology-pill">{c}</span>' for c in concepts[:12])
    st.markdown(f'<div style="margin:8px 0;">{pills}</div>', unsafe_allow_html=True)


def render_rag_sources(passages: list):
    if not passages:
        return
    with st.expander("📚 Retrieved Knowledge Base Evidence", expanded=False):
        for i, p in enumerate(passages, 1):
            st.markdown(f"""
            <div style="background:#16213e; border-radius:8px; padding:12px 16px; margin:6px 0;
                        border-left:3px solid #533483;">
                <div style="color:#a0a0c0; font-size:12px; margin-bottom:6px;">
                    📄 Source: <b>{p['source']}</b> &nbsp; Passage #{i}
                </div>
                <div style="color:#c0c0d8; font-size:14px; line-height:1.6;">{p['content'][:300]}…</div>
            </div>
            """, unsafe_allow_html=True)


def render_explanation(explanation: str):
    st.markdown("### 💡 Detailed Feedback")
    st.markdown(f'<div class="explanation-box">{explanation}</div>',
                unsafe_allow_html=True)