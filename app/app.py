
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from src.preprocess import preprocess_text
from src.predict import get_topics, get_all_topics, get_model_and_dict, get_corpus
from src.visualize import plot_topics
from src.visualize_2d import plot_topic_space

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="LDA Topic Analyzer",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global styles — modern, clean, light dashboard
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Manrope:wght@600;700;800&display=swap');

    :root {
        --bg: #f7f8fb;
        --surface: #ffffff;
        --border: #e6e8ef;
        --text: #0f172a;
        --muted: #64748b;
        --primary: #4f46e5;
        --primary-soft: #eef2ff;
        --accent: #0ea5e9;
        --success: #10b981;
        --shadow: 0 1px 2px rgba(15,23,42,0.04), 0 8px 24px rgba(15,23,42,0.06);
    }

    .stApp { background: var(--bg); }

    html, body, [class*="css"]  {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: var(--text);
    }

    #MainMenu, footer, header { visibility: hidden; }

    .block-container {
        padding-top: 2.5rem;
        padding-bottom: 4rem;
        max-width: 1200px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: var(--surface);
        border-right: 1px solid var(--border);
    }
    .sidebar-brand {
        display: flex; align-items: center; gap: 10px;
        font-family: 'Manrope', sans-serif; font-weight: 800;
        font-size: 1.05rem; color: var(--text);
        padding: 8px 4px 18px 4px;
        border-bottom: 1px solid var(--border);
        margin-bottom: 16px;
    }
    .sidebar-brand .logo {
        width: 32px; height: 32px; border-radius: 8px;
        background: linear-gradient(135deg, var(--primary), var(--accent));
        display: inline-flex; align-items: center; justify-content: center;
        color: white; font-weight: 800; font-size: 14px;
        box-shadow: var(--shadow);
    }
    .sidebar-section-title {
        font-size: 0.72rem; font-weight: 700;
        letter-spacing: 0.08em; text-transform: uppercase;
        color: var(--muted); margin: 14px 0 8px 0;
    }
    .sidebar-list { list-style: none; padding: 0; margin: 0; font-size: 0.9rem; }
    .sidebar-list li {
        padding: 6px 0; display: flex; gap: 8px; align-items: flex-start;
        color: #334155;
    }
    .sidebar-list li::before {
        content: ""; width: 6px; height: 6px; border-radius: 50%;
        background: var(--primary); margin-top: 8px; flex-shrink: 0;
    }
    .sidebar-tip {
        margin-top: 18px; padding: 12px 14px;
        background: var(--primary-soft); border: 1px solid #e0e7ff;
        border-radius: 10px; color: #3730a3;
        font-size: 0.82rem; line-height: 1.4;
    }

    /* Header */
    .page-header { margin-bottom: 28px; }
    .eyebrow {
        display: inline-flex; align-items: center; gap: 8px;
        font-size: 0.75rem; font-weight: 600;
        letter-spacing: 0.06em; text-transform: uppercase;
        color: var(--primary); background: var(--primary-soft);
        padding: 6px 12px; border-radius: 999px; margin-bottom: 14px;
    }
    .eyebrow .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--primary); }
    .page-title {
        font-family: 'Manrope', sans-serif;
        font-size: 2.4rem; font-weight: 800;
        line-height: 1.15; letter-spacing: -0.02em;
        color: var(--text); margin: 0 0 8px 0;
    }
    .page-subtitle {
        font-size: 1rem; color: var(--muted);
        max-width: 640px; margin: 0;
    }

    /* Sections */
    .section-title {
        display: flex; align-items: center; gap: 10px;
        font-family: 'Manrope', sans-serif; font-weight: 700;
        font-size: 1.05rem; color: var(--text);
        margin: 26px 0 12px 0;
    }
    .section-title .badge {
        font-size: 0.7rem; font-weight: 600;
        color: var(--primary); background: var(--primary-soft);
        padding: 3px 8px; border-radius: 6px;
    }

    /* Text area */
    .stTextArea textarea {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        padding: 14px 16px !important;
        color: var(--text) !important;
        font-size: 0.95rem !important;
        font-family: 'Inter', sans-serif !important;
        box-shadow: var(--shadow);
        transition: border-color .15s ease, box-shadow .15s ease;
    }
    .stTextArea textarea:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 4px rgba(79,70,229,0.12) !important;
        outline: none !important;
    }
    .stTextArea label { display: none; }

    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        border: 1px solid var(--border);
        background: var(--surface);
        color: var(--text);
        font-weight: 600; font-size: 0.92rem;
        padding: 10px 18px; height: 44px;
        transition: all .15s ease;
        box-shadow: var(--shadow);
    }
    .stButton > button:hover {
        border-color: #cbd5e1;
        transform: translateY(-1px);
    }
    div[data-testid="column"]:nth-of-type(1) .stButton > button {
        background: var(--primary); color: #ffffff; border-color: var(--primary);
    }
    div[data-testid="column"]:nth-of-type(1) .stButton > button:hover {
        background: #4338ca; border-color: #4338ca; color: #ffffff;
    }

    /* Main topic card */
    .main-topic {
        background: linear-gradient(135deg, #ffffff 0%, #f5f3ff 100%);
        border: 1px solid #e0e7ff;
        border-radius: 14px;
        padding: 26px 24px;
        box-shadow: var(--shadow);
        height: 100%;
    }
    .main-topic .label-row {
        display: flex; align-items: center; justify-content: space-between;
        margin-bottom: 12px;
    }
    .main-topic .pill {
        font-size: 0.7rem; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.06em;
        color: var(--primary); background: var(--primary-soft);
        padding: 4px 10px; border-radius: 999px;
    }
    .main-topic h2 {
        font-family: 'Manrope', sans-serif;
        font-size: 1.7rem; font-weight: 800;
        line-height: 1.2; color: var(--text);
        margin: 6px 0 16px 0; letter-spacing: -0.01em;
    }
    .confidence-track {
        height: 8px; background: #eef2ff;
        border-radius: 999px; overflow: hidden;
        margin: 8px 0 8px 0;
    }
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, var(--primary), var(--accent));
        border-radius: 999px;
    }
    .confidence-meta {
        display: flex; justify-content: space-between;
        font-size: 0.82rem; color: var(--muted);
    }

    /* Related cards */
    .related-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 18px;
        box-shadow: var(--shadow);
        height: 100%;
        transition: transform .15s ease, border-color .15s ease;
    }
    .related-card:hover {
        transform: translateY(-2px);
        border-color: #c7d2fe;
    }
    .related-card .rank {
        font-size: 0.7rem; font-weight: 700;
        color: var(--muted); letter-spacing: 0.08em;
        text-transform: uppercase; margin-bottom: 10px;
    }
    .related-card .keywords {
        font-family: 'Manrope', sans-serif;
        font-size: 1rem; font-weight: 700;
        color: var(--text); margin-bottom: 10px;
        line-height: 1.3;
    }
    .related-card .prob {
        display: flex; align-items: center; gap: 8px;
        font-size: 0.85rem; color: var(--muted);
    }
    .related-card .prob .num {
        font-weight: 700; color: var(--primary); font-size: 0.95rem;
    }

    /* Empty state */
    .empty-state {
        text-align: center; padding: 48px 20px;
        background: var(--surface);
        border: 1px dashed var(--border);
        border-radius: 14px; color: var(--muted);
    }
    .empty-state h3 {
        color: var(--text);
        font-family: 'Manrope', sans-serif;
        font-weight: 700; margin: 6px 0;
    }

    /* Hide anchor link icon beside headings */
a[href^="#"] {
    display: none !important;
}

/* Extra safety (newer Streamlit versions) */
.css-1b0udgb a, 
.css-1v0mbdj a {
    display: none !important;
}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <span class="logo">LD</span>
            <span>LDA Topic Analyzer</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-section-title">About</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <ul class="sidebar-list">
            <li>Analyze any text using Latent Dirichlet Allocation</li>
            <li>Surfaces the dominant topic and related themes</li>
            <li>Trained on the BBC News dataset</li>
        </ul>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-section-title">How to use</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <ul class="sidebar-list">
            <li>Paste an article or paragraph</li>
            <li>Click <b>Analyze</b> to compute topics</li>
            <li>Review distribution & related themes</li>
        </ul>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-tip">💡 Tip: longer, well-written passages yield more confident topic predictions.</div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div ">
        <span class="eyebrow"><span class="dot"></span> Topic Modeling</span>
        <h1>Implementation of Latent Dirichlet Allocation (LDA) for Topic Modeling and Dimensionality Reduction with Visualization</h1>
        <p class="page-subtitle">Discover the dominant themes in your text and explore how they relate across the BBC topic space.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
st.markdown("""
<div class="section-title">
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
    <path d="M12 20h9" stroke="currentColor" stroke-width="2"/>
    <path d="M16.5 3.5a2.1 2.1 0 1 1 3 3L7 19l-4 1 1-4 12.5-12.5z" stroke="currentColor" stroke-width="2"/>
  </svg>
  Enter Text <span class="badge">Input</span>
</div>
""", unsafe_allow_html=True)

text = st.text_area(
    "Input text",
    height=180,
    placeholder="Paste an article, paragraph, or any block of text you'd like to analyze…",
    label_visibility="collapsed",
)

col1, col2, _ = st.columns([1, 1, 4])

with col1:
    analyze = st.button("🔍 Analyze", use_container_width=True, key="analyze_btn")

with col2:
    clear = st.button("🔄 Reset", use_container_width=True, key="reset_btn")

st.markdown("""
<style>

/* TARGET buttons inside THIS specific row only */
div[data-testid="column"]:nth-of-type(1) div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #4f8cff, #6a5cff) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    height: 44px;
}

div[data-testid="column"]:nth-of-type(1) div[data-testid="stButton"] > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}

/* Reset button */
div[data-testid="column"]:nth-of-type(2) div[data-testid="stButton"] > button {
    background: #f1f3f5 !important;
    color: #333 !important;
    border: 1px solid #ddd !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    height: 44px;
}

div[data-testid="column"]:nth-of-type(2) div[data-testid="stButton"] > button:hover {
    background: #e9ecef !important;
    box-shadow: 0 3px 10px rgba(0,0,0,0.08);
}

</style>
""", unsafe_allow_html=True)
if clear:
    st.rerun()

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
if analyze:
    if not text.strip():
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing topics…"):
            tokens = preprocess_text(text)
            topics = get_topics(tokens)
            all_topics = get_all_topics()
            topics = sorted(topics, key=lambda x: x[1], reverse=True)

        # Distribution + Main topic
        left, right = st.columns([1, 1], gap="large")

        with left:
            st.markdown("""
<div class="section-title">
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
    <rect x="3" y="10" width="4" height="10" stroke="currentColor" stroke-width="2"/>
    <rect x="10" y="4" width="4" height="16" stroke="currentColor" stroke-width="2"/>
    <rect x="17" y="7" width="4" height="13" stroke="currentColor" stroke-width="2"/>
  </svg>
  Topic Distribution
</div>
""", unsafe_allow_html=True)
            st.pyplot(plot_topics(topics, all_topics))

        with right:
            st.markdown("""
<div class="section-title">
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
    <circle cx="12" cy="12" r="8" stroke="currentColor" stroke-width="2"/>
    <circle cx="12" cy="12" r="3" stroke="currentColor" stroke-width="2"/>
  </svg>
  Main Topic
</div>
""", unsafe_allow_html=True)

            main_topic_id, main_prob = topics[0]
            words = all_topics[main_topic_id].split('+')
            words = [w.split('*')[1].replace('"', '') for w in words[:3]]
            main_label = " · ".join(words)
            pct = main_prob * 100

            st.markdown(
                f"""
                <div class="main-topic">
                    <div class="label-row">
                        <span class="pill">Top match</span>
                        <span style="font-size:0.78rem;color:var(--muted);">Topic #{main_topic_id}</span>
                    </div>
                    <h2>{main_label}</h2>
                    <div class="confidence-track">
                        <div class="confidence-fill" style="width:{pct:.1f}%;"></div>
                    </div>
                    <div class="confidence-meta">
                        <span>Confidence</span>
                        <span style="color:var(--text);font-weight:600;">{pct:.1f}%</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Topic Space
        st.markdown("""
<div class="section-title">
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
    <circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="2"/>
    <path d="M2 12h20" stroke="currentColor" stroke-width="2"/>
    <path d="M12 2a15 15 0 0 1 0 20" stroke="currentColor" stroke-width="2"/>
  </svg>
  Topic Space Visualization
</div>
""", unsafe_allow_html=True)
        lda_model, dictionary = get_model_and_dict()
        corpus = get_corpus()
        st.pyplot(plot_topic_space(lda_model, dictionary, corpus, tokens))

        # Related Topics
        st.markdown("""
<div class="section-title">
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
    <path d="M9 12l2 2 4-4" stroke="currentColor" stroke-width="2"/>
    <path d="M21 12c0 4.97-4.03 9-9 9s-9-4.03-9-9 4.03-9 9-9 9 4.03 9 9z" stroke="currentColor" stroke-width="2"/>
  </svg>
  Related Topics
</div>
""", unsafe_allow_html=True)

        related = topics[1:4]
        if related:
            cols = st.columns(len(related), gap="medium")
            for i, (topic_id, prob) in enumerate(related):
                words = all_topics[topic_id].split('+')
                words = [w.split('*')[1].replace('"', '') for w in words[:3]]
                label = " · ".join(words)

                with cols[i]:
                    st.markdown(
                        f"""
                        <div class="related-card">
                            <div class="rank">#{i+2} · Topic {topic_id +1 }</div>
                            <div class="keywords">{label}</div>
                            <div class="prob">
                                <span class="num">{prob*100:.1f}%</span>
                                <span>relevance</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
        else:
            st.markdown(
                '<div class="empty-state"><h3>No related topics</h3><p>Only one dominant topic was detected.</p></div>',
                unsafe_allow_html=True,
            )
else:
    st.markdown(
        """
        <div class="empty-state" style="margin-top:20px;">
            <h3>Ready when you are</h3>
            <p>Paste your text above and click <b>Analyze</b> to see topic distribution, the main theme, and related topics.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
