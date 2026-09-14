"""
Interactive dashboard for the multi-method fake news detection system.

Features:
- Tab 1: Single Article & URL Analyzer (with auto-scraping, visual gauge,
  per-method breakdown, dual explainability LIME + BiLSTM attention, and human feedback submission)
- Tab 2: Live News Feed Monitor (RSS scanning, real-time article scoring, alert stream)
- Tab 3: Model Architecture & Benchmarks (learned meta-ensemble weights and cross-method evaluation metrics)

Run with:
  streamlit run dashboard/app.py
"""
import sys
import os
import csv
import time
from urllib.parse import urlparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import torch
import feedparser
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from src.models.classical_ml import ClassicalEnsemble  # noqa: F401
from src.models.ensemble import MetaEnsemble  # noqa: F401
from src.models.deep_learning import DeepTextClassifier
from src.explainability import Explainer
from src.pipeline import FakeNewsDetector
from utils.scraper import extract_article

st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODELS_DIR = "models"
FEEDBACK_LOG = "data/feedback_log.csv"

PRESETS = {
    "Clickbait / Viral Scam": {
        "text": "SHOCKING: Doctors HATE this one weird natural fruit trick that cures all illnesses overnight! Big Pharma and corrupt officials are trying to ban it before everyone finds out. Share this before it gets permanently censored from the internet!!!",
        "source": "viral-alert24.com",
    },
    "Legitimate News Wire": {
        "text": "The European Central Bank kept interest rates unchanged at its policy meeting on Thursday, reiterating that future decisions will continue to follow a data-dependent approach based on incoming economic indicators and inflation trends.",
        "source": "reuters.com",
    },
    "Conspiracy Narrative": {
        "text": "BREAKING: An anonymous insider from deep within the agency confirms that secret weather modification facilities have been operating without public oversight. Whistleblowers state that documents reveal high-level collusion.",
        "source": "unfiltered-daily.net",
    },
    "Neutral Local Report": {
        "text": "The Department of Transportation announced that routine bridge inspection and maintenance will begin on Monday morning. Motorists should anticipate minor delays during morning commute hours while lane closures are in effect.",
        "source": "dot.gov",
    }
}


@st.cache_resource
def load_detector():
    try:
        classical_model = joblib.load(f"{MODELS_DIR}/classical_ensemble.joblib")
        meta_model = joblib.load(f"{MODELS_DIR}/meta_ensemble.joblib")
        bundle = torch.load(f"{MODELS_DIR}/bilstm.pt", map_location="cpu", weights_only=False)
        deep_model = DeepTextClassifier.load(bundle)
        explainer = Explainer(classical_model)
        return FakeNewsDetector(classical_model, deep_model, meta_model, explainer)
    except Exception as e:
        return e


st.title("🕵️ Multi-Method Fake News Detection System")
st.caption(
    "Calibrated real-time misinformation detection fusing **five heterogeneous methods**: "
    "Classical ML Ensemble, BiLSTM Attention, Stylometric Rules, Source Credibility, and External Fact-Checking."
)

detector = load_detector()
if isinstance(detector, Exception):
    st.error(f"⚠️ Could not load models ({detector}). If training is in progress, please wait a moment and refresh.")
    st.stop()

tab_analyze, tab_feed, tab_benchmarks = st.tabs([
    "🔍 Article & URL Analyzer",
    "📡 Live Feed Monitor",
    "⚖️ Stacked Ensemble & Benchmarks"
])

# ==============================================================================
# TAB 1: ARTICLE & URL ANALYZER
# ==============================================================================
with tab_analyze:
    col_mode, col_preset = st.columns([1, 1])
    with col_mode:
        input_mode = st.radio("Input Source:", ["Paste Text / Claim", "Scrape URL"], horizontal=True)
    with col_preset:
        preset_choice = st.selectbox("Or load a preset example:", ["-- Select Preset --"] + list(PRESETS.keys()))

    target_text = ""
    target_source = ""
    if preset_choice and preset_choice != "-- Select Preset --":
        target_text = PRESETS[preset_choice]["text"]
        target_source = PRESETS[preset_choice]["source"]

    col_input, col_meta = st.columns([2, 1])
    with col_input:
        if input_mode == "Paste Text / Claim":
            input_text = st.text_area(
                "Article Text or Claim:",
                value=target_text,
                height=220,
                placeholder="Paste news headline and full article text here..."
            )
            scraped_meta = None
        else:
            url_input = st.text_input("News Article URL:", placeholder="https://example.com/news/article-headline")
            scrape_btn = st.button("🌐 Fetch & Extract Article")
            if "scraped_data" not in st.session_state:
                st.session_state["scraped_data"] = None

            if scrape_btn and url_input.strip():
                with st.spinner("Fetching and extracting article content..."):
                    try:
                        st.session_state["scraped_data"] = extract_article(url_input.strip())
                        st.success(f"Extracted article: {st.session_state['scraped_data']['title'] or url_input}")
                    except Exception as e:
                        st.error(f"Failed to scrape URL: {e}")
                        st.session_state["scraped_data"] = None

            scraped_meta = st.session_state.get("scraped_data")
            if scraped_meta:
                st.info(f"**Title**: {scraped_meta['title']}\n\n**Extracted Domain**: `{scraped_meta['domain']}`")
                input_text = st.text_area("Extracted Text:", value=scraped_meta["text"], height=160)
            else:
                input_text = ""

    with col_meta:
        default_src = scraped_meta["domain"] if scraped_meta else target_source
        source_input = st.text_input("Source Domain (optional):", value=default_src, placeholder="e.g. reuters.com")
        use_fact_check = st.checkbox("Enable External Fact-Check Lookup", value=True)
        show_explainer = st.checkbox("Compute Explainability (LIME + Attention)", value=True)
        analyze_button = st.button("🚀 Analyze Now", type="primary", use_container_width=True)

    if analyze_button:
        text_to_eval = input_text.strip()
        if not text_to_eval:
            st.warning("Please provide article text or enter a valid URL to analyze.")
        else:
            with st.spinner("Running 5 detection methods and meta-classifier fusion..."):
                t_start = time.time()
                result = detector.predict(
                    text_to_eval,
                    source=source_input.strip(),
                    use_fact_check=use_fact_check,
                    explain=show_explainer
                )
                latency = (time.time() - t_start) * 1000

            verdict = result["verdict"]
            prob = result["fake_probability"]
            conf = result["confidence"]
            color = "#d62728" if verdict == "FAKE" else "#2ca02c"

            st.markdown("---")
            st.markdown(
                f"### Verdict: <span style='color:{color}; font-weight:800;'>{verdict}</span> "
                f"({conf * 100:.1f}% confidence &bull; Latency: {latency:.1f}ms)",
                unsafe_allow_html=True
            )

            # Gauge & Per-method Bar
            g1, g2 = st.columns([1, 1])
            with g1:
                gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    title={"text": "Calibrated Fake Probability (%)"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": color},
                        "steps": [
                            {"range": [0, 35], "color": "#e8f8f5"},
                            {"range": [35, 65], "color": "#fef9e7"},
                            {"range": [65, 100], "color": "#fdedec"}
                        ],
                        "threshold": {
                            "line": {"color": "red", "width": 4},
                            "thickness": 0.75,
                            "value": 50
                        }
                    }
                ))
                gauge.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=10))
                st.plotly_chart(gauge, use_container_width=True)

            with g2:
                breakdown = result["method_breakdown"]
                methods = [
                    "Classical ML Ensemble",
                    "BiLSTM Deep Learning",
                    "Stylometric Anomaly",
                    "Source Reputation Risk",
                    "External Fact-Check"
                ]
                fc_val = 50.0 if not breakdown["fact_check"]["available"] else (
                    95.0 if any("false" in (m.get("rating") or "").lower() for m in breakdown["fact_check"]["matches"]) else 50.0
                )
                vals = [
                    breakdown["classical_ml"]["fake_probability"] * 100,
                    breakdown["deep_learning"]["fake_probability"] * 100,
                    breakdown["linguistic_analysis"]["anomaly_score"] * 100,
                    (1.0 - breakdown["source_credibility"]["score"]) * 100,
                    fc_val
                ]
                bar_fig = go.Figure(go.Bar(
                    x=vals,
                    y=methods,
                    orientation="h",
                    marker_color=["#1f77b4", "#ff7f0e", "#9467bd", "#e377c2", "#17becf"],
                    text=[f"{v:.1f}%" for v in vals],
                    textposition="auto"
                ))
                bar_fig.update_layout(
                    title="Per-Method Disinformation Signals",
                    xaxis=dict(range=[0, 100], title="Fake Signal (%)"),
                    height=260,
                    margin=dict(l=10, r=10, t=40, b=10)
                )
                st.plotly_chart(bar_fig, use_container_width=True)

            # Detailed Breakdown Cards
            st.markdown("#### 🔬 Detailed Method Inspection")
            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown("**1. Classical ML Algorithms**")
                pm = breakdown["classical_ml"]["per_model"]
                clf_df = pd.DataFrame([
                    {"Algorithm": k.replace("_", " ").title(), "Fake Prob": f"{v*100:.1f}%"}
                    for k, v in pm.items()
                ])
                st.dataframe(clf_df, hide_index=True, use_container_width=True)

            with c2:
                st.markdown("**2. Source Credibility**")
                cred = breakdown["source_credibility"]
                st.write(f"Domain: `{cred.get('domain', 'None')}`")
                st.write(f"Trust Tier: **{cred.get('tier', 'unknown').upper()}** (Score: {cred.get('score', 0.5):.2f})")
                for reason in cred.get("reasons", []):
                    st.caption(f"&bull; {reason}")

            with c3:
                st.markdown("**3. Stylometric & Linguistic Red Flags**")
                ling = breakdown["linguistic_analysis"]["features"]
                st.write(f"Clickbait keywords: **{int(ling.get('clickbait_phrase_count', 0))}**")
                st.write(f"ALL-CAPS words: **{ling.get('caps_word_ratio', 0)*100:.1f}%**")
                st.write(f"Exclamation density: **{ling.get('exclamation_ratio', 0):.2f} / sent**")
                st.write(f"Subjectivity score: **{ling.get('sentiment_subjectivity', 0):.2f}**")
                st.write(f"Vague sourcing count: **{int(ling.get('vague_sourcing_count', 0))}**")

            # Explainability Section
            if show_explainer and "explanation" in result and "error" not in result["explanation"]:
                st.markdown("---")
                st.markdown("#### 🔦 Explainability Signals (Dual AI Attribution)")
                e_col1, e_col2 = st.columns(2)
                exp = result["explanation"]

                with e_col1:
                    st.markdown("**LIME Feature Attribution (Classical Ensemble)**")
                    lime_words = exp.get("lime", {}).get("top_words", [])
                    if lime_words:
                        for item in lime_words:
                            w = item["word"]
                            wt = item["weight"]
                            badge = "🔴 FAKE" if wt > 0 else "🟢 REAL"
                            st.write(f"{badge} &bull; `{w}` (impact: `{wt:+.3f}`)")
                    else:
                        st.caption("No significant LIME features detected.")

                with e_col2:
                    st.markdown("**BiLSTM Attention Mechanism Focus**")
                    attn_words = exp.get("attention", {}).get("top_attended_words", [])
                    if attn_words:
                        for token, score in attn_words:
                            st.write(f"🔎 `{token}` &bull; attention weight: `{score:.4f}`")
                    else:
                        st.caption("Attention weights not available.")

            # Human Feedback Logging
            st.markdown("---")
            with st.expander("📝 Human Feedback / Report Correction (Continuous Learning Loop)"):
                st.write("Help improve future model iterations by reporting false positives or false negatives:")
                fb_col1, fb_col2 = st.columns([1, 2])
                with fb_col1:
                    correct_choice = st.selectbox("Actual Label should be:", ["REAL", "FAKE"])
                with fb_col2:
                    fb_notes = st.text_input("Reviewer notes:", placeholder="Reason or missing context...")

                if st.button("Submit Feedback"):
                    os.makedirs(os.path.dirname(FEEDBACK_LOG), exist_ok=True)
                    has_header = os.path.isfile(FEEDBACK_LOG)
                    with open(FEEDBACK_LOG, "a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        if not has_header:
                            writer.writerow(["text", "predicted_verdict", "correct_verdict", "notes", "timestamp"])
                        writer.writerow([text_to_eval, verdict, correct_choice, fb_notes, time.time()])
                    st.success("✅ Correction logged successfully! It will be included in the next training pass.")


# ==============================================================================
# TAB 2: LIVE FEED MONITOR
# ==============================================================================
with tab_feed:
    st.markdown("### 📡 Live RSS Feed Monitoring")
    st.write("Scan online news feeds to detect, score, and stream flagged articles in real time.")

    feed_col1, feed_col2, feed_col3 = st.columns([3, 1, 1])
    with feed_col1:
        rss_url = st.text_input("RSS Feed URL:", value="http://feeds.bbci.co.uk/news/rss.xml")
    with feed_col2:
        alert_thresh = st.slider("Alert Threshold (Fake Prob):", min_value=0.5, max_value=0.95, value=0.65, step=0.05)
    with feed_col3:
        max_entries = st.number_input("Max Articles to Scan:", min_value=3, max_value=25, value=8)

    scan_btn = st.button("⚡ Scan Feed Now", type="primary")

    if scan_btn and rss_url.strip():
        with st.spinner("Fetching and analyzing feed articles..."):
            try:
                parsed = feedparser.parse(rss_url.strip())
                entries = parsed.entries[:int(max_entries)]
                if not entries:
                    st.warning("No articles found in this feed or feed could not be parsed.")
                else:
                    feed_results = []
                    for entry in entries:
                        title = entry.get("title", "")
                        summary = entry.get("summary", "")
                        link = entry.get("link", "")
                        text_content = f"{title}. {summary}" if summary else title
                        src = urlparse(link).netloc.replace("www.", "") if link else ""

                        res = detector.predict(text_content, source=src, use_fact_check=False, explain=False)
                        feed_results.append({
                            "title": title,
                            "link": link,
                            "source": src,
                            "fake_probability": res["fake_probability"],
                            "verdict": res["verdict"],
                            "confidence": res["confidence"]
                        })

                    st.markdown(f"**Scanned {len(feed_results)} articles from `{rss_url}`:**")
                    for item in feed_results:
                        is_fake = item["verdict"] == "FAKE"
                        badge_color = "red" if is_fake else "green"
                        prob_pct = item["fake_probability"] * 100

                        with st.container():
                            c_title, c_score = st.columns([4, 1])
                            with c_title:
                                st.markdown(f"**[{item['title']}]({item['link']})**")
                                st.caption(f"Source: `{item['source']}` &bull; Verdict: :{badge_color}[{item['verdict']}]")
                            with c_score:
                                st.metric("Fake Probability", f"{prob_pct:.1f}%")
                            st.divider()

            except Exception as e:
                st.error(f"Error parsing feed: {e}")


# ==============================================================================
# TAB 3: MODEL ARCHITECTURE & BENCHMARKS
# ==============================================================================
with tab_benchmarks:
    st.markdown("### ⚖️ Meta-Classifier Stacking Weights & Method Comparison")
    st.write(
        "Unlike naive models that rely on a single signal or arbitrary fixed averages, "
        "our system trains a **Logistic Regression Stacking Classifier** on Out-Of-Fold (OOF) "
        "cross-validation predictions to determine mathematically optimal trust weights."
    )

    weights = detector.meta_model.explain_weights()
    w_df = pd.DataFrame([
        {"Method": k.replace("_", " ").title(), "Learned Weight": round(v, 4)}
        for k, v in weights.items()
    ])

    b1, b2 = st.columns([1, 1])
    with b1:
        st.markdown("#### Learned Meta-Ensemble Stacking Weights")
        fig_w = px.bar(
            w_df,
            x="Learned Weight",
            y="Method",
            orientation="h",
            color="Learned Weight",
            color_continuous_scale="Blues",
            title="Logistic Regression Meta-Classifier Trust Weights"
        )
        fig_w.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_w, use_container_width=True)

    with b2:
        st.markdown("#### Method Independence & Complementary Strengths")
        st.markdown(
            """
            | Method | Primary Blind Spot | How Other Methods Compensate |
            | :--- | :--- | :--- |
            | **Classical ML** | Fooled by paraphrasing | BiLSTM captures sequential narrative |
            | **Deep BiLSTM** | Misses publisher credibility | Source reputation flags low-trust domains |
            | **Stylometric** | Sober disinformation passes | Word/char n-grams catch deceptive topics |
            | **Source Credibility** | Unknown/new domains have no history | Content and sequence models analyze text |
            | **Fact-Check API** | Only covers previously debunked claims | Base classifiers handle novel claims |
            """
        )

    # Held-out evaluation metrics
    eval_path = f"{MODELS_DIR}/eval_results.joblib"
    if os.path.exists(eval_path):
        st.markdown("---")
        st.markdown("#### 📈 Held-Out Test Set Performance Benchmarks")
        try:
            eval_data = joblib.load(eval_path)
            metric_rows = []
            for name, metrics in eval_data.items():
                clean_name = name.replace("_", " ").title()
                metric_rows.append({
                    "Detection Method": clean_name,
                    "Accuracy": f"{metrics.get('accuracy', 0)*100:.2f}%",
                    "Precision": f"{metrics.get('precision', 0)*100:.2f}%",
                    "Recall": f"{metrics.get('recall', 0)*100:.2f}%",
                    "F1 Score": f"{metrics.get('f1', 0)*100:.2f}%",
                    "ROC-AUC": f"{metrics.get('auc', 0)*100:.2f}%"
                })
            st.dataframe(pd.DataFrame(metric_rows), hide_index=True, use_container_width=True)
        except Exception as e:
            st.caption(f"Could not load evaluation metrics: {e}")
