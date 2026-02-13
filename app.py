import streamlit as st
import pandas as pd
import numpy as np
import eikon as ek
import plotly.express as px
import plotly.graph_objects as go
from transformers import pipeline
import re
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. CORE SETUP & CONCALL.IN UI/UX
# ==========================================
st.set_page_config(page_title="Corporate Narrative Analyzer", layout="wide", initial_sidebar_state="expanded")

# Modern SaaS / Concall.in Light Theme CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    .stApp { background-color: #f3f4f6; color: #1f2937; font-family: 'Inter', sans-serif; }
    h1, h2, h3, h4 { color: #111827; font-weight: 600; }
    
    /* Clean Card UI for Metrics and Visuals */
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    /* Buttons */
    .stButton>button { 
        background-color: #4f46e5; color: white; border-radius: 6px; 
        font-weight: 500; border: none; transition: 0.2s;
    }
    .stButton>button:hover { background-color: #4338ca; color: white; }
    
    /* Tabs (Concall.in Style) */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; background-color: transparent; }
    .stTabs [data-baseweb="tab"] { 
        background-color: #ffffff; border: 1px solid #e5e7eb; border-bottom: none;
        border-radius: 6px 6px 0 0; padding: 10px 20px; color: #6b7280; font-weight: 500;
    }
    .stTabs [aria-selected="true"] { 
        background-color: #f3f4f6; color: #4f46e5; border-top: 3px solid #4f46e5;
    }
    
    /* Dataframes */
    .dataframe { border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. THE AI NLP PIPELINE (CACHED)
# ==========================================
@st.cache_resource(show_spinner=False)
def load_ai_models():
    """Loads Advanced Hugging Face Transformers into memory."""
    return {
        "sentiment": pipeline("sentiment-analysis", model="ProsusAI/finbert"),
        "summarizer": pipeline("summarization", model="facebook/bart-large-cnn"),
        "aspect": pipeline("zero-shot-classification", model="facebook/bart-large-mnli"),
        "ner": pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
    }

# ==========================================
# 3. DATA EXTRACTION (TIME-BOUND)
# ==========================================
@st.cache_data(show_spinner=False)
def fetch_narrative_data(api_key, ticker, days):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=days)
    
    try:
        ek.set_app_key(api_key)
        df_news = ek.get_news_headlines(
            query=f"R:{ticker} AND Language:LEN", 
            date_from=start_date.strftime('%Y-%m-%d'),
            date_to=end_date.strftime('%Y-%m-%d'),
            count=100
        )
        
        transcript_raw = ek.get_data(instruments=[ticker], fields=['TR.EventTranscript'])
        if pd.isna(transcript_raw[0].iloc[0]['EventTranscript']):
            raise ValueError("Transcript restricted.")
            
        return df_news, {"Latest Quarter": str(transcript_raw[0].iloc[0]['EventTranscript'])}

    except Exception as e:
        # --- MOCK FALLBACK ---
        st.warning(f"⚠️ Eikon Premium API Access Offline. Booting Simulation Data for {ticker}.")
        
        mock_headlines = [
            f"{ticker} reports surge in OTT subscriber base, offsetting traditional multiplex drops.",
            f"SEBI regulations delay {ticker}'s proposed network merger by another three months.",
            f"Analysts upgrade {ticker} following aggressive content acquisition strategy.",
            f"Ad-revenue challenges in the mid-cap broadcast space continue to pressure {ticker}.",
            f"Management at {ticker} announces major pivot to direct-to-consumer digital models."
        ] * (days // 5 + 2)
        
        dates = pd.date_range(start=start_date, end=end_date, periods=len(mock_headlines))
        df_news = pd.DataFrame({"versionCreated": dates, "text": mock_headlines}).set_index("versionCreated")
        
        transcripts_4q = {
            "Q4 (Latest)": f"Good morning. {ticker} had a transformative Q4. Our OTT digital platforms saw a 22% surge in active users, directly cannibalizing traditional linear TV and cinema viewership as we predicted. We are aggressively deploying machine learning algorithms to reduce telecom churn and improve targeted ad-spends. This digital transformation will be our primary revenue driver moving into FY27. However, changes in compliance frameworks regarding our pending merger have created short-term friction.",
            "Q3": f"Welcome to the Q3 call. Top-line revenue remained flat this quarter. While our traditional broadcast networks faced ad-spend headwinds, our investments in streaming infrastructure are yielding strong cash flow. We anticipate closing the strategic merger by Q4, which will unlock significant synergies in our content library. We plan to aggressively expand our digital footprint in tier-2 cities.",
            "Q2": f"Q2 was an investment phase for {ticker}. We incurred heavy upfront costs acquiring streaming rights to major sporting events to bolster our OTT offering. We expect these investments to yield high double-digit subscriber growth. We are strictly monitoring our balance sheet and will execute cost-cutting measures in our legacy cinema-distribution arms to fund this digital pivot.",
            "Q1": f"Starting the fiscal year, {ticker} delivered steady Q1 results. Our legacy media assets provided a strong cash baseline. We have initiated a strategic review of our portfolio to address the rapidly changing entertainment landscape. We project that over the next 12 to 18 months, capital allocation will shift heavily away from traditional cinema and into proprietary digital streaming technologies."
        }
        return df_news, transcripts_4q

# ==========================================
# 4. SIDEBAR CONTROLS
# ==========================================
with st.sidebar:
    st.markdown("### 🔑 LSEG Connect")
    eikon_key = st.text_input("Eikon API Key", type="password", value="")
    
    st.markdown("### 📊 Parameters")
    ticker_input = st.text_input("NSE Ticker ID", value="ZEEL.NS").upper()
    horizon = st.selectbox("News Timeline", [7, 30, 90, 180, 365], index=2, format_func=lambda x: f"Last {x} Days")
    
    st.caption("Concall Transcripts are strictly locked to Last 4 Quarters.")
    run_btn = st.button("Analyze Narrative", use_container_width=True)
    
    st.divider()
    with st.expander("🔬 View Quant Methodology"):
        st.markdown("""
        **NLP Architecture:**
        * **Sentiment:** `ProsusAI/finbert` (Trained on 1.8M financial articles).
        * **Strategic Tagging:** `bart-large-mnli` (Zero-shot classification).
        * **Entity Linking:** `bert-base-NER` (Filtered for corporate stopwords).
        * **Summarization:** `bart-large-cnn` (Chunked for token-limit safety).
        """)

# ==========================================
# 5. DASHBOARD EXECUTION & NLP
# ==========================================
if run_btn:
    with st.spinner("Processing NLP Pipelines (FinBERT, BART, NER)..."):
        ai_models = load_ai_models()
        df_news, transcripts_4q = fetch_narrative_data(eikon_key, ticker_input, horizon)

        if df_news.empty:
            st.error(f"No news events found for {ticker_input} in the last {horizon} days. Please expand your timeline.")
            st.stop()

        # ------------------------------------------
        # FEATURE 1: News Analysis
        # ------------------------------------------
        aspect_categories = ["Digital Transformation", "Strategic Mergers", "Regulatory", "Financial Margins"]
        aspect_counts = {k: 0 for k in aspect_categories}
        sentiment_data = []
        entity_sentiment_map = {} 
        
        junk_entities = ['The', 'Board', 'Inc', 'Ltd', 'Company', 'Group', 'Management', 'Limited', 'Corp', 'SEBI']
        
        for index, row in df_news.iterrows():
            headline = str(row['text'])
            date_str = index.strftime('%b %d, %Y') if pd.notnull(index) else "N/A"
            
            sent_result = ai_models['sentiment'](headline, truncation=True, max_length=512)[0]
            label = sent_result['label'].capitalize()
            score_val = sent_result['score'] if label == 'Positive' else (-sent_result['score'] if label == 'Negative' else 0)
            
            aspect_result = ai_models['aspect'](headline, candidate_labels=aspect_categories)
            dominant_aspect = aspect_result['labels'][0]
            aspect_counts[dominant_aspect] += 1
            
            entities = ai_models['ner'](headline)
            for e in entities:
                if e['entity_group'] in ['ORG', 'PER']:
                    word = e['word'].replace("##", "").strip()
                    if len(word) > 2 and word not in junk_entities:
                        if word not in entity_sentiment_map:
                            entity_sentiment_map[word] = []
                        entity_sentiment_map[word].append(score_val)
                
            sentiment_data.append({
                "Date": date_str, "Headline": headline, "Strategic Aspect": dominant_aspect,
                "FinBERT Sentiment": label, "Confidence": f"{sent_result['score']:.2%}"
            })
            
        df_analyzed_news = pd.DataFrame(sentiment_data)
        
        entity_records = []
        for ent, scores in entity_sentiment_map.items():
            if len(scores) >= 2:
                entity_records.append({"Entity": ent, "Mentions": len(scores), "Average Sentiment": np.mean(scores)})
        df_entities = pd.DataFrame(entity_records).sort_values(by="Mentions", ascending=False).head(8)

        # ------------------------------------------
        # FEATURE 2: 4-Quarter Transcript Engine
        # ------------------------------------------
        fls_keywords = ['expect', 'anticipate', 'will', 'future', 'project', 'guidance', 'forecast', 'plan', 'ahead']
        transcript_insights = {}
        
        for quarter, text in transcripts_4q.items():
            sentences = [s.strip() + '.' for s in re.split(r'(?<=[.!?]) +', text) if len(s.split()) > 5]
            fls_sentences = [s for s in sentences if any(k in s.lower() for k in fls_keywords)]
            
            fls_data = []
            for s in fls_sentences:
                res = ai_models['sentiment'](s, truncation=True, max_length=512)[0]
                fls_data.append({"Statement": s, "Sentiment": res['label'].capitalize()})
            
            words = text.split()
            chunks = [' '.join(words[i:i+400]) for i in range(0, len(words), 400)] 
            
            chunk_summaries = []
            for chunk in chunks[:3]:
                out = ai_models['summarizer'](chunk, max_length=60, min_length=20, do_sample=False, truncation=True)
                chunk_summaries.append(out[0]['summary_text'])
                
            bullet_points = [s.strip() + '.' for s in " ".join(chunk_summaries).split('.') if len(s.strip()) > 10][:3]
            
            transcript_insights[quarter] = {"summary": bullet_points, "fls_df": pd.DataFrame(fls_data), "raw_text": text}

    # ==========================================
    # 6. UI RENDERING (CONCALL.IN STYLE)
    # ==========================================
    st.markdown(f"## {ticker_input} Insights & Analysis")
    st.caption(f"News: **Last {horizon} Days** • Transcripts: **Last 4 Quarters**")
    st.write("") 

    # --- SECTION 1: News & Entity Sentiment ---
    st.markdown("#### Entity & Aspect Intelligence")
    
    c1, c2 = st.columns([1, 1.5])
    with c1:
        fig_radar = go.Figure(data=go.Scatterpolar(r=list(aspect_counts.values()), theta=list(aspect_counts.keys()), fill='toself', line_color='#4f46e5'))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=False), bgcolor='#ffffff'), margin=dict(t=20, b=20, l=20, r=20), height=280, template='plotly_white', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_radar, use_container_width=True)
        
    with c2:
        if not df_entities.empty:
            fig_elsa = px.scatter(df_entities, x="Mentions", y="Average Sentiment", text="Entity", size="Mentions", 
                                  color="Average Sentiment", color_continuous_scale=['#ef4444', '#9ca3af', '#10b981'], range_color=[-1, 1])
            fig_elsa.update_traces(textposition='top center', marker=dict(line=dict(width=1, color='DarkSlateGrey')))
            fig_elsa.update_layout(height=280, margin=dict(t=20, b=0, l=0, r=0), template='plotly_white', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_elsa, use_container_width=True)
        else:
            st.info("Insufficient entity data in current news cycle to plot ELSA matrix.")

    st.write("")
    st.divider()

    # --- SECTION 2: 4-Quarter Transcript Engine ---
    st.markdown("#### Earnings Call Transcripts & Forward Guidance")
    
    tabs = st.tabs(list(transcript_insights.keys()))
    
    for i, (quarter, data) in enumerate(transcript_insights.items()):
        with tabs[i]:
            st.write("")
            colA, colB = st.columns([1, 1])
            with colA:
                st.markdown(f"**AI Executive Summary ({quarter}):**")
                for bullet in data["summary"]:
                    st.markdown(f"- <span style='color:#4b5563;'>{bullet}</span>", unsafe_allow_html=True)
                
                with st.expander("📄 View Raw Transcript"):
                    st.write(data["raw_text"])
                    
            with colB:
                st.markdown(f"**Forward-Looking Statements:**")
                if not data["fls_df"].empty:
                    st.dataframe(
                        data["fls_df"].style.applymap(
                            lambda v: 'color: #10b981; font-weight:600' if v == 'Positive' else ('color: #ef4444; font-weight:600' if v == 'Negative' else 'color: #6b7280'), 
                            subset=['Sentiment']
                        ), 
                        use_container_width=True
                    )
                else:
                    st.write("No forward-looking statements detected.")

    st.divider()

    # --- SECTION 3: Live Feed ---
    st.markdown(f"#### Live News Feed Log (Past {horizon} Days)")
    st.dataframe(
        df_analyzed_news.style.applymap(
            lambda v: 'color: #10b981; font-weight:600' if v == 'Positive' else ('color: #ef4444; font-weight:600' if v == 'Negative' else 'color: #6b7280'), 
            subset=['FinBERT Sentiment']
        ), 
        use_container_width=True, 
        height=350
    )
