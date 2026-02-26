# streamlit_brand_instagram_sentiment_with_video_refactor_mystic.py
# Refactored + Fantasy UI + FIXED PLATFORM FILTER

import os
import re
import shutil
import tempfile
from datetime import datetime
from collections import Counter

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# optional heavy imports cached as resources
import instaloader
import easyocr
from transformers import pipeline

# video/audio imports
from moviepy.editor import VideoFileClip
import speech_recognition as sr

# -----------------------------
# Page config (mystic)
# -----------------------------
st.set_page_config(
    page_title="Mystic Brand Oracle",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# High-drama fantasy CSS + particles + glows + animations
# -----------------------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #1a0033 0%, #0d001a 100%);
        color: #e9d5ff;
        overflow: hidden;
    }

    /* Floating particle stars/glows */
    .particles {
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        pointer-events: none;
        background: transparent;
        z-index: -1;
    }
    .particle {
        position: absolute;
        background: white;
        border-radius: 50%;
        opacity: 0.7;
        animation: float 25s infinite linear;
    }
    @keyframes float {
        0%   { transform: translateY(100vh) scale(0.2); opacity: 0; }
        10%  { opacity: 0.8; }
        90%  { opacity: 0.8; }
        100% { transform: translateY(-30vh) scale(0.4); opacity: 0; }
    }

    /* Glowing animated headers */
    h1, h2, h3 {
        color: #d8b4fe;
        text-shadow: 0 0 20px #c084fc, 0 0 40px #a855f7;
        animation: glowPulse 3s ease-in-out infinite alternate;
    }
    @keyframes glowPulse {
        from { text-shadow: 0 0 15px #c084fc, 0 0 30px #a855f7; }
        to   { text-shadow: 0 0 30px #d8b4fe, 0 0 60px #c084fc; }
    }

    /* Magical orb buttons */
    .stButton > button {
        background: linear-gradient(45deg, #a855f7, #ec4899, #f472b6);
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 999px;
        padding: 16px 40px;
        font-size: 18px;
        box-shadow: 0 10px 30px rgba(168,85,247,0.6);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        animation: subtleFloat 4s ease-in-out infinite;
    }
    .stButton > button:hover {
        transform: scale(1.12) rotate(3deg);
        box-shadow: 0 20px 50px rgba(236,72,153,0.7);
        background: linear-gradient(45deg, #c084fc, #f472b6, #ec4899);
    }
    @keyframes subtleFloat {
        0%,100% { transform: translateY(0); }
        50%     { transform: translateY(-8px); }
    }

    /* Inputs with neon border glow */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: #2d0a4e;
        color: #f3e8ff;
        border: 2px solid #a855f7;
        border-radius: 16px;
        transition: all 0.4s;
    }
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #ec4899;
        box-shadow: 0 0 25px #ec4899, inset 0 0 15px rgba(236,72,153,0.3);
    }

    /* Sidebar glow frame */
    section[data-testid="stSidebar"] {
        background: linear-gradient(#2d0a4e, #1a0033) !important;
        border-right: 3px solid #c084fc;
        box-shadow: 5px 0 30px rgba(192,132,252,0.4);
    }

    /* Dataframe / table with hover magic */
    .stDataFrame, table {
        background: #2d0a4e;
        border-radius: 16px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.6);
        transition: transform 0.4s;
    }
    tr:hover {
        background: #4c1d95 !important;
        transform: scale(1.02);
        box-shadow: 0 0 25px #a855f7;
    }

    /* Fade-in + slide-up for content */
    @keyframes appear {
        from { opacity: 0; transform: translateY(40px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    div.block-container > div:first-child {
        animation: appear 1.2s ease-out;
    }
</style>

<div class="particles"></div>
""", unsafe_allow_html=True)

# Add some random floating particles (pure JS injection via st.components)
st.components.v1.html("""
<script>
    (function(){
        const container = document.querySelector('.particles');
        if (!container) return;
        for (let i = 0; i < 60; i++) {
            const p = document.createElement('div');
            p.className = 'particle';
            p.style.width = p.style.height = Math.random() * 4 + 1 + 'px';
            p.style.left = Math.random() * 100 + 'vw';
            p.style.animationDelay = Math.random() * 20 + 's';
            p.style.animationDuration = (Math.random() * 20 + 20) + 's';
            container.appendChild(p);
        }
    })();
</script>
""", height=0)

# -----------------------------
# Persistent session state defaults
# -----------------------------
if 'mode' not in st.session_state:
    st.session_state.mode = 'home'
if 'ocr_text' not in st.session_state:
    st.session_state.ocr_text = ''
if 'video_transcript' not in st.session_state:
    st.session_state.video_transcript = ''
if 'insta_media_text' not in st.session_state:
    st.session_state.insta_media_text = ''

# -----------------------------
# Shared sidebar controls (kept minimal)
# -----------------------------
st.sidebar.header("✨ Mystic Controls")
start_date = st.sidebar.date_input("From", datetime(2024, 1, 1))
end_date   = st.sidebar.date_input("To", datetime.today())
platform = st.sidebar.multiselect(
    "Select Platform / Category",
    options=["All", "Twitter", "Facebook", "Instagram", "Reviews"],
    default=["All"],
)
uploaded_file = st.sidebar.file_uploader("Upload CSV or JSON for Dashboard (must contain a `text` column)", type=["csv", "json"])

# -----------------------------
# Cached resources
# -----------------------------
@st.cache_resource
def load_ocr(gpu=False):
    return easyocr.Reader(["en"], gpu=gpu)
#, model="cardiffnlp/twitter-xlm-roberta-base-sentiment")
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis")

ocr_reader = load_ocr(gpu=False)
sentiment_model = load_sentiment_model()

# -----------------------------
# Utility functions (unchanged behavior, cleaned)
# -----------------------------
def extract_shortcode(url):
    match = re.search(r"instagram\.com/(?:p|reel|tv)/([^/?#&]+)", url)
    return match.group(1) if match else None


def extract_speech_from_video_file(video_path: str) -> str:
    try:
        clip = VideoFileClip(video_path)
    except Exception as e:
        st.warning(f"moviepy failed to open video: {e}")
        return ""

    audio_path = tempfile.mktemp(suffix=".wav")
    try:
        clip.audio.write_audiofile(audio_path, logger=None)
    except Exception as e:
        st.warning(f"Failed to write audio: {e}")
        return ""
    finally:
        try:
            clip.reader.close()
        except Exception:
            pass
        try:
            if clip.audio:
                clip.audio.reader.close_proc()
        except Exception:
            pass

    r = sr.Recognizer()
    transcript = ""
    try:
        with sr.AudioFile(audio_path) as source:
            audio = r.record(source)
        transcript = r.recognize_google(audio)
    except sr.UnknownValueError:
        transcript = ""
    except Exception as e:
        st.warning(f"Speech recognition failed: {e}")
    finally:
        try:
            os.remove(audio_path)
        except Exception:
            pass

    return transcript


def extract_text_from_instagram_media(url):
    shortcode = extract_shortcode(url)
    if not shortcode:
        raise ValueError("Couldn't parse shortcode from the Instagram URL.")

    loader = instaloader.Instaloader(
        download_pictures=True,
        download_videos=True,
        download_comments=False,
        save_metadata=False,
        quiet=True,
    )

    try:
        post = instaloader.Post.from_shortcode(loader.context, shortcode)
        target_folder = f"{shortcode}"
        loader.download_post(post, target=target_folder)

        extracted_text = ""
        for file in os.listdir(target_folder):
            path = os.path.join(target_folder, file)
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                try:
                    text_list = ocr_reader.readtext(path, detail=0)
                    if text_list:
                        extracted_text += " ".join(text_list) + " "
                except Exception as e:
                    st.warning(f"OCR failed on {file}: {e}")
            if file.lower().endswith((".mp4", ".mov", ".mkv")):
                try:
                    speech = extract_speech_from_video_file(path)
                    if speech:
                        extracted_text += speech + " "
                except Exception as e:
                    st.warning(f"Video transcription failed on {file}: {e}")

        shutil.rmtree(target_folder, ignore_errors=True)
        return extracted_text.strip()

    except Exception as e:
        try:
            shutil.rmtree(shortcode, ignore_errors=True)
        except Exception:
            pass
        raise


def extract_speech_from_uploaded_video(uploaded_file) -> str:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[-1]) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        transcript = extract_speech_from_video_file(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
    return transcript

# -----------------------------
# Sentiment helpers
# -----------------------------
@st.cache_data
def analyze_sentiment_batch(texts):
    results = []
    for t in texts:
        try:
            res = sentiment_model(str(t))
            if isinstance(res, list):
                res = res[0]
            results.append(res)
        except Exception:
            results.append({'label': 'NEUTRAL', 'score': 0.0})
    return results

# -----------------------------
# UI: Mode navigation (mystic labels)
# -----------------------------
st.title("🌙 Mystic Brand Oracle ✨")
st.markdown("Choose your path... Use the three big orbs below to open a workspace. Sidebar contains shared filters and file upload for the Dashboard.")

c1, c2, c3 = st.columns([1,1,1])
with c1:
    if st.button("📜 Dashboard Realm"):
        st.session_state.mode = 'dashboard'
with c2:
    if st.button("📸 Vision Divination"):
        st.session_state.mode = 'instagram'
with c3:
    if st.button("🎥 Echo Transcription"):
        st.session_state.mode = 'video'

st.divider()

# -----------------------------
# Mode: Dashboard (Dataset upload + visualizations) — UPDATED with platform filter
# -----------------------------
def show_dashboard():
    st.header("📜 Enchanted Dashboard")

    df = None
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_json(uploaded_file)

            if 'text' not in df.columns:
                st.error("Uploaded file must contain a 'text' column.")
                return

            # Apply platform filtering
            if "All" not in platform:
                # Look for possible column names (case insensitive match)
                possible_platform_cols = ['platform', 'Platform', 'source', 'Source', 'social', 'channel']
                platform_col = None
                for col in possible_platform_cols:
                    if col in df.columns:
                        platform_col = col
                        break

                if platform_col is None:
                    st.warning("No 'platform', 'source', 'social' or 'channel' column found → platform filter ignored.")
                else:
                    # Case-insensitive filtering
                    df['platform_lower'] = df[platform_col].astype(str).str.lower()
                    selected_lower = [p.lower() for p in platform]
                    df = df[df['platform_lower'].isin(selected_lower)].copy()
                    df = df.drop(columns=['platform_lower'])

            # Date filtering
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df = df[(df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))]

            st.success(f"✨ {len(df):,} visions summoned from {uploaded_file.name}")

        except Exception as e:
            st.error(f"Failed to load file: {e}")
            return

    if df is None:
        st.info("Summon a scroll via sidebar")
        return

    # run sentiment if needed
    if 'label' not in df.columns or 'score' not in df.columns:
        with st.spinner("Reading auras..."):
            results = analyze_sentiment_batch(df['text'].astype(str).tolist())
            df['label'] = [r.get('label', 'NEUTRAL') for r in results]
            df['score'] = [r.get('score', 0.0) for r in results]

    st.subheader("Recent Visions")
    st.dataframe(df.head(200))

    st.subheader("🌌 Aura Flow Over Time")
    if 'date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['date']):
        trend_df = df.groupby(df['date'].dt.date).size().reset_index(name='count')
        trend_df['date'] = pd.to_datetime(trend_df['date'])
    else:
        trend_df = pd.DataFrame({'date': pd.date_range(end=datetime.today(), periods=len(df)), 'count': 1}).groupby('date').sum().reset_index()

    trend_df['count'] = trend_df['count'].rolling(window=3, min_periods=1).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trend_df['date'], y=trend_df['count'], mode='lines', line=dict(color='#ec4899', width=4.5, shape='spline')))
    fig.update_layout(
        height=480,
        title={'text':"Aura Flow","font":dict(size=24,color="#d8b4fe"),'x':0.5},
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=50, r=30, t=50, b=40),
        font_color="#e9d5ff",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🔮 Sentiment Orbs")
    pie = px.pie(df, names='label', values='score', hole=0.42,
                 color_discrete_map={'POSITIVE':'#a855f7','NEGATIVE':'#f472b6','NEUTRAL':'#6b7280'})
    pie.update_traces(textposition='inside', textinfo='percent+label', pull=[0.08]*3, rotation=35,
                      marker_line=dict(color='#1a0033',width=3))
    pie.update_layout(
        title={'text':"Sentiment Orbs",'font':dict(size=22,color="#d8b4fe"),'x':0.5},
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font_color="#e9d5ff",
        margin=dict(t=60,b=40),
        transition_duration=700
    )
    st.plotly_chart(pie, use_container_width=True)

    st.subheader("🪄 Power Words")
    def top_words(txts, n=18):
        w = " ".join(txts).lower()
        t = [x for x in re.findall(r'\w+', w) if len(x)>4 and x not in {"which","the","and","for","with","this","that","are","was","from","is","but","of","in","to"}]
        return Counter(t).most_common(n)

    cA, cB = st.columns(2)
    with cA:
        pos = df[df.label=='POSITIVE']
        if not pos.empty:
            d = pd.DataFrame(top_words(pos.text.tolist()), columns=['Word','Power'])
            figp = px.bar(d, x='Power', y='Word', orientation='h', title='Light Words', color_discrete_sequence=['#a855f7'])
            figp.update_layout(transition_duration=600, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                               font_color="#e9d5ff", title_font_color="#d8b4fe")
            st.plotly_chart(figp, use_container_width=True)

    with cB:
        neg = df[df.label=='NEGATIVE']
        if not neg.empty:
            d = pd.DataFrame(top_words(neg.text.tolist()), columns=['Word','Power'])
            fign = px.bar(d, x='Power', y='Word', orientation='h', title='Shadow Words', color_discrete_sequence=['#f472b6'])
            fign.update_layout(transition_duration=600, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                               font_color="#e9d5ff", title_font_color="#d8b4fe")
            st.plotly_chart(fign, use_container_width=True)

    # Summary
    tot = len(df)
    p = round(len(df[df.label=='POSITIVE'])/tot*100,1) if tot else 0
    n = round(len(df[df.label=='NEGATIVE'])/tot*100,1) if tot else 0
    st.markdown(f"*Total Visions* {tot:,}   *Light* {p}%   *Shadow* {n}%")
    if n > 40: st.error("Shadow energy is strong — caution advised")
    if p > 60: st.success("Radiant aura detected — powerful resonance!")

# -----------------------------
# Mode: Instagram Media OCR workspace
# -----------------------------
def show_instagram_media():
    st.header("📸 Instagram Media OCR & Transcription")
    st.markdown("Paste an Instagram post URL and choose to extract *Images*, *Videos*, or *Both*. Results are saved to session state for later analysis.")

    insta_url = st.text_input("Instagram post URL")
    run_images = st.button("Extract images text")
    run_videos = st.button("Extract videos speech")
    run_both = st.button("Extract both (images + video)")

    if run_images or run_videos or run_both:
        if not insta_url.strip():
            st.error("Please paste an Instagram URL first.")
        else:
            mode = 'both' if run_both else ('images' if run_images else 'videos')
            with st.spinner("Downloading and extracting media — this may take a moment"):
                try:
                    combined = ''
                    shortcode = extract_shortcode(insta_url.strip())
                    loader = instaloader.Instaloader(download_pictures=True, download_videos=True, download_comments=False, save_metadata=False, quiet=True)
                    post = instaloader.Post.from_shortcode(loader.context, shortcode)
                    target_folder = f"{shortcode}"
                    loader.download_post(post, target=target_folder)

                    for file in os.listdir(target_folder):
                        path = os.path.join(target_folder, file)
                        if mode in ['both','images'] and file.lower().endswith((".jpg", ".jpeg", ".png")):
                            try:
                                text_list = ocr_reader.readtext(path, detail=0)
                                if text_list:
                                    combined += " ".join(text_list) + " "
                            except Exception as e:
                                st.warning(f"OCR failed on {file}: {e}")
                        if mode in ['both','videos'] and file.lower().endswith((".mp4", ".mov", ".mkv")):
                            try:
                                speech = extract_speech_from_video_file(path)
                                if speech:
                                    combined += speech + " "
                            except Exception as e:
                                st.warning(f"Video transcription failed on {file}: {e}")

                    shutil.rmtree(target_folder, ignore_errors=True)

                    if combined:
                        st.session_state.insta_media_text = combined.strip()
                        st.success("Extraction complete — saved to session state")
                        st.text_area("📎 Extracted media text", st.session_state.insta_media_text, height=200)
                    else:
                        st.info("No extractable text/speech found in that post.")

                except Exception as e:
                    st.error(f"Failed: {e}")

    # quick actions after extraction
    if st.session_state.insta_media_text:
        st.markdown("**Saved Instagram media text available.** Use it below for quick sentiment check or copy to manual input for later batch analysis.")
        if st.button("Analyze saved Instagram text (single)"):
            with st.spinner("Analyzing sentiment..."):
                try:
                    r = sentiment_model(st.session_state.insta_media_text[:512])
                    if isinstance(r, list):
                        r = r[0]
                    label = r.get('label')
                    score = round(r.get('score', 0.0) * 100, 2)
                    st.write(st.session_state.insta_media_text)
                    if label == 'POSITIVE':
                        st.success(f"😊 Positive ({score}%)")
                    elif label == 'NEGATIVE':
                        st.error(f"😡 Negative ({score}%)")
                    else:
                        st.info(f"😐 Neutral ({score}%)")
                except Exception as e:
                    st.error(f"Sentiment analysis failed: {e}")

# -----------------------------
# Mode: Video Transcription workspace
# -----------------------------
def show_video_transcription():
    st.header("🎥 Video Transcription")
    st.markdown("Upload a local video to transcribe speech → text. Result is stored in session state for quick reuse.")

    uploaded_video = st.file_uploader("Upload video (mp4 / mov / mkv)")
    if uploaded_video is not None:
        if st.button("Transcribe uploaded video"):
            with st.spinner("Transcribing — this may take some time depending on file length"):
                try:
                    st.session_state.video_transcript = extract_speech_from_uploaded_video(uploaded_video)
                    if st.session_state.video_transcript:
                        st.success("Transcription saved to session state")
                        st.text_area("🎤 Transcript", st.session_state.video_transcript, height=300)
                    else:
                        st.info("No speech detected or transcription returned empty.")
                except Exception as e:
                    st.error(f"Transcription failed: {e}")

    if st.session_state.video_transcript:
        st.markdown("**Saved transcript available.** You can analyze it or copy into your dataset for batch processing.")
        if st.button("Analyze saved transcript (single)"):
            with st.spinner("Running sentiment on transcript..."):
                try:
                    r = sentiment_model(st.session_state.video_transcript[:512])
                    if isinstance(r, list):
                        r = r[0]
                    label = r.get('label')
                    score = round(r.get('score', 0.0) * 100, 2)
                    st.write(st.session_state.video_transcript)
                    if label == 'POSITIVE':
                        st.success(f"😊 Positive ({score}%)")
                    elif label == 'NEGATIVE':
                        st.error(f"😡 Negative ({score}%)")
                    else:
                        st.info(f"😐 Neutral ({score}%)")
                except Exception as e:
                    st.error(f"Sentiment analysis failed: {e}")

# -----------------------------
# Router: show the selected mode
# -----------------------------
if st.session_state.mode == 'dashboard':
    show_dashboard()
elif st.session_state.mode == 'instagram':
    show_instagram_media()
elif st.session_state.mode == 'video':
    show_video_transcription()
else:
    st.info("Choose your path above to begin.")

st.divider()
st.caption("Forged in starfire • 2025–2026")