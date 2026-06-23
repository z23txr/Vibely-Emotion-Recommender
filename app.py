import streamlit as st
import pandas as pd
import os
import html
import unicodedata
import base64
import numpy as np
import cv2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from app.core.emotion_detector import create_emotion_detector

# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Vibely", layout="wide", initial_sidebar_state="collapsed")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def b64(fname):
    p = os.path.join(BASE_DIR, fname)
    if not os.path.exists(p): return ""
    with open(p, "rb") as f: d = base64.b64encode(f.read()).decode()
    mime = "image/webp" if fname.endswith(".webp") else "image/png"
    return f"data:{mime};base64,{d}"

def read_emotion():
    try:
        with open(os.path.join(BASE_DIR,"detected_emotion.txt")) as f:
            return f.read().strip().lower()
    except: return "none"

# ──────────────────────── session state ───────────────────────────────────────
if "page" not in st.session_state: st.session_state.page = "home"
if "emotion" not in st.session_state: st.session_state.emotion = read_emotion()

# ──────────────────────── data ────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_movies():
    df = pd.read_csv(os.path.join(BASE_DIR,"fully_cleaned_movies.csv"), encoding="utf-8")
    df.fillna("", inplace=True)
    def cg(g):
        return [x.strip() for x in unicodedata.normalize("NFKD",str(g)).replace("\xa0"," ").split(",") if x.strip()]
    df["gl"] = df["genre"].apply(cg)
    df["combined"] = df["genre"] + " " + df["overview"]
    return df

@st.cache_data(show_spinner=False)
def get_sim(_df):
    m = TfidfVectorizer(stop_words="english", max_features=12000).fit_transform(_df["combined"])
    return cosine_similarity(m)

EG = {
    "happiness":["Comedy","Adventure","Family","Animation"],
    "happy":    ["Comedy","Adventure","Family","Animation"],
    "sadness":  ["Drama","Romance"], "sad":["Drama","Romance"],
    "anger":    ["Action","Thriller","Crime"], "angry":["Action","Thriller","Crime"],
    "fear":     ["Horror","Mystery","Thriller"],
    "surprise": ["Science Fiction","Fantasy","Adventure"],
    "disgust":  ["Comedy","Drama"], "neutral":[],
}

def pick(df, emotion):
    g = EG.get(emotion.lower(),[])
    if not g: return df.sample(min(20,len(df)))
    low = {x.lower() for x in g}
    res = df[df["gl"].apply(lambda gl: any(x.lower() in low for x in gl))]
    return res.sample(min(20,len(res))) if len(res)>=20 else (res if not res.empty else df.sample(20))

# ──────────────────────── movie grid HTML (Pure CSS Checkbox Modal) ──────────
PH = "https://via.placeholder.com/300x450/0d0d1a/6d28d9?text=No+Image"

def movie_grid(df, pfx="m"):
    if df.empty:
        return "<div style='color:#888;text-align:center;padding:60px;font-family:sans-serif'>No movies found.</div>"

    html_content = ""
    for i, (_, r) in enumerate(df.iterrows()):
        mid = f"{pfx}_{i}"
        nm  = html.escape(str(r.get("name","") or "N/A"))
        gn  = html.escape(str(r.get("genre","") or ""))
        ov  = html.escape(str(r.get("overview","") or "No overview."))
        po  = html.escape(str(r.get("Poster_URL","") or PH))
        rel = html.escape(str(r.get("date_of_release","") or "—"))
        lng = html.escape(str(r.get("language","") or "—"))
        sc  = html.escape(str(r.get("score","")) if r.get("score","") else "N/A")
        gns = (gn[:28]+"...") if len(gn)>28 else gn

        html_content += f"""
        <div class="movie-item">
          <input type="checkbox" id="chk_{mid}" class="modal-toggle">
          
          <label for="chk_{mid}" class="card">
            <div class="thumb">
              <img src="{po}" onerror="this.src='{PH}'" alt="{nm}" loading="lazy">
              <div class="thumb-hover"><span class="eye-btn">View Details</span></div>
            </div>
            <div class="info">
              <div class="nm">{nm}</div>
              <div class="gn">{gns}</div>
              {f'<div class="sc">Score: {sc}</div>' if sc!="N/A" else ""}
            </div>
          </label>

          <div class="modal">
            <label for="chk_{mid}" class="modal-bg"></label>
            <div class="mbox">
              <label for="chk_{mid}" class="xbtn">x</label>
              <div class="mrow">
                <img src="{po}" onerror="this.src='{PH}'" class="mpic" alt="{nm}">
                <div class="mdet">
                  <h2 class="mttl">{nm}</h2>
                  <div class="mline"><span class="lbl">Genre</span><span>{gn or "—"}</span></div>
                  <div class="mline"><span class="lbl">Released</span><span>{rel}</span></div>
                  <div class="mline"><span class="lbl">Language</span><span>{lng}</span></div>
                  <div class="mline"><span class="lbl">Score</span><span>{sc}</span></div>
                  <p class="mov">{ov}</p>
                </div>
              </div>
            </div>
          </div>
        </div>"""

    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Outfit:wght@800;900&display=swap" rel="stylesheet">
<style>
/* scrollbar hidden but scroll works */
html {{ overflow-y: auto; scrollbar-width: none; -ms-overflow-style: none; }}
html::-webkit-scrollbar {{ display: none; }}
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ background: transparent; font-family: 'Inter', sans-serif; }}

.grid {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 20px; padding: 10px 5px 30px; }}
@media (max-width:1200px) {{ .grid {{ grid-template-columns: repeat(4,1fr); }} }}
@media (max-width:900px)  {{ .grid {{ grid-template-columns: repeat(3,1fr); }} }}
@media (max-width:600px)  {{ .grid {{ grid-template-columns: repeat(2,1fr); }} }}

.movie-item {{ display: block; height: 100%; }}
.modal-toggle {{ display: none; }}

.card {{
  display: block; background: rgba(20,20,30,0.6); border: 1px solid rgba(255,255,255,0.08);
  border-radius: 12px; overflow: hidden; cursor: pointer; height: 100%;
  transition: transform .2s, box-shadow .2s, border-color .2s;
}}
.card:hover {{ transform: translateY(-6px); box-shadow: 0 15px 30px rgba(0,0,0,0.8); border-color: rgba(139,92,246,.5); }}
.thumb {{ position: relative; overflow: hidden; }}
.thumb img {{ width: 100%; aspect-ratio: 2/3; object-fit: cover; display: block; transition: transform .3s; }}
.card:hover .thumb img {{ transform: scale(1.04); }}
.thumb-hover {{
  position: absolute; inset: 0; background: rgba(0,0,0,0.6);
  opacity: 0; transition: opacity .2s; display: flex; align-items: center; justify-content: center;
}}
.card:hover .thumb-hover {{ opacity: 1; }}
.eye-btn {{ background: #6d28d9; color: #fff; border-radius: 6px; padding: 8px 16px; font-size: 13px; font-weight: 600; }}
.info {{ padding: 12px; }}
.nm {{ font-size: 14px; font-weight: 700; color: #fff; line-height: 1.3; margin-bottom: 4px; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }}
.gn {{ font-size: 11px; color: #9ca3af; }}
.sc {{ font-size: 12px; color: #fbbf24; font-weight: 700; margin-top: 6px; }}

/* ── Pure CSS Modal ── */
.modal {{
  display: none; position: fixed; inset: 0; z-index: 99999;
  align-items: center; justify-content: center;
}}
.modal-toggle:checked ~ .modal {{ display: flex; }}

.modal-bg {{
  position: absolute; inset: 0; background: rgba(0,0,0,0.85); backdrop-filter: blur(8px);
  cursor: default; display: block;
}}

.mbox {{
  background: #0f0f1a; border: 1px solid #4c1d95; border-radius: 16px;
  width: 90%; max-width: 800px; max-height: 85vh; overflow-y: auto; scrollbar-width: none;
  padding: 30px; position: relative; z-index: 2;
  animation: slideUp 0.2s;
}}
.mbox::-webkit-scrollbar {{ display: none; }}
@keyframes slideUp {{ from {{transform: translateY(20px); opacity:0;}} to {{transform: translateY(0); opacity:1;}} }}

.xbtn {{
  position: absolute; top: 15px; right: 15px; width: 32px; height: 32px; border-radius: 50%;
  background: rgba(255,255,255,0.1); color: #fff; font-size: 16px; font-weight: bold;
  cursor: pointer; display: flex; align-items: center; justify-content: center; transition: background 0.2s;
}}
.xbtn:hover {{ background: #ef4444; }}

.mrow {{ display: flex; gap: 24px; flex-wrap: wrap; }}
.mpic {{ width: 160px; border-radius: 8px; object-fit: cover; flex-shrink: 0; }}
.mdet {{ flex: 1; min-width: 200px; }}
.mttl {{ font-family:'Outfit',sans-serif; font-size: 24px; font-weight: 800; color: #fff; margin-bottom: 12px; }}
.mline {{ display: flex; gap: 10px; margin-bottom: 8px; font-size: 14px; }}
.lbl {{ font-weight: 700; color: #a78bfa; width: 80px; flex-shrink: 0; text-transform: uppercase; font-size: 11px; letter-spacing: 1px; }}
.mline span:last-child {{ color: #e5e7eb; }}
.mov {{ font-size: 14px; color: #d1d5db; line-height: 1.6; margin-top: 16px; border-top: 1px solid rgba(255,255,255,.1); padding-top: 16px; }}
</style></head><body>
<div class="grid">{html_content}</div>
</body></html>"""

# ──────────────────────── global streamlit CSS ────────────────────────────────
GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Outfit:wght@700;900&display=swap');

/* Hide default streamlit padding and elements */
.block-container { padding: 3rem 4rem !important; max-width: 1400px !important; }
header, footer, #MainMenu { display: none !important; }

/* Custom global styles */
body, .stApp { font-family: 'Inter', sans-serif !important; color: #fff; }

/* Clean, visible buttons that don't get cut off */
div.stButton > button {
    background: rgba(30, 30, 45, 0.8) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    color: white !important;
    padding: 10px 24px !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    transition: all 0.2s !important;
    width: 100% !important;
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
}
div.stButton > button:hover {
    background: rgba(255, 255, 255, 0.15) !important;
    border-color: rgba(255, 255, 255, 0.5) !important;
    transform: translateY(-2px);
}

/* Hide scrollbars but allow scrolling */
::-webkit-scrollbar { display: none; }
* { scrollbar-width: none; }

</style>
"""

def apply_bg(img_name, opacity=0.0):
    img_data = b64(img_name)
    overlay = f"rgba(0,0,0,{opacity})" if opacity > 0 else "transparent"
    st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("{img_data}") !important;
        background-size: cover !important;
        background-position: center !important;
        background-attachment: fixed !important;
    }}
    .stApp::before {{
        content: ''; position: fixed; inset: 0; background: {overlay}; pointer-events: none; z-index: 0;
    }}
    </style>
    """, unsafe_allow_html=True)

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  HOME PAGE
# ══════════════════════════════════════════════════════════════════════════════
def page_home():
    apply_bg("bnm.png", opacity=0.3)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family: 'Inter', sans-serif;">
        <h3 style="font-weight: 400; color: #ddd; margin-bottom: 0;">Welcome to</h3>
        <h1 style="font-family: 'Outfit', sans-serif; font-size: 5rem; font-weight: 900; line-height: 1; margin-top: 5px; margin-bottom: 10px;">Vibely</h1>
        <h2 style="font-weight: 600; font-style: italic; color: #ccc; margin-top: 0;">Let's have fun...!</h2>
        <p style="font-size: 1.2rem; color: #aaa; margin-top: 2rem; margin-bottom: 1.5rem;">What would you prefer?</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, _ = st.columns([1, 1, 4])
    with col1:
        if st.button("Search Manually", use_container_width=True):
            st.session_state.page = "search"
            st.rerun()
    with col2:
        if st.button("Detect Emotion", use_container_width=True):
            st.session_state.page = "detecting"
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
#  DETECTING PAGE
# ══════════════════════════════════════════════════════════════════════════════
def page_detecting():
    apply_bg("bnm.png", opacity=0.6)

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family: 'Inter', sans-serif;">
        <h1 style="font-family: 'Outfit', sans-serif; font-size: 4rem; font-weight: 900; margin-bottom: 1rem;">Emotion Detection</h1>
        <p style="font-size: 1.2rem; color: #ccc; line-height: 1.6;">
            Take a quick selfie and the AI will instantly detect your emotion!
        </p>
    </div>
    """, unsafe_allow_html=True)

    picture = st.camera_input("Take a photo")

    if picture is not None:
        with st.spinner("Analyzing your mood..."):
            bytes_data = picture.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            
            model_path = os.path.join(BASE_DIR, "emotion_model.h5")
            detector = create_emotion_detector(model_path)
            
            img_with_emotion, emotion = detector.detect_emotion(cv2_img)
            
            if emotion:
                st.session_state.emotion = emotion.lower()
            else:
                st.session_state.emotion = "none"
            
            st.session_state.page = "results"
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
#  RESULTS PAGE
# ══════════════════════════════════════════════════════════════════════════════
def page_results():
    df = load_movies()
    emotion = st.session_state.emotion

    apply_bg("background.webp", opacity=0.7)

    # All three buttons perfectly aligned in equal-width columns across the screen
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Back to Home", use_container_width=True): st.session_state.page="home"; st.rerun()
    with col2:
        if st.button("Detect Again", use_container_width=True): st.session_state.page="detecting"; st.rerun()
    with col3:
        if st.button("Search Movies", use_container_width=True): st.session_state.page="search"; st.rerun()

    st.markdown(f"""
    <div style="text-align: center; margin: 3rem 0 2rem;">
        <h1 style="font-family: 'Outfit', sans-serif; font-size: 4rem; margin: 0.5rem 0;">{emotion.title()} Mood</h1>
        <p style="color: #aaa; font-size: 1.1rem;">Movies handpicked for your current feeling</p>
    </div>
    """, unsafe_allow_html=True)

    movies = pick(df, emotion)
    # Using iframe so JS works perfectly for the modal!
    st.components.v1.html(movie_grid(movies, "er"), height=750, scrolling=True)

# ══════════════════════════════════════════════════════════════════════════════
#  SEARCH PAGE
# ══════════════════════════════════════════════════════════════════════════════
def page_search():
    df = load_movies()
    apply_bg("bnm.png", opacity=0.6)

    col1, _ = st.columns([1, 7])
    with col1:
        if st.button("Back to Home", use_container_width=True): st.session_state.page="home"; st.rerun()

    st.markdown("""
    <div style="margin: 2rem 0 1rem;">
        <h1 style="font-family: 'Outfit', sans-serif; font-size: 3rem; margin-bottom: 0;">Search Movies</h1>
        <p style="color: #aaa; margin-top: 0.5rem;">Pick any movie and get AI-powered similar recommendations</p>
    </div>
    """, unsafe_allow_html=True)

    names = sorted(df["name"].dropna().unique().tolist())
    
    col_search, col_btn = st.columns([4, 1])
    with col_search:
        sel = st.selectbox("Choose a movie:", names, index=None, placeholder="Type to search...")
    with col_btn:
        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
        go = st.button("Find Similar", use_container_width=True)

    if go and sel:
        with st.spinner("Finding similar movies..."):
            sm = get_sim(df)
        idx = df[df["name"]==sel].index[0]
        scs = sorted(enumerate(sm[idx]), key=lambda x:x[1], reverse=True)[1:21]
        res = df.iloc[[s[0] for s in scs]]

        st.markdown(f"### Similar to <span style='color:#c4b5fd;'>{sel}</span>", unsafe_allow_html=True)
        st.components.v1.html(movie_grid(res, "sr"), height=750, scrolling=True)

    st.markdown("<hr style='border-color: rgba(255,255,255,0.1); margin: 3rem 0;'><p style='color: #aaa; text-align: left;'>OR BROWSE BY GENRE</p>", unsafe_allow_html=True)
    GENRES = ["Action","Comedy","Drama","Horror","Romance","Science Fiction","Thriller","Adventure","Animation","Fantasy"]
    
    # Take up full width matching the search input row above
    chosen = st.selectbox("Select Genre:", GENRES, index=None, placeholder="Select a genre...")
        
    if chosen:
        gl = chosen.lower()
        gdf = df[df["gl"].apply(lambda g: any(x.lower()==gl for x in g))]
        if gdf.empty:
            gdf = df[df["genre"].str.contains(chosen, case=False, na=False)]
        samp = gdf.sample(min(20,len(gdf))) if len(gdf)>20 else gdf
        st.markdown(f"### Top <span style='color:#c4b5fd;'>{chosen}</span> Movies", unsafe_allow_html=True)
        st.components.v1.html(movie_grid(samp, "gb"), height=750, scrolling=True)

# ══════════════════════════════════════════════════════════════════════════════
#  ROUTER
# ══════════════════════════════════════════════════════════════════════════════
p = st.session_state.page
if p == "home": page_home()
elif p == "detecting": page_detecting()
elif p == "results": page_results()
elif p == "search": page_search()
else: st.session_state.page = "home"; st.rerun()
