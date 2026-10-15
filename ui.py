"""Streamlit UI for AI Government Service Navigator. Run: streamlit run ui.py"""
from pathlib import Path
import asyncio
import hashlib
import os
import re
import tempfile

import edge_tts
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

import app as backend

load_dotenv()
st.set_page_config(page_title="Government Service Navigator", page_icon="🏛️",
                   layout="centered", initial_sidebar_state="expanded")

# ------------------------------------------------------------------ styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Mukta:wght@400;600;700&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Mukta', system-ui, sans-serif; }
:root { --ink:#1b2a4a; --saffron:#f28c28; --green:#1b8a55; --teal:#0e8f9c;
        --amber:#b7791f; --red:#c0392b; --line:#dfe5ee; }

/* page + text */
.stApp { background: linear-gradient(180deg,#fff4e6 0%,#f3f8ff 280px,#f6fbf8 100%); }
.block-container { max-width: 880px; padding-top: 1.2rem; }
.stApp, .stApp p, .stApp span, .stApp label, .stApp li, .stApp h1, .stApp h2,
.stApp h3, .stApp h4, [data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"],
[data-testid="stMetricValue"], [data-testid="stMetricLabel"],
[data-testid="stExpander"] summary, [data-testid="stExpander"] p { color:#1b2a4a !important; }
[data-testid="stCaptionContainer"], .field-label { color:#5b6b88 !important; }

/* hero banner */
.hero { background: linear-gradient(115deg,#1b2a4a 0%,#0e8f9c 60%,#1b8a55 100%);
        border-radius:20px; padding:1.5rem 1.8rem; margin-bottom:1.1rem;
        box-shadow:0 10px 26px rgba(27,42,74,.22); border-bottom:6px solid #f28c28; }
.stApp .hero .app-title, .stApp .hero .app-sub { color:#ffffff !important; }
.app-title { font-size:2rem; font-weight:700; margin:0; line-height:1.15; }
.app-sub { font-size:1.05rem; margin:.3rem 0 0 0; opacity:.95; }

/* sidebar */
section[data-testid="stSidebar"] { background: linear-gradient(180deg,#e8f6f7,#fff4e6); border-right:3px solid #f28c28; }
section[data-testid="stSidebar"] * { color:#1b2a4a !important; }

/* tabs */
.stTabs [data-baseweb="tab-list"] { gap:8px; }
.stTabs [data-baseweb="tab"] { background:#ffffff; border:1px solid var(--line); border-radius:12px 12px 0 0;
                               padding:8px 18px; font-weight:600; }
.stTabs [aria-selected="true"] { background:#1b2a4a !important; }
.stTabs [aria-selected="true"] * { color:#ffffff !important; }
.stTabs [data-baseweb="tab-highlight"] { background:#f28c28 !important; height:4px; }

/* buttons */
.stButton > button { border-radius:12px; font-weight:600; border:1px solid #f0c48a; background:#fff8ee; }
.stButton > button:hover { border-color:#f28c28; background:#ffeccf; }
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primaryFormSubmit"] {
    background: linear-gradient(90deg,#f28c28,#e9622b); border:0; box-shadow:0 4px 12px rgba(233,98,43,.35); }
.stButton > button[kind="primary"] *, .stFormSubmitButton > button * { color:#ffffff !important; }

/* answer card + scheme cards */
[data-testid="stVerticalBlockBorderWrapper"] { background:#ffffff; border:1px solid #bfe3e7 !important;
        border-left:6px solid #0e8f9c !important; border-radius:14px; box-shadow:0 4px 14px rgba(14,143,156,.10); }
[data-testid="stExpander"] { background:#ffffff; border:1px solid var(--line); border-left:6px solid #f28c28;
                             border-radius:12px; box-shadow:0 2px 8px rgba(27,42,74,.06); }
[data-testid="stExpander"]:nth-of-type(even) { border-left-color:#1b8a55; }
[data-testid="stExpander"] summary:hover { background:#fff4e6; }

/* metrics */
[data-testid="stMetric"] { background:#ffffff; border:1px solid var(--line); border-radius:14px; padding:12px 16px;
                           border-top:5px solid #1b8a55; box-shadow:0 3px 10px rgba(27,42,74,.07); }
[data-testid="stHorizontalBlock"] > div:nth-child(2) [data-testid="stMetric"] { border-top-color:#f28c28; }
[data-testid="stHorizontalBlock"] > div:nth-child(3) [data-testid="stMetric"] { border-top-color:#c0392b; }

/* badges */
.badge { display:inline-block; padding:3px 12px; border-radius:999px; font-size:.82rem; font-weight:600; margin-right:6px; }
.badge.b-ok   { background:#dcf3e6; color:#13673f !important; }
.badge.b-warn { background:#fdebc8; color:#8a5a0a !important; }
.badge.b-bad  { background:#f9dcd8; color:#a02c20 !important; }
.badge.b-info { background:#dff0fb; color:#17507a !important; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------ resources
DATA_FILE, DB_DIR = backend.DATA_FILE, backend.DB_DIR


@st.cache_data
def load_data_cached(path):
    return backend.load_data(path)


@st.cache_resource
def load_resources_cached(db_dir):
    return backend.load_search_resources(db_dir)


if not DATA_FILE.exists():
    st.error(f"Dataset not found: {DATA_FILE.name}. Place it next to ui.py.")
    st.stop()
if not DB_DIR.exists():
    st.error("Vector database not found. Run semantic_search.py first.")
    st.stop()
try:
    df = load_data_cached(str(DATA_FILE))
    model, collection = load_resources_cached(str(DB_DIR))
except Exception as exc:
    st.error(f"Could not load search resources ({type(exc).__name__}: {exc}).")
    st.stop()
if collection.count() != len(df):
    st.error("CSV and vector database have different record counts. Rebuild the vector database.")
    st.stop()
backend.set_resources(df, model, collection)

clean, safe_url = backend.clean, backend.safe_url
STATUS_MATCH, STATUS_VERIFY, STATUS_MISMATCH = backend.STATUS_MATCH, backend.STATUS_VERIFY, backend.STATUS_MISMATCH

LANG_CODES = {"English": "en", "Hindi": "hi", "Marathi": "mr"}
VOICES = {"English": "en-IN-NeerjaNeural", "Hindi": "hi-IN-SwaraNeural", "Marathi": "mr-IN-AarohiNeural"}
EXAMPLES = {
    "🎓 Student scholarship": "I am a student from Maharashtra looking for scholarship assistance.",
    "🌾 Farmer subsidy": "I am a farmer from Maharashtra looking for a drip irrigation subsidy.",
    "💼 Start a business": "I want to start a small business and need a loan or subsidy.",
    "♿ Disability support": "I have a disability and need financial or education support.",
}

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    voice_language = st.selectbox("Language you will speak", list(LANG_CODES), key="voice_language",
                                  help="Pick this before recording a voice question.")
    top_k = st.slider("Schemes to show", 3, 10, 5)
    st.divider()
    st.caption(f"📚 {len(df)} schemes in the dataset")
    st.info("Preliminary guidance only. Always confirm on the official website before applying.")

# ------------------------------------------------------------------ header
st.markdown(
    '<div class="hero"><p class="app-title">🏛️ Government Service Navigator</p>'
    '<p class="app-sub">Tell us about yourself. We find schemes that may help, '
    'in English, हिंदी or मराठी.</p></div>',
    unsafe_allow_html=True,
)

tab_find, tab_elig, tab_about = st.tabs(["🔎 Find schemes", "🧾 Check eligibility", "ℹ️ How it works"])


# ------------------------------------------------------------------ helpers
def badge(text, kind):
    return f'<span class="badge b-{kind}">{text}</span>'


def set_query(text):
    st.session_state["query"] = text


def speech_text(text):
    text = re.sub(r"```[\s\S]*?```", " ", str(text))
    text = re.sub(r"!?\[([^\]]+)\]\(https?://[^)]+\)", r"\1", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"(?m)^\s*(#{1,6}|[-+•]|\d+[.)])\s+", "", text)
    text = re.sub(r"[*_~`|]", "", text)
    return re.sub(r"\s+", " ", text).strip()[:2500]


def make_speech(text, voice):
    async def run(path):
        await edge_tts.Communicate(text, voice).save(path)
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        path = f.name
    try:
        asyncio.run(run(path))
        with open(path, "rb") as f:
            return f.read()
    finally:
        if os.path.exists(path):
            os.remove(path)


def transcribe(audio_bytes, lang):
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY missing in .env")
    out = Groq(api_key=key).audio.transcriptions.create(
        file=("voice.wav", audio_bytes, "audio/wav"), model="whisper-large-v3",
        language=LANG_CODES[lang], response_format="text")
    return (out if isinstance(out, str) else getattr(out, "text", "")).strip()


# ================================================================== TAB 1
with tab_find:
    st.markdown("**Try an example**")
    cols = st.columns(len(EXAMPLES))
    for col, (label, text) in zip(cols, EXAMPLES.items()):
        col.button(label, on_click=set_query, args=(text,), use_container_width=True)

    with st.expander("🎙️ Prefer to speak? Record your question"):
        st.caption(f"Speaking language: **{voice_language}** (change it in the sidebar).")
        audio = st.audio_input("Press the mic, speak, then stop")
        if audio is not None:
            data = audio.getvalue()
            digest = hashlib.md5(data).hexdigest() + voice_language
            if st.session_state.get("last_audio") != digest:
                try:
                    with st.spinner("Converting your voice to text..."):
                        text = transcribe(data, voice_language)
                    st.session_state["last_audio"] = digest
                    if text:
                        st.session_state["query"] = text
                        st.session_state["voice_text"] = text
                        st.session_state["voice_lang"] = voice_language
                        st.success("Done. Check the text below, edit if needed, then search.")
                    else:
                        st.warning("We could not hear any speech. Please record again.")
                except Exception as exc:
                    st.error(f"Voice conversion failed ({type(exc).__name__}). Check your internet and API key.")

    query = st.text_area("What help do you need?", key="query", height=110,
                         placeholder="Example: I am a farmer from Maharashtra looking for a drip irrigation subsidy.")
    go = st.button("🔎 Find schemes", type="primary", use_container_width=True)

    if go:
        q = query.strip()
        if not q:
            st.warning("Please type your question or pick an example first.")
        else:
            try:
                with st.status("Working on it...", expanded=True) as status:
                    st.write("Searching the scheme database")
                    results = backend.search_schemes(q, top_k)
                    texts = [backend.build_scheme_text(i, r) for i in results
                             if (r := backend.get_row(i)) is not None]
                    st.write("Writing your guidance")
                    hint = (st.session_state.get("voice_lang")
                            if q == st.session_state.get("voice_text", "").strip() else None)
                    lang = backend.detect_query_language(q, hint)
                    answer = backend.generate_rag_answer(q, texts, lang)
                    status.update(label="Done", state="complete", expanded=False)
                st.session_state["search"] = {"results": results, "answer": answer, "language": lang, "query": q}
                st.session_state.pop("tts", None)
            except Exception as exc:
                st.error(f"Search failed ({type(exc).__name__}: {exc}). Please try again.")

    state = st.session_state.get("search")
    if state:
        lang = state["language"]
        with st.container(border=True):
            head, btn = st.columns([3, 2])
            head.markdown(f"#### 🤖 Guidance for you  \n{badge(lang, 'info')}", unsafe_allow_html=True)
            if btn.button("🔊 Listen to this answer", use_container_width=True):
                try:
                    with st.spinner("Preparing audio..."):
                        st.session_state["tts"] = make_speech(speech_text(state["answer"]), VOICES[lang])
                except Exception as exc:
                    st.error(f"Audio failed ({type(exc).__name__}). Check your internet connection.")
            if st.session_state.get("tts"):
                st.audio(st.session_state["tts"], format="audio/mp3")
            st.write(state["answer"])
            st.caption("AI-written from the dataset only. It may be incomplete. Verify on official sources.")

        st.markdown(f"#### 📋 {len(state['results'])} schemes that may fit")
        if not state["results"]:
            st.info("Nothing found. Add your state, work, or the kind of help you need and try again.")
        for rank, item in enumerate(state["results"], 1):
            row = backend.get_row(item)
            if row is None:
                continue
            meta = item["metadata"]
            name = clean(meta.get("scheme_name")) or clean(row.get("scheme_name")) or "Unnamed scheme"
            level = clean(meta.get("government_level") or row.get("government_level"))
            category = clean(meta.get("category") or row.get("category"))
            with st.expander(f"{rank}. {name}"):
                st.markdown((badge(category, "info") if category else "") + (badge(level, "ok") if level else ""),
                            unsafe_allow_html=True)
                for label, val in [
                    ("What it is", row.get("scheme_description")),
                    ("Benefit", meta.get("benefit") or row.get("benefit")),
                    ("Who can apply", row.get("target_beneficiary")),
                    ("Eligibility", meta.get("eligibility") or row.get("income_or_eligibility_condition")),
                ]:
                    st.markdown(f'<p class="field-label">{label}</p>', unsafe_allow_html=True)
                    st.write(clean(val) or "Not mentioned in the available data.")
                src, app_url = safe_url(row.get("official_source")), safe_url(row.get("application_url"))
                b1, b2 = st.columns(2)
                if src:
                    b1.link_button("🔗 Official source", src, use_container_width=True)
                if app_url:
                    b2.link_button("📝 Application link", app_url, use_container_width=True)
                if not (src or app_url):
                    st.caption("No official link in the dataset. Search the scheme name on the government portal.")
                st.caption("Status and rules are not independently verified.")

# ================================================================== TAB 2
with tab_elig:
    st.write("Enter your details for a quick first check. This is a screening, not an official decision.")
    with st.form("eligibility_form"):
        c1, c2 = st.columns(2)
        age = c1.number_input("Age", 0, 120, 22)
        state_name = c2.selectbox("State", backend.INDIAN_STATES,
                                  index=backend.INDIAN_STATES.index("Maharashtra"))
        income = c1.number_input("Annual family income (₹)", min_value=0, value=150000, step=10000)
        gender = c2.selectbox("Gender", ["Prefer not to say", "Female", "Male", "Other"])
        profile = st.selectbox("I am a...", list(backend.PROFILE_KEYWORDS))
        submitted = st.form_submit_button("Check eligibility", type="primary", use_container_width=True)

    if submitted:
        rows = []
        with st.spinner("Checking all schemes..."):
            for _, sch in df.iterrows():
                status_, notes, relevant = backend.assess_scheme(sch, age, state_name, income, gender, profile)
                if relevant:
                    rows.append({"Scheme": clean(sch.get("scheme_name")) or "Unnamed scheme",
                                 "Result": status_, "Why": "; ".join(notes),
                                 "Official source": safe_url(sch.get("official_source"))})
        out = pd.DataFrame(rows, columns=["Scheme", "Result", "Why", "Official source"])
        if not out.empty:
            order = {STATUS_MATCH: 0, STATUS_VERIFY: 1, STATUS_MISMATCH: 2}
            out = out.sort_values("Result", key=lambda s: s.map(order)).reset_index(drop=True)
        st.session_state["elig_results"] = out

    res = st.session_state.get("elig_results")
    if res is None:
        st.info("Fill in the form and press **Check eligibility** to see results.")
    elif res.empty:
        st.info("No schemes in this dataset relate to the profile you chose. Try 'Other'.")
    else:
        counts = res["Result"].value_counts()
        m1, m2, m3 = st.columns(3)
        m1.metric("✅ Likely fit", int(counts.get(STATUS_MATCH, 0)))
        m2.metric("⚠️ Check details", int(counts.get(STATUS_VERIFY, 0)))
        m3.metric("❌ Probably not", int(counts.get(STATUS_MISMATCH, 0)))

        pick = st.radio("Show", ["Likely fit + check details", "Likely fit", "Check details", "Probably not", "All"],
                        horizontal=True)
        show = {"Likely fit + check details": res[res["Result"] != STATUS_MISMATCH],
                "Likely fit": res[res["Result"] == STATUS_MATCH],
                "Check details": res[res["Result"] == STATUS_VERIFY],
                "Probably not": res[res["Result"] == STATUS_MISMATCH],
                "All": res}[pick]
        search_term = st.text_input("Search within results", placeholder="e.g. scholarship")
        if search_term:
            show = show[show["Scheme"].str.contains(search_term, case=False, na=False)]

        st.dataframe(show, use_container_width=True, hide_index=True,
                     column_config={"Official source": st.column_config.LinkColumn("Official source", display_text="Open"),
                                    "Why": st.column_config.TextColumn("Why", width="large")})
        st.download_button("⬇️ Download results (CSV)", show.to_csv(index=False).encode("utf-8"),
                           "eligibility_results.csv", "text/csv")

# ================================================================== TAB 3
with tab_about:
    st.markdown("""
**How it works**
1. You describe your situation by typing or speaking.
2. The app finds the closest schemes in a dataset of government schemes (semantic search plus keywords).
3. An AI writes a short answer using only those records, in your language.
4. You can listen to the answer and open the official source links.

**Please remember**
- The dataset may be incomplete or out of date.
- Eligibility results are a first screening. "Likely fit" does not guarantee approval.
- Always confirm rules, dates and documents on the official website.
""")

st.caption("Scheme records may be incomplete or outdated. Verify on official sources before applying.")
