"""
AllDoors Intelligence — RAG Chatbot for CRM & Business Data
API key lives ONLY in Streamlit Cloud Secrets as GEMINI_API_KEY.
Never printed, logged, or shown in the UI.
"""

import json, re, textwrap
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import faiss
import pdfplumber
import docx
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="AllDoors Intelligence",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,500;1,300;1,400&family=Jost:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, .stApp {
  background: #FAF8F4;
  color: #1A1A1A;
  font-family: 'Jost', sans-serif;
  font-weight: 300;
}

#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"] { display: none !important; }

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
  background: #F2EFE9;
  border-right: 1px solid #D8D0C4;
  padding: 0;
}
[data-testid="stSidebar"] > div:first-child {
  padding: 40px 28px 32px;
}

.sb-logo {
  font-family: 'Cormorant Garamond', serif;
  font-size: 1.5rem;
  font-weight: 400;
  letter-spacing: 0.08em;
  color: #1A1A1A;
  text-transform: uppercase;
}
.sb-rule {
  border: none;
  border-top: 1px solid #D8D0C4;
  margin: 20px 0;
}
.sb-label {
  font-family: 'Jost', sans-serif;
  font-size: 0.62rem;
  font-weight: 500;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: #9A9080;
  margin-bottom: 10px;
  display: block;
}
.sb-foot {
  font-family: 'Cormorant Garamond', serif;
  font-style: italic;
  font-size: 0.8rem;
  color: #B8AFA0;
  text-align: center;
  margin-top: 32px;
}

/* stat cards */
.stats { display: flex; gap: 8px; margin: 6px 0 20px; }
.stat {
  flex: 1; background: #EAE6DF;
  border: 1px solid #D8D0C4;
  padding: 12px 8px; text-align: center;
}
.stat-n {
  font-family: 'Cormorant Garamond', serif;
  font-size: 1.6rem; font-weight: 300;
  color: #1A1A1A; display: block;
}
.stat-l {
  font-size: 0.58rem; font-weight: 500;
  letter-spacing: 0.15em; text-transform: uppercase;
  color: #9A9080;
}

/* ── BUTTONS ── */
.stButton > button {
  font-family: 'Jost', sans-serif !important;
  font-weight: 400 !important;
  font-size: 0.82rem !important;
  letter-spacing: 0.06em !important;
  border-radius: 0 !important;
  transition: all 0.2s !important;
}
.stButton > button[kind="primary"] {
  background: #1A1A1A !important;
  color: #FAF8F4 !important;
  border: 1px solid #1A1A1A !important;
  padding: 11px 0 !important;
}
.stButton > button[kind="primary"]:hover {
  background: #3A3530 !important;
}
.stButton > button:not([kind="primary"]) {
  background: transparent !important;
  color: #3A3530 !important;
  border: 1px solid #C8C0B4 !important;
  padding: 10px 0 !important;
}
.stButton > button:not([kind="primary"]):hover {
  border-color: #1A1A1A !important;
  color: #1A1A1A !important;
}

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] {
  background: #EAE6DF !important;
  border: 1px dashed #C8C0B4 !important;
  border-radius: 0 !important;
}

/* ── MAIN AREA ── */
.main-wrap { padding: 52px 60px 20px; max-width: 1000px; }

.eyebrow {
  font-family: 'Jost', sans-serif;
  font-size: 0.62rem;
  font-weight: 500;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  color: #9A9080;
  margin-bottom: 12px;
}
.headline {
  font-family: 'Cormorant Garamond', serif;
  font-size: 3.2rem;
  font-weight: 300;
  line-height: 1.08;
  color: #1A1A1A;
  letter-spacing: -0.01em;
}
.headline em { font-style: italic; }
.divider {
  border: none;
  border-top: 1px solid #1A1A1A;
  margin: 28px 0 36px;
}

/* suggestion grid */
.sug-section-label {
  font-family: 'Jost', sans-serif;
  font-size: 0.6rem;
  font-weight: 500;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: #9A9080;
  margin-bottom: 14px;
}
.sug-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 1px;
  background: #D8D0C4;
  border: 1px solid #D8D0C4;
  margin-bottom: 40px;
}
.sug-item {
  background: #FAF8F4;
  padding: 18px 20px;
  cursor: pointer;
  transition: background 0.15s;
}
.sug-item:hover { background: #F2EFE9; }
.sug-text {
  font-family: 'Cormorant Garamond', serif;
  font-size: 1rem;
  font-weight: 400;
  color: #3A3530;
  line-height: 1.3;
}

/* ── CHAT ── */
[data-testid="stChatMessage"] {
  background: transparent !important;
  border: none !important;
  border-bottom: 1px solid #E8E3DC !important;
  padding: 24px 0 !important;
  border-radius: 0 !important;
}
[data-testid="stChatMessage"] p {
  font-family: 'Jost', sans-serif !important;
  font-weight: 300 !important;
  font-size: 0.95rem !important;
  line-height: 1.7 !important;
  color: #1A1A1A !important;
}
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] { display: none !important; }

/* ── CHAT INPUT ── */
[data-testid="stChatInput"] {
  background: #FAF8F4;
  border-top: 1px solid #D8D0C4;
  padding-top: 16px;
}
[data-testid="stChatInput"] textarea {
  background: #F2EFE9 !important;
  border: 1px solid #C8C0B4 !important;
  border-radius: 0 !important;
  color: #1A1A1A !important;
  font-family: 'Jost', sans-serif !important;
  font-weight: 300 !important;
  font-size: 0.95rem !important;
}
[data-testid="stChatInput"] textarea::placeholder {
  color: #A8A09A !important;
  font-style: italic;
}

/* source pills */
.pill {
  display: inline-block;
  border: 1px solid #D8D0C4;
  padding: 2px 10px;
  font-family: 'Jost', sans-serif;
  font-size: 0.68rem;
  font-weight: 400;
  letter-spacing: 0.05em;
  color: #9A9080;
  margin: 4px 4px 0 0;
}

/* text input fallback */
[data-testid="stTextInput"] input {
  background: #EAE6DF !important;
  border: 1px solid #C8C0B4 !important;
  border-radius: 0 !important;
  font-family: 'Jost', sans-serif !important;
  font-weight: 300 !important;
  color: #1A1A1A !important;
}
</style>
""", unsafe_allow_html=True)

# ── constants ────────────────────────────────────────────────
EMBED_MODEL  = "all-MiniLM-L6-v2"
CHUNK_SIZE   = 400
OVERLAP      = 80
TOP_K        = 8
GEMINI_MODEL = "gemini-2.0-flash"
CHART_KW     = ["chart","plot","graph","visuali","bar","pie","line",
                 "scatter","histogram","show","compare","distribution","trend"]

for k, v in {
    "chunks":[], "index":None, "embeddings":None,
    "history":[], "dataframes":{}, "ready":False, "embed_model":None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

@st.cache_resource(show_spinner="Loading model…")
def load_embed():
    return SentenceTransformer(EMBED_MODEL)

# ── parsers ──────────────────────────────────────────────────
def chunk(text, src, meta):
    out, s = [], 0
    while s < len(text):
        c = text[s:s+CHUNK_SIZE].strip()
        if c: out.append({"text":c,"source":src,"meta":meta.copy()})
        s += CHUNK_SIZE - OVERLAP
    return out

def parse_csv(f, name):
    df = pd.read_csv(f)
    st.session_state["dataframes"][name] = df
    rows = [{"text":f"File:{name}\nCols:{','.join(df.columns)}\nRows:{len(df)}","source":name,"meta":{"type":"header"}}]
    for i in range(0,len(df),10):
        b = df.iloc[i:i+10]
        rows.append({"text":f"[{name} rows {i+1}-{i+len(b)}]\n{b.to_string(index=False)}","source":name,"meta":{"rows":f"{i+1}-{i+len(b)}"}})
    return rows

def parse_excel(f, name):
    xl, out = pd.ExcelFile(f), []
    for sh in xl.sheet_names:
        df = xl.parse(sh)
        key = f"{name} / {sh}"
        st.session_state["dataframes"][key] = df
        out.append({"text":f"File:{name} Sheet:{sh}\nCols:{','.join(str(c) for c in df.columns)}\nRows:{len(df)}","source":key,"meta":{"type":"header"}})
        for i in range(0,len(df),10):
            b = df.iloc[i:i+10]
            out.append({"text":f"[{key} rows {i+1}-{i+len(b)}]\n{b.to_string(index=False)}","source":key,"meta":{"rows":f"{i+1}-{i+len(b)}"}})
    return out

def parse_pdf(f, name):
    out = []
    with pdfplumber.open(f) as pdf:
        for i, pg in enumerate(pdf.pages):
            t = pg.extract_text() or ""
            if t.strip(): out.extend(chunk(t, name, {"page":i+1}))
    return out

def parse_docx(f, name):
    d = docx.Document(f)
    return chunk("\n".join(p.text for p in d.paragraphs if p.text.strip()), name, {"type":"doc"})

def parse_json(f, name):
    return chunk(json.dumps(json.load(f), indent=2), name, {"type":"json"})

def parse_text(f, name):
    return chunk(f.read().decode("utf-8", errors="ignore"), name, {"type":"text"})

PARSERS = {
    ".csv":parse_csv,".xlsx":parse_excel,".xls":parse_excel,
    ".pdf":parse_pdf,".docx":parse_docx,".json":parse_json,
    ".tsv":lambda f,n:parse_csv(f,n),".md":parse_text,".txt":parse_text,
}

def build_index(chunks, model):
    emb = model.encode([c["text"] for c in chunks], show_progress_bar=False, batch_size=64).astype(np.float32)
    faiss.normalize_L2(emb)
    idx = faiss.IndexFlatIP(emb.shape[1])
    idx.add(emb)
    return idx, emb

def retrieve(q, model, idx, emb, chunks):
    qe = model.encode([q]).astype(np.float32)
    faiss.normalize_L2(qe)
    scores, ids = idx.search(qe, TOP_K)
    return [{**chunks[i],"score":float(s)} for s,i in zip(scores[0],ids[0]) if i>=0]

def ask(question, ctx, history):
    context = "\n\n---\n\n".join(
        f"[SOURCE {i+1}: {c['source']}" +
        (f" | {', '.join(f'{k}:{v}' for k,v in c['meta'].items())}" if c['meta'] else "") +
        f"]\n{c['text']}" for i,c in enumerate(ctx))
    hist = "".join(
        f"{'User' if t['role']=='user' else 'Assistant'}: {t['content']}\n"
        for t in history[-6:])
    prompt = textwrap.dedent(f"""
        You are AllDoors Intelligence, a precise real estate and business data analyst.
        Answer ONLY from the provided sources.
        If the answer is not there say: "I don't see that information in the uploaded files."
        Cite every fact with [SOURCE N].
        For chart requests include:
        ```chart
        {{"type":"bar","x":"column","y":"column","source":"filename"}}
        ```
        HISTORY: {hist}
        SOURCES: {context}
        QUESTION: {question}
        ANSWER:""").strip()
    return genai.GenerativeModel(GEMINI_MODEL).generate_content(prompt).text

def wants_chart(q):
    return any(k in q.lower() for k in CHART_KW)

def get_spec(answer):
    m = re.search(r"```chart\s*(\{.*?\})\s*```", answer, re.DOTALL)
    if m:
        try: return json.loads(m.group(1))
        except: pass
    return None

CHART_LAYOUT = dict(
    paper_bgcolor="#FAF8F4", plot_bgcolor="#F2EFE9",
    font=dict(family="Jost, sans-serif", color="#3A3530", size=12),
    title_font=dict(family="Cormorant Garamond, serif", size=18, color="#1A1A1A"),
    colorway=["#3A3530","#8B7355","#C4A882","#1A1A1A","#6B5B45","#D4C4B0"],
    margin=dict(t=48, b=32, l=32, r=32),
)

def render_chart(question, answer):
    dfs = st.session_state["dataframes"]
    if not dfs: return None
    spec = get_spec(answer)
    if spec:
        df = next((dfs[k] for k in dfs if spec.get("source","").lower() in k.lower()), list(dfs.values())[0])
        x,y,t = spec.get("x"),spec.get("y"),spec.get("type","bar")
        if x in df.columns and y in df.columns:
            fns = {"bar":px.bar,"line":px.line,"pie":px.pie,"scatter":px.scatter}
            kw = {"names":x,"values":y} if t=="pie" else {"x":x,"y":y}
            fig = fns.get(t, px.bar)(df, title=f"{y} by {x}", **kw)
            fig.update_layout(**CHART_LAYOUT)
            return fig
    label,df = max(dfs.items(), key=lambda kv:len(kv[1]))
    num = df.select_dtypes(include="number").columns.tolist()
    cat = df.select_dtypes(exclude="number").columns.tolist()
    if not num: return None
    y_col,x_col = num[0], (cat[0] if cat else df.columns[0])
    try:
        agg = df.groupby(x_col)[y_col].sum().reset_index().sort_values(y_col,ascending=False).head(20)
        fig = px.bar(agg, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
        fig.update_layout(**CHART_LAYOUT)
        return fig
    except: return None

# ════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("<div class='sb-logo'>AllDoors<br><span style='font-size:0.65rem;letter-spacing:0.22em;color:#9A9080;font-family:Jost,sans-serif;font-weight:400'>INTELLIGENCE</span></div>", unsafe_allow_html=True)
    st.markdown("<hr class='sb-rule'>", unsafe_allow_html=True)

    # API key — read silently from Streamlit secrets, never displayed
    _key = ""
    try:
        _key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        pass
    if not _key:
        st.markdown("<span class='sb-label'>API Key</span>", unsafe_allow_html=True)
        _key = st.text_input("key", type="password",
                             placeholder="Paste Gemini key…",
                             label_visibility="collapsed")
        st.caption("Free key at [aistudio.google.com](https://aistudio.google.com)")
    if _key:
        genai.configure(api_key=_key)
        st.session_state["ready"] = True

    st.markdown("<span class='sb-label'>Documents</span>", unsafe_allow_html=True)
    st.caption("CSV · Excel · PDF · Word · JSON")

    files = st.file_uploader("docs", label_visibility="collapsed",
        type=["csv","xlsx","xls","pdf","docx","json","tsv","md","txt"],
        accept_multiple_files=True)

    if files:
        if st.button("Index Documents", use_container_width=True, type="primary"):
            model = load_embed()
            all_chunks, bar = [], st.progress(0)
            status = st.empty()
            for i, f in enumerate(files):
                ext = Path(f.name).suffix.lower()
                parser = PARSERS.get(ext)
                if parser:
                    try:
                        nc = parser(f, f.name)
                        all_chunks.extend(nc)
                        status.caption(f"{f.name} — {len(nc)} passages")
                    except Exception as e:
                        st.warning(f"{f.name}: {e}")
                bar.progress((i+1)/len(files))
            if all_chunks:
                with st.spinner("Building index…"):
                    idx, emb = build_index(all_chunks, model)
                st.session_state.update({"chunks":all_chunks,"index":idx,"embeddings":emb,"embed_model":model})
                bar.empty(); status.empty()
                st.success(f"{len(all_chunks)} passages indexed.")
            else:
                st.error("No content extracted.")

    if st.session_state["chunks"]:
        st.markdown(f"""
        <div class='stats'>
          <div class='stat'><span class='stat-n'>{len(st.session_state['chunks'])}</span><span class='stat-l'>Passages</span></div>
          <div class='stat'><span class='stat-n'>{len(st.session_state['dataframes'])}</span><span class='stat-l'>Tables</span></div>
        </div>""", unsafe_allow_html=True)
        if st.button("Clear session", use_container_width=True):
            for k in ["chunks","index","embeddings","history","dataframes","embed_model"]:
                st.session_state[k] = [] if k in ("chunks","history") else ({} if k=="dataframes" else None)
            st.rerun()

    st.markdown("<div class='sb-foot'>alldoors.in</div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════
st.markdown("""
<div class='main-wrap'>
  <div class='eyebrow'>Real Estate Intelligence</div>
  <div class='headline'>AllDoors <em>Intelligence</em></div>
  <hr class='divider'>
</div>""", unsafe_allow_html=True)

SUGGESTIONS = [
    "What was the total closed-won revenue?",
    "Who is the top rep by pipeline value?",
    "Show me a bar chart of deals by stage",
    "List the at-risk accounts",
    "Does the PDF contract total match the deals sheet?",
    "Which lead sources convert best?",
]

if not st.session_state["history"]:
    st.markdown("<div class='main-wrap'><div class='sug-section-label'>Suggested Queries</div>", unsafe_allow_html=True)
    cols = st.columns(3, gap="small")
    for i, s in enumerate(SUGGESTIONS):
        with cols[i % 3]:
            if st.button(s, use_container_width=True, key=f"s{i}"):
                st.session_state["_q"] = s
                st.rerun()
    st.markdown("""
    <p style='font-family:Cormorant Garamond,serif;font-style:italic;
    color:#B8AFA0;font-size:0.9rem;text-align:center;margin-top:40px'>
    Upload your documents in the sidebar, index them, and begin your enquiry.
    </p></div>""", unsafe_allow_html=True)

for turn in st.session_state["history"]:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("sources"):
            st.markdown(" ".join(f"<span class='pill'>{s}</span>" for s in turn["sources"]), unsafe_allow_html=True)
        if turn.get("fig"):
            st.plotly_chart(turn["fig"], use_container_width=True)

question = st.chat_input("Enter your enquiry…") or st.session_state.pop("_q", None)

if question:
    if not st.session_state["ready"]:
        st.error("Please add your Gemini API key in the sidebar.")
        st.stop()
    if not st.session_state["chunks"]:
        st.error("Please upload and index your documents first.")
        st.stop()

    with st.chat_message("user"):
        st.markdown(question)
    st.session_state["history"].append({"role":"user","content":question})

    with st.chat_message("assistant"):
        with st.spinner("Searching…"):
            ctx = retrieve(question, st.session_state["embed_model"],
                           st.session_state["index"], st.session_state["embeddings"],
                           st.session_state["chunks"])
        with st.spinner("Composing answer…"):
            raw = ask(question, ctx, st.session_state["history"])

        display = re.sub(r"```chart.*?```","", raw, flags=re.DOTALL).strip()
        st.markdown(display)
        sources = list(dict.fromkeys(c["source"] for c in ctx))
        st.markdown(" ".join(f"<span class='pill'>{s}</span>" for s in sources), unsafe_allow_html=True)

        fig = None
        if wants_chart(question):
            fig = render_chart(question, raw)
            if fig: st.plotly_chart(fig, use_container_width=True)

    st.session_state["history"].append({"role":"assistant","content":display,"sources":sources,"fig":fig})