"""
AllDoors Intelligence — RAG Chatbot for CRM & Business Data
Stack: Streamlit · Gemini 2.0 Flash · sentence-transformers · FAISS · Plotly
API key is read ONLY from Streamlit Cloud secrets (GEMINI_API_KEY).
It is never stored in code or exposed in the UI.
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

# ── page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="AllDoors Intelligence",
    page_icon="🚪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── design: old-money beige / cream / warm serif ─────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;1,400&family=EB+Garamond:ital,wght@0,400;0,500;1,400&display=swap');

/* ── global ── */
html, body, [class*="css"] {
  font-family: 'EB Garamond', Georgia, serif;
  background-color: #f5f0e8;
  color: #2c2416;
}
.stApp { background: #f5f0e8; }

/* ── hide streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

/* ── sidebar ── */
[data-testid="stSidebar"] {
  background: #ede8dc;
  border-right: 1px solid #c8b99a;
}
[data-testid="stSidebar"] * {
  font-family: 'EB Garamond', Georgia, serif !important;
  color: #2c2416 !important;
}

/* ── sidebar brand ── */
.brand-block {
  padding: 24px 0 12px 0;
  border-bottom: 1px solid #c8b99a;
  margin-bottom: 16px;
}
.brand-name {
  font-family: 'Playfair Display', Georgia, serif;
  font-size: 1.35rem;
  font-weight: 600;
  color: #2c2416;
  letter-spacing: 0.5px;
}
.brand-sub {
  font-family: 'EB Garamond', Georgia, serif;
  font-size: 0.78rem;
  color: #7a6a52;
  letter-spacing: 1.5px;
  text-transform: uppercase;
  margin-top: 3px;
}

/* ── sidebar section labels ── */
.sec-label {
  font-family: 'EB Garamond', Georgia, serif;
  font-size: 0.72rem;
  letter-spacing: 1.8px;
  text-transform: uppercase;
  color: #9a8a6a;
  margin: 18px 0 6px 0;
}

/* ── file uploader ── */
[data-testid="stFileUploader"] {
  background: #f0ebe0 !important;
  border: 1px dashed #c8b99a !important;
  border-radius: 4px !important;
}

/* ── primary button (Ingest) ── */
.stButton > button[kind="primary"] {
  background: #2c2416 !important;
  color: #f5f0e8 !important;
  border: none !important;
  border-radius: 3px !important;
  font-family: 'EB Garamond', Georgia, serif !important;
  font-size: 0.95rem !important;
  letter-spacing: 0.5px !important;
  padding: 10px 0 !important;
  margin-top: 8px;
}
.stButton > button[kind="primary"]:hover {
  background: #4a3c28 !important;
}

/* ── secondary buttons ── */
.stButton > button {
  border: 1px solid #c8b99a !important;
  background: transparent !important;
  color: #2c2416 !important;
  border-radius: 3px !important;
  font-family: 'EB Garamond', Georgia, serif !important;
  font-size: 0.9rem !important;
}
.stButton > button:hover {
  background: #ede8dc !important;
  border-color: #2c2416 !important;
}

/* ── stat cards ── */
.stat-row { display: flex; gap: 10px; margin: 10px 0; }
.stat-card {
  flex: 1;
  background: #f0ebe0;
  border: 1px solid #c8b99a;
  border-radius: 3px;
  padding: 10px 8px;
  text-align: center;
}
.stat-num {
  font-family: 'Playfair Display', Georgia, serif;
  font-size: 1.5rem;
  color: #2c2416;
  display: block;
}
.stat-lbl {
  font-size: 0.65rem;
  letter-spacing: 1.2px;
  text-transform: uppercase;
  color: #9a8a6a;
}

/* ── main header ── */
.main-header {
  padding: 40px 0 8px 0;
  border-bottom: 2px solid #2c2416;
  margin-bottom: 32px;
}
.main-title {
  font-family: 'Playfair Display', Georgia, serif;
  font-size: 2.4rem;
  font-weight: 600;
  color: #2c2416;
  letter-spacing: 0.3px;
  line-height: 1.15;
}
.main-title em {
  font-style: italic;
  font-weight: 400;
}
.main-sub {
  font-family: 'EB Garamond', Georgia, serif;
  font-size: 0.8rem;
  letter-spacing: 2px;
  text-transform: uppercase;
  color: #9a8a6a;
  margin-top: 6px;
}

/* ── suggestion cards ── */
.sug-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin: 20px 0 32px; }
.sug-card {
  background: #ede8dc;
  border: 1px solid #c8b99a;
  border-radius: 3px;
  padding: 14px 16px;
  cursor: pointer;
  transition: border-color 0.2s, background 0.2s;
}
.sug-card:hover { background: #e5dfd0; border-color: #2c2416; }
.sug-icon { font-size: 1.1rem; margin-bottom: 6px; }
.sug-text {
  font-family: 'EB Garamond', Georgia, serif;
  font-size: 0.9rem;
  color: #2c2416;
  line-height: 1.35;
}

/* ── chat messages ── */
[data-testid="stChatMessage"] {
  background: transparent !important;
  border: none !important;
  padding: 12px 0 !important;
  border-bottom: 1px solid #ddd5c0 !important;
}

/* ── chat input ── */
[data-testid="stChatInput"] {
  border-top: 1px solid #c8b99a;
  background: #f5f0e8;
  padding-top: 12px;
}
[data-testid="stChatInput"] textarea {
  background: #ede8dc !important;
  border: 1px solid #c8b99a !important;
  border-radius: 3px !important;
  color: #2c2416 !important;
  font-family: 'EB Garamond', Georgia, serif !important;
  font-size: 1rem !important;
}

/* ── source pills ── */
.pill {
  display: inline-block;
  background: #ede8dc;
  border: 1px solid #c8b99a;
  border-radius: 2px;
  padding: 1px 9px;
  font-size: 0.72rem;
  color: #7a6a52;
  letter-spacing: 0.3px;
  margin: 3px 3px 0 0;
  font-family: 'EB Garamond', Georgia, serif;
}

/* ── empty state ── */
.empty-rule {
  border: none;
  border-top: 1px solid #c8b99a;
  margin: 8px 0 24px;
}
.empty-label {
  font-family: 'EB Garamond', Georgia, serif;
  font-size: 0.72rem;
  letter-spacing: 2px;
  text-transform: uppercase;
  color: #9a8a6a;
}
.footer-note {
  text-align: center;
  color: #b0a080;
  font-size: 0.78rem;
  font-style: italic;
  margin-top: 32px;
  letter-spacing: 0.3px;
}

/* ── API key input (fallback for local) ── */
[data-testid="stTextInput"] input {
  background: #f0ebe0 !important;
  border: 1px solid #c8b99a !important;
  border-radius: 3px !important;
  color: #2c2416 !important;
  font-family: 'EB Garamond', Georgia, serif !important;
}

/* ── alerts ── */
.stSuccess { background: #eef5ec !important; border-color: #a8c8a0 !important; }
.stWarning { background: #fdf6e3 !important; border-color: #e0c878 !important; }
</style>
""", unsafe_allow_html=True)

# ── constants ────────────────────────────────────────────────
EMBED_MODEL  = "all-MiniLM-L6-v2"
CHUNK_SIZE   = 400
OVERLAP      = 80
TOP_K        = 8
GEMINI_MODEL = "gemini-2.0-flash"
CHART_KW     = ["chart","plot","graph","visuali","bar","pie","line",
                 "scatter","histogram","show me","compare","distribution","trend"]

# ── session state ────────────────────────────────────────────
for k, v in {
    "chunks":[], "index":None, "embeddings":None,
    "history":[], "dataframes":{}, "ready":False, "embed_model":None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── embedding model (cached) ─────────────────────────────────
@st.cache_resource(show_spinner="Preparing intelligence layer…")
def load_embed():
    return SentenceTransformer(EMBED_MODEL)

# ════════════════════════════════════════════════════════════
# PARSERS
# ════════════════════════════════════════════════════════════
def chunk(text, src, meta):
    out, s = [], 0
    while s < len(text):
        c = text[s:s+CHUNK_SIZE].strip()
        if c:
            out.append({"text":c, "source":src, "meta":meta.copy()})
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
        key = f"{name} › {sh}"
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
            if t.strip():
                out.extend(chunk(t, name, {"page":i+1}))
    return out

def parse_docx(f, name):
    d = docx.Document(f)
    return chunk("\n".join(p.text for p in d.paragraphs if p.text.strip()), name, {"type":"doc"})

def parse_json(f, name):
    return chunk(json.dumps(json.load(f), indent=2), name, {"type":"json"})

def parse_text(f, name):
    return chunk(f.read().decode("utf-8", errors="ignore"), name, {"type":"text"})

PARSERS = {
    ".csv":parse_csv, ".xlsx":parse_excel, ".xls":parse_excel,
    ".pdf":parse_pdf, ".docx":parse_docx, ".json":parse_json,
    ".tsv":lambda f,n: parse_csv(f,n),
    ".md":parse_text, ".txt":parse_text,
}

# ── index ────────────────────────────────────────────────────
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

# ── Gemini ───────────────────────────────────────────────────
def ask(question, ctx, history):
    context = "\n\n---\n\n".join(
        f"[SOURCE {i+1}: {c['source']}" +
        (f" | {', '.join(f'{k}:{v}' for k,v in c['meta'].items())}" if c['meta'] else "") +
        f"]\n{c['text']}"
        for i,c in enumerate(ctx)
    )
    hist = "".join(
        f"{'User' if t['role']=='user' else 'Assistant'}: {t['content']}\n"
        for t in history[-6:]
    )
    prompt = textwrap.dedent(f"""
        You are AllDoors Intelligence, a precise real estate and business data analyst.
        Answer ONLY from the provided source documents.
        If the answer is not there, say: "I don't see that information in the uploaded files."
        Cite every fact with [SOURCE N].
        For chart requests include a JSON block:
        ```chart
        {{"type":"bar","x":"column_name","y":"column_name","source":"filename"}}
        ```

        HISTORY:
        {hist}

        SOURCES:
        {context}

        QUESTION: {question}
        ANSWER:""").strip()
    return genai.GenerativeModel(GEMINI_MODEL).generate_content(prompt).text

# ── chart ────────────────────────────────────────────────────
def wants_chart(q):
    return any(k in q.lower() for k in CHART_KW)

def get_spec(answer):
    m = re.search(r"```chart\s*(\{.*?\})\s*```", answer, re.DOTALL)
    if m:
        try: return json.loads(m.group(1))
        except: pass
    return None

def render_chart(question, answer):
    dfs = st.session_state["dataframes"]
    if not dfs: return None
    spec = get_spec(answer)
    q = question.lower()

    # Warm beige chart theme
    LAYOUT = dict(
        paper_bgcolor="#f5f0e8", plot_bgcolor="#ede8dc",
        font=dict(family="EB Garamond, Georgia, serif", color="#2c2416"),
        title_font=dict(family="Playfair Display, Georgia, serif", size=16, color="#2c2416"),
        colorway=["#8B6914","#C4A35A","#4a3c28","#b8860b","#d2b48c","#6b4c11"],
    )

    if spec:
        df = next((dfs[k] for k in dfs if spec.get("source","").lower() in k.lower()), list(dfs.values())[0])
        x, y, t = spec.get("x"), spec.get("y"), spec.get("type","bar")
        if x in df.columns and y in df.columns:
            fns = {"bar":px.bar,"line":px.line,"pie":px.pie,"scatter":px.scatter}
            fn = fns.get(t, px.bar)
            kw = {"names":x,"values":y} if t=="pie" else {"x":x,"y":y}
            fig = fn(df, title=f"{y} by {x}", **kw)
            fig.update_layout(**LAYOUT)
            return fig

    label, df = max(dfs.items(), key=lambda kv: len(kv[1]))
    num = df.select_dtypes(include="number").columns.tolist()
    cat = df.select_dtypes(exclude="number").columns.tolist()
    if not num: return None
    y_col = num[0]; x_col = cat[0] if cat else df.columns[0]
    try:
        agg = df.groupby(x_col)[y_col].sum().reset_index().sort_values(y_col,ascending=False).head(20)
        fig = px.bar(agg, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
        fig.update_layout(**LAYOUT)
        return fig
    except:
        return None

# ════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════
with st.sidebar:
    # Brand
    st.markdown("""
    <div class='brand-block'>
      <div class='brand-name'>AllDoors</div>
      <div class='brand-sub'>Intelligence Suite</div>
    </div>
    """, unsafe_allow_html=True)

    # ── API KEY — read silently from secrets, never shown ────
    # The key lives ONLY in Streamlit Cloud → App Settings → Secrets
    # as: GEMINI_API_KEY = "your-key"
    # It is never printed, logged, or exposed in the UI.
    _key = ""
    try:
        _key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        pass

    if not _key:
        # Fallback input for local development only
        st.markdown("<div class='sec-label'>API Key</div>", unsafe_allow_html=True)
        _key = st.text_input("key", type="password",
                             placeholder="Paste Gemini key…",
                             label_visibility="collapsed")
        st.caption("Get a free key at [aistudio.google.com](https://aistudio.google.com)")

    if _key:
        genai.configure(api_key=_key)
        st.session_state["ready"] = True

    # ── File upload ───────────────────────────────────────────
    st.markdown("<div class='sec-label'>Documents</div>", unsafe_allow_html=True)
    st.caption("CSV · Excel · PDF · Word · JSON · TSV")

    files = st.file_uploader(
        "docs", label_visibility="collapsed",
        type=["csv","xlsx","xls","pdf","docx","json","tsv","md","txt"],
        accept_multiple_files=True,
    )

    if files:
        if st.button("⟳  Index Documents", use_container_width=True, type="primary"):
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
                        status.caption(f"✓ {f.name}")
                    except Exception as e:
                        st.warning(f"{f.name}: {e}")
                bar.progress((i+1)/len(files))

            if all_chunks:
                with st.spinner("Building index…"):
                    idx, emb = build_index(all_chunks, model)
                st.session_state.update({
                    "chunks":all_chunks, "index":idx,
                    "embeddings":emb, "embed_model":model,
                })
                bar.empty(); status.empty()
                st.success(f"{len(all_chunks)} passages indexed across {len(files)} file(s).")
            else:
                st.error("No content extracted.")

    # ── Stats ─────────────────────────────────────────────────
    if st.session_state["chunks"]:
        st.markdown(f"""
        <div class='stat-row'>
          <div class='stat-card'>
            <span class='stat-num'>{len(st.session_state['chunks'])}</span>
            <span class='stat-lbl'>Passages</span>
          </div>
          <div class='stat-card'>
            <span class='stat-num'>{len(st.session_state['dataframes'])}</span>
            <span class='stat-lbl'>Tables</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:24px'></div>", unsafe_allow_html=True)

    if st.session_state["chunks"]:
        if st.button("Clear session", use_container_width=True):
            for k in ["chunks","index","embeddings","history","dataframes","embed_model"]:
                st.session_state[k] = [] if k in ("chunks","history") else ({} if k=="dataframes" else None)
            st.rerun()

    st.markdown("""
    <div style='position:absolute;bottom:20px;left:0;right:0;text-align:center;
    font-size:0.68rem;color:#b0a080;font-style:italic;letter-spacing:0.5px;
    font-family:EB Garamond,Georgia,serif'>
    alldoors.in · private & confidential
    </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════
st.markdown("""
<div class='main-header'>
  <div class='main-title'>AllDoors <em>Intelligence</em></div>
  <div class='main-sub'>Data Analysis · Cited Answers · Live Charts</div>
</div>
""", unsafe_allow_html=True)

# ── Empty state suggestions ───────────────────────────────────
if not st.session_state["history"]:
    st.markdown("<div class='empty-label'>Ask anything about your data</div>", unsafe_allow_html=True)
    st.markdown("<hr class='empty-rule'>", unsafe_allow_html=True)

    SUGGESTIONS = [
        ("💰", "What was the total closed-won revenue?"),
        ("🏆", "Who is the top rep by pipeline value?"),
        ("📊", "Show me a bar chart of deals by stage"),
        ("⚠️",  "List the at-risk accounts"),
        ("🔗", "Does the PDF contract total match the deals sheet?"),
        ("📈", "Which lead sources convert best?"),
    ]
    cols = st.columns(3)
    for i, (icon, s) in enumerate(SUGGESTIONS):
        with cols[i % 3]:
            if st.button(f"{icon}  {s}", use_container_width=True, key=f"sug{i}"):
                st.session_state["_q"] = s
                st.rerun()

    st.markdown("""
    <div class='footer-note'>
      Upload your files in the sidebar, index them, and begin your enquiry below.
    </div>""", unsafe_allow_html=True)

# ── Chat history ──────────────────────────────────────────────
for turn in st.session_state["history"]:
    avatar = "👤" if turn["role"] == "user" else "🚪"
    with st.chat_message(turn["role"], avatar=avatar):
        st.markdown(turn["content"])
        if turn.get("sources"):
            pills = " ".join(f"<span class='pill'>📄 {s}</span>" for s in turn["sources"])
            st.markdown(pills, unsafe_allow_html=True)
        if turn.get("fig"):
            st.plotly_chart(turn["fig"], use_container_width=True)

# ── Input ─────────────────────────────────────────────────────
question = st.chat_input("Enter your enquiry…") or st.session_state.pop("_q", None)

if question:
    if not st.session_state["ready"]:
        st.error("Please add your Gemini API key in the sidebar.")
        st.stop()
    if not st.session_state["chunks"]:
        st.error("Please upload and index your documents first.")
        st.stop()

    with st.chat_message("user", avatar="👤"):
        st.markdown(question)
    st.session_state["history"].append({"role":"user","content":question})

    with st.chat_message("assistant", avatar="🚪"):
        with st.spinner("Searching documents…"):
            ctx = retrieve(question, st.session_state["embed_model"],
                           st.session_state["index"], st.session_state["embeddings"],
                           st.session_state["chunks"])
        with st.spinner("Composing answer…"):
            raw = ask(question, ctx, st.session_state["history"])

        display = re.sub(r"```chart.*?```","", raw, flags=re.DOTALL).strip()
        st.markdown(display)

        sources = list(dict.fromkeys(c["source"] for c in ctx))
        pills = " ".join(f"<span class='pill'>📄 {s}</span>" for s in sources)
        st.markdown(pills, unsafe_allow_html=True)

        fig = None
        if wants_chart(question):
            fig = render_chart(question, raw)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

    st.session_state["history"].append({
        "role":"assistant","content":display,
        "sources":sources,"fig":fig,
    })