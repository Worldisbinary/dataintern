"""
DataIntern — RAG Chatbot for CRM & Business Data
Stack: Streamlit · Gemini 2.0 Flash · sentence-transformers · FAISS · Plotly
"""

import json, re, textwrap
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import faiss
import pdfplumber
import docx
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# ── page config ─────────────────────────────────────────────
st.set_page_config(
    page_title="DataIntern",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
  /* hide default Streamlit header & footer */
  #MainMenu, footer, header { visibility: hidden; }

  /* overall background */
  .stApp { background: #0f1117; }

  /* sidebar */
  [data-testid="stSidebar"] {
    background: #16181f;
    border-right: 1px solid #2a2d3a;
  }

  /* chat input */
  [data-testid="stChatInput"] textarea {
    background: #1e2130 !important;
    border: 1px solid #3a3d4a !important;
    color: #e0e0e0 !important;
    border-radius: 12px !important;
  }

  /* suggestion buttons */
  .sug-btn button {
    background: #1e2130 !important;
    border: 1px solid #3a3d4a !important;
    color: #c0c4d6 !important;
    border-radius: 10px !important;
    text-align: left !important;
    font-size: 0.82rem !important;
    padding: 10px 14px !important;
    transition: border-color 0.2s;
  }
  .sug-btn button:hover {
    border-color: #6c8cff !important;
    color: #fff !important;
  }

  /* source pills */
  .pill {
    display: inline-block;
    background: #1e2130;
    border: 1px solid #3a3d4a;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.72rem;
    color: #8892b0;
    margin: 2px 3px 0 0;
  }

  /* ingest success */
  .ingest-ok {
    background: #0d2b1f;
    border: 1px solid #1a4a30;
    border-radius: 8px;
    padding: 10px 14px;
    color: #4ade80;
    font-size: 0.82rem;
    margin-top: 8px;
  }

  /* stat cards in sidebar */
  .stat { text-align: center; padding: 8px 0; }
  .stat b { display: block; font-size: 1.4rem; color: #6c8cff; }
  .stat span { font-size: 0.7rem; color: #8892b0; text-transform: uppercase; letter-spacing: 0.5px; }

  h1 { font-size: 1.55rem !important; color: #e0e0e0 !important; }
  p, li { color: #a0a4b8; }
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
    "chunks": [], "index": None, "embeddings": None,
    "history": [], "dataframes": {}, "ready": False, "embed_model": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── embedding model ──────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model…")
def load_embed():
    return SentenceTransformer(EMBED_MODEL)

# ── parsers ──────────────────────────────────────────────────
def chunk(text, src, meta):
    out, s = [], 0
    while s < len(text):
        c = text[s:s+CHUNK_SIZE].strip()
        if c:
            out.append({"text": c, "source": src, "meta": meta.copy()})
        s += CHUNK_SIZE - OVERLAP
    return out

def parse_csv(f, name):
    df = pd.read_csv(f)
    st.session_state["dataframes"][name] = df
    rows = [{"text": f"File:{name}\nCols:{','.join(df.columns)}\nRows:{len(df)}", "source": name, "meta": {"type":"header"}}]
    for i in range(0, len(df), 10):
        b = df.iloc[i:i+10]
        rows.append({"text": f"[{name} rows {i+1}-{i+len(b)}]\n{b.to_string(index=False)}", "source": name, "meta": {"rows": f"{i+1}-{i+len(b)}"}})
    return rows

def parse_excel(f, name):
    xl, out = pd.ExcelFile(f), []
    for sh in xl.sheet_names:
        df = xl.parse(sh)
        key = f"{name} › {sh}"
        st.session_state["dataframes"][key] = df
        out.append({"text": f"File:{name} Sheet:{sh}\nCols:{','.join(str(c) for c in df.columns)}\nRows:{len(df)}", "source": key, "meta": {"type":"header"}})
        for i in range(0, len(df), 10):
            b = df.iloc[i:i+10]
            out.append({"text": f"[{key} rows {i+1}-{i+len(b)}]\n{b.to_string(index=False)}", "source": key, "meta": {"rows": f"{i+1}-{i+len(b)}"}})
    return out

def parse_pdf(f, name):
    out = []
    with pdfplumber.open(f) as pdf:
        for i, pg in enumerate(pdf.pages):
            t = pg.extract_text() or ""
            if t.strip():
                out.extend(chunk(t, name, {"page": i+1}))
    return out

def parse_docx(f, name):
    d = docx.Document(f)
    return chunk("\n".join(p.text for p in d.paragraphs if p.text.strip()), name, {"type":"doc"})

def parse_json(f, name):
    return chunk(json.dumps(json.load(f), indent=2), name, {"type":"json"})

def parse_text(f, name):
    return chunk(f.read().decode("utf-8", errors="ignore"), name, {"type":"text"})

PARSERS = {".csv": parse_csv, ".xlsx": parse_excel, ".xls": parse_excel,
           ".pdf": parse_pdf, ".docx": parse_docx, ".json": parse_json,
           ".tsv": lambda f, n: parse_csv(f, n),
           ".md": parse_text, ".txt": parse_text}

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
    return [{**chunks[i], "score": float(s)} for s, i in zip(scores[0], ids[0]) if i >= 0]

# ── Gemini ───────────────────────────────────────────────────
def ask(question, ctx, history):
    context = "\n\n---\n\n".join(
        f"[SOURCE {i+1}: {c['source']}" +
        (f" | {', '.join(f'{k}:{v}' for k,v in c['meta'].items())}" if c['meta'] else "") +
        f"]\n{c['text']}"
        for i, c in enumerate(ctx)
    )
    hist = "".join(
        f"{'User' if t['role']=='user' else 'Assistant'}: {t['content']}\n"
        for t in history[-6:]
    )
    prompt = textwrap.dedent(f"""
        You are DataIntern, a precise business data analyst.
        Answer ONLY from the provided sources. If the answer isn't there, say:
        "I don't see that information in the uploaded files."
        Cite every fact with [SOURCE N]. For charts include:
        ```chart
        {{"type":"bar","x":"column","y":"column","source":"filename"}}
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

def get_chart_spec(answer):
    m = re.search(r"```chart\s*(\{.*?\})\s*```", answer, re.DOTALL)
    if m:
        try: return json.loads(m.group(1))
        except: pass
    return None

def render_chart(question, answer):
    dfs = st.session_state["dataframes"]
    if not dfs: return None
    spec = get_chart_spec(answer)
    q = question.lower()

    if spec:
        df = next((dfs[k] for k in dfs if spec.get("source","").lower() in k.lower()), list(dfs.values())[0])
        x, y, t = spec.get("x"), spec.get("y"), spec.get("type","bar")
        if x in df.columns and y in df.columns:
            fns = {"bar": px.bar, "line": px.line, "pie": px.pie, "scatter": px.scatter}
            fn = fns.get(t, px.bar)
            kw = {"names": x, "values": y} if t == "pie" else {"x": x, "y": y}
            return fn(df, title=f"{y} by {x}", **kw)

    # heuristic fallback
    label, df = max(dfs.items(), key=lambda kv: len(kv[1]))
    num = df.select_dtypes(include="number").columns.tolist()
    cat = df.select_dtypes(exclude="number").columns.tolist()
    if not num: return None
    y_col = num[0]; x_col = cat[0] if cat else df.columns[0]
    try:
        agg = df.groupby(x_col)[y_col].sum().reset_index().sort_values(y_col, ascending=False).head(20)
        return px.bar(agg, x=x_col, y=y_col, title=f"{y_col} by {x_col} (from {label})")
    except:
        return None

# ════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════
with st.sidebar:
    # Brand
    st.markdown("## 🔍 DataIntern")
    st.markdown("<p style='color:#8892b0;font-size:0.8rem;margin-top:-10px'>RAG Chatbot · CRM & Business Data</p>", unsafe_allow_html=True)
    st.divider()

    # ── silent API key load (no banner) ──────────────────────
    _key = ""
    try:
        _key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        pass

    if not _key:
        _key = st.text_input("Gemini API Key", type="password",
                             placeholder="Paste key from aistudio.google.com",
                             label_visibility="collapsed")
        st.caption("🔑 [Get a free key →](https://aistudio.google.com)")

    if _key:
        genai.configure(api_key=_key)
        st.session_state["ready"] = True

    st.divider()

    # ── file upload ───────────────────────────────────────────
    st.markdown("**📁 Upload Files**")
    st.caption("CSV · Excel · PDF · Word · JSON · TSV · Markdown")

    files = st.file_uploader(
        "files", label_visibility="collapsed",
        type=["csv","xlsx","xls","pdf","docx","json","tsv","md","txt"],
        accept_multiple_files=True,
    )

    if files:
        if st.button("⚡ Ingest & Index", use_container_width=True, type="primary"):
            model = load_embed()
            all_chunks, bar = [], st.progress(0)
            log = st.empty()
            for i, f in enumerate(files):
                ext = Path(f.name).suffix.lower()
                parser = PARSERS.get(ext)
                if parser:
                    try:
                        nc = parser(f, f.name)
                        all_chunks.extend(nc)
                        log.markdown(f"<div class='ingest-ok'>✓ {f.name} — {len(nc)} chunks</div>", unsafe_allow_html=True)
                    except Exception as e:
                        st.warning(f"⚠ {f.name}: {e}")
                bar.progress((i+1)/len(files))

            if all_chunks:
                with st.spinner("Building vector index…"):
                    idx, emb = build_index(all_chunks, model)
                st.session_state.update({"chunks": all_chunks, "index": idx,
                                          "embeddings": emb, "embed_model": model})
                log.empty(); bar.empty()
                st.success(f"✅ {len(all_chunks)} chunks · {len(files)} files")
            else:
                st.error("No content extracted.")

    # ── stats ─────────────────────────────────────────────────
    if st.session_state["chunks"]:
        st.divider()
        c1, c2 = st.columns(2)
        c1.markdown(f"<div class='stat'><b>{len(st.session_state['chunks'])}</b><span>Chunks</span></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='stat'><b>{len(st.session_state['dataframes'])}</b><span>Tables</span></div>", unsafe_allow_html=True)
        st.divider()

    # ── clear ─────────────────────────────────────────────────
    if st.button("🗑 Clear all", use_container_width=True):
        for k in ["chunks","index","embeddings","history","dataframes","embed_model"]:
            st.session_state[k] = [] if k in ("chunks","history") else ({} if k=="dataframes" else None)
        st.rerun()

# ════════════════════════════════════════════════════════════
# MAIN AREA
# ════════════════════════════════════════════════════════════
st.markdown("# 🔍 DataIntern")
st.markdown("<p style='color:#8892b0;margin-top:-12px'>Ask questions about your business data — cited answers & live charts</p>", unsafe_allow_html=True)

# ── empty state ───────────────────────────────────────────────
if not st.session_state["history"]:
    st.markdown("---")
    st.markdown("#### Try asking…")
    SUGGESTIONS = [
        ("💰", "What was the total closed-won revenue?"),
        ("🏆", "Who is the top rep by pipeline value?"),
        ("📊", "Show me a bar chart of deals by stage"),
        ("⚠️", "List the at-risk accounts"),
        ("🔗", "Does the PDF contract total match the deals sheet?"),
        ("📈", "Which lead sources convert best?"),
    ]
    cols = st.columns(3)
    for i, (icon, s) in enumerate(SUGGESTIONS):
        with cols[i % 3]:
            st.markdown("<div class='sug-btn'>", unsafe_allow_html=True)
            if st.button(f"{icon} {s}", use_container_width=True, key=f"sug{i}"):
                st.session_state["_q"] = s
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<p style='text-align:center;color:#4a4e6a;font-size:0.8rem'>Upload files in the sidebar → click Ingest & Index → start chatting</p>", unsafe_allow_html=True)

# ── chat history ──────────────────────────────────────────────
for turn in st.session_state["history"]:
    with st.chat_message(turn["role"], avatar="👤" if turn["role"]=="user" else "🔍"):
        st.markdown(turn["content"])
        if turn.get("sources"):
            st.markdown(" ".join(f"<span class='pill'>📄 {s}</span>" for s in turn["sources"]), unsafe_allow_html=True)
        if turn.get("fig"):
            st.plotly_chart(turn["fig"], use_container_width=True)

# ── input ─────────────────────────────────────────────────────
question = st.chat_input("Ask about your data…") or st.session_state.pop("_q", None)

if question:
    if not st.session_state["ready"]:
        st.error("Add your Gemini API key in the sidebar first.")
        st.stop()
    if not st.session_state["chunks"]:
        st.error("Upload and ingest files first.")
        st.stop()

    with st.chat_message("user", avatar="👤"):
        st.markdown(question)
    st.session_state["history"].append({"role": "user", "content": question})

    with st.chat_message("assistant", avatar="🔍"):
        with st.spinner("Searching…"):
            ctx = retrieve(question, st.session_state["embed_model"],
                           st.session_state["index"], st.session_state["embeddings"],
                           st.session_state["chunks"])
        with st.spinner("Thinking…"):
            raw = ask(question, ctx, st.session_state["history"])

        display = re.sub(r"```chart.*?```", "", raw, flags=re.DOTALL).strip()
        st.markdown(display)

        sources = list(dict.fromkeys(c["source"] for c in ctx))
        st.markdown(" ".join(f"<span class='pill'>📄 {s}</span>" for s in sources), unsafe_allow_html=True)

        fig = None
        if wants_chart(question):
            fig = render_chart(question, raw)
            if fig:
                fig.update_layout(
                    paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                    font_color="#c0c4d6", title_font_color="#e0e0e0",
                )
                st.plotly_chart(fig, use_container_width=True)

    st.session_state["history"].append({
        "role": "assistant", "content": display,
        "sources": sources, "fig": fig,
    })