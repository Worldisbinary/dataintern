"""
============================================================
DataIntern — RAG Chatbot for CRM & Business Data
============================================================
Stack:
  - Streamlit          : UI + file upload
  - Google Gemini API  : LLM (gemini-1.5-flash, free tier)
  - sentence-transformers: local free embeddings (all-MiniLM-L6-v2)
  - FAISS              : in-memory vector index
  - Plotly             : inline charts
  - pdfplumber         : PDF parsing
  - python-docx        : Word doc parsing
  - openpyxl / pandas  : Excel + CSV parsing

Flow per query:
  1. Embed the question
  2. FAISS top-k retrieval from the chunk index
  3. Build a grounded prompt (context + citations)
  4. Gemini generates a cited answer
  5. If chart is detected → Plotly renders inline
============================================================
"""

import os, json, re, io, textwrap
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

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="DataIntern — CRM Chatbot",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── minimal custom CSS (clean, dark-accent header) ──────────
st.markdown("""
<style>
  .block-container { padding-top: 1.5rem; }
  h1 { font-size: 1.6rem !important; }
  .source-pill {
    display: inline-block; background: #f0f2f6;
    border-radius: 4px; padding: 2px 8px;
    font-size: 0.75rem; color: #444; margin: 2px;
  }
  .stChatMessage { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# CONSTANTS
# ============================================================
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"   # free, runs locally, 384-dim
CHUNK_SIZE       = 400                    # characters per chunk
CHUNK_OVERLAP    = 80
TOP_K            = 8                      # chunks retrieved per query
GEMINI_MODEL     = "gemini-1.5-flash"    # free-tier model

# Chart-request keywords — if any appear, we try to plot
CHART_KEYWORDS = [
    "chart", "plot", "graph", "visuali", "bar", "pie",
    "line", "scatter", "histogram", "show me", "compare",
    "distribution", "trend",
]

# ============================================================
# SESSION STATE INITIALISATION
# ============================================================
def _init_state():
    defaults = {
        "chunks":       [],   # list of {"text", "source", "meta"}
        "index":        None, # FAISS index
        "embeddings":   None, # np array of chunk embeddings
        "chat_history": [],   # list of {"role", "content", "sources"}
        "dataframes":   {},   # filename → pd.DataFrame (for charting)
        "gemini_ready": False,
        "embed_model":  None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ============================================================
# EMBEDDING MODEL (cached so it loads only once per session)
# ============================================================
@st.cache_resource(show_spinner="Loading embedding model…")
def load_embed_model():
    return SentenceTransformer(EMBED_MODEL_NAME)

# ============================================================
# FILE PARSERS — each returns list of {"text", "source", "meta"}
# ============================================================

def _chunk_text(text: str, source: str, meta: dict) -> list[dict]:
    """Splits a long string into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end].strip()
        if chunk:
            chunks.append({"text": chunk, "source": source, "meta": meta.copy()})
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def parse_csv(file, filename: str) -> list[dict]:
    """CSV → row-by-row text chunks + stores DataFrame for charting."""
    df = pd.read_csv(file)
    st.session_state["dataframes"][filename] = df
    chunks = []
    # Header description chunk
    header_text = f"File: {filename}\nColumns: {', '.join(df.columns)}\nRow count: {len(df)}"
    chunks.append({"text": header_text, "source": filename, "meta": {"type": "header"}})
    # Every 10 rows → one chunk
    for i in range(0, len(df), 10):
        batch = df.iloc[i : i + 10]
        text = f"[{filename} rows {i+1}–{i+len(batch)}]\n" + batch.to_string(index=False)
        chunks.append({"text": text, "source": filename, "meta": {"rows": f"{i+1}-{i+len(batch)}"}})
    return chunks


def parse_excel(file, filename: str) -> list[dict]:
    """Excel → parse every sheet; store each as a DataFrame."""
    xl = pd.ExcelFile(file)
    chunks = []
    for sheet in xl.sheet_names:
        df = xl.parse(sheet)
        key = f"{filename} › {sheet}"
        st.session_state["dataframes"][key] = df
        chunks.append({
            "text": f"File: {filename}, Sheet: {sheet}\nColumns: {', '.join(str(c) for c in df.columns)}\nRows: {len(df)}",
            "source": key, "meta": {"type": "header"},
        })
        for i in range(0, len(df), 10):
            batch = df.iloc[i : i + 10]
            text = f"[{key} rows {i+1}–{i+len(batch)}]\n" + batch.to_string(index=False)
            chunks.append({"text": text, "source": key, "meta": {"rows": f"{i+1}-{i+len(batch)}"}})
    return chunks


def parse_pdf(file, filename: str) -> list[dict]:
    """PDF → page-by-page text chunks using pdfplumber."""
    chunks = []
    with pdfplumber.open(file) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            if text.strip():
                for c in _chunk_text(text, filename, {"page": i + 1}):
                    chunks.append(c)
    return chunks


def parse_docx(file, filename: str) -> list[dict]:
    """Word doc → paragraph text chunks."""
    doc = docx.Document(file)
    full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return _chunk_text(full_text, filename, {"type": "paragraph"})


def parse_json(file, filename: str) -> list[dict]:
    """JSON → pretty-printed chunks."""
    data = json.load(file)
    text = json.dumps(data, indent=2)
    return _chunk_text(text, filename, {"type": "json"})


def parse_tsv(file, filename: str) -> list[dict]:
    """TSV treated the same as CSV."""
    df = pd.read_csv(file, sep="\t")
    st.session_state["dataframes"][filename] = df
    chunks = []
    chunks.append({
        "text": f"File: {filename}\nColumns: {', '.join(df.columns)}\nRow count: {len(df)}",
        "source": filename, "meta": {"type": "header"},
    })
    for i in range(0, len(df), 10):
        batch = df.iloc[i : i + 10]
        text = f"[{filename} rows {i+1}–{i+len(batch)}]\n" + batch.to_string(index=False)
        chunks.append({"text": text, "source": filename, "meta": {"rows": f"{i+1}-{i+len(batch)}"}})
    return chunks


def parse_md(file, filename: str) -> list[dict]:
    """Markdown / plain text → text chunks."""
    text = file.read().decode("utf-8", errors="ignore")
    return _chunk_text(text, filename, {"type": "markdown"})


PARSERS = {
    ".csv":  parse_csv,
    ".xlsx": parse_excel,
    ".xls":  parse_excel,
    ".pdf":  parse_pdf,
    ".docx": parse_docx,
    ".json": parse_json,
    ".tsv":  parse_tsv,
    ".md":   parse_md,
    ".txt":  parse_md,
}

# ============================================================
# INDEX BUILDER
# ============================================================

def build_index(chunks: list[dict], model) -> faiss.IndexFlatIP:
    """Embeds all chunks and stores them in a FAISS inner-product index."""
    texts = [c["text"] for c in chunks]
    # Batch embed (sentence-transformers handles batching internally)
    embeddings = model.encode(texts, show_progress_bar=False, batch_size=64)
    embeddings = embeddings.astype(np.float32)
    # Normalise → inner product == cosine similarity
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings


def retrieve(query: str, model, index, embeddings, chunks, k=TOP_K) -> list[dict]:
    """Returns the top-k most relevant chunks for a query."""
    q_emb = model.encode([query]).astype(np.float32)
    faiss.normalize_L2(q_emb)
    scores, idxs = index.search(q_emb, k)
    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx >= 0:
            results.append({**chunks[idx], "score": float(score)})
    return results

# ============================================================
# GEMINI ANSWER GENERATION
# ============================================================

def ask_gemini(question: str, context_chunks: list[dict], history: list[dict]) -> str:
    """
    Builds a grounded prompt from retrieved chunks and chat history,
    calls Gemini, and returns the answer string.
    """
    # Build context block with source labels
    context_parts = []
    for i, c in enumerate(context_chunks):
        meta_str = ", ".join(f"{k}: {v}" for k, v in c["meta"].items()) if c["meta"] else ""
        label = f"[SOURCE {i+1}: {c['source']}" + (f" | {meta_str}" if meta_str else "") + "]"
        context_parts.append(f"{label}\n{c['text']}")
    context_block = "\n\n---\n\n".join(context_parts)

    # Build chat history string (last 6 turns for context window efficiency)
    history_str = ""
    for turn in history[-6:]:
        role = "User" if turn["role"] == "user" else "Assistant"
        history_str += f"{role}: {turn['content']}\n"

    system_prompt = textwrap.dedent(f"""
        You are DataIntern, a precise business data analyst.
        Answer ONLY from the provided source documents.
        If the answer is not in the sources, say exactly:
        "I don't see that information in the uploaded files."

        Rules:
        - Cite every fact with [SOURCE N] where N matches the context block.
        - For numbers, quote them exactly as they appear in the source.
        - Be concise but complete.
        - If asked to chart/visualise, describe what to chart and include
          a JSON block like: ```chart\n{{"type":"bar","x":"column","y":"column","source":"filename"}}\n```
    """).strip()

    full_prompt = f"""{system_prompt}

--- CONVERSATION HISTORY ---
{history_str}

--- RETRIEVED CONTEXT ---
{context_block}

--- QUESTION ---
{question}

--- ANSWER ---"""

    model = genai.GenerativeModel(GEMINI_MODEL)
    response = model.generate_content(full_prompt)
    return response.text

# ============================================================
# CHART DETECTION & RENDERING
# ============================================================

def _wants_chart(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in CHART_KEYWORDS)


def _extract_chart_spec(answer: str) -> dict | None:
    """Pulls the ```chart ... ``` JSON block from the LLM answer if present."""
    match = re.search(r"```chart\s*(\{.*?\})\s*```", answer, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            return None
    return None


def render_chart(question: str, answer: str) -> go.Figure | None:
    """
    Tries to render a chart.
    Priority: (1) LLM-specified chart spec, (2) heuristic from question keywords.
    Returns a Plotly figure or None.
    """
    dfs = st.session_state["dataframes"]
    if not dfs:
        return None

    spec = _extract_chart_spec(answer)

    # If LLM gave us a spec, use it
    if spec:
        src = spec.get("source", "")
        # Find the best matching DataFrame
        df = None
        for key in dfs:
            if src.lower() in key.lower() or key.lower() in src.lower():
                df = dfs[key]
                break
        if df is None:
            df = list(dfs.values())[0]  # fall back to first

        x_col = spec.get("x")
        y_col = spec.get("y")
        chart_type = spec.get("type", "bar")

        # Validate columns exist
        if x_col in df.columns and y_col in df.columns:
            if chart_type == "bar":
                return px.bar(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
            elif chart_type == "line":
                return px.line(df, x=x_col, y=y_col, title=f"{y_col} over {x_col}")
            elif chart_type == "pie":
                return px.pie(df, names=x_col, values=y_col, title=f"{y_col} by {x_col}")
            elif chart_type == "scatter":
                return px.scatter(df, x=x_col, y=y_col, title=f"{x_col} vs {y_col}")

    # Heuristic fallback: pick the largest numeric DataFrame and guess columns
    q = question.lower()
    best_df = max(dfs.items(), key=lambda kv: len(kv[1]), default=(None, None))
    if best_df[1] is None:
        return None
    label, df = best_df
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()

    if not num_cols:
        return None

    y_col = num_cols[0]
    # Try to pick a sensible x column
    x_col = cat_cols[0] if cat_cols else df.columns[0]

    title = f"{y_col} by {x_col} (from {label})"

    if "pie" in q:
        return px.pie(df.head(15), names=x_col, values=y_col, title=title)
    elif "line" in q or "trend" in q:
        return px.line(df, x=x_col, y=y_col, title=title)
    elif "scatter" in q:
        if len(num_cols) >= 2:
            return px.scatter(df, x=num_cols[0], y=num_cols[1], title=f"{num_cols[0]} vs {num_cols[1]}")
    else:
        # Default: bar chart, aggregate if needed
        try:
            agg = df.groupby(x_col)[y_col].sum().reset_index().sort_values(y_col, ascending=False).head(20)
            return px.bar(agg, x=x_col, y=y_col, title=title)
        except Exception:
            return px.bar(df.head(20), x=x_col, y=y_col, title=title)

# ============================================================
# SIDEBAR — API key + file upload
# ============================================================

with st.sidebar:
    st.title("⚙️ DataIntern Setup")

    # Auto-load from Streamlit Cloud secrets if available
    _secret_key = st.secrets.get("GEMINI_API_KEY", "") if hasattr(st, "secrets") else ""

    # API key input
    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        value=_secret_key,
        placeholder="Paste your key here…",
        help="Get it free at aistudio.google.com → Get API Key",
    )
    if api_key:
        try:
            genai.configure(api_key=api_key)
            # Quick validation ping
            genai.GenerativeModel(GEMINI_MODEL).generate_content("ping")
            st.success("✅ API key valid")
            st.session_state["gemini_ready"] = True
        except Exception as e:
            st.error(f"❌ Invalid key: {e}")
            st.session_state["gemini_ready"] = False

    st.divider()

    # File uploader
    uploaded_files = st.file_uploader(
        "Upload your files",
        type=["csv", "xlsx", "xls", "pdf", "docx", "json", "tsv", "md", "txt"],
        accept_multiple_files=True,
        help="Supported: CSV, Excel, PDF, Word, JSON, TSV, Markdown",
    )

    if uploaded_files:
        if st.button("📥 Ingest files", use_container_width=True):
            model = load_embed_model()
            all_chunks = []
            progress = st.progress(0)
            for i, f in enumerate(uploaded_files):
                ext = Path(f.name).suffix.lower()
                parser = PARSERS.get(ext)
                if parser:
                    try:
                        new_chunks = parser(f, f.name)
                        all_chunks.extend(new_chunks)
                        st.write(f"✓ {f.name} → {len(new_chunks)} chunks")
                    except Exception as e:
                        st.warning(f"⚠️ {f.name}: {e}")
                else:
                    st.warning(f"⚠️ Unsupported format: {f.name}")
                progress.progress((i + 1) / len(uploaded_files))

            if all_chunks:
                with st.spinner("Building vector index…"):
                    index, embeddings = build_index(all_chunks, model)
                st.session_state["chunks"]     = all_chunks
                st.session_state["index"]      = index
                st.session_state["embeddings"] = embeddings
                st.session_state["embed_model"] = model
                st.success(f"✅ {len(all_chunks)} chunks indexed from {len(uploaded_files)} file(s)")
            else:
                st.error("No chunks extracted — check file formats.")

    # Stats
    if st.session_state["chunks"]:
        st.divider()
        st.metric("Chunks indexed", len(st.session_state["chunks"]))
        st.metric("Files with tables", len(st.session_state["dataframes"]))

    # Clear button
    if st.button("🗑️ Clear everything", use_container_width=True):
        for k in ["chunks", "index", "embeddings", "chat_history", "dataframes", "embed_model"]:
            st.session_state[k] = [] if k in ("chunks", "chat_history") else ({} if k == "dataframes" else None)
        st.rerun()

# ============================================================
# MAIN CHAT UI
# ============================================================

st.title("📊 DataIntern — CRM & Business Data Chatbot")
st.caption("Ask questions about your uploaded files. Get cited answers + live charts.")

# Show starter suggestions if no conversation yet
if not st.session_state["chat_history"]:
    st.info("👆 Upload your files in the sidebar and click **Ingest files**, then ask anything below.")
    cols = st.columns(2)
    suggestions = [
        "What was the total closed-won revenue?",
        "Who is the top rep by pipeline value?",
        "Show me a bar chart of deals by stage",
        "List the at-risk accounts",
        "Which lead sources convert best?",
        "Does the PDF contract total match the deals sheet?",
    ]
    for i, s in enumerate(suggestions):
        if cols[i % 2].button(s, use_container_width=True, key=f"sug_{i}"):
            st.session_state["_pending_question"] = s
            st.rerun()

# Render existing chat history
for turn in st.session_state["chat_history"]:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        # Show source pills
        if turn.get("sources"):
            src_html = " ".join(f'<span class="source-pill">📄 {s}</span>' for s in turn["sources"])
            st.markdown(src_html, unsafe_allow_html=True)
        # Re-render chart if stored
        if turn.get("figure"):
            st.plotly_chart(turn["figure"], use_container_width=True)

# ── chat input ───────────────────────────────────────────────
question = st.chat_input("Ask about your data…") or st.session_state.pop("_pending_question", None)

if question:
    # Guard checks
    if not st.session_state["gemini_ready"]:
        st.error("Please enter a valid Gemini API key in the sidebar first.")
        st.stop()
    if not st.session_state["chunks"]:
        st.error("Please upload and ingest at least one file first.")
        st.stop()

    # Show user message
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state["chat_history"].append({"role": "user", "content": question})

    # Retrieve + generate
    with st.chat_message("assistant"):
        with st.spinner("Searching your files…"):
            retrieved = retrieve(
                question,
                st.session_state["embed_model"],
                st.session_state["index"],
                st.session_state["embeddings"],
                st.session_state["chunks"],
            )

        with st.spinner("Generating answer…"):
            answer = ask_gemini(question, retrieved, st.session_state["chat_history"])

        # Clean up the chart spec block from displayed answer
        display_answer = re.sub(r"```chart.*?```", "", answer, flags=re.DOTALL).strip()
        st.markdown(display_answer)

        # Unique sources cited
        sources = list(dict.fromkeys(c["source"] for c in retrieved))
        src_html = " ".join(f'<span class="source-pill">📄 {s}</span>' for s in sources)
        st.markdown(src_html, unsafe_allow_html=True)

        # Chart rendering
        fig = None
        if _wants_chart(question):
            with st.spinner("Rendering chart…"):
                fig = render_chart(question, answer)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

    # Store in history
    st.session_state["chat_history"].append({
        "role": "assistant",
        "content": display_answer,
        "sources": sources,
        "figure": fig,
    })
