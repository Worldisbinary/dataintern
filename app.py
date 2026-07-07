"""
AllDoors Intelligence — RAG Chatbot for CRM & Business Data
API key: Streamlit Cloud Secrets only → GEMINI_API_KEY
Never logged, printed, or shown in UI.
"""

import json, re, textwrap, time
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import faiss
import pdfplumber
import docx
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="AllDoors Intelligence",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* force sidebar always open */
[data-testid="stSidebar"] {
  min-width: 260px !important;
  max-width: 260px !important;
}
[data-testid="collapsedControl"] {
  display: none !important;
}

@import url('https://fonts...
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, .stApp {
  background: #141414;
  color: #E5E5E5;
  font-family: 'Inter', sans-serif;
  font-weight: 300;
}

#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"] { display: none !important; }

/* ══════════════════════════════
   SIDEBAR
══════════════════════════════ */
[data-testid="stSidebar"] {
  background: #000000;
  border-right: 1px solid #222;
}
[data-testid="stSidebar"] > div:first-child {
  padding: 32px 24px;
}
[data-testid="stSidebar"] * {
  font-family: 'Inter', sans-serif !important;
  color: #E5E5E5 !important;
}

.sb-logo {
  font-family: 'Bebas Neue', sans-serif !important;
  font-size: 1.6rem !important;
  letter-spacing: 0.12em;
  color: #FFFFFF !important;
}
.sb-logo span {
  color: #E50914 !important;
}
.sb-divider {
  border: none;
  border-top: 1px solid #222;
  margin: 20px 0;
}
.sb-label {
  font-size: 0.6rem !important;
  font-weight: 500 !important;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: #808080 !important;
  display: block;
  margin-bottom: 8px;
}

/* sidebar buttons */
[data-testid="stSidebar"] .stButton > button {
  font-family: 'Inter', sans-serif !important;
  font-size: 0.8rem !important;
  font-weight: 500 !important;
  letter-spacing: 0.08em !important;
  border-radius: 2px !important;
  text-transform: uppercase !important;
  padding: 10px 0 !important;
  transition: all 0.15s !important;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
  background: #E50914 !important;
  color: #FFFFFF !important;
  border: none !important;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
  background: #F40612 !important;
}
[data-testid="stSidebar"] .stButton > button:not([kind="primary"]) {
  background: transparent !important;
  color: #808080 !important;
  border: 1px solid #333 !important;
}
[data-testid="stSidebar"] .stButton > button:not([kind="primary"]):hover {
  color: #E5E5E5 !important;
  border-color: #555 !important;
}

[data-testid="stSidebar"] [data-testid="stFileUploader"] {
  background: #1A1A1A !important;
  border: 1px dashed #333 !important;
  border-radius: 2px !important;
}
[data-testid="stSidebar"] [data-testid="stTextInput"] input {
  background: #1A1A1A !important;
  border: 1px solid #333 !important;
  border-radius: 2px !important;
  color: #E5E5E5 !important;
}

/* stat cards */
.sb-stats { display: flex; gap: 8px; margin: 10px 0 16px; }
.sb-stat {
  flex: 1;
  background: #1A1A1A;
  border: 1px solid #222;
  padding: 12px 6px;
  text-align: center;
  border-radius: 2px;
}
.sb-stat-n {
  font-family: 'Bebas Neue', sans-serif !important;
  font-size: 1.6rem !important;
  color: #E50914 !important;
  display: block;
  letter-spacing: 0.05em;
}
.sb-stat-l {
  font-size: 0.55rem !important;
  letter-spacing: 0.15em;
  text-transform: uppercase;
  color: #555 !important;
}
.sb-foot {
  font-size: 0.7rem !important;
  color: #444 !important;
  text-align: center;
  margin-top: 24px;
  letter-spacing: 0.08em;
}

/* ══════════════════════════════
   HERO
══════════════════════════════ */
.hero {
  background: linear-gradient(
    to bottom,
    rgba(0,0,0,0.7) 0%,
    rgba(20,20,20,0.4) 60%,
    #141414 100%
  ),
  url('https://images.unsplash.com/photo-1486325212027-8081e485255e?w=1600&q=80&fit=crop') center/cover no-repeat;
  padding: 100px 60px 80px;
  position: relative;
}
.hero-tag {
  font-size: 0.62rem;
  font-weight: 500;
  letter-spacing: 0.25em;
  text-transform: uppercase;
  color: #E50914;
  margin-bottom: 16px;
}
.hero-title {
  font-family: 'Bebas Neue', sans-serif;
  font-size: 5.5rem;
  line-height: 0.95;
  letter-spacing: 0.04em;
  color: #FFFFFF;
  text-shadow: 0 2px 20px rgba(0,0,0,0.8);
}
.hero-title span { color: #E50914; }
.hero-sub {
  font-size: 1rem;
  font-weight: 300;
  color: #B3B3B3;
  margin-top: 16px;
  letter-spacing: 0.02em;
  max-width: 480px;
  line-height: 1.6;
}
.hero-rule {
  border: none;
  border-top: 3px solid #E50914;
  width: 48px;
  margin: 24px 0 0;
}

/* ══════════════════════════════
   CONTENT
══════════════════════════════ */
.content { padding: 40px 60px 20px; }

.row-label {
  font-size: 0.65rem;
  font-weight: 500;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: #808080;
  margin-bottom: 16px;
  display: flex;
  align-items: center;
  gap: 14px;
}
.row-label::after {
  content: '';
  flex: 1;
  border-top: 1px solid #222;
}

/* suggestion cards — Netflix tile style */
.tile-row {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 4px;
  margin-bottom: 40px;
}
.tile {
  background: #1F1F1F;
  padding: 20px 22px;
  cursor: pointer;
  border-left: 3px solid transparent;
  transition: all 0.18s;
  position: relative;
}
.tile:hover {
  background: #2A2A2A;
  border-left-color: #E50914;
  transform: scale(1.015);
}
.tile-n {
  font-family: 'Bebas Neue', sans-serif;
  font-size: 1.8rem;
  color: #333;
  line-height: 1;
  margin-bottom: 8px;
  letter-spacing: 0.05em;
  transition: color 0.18s;
}
.tile:hover .tile-n { color: #E50914; }
.tile-t {
  font-size: 0.88rem;
  font-weight: 300;
  color: #B3B3B3;
  line-height: 1.4;
}
.tile:hover .tile-t { color: #E5E5E5; }

/* stButton INSIDE tile — invisible click layer */
.stButton > button {
  position: absolute !important;
  inset: 0 !important;
  opacity: 0 !important;
  width: 100% !important;
  height: 100% !important;
  cursor: pointer !important;
  border: none !important;
  background: transparent !important;
}

/* ══════════════════════════════
   CHAT
══════════════════════════════ */
[data-testid="stChatMessage"] {
  background: transparent !important;
  border: none !important;
  border-bottom: 1px solid #1F1F1F !important;
  padding: 28px 0 !important;
  border-radius: 0 !important;
}
[data-testid="stChatMessage"] p {
  font-family: 'Inter', sans-serif !important;
  font-weight: 300 !important;
  font-size: 0.95rem !important;
  line-height: 1.8 !important;
  color: #E5E5E5 !important;
}
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] { display: none !important; }

[data-testid="stChatInput"] {
  background: #0A0A0A;
  border-top: 1px solid #222;
  padding: 16px 60px !important;
}
[data-testid="stChatInput"] textarea {
  background: #1A1A1A !important;
  border: 1px solid #333 !important;
  border-radius: 2px !important;
  color: #E5E5E5 !important;
  font-family: 'Inter', sans-serif !important;
  font-weight: 300 !important;
  font-size: 0.95rem !important;
}
[data-testid="stChatInput"] textarea::placeholder {
  color: #555 !important;
}

/* source pills */
.pill {
  display: inline-block;
  background: #1F1F1F;
  border-left: 2px solid #E50914;
  padding: 2px 10px;
  font-size: 0.68rem;
  font-weight: 400;
  letter-spacing: 0.04em;
  color: #808080;
  margin: 4px 6px 0 0;
}

/* error / rate limit box */
.rate-box {
  background: #1A0A0A;
  border: 1px solid #E50914;
  border-radius: 2px;
  padding: 16px 20px;
  font-size: 0.88rem;
  color: #E5E5E5;
  margin-top: 8px;
}
</style>
""", unsafe_allow_html=True)

# ── constants ────────────────────────────────────────────────
EMBED_MODEL  = "all-MiniLM-L6-v2"
CHUNK_SIZE   = 400; OVERLAP = 80; TOP_K = 8
GEMINI_MODEL = "gemini-2.0-flash"
CHART_KW     = ["chart","plot","graph","visuali","bar","pie","line",
                 "scatter","histogram","show","compare","distribution","trend"]
MAX_RETRIES  = 3   # retry on rate limit

for k,v in {"chunks":[],"index":None,"embeddings":None,
             "history":[],"dataframes":{},"ready":False,"embed_model":None}.items():
    if k not in st.session_state: st.session_state[k] = v

@st.cache_resource(show_spinner="Loading model…")
def load_embed(): return SentenceTransformer(EMBED_MODEL)

# ── parsers ──────────────────────────────────────────────────
def chunk(text, src, meta):
    out,s = [],0
    while s < len(text):
        c = text[s:s+CHUNK_SIZE].strip()
        if c: out.append({"text":c,"source":src,"meta":meta.copy()})
        s += CHUNK_SIZE - OVERLAP
    return out

def parse_csv(f,name):
    df=pd.read_csv(f); st.session_state["dataframes"][name]=df
    rows=[{"text":f"File:{name}\nCols:{','.join(df.columns)}\nRows:{len(df)}","source":name,"meta":{"type":"header"}}]
    for i in range(0,len(df),10):
        b=df.iloc[i:i+10]
        rows.append({"text":f"[{name} rows {i+1}-{i+len(b)}]\n{b.to_string(index=False)}","source":name,"meta":{"rows":f"{i+1}-{i+len(b)}"}})
    return rows

def parse_excel(f,name):
    xl,out=pd.ExcelFile(f),[]
    for sh in xl.sheet_names:
        df=xl.parse(sh); key=f"{name}/{sh}"; st.session_state["dataframes"][key]=df
        out.append({"text":f"File:{name} Sheet:{sh}\nCols:{','.join(str(c) for c in df.columns)}\nRows:{len(df)}","source":key,"meta":{"type":"header"}})
        for i in range(0,len(df),10):
            b=df.iloc[i:i+10]
            out.append({"text":f"[{key} rows {i+1}-{i+len(b)}]\n{b.to_string(index=False)}","source":key,"meta":{"rows":f"{i+1}-{i+len(b)}"}})
    return out

def parse_pdf(f,name):
    out=[]
    with pdfplumber.open(f) as pdf:
        for i,pg in enumerate(pdf.pages):
            t=pg.extract_text() or ""
            if t.strip(): out.extend(chunk(t,name,{"page":i+1}))
    return out

def parse_docx(f,name):
    d=docx.Document(f)
    return chunk("\n".join(p.text for p in d.paragraphs if p.text.strip()),name,{"type":"doc"})

def parse_json(f,name):
    return chunk(json.dumps(json.load(f),indent=2),name,{"type":"json"})

def parse_text(f,name):
    return chunk(f.read().decode("utf-8",errors="ignore"),name,{"type":"text"})

PARSERS={".csv":parse_csv,".xlsx":parse_excel,".xls":parse_excel,
         ".pdf":parse_pdf,".docx":parse_docx,".json":parse_json,
         ".tsv":lambda f,n:parse_csv(f,n),".md":parse_text,".txt":parse_text}

def build_index(chunks,model):
    emb=model.encode([c["text"] for c in chunks],show_progress_bar=False,batch_size=64).astype(np.float32)
    faiss.normalize_L2(emb); idx=faiss.IndexFlatIP(emb.shape[1]); idx.add(emb)
    return idx,emb

def retrieve(q,model,idx,emb,chunks):
    qe=model.encode([q]).astype(np.float32); faiss.normalize_L2(qe)
    scores,ids=idx.search(qe,TOP_K)
    return [{**chunks[i],"score":float(s)} for s,i in zip(scores[0],ids[0]) if i>=0]

def ask(question, ctx, history):
    """Calls Gemini with exponential back-off on rate-limit errors."""
    context="\n\n---\n\n".join(
        f"[SOURCE {i+1}: {c['source']}"+(f" | {', '.join(f'{k}:{v}' for k,v in c['meta'].items())}" if c['meta'] else "")+f"]\n{c['text']}"
        for i,c in enumerate(ctx))
    hist="".join(f"{'User' if t['role']=='user' else 'Assistant'}: {t['content']}\n" for t in history[-6:])
    prompt=textwrap.dedent(f"""
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

    model_obj = genai.GenerativeModel(GEMINI_MODEL)
    for attempt in range(MAX_RETRIES):
        try:
            return model_obj.generate_content(prompt).text
        except ResourceExhausted:
            if attempt < MAX_RETRIES - 1:
                wait = 15 * (attempt + 1)   # 15s, 30s, 45s
                time.sleep(wait)
            else:
                return (
                    "**Rate limit reached.** Gemini's free tier allows 15 requests/minute. "
                    "Please wait 30–60 seconds and try again. "
                    "For higher limits, add billing at [console.cloud.google.com](https://console.cloud.google.com)."
                )
        except Exception as e:
            return f"**Error:** {str(e)[:200]}"

def wants_chart(q): return any(k in q.lower() for k in CHART_KW)

def get_spec(answer):
    m=re.search(r"```chart\s*(\{.*?\})\s*```",answer,re.DOTALL)
    if m:
        try: return json.loads(m.group(1))
        except: pass
    return None

CLAYOUT=dict(
    paper_bgcolor="#141414", plot_bgcolor="#1F1F1F",
    font=dict(family="Inter,sans-serif",color="#B3B3B3",size=12),
    title_font=dict(family="Bebas Neue,sans-serif",size=22,color="#FFFFFF",letterSpacing=2),
    colorway=["#E50914","#B3B3B3","#E5E5E5","#831010","#666666","#FF4444"],
    margin=dict(t=52,b=32,l=32,r=32),
)

def render_chart(question,answer):
    dfs=st.session_state["dataframes"]
    if not dfs: return None
    spec=get_spec(answer)
    if spec:
        df=next((dfs[k] for k in dfs if spec.get("source","").lower() in k.lower()),list(dfs.values())[0])
        x,y,t=spec.get("x"),spec.get("y"),spec.get("type","bar")
        if x in df.columns and y in df.columns:
            fns={"bar":px.bar,"line":px.line,"pie":px.pie,"scatter":px.scatter}
            kw={"names":x,"values":y} if t=="pie" else {"x":x,"y":y}
            fig=fns.get(t,px.bar)(df,title=f"{y} by {x}",**kw)
            fig.update_layout(**CLAYOUT); return fig
    label,df=max(dfs.items(),key=lambda kv:len(kv[1]))
    num=df.select_dtypes(include="number").columns.tolist()
    cat=df.select_dtypes(exclude="number").columns.tolist()
    if not num: return None
    y_col=num[0]; x_col=cat[0] if cat else df.columns[0]
    try:
        agg=df.groupby(x_col)[y_col].sum().reset_index().sort_values(y_col,ascending=False).head(20)
        fig=px.bar(agg,x=x_col,y=y_col,title=f"{y_col} by {x_col}")
        fig.update_layout(**CLAYOUT); return fig
    except: return None

# ════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class='sb-logo'>All<span>Doors</span></div>
    <hr class='sb-divider'>
    """, unsafe_allow_html=True)

    # API key — silent from secrets, never shown
    _key=""
    try: _key=st.secrets.get("GEMINI_API_KEY","")
    except: pass
    if not _key:
        st.markdown("<span class='sb-label'>API Key</span>",unsafe_allow_html=True)
        _key=st.text_input("k",type="password",placeholder="Gemini key…",label_visibility="collapsed")
        st.caption("Free at [aistudio.google.com](https://aistudio.google.com)")
    if _key:
        genai.configure(api_key=_key)
        st.session_state["ready"]=True

    st.markdown("<hr class='sb-divider'><span class='sb-label'>Documents</span>",unsafe_allow_html=True)
    st.caption("CSV · Excel · PDF · Word · JSON")

    files=st.file_uploader("f",label_visibility="collapsed",
        type=["csv","xlsx","xls","pdf","docx","json","tsv","md","txt"],
        accept_multiple_files=True)

    if files:
        if st.button("Index Documents",use_container_width=True,type="primary"):
            model=load_embed(); all_chunks,bar=[],st.progress(0); status=st.empty()
            for i,f in enumerate(files):
                ext=Path(f.name).suffix.lower(); parser=PARSERS.get(ext)
                if parser:
                    try:
                        nc=parser(f,f.name); all_chunks.extend(nc)
                        status.caption(f"{f.name} — {len(nc)} passages")
                    except Exception as e: st.warning(f"{f.name}: {e}")
                bar.progress((i+1)/len(files))
            if all_chunks:
                with st.spinner("Building index…"):
                    idx,emb=build_index(all_chunks,model)
                st.session_state.update({"chunks":all_chunks,"index":idx,"embeddings":emb,"embed_model":model})
                bar.empty(); status.empty()
                st.success(f"{len(all_chunks)} passages indexed.")
            else: st.error("No content extracted.")

    if st.session_state["chunks"]:
        st.markdown(f"""
        <div class='sb-stats'>
          <div class='sb-stat'><span class='sb-stat-n'>{len(st.session_state['chunks'])}</span><span class='sb-stat-l'>Passages</span></div>
          <div class='sb-stat'><span class='sb-stat-n'>{len(st.session_state['dataframes'])}</span><span class='sb-stat-l'>Tables</span></div>
        </div>""",unsafe_allow_html=True)
        if st.button("Clear session",use_container_width=True):
            for k in ["chunks","index","embeddings","history","dataframes","embed_model"]:
                st.session_state[k]=[] if k in ("chunks","history") else ({} if k=="dataframes" else None)
            st.rerun()

    st.markdown("<div class='sb-foot'>alldoors.in &nbsp;·&nbsp; Team C</div>",unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# HERO
# ════════════════════════════════════════════════════════════
st.markdown("""
<div class='hero'>
  <div class='hero-tag'>AllDoors &nbsp;·&nbsp; Real Estate Intelligence</div>
  <div class='hero-title'>We Are<br><span>Team C.</span></div>
  <div class='hero-sub'>
    Ask questions about your CRM data. Get cited answers and live charts — instantly.
  </div>
  <hr class='hero-rule'>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# SUGGESTIONS
# ════════════════════════════════════════════════════════════
SUGGESTIONS=[
    "What was the total closed-won revenue?",
    "Who is the top rep by pipeline value?",
    "Show a bar chart of deals by stage",
    "List the at-risk accounts",
    "Does the PDF contract total match the deals sheet?",
    "Which lead sources convert best?",
]

if not st.session_state["history"]:
    st.markdown("<div class='content'><div class='row-label'>Suggested Queries</div>", unsafe_allow_html=True)
    cols=st.columns(3,gap="small")
    for i,s in enumerate(SUGGESTIONS):
        with cols[i%3]:
            st.markdown(f"""
            <div class='tile' style='position:relative'>
              <div class='tile-n'>0{i+1}</div>
              <div class='tile-t'>{s}</div>
            </div>""", unsafe_allow_html=True)
            if st.button(s,key=f"s{i}",use_container_width=True):
                st.session_state["_q"]=s; st.rerun()

    st.markdown("""
    <p style='font-family:Inter,sans-serif;font-weight:300;color:#555;
    font-size:0.82rem;text-align:center;margin-top:36px;letter-spacing:0.04em'>
    Upload your documents in the sidebar — index them — then begin your enquiry below.
    </p></div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# CHAT
# ════════════════════════════════════════════════════════════
st.markdown("<div class='content'>",unsafe_allow_html=True)

for turn in st.session_state["history"]:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("sources"):
            st.markdown(" ".join(f"<span class='pill'>{s}</span>" for s in turn["sources"]),unsafe_allow_html=True)
        if turn.get("fig"):
            st.plotly_chart(turn["fig"],use_container_width=True)

st.markdown("</div>",unsafe_allow_html=True)

question=st.chat_input("Ask about your data…") or st.session_state.pop("_q",None)

if question:
    if not st.session_state["ready"]:
        st.error("Add your Gemini API key in the sidebar.")
        st.stop()
    if not st.session_state["chunks"]:
        st.error("Upload and index documents first.")
        st.stop()

    with st.chat_message("user"): st.markdown(question)
    st.session_state["history"].append({"role":"user","content":question})

    with st.chat_message("assistant"):
        with st.spinner("Searching…"):
            ctx=retrieve(question,st.session_state["embed_model"],
                         st.session_state["index"],st.session_state["embeddings"],
                         st.session_state["chunks"])
        with st.spinner("Composing answer…"):
            raw=ask(question,ctx,st.session_state["history"])

        display=re.sub(r"```chart.*?```","",raw,flags=re.DOTALL).strip()
        st.markdown(display)
        sources=list(dict.fromkeys(c["source"] for c in ctx))
        st.markdown(" ".join(f"<span class='pill'>{s}</span>" for s in sources),unsafe_allow_html=True)

        fig=None
        if wants_chart(question):
            fig=render_chart(question,raw)
            if fig: st.plotly_chart(fig,use_container_width=True)

    st.session_state["history"].append({"role":"assistant","content":display,"sources":sources,"fig":fig})