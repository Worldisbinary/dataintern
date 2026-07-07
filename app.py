"""
AllDoors Intelligence — RAG Chatbot for CRM & Business Data
API key: Streamlit Cloud Secrets only → GEMINI_API_KEY
Never logged, printed, or shown in UI.
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
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300;1,400&family=Jost:wght@200;300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, .stApp {
  font-family: 'Jost', sans-serif;
  font-weight: 300;
  background: #F0EBE1;
  color: #1C1A16;
}
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"] { display: none !important; }

/* ═══════════════════════════════════════
   SIDEBAR
═══════════════════════════════════════ */
[data-testid="stSidebar"] {
  background: #1C1A16;
  border-right: none;
}
[data-testid="stSidebar"] > div:first-child {
  padding: 36px 24px 32px;
}
[data-testid="stSidebar"] * {
  color: #E8E0D0 !important;
  font-family: 'Jost', sans-serif !important;
}

.sb-brand {
  font-family: 'Cormorant Garamond', serif !important;
  font-size: 1.4rem !important;
  font-weight: 400 !important;
  letter-spacing: 0.12em;
  color: #E8E0D0 !important;
  text-transform: uppercase;
}
.sb-sub {
  font-size: 0.58rem !important;
  font-weight: 500 !important;
  letter-spacing: 0.25em;
  text-transform: uppercase;
  color: #7A7060 !important;
  margin-top: 4px;
  display: block;
}
.sb-rule { border: none; border-top: 1px solid #2E2B24; margin: 20px 0; }
.sb-lbl {
  font-size: 0.58rem !important;
  font-weight: 500 !important;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: #7A7060 !important;
  display: block;
  margin-bottom: 8px;
}

/* sidebar buttons */
[data-testid="stSidebar"] .stButton > button {
  font-family: 'Jost', sans-serif !important;
  font-size: 0.78rem !important;
  font-weight: 400 !important;
  letter-spacing: 0.1em !important;
  border-radius: 0 !important;
  text-transform: uppercase !important;
  padding: 10px 0 !important;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
  background: #C4A96A !important;
  color: #1C1A16 !important;
  border: none !important;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
  background: #D4B97A !important;
}
[data-testid="stSidebar"] .stButton > button:not([kind="primary"]) {
  background: transparent !important;
  color: #7A7060 !important;
  border: 1px solid #2E2B24 !important;
}
[data-testid="stSidebar"] .stButton > button:not([kind="primary"]):hover {
  border-color: #7A7060 !important;
  color: #E8E0D0 !important;
}

/* file uploader inside sidebar */
[data-testid="stSidebar"] [data-testid="stFileUploader"] {
  background: #242018 !important;
  border: 1px dashed #3A3628 !important;
  border-radius: 0 !important;
}

/* stat cards */
.sb-stats { display: flex; gap: 6px; margin: 8px 0 16px; }
.sb-stat {
  flex: 1;
  background: #242018;
  border: 1px solid #2E2B24;
  padding: 12px 6px;
  text-align: center;
}
.sb-stat-n {
  font-family: 'Cormorant Garamond', serif !important;
  font-size: 1.5rem !important;
  color: #C4A96A !important;
  display: block;
}
.sb-stat-l {
  font-size: 0.55rem !important;
  letter-spacing: 0.15em;
  text-transform: uppercase;
  color: #7A7060 !important;
}
.sb-foot {
  font-family: 'Cormorant Garamond', serif !important;
  font-style: italic;
  font-size: 0.8rem !important;
  color: #3A3628 !important;
  text-align: center;
  margin-top: 28px;
}

/* text input in sidebar */
[data-testid="stSidebar"] [data-testid="stTextInput"] input {
  background: #242018 !important;
  border: 1px solid #3A3628 !important;
  border-radius: 0 !important;
  color: #E8E0D0 !important;
}

/* ═══════════════════════════════════════
   HERO — origami / folded geometry
═══════════════════════════════════════ */
.hero-outer {
  position: relative;
  background: #F0EBE1;
  padding: 0;
  overflow: hidden;
}

/* large folded triangle top-right */
.hero-outer::before {
  content: '';
  position: absolute;
  top: 0; right: 0;
  width: 0; height: 0;
  border-style: solid;
  border-width: 0 280px 280px 0;
  border-color: transparent #C4A96A transparent transparent;
  z-index: 0;
}
/* inner fold shadow */
.hero-outer::after {
  content: '';
  position: absolute;
  top: 0; right: 0;
  width: 0; height: 0;
  border-style: solid;
  border-width: 0 240px 240px 0;
  border-color: transparent #D4B97A transparent transparent;
  z-index: 1;
}

.hero-content {
  position: relative;
  z-index: 2;
  padding: 64px 60px 48px;
}
.hero-eyebrow {
  font-size: 0.62rem;
  font-weight: 500;
  letter-spacing: 0.28em;
  text-transform: uppercase;
  color: #9A8A6A;
  margin-bottom: 16px;
}
.hero-heading {
  font-family: 'Cormorant Garamond', serif;
  font-size: 4rem;
  font-weight: 300;
  line-height: 1.0;
  color: #1C1A16;
  letter-spacing: -0.01em;
}
.hero-heading em {
  font-style: italic;
  color: #7A6040;
}
.hero-rule {
  border: none;
  border-top: 1px solid #1C1A16;
  margin: 28px 0 0;
  width: 100%;
}

/* ═══════════════════════════════════════
   CONTENT GRID — origami fold cards
═══════════════════════════════════════ */
.content-area { padding: 0 60px 40px; background: #F0EBE1; }

.section-tag {
  font-size: 0.6rem;
  font-weight: 500;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  color: #9A8A6A;
  margin: 36px 0 16px;
  display: flex;
  align-items: center;
  gap: 12px;
}
.section-tag::after {
  content: '';
  flex: 1;
  border-top: 1px solid #D8D0C0;
}

/* suggestion grid — origami card style */
.sug-outer {
  display: grid;
  grid-template-columns: repeat(3,1fr);
  gap: 2px;
  background: #C4A96A;
  margin-bottom: 8px;
}
.sug-card {
  background: #F0EBE1;
  padding: 20px 22px;
  position: relative;
  overflow: hidden;
  cursor: pointer;
  transition: background 0.18s;
}
.sug-card::after {
  content: '';
  position: absolute;
  bottom: 0; right: 0;
  width: 0; height: 0;
  border-style: solid;
  border-width: 0 0 20px 20px;
  border-color: transparent transparent #C4A96A transparent;
  transition: all 0.18s;
}
.sug-card:hover { background: #EAE4D8; }
.sug-card:hover::after { border-width: 0 0 28px 28px; }
.sug-num {
  font-family: 'Cormorant Garamond', serif;
  font-size: 0.75rem;
  color: #B8A880;
  margin-bottom: 6px;
  display: block;
}
.sug-txt {
  font-family: 'Cormorant Garamond', serif;
  font-size: 1.05rem;
  font-weight: 400;
  color: #2C2618;
  line-height: 1.3;
}

/* ── MAIN BUTTONS (suggestion) ── */
.stButton > button {
  background: transparent !important;
  border: none !important;
  padding: 0 !important;
  width: 100%;
  height: 100%;
}

/* ═══════════════════════════════════════
   CHAT
═══════════════════════════════════════ */
.chat-area { padding: 0 60px; background: #F0EBE1; }

[data-testid="stChatMessage"] {
  background: transparent !important;
  border: none !important;
  border-bottom: 1px solid #D8D0C0 !important;
  padding: 28px 0 !important;
  border-radius: 0 !important;
}
[data-testid="stChatMessage"] p {
  font-family: 'Jost', sans-serif !important;
  font-weight: 300 !important;
  font-size: 0.95rem !important;
  line-height: 1.75 !important;
  color: #2C2618 !important;
}
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] { display: none !important; }

/* ── CHAT INPUT ── */
[data-testid="stChatInput"] {
  background: #E8E2D6;
  border-top: 1px solid #C4A96A;
  padding: 14px 60px !important;
}
[data-testid="stChatInput"] textarea {
  background: #F0EBE1 !important;
  border: 1px solid #C8C0B0 !important;
  border-radius: 0 !important;
  color: #1C1A16 !important;
  font-family: 'Cormorant Garamond', serif !important;
  font-size: 1.05rem !important;
  font-style: italic;
}
[data-testid="stChatInput"] textarea::placeholder {
  color: #A8A090 !important;
}

/* source pills */
.pill {
  display: inline-block;
  border-left: 2px solid #C4A96A;
  padding: 1px 8px;
  font-family: 'Jost', sans-serif;
  font-size: 0.68rem;
  font-weight: 400;
  letter-spacing: 0.06em;
  color: #7A6A50;
  margin: 4px 6px 0 0;
  background: #EAE4D8;
}
</style>
""", unsafe_allow_html=True)

# ── constants ────────────────────────────────────────────────
EMBED_MODEL  = "all-MiniLM-L6-v2"
CHUNK_SIZE   = 400; OVERLAP = 80; TOP_K = 8
GEMINI_MODEL = "gemini-2.0-flash"
CHART_KW     = ["chart","plot","graph","visuali","bar","pie","line",
                 "scatter","histogram","show","compare","distribution","trend"]

for k,v in {"chunks":[],"index":None,"embeddings":None,
             "history":[],"dataframes":{},"ready":False,"embed_model":None}.items():
    if k not in st.session_state: st.session_state[k] = v

@st.cache_resource(show_spinner="Loading intelligence layer…")
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

def ask(question,ctx,history):
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
    return genai.GenerativeModel(GEMINI_MODEL).generate_content(prompt).text

def wants_chart(q): return any(k in q.lower() for k in CHART_KW)

def get_spec(answer):
    m=re.search(r"```chart\s*(\{.*?\})\s*```",answer,re.DOTALL)
    if m:
        try: return json.loads(m.group(1))
        except: pass
    return None

CLAYOUT=dict(paper_bgcolor="#F0EBE1",plot_bgcolor="#EAE4D8",
    font=dict(family="Jost,sans-serif",color="#2C2618",size=12),
    title_font=dict(family="Cormorant Garamond,serif",size=18,color="#1C1A16"),
    colorway=["#C4A96A","#1C1A16","#7A6040","#D4B97A","#4A4030","#E8D4A0"],
    margin=dict(t=48,b=32,l=32,r=32))

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
    <div class='sb-brand'>AllDoors</div>
    <span class='sb-sub'>Intelligence</span>
    <hr class='sb-rule'>
    """, unsafe_allow_html=True)

    # API key — silent, never displayed
    _key=""
    try: _key=st.secrets.get("GEMINI_API_KEY","")
    except: pass
    if not _key:
        st.markdown("<span class='sb-lbl'>API Key</span>", unsafe_allow_html=True)
        _key=st.text_input("k",type="password",placeholder="Gemini key…",label_visibility="collapsed")
        st.caption("Free at [aistudio.google.com](https://aistudio.google.com)")
    if _key:
        genai.configure(api_key=_key)
        st.session_state["ready"]=True

    st.markdown("<hr class='sb-rule'><span class='sb-lbl'>Documents</span>", unsafe_allow_html=True)

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

    st.markdown("<div class='sb-foot'>alldoors.in</div>",unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════

# Hero with origami fold
st.markdown("""
<div class='hero-outer'>
  <div class='hero-content'>
    <div class='hero-eyebrow'>AllDoors &nbsp;·&nbsp; Real Estate Intelligence &nbsp;·&nbsp; Team C</div>
    <div class='hero-heading'>We are<br><em>Team C.</em></div>
    <hr class='hero-rule'>
  </div>
</div>
""", unsafe_allow_html=True)

# Suggestions
if not st.session_state["history"]:
    st.markdown("""
    <div class='content-area'>
      <div class='section-tag'>Suggested Queries</div>
    </div>""", unsafe_allow_html=True)

    SUGGESTIONS=[
        "What was the total closed-won revenue?",
        "Who is the top rep by pipeline value?",
        "Show a bar chart of deals by stage",
        "List the at-risk accounts",
        "Does the PDF contract total match the deals sheet?",
        "Which lead sources convert best?",
    ]

    # Render as native buttons styled via CSS
    cols=st.columns(3,gap="small")
    for i,s in enumerate(SUGGESTIONS):
        with cols[i%3]:
            # wrap each button in a styled card div
            st.markdown(f"""
            <div class='sug-card'>
              <span class='sug-num'>0{i+1}</span>
              <span class='sug-txt'>{s}</span>
            </div>""", unsafe_allow_html=True)
            if st.button(s, key=f"s{i}", use_container_width=True):
                st.session_state["_q"]=s; st.rerun()

    st.markdown("""
    <div class='content-area'>
      <p style='font-family:Cormorant Garamond,serif;font-style:italic;
      color:#9A8A6A;font-size:0.95rem;text-align:center;margin-top:32px;margin-bottom:24px'>
      Upload your documents in the sidebar, index them, and begin your enquiry.
      </p>
    </div>""", unsafe_allow_html=True)

# Chat history
for turn in st.session_state["history"]:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("sources"):
            st.markdown(" ".join(f"<span class='pill'>{s}</span>" for s in turn["sources"]),unsafe_allow_html=True)
        if turn.get("fig"):
            st.plotly_chart(turn["fig"],use_container_width=True)

# Input
question=st.chat_input("Enter your enquiry…") or st.session_state.pop("_q",None)

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
        with st.spinner("Searching documents…"):
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