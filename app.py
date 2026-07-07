"""
AllDoors Intelligence — RAG Chatbot for CRM & Business Data
API key: Streamlit Cloud Secrets → GEMINI_API_KEY (never shown in UI)
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

# System fonts only — no Google Fonts CDN needed
st.markdown("""
<style>
/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp {
  background: #141414 !important;
  color: #E5E5E5 !important;
  font-family: -apple-system, 'Helvetica Neue', Arial, sans-serif;
}

/* hide streamlit chrome */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="collapsedControl"] { display: none !important; visibility: hidden !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: #000000 !important;
  border-right: 1px solid #1F1F1F !important;
  min-width: 240px !important;
}
[data-testid="stSidebar"] > div:first-child {
  padding: 28px 20px !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div {
  color: #B3B3B3 !important;
  font-family: -apple-system, 'Helvetica Neue', Arial, sans-serif !important;
}

/* sidebar buttons */
[data-testid="stSidebar"] .stButton > button {
  font-family: -apple-system, 'Helvetica Neue', Arial, sans-serif !important;
  font-size: 0.8rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.1em !important;
  text-transform: uppercase !important;
  border-radius: 3px !important;
  padding: 10px 16px !important;
  width: 100% !important;
  transition: all 0.15s !important;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
  background: #E50914 !important;
  color: #FFFFFF !important;
  border: none !important;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
  background: #C2070F !important;
}
[data-testid="stSidebar"] .stButton > button:not([kind="primary"]) {
  background: transparent !important;
  color: #808080 !important;
  border: 1px solid #333 !important;
}
[data-testid="stSidebar"] .stButton > button:not([kind="primary"]):hover {
  color: #E5E5E5 !important;
  border-color: #666 !important;
}

/* sidebar file uploader */
[data-testid="stSidebar"] [data-testid="stFileUploader"] {
  background: #1A1A1A !important;
  border: 1px dashed #333 !important;
  border-radius: 3px !important;
}
[data-testid="stSidebar"] [data-testid="stTextInput"] input {
  background: #1A1A1A !important;
  border: 1px solid #333 !important;
  border-radius: 3px !important;
  color: #E5E5E5 !important;
}

/* ── Main content area ── */
.main .block-container {
  padding: 0 !important;
  max-width: 100% !important;
}

/* ── Suggestion buttons — styled as tiles ── */
.stButton > button {
  background: #1F1F1F !important;
  color: #B3B3B3 !important;
  border: none !important;
  border-left: 3px solid #333 !important;
  border-radius: 0 !important;
  font-family: -apple-system, 'Helvetica Neue', Arial, sans-serif !important;
  font-size: 0.88rem !important;
  font-weight: 400 !important;
  text-align: left !important;
  padding: 20px 18px !important;
  width: 100% !important;
  height: 80px !important;
  line-height: 1.4 !important;
  transition: all 0.15s !important;
}
.stButton > button:hover {
  background: #2A2A2A !important;
  border-left-color: #E50914 !important;
  color: #FFFFFF !important;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
  background: transparent !important;
  border: none !important;
  border-bottom: 1px solid #222 !important;
  border-radius: 0 !important;
  padding: 24px 48px !important;
}
[data-testid="stChatMessage"] p {
  font-size: 0.95rem !important;
  font-weight: 300 !important;
  line-height: 1.75 !important;
  color: #E5E5E5 !important;
}
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] { display: none !important; }

/* ── Chat input ── */
[data-testid="stChatInput"] {
  background: #0A0A0A !important;
  border-top: 1px solid #222 !important;
  padding: 16px 48px !important;
}
[data-testid="stChatInput"] textarea {
  background: #1A1A1A !important;
  border: 1px solid #333 !important;
  border-radius: 3px !important;
  color: #E5E5E5 !important;
  font-size: 0.95rem !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #555 !important; }

/* ── Source pills ── */
.pill {
  display: inline-block;
  background: #1F1F1F;
  border-left: 2px solid #E50914;
  padding: 2px 10px;
  font-size: 0.68rem;
  color: #808080;
  margin: 4px 4px 0 0;
  letter-spacing: 0.04em;
}

/* ── Alerts ── */
.stSuccess { background: #0D1F0D !important; border-color: #1A5C1A !important; }
.stWarning { background: #1F150D !important; }
.stError   { background: #1F0D0D !important; }
</style>
""", unsafe_allow_html=True)

# ── constants ────────────────────────────────────────────────
EMBED_MODEL  = "all-MiniLM-L6-v2"
CHUNK_SIZE   = 400; OVERLAP = 80; TOP_K = 8
GEMINI_MODEL = "gemini-2.0-flash"
CHART_KW     = ["chart","plot","graph","visuali","bar","pie","line",
                 "scatter","histogram","show","compare","distribution","trend"]
MAX_RETRIES  = 3

for k,v in {"chunks":[],"index":None,"embeddings":None,
             "history":[],"dataframes":{},"ready":False,"embed_model":None}.items():
    if k not in st.session_state: st.session_state[k] = v

@st.cache_resource(show_spinner="Loading model…")
def load_embed(): return SentenceTransformer(EMBED_MODEL)

# ── parsers ──────────────────────────────────────────────────
def chunk(text, src, meta):
    out,s=[],0
    while s<len(text):
        c=text[s:s+CHUNK_SIZE].strip()
        if c: out.append({"text":c,"source":src,"meta":meta.copy()})
        s+=CHUNK_SIZE-OVERLAP
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
        f"[SOURCE {i+1}: {c['source']}"+(f" | {','.join(f'{k}:{v}' for k,v in c['meta'].items())}" if c['meta'] else "")+f"]\n{c['text']}"
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
    m=genai.GenerativeModel(GEMINI_MODEL)
    for attempt in range(MAX_RETRIES):
        try: return m.generate_content(prompt).text
        except ResourceExhausted:
            if attempt<MAX_RETRIES-1: time.sleep(15*(attempt+1))
            else: return "**Rate limit reached.** Gemini free tier allows 15 requests/minute. Please wait 30–60 seconds and try again."
        except Exception as e: return f"**Error:** {str(e)[:200]}"

def wants_chart(q): return any(k in q.lower() for k in CHART_KW)

def get_spec(answer):
    m=re.search(r"```chart\s*(\{.*?\})\s*```",answer,re.DOTALL)
    if m:
        try: return json.loads(m.group(1))
        except: pass
    return None

CLAYOUT=dict(paper_bgcolor="#141414",plot_bgcolor="#1F1F1F",
    font=dict(family="-apple-system,Helvetica Neue,Arial,sans-serif",color="#B3B3B3",size=12),
    title_font=dict(size=18,color="#FFFFFF"),
    colorway=["#E50914","#B3B3B3","#E5E5E5","#831010","#666","#FF6B6B"],
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
    st.markdown("## AllDoors")
    st.caption("INTELLIGENCE SUITE")
    st.divider()

    # API key — silent from secrets
    _key=""
    try: _key=st.secrets.get("GEMINI_API_KEY","")
    except: pass
    if not _key:
        st.caption("API KEY")
        _key=st.text_input("k",type="password",
                           placeholder="Paste Gemini key…",
                           label_visibility="collapsed")
        st.caption("Free key → [aistudio.google.com](https://aistudio.google.com)")
    if _key:
        genai.configure(api_key=_key)
        st.session_state["ready"]=True

    st.divider()
    st.caption("DOCUMENTS")
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
                        status.caption(f"✓ {f.name}")
                    except Exception as e: st.warning(f"{f.name}: {e}")
                bar.progress((i+1)/len(files))
            if all_chunks:
                with st.spinner("Building index…"):
                    idx,emb=build_index(all_chunks,model)
                st.session_state.update({"chunks":all_chunks,"index":idx,"embeddings":emb,"embed_model":model})
                bar.empty(); status.empty()
                st.success(f"{len(all_chunks)} passages indexed across {len(files)} file(s)")
            else: st.error("No content extracted.")

    if st.session_state["chunks"]:
        c1,c2=st.columns(2)
        c1.metric("Passages",len(st.session_state["chunks"]))
        c2.metric("Tables",len(st.session_state["dataframes"]))
        st.divider()
        if st.button("Clear session",use_container_width=True):
            for k in ["chunks","index","embeddings","history","dataframes","embed_model"]:
                st.session_state[k]=[] if k in ("chunks","history") else ({} if k=="dataframes" else None)
            st.rerun()

    st.markdown("<br><br>",unsafe_allow_html=True)
    st.caption("alldoors.in · Team C")

# ════════════════════════════════════════════════════════════
# MAIN — HERO
# ════════════════════════════════════════════════════════════
st.markdown("""
<div style="
  background: linear-gradient(to bottom, rgba(0,0,0,0.75) 0%, rgba(20,20,20,0.5) 60%, #141414 100%),
  url('https://images.unsplash.com/photo-1486325212027-8081e485255e?w=1600&q=80&fit=crop') center/cover no-repeat;
  padding: 90px 48px 72px;
">
  <div style="font-size:0.6rem;font-weight:600;letter-spacing:0.28em;text-transform:uppercase;
    color:#E50914;margin-bottom:14px">
    AllDoors &nbsp;·&nbsp; Real Estate Intelligence
  </div>
  <div style="font-size:4.5rem;font-weight:800;line-height:0.95;letter-spacing:-0.02em;
    color:#FFFFFF;text-shadow:0 2px 24px rgba(0,0,0,0.9)">
    WE ARE<br><span style="color:#E50914">TEAM C.</span>
  </div>
  <div style="font-size:0.95rem;font-weight:300;color:#B3B3B3;margin-top:16px;
    max-width:440px;line-height:1.6">
    Ask questions about your CRM data.<br>Get cited answers and live charts — instantly.
  </div>
  <div style="width:40px;border-top:3px solid #E50914;margin-top:24px"></div>
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
    st.markdown("""
    <div style="padding:32px 48px 16px;font-size:0.6rem;font-weight:600;
      letter-spacing:0.2em;text-transform:uppercase;color:#808080">
      Suggested Queries
    </div>""",unsafe_allow_html=True)

    col1,col2,col3=st.columns(3,gap="small")
    cols=[col1,col2,col3]
    for i,s in enumerate(SUGGESTIONS):
        with cols[i%3]:
            if st.button(s,key=f"s{i}",use_container_width=True):
                st.session_state["_q"]=s; st.rerun()

    st.markdown("""
    <div style="padding:32px 48px 0;font-size:0.8rem;font-weight:300;
      color:#555;text-align:center">
      Upload documents in the sidebar → Index → ask anything below
    </div>""",unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# CHAT
# ════════════════════════════════════════════════════════════
for turn in st.session_state["history"]:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("sources"):
            st.markdown(" ".join(f"<span class='pill'>{s}</span>" for s in turn["sources"]),unsafe_allow_html=True)
        if turn.get("fig"):
            st.plotly_chart(turn["fig"],use_container_width=True)

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