# DataIntern — RAG Chatbot for CRM & Business Data

A Retrieval-Augmented Generation (RAG) chatbot that ingests mixed business files and answers natural-language questions with **cited sources** and **live charts**.

**Live demo:** *(paste your Streamlit Cloud URL here after deployment)*

---

## What it does

| Capability | Detail |
|---|---|
| Multi-format ingestion | CSV, Excel (all sheets), PDF, Word, JSON, TSV, Markdown |
| RAG pipeline | sentence-transformers embeddings → FAISS vector index → top-k retrieval |
| Cited answers | Every fact tagged with `[SOURCE N: filename \| page/row]` |
| Live charts | Bar, line, pie, scatter — rendered inline with Plotly |
| Multi-turn memory | Last 6 conversation turns kept in context |
| Refuses hallucination | "I don't see that in your files" when context is missing |

## Stack

- **LLM:** Google Gemini 1.5 Flash (free tier — 15 req/min)
- **Embeddings:** `all-MiniLM-L6-v2` via sentence-transformers (runs locally, free)
- **Vector search:** FAISS (in-memory, no server needed)
- **UI:** Streamlit
- **Charts:** Plotly Express
- **Parsing:** pdfplumber, python-docx, pandas, openpyxl

## Run locally

```bash
git clone https://github.com/YOUR_USERNAME/dataintern.git
cd dataintern
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501, paste your Gemini API key in the sidebar, upload files, and start asking.

## Deploy on Streamlit Cloud (free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Select your repo, branch `main`, file `app.py`
4. Under **Advanced settings → Secrets**, add:
   ```toml
   GEMINI_API_KEY = "your-key-here"
   ```
5. Click **Deploy** — live in ~3 minutes

> The app reads `GEMINI_API_KEY` from Streamlit secrets automatically if present, so users don't need to paste it each time. The sidebar input is a fallback for local runs.

## Get a free Gemini API key

1. Go to [aistudio.google.com](https://aistudio.google.com)
2. Click **Get API Key → Create API key**
3. Copy and paste into the sidebar

## Example questions (use the sample CRM files)

- "What was total closed-won revenue?"
- "Who is the top rep by pipeline value?"
- "Show me a bar chart of deals by stage"
- "Does the PDF contract total match the deals sheet?"
- "List the at-risk accounts"
- "Which lead sources convert best?"

## File structure

```
dataintern/
├── app.py            # Full Streamlit RAG app (single file)
├── requirements.txt  # Python dependencies
└── README.md
```
