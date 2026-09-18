# B-RAG AI

**Retrieval-Augmented Generation · Document Intelligence · Built by Bibek Subedi**

---

## Overview

B-RAG AI is a production-grade document intelligence application that allows users to upload a PDF or DOCX and conduct a natural language conversation with its contents. The system combines Hugging Face embedding models, Google Gemini's language model, and a lightweight NumPy vector store to deliver fast, accurate, and cost-controlled retrieval-augmented generation — all within a single-page Streamlit interface.

The application is designed to look and behave like a professional product, not a prototype.

> **v2.0 migration note:** The original build ran on OpenAI's embedding and chat APIs with a FAISS vector store. Following the OpenAI API credit expiring, the stack was migrated to Hugging Face (embeddings), Google Gemini (chat), and a pure NumPy cosine-similarity search in place of FAISS — removing a native-binary dependency entirely while keeping the same retrieval behavior.

---

## Live Features

- PDF and DOCX upload with a three-rule validation gate (size, page count, text density)
- Automatic text chunking and vector indexing on first upload
- Semantic retrieval of the most relevant document chunks per query
- Token-aware context assembly capped at 4,000 tokens per request
- Streaming chat responses via Gemini's OpenAI-compatible endpoint
- Persistent visit counter backed by Supabase
- Session-cached embeddings to eliminate redundant API calls
- Fully branded sidebar with developer profile and social links
- Custom dark-theme UI with amber accent palette

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Language Model | Google `gemini-3.5-flash-lite` (via OpenAI-compatible endpoint) |
| Embeddings | Hugging Face `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store | NumPy (cosine similarity) |
| PDF Parsing | pypdf |
| DOCX Parsing | python-docx |
| Token Counting | tiktoken |
| Visit Counter | Supabase (PostgreSQL) |
| Hosting | Streamlit Community Cloud |



---

## Project Structure

```
b_rag_ai/
├── app.py
├── requirements.txt
├── .gitignore
├── .streamlit/
│   ├── secrets.toml
│   └── config.toml
└── assets/
    ├── cover.jpg
    ├── favicon.png
    └── photo.jpg
```

---

## Local Setup

**Step 1 — Clone the repository**

```bash
git clone https://github.com/BibekSubediCR7/b-rag-ai.git
cd b-rag-ai
```

**Step 2 — Create a virtual environment**

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux
```

**Step 3 — Install dependencies**

```bash
pip install -r requirements.txt
```

If `pip` or `streamlit` is blocked by an Application Control policy (common on managed Windows machines), run them as Python modules instead:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

**Step 4 — Configure secrets**

Create `.streamlit/secrets.toml` and add the following:

```toml
GEMINI_API_KEY = "..."
HF_API_KEY     = "hf_..."
SUPABASE_URL   = "https://your-project-id.supabase.co"
SUPABASE_KEY   = "your-anon-public-key"
```

- Get a free Gemini key at [aistudio.google.com](https://aistudio.google.com) — no credit card required for the free tier.
- Get a free Hugging Face token (Read scope is enough) at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

**Step 5 — Run the application**

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Supabase Setup

The visit counter requires two objects in your Supabase project. Run the following in the SQL Editor.

**Create the visits table:**

```sql
CREATE TABLE visits (
  id    INT PRIMARY KEY DEFAULT 1,
  count BIGINT DEFAULT 0
);

INSERT INTO visits (id, count) VALUES (1, 0);
```

**Create the atomic increment function:**

```sql
CREATE OR REPLACE FUNCTION increment_visits()
RETURNS BIGINT AS $$
  UPDATE visits SET count = count + 1 WHERE id = 1 RETURNING count;
$$ LANGUAGE SQL;
```

---

## Deployment — Streamlit Community Cloud

1. Push the repository to GitHub. The `.gitignore` already excludes `secrets.toml` and `visit_counter.json`.
2. Go to [share.streamlit.io](https://share.streamlit.io) and select **New app**.
3. Choose your repository and set the main file path to `app.py`.
4. Open **Settings → Secrets** and paste all four key-value pairs from your local `secrets.toml`.
5. Click **Deploy**.

The app will be live within 60 seconds at a `*.streamlit.app` URL.

---

## Cost Controls

Every request is subject to three cost guardrails.

**Model selection** — Gemini's free tier and Hugging Face's free inference tier cover typical document-chat traffic at no cost. `sentence-transformers/all-MiniLM-L6-v2` is a lightweight, low-latency embedding model well suited to this workload.

**Token budget** — Retrieved chunks are assembled and trimmed by `tiktoken` before being sent to the model. The total context per request never exceeds 4,000 tokens.

**Response cap** — `max_tokens` is hard-set to 800, preventing runaway generation from draining API quota.

**Session caching** — Embeddings are stored in `st.session_state` after the first run, keyed by document hash. Subsequent questions against the same document make zero embedding API calls, and re-uploading an already-seen file makes zero new calls at all.

---

## Document Validation Rules

The uploader enforces three rules before any processing begins.

| Rule | Limit | Enforcement |
|---|---|---|
| File size | 10 MB | `file.size` checked in bytes |
| Page count | 300 pages | `pypdf.PdfReader` page length |
| Text density | 50 characters per page minimum | Heuristic for scanned image detection |

If any rule is violated, execution stops immediately and a styled security alert is displayed. No API calls are made.

---

## Developer

**Bibek Subedi**
Aspiring AI Researcher and Data Scientist

- GitHub — [github.com/BibekSubediCR7](https://github.com/BibekSubediCR7)
- LinkedIn — [linkedin.com/in/bibeksubedicr7](https://www.linkedin.com/in/bibeksubedicr7/)
- Facebook — [facebook.com/profile.php?id=100015784387352](https://www.facebook.com/profile.php?id=100015784387352)

---

## License

This project is personal and open for reference. If you use it as a base for your own work, attribution is appreciated.

---

*Developed by Bibek Subedi · 2026 · All Rights Reserved*