# B-RAG AI

**Retrieval-Augmented Generation · Document Intelligence · Built by Bibek Subedi**

---

## Overview

B-RAG AI is a production-grade document intelligence application that allows users to upload a PDF and conduct a natural language conversation with its contents. The system combines OpenAI's embedding and language models with a FAISS vector store to deliver fast, accurate, and cost-controlled retrieval-augmented generation — all within a single-page Streamlit interface.

The application is designed to look and behave like a professional product, not a prototype.

---

## Live Features

- PDF upload with a three-rule validation gate (size, page count, text density)
- Automatic text chunking and FAISS vector indexing on first upload
- Semantic retrieval of the most relevant document chunks per query
- Token-aware context assembly capped at 4,000 tokens per request
- Streaming chat responses via `gpt-4o-mini`
- Persistent visit counter backed by Supabase
- Session-cached vector store to eliminate redundant embedding calls
- Fully branded sidebar with developer profile and social links
- Custom dark-theme UI with amber accent palette

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Language Model | OpenAI `gpt-4o-mini` |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector Store | FAISS (CPU) |
| PDF Parsing | pypdf |
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

**Step 4 — Configure secrets**

Create `.streamlit/secrets.toml` and add the following:

```toml
OPENAI_API_KEY = "sk-..."
SUPABASE_URL   = "https://your-project-id.supabase.co"
SUPABASE_KEY   = "your-anon-public-key"
```

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
4. Open **Settings → Secrets** and paste all three key-value pairs from your local `secrets.toml`.
5. Click **Deploy**.

The app will be live within 60 seconds at a `*.streamlit.app` URL.

---

## Cost Controls

Every request is subject to three cost guardrails.

**Model selection** — `gpt-4o-mini` costs approximately 40x less than `gpt-4o` with comparable performance on document Q&A tasks. `text-embedding-3-small` is the lowest-cost OpenAI embedding model.

**Token budget** — Retrieved chunks are assembled and trimmed by `tiktoken` before being sent to the model. The total context per request never exceeds 4,000 tokens.

**Response cap** — `max_tokens` is hard-set to 800, preventing runaway generation from draining API credits.

**Session caching** — The FAISS index is stored in `st.session_state` after the first embedding run. Subsequent questions against the same document make zero embedding API calls.

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
