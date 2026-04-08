# B-RAG AI
**Chat with your PDF documents — built by Bibek Subedi**

Powered by OpenAI `gpt-4o-mini` + `text-embedding-3-small` + FAISS.

---

## Project Structure

```
b_rag_ai/
├── app.py
├── requirements.txt
├── .gitignore
├── .streamlit/
│   └── secrets.toml      ← your OpenAI key goes here (never commit)
└── assets/
    ├── favicon.png        ← 32×32 icon (optional)
    └── photo.jpg          ← your profile photo (optional)
```

## Local Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push repo to GitHub (secrets.toml is git-ignored).
2. Go to share.streamlit.io → New app → select repo.
3. Add `OPENAI_API_KEY` in the Secrets panel (Settings → Secrets).
4. Deploy.
