"""
B-RAG AI — Chat with your Documents
Developed by Bibek Subedi
© 2026 All Rights Reserved
"""
import requests 
import streamlit as st
import time
import json
import os
import base64
from pathlib import Path
from datetime import datetime

# ── Third-party ─────────────────────────────────────────────────────────────
import tiktoken
from pypdf import PdfReader
from openai import OpenAI
import faiss
import numpy as np

# ════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG  (must be the very first Streamlit call)
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="B-RAG AI",
    page_icon="assets/favicon.png",  
    layout="wide",
    initial_sidebar_state="expanded",
)

#══════════════════════════════════

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=representation",
}

def load_visits() -> int:
    """Read current visit count from Supabase."""
    r = requests.get(
        f"{SUPABASE_URL}/rest/v1/visits?id=eq.1&select=count",
        headers=HEADERS,
    )
    return r.json()[0]["count"] if r.ok and r.json() else 0

def increment_visits() -> int:
    """Atomically increment and return the new count using Supabase RPC."""
    # Atomic increment — no race condition
    r = requests.post(
        f"{SUPABASE_URL}/rest/v1/rpc/increment_visits",
        headers=HEADERS,
        json={},
    )
    if r.ok:
        return r.json()          # returns new count
    return load_visits()         # fallback: just read current

if "visit_counted" not in st.session_state:
    st.session_state["visit_counted"] = True
    st.session_state["total_visits"] = increment_visits()

VISITS = st.session_state.get("total_visits", load_visits())
# ════════════════════════════════════════════════════════════════════════════


st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── CSS Variables ── */
:root {
    --navy:      #0B1120;
    --navy-mid:  #121C30;
    --navy-card: #1A2540;
    --amber:     #F5A623;
    --amber-dim: #C4841C;
    --ivory:     #F5F0E8;
    --muted:     #8A95A8;
    --rule-line: #2A3550;
    --success:   #2ECC71;
    --danger:    #E74C3C;
    --font-head: 'Syne', sans-serif;
    --font-body: 'DM Sans', sans-serif;
    --font-mono: 'JetBrains Mono', monospace;
}

/* ── Base reset ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--navy) !important;
    color: var(--ivory) !important;
    font-family: var(--font-body);
}

[data-testid="stSidebar"] {
    background-color: var(--navy-mid) !important;
    border-right: 1px solid var(--rule-line);
}

/* ── Visit counter badge ── */
.visit-badge {
    position: fixed;
    top: 14px;
    right: 18px;
    background: var(--navy-card);
    border: 1px solid var(--amber);
    color: var(--amber);
    font-family: var(--font-mono);
    font-size: 11px;
    font-weight: 500;
    padding: 4px 12px;
    border-radius: 20px;
    z-index: 9999;
    letter-spacing: 0.05em;
}

/* ── Main title block ── */
.hero-block {
    padding: 40px 0 10px 0;
    border-bottom: 2px solid var(--rule-line);
    margin-bottom: 30px;
}
.hero-title {
    font-family: var(--font-head);
    font-size: 52px;
    font-weight: 800;
    color: var(--ivory);
    letter-spacing: -1.5px;
    line-height: 1.1;
    margin: 0;
}
.hero-title span {
    color: var(--amber);
}
.hero-sub {
    font-family: var(--font-body);
    font-size: 16px;
    font-weight: 300;
    color: var(--muted);
    margin-top: 10px;
    letter-spacing: 0.01em;
}

/* ── Rules box — custom card ── */
.rules-card {
    background: var(--navy-card);
    border-left: 4px solid var(--amber);
    border-radius: 0 8px 8px 0;
    padding: 20px 24px;
    margin: 24px 0;
}
.rules-card h4 {
    font-family: var(--font-head);
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--amber);
    margin: 0 0 14px 0;
}
.rules-card ul {
    margin: 0;
    padding-left: 16px;
}
.rules-card ul li {
    font-family: var(--font-body);
    font-size: 14px;
    color: #B0BCCC;
    line-height: 2;
}
.rules-card ul li strong {
    color: var(--ivory);
    font-weight: 500;
}

/* ── Security alert box ── */
.alert-card {
    background: #1E0A0A;
    border: 1px solid var(--danger);
    border-radius: 8px;
    padding: 18px 22px;
    margin: 16px 0;
}
.alert-card h4 {
    font-family: var(--font-head);
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--danger);
    margin: 0 0 6px 0;
}
.alert-card p {
    font-size: 14px;
    color: #E8A0A0;
    margin: 0;
}

/* ── Success badge ── */
.success-card {
    background: #061A10;
    border: 1px solid var(--success);
    border-radius: 8px;
    padding: 14px 20px;
    font-size: 14px;
    color: #8AEAB8;
    margin: 12px 0;
    font-family: var(--font-mono);
    letter-spacing: 0.03em;
}

/* ── Stat chips ── */
.stat-row {
    display: flex;
    gap: 12px;
    margin: 16px 0;
    flex-wrap: wrap;
}
.stat-chip {
    background: var(--navy-card);
    border: 1px solid var(--rule-line);
    border-radius: 6px;
    padding: 10px 16px;
    font-family: var(--font-mono);
    font-size: 12px;
    color: var(--muted);
}
.stat-chip strong {
    display: block;
    font-size: 18px;
    color: var(--amber);
    font-weight: 600;
    margin-bottom: 2px;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: var(--navy-card) !important;
    border: 1px solid var(--rule-line) !important;
    border-radius: 10px !important;
    margin-bottom: 12px !important;
    padding: 14px 18px !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] textarea {
    background: var(--navy-card) !important;
    color: var(--ivory) !important;
    border: 1px solid var(--rule-line) !important;
    border-radius: 8px !important;
    font-family: var(--font-body) !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: var(--amber) !important;
    box-shadow: 0 0 0 2px rgba(245,166,35,0.15) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploadDropzone"] {
    background: var(--navy-card) !important;
    border: 2px dashed var(--rule-line) !important;
    border-radius: 12px !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploadDropzone"]:hover {
    border-color: var(--amber) !important;
}

#/* ── Sidebar profile card ── */
#.profile-photo-placeholder {
 #  height: 96px;
  #  border-radius: 50%;
   # background: var(--navy-card);
    #border: 2px solid var(--amber);
#    display: flex;
 #   align-items: center;
  #  justify-content: center;
   # margin: 0 auto 16px auto;
##   font-size: 32px;
#    font-weight: 800;
 #   color: var(--amber);
#  letter-spacing: -1px;
#} 
.profile-name {
    font-family: var(--font-head);
    font-size: 17px;
    font-weight: 700;
    color: var(--ivory);
    text-align: center;
    margin-bottom: 4px;
}
.profile-tagline {
    font-size: 12px;
    color: var(--muted);
    text-align: center;
    line-height: 1.5;
    margin-bottom: 16px;
}
.social-grid {
    display: flex;
    flex-direction: column;
    gap: 8px;
    margin-top: 12px;
}
.social-btn {
    display: block;
    background: var(--navy);
    border: 1px solid var(--rule-line);
    border-radius: 6px;
    padding: 8px 14px;
    font-family: var(--font-body);
    font-size: 12px;
    font-weight: 500;
    color: var(--muted) !important;
    text-decoration: none !important;
    text-align: center;
    transition: all 0.2s;
    letter-spacing: 0.04em;
    cursor: pointer;
}
.social-btn:hover {
    border-color: var(--amber);
    color: var(--amber) !important;
    background: rgba(245,166,35,0.06);
}
.sidebar-divider {
    border: none;
    border-top: 1px solid var(--rule-line);
    margin: 20px 0;
}
.sidebar-section-label {
    font-family: var(--font-head);
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--amber);
    margin-bottom: 12px;
    display: block;
}

/* ── Footer ── */
.footer-bar {
    text-align: center;
    padding: 40px 0 20px 0;
    border-top: 1px solid var(--rule-line);
    margin-top: 60px;
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--muted);
    letter-spacing: 0.06em;
}

/* ── Streamlit overrides ── */
.stSpinner > div { color: var(--amber) !important; }
div[data-testid="stMarkdownContainer"] p { color: var(--ivory); }
button[kind="primary"] {
    background: var(--amber) !important;
    color: var(--navy) !important;
    font-family: var(--font-head) !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 6px !important;
}
button[kind="secondary"] {
    background: transparent !important;
    color: var(--amber) !important;
    border: 1px solid var(--amber) !important;
    font-family: var(--font-head) !important;
    font-weight: 600 !important;
    border-radius: 6px !important;
}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
#  VISIT COUNTER BADGE  (top-right corner)
# ════════════════════════════════════════════════════════════════════════════
st.markdown(
    f"""
    <style>
        /* Define the variables in case they aren't set in your theme */
        :root {{
            --navy-card: #0e1117; 
            --amber: #ffbf00;
            --font-mono: 'Source Code Pro', monospace;
        }}

        .visit-badge {{
            position: fixed;
            top: 20px; /* Lowered to sit just below the Streamlit top bar */
            right: 18px;
            background: var(--navy-card);
            border: 1px solid var(--amber);
            color: var(--amber);
            font-family: var(--font-mono);
            font-size: 11px;
            font-weight: 500;
            padding: 4px 12px;
            border-radius: 20px;
            z-index: 999999; /* Increased to ensure it stays on top */
            letter-spacing: 0.05em;
            box-shadow: 0px 2px 10px rgba(0,0,0,0.3);
        }}
    </style>
    
    <div class="visit-badge">
        VISITS &nbsp;{VISITS:,}
    </div>
    """,
    unsafe_allow_html=True,
)


# ════════════════════════════════════════════════════════════════════════════
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ════════════════════════════════════════════════════════════════════════════
EMBED_MODEL   = "text-embedding-3-small"   # cheapest OpenAI embeddings
CHAT_MODEL    = "gpt-4o-mini"              # most cost-efficient GPT-4 class
MAX_TOKENS    = 800                        # hard cap on response length
MAX_CTX_TOKS  = 4000                       # max context tokens sent to LLM
MAX_FILE_MB   = 10                         # file size gate
MAX_PAGES     = 300                        # page count gate
CHUNK_SIZE    = 500                        # words per text chunk
CHUNK_OVERLAP = 50                         # overlap between chunks
EMBED_DIM     = 1536                       # text-embedding-3-small dimension

SYSTEM_PROMPT = """You are an elite academic assistant with complete mastery of the provided document. Your role is to deliver precise, insightful analysis — not to describe what a document says.

**Formatting Standards:**
- Always structure responses with Markdown.
- Use **bold** for key terms and **## Bold Headers** for major sections.
- Present lists as clean bullet points.
- When data contains numbers or comparisons, format them as Markdown tables.
- Break long explanations into 2–3 focused paragraphs for readability.

**Tone & Voice:**
- Speak as a domain expert, not a document reader.
- Never say "Based on the text," "According to the document," or similar disclaimers.
- Be direct, confident, and professional.
- If information is outside the scope of the document, state that clearly and concisely."""

# ════════════════════════════════════════════════════════════════════════════
#  SESSION STATE INITIALISATION
# ════════════════════════════════════════════════════════════════════════════
for key, default in {
    "vector_store":   None,
    "chunks":         [],
    "chat_history":   [],
    "doc_stats":      {},
    "file_name":      None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:

    # ── Developer Profile ───────────────────────────────────────────────
    st.markdown('<span class="sidebar-section-label">Developer Profile</span>', unsafe_allow_html=True)

    
    st.image("assets/photo.jpg", use_container_width=True)
   
  #  st.markdown(
 #      '<div class="profile-photo-placeholder">BS</div>',
  #      unsafe_allow_html=True,
  #  )

    st.markdown('<div class="profile-name">Bibek Subedi</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="profile-tagline">Aspiring AI Researcher<br>& Data Scientist</div>',
        unsafe_allow_html=True,
    )

    st.markdown("""
    <div class="social-grid">
        <a class="social-btn" href="https://github.com/BibekSubediCR7" target="_blank">
            GitHub &rarr;
        </a>
        <a class="social-btn" href="https://www.linkedin.com/in/bibeksubedicr7/" target="_blank">
            LinkedIn &rarr;
        </a>
        <a class="social-btn" href="https://www.facebook.com/profile.php?id=100015784387352" target="_blank">
            Facebook &rarr;
        </a>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # ── App Controls ────────────────────────────────────────────────────
    st.markdown('<span class="sidebar-section-label">App Controls</span>', unsafe_allow_html=True)

    if st.session_state["doc_stats"]:
        stats = st.session_state["doc_stats"]
        st.markdown(f"""
        <div class="stat-row">
            <div class="stat-chip"><strong>{stats.get('pages', '—')}</strong>Pages</div>
            <div class="stat-chip"><strong>{stats.get('chunks', '—')}</strong>Chunks</div>
            <div class="stat-chip"><strong>{stats.get('tokens', '—')}</strong>Tokens</div>
        </div>
        """, unsafe_allow_html=True)

    if st.session_state["vector_store"] is not None:
        if st.button("Clear Document & Reset", type="secondary", use_container_width=True):
            for k in ["vector_store", "chunks", "chat_history", "doc_stats", "file_name"]:
                st.session_state[k] = [] if k in ["chunks", "chat_history"] else (None if k != "doc_stats" else {})
            st.rerun()

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # ── Model Info ──────────────────────────────────────────────────────
    st.markdown('<span class="sidebar-section-label">Model Configuration</span>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-family: var(--font-mono); font-size: 11px; color: var(--muted); line-height: 2;">
        Chat &nbsp;&nbsp;&nbsp;&nbsp;: {CHAT_MODEL}<br>
        Embed &nbsp;&nbsp;&nbsp;: {EMBED_MODEL}<br>
        Max Tok : {MAX_TOKENS}<br>
        Ctx Tok &nbsp;: {MAX_CTX_TOKS}
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
#  MAIN CONTENT
# ════════════════════════════════════════════════════════════════════════════

# ── Hero ─────────────────────────────────────────────────────────────────


def get_base64_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

try:
    img_b64 = get_base64_image("assets/cover.jpg")
    bg_css = f"url('data:image/jpeg;base64,{img_b64}')"
except FileNotFoundError:
    bg_css = "none"

st.markdown(f"""
<div class="hero-block" style="
    background-image: {bg_css};
    background-size: cover;
    background-position: center;
    border-radius: 12px;
    padding: 60px 40px;
    position: relative;
    overflow: hidden;
    margin-bottom: 30px;
">
    <div style="
        position: absolute;
        inset: 0;
        background: linear-gradient(
            to right,
            rgba(11,17,32,0.92) 40%,
            rgba(11,17,32,0.55) 100%
        );
        border-radius: 12px;
    "></div>
    <div style="position: relative; z-index: 1;">
        <h1 class="hero-title">B-<span>RAG</span> AI</h1>
        <p class="hero-sub">An intelligent RAG system that understands your documents.</p>
    </div>
</div>
""", unsafe_allow_html=True)
# ── Submission Rules ─────────────────────────────────────────────────────
st.markdown("""
<div class="rules-card">
    <h4>Document Submission Policy</h4>
    <ul>
        <li><strong>File size</strong> must not exceed <strong>10 MB</strong>.</li>
        <li><strong>Page count</strong> must not exceed <strong>300 pages</strong>.</li>
        <li>The document must be <strong>text-based</strong>. Scanned image PDFs are not supported.</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════
def get_base64_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()
    
def security_alert(message: str):
    st.markdown(f"""
    <div class="alert-card">
        <h4>Security Alert</h4>
        <p>{message}</p>
    </div>
    """, unsafe_allow_html=True)


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i : i + size])
        chunks.append(chunk)
        i += size - overlap
    return chunks


def get_embeddings(texts: list[str]) -> np.ndarray:
    """Batch-embed texts and return a float32 numpy array."""
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return np.array([item.embedding for item in response.data], dtype=np.float32)


def build_faiss_index(chunks: list[str]) -> faiss.IndexFlatIP:
    """Build a cosine-similarity FAISS index over text chunks."""
    embeddings = get_embeddings(chunks)
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(embeddings)
    return index


def retrieve(query: str, index: faiss.IndexFlatIP, chunks: list[str], k: int = 5) -> list[str]:
    """Retrieve the top-k most relevant chunks for a query."""
    q_emb = get_embeddings([query])
    faiss.normalize_L2(q_emb)
    _, indices = index.search(q_emb, k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]


def trim_context_to_budget(chunks: list[str], budget: int = MAX_CTX_TOKS) -> str:
    """
    Join retrieved chunks and trim so the total token count
    stays under the budget (cost guard).
    """
    enc = tiktoken.encoding_for_model("gpt-4o-mini")
    context, tokens_used = [], 0
    for chunk in chunks:
        chunk_toks = len(enc.encode(chunk))
        if tokens_used + chunk_toks > budget:
            break
        context.append(chunk)
        tokens_used += chunk_toks
    return "\n\n---\n\n".join(context)


def count_extractable_chars(reader: PdfReader) -> int:
    """Return total number of extracted characters (proxy for text density)."""
    total = 0
    for page in reader.pages:
        total += len(page.extract_text() or "")
    return total


def extract_full_text(reader: PdfReader) -> str:
    parts = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            parts.append(text)
    return "\n".join(parts)


def estimate_tokens(text: str) -> int:
    enc = tiktoken.encoding_for_model("gpt-4o-mini")
    return len(enc.encode(text))

# ════════════════════════════════════════════════════════════════════════════
#  FILE UPLOAD & VALIDATION ("BOUNCER")
# ════════════════════════════════════════════════════════════════════════════
uploaded_file = st.file_uploader(
    "Upload PDF Document",
    type=["pdf"],
    label_visibility="collapsed",
)

if uploaded_file is not None:

    # ── Rule 1: File size ────────────────────────────────────────────────
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > MAX_FILE_MB:
        security_alert(
            f"File size <strong>{file_size_mb:.2f} MB</strong> exceeds the "
            f"{MAX_FILE_MB} MB limit. Upload a smaller document."
        )
        st.stop()

    # ── Rule 2: Page count ───────────────────────────────────────────────
    reader = PdfReader(uploaded_file)
    page_count = len(reader.pages)
    if page_count > MAX_PAGES:
        security_alert(
            f"Document has <strong>{page_count} pages</strong>, exceeding "
            f"the {MAX_PAGES}-page limit."
        )
        st.stop()

    # ── Rule 3: Text density ─────────────────────────────────────────────
    char_count = count_extractable_chars(reader)
    chars_per_page = char_count / max(page_count, 1)
    if chars_per_page < 50:                              # heuristic for scanned PDF
        security_alert(
            "This document appears to contain scanned images rather than "
            "selectable text. Please upload a text-based PDF."
        )
        st.stop()

    # ── Embed only if new file ───────────────────────────────────────────
    if st.session_state["file_name"] != uploaded_file.name:
        st.session_state["vector_store"]  = None
        st.session_state["chunks"]        = []
        st.session_state["chat_history"]  = []
        st.session_state["doc_stats"]     = {}
        st.session_state["file_name"]     = uploaded_file.name

    if st.session_state["vector_store"] is None:
        with st.spinner("Analyzing document..."):
            full_text = extract_full_text(reader)
            chunks    = chunk_text(full_text)
            index     = build_faiss_index(chunks)

            st.session_state["chunks"]       = chunks
            st.session_state["vector_store"] = index
            st.session_state["doc_stats"]    = {
                "pages":  page_count,
                "chunks": len(chunks),
                "tokens": estimate_tokens(full_text),
            }

        st.rerun()   # refresh sidebar stats

    # ── Document ready feedback ──────────────────────────────────────────
    stats = st.session_state["doc_stats"]
    st.markdown(f"""
    <div class="success-card">
        READY &nbsp;|&nbsp; {uploaded_file.name} &nbsp;|&nbsp;
        {stats.get('pages')} pages &nbsp;/&nbsp;
        {stats.get('chunks')} chunks &nbsp;/&nbsp;
        ~{stats.get('tokens'):,} tokens
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

# ════════════════════════════════════════════════════════════════════════════
#  CHAT INTERFACE
# ════════════════════════════════════════════════════════════════════════════
if st.session_state["vector_store"] is not None:

    # ── Render history ───────────────────────────────────────────────────
    for msg in st.session_state["chat_history"]:
     with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── Chat input ───────────────────────────────────────────────────────
    user_input = st.chat_input("Ask anything about your document...")

    if user_input:
        # Append & display user message
        st.session_state["chat_history"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # ── Retrieve & trim context ──────────────────────────────────────
        relevant_chunks = retrieve(
            user_input,
            st.session_state["vector_store"],
            st.session_state["chunks"],
        )
        context = trim_context_to_budget(relevant_chunks)

        # ── Build messages payload ───────────────────────────────────────
        messages_payload = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"**Context from document:**\n\n{context}\n\n"
                    f"---\n\n**Question:** {user_input}"
                ),
            },
        ]

        # ── Stream response ──────────────────────────────────────────────
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""

            stream = client.chat.completions.create(
                model       = CHAT_MODEL,
                messages    = messages_payload,
                max_tokens  = MAX_TOKENS,
                temperature = 0.3,
                stream      = True,
            )

            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                full_response += delta
                response_placeholder.markdown(full_response + "▌")

            response_placeholder.markdown(full_response)

        st.session_state["chat_history"].append(
            {"role": "assistant", "content": full_response}
        )

else:
    # ── No document uploaded yet ─────────────────────────────────────────
    st.markdown("""
    <div style="
        text-align: center;
        padding: 60px 20px;
        color: var(--muted);
        font-family: var(--font-body);
        font-size: 15px;
        line-height: 1.8;
    ">
        Upload a PDF above to activate the chat interface.<br>
        <span style="font-family: var(--font-mono); font-size: 12px; color: #3A4560;">
            Awaiting document...
        </span>
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
#  FOOTER
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="footer-bar">
    Developed by Bibek Subedi &nbsp;|&nbsp; &copy; 2026 All Rights Reserved
</div>
""", unsafe_allow_html=True)
