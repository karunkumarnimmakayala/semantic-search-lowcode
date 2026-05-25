# Semantic Search for Low-Code Developers

A RAG-powered semantic search tool that lets developers query **Pega** and **Mendix** documentation using natural language — instead of keyword search.

Built as a portfolio project to demonstrate hands-on RAG, vector search, and LLM integration skills.

---

## What it does

- User types a business requirement or question in plain English
- The tool retrieves the most semantically relevant documentation chunks using vector search
- Passes them to an LLM to synthesise a practical, accurate answer
- Returns the answer alongside source references so the user can verify

---

## Tech Stack

| Layer | Technology |
|---|---|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` via Hugging Face |
| Vector store | Qdrant Cloud (free tier) |
| LLM Generation | Kimi K2 via NVIDIA NIM API |
| Backend | FastAPI (Python) |
| Frontend | React (coming soon) |

---

## RAG Architecture

 INGESTION (offline, runs once)
──────────────────────────────
Pega PDFs + Mendix GitHub docs
→ chunk (500 words, 50 overlap)
→ embed (384-dim vectors)
→ store in Qdrant Cloud
QUERY (online, per request)
────────────────────────────
User question
→ embed question
→ cosine similarity search → top-k chunks
→ LLM generates answer from chunks only
→ return answer + source references

---

## Knowledge Base

| Collection | Source | Chunks |
|---|---|---|
| `pega_docs` | Official Pega Platform PDFs (v8.4, v8.6, v8.7) | 899 |
| `mendix_docs` | Mendix GitHub docs repo (markdown) | 819 |
| **Total** | | **1,718 chunks** |

---

## Project Structure

semantic_search/
├── backend/
│   ├── main.py              ← FastAPI app + RAG pipeline
│   ├── requirements.txt     ← Python dependencies
│   └── test_connections.py  ← verify HF + Qdrant connections
├── scripts/
│   ├── ingest.py            ← data ingestion pipeline
│   └── test_search.py       ← verify vector search works
├── data/
│   └── pega_pdfs/           ← Pega PDF source files (not committed)
├── frontend/                ← React app (coming soon)
├── .env.example             ← environment variable template
└── README.md


---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/karunkumarnimmakayala/semantic-search-lowcode.git
cd semantic-search-lowcode
```

### 2. Create your `.env` file

```env
HF_API_TOKEN=your_huggingface_token
QDRANT_URL=your_qdrant_cluster_url
QDRANT_API_KEY=your_qdrant_api_key
NVIDIA_API_KEY=your_nvidia_nim_key
PEGA_COLLECTION=pega_docs
MENDIX_COLLECTION=mendix_docs
```

### 3. Install dependencies

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
```

### 4. Run ingestion (seeds Qdrant)

```bash
cd scripts
python ingest.py
```

### 5. Start the backend

```bash
cd backend
uvicorn main:app --reload --port 8000
```

### 6. Test the API

Visit `http://localhost:8000/docs` for the interactive Swagger UI.

---

## API Endpoints

### `GET /health`
Returns API status and collection names.

### `POST /search`
```json
{
  "question": "How do stages work in Pega case management?",
  "source": "pega",
  "top_k": 3
}
```

**Response:**
```json
{
  "answer": "Stages in Pega represent major phases...",
  "sources": [
    {
      "title": "Case Management 87",
      "source": "https://docs.pega.com/Case_management-87",
      "text": "...",
      "score": 0.705
    }
  ],
  "collection_searched": "pega_docs"
}
```

---

## Why this project exists

Portfolio project demonstrating hands-on RAG, vector search, and LLM integration — relevant for roles at companies like **Qdrant**, **Rabobank**, **Prosus**, and any AI-first engineering team.

Buit by a developer we is curious to learn RAG 
---

## Accounts needed to run this

- [Hugging Face](https://huggingface.co) — free account + API token
- [Qdrant Cloud](https://cloud.qdrant.io) — free tier cluster
- [NVIDIA NIM](https://build.nvidia.com) — free API credits