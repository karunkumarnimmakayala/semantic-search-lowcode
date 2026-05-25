import os
import time
import requests
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from qdrant_client import QdrantClient

load_dotenv(dotenv_path="../.env")

# ── Config ───────────────────────────────────────────────────
QDRANT_URL        = os.getenv("QDRANT_URL")
QDRANT_API_KEY    = os.getenv("QDRANT_API_KEY")
HF_API_TOKEN      = os.getenv("HF_API_TOKEN")
PEGA_COLLECTION   = os.getenv("PEGA_COLLECTION")
MENDIX_COLLECTION = os.getenv("MENDIX_COLLECTION")

HF_EMBED_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
#***HF_LLM_URL   = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3/v1/chat/completions"***

VECTOR_SIZE = 384
TOP_K       = 3

# ── Clients ──────────────────────────────────────────────────
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# ── FastAPI app ──────────────────────────────────────────────
app = FastAPI(
    title="Semantic Search for Low-Code Developers",
    description="RAG-powered search over Pega and Mendix documentation",
    version="1.0.0"
)

# ── CORS — allows React frontend to call this API ────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173",  # Vite dev server
                   "http://localhost:3000",  # CRA dev server
                   "*"],                     # all origins for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request / Response models ─────────────────────────────────
class SearchRequest(BaseModel):
    question: str
    source: str = "pega"  # "pega", "mendix", or "both"
    top_k: int = 3

class SourceReference(BaseModel):
    title: str
    source: str
    text: str
    score: float

class SearchResponse(BaseModel):
    answer: str
    sources: list[SourceReference]
    collection_searched: str

# ── Helper: embed a question ──────────────────────────────────
def embed_question(text: str) -> list[float]:
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json"
    }
    print(f"Embedding question: {text[:50]}")
    print(f"HF Token loaded: {HF_API_TOKEN[:8] if HF_API_TOKEN else 'MISSING'}")
    response = requests.post(
        HF_EMBED_URL,
        headers=headers,
        json={"inputs": [text]}
    )
    print(f"Embed status: {response.status_code}")
    print(f"Embed response: {response.text[:200]}")
    if response.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Embedding failed: {response.text}")
    return response.json()[0]
# ── Helper: search Qdrant ─────────────────────────────────────
def search_collection(vector: list[float], collection: str, top_k: int):
    response = qdrant.query_points(
        collection_name=collection,
        query=vector,
        limit=top_k
    )
    return response.points

# ── Helper: generate answer with Mistral ──────────────────────

def generate_answer(question: str, chunks: list) -> str:
    context = "\n\n".join([
        f"[Source: {c.payload['title']}]\n{c.payload['text']}"
        for c in chunks
    ])

    prompt = f"""You are a helpful assistant for low-code developers working with Pega and Mendix.
Answer the question based ONLY on the provided documentation context.
If the context does not contain enough information, say so clearly.
Be concise, practical, and specific.

Documentation context:
{context}

Question: {question}

Answer:"""

    from openai import OpenAI
    llm_client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.getenv("NVIDIA_API_KEY")
    )

    response = llm_client.chat.completions.create(
        model="moonshotai/kimi-k2-instruct-0905",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=512,
        stream=False
    )

    return response.choices[0].message.content.strip()

# ── Routes ────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "collections": {
            "pega":   PEGA_COLLECTION,
            "mendix": MENDIX_COLLECTION
        }
    }

@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if request.source not in ("pega", "mendix", "both"):
        raise HTTPException(status_code=400, detail="source must be 'pega', 'mendix', or 'both'")

    # Embed the question
    vector = embed_question(request.question)

    # Search relevant collections
    chunks = []
    if request.source == "pega":
        chunks = search_collection(vector, PEGA_COLLECTION, request.top_k)
        collection_searched = PEGA_COLLECTION

    elif request.source == "mendix":
        chunks = search_collection(vector, MENDIX_COLLECTION, request.top_k)
        collection_searched = MENDIX_COLLECTION

    elif request.source == "both":
        pega_chunks   = search_collection(vector, PEGA_COLLECTION, request.top_k)
        mendix_chunks = search_collection(vector, MENDIX_COLLECTION, request.top_k)
        # Merge and sort by score, take top_k overall
        chunks = sorted(
            pega_chunks + mendix_chunks,
            key=lambda x: x.score,
            reverse=True
        )[:request.top_k]
        collection_searched = "both"

    if not chunks:
        raise HTTPException(status_code=404, detail="No relevant chunks found")

    # Generate answer
    answer = generate_answer(request.question, chunks)

    # Build source references
    sources = [
        SourceReference(
            title=c.payload.get("title", "Unknown"),
            source=c.payload.get("source", ""),
            text=c.payload["text"][:300],
            score=round(c.score, 3)
        )
        for c in chunks
    ]

    return SearchResponse(
        answer=answer,
        sources=sources,
        collection_searched=collection_searched
    )