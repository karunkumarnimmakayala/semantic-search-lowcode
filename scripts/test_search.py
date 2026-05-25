import os
import requests
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv(dotenv_path="../.env")

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

HF_EMBED_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"

def embed(text):
    headers = {
        "Authorization": f"Bearer {os.getenv('HF_API_TOKEN')}",
        "Content-Type": "application/json"
    }
    response = requests.post(HF_EMBED_URL, headers=headers, json={"inputs": [text]})
    return response.json()[0]

def search(question, collection, top_k=3):
    vector = embed(question)
    response = client.query_points(
        collection_name=collection,
        query=vector,
        limit=top_k
    )
    return response.points

def print_results(question, collection):
    print(f"Question: {question}\n")
    results = search(question, collection)
    for i, r in enumerate(results):
        print(f"Result {i+1} (score: {r.score:.3f})")
        print(f"Source: {r.payload['source']}")
        print(f"Text:   {r.payload['text'][:200]}...")
        print()

# ── Test Pega ────────────────────────────────────────────────
print("=" * 50)
print("PEGA SEARCH TEST")
print("=" * 50)
print_results("How do stages work in Pega case management?", "pega_docs")

# ── Test Mendix ──────────────────────────────────────────────
print("=" * 50)
print("MENDIX SEARCH TEST")
print("=" * 50)
print_results("How do I create a microflow in Mendix?", "mendix_docs")