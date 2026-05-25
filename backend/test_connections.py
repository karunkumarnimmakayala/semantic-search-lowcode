import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
import requests

# Load .env from project root (one level up from backend)
load_dotenv(dotenv_path="../.env")

# ── Test 1: Qdrant ──────────────────────────────────────────
print("Testing Qdrant connection...")
client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)
collections = client.get_collections()
print(f"✅ Qdrant connected — collections: {collections}")

# ── Test 2: Hugging Face ────────────────────────────────────

print("\nTesting Hugging Face connection...")
headers = {
    "Authorization": f"Bearer {os.getenv('HF_API_TOKEN')}",
    "Content-Type": "application/json"
}
response = requests.post(
    "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction",
    headers=headers,
    json={"inputs": ["test connection"]}
)
print(f"HF status code: {response.status_code}")
if response.status_code == 200:
    result = response.json()
    print(f"✅ Hugging Face connected — vector length: {len(result[0])}")
else:
    print(f"❌ HF error: {response.text}")

# ── Test 3: ENV vars loaded ─────────────────────────────────
print("\nChecking environment variables...")
vars_to_check = ["HF_API_TOKEN", "QDRANT_URL", "QDRANT_API_KEY", "PEGA_COLLECTION", "MENDIX_COLLECTION"]
for var in vars_to_check:
    value = os.getenv(var)
    if value:
        # Only show first 8 chars for security
        preview = value[:8] + "..." if len(value) > 8 else value
        print(f"✅ {var} = {preview}")
    else:
        print(f"❌ {var} is missing")