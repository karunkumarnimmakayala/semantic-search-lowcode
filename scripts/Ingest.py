import os
import time
import requests
import shutil
import hashlib
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from bs4 import BeautifulSoup
from git import Repo

# ── Load environment variables ──────────────────────────────
load_dotenv(dotenv_path="../.env")

HF_API_TOKEN    = os.getenv("HF_API_TOKEN")
QDRANT_URL      = os.getenv("QDRANT_URL")
QDRANT_API_KEY  = os.getenv("QDRANT_API_KEY")
PEGA_COLLECTION    = os.getenv("PEGA_COLLECTION")
MENDIX_COLLECTION  = os.getenv("MENDIX_COLLECTION")

HF_EMBED_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
VECTOR_SIZE  = 384
CHUNK_SIZE   = 500   # words per chunk
CHUNK_OVERLAP = 50   # words overlap between chunks

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# ── STEP 1: Chunking ─────────────────────────────────────────
def chunk_text(text, source_url, title=""):
    """Split text into overlapping word chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + CHUNK_SIZE
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)
        if len(chunk_text.strip()) > 100:  # skip tiny chunks
            chunks.append({
                "text": chunk_text,
                "source": source_url,
                "title": title
            })
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

# ── STEP 2: Embedding ────────────────────────────────────────
def embed_texts(texts):
    """Send texts to HF API and get back 384-dim vectors."""
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json"
    }
    response = requests.post(
        HF_EMBED_URL,
        headers=headers,
        json={"inputs": texts}
    )
    if response.status_code != 200:
        print(f"❌ Embedding error: {response.text}")
        return None
    return response.json()

# ── STEP 3: Qdrant collection setup ─────────────────────────
def create_collection(name, recreate=False):
    """Create a Qdrant collection, optionally wiping it first."""
    existing = [c.name for c in client.get_collections().collections]
    if name in existing:
        if recreate:
            client.delete_collection(name)
            print(f"🗑️  Deleted existing collection: {name}")
        else:
            print(f"Collection '{name}' already exists — skipping")
            return
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
    )
    print(f"✅ Created collection: {name}")

# ── STEP 4: Seed chunks into Qdrant ─────────────────────────
def seed_collection(collection_name, chunks):
    """Embed and upsert all chunks into a Qdrant collection."""
    print(f"\nSeeding {len(chunks)} chunks into '{collection_name}'...")
    batch_size = 16  # HF free tier handles 16 at a time comfortably
    points = []
    point_id = 0

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [c["text"] for c in batch]

        print(f"  Embedding batch {i // batch_size + 1} / {(len(chunks) - 1) // batch_size + 1}...")
        vectors = embed_texts(texts)
        if vectors is None:
            continue

        for chunk, vector in zip(batch, vectors):
            # Deterministic ID from content hash — re-upserting same
            # chunk overwrites rather than duplicates
            chunk_hash = hashlib.md5(
                (chunk["source"] + chunk["text"][:100]).encode()
            ).hexdigest()
            # Qdrant needs integer or UUID — convert hex to int
            point_id = int(chunk_hash[:16], 16)

            points.append(PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "title": chunk["title"]
                }
            ))

        time.sleep(1)  # be polite to HF free tier

    if not points:
        print(f"⚠️ No points to seed into '{collection_name}' — skipping")
        return
    client.upsert(collection_name=collection_name, points=points)
    print(f"✅ Seeded {len(points)} chunks into '{collection_name}'")

# ── MENDIX LOADER ────────────────────────────────────────────
def load_mendix_docs():
    """Clone mendix/docs from GitHub and read markdown files."""
    repo_path = Path("../data/mendix_docs")

    if repo_path.exists():
        print("Mendix repo already cloned — skipping clone")
    else:
        print("Cloning Mendix docs from GitHub (this takes ~1-2 min)...")
        Repo.clone_from(
            "https://github.com/mendix/docs.git",
            str(repo_path),
            depth=1  # shallow clone — only latest snapshot, much faster
        )
        print("✅ Mendix repo cloned")

    # Read all markdown files under content/en/docs/
    docs_path = repo_path / "content" / "en" / "docs"
    chunks = []
    md_files = list(docs_path.rglob("*.md"))
    print(f"Found {len(md_files)} markdown files")

    for md_file in md_files[:200]:  # limit to 200 files for free tier
        try:
            text = md_file.read_text(encoding="utf-8", errors="ignore")
            # Strip Hugo frontmatter (--- block at top)
            if text.startswith("---"):
                end = text.find("---", 3)
                if end != -1:
                    text = text[end + 3:].strip()
            # Use file path as source URL
            relative = md_file.relative_to(repo_path)
            source = f"https://docs.mendix.com/{relative}"
            title = md_file.stem.replace("-", " ").title()
            chunks.extend(chunk_text(text, source, title))
        except Exception as e:
            print(f"Skipping {md_file.name}: {e}")

    print(f"✅ Mendix: {len(chunks)} chunks ready")
    return chunks

# ── PEGA LOADER ──────────────────────────────────────────────
def load_pega_docs():
    """Extract text from Pega PDF files and chunk them."""
    import PyPDF2

    pdf_folder = Path("../data/pega_pdfs")

    if not pdf_folder.exists():
        print("❌ data/pega_pdfs/ folder not found")
        return []

    pdf_files = list(pdf_folder.glob("*.pdf"))
    if not pdf_files:
        print("❌ No PDF files found in data/pega_pdfs/")
        return []

    print(f"Found {len(pdf_files)} PDF files")
    all_chunks = []

    for pdf_path in pdf_files:
        print(f"  Reading: {pdf_path.name}")
        try:
            with open(pdf_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                total_pages = len(reader.pages)
                print(f"    Pages: {total_pages}")

                full_text = ""
                for page_num, page in enumerate(reader.pages):
                    try:
                        text = page.extract_text()
                        if text:
                            full_text += text + "\n"
                    except Exception as e:
                        print(f"    ⚠️ Skipping page {page_num}: {e}")
                        continue

                if len(full_text.strip()) < 100:
                    print(f"    ⚠️ Too little text extracted from {pdf_path.name}")
                    continue

                # Use filename as source reference
                source = f"https://docs.pega.com/{pdf_path.stem}"
                title = pdf_path.stem.replace("-", " ").title()

                chunks = chunk_text(full_text, source, title)
                all_chunks.extend(chunks)
                print(f"    ✅ {len(chunks)} chunks from {pdf_path.name}")

        except Exception as e:
            print(f"  ❌ Error reading {pdf_path.name}: {e}")
            continue

    print(f"✅ Pega: {len(all_chunks)} chunks total")
    return all_chunks  

# ── MAIN ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG Ingestion Pipeline")
    parser.add_argument("--source", choices=["all", "mendix", "pega"],
                        default="all", help="Which source to ingest")
    parser.add_argument("--recreate", action="store_true",
                        help="Wipe and recreate Qdrant collections")
    args = parser.parse_args()

    print("=" * 50)
    print("RAG Ingestion Pipeline")
    print(f"Source: {args.source} | Recreate: {args.recreate}")
    print("=" * 50)

    mendix_chunks = []
    pega_chunks = []

    if args.source in ("all", "mendix"):
        create_collection(MENDIX_COLLECTION, recreate=args.recreate)
        print("\n── Mendix Docs ──────────────────────────")
        mendix_chunks = load_mendix_docs()
        seed_collection(MENDIX_COLLECTION, mendix_chunks)

    if args.source in ("all", "pega"):
        create_collection(PEGA_COLLECTION, recreate=args.recreate)
        print("\n── Pega Docs ────────────────────────────")
        pega_chunks = load_pega_docs()
        seed_collection(PEGA_COLLECTION, pega_chunks)

    print("\n" + "=" * 50)
    print("✅ Ingestion complete!")
    if mendix_chunks:
        print(f"   Mendix chunks: {len(mendix_chunks)}")
    if pega_chunks:
        print(f"   Pega chunks:   {len(pega_chunks)}")
    print("=" * 50)