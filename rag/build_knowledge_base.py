"""Index medical_knowledge/*.txt with Ollama embeddings and ChromaDB.
Run from project root: python rag/build_knowledge_base.py
Prerequisites: ollama pull nomic-embed-text; pip install -r requirements.txt
"""
from pathlib import Path
import hashlib
import requests
import chromadb

ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_DIR = ROOT / "medical_knowledge"
DB_DIR = ROOT / "rag" / "chroma_db"
COLLECTION = "skin_disease_medical_knowledge"
OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
CHUNK_SIZE, OVERLAP = 1000, 150


def chunks(text):
    text = " ".join(text.split())
    out, start = [], 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        if text[start:end].strip():
            out.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(end - OVERLAP, start + 1)
    return out


def embed(text):
    r = requests.post(f"{OLLAMA_URL}/api/embed", json={"model": EMBED_MODEL, "input": text}, timeout=120)
    r.raise_for_status()
    data = r.json()
    if not data.get("embeddings"):
        raise RuntimeError(f"Ollama returned no embedding: {data}")
    return data["embeddings"][0]


def main():
    files = sorted(KNOWLEDGE_DIR.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No knowledge .txt files in {KNOWLEDGE_DIR}")
    client = chromadb.PersistentClient(path=str(DB_DIR))
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    ids, docs, metas, vectors = [], [], [], []
    for path in files:
        for i, part in enumerate(chunks(path.read_text(encoding="utf-8"))):
            ids.append(hashlib.sha1(f"{path.name}:{i}:{part}".encode()).hexdigest())
            docs.append(part)
            metas.append({"source": path.name, "chunk": i})
            vectors.append(embed(part))
            print(f"Indexed {path.name} chunk {i + 1}")
    collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=vectors)
    print(f"Done: indexed {len(docs)} chunks in {DB_DIR}")


if __name__ == "__main__":
    main()
