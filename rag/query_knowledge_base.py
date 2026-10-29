"""Ask questions against retrieved medical text, using local Ollama generation.
Run from project root: python rag/query_knowledge_base.py
"""
from pathlib import Path
import requests
import chromadb

ROOT = Path(__file__).resolve().parents[1]
DB_DIR = ROOT / "rag" / "chroma_db"
COLLECTION = "skin_disease_medical_knowledge"
OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2:3b"


def embed(text):
    r = requests.post(f"{OLLAMA_URL}/api/embed", json={"model": EMBED_MODEL, "input": text}, timeout=120)
    r.raise_for_status()
    data = r.json()
    if not data.get("embeddings"):
        raise RuntimeError(f"Ollama returned no embedding: {data}")
    return data["embeddings"][0]


def retrieve(question, k=4):
    client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = client.get_collection(COLLECTION)
    if collection.count() == 0:
        return []
    result = collection.query(query_embeddings=[embed(question)], n_results=min(k, collection.count()), include=["documents", "metadatas", "distances"])
    return [{"text": d, "source": m.get("source", "unknown"), "distance": dist}
            for d, m, dist in zip(result["documents"][0], result["metadatas"][0], result["distances"][0])]


def generate(question, hits):
    context = "\n\n".join(f"[Source file: {h['source']}]\n{h['text']}" for h in hits)
    prompt = f"""You are a cautious educational health-information assistant. Use the supplied context as the primary source.
Answer with headings: What it is; Common symptoms; Possible causes/contributing factors; General care and precautions; What to avoid; Treatment options to discuss with a clinician; When to seek medical help.
Do not diagnose a person, prescribe medicine or dosages, or claim an image proves a disease. If context lacks information, say so. Explain that similar skin conditions can look alike and clinician assessment may be needed. Recommend urgent local care for emergency warning signs. End by stating this is educational information, not a substitute for professional medical advice. Mention source filenames; do not invent citations.

CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nAnswer:"""
    r = requests.post(f"{OLLAMA_URL}/api/generate", json={"model": LLM_MODEL, "prompt": prompt, "stream": False}, timeout=240)
    r.raise_for_status()
    return r.json().get("response", "").strip()


def main():
    print("Skin Disease RAG (local Ollama). Type 'exit' to quit.")
    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        try:
            hits = retrieve(question)
            if not hits:
                print("No relevant information found. Build the knowledge base first.")
                continue
            print("\n" + generate(question, hits))
            print("\nRetrieved files:")
            for source in sorted({h["source"] for h in hits}):
                print("-", source)
        except Exception as e:
            print(f"\nError: {e}\nCheck Ollama is running, models are downloaded, and the index was built.")


if __name__ == "__main__":
    main()
