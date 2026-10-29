# Skin Disease RAG Knowledge Base (Ollama + ChromaDB)

Included: six starter medical knowledge files, `sources.csv`, and two scripts to index/retrieve content and generate answers with a local Ollama LLM.

## Setup (run commands from this folder)
1. Install [Ollama](https://ollama.com/) and ensure it is running.
2. Install Python packages:
   ```bash
   python -m pip install -r requirements.txt
   ```
3. Download local models:
   ```bash
   ollama pull nomic-embed-text
   ollama pull llama3.2:3b
   ```
4. Build the vector index:
   ```bash
   python rag/build_knowledge_base.py
   ```
5. Ask questions:
   ```bash
   python rag/query_knowledge_base.py
   ```
   Type `exit` to quit.

Edit the `.txt` files to expand the content; rerun the build script after edits. `sources.csv` records the reference pages used for this starter summary.

## How it works
Knowledge text -> chunks -> Ollama embeddings -> ChromaDB retrieval -> Ollama generates a grounded explanation. This does not fine-tune/train the LLM. It is separate from the image CNN: CNN predicts a class; RAG retrieves text; LLM explains the retrieved text.

## Medical limitation
These files are educational starter material, not a diagnosis or personal treatment plan. A skin image can be ambiguous and a model can be wrong. Have a qualified healthcare professional review content before public deployment. Do not self-prescribe or delay care based on AI output.
