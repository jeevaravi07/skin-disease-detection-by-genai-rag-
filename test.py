import os
import json
import hashlib
import threading
import traceback
import requests
import chromadb
import webbrowser
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from urllib.parse import quote_plus

import numpy as np
from PIL import Image, ImageTk

try:
    import tensorflow as tf
except ImportError:
    tf = None


# ============================================================
# CONFIGURATION
# ============================================================

IMG_SIZE = (224, 224)
OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2:3b"

CHUNK_CHARS = 1000
CHUNK_OVERLAP = 150
TOP_K = 5

DISCLAIMER = (
    "AI-assisted screening and general information only. This is not a confirmed "
    "medical diagnosis or prescription. A skin image alone cannot establish the "
    "exact cause or appropriate treatment."
)

REQUEST_TIMEOUT = 180
GENERATION_TIMEOUT = 300


# ============================================================
# OLLAMA TESTS
# ============================================================

def check_ollama_connection():
    """Check that the local Ollama HTTP API is responding."""
    response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
    response.raise_for_status()
    payload = response.json()
    installed = {
        item.get("name", "").split(":")[0]
        for item in payload.get("models", [])
    }

    if LLM_MODEL.split(":")[0] not in installed:
        raise RuntimeError(
            f"Required LLM '{LLM_MODEL}' was not found. Run: "
            f"ollama pull {LLM_MODEL}"
        )

    if EMBED_MODEL.split(":")[0] not in installed:
        raise RuntimeError(
            f"Required embedding model '{EMBED_MODEL}' was not found. Run: "
            f"ollama pull {EMBED_MODEL}"
        )

    return payload


def ollama_embedding(text):
    """Return one vector from Ollama's /api/embed endpoint."""
    response = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": text},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()
    embeddings = data.get("embeddings")

    if not embeddings or not embeddings[0]:
        raise RuntimeError(
            f"Ollama returned no embedding for '{EMBED_MODEL}'. "
            "Check the Ollama API and installed model."
        )

    return embeddings[0]


def ollama_generate(prompt):
    """Generate text with the local Ollama language model."""
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False,
        },
        timeout=GENERATION_TIMEOUT,
    )
    response.raise_for_status()
    answer = response.json().get("response", "").strip()

    if not answer:
        raise RuntimeError(
            f"Ollama model '{LLM_MODEL}' returned an empty response."
        )

    return answer


def run_ollama_test():
    """Test API connectivity, embeddings, and text generation."""
    check_ollama_connection()
    vector = ollama_embedding("Skin disease symptoms and causes")
    if not vector:
        raise RuntimeError("Embedding test returned an empty vector.")

    answer = ollama_generate(
        "Explain eczema in two simple sentences. Keep the answer brief."
    )
    return (
        "Ollama test successful.\n\n"
        f"LLM model: {LLM_MODEL}\n"
        f"Embedding model: {EMBED_MODEL}\n"
        f"Embedding dimensions: {len(vector)}\n\n"
        "Generation test response:\n"
        f"{answer}"
    )


# ============================================================
# RAG / CHROMADB
# ============================================================

def split_text(text, chunk_size=CHUNK_CHARS, overlap=CHUNK_OVERLAP):
    text = " ".join(text.split())
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(end - overlap, start + 1)

    return chunks


def get_rag_collection(progress_callback=None):
    """
    Load the persistent ChromaDB collection.
    On first run, embed and index .txt files in ./medical_knowledge.
    """
    app_dir = Path(__file__).resolve().parent
    knowledge_dir = app_dir / "medical_knowledge"
    db_dir = app_dir / "rag" / "chroma_db"

    if not knowledge_dir.is_dir():
        raise FileNotFoundError(
            "The medical_knowledge folder was not found beside app.py.\n"
            f"Expected folder: {knowledge_dir}"
        )

    files = sorted(knowledge_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(
            "No .txt medical knowledge files were found in:\n"
            f"{knowledge_dir}"
        )

    check_ollama_connection()

    db_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(db_dir))
    collection = client.get_or_create_collection(
        name="skin_disease_medical_knowledge",
        metadata={"hnsw:space": "cosine"},
    )

    # Index on first run. If you change the knowledge files later and want
    # a complete rebuild, close the app and delete rag/chroma_db.
    if collection.count() == 0:
        docs, ids, metas, vectors = [], [], [], []
        total_files = len(files)

        for file_index, file_path in enumerate(files, start=1):
            if progress_callback:
                progress_callback(
                    f"RAG indexing: reading {file_index}/{total_files}: "
                    f"{file_path.name}"
                )

            content = file_path.read_text(encoding="utf-8", errors="replace")
            chunks = split_text(content)

            for chunk_no, chunk in enumerate(chunks):
                chunk_id = hashlib.sha1(
                    f"{file_path.name}:{chunk_no}:{chunk}".encode("utf-8")
                ).hexdigest()

                if progress_callback:
                    progress_callback(
                        f"RAG indexing: embedding {file_path.name}, "
                        f"chunk {chunk_no + 1}/{len(chunks)}"
                    )

                docs.append(chunk)
                ids.append(chunk_id)
                metas.append({
                    "source": file_path.name,
                    "chunk": chunk_no,
                })
                vectors.append(ollama_embedding(chunk))

        if not docs:
            raise RuntimeError(
                "No text chunks were created. Check that the knowledge files "
                "contain readable text."
            )

        # Add in batches to avoid overly large ChromaDB requests.
        batch_size = 64
        for start in range(0, len(docs), batch_size):
            end = start + batch_size
            collection.add(
                ids=ids[start:end],
                documents=docs[start:end],
                metadatas=metas[start:end],
                embeddings=vectors[start:end],
            )

    if collection.count() == 0:
        raise RuntimeError("The ChromaDB knowledge base is empty.")

    return collection


def retrieve_medical_context(disease, progress_callback=None, top_k=TOP_K):
    question = (
        f"{disease}: what it is, common symptoms, possible causes and "
        "contributing factors, general care and precautions, what to avoid, "
        "treatment options to discuss with a clinician, and when to seek help."
    )

    if progress_callback:
        progress_callback("RAG: loading medical knowledge and searching documents...")

    collection = get_rag_collection(progress_callback=progress_callback)
    count = collection.count()

    query_vector = ollama_embedding(question)
    result = collection.query(
        query_embeddings=[query_vector],
        n_results=min(top_k, count),
        include=["documents", "metadatas"],
    )

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]

    context_parts = []
    sources = set()

    for document, metadata in zip(documents, metadatas):
        source = (metadata or {}).get("source", "unknown")
        context_parts.append(f"[Source file: {source}]\n{document}")
        sources.add(source)

    if not context_parts:
        raise RuntimeError(
            "RAG did not retrieve any documents. Check the medical_knowledge "
            "text files and ChromaDB index."
        )

    return "\n\n".join(context_parts), sorted(sources)


def generate_genai_explanation(disease, score, progress_callback=None):
    context, sources = retrieve_medical_context(
        disease,
        progress_callback=progress_callback,
    )

    prompt = f"""You are a cautious health-information assistant for an educational
skin-disease screening project.

The image classifier predicted: {disease}
Raw model output score: {score:.4f}. This is NOT a calibrated probability.

Use the retrieved context as your primary source. Explain:
1. What the condition is
2. Common symptoms
3. Possible causes and contributing factors
4. General care and precautions
5. What to avoid
6. Treatment options to discuss with a qualified clinician
7. When to seek medical help

Safety rules:
- Do not confirm a diagnosis from an image or model prediction.
- Do not prescribe medicine or provide dosages.
- Do not recommend stopping prescribed treatment.
- Similar skin conditions can look alike; recommend clinical assessment.
- Do not invent details that are absent from the retrieved context.
- Recommend prompt medical care for severe or worsening symptoms or signs
  of infection.
- State that the answer is educational and not a substitute for professional
  medical advice.
- End with a list of the source filenames used.

RETRIEVED MEDICAL CONTEXT:
{context}

Write a clear, well-organized answer in plain English."""

    if progress_callback:
        progress_callback("GenAI: asking local Ollama model to write the explanation...")

    answer = ollama_generate(prompt)
    answer += "\n\n" + DISCLAIMER

    if sources:
        answer += "\n\nRetrieved knowledge files: " + ", ".join(sources)

    return answer


# ============================================================
# TKINTER APPLICATION
# ============================================================

class SkinDiseaseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Skin Disease Detection Using Generative AI")
        self.root.geometry("1450x900")
        self.root.minsize(1050, 720)
        self.root.option_add("*Font", ("Segoe UI", 10))

        self.model = None
        self.class_names = None
        self.model_path = tk.StringVar()
        self.classes_path = tk.StringVar()
        self.location = tk.StringVar()
        self.image_path = None
        self.preview = None
        self.busy = False

        self._configure_style()
        self._build_ui()
        self._auto_find_model()

    def _configure_style(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"))
        style.configure("Section.TLabelframe.Label", font=("Segoe UI", 11, "bold"))
        style.configure("Action.TButton", padding=(8, 7))
        style.configure("Predict.TButton", padding=(8, 9), font=("Segoe UI", 10, "bold"))

    def _build_ui(self):
        """Simplified interface: location, image upload, prediction, and GenAI result."""
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        outer = ttk.Frame(self.root, padding=14)
        outer.grid(row=0, column=0, sticky="nsew")
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(3, weight=1)

        ttk.Label(
            outer, text="Skin Disease Detection Using Generative AI",
            style="Title.TLabel",
        ).grid(row=0, column=0, sticky="w", pady=(0, 5))

        ttk.Label(
            outer,
            text="Upload a skin image to receive an AI-assisted screening result and a GenAI explanation grounded in your medical knowledge files.",
            wraplength=1250,
        ).grid(row=1, column=0, sticky="w", pady=(0, 12))

        location_box = ttk.LabelFrame(
            outer, text="1. Nearby dermatologist search (optional)",
            padding=10, style="Section.TLabelframe",
        )
        location_box.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        location_box.columnconfigure(1, weight=1)

        ttk.Label(location_box, text="City / area / postcode:").grid(
            row=0, column=0, sticky="w", padx=(0, 8)
        )
        ttk.Entry(location_box, textvariable=self.location).grid(
            row=0, column=1, sticky="ew", padx=5
        )
        ttk.Button(
            location_box, text="Find nearby dermatologists",
            command=self.open_maps, style="Action.TButton",
        ).grid(row=0, column=2, padx=(5, 0))

        work = ttk.Frame(outer)
        work.grid(row=3, column=0, sticky="nsew")
        work.columnconfigure(0, weight=0, minsize=330)
        work.columnconfigure(1, weight=1, minsize=500)
        work.rowconfigure(0, weight=1)

        image_box = ttk.LabelFrame(
            work, text="2. Skin image", padding=10,
            style="Section.TLabelframe",
        )
        image_box.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        image_box.columnconfigure(0, weight=1)
        image_box.rowconfigure(2, weight=1)

        ttk.Button(
            image_box, text="Upload skin image",
            command=self.browse_image, style="Action.TButton",
        ).grid(row=0, column=0, sticky="ew", pady=(0, 6))

        self.predict_button = ttk.Button(
            image_box, text="Predict + GenAI Explain",
            command=self.start_prediction, state="disabled",
            style="Predict.TButton",
        )
        self.predict_button.grid(row=1, column=0, sticky="ew", pady=(0, 10))

        self.image_label = ttk.Label(
            image_box, text="No image selected", anchor="center", justify="center"
        )
        self.image_label.grid(row=2, column=0, sticky="nsew")

        result_box = ttk.LabelFrame(
            work, text="3. AI prediction and GenAI explanation",
            padding=12, style="Section.TLabelframe",
        )
        result_box.grid(row=0, column=1, sticky="nsew")
        result_box.columnconfigure(0, weight=1)
        result_box.rowconfigure(3, weight=1)

        self.prediction_label = ttk.Label(
            result_box, text="Predicted disease: —",
            font=("Segoe UI", 12, "bold"), wraplength=900,
        )
        self.prediction_label.grid(row=0, column=0, sticky="w", pady=(0, 4))

        self.score_label = ttk.Label(
            result_box, text="Model score: —", wraplength=900
        )
        self.score_label.grid(row=1, column=0, sticky="w", pady=3)

        self.conf_label = ttk.Label(
            result_box, text="Screening indicator: —", wraplength=900
        )
        self.conf_label.grid(row=2, column=0, sticky="w", pady=(3, 8))

        text_frame = ttk.Frame(result_box)
        text_frame.grid(row=3, column=0, sticky="nsew")
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)

        self.output = tk.Text(
            text_frame, wrap="word", font=("Segoe UI", 10),
            padx=10, pady=10, undo=False,
        )
        output_scroll = ttk.Scrollbar(
            text_frame, orient="vertical", command=self.output.yview
        )
        self.output.configure(yscrollcommand=output_scroll.set)
        self.output.grid(row=0, column=0, sticky="nsew")
        output_scroll.grid(row=0, column=1, sticky="ns")

        self.output.insert(
            "1.0",
            "HOW TO USE THIS APP\\n"
            "1. Wait for the AI model to load automatically.\\n"
            "2. Enter a city, area or postcode if you want to find a dermatologist.\\n"
            "3. Click Upload skin image and select a skin photo.\\n"
            "4. Click Predict + GenAI Explain.\\n\\n"
            "The CNN predicts a class. RAG retrieves relevant medical text, and the local generative AI creates the explanation.\\n\\n"
            + DISCLAIMER,
        )
        self.output.configure(state="disabled")

        bottom = ttk.Frame(outer)
        bottom.grid(row=4, column=0, sticky="ew", pady=(8, 0))
        bottom.columnconfigure(0, weight=1)

        self.status = tk.StringVar(value="Loading AI model...")
        ttk.Label(
            bottom, textvariable=self.status, relief="sunken", anchor="w"
        ).grid(row=0, column=0, sticky="ew")

        ttk.Label(
            bottom, text=DISCLAIMER, foreground="#8a3b12",
            wraplength=1300, justify="left",
        ).grid(row=1, column=0, sticky="w", pady=(5, 0))

    def set_status(self, message):
        self.root.after(0, lambda msg=message: self.status.set(msg))

    def _auto_find_model(self):
        """Load the CNN model automatically; keep model filenames out of the UI."""
        base = Path(__file__).resolve().parent
        model_file = base / "outputs" / "skin_disnet_final.keras"
        classes_file = base / "outputs" / "class_names.json"

        if not model_file.is_file() or not classes_file.is_file():
            alt_model = base / "skin_disnet_final.keras"
            alt_classes = base / "class_names.json"
            if alt_model.is_file() and alt_classes.is_file():
                model_file, classes_file = alt_model, alt_classes

        if not model_file.is_file():
            self.status.set("AI model file not found. Check the outputs folder.")
            messagebox.showerror(
                "AI model not found",
                f"Expected model file:\\n{model_file}\\n\\n"
                "Place skin_disnet_final.keras in the outputs folder and restart.",
            )
            return

        if not classes_file.is_file():
            self.status.set("Class names file not found.")
            messagebox.showerror(
                "Class names not found",
                f"Expected class_names.json at:\\n{classes_file}",
            )
            return

        self.model_path.set(str(model_file))
        self.classes_path.set(str(classes_file))
        self.status.set("Loading AI model automatically...")
        threading.Thread(
            target=self._load_model_worker,
            args=(str(model_file), str(classes_file)),
            daemon=True,
        ).start()

    def _load_model_worker(self, model_file, classes_file):
        try:
            if tf is None:
                raise RuntimeError(
                    "TensorFlow is not installed. Run: py -3.11 -m pip install tensorflow"
                )

            with open(classes_file, "r", encoding="utf-8") as file:
                names = self.normalize_class_names(json.load(file))

            model = tf.keras.models.load_model(model_file)
            output_count = int(model.output_shape[-1])
            if len(names) != output_count:
                raise ValueError(
                    f"class_names.json has {len(names)} classes but the model "
                    f"has {output_count} outputs. Use the matching JSON."
                )

            self.root.after(0, lambda m=model, n=names: self._model_loaded(m, n))
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
            self.root.after(0, lambda err=detail: self._model_load_failed(err))

    def _model_loaded(self, model, names):
        self.model = model
        self.class_names = names
        self.predict_button.configure(state="normal")
        self.status.set(
            f"AI model ready ({len(names)} classes). Upload a skin image to begin."
        )

    def _model_load_failed(self, error):
        self.status.set("AI model could not be loaded.")
        messagebox.showerror("AI model loading error", error)

    @staticmethod
    def normalize_class_names(names):
        if isinstance(names, list):
            return names
        if isinstance(names, dict):
            try:
                return [names[key] for key in sorted(names, key=lambda x: int(x))]
            except (ValueError, TypeError):
                return list(names.values())
        raise ValueError("class_names.json must contain a JSON list or object.")

    def browse_image(self):
        path = filedialog.askopenfilename(
            title="Select skin image",
            filetypes=[
                ("Images", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        try:
            img = Image.open(path).convert("RGB")
            thumbnail = img.copy()
            thumbnail.thumbnail((320, 360))
            self.preview = ImageTk.PhotoImage(thumbnail)
            self.image_label.configure(image=self.preview, text="")
            self.image_path = path
            self.status.set(f"Selected image: {os.path.basename(path)}")
        except Exception as exc:
            messagebox.showerror("Image error", str(exc))

    def start_prediction(self):
        if self.busy:
            return

        if self.model is None:
            messagebox.showwarning(
                "Model not loaded",
                "Click Load model and wait for the success message first.",
            )
            return

        if not self.image_path:
            messagebox.showwarning("Image required", "Upload a skin image first.")
            return

        self.busy = True
        self.predict_button.configure(state="disabled")
        self.status.set("Starting CNN prediction...")
        threading.Thread(target=self.predict_worker, daemon=True).start()

    def predict_worker(self):
        try:
            self.set_status("Stage 1/3: Running CNN prediction...")

            img = Image.open(self.image_path).convert("RGB").resize(IMG_SIZE)
            array = np.asarray(img, dtype=np.float32)
            array = np.expand_dims(array, axis=0)

            # IMPORTANT: preprocessing must match your model training pipeline.
            # If training images were divided by 255, uncomment this:
            # array = array / 255.0

            scores = np.asarray(self.model.predict(array, verbose=0)[0])
            if scores.ndim != 1 or len(scores) != len(self.class_names):
                raise ValueError(
                    "Unexpected model output shape. Check the saved model and class_names.json."
                )

            idx = int(np.argmax(scores))
            disease = str(self.class_names[idx])
            score = float(scores[idx])

            indicator = (
                "Higher model score" if score >= 0.80 else
                "Moderate model score" if score >= 0.55 else
                "Low model score — uncertain prediction"
            )

            self.root.after(
                0,
                lambda d=disease: self.prediction_label.configure(
                    text=f"CNN predicted class: {d}"
                ),
            )
            self.root.after(
                0,
                lambda s=score: self.score_label.configure(
                    text=f"Raw model score: {s:.4f} (not a calibrated probability)"
                ),
            )

            def progress(message):
                self.set_status(message)

            explanation = generate_genai_explanation(
                disease, score, progress_callback=progress
            )

            self.root.after(
                0,
                lambda d=disease, s=score, ind=indicator, exp=explanation:
                    self.show_results(d, s, ind, exp),
            )

        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
            self.root.after(0, lambda err=detail: self.show_error(err))

    def show_results(self, disease, score, indicator, explanation):
        self.prediction_label.configure(text=f"Predicted disease: {disease}")
        self.score_label.configure(
            text=f"Raw model score: {score:.4f} (not a calibrated probability)"
        )
        self.conf_label.configure(
            text=f"Model-score indicator: {indicator}; not a medical risk level."
        )

        self.output.configure(state="normal")
        self.output.delete("1.0", "end")
        self.output.insert("1.0", explanation)
        self.output.configure(state="disabled")

        self.status.set(
            "Prediction and GenAI explanation complete. This is not a confirmed diagnosis."
        )
        self.busy = False
        self.predict_button.configure(state="normal")

    def show_error(self, error):
        self.status.set("Operation failed. See the error dialog for details.")
        self.busy = False
        if self.model is not None:
            self.predict_button.configure(state="normal")
        messagebox.showerror(
            "Prediction / GenAI error",
            f"{error}\n\n"
            "Check that Ollama is running, both required models are installed, "
            "and medical_knowledge contains .txt files beside app.py.",
        )

    def start_ollama_test(self):
        if self.busy:
            messagebox.showinfo(
                "Busy",
                "Wait for the current operation to finish before running a test.",
            )
            return

        self.busy = True
        self.status.set("Testing Ollama connection, embeddings, and generation...")
        threading.Thread(target=self.ollama_test_worker, daemon=True).start()

    def ollama_test_worker(self):
        try:
            result = run_ollama_test()
            self.root.after(0, lambda text=result: self.show_ollama_test(text))
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
            self.root.after(0, lambda err=detail: self.show_ollama_test_error(err))

    def show_ollama_test(self, result):
        self.busy = False
        self.status.set("Ollama API, embedding, and text generation tests passed.")
        self.output.configure(state="normal")
        self.output.delete("1.0", "end")
        self.output.insert("1.0", result)
        self.output.configure(state="disabled")
        messagebox.showinfo(
            "Ollama test passed",
            "Ollama responded successfully to embedding and generation requests.\n"
            "Next, test the RAG knowledge folder by clicking Predict + Explain.",
        )

    def show_ollama_test_error(self, error):
        self.busy = False
        self.status.set("Ollama test failed.")
        messagebox.showerror(
            "Ollama test failed",
            f"{error}\n\n"
            "Try these commands in PowerShell:\n"
            "ollama list\n"
            "ollama run llama3.2:3b\n"
            "Then type /bye and retry the test.\n\n"
            "For a connection error, confirm that Ollama is running.",
        )

    def open_maps(self):
        location = self.location.get().strip()
        if not location:
            messagebox.showwarning(
                "Location required", "Enter a city, area or postcode."
            )
            return

        url = (
            "https://www.google.com/maps/search/"
            + quote_plus("dermatologist near " + location)
        )
        webbrowser.open(url)


if __name__ == "__main__":
    root = tk.Tk()
    app = SkinDiseaseApp(root)
    root.mainloop()
