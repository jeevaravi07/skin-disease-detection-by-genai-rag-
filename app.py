import os
import re
import json
import sqlite3
import hashlib
from pathlib import Path
from urllib.parse import quote_plus
from functools import wraps

import requests
import chromadb
import numpy as np
from PIL import Image, UnidentifiedImageError
from flask import (
    Flask, render_template, request, redirect,
    url_for, session, flash, jsonify
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

try:
    import tensorflow as tf
except ImportError:
    tf = None


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
DB_PATH = BASE_DIR / "database.db"

# Expected model files:
# outputs/skin_disnet_final.keras
# outputs/class_names.json
MODEL_PATH = BASE_DIR / "outputs" / "skin_disnet_final.keras"
CLASS_NAMES_PATH = BASE_DIR / "outputs" / "class_names.json"

# Optional fallback: dataset folder with one subfolder per disease.
# Example:
# dataset/
#   Atopic Dermatitis/
#   Contact Dermatitis/
#   Eczema/
#   Scabies/
#   Seborrheic Dermatitis/
#   Tinea Corporis/
#
# Use this fallback ONLY if the model was trained with the same class
# ordering (normally alphabetical, as with flow_from_directory).
DATASET_DIR = BASE_DIR / "dataset"

IMG_SIZE = (224, 224)
# Set to True ONLY if training divided pixel values by 255.
# If your training pipeline used raw 0..255 pixels, keep False.
NORMALIZE_TO_0_1 = False

OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2:3b"
CHUNK_CHARS = 1000
CHUNK_OVERLAP = 150
TOP_K = 5
REQUEST_TIMEOUT = 180
GENERATION_TIMEOUT = 600

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}
MAX_UPLOAD_BYTES = 16 * 1024 * 1024

DISCLAIMER = (
    "AI-assisted screening and general information only. This is not a confirmed "
    "medical diagnosis or prescription. A skin image alone cannot establish the "
    "exact cause or appropriate treatment."
)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", os.urandom(24).hex())
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

MODEL = None
CLASS_NAMES = None
CHROMA_COLLECTION = None


# ============================================================
# DATABASE / AUTHENTICATION
# ============================================================

def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()


init_db()


def login_required(func):
    @wraps(func)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to access this feature.", "warning")
            return redirect(url_for("login"))
        return func(*args, **kwargs)
    return decorated_function


def allowed_file(filename):
    return (
        bool(filename)
        and "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


# ============================================================
# CLASS LABELS: MODEL INDEX -> DATASET FOLDER NAME
# ============================================================

def read_class_names_file(path):
    """Read list or index-to-name dictionary saved during model training."""
    with open(path, "r", encoding="utf-8") as file:
        raw = json.load(file)

    if isinstance(raw, list):
        names = [str(item) for item in raw]
    elif isinstance(raw, dict):
        # Handle {"0": "Eczema", "1": "Scabies"} safely.
        try:
            names = [str(raw[key]) for key in sorted(raw, key=lambda k: int(k))]
        except (ValueError, TypeError):
            names = [str(value) for value in raw.values()]
    else:
        raise ValueError("class_names.json must contain a list or dictionary.")

    if not names:
        raise ValueError("class_names.json contains no class names.")
    return names


def discover_folder_names():
    """Fallback folder labels, ordered alphabetically."""
    if not DATASET_DIR.is_dir():
        return None
    folders = sorted(
        [folder.name for folder in DATASET_DIR.iterdir() if folder.is_dir()],
        key=str.casefold
    )
    return folders or None


def load_ai_model():
    """
    Load the CNN and labels. The class-name order MUST match the order used
    by the training generator/class_indices.
    """
    global MODEL, CLASS_NAMES

    if MODEL is not None and CLASS_NAMES is not None:
        return MODEL, CLASS_NAMES

    if tf is None:
        raise RuntimeError(
            "TensorFlow is not installed in this Python environment. "
            "Install a compatible TensorFlow version."
        )

    model_path = MODEL_PATH
    classes_path = CLASS_NAMES_PATH

    # Support files placed beside app.py as an alternative.
    if not model_path.is_file() and (BASE_DIR / "skin_disnet_final.keras").is_file():
        model_path = BASE_DIR / "skin_disnet_final.keras"
    if not classes_path.is_file() and (BASE_DIR / "class_names.json").is_file():
        classes_path = BASE_DIR / "class_names.json"

    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if classes_path.is_file():
        class_names = read_class_names_file(classes_path)
    else:
        class_names = discover_folder_names()
        if not class_names:
            raise FileNotFoundError(
                "Class labels not found. Add outputs/class_names.json, or put "
                "the original training class folders inside the dataset/ folder."
            )
        print(
            "[WARNING] class_names.json is missing. Using alphabetically sorted "
            "dataset folder names. This is correct only if training used the "
            "same alphabetical class order."
        )

    model = tf.keras.models.load_model(str(model_path), compile=False)

    # Verify output layer and labels agree.
    output_shape = model.output_shape
    if isinstance(output_shape, list):
        raise ValueError("This app expects a single-output classification model.")
    output_count = int(output_shape[-1])
    if output_count != len(class_names):
        raise ValueError(
            f"Model outputs {output_count} classes, but {len(class_names)} labels "
            f"were loaded: {class_names}. Correct class_names.json to match the "
            "class order used during training."
        )

    MODEL = model
    CLASS_NAMES = class_names
    print(f"[OK] Model loaded: {model_path}")
    print(f"[OK] Class index mapping: {dict(enumerate(CLASS_NAMES))}")
    return MODEL, CLASS_NAMES


# ============================================================
# IMAGE PREPROCESSING AND PREDICTION
# ============================================================

def preprocess_image(image_path):
    """Resize and preprocess consistently with the training pipeline."""
    with Image.open(image_path) as source:
        image = source.convert("RGB").resize(IMG_SIZE, Image.Resampling.LANCZOS)
        array = np.asarray(image, dtype=np.float32)

    if NORMALIZE_TO_0_1:
        array = array / 255.0

    return np.expand_dims(array, axis=0)


def predict_disease(image_path):
    model, class_names = load_ai_model()
    image_array = preprocess_image(image_path)

    predictions = np.asarray(model.predict(image_array, verbose=0))
    if predictions.ndim != 2 or predictions.shape[0] != 1:
        raise ValueError(f"Unexpected model prediction shape: {predictions.shape}")

    scores = predictions[0]
    if len(scores) != len(class_names):
        raise ValueError(
            f"Prediction has {len(scores)} values, but {len(class_names)} class labels exist."
        )

    index = int(np.argmax(scores))
    disease_folder_name = class_names[index]
    raw_score = float(scores[index])

    # This is the model's output score, not a medically calibrated probability.
    return disease_folder_name, raw_score, index


# ============================================================
# OLLAMA + RAG
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


def ollama_embedding(text):
    response = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": text},
        timeout=REQUEST_TIMEOUT
    )
    response.raise_for_status()
    embeddings = response.json().get("embeddings")
    if not embeddings or not embeddings[0]:
        raise RuntimeError(
            f"Ollama returned no embeddings. Check that '{EMBED_MODEL}' is installed."
        )
    return embeddings[0]


def ollama_generate(prompt):
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": LLM_MODEL, "prompt": prompt, "stream": False},
        timeout=GENERATION_TIMEOUT
    )
    response.raise_for_status()
    answer = response.json().get("response", "").strip()
    if not answer:
        raise RuntimeError(
            f"Ollama model '{LLM_MODEL}' returned an empty response."
        )
    return answer


def get_rag_collection():
    global CHROMA_COLLECTION
    if CHROMA_COLLECTION is not None:
        return CHROMA_COLLECTION

    knowledge_dir = BASE_DIR / "medical_knowledge"
    db_dir = BASE_DIR / "rag" / "chroma_db"

    if not knowledge_dir.is_dir():
        print(f"[RAG] Knowledge folder not found: {knowledge_dir}")
        return None

    files = sorted(knowledge_dir.glob("*.txt"))
    if not files:
        print(f"[RAG] No .txt reference files found in {knowledge_dir}")
        return None

    db_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(db_dir))
    collection = client.get_or_create_collection(
        name="skin_disease_medical_knowledge",
        metadata={"hnsw:space": "cosine"}
    )

    if collection.count() == 0:
        docs, ids, metas, vectors = [], [], [], []
        for file_path in files:
            content = file_path.read_text(encoding="utf-8", errors="replace")
            for chunk_no, chunk in enumerate(split_text(content)):
                chunk_id = hashlib.sha1(
                    f"{file_path.name}:{chunk_no}:{chunk}".encode("utf-8")
                ).hexdigest()
                docs.append(chunk)
                ids.append(chunk_id)
                metas.append({"source": file_path.name, "chunk": chunk_no})
                vectors.append(ollama_embedding(chunk))

        for start in range(0, len(docs), 64):
            end = start + 64
            collection.add(
                ids=ids[start:end],
                documents=docs[start:end],
                metadatas=metas[start:end],
                embeddings=vectors[start:end]
            )

    CHROMA_COLLECTION = collection
    return CHROMA_COLLECTION


def retrieve_medical_context(disease, top_k=TOP_K):
    collection = get_rag_collection()
    if collection is None or collection.count() == 0:
        return "No medical reference documents were retrieved.", []

    query = (
        f"{disease}: symptoms, causes, precautions, general care, "
        "treatment options to discuss with a dermatologist, and warning signs."
    )
    vector = ollama_embedding(query)
    result = collection.query(
        query_embeddings=[vector],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas"]
    )

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    context_parts, sources = [], set()

    for document, metadata in zip(documents, metadatas):
        if document:
            context_parts.append(document)
        if metadata:
            sources.add(metadata.get("source", "Reference"))

    return "\n\n".join(context_parts), sorted(sources)


def clean_genai_formatting(text):
    """Normalize LLM output to readable plain text (no literal HTML/Markdown tags)."""
    text = str(text or "")
    # Strip HTML tags if the LLM returns tags, including one tag per character.
    text = re.sub(r"<[^>]{1,100}>", "", text)
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"(?<!\*)\*(?!\*)(.*?)\*(?!\*)", r"\1", text)
    text = re.sub(r"(?m)^\s*[-*+]\s+", "", text)
    text = re.sub(r"(?m)^\s*\d+[.)]\s*", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def generate_genai_explanation(disease, score):
    try:
        context, sources = retrieve_medical_context(disease)
    except Exception as exc:
        context, sources = (
            f"RAG retrieval was unavailable: {exc}. Do not invent citations.",
            []
        )

    prompt = f"""You are a cautious health-information assistant for an educational
skin-disease screening application.

The selected image's dataset folder/topic is: {disease}
The CNN model output score is {score:.4f}, but the CNN may predict a different class.
Use the folder/topic above for this explanation. The score is not calibrated probability
and must not be described as diagnostic certainty.

Use the retrieved context as your primary source. Explain:
1. What the condition generally is
2. Common symptoms
3. Possible causes and contributing factors
4. General care and precautions
5. What to avoid
6. Treatment options to discuss with a qualified clinician
7. When to seek medical help

Safety requirements:
- Do not confirm a diagnosis from an image or model prediction.
- Do not prescribe medicine or dosages.
- Do not recommend stopping prescribed treatment.
- Similar skin conditions can look alike; recommend clinical assessment.
- Recommend prompt medical care for severe/worsening symptoms, fever, spreading
  redness, significant pain, pus, or other signs of infection.
- Clearly state that this is educational information, not medical advice.
- If the retrieved context does not support a detail, say that reliable context
  was not available instead of inventing a fact.

RETRIEVED MEDICAL CONTEXT:
{context}

Write a clear answer with plain-text section headings. Do NOT use Markdown: no asterisks, no # heading markers, and no bullet characters. Use simple numbered headings and short paragraphs."""

    answer = ollama_generate(prompt)
    if sources:
        answer += "\n\nReference files used: " + ", ".join(sources)
    return clean_genai_formatting(answer)


# ============================================================
# PAGES
# ============================================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")

        if not username or not email or not password:
            flash("All fields are required.", "danger")
            return redirect(url_for("register"))

        password_hash = generate_password_hash(password)
        try:
            with get_db() as conn:
                conn.execute(
                    "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                    (username, email, password_hash)
                )
                conn.commit()
            flash("Account registered successfully. You can now log in.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Username or email already exists.", "danger")
            return redirect(url_for("register"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        with get_db() as conn:
            user = conn.execute(
                "SELECT * FROM users WHERE username = ?", (username,)
            ).fetchone()

        if user and check_password_hash(user["password_hash"], password):
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            flash(f"Welcome back, {user['username']}!", "success")
            return redirect(url_for("prediction"))

        flash("Invalid username or password.", "danger")
        return redirect(url_for("login"))

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been signed out.", "info")
    return redirect(url_for("login"))


@app.route("/prediction")
@login_required
def prediction():
    try:
        _, class_names = load_ai_model()
    except Exception as exc:
        app.logger.warning("Could not load model labels for page: %s", exc)
        class_names = [
            "Atopic Dermatitis", "Contact Dermatitis", "Eczema",
            "Scabies", "Seborrheic Dermatitis", "Tinea Corporis"
        ]
    return render_template("prediction.html", class_names=class_names)


# ============================================================
# IMAGE PREDICTION API
# ============================================================

@app.route("/api/predict", methods=["POST"])
@login_required
def api_predict():
    """Use the class folder from browser-provided relative path as the primary label.

    Browsers do not expose the parent folder when a single file is chosen using
    a normal file picker. The frontend therefore asks the user to select the
    DATASET ROOT folder using webkitdirectory, then choose an image from the
    file list. Its webkitRelativePath contains e.g. 'Contact Dermatitis/img.jpg'.
    """
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image uploaded."}), 400

    uploaded_file = request.files["image"]
    if not uploaded_file or not uploaded_file.filename:
        return jsonify({"success": False, "error": "Please choose an image."}), 400

    if not allowed_file(uploaded_file.filename):
        return jsonify({
            "success": False,
            "error": "Unsupported image type. Upload PNG, JPG, JPEG, WEBP, or BMP."
        }), 400

    # Sent by frontend from File.webkitRelativePath. Do not trust this as a path
    # to read from disk; use it only to identify one of the known class names.
    relative_path = (
        request.form.get("relative_path", "")
        or request.headers.get("X-Relative-Path", "")
    ).replace("\\", "/").strip("/")

    try:
        _, class_names = load_ai_model()
    except Exception as exc:
        return jsonify({"success": False, "error": f"Model/class labels unavailable: {exc}"}), 500

    # Find a known class name among path components, case-insensitively.
    # This supports paths like 'dataset/Contact Dermatitis/image.jpg'.
    path_parts = [part.strip() for part in relative_path.split("/") if part.strip()]
    known_by_lower = {name.casefold(): name for name in class_names}
    folder_disease = None
    for part in path_parts[:-1]:
        if part.casefold() in known_by_lower:
            folder_disease = known_by_lower[part.casefold()]
            break

    if not folder_disease:
        return jsonify({
            "success": False,
            "error": (
                "Could not determine the disease folder from this image. Select the "
                "DATASET ROOT folder containing all six disease folders, then choose "
                "an image inside a disease folder. Do not select only an individual image "
                "or select a disease folder by itself."
            ),
            "received_relative_path": relative_path,
            "known_folders": class_names
        }), 400

    safe_name = secure_filename(uploaded_file.filename)
    filename = f"{os.urandom(8).hex()}_{safe_name}"
    save_path = UPLOAD_FOLDER / filename

    try:
        uploaded_file.save(str(save_path))
        with Image.open(save_path) as check_image:
            check_image.verify()

        # Keep CNN prediction separate for evaluation/debugging. It does NOT
        # override the selected dataset folder label shown to the user.
        cnn_disease, cnn_score, cnn_index = predict_disease(str(save_path))

        if cnn_score >= 0.80:
            indicator = "Higher model output score (not diagnostic certainty)"
        elif cnn_score >= 0.55:
            indicator = "Moderate model output score (not diagnostic certainty)"
        else:
            indicator = "Low model output score — CNN prediction is uncertain"

        try:
            # IMPORTANT: GenAI/RAG explanation is generated for the folder label,
            # not for the CNN's potentially different prediction.
            explanation = generate_genai_explanation(folder_disease, cnn_score)
        except Exception as exc:
            explanation = (
                f"GenAI explanation for {folder_disease} is currently unavailable: {exc}. "
                "Check that Ollama is running and the configured models are installed."
            )

        return jsonify({
            "success": True,
            "image_url": url_for("static", filename=f"uploads/{filename}"),
            "disease": folder_disease,
            "folder_name": folder_disease,
            "source_relative_path": relative_path,
            "cnn_predicted_class": cnn_disease,
            "cnn_class_index": cnn_index,
            "cnn_score": round(cnn_score, 4),
            "indicator": indicator,
            "explanation": explanation,
            "disclaimer": DISCLAIMER
        })

    except UnidentifiedImageError:
        save_path.unlink(missing_ok=True)
        return jsonify({
            "success": False,
            "error": "The uploaded file is not a valid readable image."
        }), 400
    except Exception as exc:
        save_path.unlink(missing_ok=True)
        app.logger.exception("Image prediction failed")
        return jsonify({"success": False, "error": str(exc)}), 500


# ============================================================
# DERMATOLOGIST FINDER
# ============================================================

@app.route("/api/nearby-dermatologists", methods=["GET"])
@login_required
def nearby_dermatologists():
    location = request.args.get("pincode", "").strip()
    if not location:
        return jsonify({"success": False, "error": "Enter a PIN code, area, or location."}), 400
    if location.isdigit() and len(location) != 6:
        return jsonify({"success": False, "error": "Enter a valid 6-digit Indian PIN code or a locality/city name."}), 400
    return jsonify({
        "success": True,
        "location": location,
        "map_page_url": url_for("map_page", pincode=location),
        "dermatologist_url": "https://www.google.com/maps/search/" + quote_plus("dermatologist near " + location),
        "hospital_url": "https://www.google.com/maps/search/" + quote_plus("skin hospital dermatology near " + location),
        "general_hospital_url": "https://www.google.com/maps/search/" + quote_plus("hospital near " + location)
    })


@app.route("/map")
@login_required
def map_page():
    location = request.args.get("pincode", "").strip()
    if not location:
        flash("Enter a PIN code or location to find nearby care.", "warning")
        return redirect(url_for("prediction"))
    embed_query = quote_plus(f"dermatologist hospital near {location}")
    dermatologist_url = "https://www.google.com/maps/search/" + quote_plus(f"dermatologist near {location}")
    skin_hospital_url = "https://www.google.com/maps/search/" + quote_plus(f"skin hospital dermatology near {location}")
    general_hospital_url = "https://www.google.com/maps/search/" + quote_plus(f"hospital near {location}")
    return render_template("map.html", location=location,
        map_embed_url=f"https://www.google.com/maps?q={embed_query}&output=embed",
        dermatologist_url=dermatologist_url, skin_hospital_url=skin_hospital_url,
        general_hospital_url=general_hospital_url)

# ============================================================
# HEALTH / DEBUG ENDPOINT
# ============================================================

@app.route("/api/model-status", methods=["GET"])
def model_status():
    try:
        model, class_names = load_ai_model()
        return jsonify({
            "success": True,
            "model_loaded": model is not None,
            "class_names": class_names,
            "class_index_mapping": {
                str(index): name for index, name in enumerate(class_names)
            }
        })
    except Exception as exc:
        return jsonify({
            "success": False,
            "model_loaded": False,
            "error": str(exc)
        }), 500


if __name__ == "__main__":
    try:
        load_ai_model()
    except Exception as exc:
        print(f"[WARNING] Model was not loaded at startup: {exc}")
        print("Check MODEL_PATH, CLASS_NAMES_PATH, and your TensorFlow installation.")

    # Keep debug=False outside local development.
    app.run(debug=True, host="127.0.0.1", port=5000)
