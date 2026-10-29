"""
SkinDisNet training script with data augmentation.
Expected beside this file:
  images/
    Atopic Dermatitis/
    Contact Dermatitis/
    Eczema/
    Scabies/
    Seborrheic Dermatitis/
    Tinea Corporis/
  balanced_metadata.csv

Run: python train.py
Outputs are saved to ./outputs/
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import json
import random
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_fscore_support, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize

# ---------------- SETTINGS ----------------
ROOT = Path(__file__).resolve().parent
IMAGE_DIR = ROOT / "images"
CSV_PATH = ROOT / "balanced_metadata.csv"
OUT = ROOT / "outputs"
MODEL_PATH = OUT / "skin_disnet_efficientnetb0_augmented.keras"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
HEAD_EPOCHS = 12
FINE_TUNE_EPOCHS = 8
SEED = 42
USE_IMAGENET_WEIGHTS = True
AUTOTUNE = tf.data.AUTOTUNE

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
OUT.mkdir(exist_ok=True)

def find_col(df, candidates):
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lookup:
            return lookup[c.lower()]
    return None

def scan_images():
    if not IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Cannot find {IMAGE_DIR}. Save train.py beside your images folder."
        )
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    rows = []
    for folder in sorted(p for p in IMAGE_DIR.iterdir() if p.is_dir()):
        for p in sorted(folder.rglob("*")):
            if p.is_file() and p.suffix.lower() in exts:
                rows.append({"path": str(p), "label": folder.name, "stem": p.stem})
    if not rows:
        raise RuntimeError("No images found. Expected images/<disease class>/<image file>.")
    return rows

def match_patients(rows):
    # Group images by Patient_id when metadata filenames match.
    for r in rows:
        r["patient_id"] = "file_" + r["stem"]
    if not CSV_PATH.exists():
        warnings.warn(f"{CSV_PATH.name} not found. Patient-level leakage cannot be checked.")
        return rows

    df = pd.read_csv(CSV_PATH)
    image_col = find_col(df, ["Image_id", "image_id", "filename", "file_name", "image"])
    patient_col = find_col(df, ["Patient_id", "patient_id", "patient", "patientid"])
    folder_col = find_col(df, ["Folder_name", "folder_name", "folder"])
    if image_col is None or patient_col is None:
        warnings.warn("Could not identify Image_id and Patient_id columns in CSV.")
        return rows

    by_key = {}
    by_stem = {}
    for _, row in df.iterrows():
        stem = Path(str(row[image_col]).strip()).stem.casefold()
        patient = str(row[patient_col]).strip()
        if not patient or patient.lower() in {"nan", "none"}:
            patient = "missing_" + stem
        by_stem.setdefault(stem, patient)
        if folder_col is not None:
            by_key[(str(row[folder_col]).strip().casefold(), stem)] = patient

    matched = 0
    for r in rows:
        stem = Path(r["path"]).stem.casefold()
        folder = Path(r["path"]).parent.name.casefold()
        patient = by_key.get((folder, stem), by_stem.get(stem))
        if patient is not None:
            r["patient_id"] = patient
            matched += 1
    print(f"Metadata patient matches: {matched}/{len(rows)}")
    if matched < len(rows) * 0.95:
        warnings.warn("Less than 95% of image files matched metadata. Check CSV image IDs.")
    return rows

def split_data(rows):
    y = np.array([r["label"] for r in rows])
    groups = np.array([r["patient_id"] for r in rows])
    try:
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        trainval, test = next(splitter.split(np.zeros(len(y)), y, groups))
        splitter2 = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED + 1)
        train_rel, val_rel = next(
            splitter2.split(np.zeros(len(trainval)), y[trainval], groups[trainval])
        )
        train = trainval[train_rel]
        val = trainval[val_rel]
        if set(groups[train]) & set(groups[val]) or set(groups[train]) & set(groups[test]) or set(groups[val]) & set(groups[test]):
            raise ValueError("Patient overlap after split")
        return train, val, test, "patient-level stratified group split"
    except Exception as e:
        warnings.warn(f"Patient-level split failed: {e}. Using stratified image split.")
        idx = np.arange(len(rows))
        trainval, test = train_test_split(idx, test_size=0.15, random_state=SEED, stratify=y)
        train, val = train_test_split(
            trainval, test_size=0.1765, random_state=SEED, stratify=y[trainval]
        )
        return train, val, test, "image-level stratified fallback; patient leakage may remain"

def read_image(path, label):
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, IMG_SIZE)
    return tf.cast(img, tf.float32), label

def make_ds(rows, class_map, training=False):
    paths = [r["path"] for r in rows]
    labels = [class_map[r["label"]] for r in rows]
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(read_image, num_parallel_calls=AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

def build_model(n_classes):
    """
    EfficientNetB0 CNN with on-the-fly augmentation.
    Augmentation layers run only during training; validation/test images remain unchanged.
    """
    inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name="image")

    x = layers.RandomFlip("horizontal", seed=42)(inputs)
    x = layers.RandomRotation(
        factor=0.05, fill_mode="reflect", seed=43
    )(x)
    x = layers.RandomTranslation(
        height_factor=0.05, width_factor=0.05,
        fill_mode="reflect", seed=44
    )(x)
    x = layers.RandomZoom(
        height_factor=(-0.10, 0.10),
        width_factor=(-0.10, 0.10),
        fill_mode="reflect", seed=45
    )(x)
    x = layers.RandomContrast(factor=0.10, seed=46)(x)

    backbone = keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet" if USE_IMAGENET_WEIGHTS else None,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    backbone.trainable = False

    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.40)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.30)(x)
    outputs = layers.Dense(
        n_classes, activation="softmax", name="class_probabilities"
    )(x)

    model = keras.Model(inputs, outputs, name="SkinDisNet_EfficientNetB0_Augmented")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model, backbone

def plot_training(history, out):
    epochs = range(1, len(history["loss"]) + 1)
    for metric, title, filename in [
        ("accuracy", "Training and Validation Accuracy", "training_validation_accuracy.png"),
        ("loss", "Training and Validation Loss", "training_validation_loss.png")
    ]:
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, history[metric], label="Training " + metric)
        plt.plot(epochs, history["val_" + metric], label="Validation " + metric)
        plt.xlabel("Epoch")
        plt.ylabel(metric.title())
        plt.title(title)
        plt.legend()
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(out / filename, dpi=180)
        plt.close()

def main():
    started = time.time()
    rows = match_patients(scan_images())
    classes = sorted({r["label"] for r in rows})
    class_map = {name: i for i, name in enumerate(classes)}
    id_map = {str(i): name for name, i in class_map.items()}
    print(f"Found {len(rows)} images in {len(classes)} classes: {classes}")
    print("Outputs:", OUT)

    counts = pd.Series([r["label"] for r in rows]).value_counts().reindex(classes, fill_value=0)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=counts.index, y=counts.values)
    plt.title("Dataset Class Distribution")
    plt.xlabel("Disease class"); plt.ylabel("Images")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout(); plt.savefig(OUT / "class_distribution.png", dpi=180); plt.close()

    train_i, val_i, test_i, split_method = split_data(rows)
    train_rows = [rows[i] for i in train_i]
    val_rows = [rows[i] for i in val_i]
    test_rows = [rows[i] for i in test_i]
    for name, subset in [("train", train_rows), ("validation", val_rows), ("test", test_rows)]:
        print(f"{name}: {len(subset)} images; {pd.Series([r['label'] for r in subset]).value_counts().to_dict()}")

    groups = {s: set(r["patient_id"] for r in rs) for s, rs in [
        ("train", train_rows), ("validation", val_rows), ("test", test_rows)
    ]}
    no_overlap = not (groups["train"] & groups["validation"] or groups["train"] & groups["test"] or groups["validation"] & groups["test"])
    print("Patient groups do not overlap:", no_overlap)
    split_rows = []
    for split_name, subset in [("train", train_rows), ("validation", val_rows), ("test", test_rows)]:
        for r in subset:
            split_rows.append({
                "image_path": os.path.relpath(r["path"], ROOT),
                "label": r["label"],
                "patient_id": r["patient_id"],
                "split": split_name,
            })
    pd.DataFrame(split_rows).to_csv(OUT / "split_manifest.csv", index=False)

    split_counts = pd.DataFrame([
        {"split": split, "label": r["label"]}
        for split, subset in [("train", train_rows), ("validation", val_rows), ("test", test_rows)]
        for r in subset
    ]).groupby(["split", "label"]).size().unstack(fill_value=0)
    split_counts.plot(kind="bar", figsize=(10, 5))
    plt.title("Class Distribution by Split"); plt.xlabel("Split"); plt.ylabel("Images")
    plt.xticks(rotation=0); plt.tight_layout()
    plt.savefig(OUT / "split_class_distribution.png", dpi=180); plt.close()

    train_ds = make_ds(train_rows, class_map, training=True)
    val_ds = make_ds(val_rows, class_map)
    test_ds = make_ds(test_rows, class_map)

    model, backbone = build_model(len(classes))
    callbacks = [
        keras.callbacks.ModelCheckpoint(str(MODEL_PATH), monitor="val_loss", save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7, verbose=1),
        keras.callbacks.CSVLogger(str(OUT / "training_log.csv"))
    ]
    print("\nStage 1: train classification head")
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=12, callbacks=callbacks)
    print("\nStage 2: fine-tune final backbone layers")
    backbone.trainable = True
    for layer in backbone.layers[:-30]:
        layer.trainable = False
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-6),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=8, callbacks=callbacks)
    history = {k: h1.history.get(k, []) + h2.history.get(k, []) for k in set(h1.history) | set(h2.history)}
    plot_training(history, OUT)

    best = keras.models.load_model(MODEL_PATH)
    test_loss, test_acc = best.evaluate(test_ds, verbose=0)
    y_true = np.concatenate([y.numpy() for _, y in test_ds])
    probs = best.predict(test_ds, verbose=1)
    y_pred = np.argmax(probs, axis=1)

    report = classification_report(y_true, y_pred, labels=list(range(len(classes))),
                                   target_names=classes, output_dict=True, zero_division=0)
    (OUT / "classification_report.txt").write_text(
        classification_report(y_true, y_pred, labels=list(range(len(classes))),
                              target_names=classes, zero_division=0), encoding="utf-8")
    pd.DataFrame(report).transpose().to_csv(OUT / "classification_report.csv")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title("Confusion Matrix — Test Set"); plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.xticks(rotation=35, ha="right"); plt.yticks(rotation=0)
    plt.tight_layout(); plt.savefig(OUT / "confusion_matrix.png", dpi=180); plt.close()

    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=list(range(len(classes))), zero_division=0)
    metric_df = pd.DataFrame({"class": classes, "precision": p, "recall": r, "f1_score": f1})
    metric_df.to_csv(OUT / "per_class_metrics.csv", index=False)
    ax = metric_df.set_index("class").plot(kind="bar", figsize=(10, 5), ylim=(0, 1))
    ax.set_title("Per-Class Precision, Recall and F1"); ax.set_ylabel("Score")
    plt.xticks(rotation=30, ha="right"); plt.tight_layout()
    plt.savefig(OUT / "per_class_metrics.png", dpi=180); plt.close()

    y_bin = label_binarize(y_true, classes=list(range(len(classes))))
    plt.figure(figsize=(8, 6))
    roc_auc = {}
    for i, cls in enumerate(classes):
        if len(np.unique(y_bin[:, i])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        score = auc(fpr, tpr); roc_auc[cls] = float(score)
        plt.plot(fpr, tpr, label=f"{cls} (AUC {score:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.title("One-vs-Rest ROC Curves"); plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(OUT / "roc_curves.png", dpi=180); plt.close()

    plt.figure(figsize=(8, 6))
    ap_scores = {}
    for i, cls in enumerate(classes):
        if int(y_bin[:, i].sum()) == 0:
            continue
        prec, rec, _ = precision_recall_curve(y_bin[:, i], probs[:, i])
        ap = average_precision_score(y_bin[:, i], probs[:, i]); ap_scores[cls] = float(ap)
        plt.plot(rec, prec, label=f"{cls} (AP {ap:.2f})")
    plt.title("One-vs-Rest Precision-Recall Curves"); plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig(OUT / "precision_recall_curves.png", dpi=180); plt.close()

    confidence = probs.max(axis=1)
    correct = y_true == y_pred
    plt.figure(figsize=(8, 5))
    plt.hist(confidence[correct], bins=10, alpha=.7, label="Correct")
    plt.hist(confidence[~correct], bins=10, alpha=.7, label="Incorrect")
    plt.title("Prediction Score Distribution"); plt.xlabel("Maximum softmax score"); plt.ylabel("Images")
    plt.legend(); plt.tight_layout()
    plt.savefig(OUT / "prediction_score_distribution.png", dpi=180); plt.close()

    pred_rows = []
    for row, yt, yp, score in zip(test_rows, y_true, y_pred, probs):
        pred_rows.append({
            "image_path": row["path"], "patient_id": row["patient_id"],
            "actual": classes[int(yt)], "predicted": classes[int(yp)],
            "correct": bool(yt == yp), "max_softmax_score": float(score.max()),
            **{f"score_{classes[j]}": float(score[j]) for j in range(len(classes))}
        })
    pd.DataFrame(pred_rows).to_csv(OUT / "test_predictions.csv", index=False)
    (OUT / "class_names.json").write_text(json.dumps({
        "class_to_id": class_map, "id_to_class": id_map,
        "img_size": list(IMG_SIZE),
        "warning": "Softmax scores are not calibrated probabilities of disease."
    }, indent=2), encoding="utf-8")

    summary = {
        "image_count": len(rows), "classes": classes, "split_method": split_method,
        "patient_overlap_free": no_overlap, "train_count": len(train_rows),
        "validation_count": len(val_rows), "test_count": len(test_rows),
        "test_accuracy": float(test_acc), "test_loss": float(test_loss),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "roc_auc_ovr": roc_auc, "average_precision_ovr": ap_scores,
        "model_path": str(MODEL_PATH), "elapsed_seconds": round(time.time() - started, 2),
        "note": "Research/demo only; not a confirmed diagnosis."
    }
    (OUT / "evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nTRAINING FINISHED")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Macro F1: {report['macro avg']['f1-score']:.4f}")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Graphs and reports: {OUT}")
    print("Do not interpret model scores as a confirmed diagnosis.")

if __name__ == "__main__":
    main()
