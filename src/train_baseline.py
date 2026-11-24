"""Baseline pipeline from the project report: TF-IDF (1-2 grams) + Logistic Regression
on the full merged dataset with an 80/20 split.

Use --remove-leakage to strip the Reuters dateline / source words, and --dedupe to drop
duplicate articles, to measure how much accuracy comes from dataset artefacts rather
than from the writing itself.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from common import DEFAULT_DATA_DIR, DEFAULT_RESULTS_DIR, clean_text, load_data, strip_source_leakage


def main(data_dir, out_dir, remove_leakage, dedupe):
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    df = load_data(data_dir, drop_duplicates=dedupe)
    text = df["text"].apply(strip_source_leakage) if remove_leakage else df["text"]
    df["clean_text"] = text.apply(clean_text)

    X = TfidfVectorizer(max_features=10000, ngram_range=(1, 2)).fit_transform(df["clean_text"])
    y = df["label"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
    pred = model.predict(X_te)
    acc = accuracy_score(y_te, pred)
    print(f"Rows: {len(df)} | train {X_tr.shape[0]} | test {X_te.shape[0]}")
    print(f"Accuracy: {acc:.4f} | Error rate: {1 - acc:.4f}")
    print(classification_report(y_te, pred, target_names=["Fake", "Real"]))

    cm = confusion_matrix(y_te, pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion Matrix")
    tag = ("_no_leakage" if remove_leakage else "") + ("_dedupe" if dedupe else "")
    plt.tight_layout(); plt.savefig(out_dir / "figures" / f"baseline_confusion_matrix{tag}.png", dpi=150); plt.close()

    (out_dir / f"baseline_metrics{tag}.json").write_text(json.dumps({
        "rows": len(df), "remove_leakage": remove_leakage, "deduplicated": dedupe,
        "accuracy": acc, "confusion_matrix": cm.tolist(),
        "report": classification_report(y_te, pred, output_dict=True)}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    p.add_argument("--remove-leakage", action="store_true", help="strip Reuters dateline/source words")
    p.add_argument("--dedupe", action="store_true", help="drop duplicate articles before splitting")
    a = p.parse_args()
    main(a.data_dir, a.out_dir, a.remove_leakage, a.dedupe)
