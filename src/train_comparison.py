"""Balanced-sample experiment: Logistic Regression vs Random Forest with threshold tuning.

Draws a class-balanced sample (default 5,000 per class), splits 80/20, trains both models
on TF-IDF (1-2 grams), compares them at t=0.5, sweeps the decision threshold for best F1,
and saves metric, confusion-matrix, threshold and feature-importance plots.

NOTE: the best threshold is chosen on the test set here (as in the report's figure), which
is slightly optimistic. For a strict evaluation, choose it on a validation split instead.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from common import DEFAULT_DATA_DIR, DEFAULT_RESULTS_DIR, clean_text, load_data, strip_source_leakage


def metrics(y, proba, t=0.5):
    pred = (proba >= t).astype(int)
    return {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred),
            "recall": recall_score(y, pred), "f1": f1_score(y, pred), "roc_auc": roc_auc_score(y, proba)}


def best_threshold(y, proba):
    ts = np.arange(0.01, 1.0, 0.01)
    f1s = [f1_score(y, (proba >= t).astype(int)) for t in ts]
    return ts, f1s, float(ts[int(np.argmax(f1s))])


def main(data_dir, out_dir, n_per_class, remove_leakage):
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tag = "_no_leakage" if remove_leakage else ""

    df = load_data(data_dir, drop_duplicates=True)
    df = pd.concat([g.sample(n_per_class, random_state=42) for _, g in df.groupby("label")]).reset_index(drop=True)
    raw = df["text"].apply(strip_source_leakage) if remove_leakage else df["text"]
    before = raw.str.split().str.len().mean()
    df["clean_text"] = raw.apply(clean_text)
    after = df["clean_text"].str.split().str.len().mean()
    print(f"Avg words/article: {before:.0f} -> {after:.0f} ({1 - after / before:.1%} reduction)")

    X_tr_txt, X_te_txt, y_tr, y_te = train_test_split(
        df["clean_text"], df["label"], test_size=0.2, stratify=df["label"], random_state=42)
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_tr, X_te = tfidf.fit_transform(X_tr_txt), tfidf.transform(X_te_txt)  # vocabulary fit on train only
    print(f"Train {X_tr.shape[0]} | Test {X_te.shape[0]}")

    models = {"Logistic Regression": LogisticRegression(max_iter=1000),
              "Random Forest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42)}
    probas, results, cms = {}, {}, {}
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        probas[name] = m.predict_proba(X_te)[:, 1]
        results[name] = metrics(y_te, probas[name])
        cms[name] = confusion_matrix(y_te, (probas[name] >= 0.5).astype(int))
        print(name, {k: round(v, 3) for k, v in results[name].items()})

    pd.DataFrame(results).T.plot(kind="bar", rot=0, figsize=(11, 4.5), ylim=(0.5, 1.05))
    plt.title("Model Comparison (threshold = 0.5)"); plt.tight_layout()
    plt.savefig(fig_dir / f"model_comparison{tag}.png", dpi=150); plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (name, cm) in zip(axes, cms.items()):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
        ax.set_title(name); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.tight_layout(); plt.savefig(fig_dir / f"confusion_matrices{tag}.png", dpi=150); plt.close()

    thresholds = {}
    plt.figure(figsize=(10, 4))
    for name, p in probas.items():
        ts, f1s, t_best = best_threshold(y_te, p)
        thresholds[name] = {"threshold": t_best, "f1": max(f1s)}
        plt.plot(ts, f1s, label=name); plt.axvline(t_best, ls="--", alpha=0.5)
    plt.xlabel("Classification threshold"); plt.ylabel("F1"); plt.legend(); plt.title("Threshold vs F1")
    plt.tight_layout(); plt.savefig(fig_dir / f"threshold_vs_f1{tag}.png", dpi=150); plt.close()
    print("Best thresholds:", thresholds)

    feats = np.array(tfidf.get_feature_names_out())
    coef = models["Logistic Regression"].coef_[0]          # positive = REAL (label 1), negative = FAKE (label 0)
    order = np.argsort(coef)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    axes[0].barh(feats[order[:20]][::-1], -coef[order[:20]][::-1], color="#e63946"); axes[0].set_title("Pushes toward FAKE (label 0)")
    axes[1].barh(feats[order[-20:]], coef[order[-20:]], color="#2a9d8f"); axes[1].set_title("Pushes toward REAL (label 1)")
    plt.tight_layout(); plt.savefig(fig_dir / f"lr_top_words{tag}.png", dpi=150); plt.close()

    imp = models["Random Forest"].feature_importances_
    top = np.argsort(imp)[-20:]
    plt.figure(figsize=(8, 6)); plt.barh(feats[top], imp[top], color="#e76f51")
    plt.title("Top 20 Random Forest Feature Importances")
    plt.tight_layout(); plt.savefig(fig_dir / f"rf_feature_importance{tag}.png", dpi=150); plt.close()
    print("LR top words -> FAKE:", list(feats[order[:10]]))
    print("LR top words -> REAL:", list(feats[order[-10:]][::-1]))

    (out_dir / f"comparison_metrics{tag}.json").write_text(json.dumps(
        {"metrics_at_0.5": results, "best_thresholds": thresholds,
         "confusion_matrices": {k: v.tolist() for k, v in cms.items()}}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    p.add_argument("--n-per-class", type=int, default=5000)
    p.add_argument("--remove-leakage", action="store_true", help="strip Reuters dateline/source words")
    a = p.parse_args()
    main(a.data_dir, a.out_dir, a.n_per_class, a.remove_leakage)
