"""Exploratory data analysis: class balance, text length, most frequent words."""
import argparse
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import DEFAULT_DATA_DIR, DEFAULT_RESULTS_DIR, clean_text, load_data


def main(data_dir, out_dir):
    out = out_dir / "figures"
    out.mkdir(parents=True, exist_ok=True)
    df = load_data(data_dir)
    df["text_length"] = df["text"].str.len()
    df["clean_text"] = df["text"].apply(clean_text)
    names = {0: "Fake", 1: "Real"}

    print(df["label"].map(names).value_counts().to_string())
    print("\nMean text length (characters):")
    print(df.groupby("label")["text_length"].mean().rename(names).round(1).to_string())
    print("\nArticles per subject:")
    print(df.groupby(["label", "subject"]).size().to_string())

    ax = df["label"].map(names).value_counts().plot(kind="bar", color=["#2a9d8f", "#e63946"], rot=0)
    ax.set_title("Fake vs Real News Distribution")
    ax.set_ylabel("count")
    plt.tight_layout(); plt.savefig(out / "eda_class_distribution.png", dpi=150); plt.close()

    for lab in (0, 1):
        df[df.label == lab]["text_length"].clip(upper=20000).plot(kind="hist", bins=50, alpha=0.6, label=names[lab])
    plt.legend(); plt.xlabel("text length (characters, clipped at 20,000)"); plt.title("Text Length Distribution")
    plt.tight_layout(); plt.savefig(out / "eda_text_length.png", dpi=150); plt.close()

    for lab in (0, 1):
        words, freqs = zip(*Counter(" ".join(df[df.label == lab]["clean_text"]).split()).most_common(15))
        plt.barh(words[::-1], freqs[::-1], color="#e63946" if lab == 0 else "#2a9d8f")
        plt.title(f"Top Words in {names[lab]} News"); plt.xlabel("count")
        plt.tight_layout(); plt.savefig(out / f"eda_top_words_{names[lab].lower()}.png", dpi=150); plt.close()
    print(f"\nFigures saved to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    a = p.parse_args()
    main(a.data_dir, a.out_dir)
