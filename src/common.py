"""Shared helpers: data loading and text cleaning."""
import re
import string
from pathlib import Path

import pandas as pd

try:  # NLTK stop-words if available, otherwise fall back to scikit-learn's list
    from nltk.corpus import stopwords
    STOP_WORDS = set(stopwords.words("english"))
except Exception:  # ImportError or missing corpus (LookupError)
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    STOP_WORDS = set(ENGLISH_STOP_WORDS)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "raw"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"

# "WASHINGTON (Reuters) - ..." datelines appear almost only in True.csv and let a
# model recognise the *source* instead of the *content* (label leakage).
_DATELINE = re.compile(r"^\s*[A-Za-z .,/'-]{2,40}\(Reuters\)\s*-?\s*", re.IGNORECASE)
_SOURCE_WORDS = re.compile(r"\breuters\b", re.IGNORECASE)


def load_data(data_dir=DEFAULT_DATA_DIR, drop_duplicates=False) -> pd.DataFrame:
    """Load Fake.csv / True.csv, label them (0 = fake, 1 = real) and shuffle."""
    data_dir = Path(data_dir)
    fake = pd.read_csv(data_dir / "Fake.csv")
    true = pd.read_csv(data_dir / "True.csv")
    fake["label"], true["label"] = 0, 1
    df = pd.concat([fake, true], ignore_index=True)
    df["text"] = df["text"].fillna("")
    if drop_duplicates:
        df = df.drop_duplicates(subset=["title", "text"])
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


def strip_source_leakage(text: str) -> str:
    """Remove the Reuters dateline and any remaining 'Reuters' mentions."""
    return _SOURCE_WORDS.sub(" ", _DATELINE.sub("", text))


def clean_text(text: str) -> str:
    """Lowercase, drop digits and punctuation, remove stop-words."""
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(w for w in text.split() if w not in STOP_WORDS)
