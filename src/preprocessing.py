"""
preprocessing.py
Data loading and TF-IDF feature extraction.

Rule: everything (TF-IDF vocabulary, IDF weights) is fitted on the TRAIN
split only and then applied to the test split, so there is no data leakage.
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

SEED = 42
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SAMPLE_PATH = os.path.join(ROOT, "data", "sample_data.csv")


# ------------------------------------------------------------------ loaders
def load_20newsgroups(seed=SEED):
    """Official train/test split. Headers, footers and quoted replies are
    removed so the classifier must learn from the message body only."""
    from sklearn.datasets import fetch_20newsgroups
    kw = dict(remove=("headers", "footers", "quotes"), random_state=seed)
    tr = fetch_20newsgroups(subset="train", **kw)
    te = fetch_20newsgroups(subset="test", **kw)
    return tr.data, te.data, tr.target, te.target, list(tr.target_names)


def load_sample(path=SAMPLE_PATH, test_size=0.25, seed=SEED):
    """Small CSV (columns: text,label) used for quick smoke tests."""
    df = pd.read_csv(path).dropna(subset=["text", "label"])
    names = sorted(df["label"].unique())
    y = df["label"].map({n: i for i, n in enumerate(names)}).to_numpy()
    X_tr, X_te, y_tr, y_te = train_test_split(
        df["text"].tolist(), y, test_size=test_size, stratify=y,
        random_state=seed)
    return X_tr, X_te, y_tr, y_te, names


# ------------------------------------------------------------------- TF-IDF
def build_tfidf(train_texts, test_texts, min_df=2):
    vec = TfidfVectorizer(stop_words="english", min_df=min_df,
                          sublinear_tf=True)
    X_train = vec.fit_transform(train_texts)      # fit on TRAIN only
    X_test = vec.transform(test_texts)
    return X_train.tocsr(), X_test.tocsr(), vec.get_feature_names_out()


def get_data(source="20newsgroups", min_df=2):
    """Return a dict with everything the experiments need."""
    if source == "20newsgroups":
        tr_txt, te_txt, y_tr, y_te, names = load_20newsgroups()
    elif source == "sample":
        tr_txt, te_txt, y_tr, y_te, names = load_sample()
        min_df = 1
    else:
        raise ValueError(f"Unknown data source: {source}")
    X_tr, X_te, feats = build_tfidf(tr_txt, te_txt, min_df=min_df)
    return {
        "X_train": X_tr, "X_test": X_te,
        "y_train": np.asarray(y_tr), "y_test": np.asarray(y_te),
        "feature_names": np.asarray(feats), "class_names": names,
        "source": source,
    }


def describe(data):
    """Basic dataset facts for the README / slides."""
    X = data["X_train"]
    return {
        "train_samples": X.shape[0],
        "test_samples": data["X_test"].shape[0],
        "features": X.shape[1],
        "classes": len(data["class_names"]),
        "density_%": round(100 * X.nnz / (X.shape[0] * X.shape[1]), 4),
        "avg_nonzero_per_doc": round(X.nnz / X.shape[0], 1),
    }


# --------------------------------------------------- sample file generation
def make_sample(n_per_class=10, path=SAMPLE_PATH, seed=SEED):
    """Write a small REAL subset of 20 Newsgroups to data/sample_data.csv."""
    from sklearn.datasets import fetch_20newsgroups
    d = fetch_20newsgroups(subset="train", random_state=seed,
                           remove=("headers", "footers", "quotes"))
    rng = np.random.RandomState(seed)
    rows = []
    for c, name in enumerate(d.target_names):
        pool = [t for t, y in zip(d.data, d.target)
                if y == c and len(t.split()) >= 20]
        for i in rng.choice(len(pool), size=n_per_class, replace=False):
            rows.append({"text": " ".join(pool[i].split()), "label": name})
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Wrote {len(rows)} real posts to {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-sample", action="store_true",
                    help="replace data/sample_data.csv with real 20NG posts")
    ap.add_argument("--n", type=int, default=10, help="posts per class")
    a = ap.parse_args()
    if a.make_sample:
        make_sample(a.n)
    else:
        print(describe(get_data("20newsgroups")))
