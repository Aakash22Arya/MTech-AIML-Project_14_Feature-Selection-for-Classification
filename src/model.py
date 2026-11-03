"""
model.py
Classifiers and feature-selection methods.

Every selection method is expressed as a *score per feature* (higher = more
useful). A selector is then just "keep the top-k" (or a natural rule for
Variance Threshold and L1-SVM). Scoring once and slicing for many values of k
makes the k-sweep cheap, even for slow scorers like Mutual Information.
"""
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

SEED = 42

# ---------------------------------------------------------------- classifiers
MODEL_FACTORIES = {
    "LinearSVC": lambda: LinearSVC(C=0.5, random_state=SEED),
    "LogisticRegression": lambda: LogisticRegression(max_iter=300,
                                                     random_state=SEED),
    "NaiveBayes": lambda: MultinomialNB(alpha=0.1),
    "RandomForest": lambda: RandomForestClassifier(
        n_estimators=100, n_jobs=-1, random_state=SEED),
    "kNN": lambda: KNeighborsClassifier(
        n_neighbors=5, metric="cosine", algorithm="brute"),
}
PRIMARY_MODEL = "LinearSVC"          # used for the headline table
DEFAULT_MODELS = ["LinearSVC", "LogisticRegression", "NaiveBayes",
                  "RandomForest"]    # add "kNN" for a fifth model


def make_model(name):
    return MODEL_FACTORIES[name]()


# ------------------------------------------------------ selection methods
VARIANCE = "Variance Threshold"
KBEST_METHODS = ["Chi-square", "ANOVA F-test", "Mutual Information",
                 "L1-SVM", "Random Forest"]
ALL_METHODS = [VARIANCE] + KBEST_METHODS
VAR_THRESHOLD = 1e-5                 # features with variance below are dropped


def _variance(X):
    mean = np.asarray(X.mean(axis=0)).ravel()
    sq_mean = np.asarray(X.multiply(X).mean(axis=0)).ravel()
    return sq_mean - mean ** 2


def score_features(method, X, y, seed=SEED):
    """Return (scores, seconds). Scores are fitted on TRAIN data only."""
    t0 = time.perf_counter()
    if method == VARIANCE:
        s = _variance(X)
    elif method == "Chi-square":
        s = chi2(X, y)[0]
    elif method == "ANOVA F-test":
        s = f_classif(X, y)[0]
    elif method == "Mutual Information":
        # MI between "term appears in document" (0/1) and the class label.
        # This is the classic text-MI and is far faster than the continuous
        # k-NN estimator on tens of thousands of sparse columns.
        Xb = X.tocsc(copy=True)
        Xb.data[:] = 1.0
        s = mutual_info_classif(Xb, y, discrete_features=True,
                                random_state=seed)
    elif method == "L1-SVM":
        svm = LinearSVC(C=0.1, penalty="l1", dual=False, max_iter=2000,
                        random_state=seed).fit(X, y)
        s = np.abs(svm.coef_).max(axis=0)
    elif method == "Random Forest":
        rf = RandomForestClassifier(n_estimators=100, n_jobs=-1,
                                    random_state=seed).fit(X, y)
        s = rf.feature_importances_
    else:
        raise ValueError(f"Unknown method: {method}")
    return np.nan_to_num(np.asarray(s, dtype=float)), time.perf_counter() - t0


def select_indices(method, scores, k, natural=True):
    """Indices of the kept features.
    natural=True : Variance and L1-SVM keep what their own rule keeps
                   (variance > threshold / non-zero weight).
    otherwise    : keep the top-k scoring features."""
    if natural and method == VARIANCE:
        idx = np.where(scores > VAR_THRESHOLD)[0]
    elif natural and method == "L1-SVM":
        idx = np.where(scores > 0)[0]
    else:
        idx = np.sort(np.argsort(-scores, kind="stable")[:k])
    return idx
