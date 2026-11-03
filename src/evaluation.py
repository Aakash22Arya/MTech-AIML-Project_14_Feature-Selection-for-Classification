"""
evaluation.py
Metrics, timing, experiment runners, result tables and figures.

Run everything from the command line:
    python -m src.evaluation --data 20newsgroups
    python -m src.evaluation --data 20newsgroups --fast
    python -m src.evaluation --data sample          (tiny smoke test)

Metrics: Number of Features, Accuracy, Macro-F1, Training Time, Inference Time
Verdict rule ("without sacrificing performance"):
    Accuracy and Macro-F1 each drop by <= TOL_PTS points vs. all features
    AND training time is at least MIN_TRAIN_SPEEDUP times faster.
"""
import argparse
import os
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from .model import (ALL_METHODS, DEFAULT_MODELS, KBEST_METHODS,
                    PRIMARY_MODEL, VAR_THRESHOLD, VARIANCE, make_model,
                    score_features, select_indices)
from .preprocessing import describe, get_data

BASELINE = "All features (baseline)"
TOL_PTS = 1.0                 # allowed drop in accuracy / Macro-F1 (points)
MIN_TRAIN_SPEEDUP = 1.10      # training must be at least 10% faster
REPEAT_CAPS = {"RandomForest": 2, "LogisticRegression": 3}   # slow models
DEFAULT_KS = [100, 250, 500, 1000, 3000, 5000, 10000]
# step-down ladder: % of the original features kept after "all features"
DEFAULT_LEVELS = [75, 50, 25, 10, 5, 2, 1]
SHORT = {BASELINE: "All", VARIANCE: "Variance", "Chi-square": "Chi²",
         "ANOVA F-test": "ANOVA", "Mutual Information": "MI",
         "L1-SVM": "L1-SVM", "Random Forest": "RF"}
INF_COL = "Inference Time (ms/1000 samples)"


# =================================================================== core
def evaluate_features(model_name, Xtr, Xte, ytr, yte, repeats=5):
    """Train + predict with one model; time both (median of repeats)."""
    repeats = max(1, min(repeats, REPEAT_CAPS.get(model_name, repeats)))
    train_t, model = [], None
    for _ in range(repeats):
        model = make_model(model_name)
        t0 = time.perf_counter()
        model.fit(Xtr, ytr)
        train_t.append(time.perf_counter() - t0)
    infer_t, pred = [], None
    for _ in range(repeats):
        t0 = time.perf_counter()
        pred = model.predict(Xte)
        infer_t.append(time.perf_counter() - t0)
    return {
        "Accuracy": accuracy_score(yte, pred),
        "Macro-F1": f1_score(yte, pred, average="macro"),
        "Train Time (s)": float(np.median(train_t)),
        INF_COL: float(np.median(infer_t)) * 1e6 / Xte.shape[0],
    }


def _setup_label(method, k, idx):
    if method == VARIANCE:
        return f"{method} (var > {VAR_THRESHOLD:g})"
    if method == "L1-SVM":
        return f"{method} (non-zero weights)"
    return f"{method} (k={k})"


def run_main_comparison(data, models=None, k=3000, repeats=5,
                        methods=None, verbose=True):
    """All features vs. each selection method, for every model.
    Returns (DataFrame, scores dict, indices dict)."""
    models = models or DEFAULT_MODELS
    methods = methods or ALL_METHODS
    Xtr, Xte = data["X_train"], data["X_test"]
    ytr, yte = data["y_train"], data["y_test"]
    k = min(k, Xtr.shape[1] - 1)

    scores, indices, sel_time = {}, {}, {}
    for m in methods:
        if verbose:
            print(f"[score] {m} ...", flush=True)
        scores[m], sel_time[m] = score_features(m, Xtr, ytr)
        indices[m] = select_indices(m, scores[m], k)

    setups = [(BASELINE, BASELINE, None, 0.0, Xtr.shape[1])]
    for m in methods:
        if len(indices[m]) == 0:
            print(f"  [skip] {m}: kept 0 features")
            continue
        setups.append((_setup_label(m, k, indices[m]), m, indices[m],
                       sel_time[m], len(indices[m])))

    rows = []
    for label, method, idx, t_sel, n_feat in setups:
        Xa = Xtr if idx is None else Xtr[:, idx]
        Xb = Xte if idx is None else Xte[:, idx]
        for model in models:
            if verbose:
                print(f"[eval] {label:<34} | {model}", flush=True)
            r = evaluate_features(model, Xa, Xb, ytr, yte, repeats)
            rows.append({"Experiment": "main", "Model": model,
                         "Method": method, "Setup": label, "k": n_feat,
                         "Features": n_feat, "Selection Time (s)": t_sel,
                         **r})
    return pd.DataFrame(rows), scores, indices


def run_k_sweep(data, scores, models=None, ks=None, methods=None, repeats=2,
                verbose=True):
    """Performance vs. number of kept features (top-k by score)."""
    models = models or DEFAULT_MODELS
    methods = [m for m in (methods or KBEST_METHODS) if m in scores]
    Xtr, Xte = data["X_train"], data["X_test"]
    ytr, yte = data["y_train"], data["y_test"]
    ks = [k for k in (ks or DEFAULT_KS) if k < Xtr.shape[1]]
    rows = []
    for m in methods:
        for k in ks:
            idx = select_indices(m, scores[m], k, natural=False)
            Xa, Xb = Xtr[:, idx], Xte[:, idx]
            for model in models:
                if verbose:
                    print(f"[sweep] {m:<18} k={k:<6} | {model}", flush=True)
                r = evaluate_features(model, Xa, Xb, ytr, yte, repeats)
                rows.append({"Experiment": "sweep", "Model": model,
                             "Method": m, "Setup": f"{m} (k={k})", "k": k,
                             "Features": len(idx), "Selection Time (s)": np.nan,
                             **r})
    return pd.DataFrame(rows)


def run_reduction_ladder(data, scores, main_df, models=None, levels=None,
                         methods=None, repeats=2, verbose=True):
    """Step-down comparison: ALL features first, then fewer and fewer.
    levels = % of the original features kept (e.g. 75, 50, 25, 10, 5, 2, 1).
    Re-uses the baseline from main_df and the scores already computed."""
    models = models or DEFAULT_MODELS
    n = data["X_train"].shape[1]
    levels = sorted({float(l) for l in (levels or DEFAULT_LEVELS) if l < 100},
                    reverse=True)
    kmap = {max(1, int(round(n * l / 100))): l for l in levels}
    ks = sorted(kmap, reverse=True)
    lad = run_k_sweep(data, scores, models, ks, methods, repeats, verbose)
    lad["Experiment"] = "ladder"
    lad["Level (%)"] = lad["k"].map(kmap)          # clean nominal level
    base = main_df[(main_df["Method"] == BASELINE)
                   & main_df["Model"].isin(models)].copy()
    base["Experiment"] = "ladder"
    base["Level (%)"] = 100.0
    return pd.concat([base, lad], ignore_index=True)


# ================================================================= tables
def add_deltas(df):
    """Compare every row with the same model's all-features baseline."""
    df = df.copy()
    new = ["Feature Reduction (%)", "Acc Change (pts)", "F1 Change (pts)",
           "Train Speedup (x)", "Inference Speedup (x)",
           "Total Speedup incl. selection (x)"]
    for c in new:
        df[c] = np.nan
    df["Verdict"] = ""
    for model in df["Model"].unique():
        m = df["Model"] == model
        b = df[m & (df["Method"] == BASELINE)].iloc[0]
        d = df[m]
        df.loc[m, "Feature Reduction (%)"] = 100 * (1 - d["Features"] / b["Features"])
        df.loc[m, "Acc Change (pts)"] = 100 * (d["Accuracy"] - b["Accuracy"])
        df.loc[m, "F1 Change (pts)"] = 100 * (d["Macro-F1"] - b["Macro-F1"])
        df.loc[m, "Train Speedup (x)"] = b["Train Time (s)"] / d["Train Time (s)"]
        df.loc[m, "Inference Speedup (x)"] = b[INF_COL] / d[INF_COL]
        df.loc[m, "Total Speedup incl. selection (x)"] = (
            b["Train Time (s)"] / (d["Selection Time (s)"].fillna(0)
                                   + d["Train Time (s)"]))
        d = df[m]
        ok = ((d["Acc Change (pts)"] >= -TOL_PTS)
              & (d["F1 Change (pts)"] >= -TOL_PTS)
              & (d["Train Speedup (x)"] >= MIN_TRAIN_SPEEDUP))
        df.loc[m, "Verdict"] = np.where(ok, "Yes", "No")
        df.loc[m & (df["Method"] == BASELINE), "Verdict"] = "-"
    return df


def headline_table(df, model=PRIMARY_MODEL):
    """The table the assignment asks for: All vs Selected, five metrics."""
    g = df[(df["Experiment"] == "main") & (df["Model"] == model)]
    t = pd.DataFrame({
        "Setup": g["Setup"].values,
        "Number of Features": g["Features"].values,
        "Accuracy": g["Accuracy"].values,
        "Macro-F1": g["Macro-F1"].values,
        "Training Time (s)": g["Train Time (s)"].values,
        INF_COL: g[INF_COL].values,
        "Selection Time (s)": g["Selection Time (s)"].values,
        "Feature Reduction (%)": g["Feature Reduction (%)"].values,
        "Acc Change (pts)": g["Acc Change (pts)"].values,
        "F1 Change (pts)": g["F1 Change (pts)"].values,
        "Train Speedup (x)": g["Train Speedup (x)"].values,
        "Inference Speedup (x)": g["Inference Speedup (x)"].values,
        "Verdict": g["Verdict"].values,
    })
    return t.round(4)


def ladder_table(df, model=PRIMARY_MODEL, method="Chi-square"):
    """Step 0 = all features, then each reduction level for one method."""
    g = df[(df["Experiment"] == "ladder") & (df["Model"] == model)
           & df["Method"].isin([BASELINE, method])]
    g = g.sort_values("Features", ascending=False)
    return pd.DataFrame({
        "Step": range(len(g)), "Level (%)": g["Level (%)"].values,
        "Number of Features": g["Features"].values,
        "Accuracy": g["Accuracy"].values, "Macro-F1": g["Macro-F1"].values,
        "Training Time (s)": g["Train Time (s)"].values,
        INF_COL: g[INF_COL].values,
        "Acc Change (pts)": g["Acc Change (pts)"].values,
        "F1 Change (pts)": g["F1 Change (pts)"].values,
        "Train Speedup (x)": g["Train Speedup (x)"].values,
        "Inference Speedup (x)": g["Inference Speedup (x)"].values,
        "Verdict": g["Verdict"].values}).round(4)


def ladder_pivot(df, model=PRIMARY_MODEL, value="Macro-F1"):
    """Methods x levels table, columns ordered 100% (all) -> smallest."""
    g = df[(df["Experiment"] == "ladder") & (df["Model"] == model)]
    base = g[g["Method"] == BASELINE][value].iloc[0]
    p = g[g["Method"] != BASELINE].pivot_table(
        index="Method", columns="Level (%)", values=value)
    p = p[sorted(p.columns, reverse=True)]
    p.insert(0, 100.0, base)
    p.columns = [f"{c:g}%" for c in p.columns]
    return p.reindex([m for m in KBEST_METHODS if m in p.index]).round(4)


def ladder_full(df):
    out = []
    for model in df["Model"].unique():
        for m in KBEST_METHODS:
            t = ladder_table(df, model, m)
            if len(t) > 1:
                t.insert(0, "Model", model)
                t.insert(1, "Method", m)
                out.append(t)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def save_results(df, out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    df.round(5).to_csv(os.path.join(out_dir, "results.csv"), index=False)
    headline_table(df).to_csv(os.path.join(out_dir, "comparison_table.csv"),
                              index=False)
    lf = ladder_full(df)
    if not lf.empty:
        lf.to_csv(os.path.join(out_dir, "reduction_ladder.csv"), index=False)


# ================================================================ figures
def _finish(fig, path, show):
    if not getattr(fig, "_custom_layout", False):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig.tight_layout()
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    if not show:
        plt.close(fig)
    return fig


def _methods_in(df):
    return [m for m in [BASELINE] + ALL_METHODS if m in set(df["Method"])]


def _grouped(ax, df, col, models, methods, log=False):
    w = 0.8 / len(models)
    x = np.arange(len(methods))
    for i, mod in enumerate(models):
        vals = [df[(df["Model"] == mod) & (df["Method"] == m)][col].mean()
                for m in methods]
        ax.bar(x - 0.4 + w * (i + 0.5), vals, w, label=mod)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[m] for m in methods], rotation=30, ha="right")
    if log:
        ax.set_yscale("log")
    ax.set_title(col)


def fig_dataset_overview(data, path=None, show=False):
    X, y, names = data["X_train"], data["y_train"], data["class_names"]
    info = describe(data)
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
    counts = np.bincount(y, minlength=len(names))
    ax[0].bar(range(len(names)), counts, color="#378ADD")
    ax[0].set_xticks(range(len(names)))
    ax[0].set_xticklabels(names, rotation=75, ha="right", fontsize=7)
    ax[0].set_title("Training samples per class")
    nnz = np.diff(X.indptr)
    ax[1].hist(nnz, bins=40, color="#1D9E75")
    ax[1].set_xlabel("Non-zero TF-IDF terms per document")
    ax[1].set_ylabel("Documents")
    ax[1].set_title(f"{info['features']:,} features, density "
                    f"{info['density_%']}%")
    return _finish(fig, path, show)


def fig_dashboard(df, path=None, show=False):
    d = df[df["Experiment"] == "main"]
    models, methods = list(d["Model"].unique()), _methods_in(d)
    fig, ax = plt.subplots(2, 2, figsize=(15, 9))
    for a, (col, log) in zip(ax.ravel(), [("Accuracy", False),
                                          ("Macro-F1", False),
                                          ("Train Time (s)", True),
                                          (INF_COL, True)]):
        _grouped(a, d, col, models, methods, log)
        if not log:
            lo = d[col].min()
            a.set_ylim(max(0, lo - 0.08), min(1.0, d[col].max() + 0.03))
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(models), fontsize=9)
    fig.suptitle("All features vs. selected features, per model")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig._custom_layout = True
    return _finish(fig, path, show)


def fig_heatmap(df, path=None, show=False):
    d = df[(df["Experiment"] == "main") & (df["Method"] != BASELINE)]
    piv = d.pivot_table(index="Model", columns="Method",
                        values="F1 Change (pts)")
    piv = piv[[m for m in ALL_METHODS if m in piv.columns]]
    v = max(1.0, float(np.nanmax(np.abs(piv.values))))
    fig, ax = plt.subplots(figsize=(1.6 * piv.shape[1] + 3, 1 * piv.shape[0] + 2))
    im = ax.imshow(piv.values, cmap="RdYlGn", vmin=-v, vmax=v, aspect="auto")
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels([SHORT[c] for c in piv.columns])
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels(piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:+.1f}", ha="center",
                    va="center", fontsize=10)
    fig.colorbar(im, label="Macro-F1 change vs. all features (points)")
    ax.set_title("Macro-F1 change: model x selection method")
    return _finish(fig, path, show)


def fig_efficiency_scatter(df, path=None, show=False):
    d = df[df["Experiment"] == "main"]
    models, methods = list(d["Model"].unique()), _methods_in(d)
    colors = plt.cm.tab10(np.arange(len(models)))
    marks = dict(zip(methods, ["*", "o", "s", "^", "D", "v", "P", "X"]))
    nfull = d["Features"].max()
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for c, mod in zip(colors, models):
        for m in methods:
            r = d[(d["Model"] == mod) & (d["Method"] == m)]
            if r.empty:
                continue
            ax.scatter(r["Train Time (s)"], r["Macro-F1"], color=c,
                       marker=marks[m],
                       s=50 + 250 * r["Features"] / nfull,
                       edgecolor="k", alpha=0.85)
    ax.set_xscale("log")
    ax.set_xlabel("Training time (s, log scale)  -  left is faster")
    ax.set_ylabel("Macro-F1  -  up is better")
    ax.set_title("Efficiency vs. performance (marker size = number of features)")
    h1 = [plt.Line2D([], [], marker="o", ls="", color=c, label=m)
          for c, m in zip(colors, models)]
    h2 = [plt.Line2D([], [], marker=marks[m], ls="", color="gray",
                     markeredgecolor="k", label=SHORT[m]) for m in methods]
    l1 = ax.legend(handles=h1, title="Model", loc="upper left",
                   bbox_to_anchor=(1.01, 1.0), fontsize=8)
    ax.add_artist(l1)
    ax.legend(handles=h2, title="Method", loc="upper left",
              bbox_to_anchor=(1.01, 0.55), fontsize=8)
    return _finish(fig, path, show)


def _ladder_axis(g):
    """Ordered levels (100% first) with their feature counts."""
    lv = (g[["Level (%)", "Features"]].drop_duplicates()
          .sort_values("Level (%)", ascending=False))
    return lv["Level (%)"].tolist(), lv["Features"].tolist()


def _ladder_xticks(ax, levels, feats):
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels([f"{l:g}%\n{f:,}" for l, f in zip(levels, feats)],
                       fontsize=8)
    ax.set_xlabel("Features kept (all features on the left)")


def fig_k_sweep(df, main_df=None, path=None, show=False):
    """Macro-F1 across the step-down ladder, one panel per model."""
    g = df[df["Experiment"] == "ladder"]
    models = list(g["Model"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(4.8 * len(models), 4.6),
                             squeeze=False)
    for a, mod in zip(axes[0], models):
        gm = g[g["Model"] == mod]
        levels, feats = _ladder_axis(gm)
        base = gm[gm["Method"] == BASELINE]["Macro-F1"].iloc[0]
        for m in KBEST_METHODS:
            r = gm[gm["Method"] == m].set_index("Level (%)")["Macro-F1"]
            ys = [base if l == 100.0 else r.get(l, np.nan) for l in levels]
            a.plot(range(len(levels)), ys, marker="o", label=SHORT[m])
        a.axhline(base, ls="--", color="k", lw=1)
        a.set_title(mod)
        _ladder_xticks(a, levels, feats)
    axes[0, 0].set_ylabel("Macro-F1")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Macro-F1 as features are reduced step by step "
                 "(dashed = all features)")
    return _finish(fig, path, show)


def fig_ladder(df, model=PRIMARY_MODEL, path=None, show=False):
    """Accuracy, Macro-F1, training and inference time down the ladder."""
    g = df[(df["Experiment"] == "ladder") & (df["Model"] == model)]
    levels, feats = _ladder_axis(g)
    base = g[g["Method"] == BASELINE].iloc[0]
    fig, ax = plt.subplots(2, 2, figsize=(15, 9))
    panels = [("Accuracy", "Accuracy", False), ("Macro-F1", "Macro-F1", False),
              ("Training time (s)", "Train Time (s)", True),
              ("Inference time (ms / 1000 samples)", INF_COL, True)]
    for a, (title, col, log) in zip(ax.ravel(), panels):
        for m in KBEST_METHODS:
            r = g[g["Method"] == m].set_index("Level (%)")[col]
            ys = [base[col] if l == 100.0 else r.get(l, np.nan) for l in levels]
            a.plot(range(len(levels)), ys, marker="o", label=SHORT[m])
        a.axhline(base[col], ls="--", color="k", lw=1)
        a.set_title(title)
        if log:
            a.set_yscale("log")
        _ladder_xticks(a, levels, feats)
    ax[0, 0].legend(fontsize=9)
    fig.suptitle(f"{model}: all features, then successive reduction")
    return _finish(fig, path, show)


def fig_speedup(df, path=None, show=False):
    d = df[(df["Experiment"] == "main") & (df["Method"] != BASELINE)]
    models, methods = list(d["Model"].unique()), _methods_in(d)
    methods = [m for m in methods if m != BASELINE]
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.8))
    for a, col in zip(ax, ["Train Speedup (x)", "Inference Speedup (x)"]):
        _grouped(a, d, col, models, methods)
        a.axhline(1, color="k", ls="--", lw=1)
        a.set_ylabel("x faster than all features")
    ax[0].legend(fontsize=8)
    return _finish(fig, path, show)


def fig_overlap(indices, path=None, show=False):
    names = list(indices)
    n = len(names)
    J = np.zeros((n, n))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            A, B = set(indices[a]), set(indices[b])
            J[i, j] = len(A & B) / max(1, len(A | B))
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(J, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([SHORT[m] for m in names], rotation=30, ha="right")
    ax.set_yticklabels([SHORT[m] for m in names])
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{J[i, j]:.2f}", ha="center", va="center",
                    fontsize=9, color="white" if J[i, j] > 0.6 else "black")
    fig.colorbar(im, label="Jaccard overlap")
    ax.set_title("Do the methods pick the same features?")
    return _finish(fig, path, show)


def fig_top_words(scores, feature_names, n=12, path=None, show=False):
    methods = [m for m in KBEST_METHODS if m in scores]
    fig, ax = plt.subplots(figsize=(3.1 * len(methods), 0.42 * n + 1.5))
    ax.axis("off")
    for j, m in enumerate(methods):
        top = np.argsort(-scores[m], kind="stable")[:n]
        ax.text(j, 1.0, SHORT[m], fontweight="bold", ha="left", va="bottom",
                transform=ax.transData)
        for i, f in enumerate(top):
            ax.text(j, 1.0 - (i + 1) / (n + 1), feature_names[f], ha="left",
                    va="center", fontsize=10)
    ax.set_xlim(-0.1, len(methods) - 0.1)
    ax.set_ylim(0, 1.1)
    ax.set_title(f"Top {n} words ranked by each method", loc="left")
    return _finish(fig, path, show)


def fig_per_class_confusion(data, indices, df, model=PRIMARY_MODEL,
                            path=None, show=False):
    """Per-class F1 change and confusion matrices: all vs. best selected."""
    sub = df[(df["Experiment"] == "main") & (df["Model"] == model)
             & (df["Method"] != BASELINE)]
    best = sub.sort_values(["Macro-F1", "Train Speedup (x)"],
                           ascending=False).iloc[0]["Method"]
    idx = indices[best]
    Xtr, Xte = data["X_train"], data["X_test"]
    ytr, yte, names = data["y_train"], data["y_test"], data["class_names"]
    p_all = make_model(model).fit(Xtr, ytr).predict(Xte)
    p_sel = make_model(model).fit(Xtr[:, idx], ytr).predict(Xte[:, idx])
    f_all = f1_score(yte, p_all, average=None)
    f_sel = f1_score(yte, p_sel, average=None)
    diff = 100 * (f_sel - f_all)
    order = np.argsort(diff)

    fig = plt.figure(figsize=(19, 6.5))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.2, 1, 1, 0.05], wspace=0.35)
    a0 = fig.add_subplot(gs[0, 0])
    a0.barh(range(len(names)), diff[order],
            color=["#E24B4A" if d < 0 else "#1D9E75" for d in diff[order]])
    a0.set_yticks(range(len(names)))
    a0.set_yticklabels([names[i] for i in order], fontsize=7)
    a0.axvline(0, color="k", lw=1)
    a0.set_xlabel("F1 change (points)")
    a0.set_title(f"Per-class F1 change\n{model}: {SHORT[best]} vs. all features")
    im = None
    for k, (p, t) in enumerate([(p_all, "All features"),
                                (p_sel, f"{SHORT[best]} ({len(idx):,} features)")]):
        a = fig.add_subplot(gs[0, 1 + k])
        cm = confusion_matrix(yte, p, normalize="true")
        im = a.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        a.set_title(f"Confusion matrix: {t}")
        a.set_xlabel("Predicted class")
        a.set_ylabel("True class")
        step = max(1, len(names) // 10)
        a.set_xticks(range(0, len(names), step))
        a.set_yticks(range(0, len(names), step))
    fig.colorbar(im, cax=fig.add_subplot(gs[0, 3]), label="Row-normalised")
    fig._custom_layout = True
    return _finish(fig, path, show), best


# ================================================================ runner
def run_all(source="20newsgroups", models=None, k=3000, levels=None, repeats=5,
            out_dir="results", fast=False, sweep_repeats=2):
    models = models or DEFAULT_MODELS
    methods, levels = None, levels or DEFAULT_LEVELS
    if fast:
        models = [m for m in models if m in ("LinearSVC", "NaiveBayes")]
        methods = [m for m in ALL_METHODS
                   if m not in ("Mutual Information", "Random Forest")]
        levels, repeats, sweep_repeats = [50, 25, 10, 5, 1], 3, 1
    fig_dir = os.path.join(out_dir, "figures")

    data = get_data(source)
    print("Dataset:", describe(data))
    fig_dataset_overview(data, os.path.join(fig_dir, "01_dataset_overview.png"))

    main_df, scores, indices = run_main_comparison(
        data, models, k, repeats, methods)
    ladder = run_reduction_ladder(
        data, scores, main_df, models, levels, repeats=sweep_repeats,
        methods=[m for m in KBEST_METHODS if m in scores])
    df = add_deltas(pd.concat([main_df, ladder], ignore_index=True))
    save_results(df, out_dir)

    main_df = df[df["Experiment"] == "main"]
    fig_dashboard(df, os.path.join(fig_dir, "02_metrics_dashboard.png"))
    fig_heatmap(df, os.path.join(fig_dir, "03_model_method_heatmap.png"))
    fig_efficiency_scatter(df, os.path.join(fig_dir, "04_efficiency_scatter.png"))
    fig_k_sweep(df, path=os.path.join(fig_dir, "05_reduction_ladder.png"))
    for mod in models:
        fig_ladder(df, mod, os.path.join(fig_dir, f"10_ladder_{mod}.png"))
    fig_speedup(df, os.path.join(fig_dir, "06_speedups.png"))
    fig_overlap(indices, os.path.join(fig_dir, "07_feature_overlap.png"))
    fig_top_words(scores, data["feature_names"],
                  path=os.path.join(fig_dir, "08_top_words.png"))
    _, best = fig_per_class_confusion(
        data, indices, df, path=os.path.join(fig_dir, "09_per_class_confusion.png"))

    print("\n=== HEADLINE TABLE (primary model: %s) ===" % PRIMARY_MODEL)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    print(headline_table(df).to_string(index=False))
    print("\n=== STEP-DOWN LADDER: Macro-F1, %s (100%% = all features) ===" % PRIMARY_MODEL)
    print(ladder_pivot(df).to_string())
    print(f"\nSaved results.csv, comparison_table.csv, reduction_ladder.csv "
          f"and figures in {out_dir}/")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="20newsgroups",
                    choices=["20newsgroups", "sample"])
    ap.add_argument("--models", default="default",
                    help="comma list, or 'default'. Options: LinearSVC,"
                         "LogisticRegression,NaiveBayes,RandomForest,kNN")
    ap.add_argument("--k", type=int, default=3000)
    ap.add_argument("--levels", default=",".join(map(str, DEFAULT_LEVELS)),
                    help="%% of features kept after 'all features'")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", default="results")
    ap.add_argument("--fast", action="store_true",
                    help="2 models, fewer methods and k values")
    a = ap.parse_args()
    run_all(a.data,
            None if a.models == "default" else a.models.split(","),
            a.k, [float(x) for x in a.levels.split(",")], a.repeats, a.out, a.fast)
