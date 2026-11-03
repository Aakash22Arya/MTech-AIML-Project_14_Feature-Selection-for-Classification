# Feature Selection for Efficient Text Classification
**AI26501 Modern AI - Mini-Project (Assignment 17)**
Group 14 | Members: Aakash Arya (MT\AI\10035\26), Tinku Raj (MT\AI\10014\26) | Instructor: Dr. Shruti Garg

## 1. Problem
Classification datasets with very many features (such as TF-IDF text) make models slow
to train and slow at prediction time. **Can we keep only a small subset of the features and
get a faster model without sacrificing predictive performance?**

## 2. Task (from the assignment)
- Choose a classification dataset with a reasonably large number of features.
- Train a model using all features, then apply one or more simple feature-selection methods.
- Determine whether reducing the input dimensionality improves efficiency without
  sacrificing predictive performance.
- **Compare:** all features vs. selected features.
- **Metrics:** Number of Features, Accuracy, Macro-F1, Training Time, Inference Time.

## 3. Setup
| Item | Choice |
|---|---|
| Dataset | 20 Newsgroups, 20 classes, official train/test split. Headers/footers/quotes removed. |
| Features | TF-IDF (stop words removed, `min_df=2`, sublinear TF): tens of thousands of features |
| Models | LinearSVC (**primary**), Logistic Regression, Multinomial Naive Bayes, Random Forest (k-NN optional) |
| Selection methods | Variance Threshold, Chi-square, ANOVA F-test, Mutual Information (filters); L1-SVM, Random Forest importance (embedded) |
| Features kept | `k = 3000` for top-k methods; Variance Threshold and L1-SVM keep what their own rule keeps |
| Step-down comparison | **All features first**, then 75%, 50%, 25%, 10%, 5%, 2% and 1% of the features, per model and method (`--levels` to change) |

**Fairness rules**
- Same train/test split and random seed for every setup.
- TF-IDF and all selectors are fitted on the **training set only** (no leakage).
- Training and inference times are the **median of repeated runs** (5; fewer for the slow
  Random Forest and Logistic Regression). Inference time is reported per 1,000 samples.
- Selection time is reported separately and also as "Total speedup incl. selection",
  so efficiency claims are honest.
- The same selected feature set is used for every model within a method.

**Verdict rule for "without sacrificing performance".** A setup gets `Yes` when
Accuracy and Macro-F1 each drop by **at most 1 point** versus all features
**and** training is at least **1.1x faster**. Otherwise `No`.

## 4. Repository structure
```
MTech-AIML-Project_14_FeatureSelection/
├── README.md
├── requirements.txt
├── data/
│   ├── README.md          dataset description and how it is loaded
│   └── sample_data.csv    tiny toy file for offline smoke tests
├── notebooks/
│   └── experiment.ipynb   full experiment with all figures
├── src/
│   ├── preprocessing.py   loading, split, TF-IDF (fit on train only)
│   ├── model.py           classifiers and feature-selection methods
│   └── evaluation.py      metrics, timing, experiments, tables, figures
├── results/
│   ├── results.csv            every model x method x k
│   ├── comparison_table.csv   headline "all vs selected" table (LinearSVC)
│   ├── reduction_ladder.csv   step-by-step table: all features, then each reduction
│   └── figures/               01 ... 09 PNG figures
└── presentation/
    └── 14_FeatureSelection.pptx
```

## 5. How to run
```bash
pip install -r requirements.txt

# Option A: notebook (recommended - shows every figure)
jupyter notebook notebooks/experiment.ipynb

# Option B: command line, full run (~20-40 min)
python -m src.evaluation --data 20newsgroups

# Quick run (2 models, fewer methods and levels)
# Custom reduction steps: python -m src.evaluation --levels 80,60,40,20,10,5
python -m src.evaluation --data 20newsgroups --fast

# Offline smoke test on the toy sample (seconds; NOT the project results)
python -m src.evaluation --data sample
```
The first run downloads 20 Newsgroups once (needs internet).
Add `--models LinearSVC,LogisticRegression,NaiveBayes,RandomForest,kNN` to include k-NN.

## 6. Figures (saved in `results/figures/`)
1. Dataset overview: class balance and TF-IDF sparsity
2. Metrics dashboard: accuracy, Macro-F1, training time, inference time per model
3. Model x method heatmap of Macro-F1 change
4. Efficiency vs. performance scatter
5. Step-down ladder: Macro-F1 as features are reduced, all features on the left
6. Training and inference speedups
7. Feature overlap between methods
8. Top words chosen by each method
9. Per-class F1 change and confusion matrices (all vs. best selected)
10. Per-model ladder: accuracy, Macro-F1, training and inference time at every step


- One dataset and TF-IDF features only; default hyper-parameters.
- Timings depend on the machine; results use one random seed.
- Future work: embeddings, wrapper methods (RFE), several seeds with error bars,
  neural models, and other data modalities (tabular, images).
