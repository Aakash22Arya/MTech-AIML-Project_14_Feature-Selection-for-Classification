# Data

## Dataset used: 20 Newsgroups
- ~18,800 newsgroup posts, 20 topic classes (official train/test split:
  11,314 train / 7,532 test).
- Loaded automatically by `sklearn.datasets.fetch_20newsgroups`
  (downloaded once and cached in `~/scikit_learn_data`). No manual download needed.
- Headers, footers and quoted replies are **removed**
  (`remove=("headers", "footers", "quotes")`). Without this, classifiers can
  score highly by reading metadata (e.g. e-mail addresses, host names) instead
  of the message topic, which would make the experiment unrealistic.
- Text is converted to TF-IDF (stop words removed, `min_df=2`, sublinear TF).
  The vocabulary and IDF weights are fitted on the **training split only**.
  This gives tens of thousands of sparse features (exact count is printed by the
  notebook and saved in the results).

## `sample_data.csv`
A tiny **synthetic** file (300 short toy documents, 5 topics, columns
`text,label`) so the whole pipeline can be smoke-tested offline in seconds:

    python -m src.evaluation --data sample

It is only a smoke test and its results are **not** the project results.
To replace it with a small subset of real 20 Newsgroups posts
(10 per class, requires internet once):

    python -m src.preprocessing --make-sample --n 10
