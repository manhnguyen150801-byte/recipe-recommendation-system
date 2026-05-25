# Notebook Guide

**File:** `notebook.ipynb`

This notebook is the complete analysis pipeline for the recipe recommendation system. It is written as a professional report: every section opens with a markdown cell explaining what is being done and why, and closes with an interpretation of the results.

---

## Sections

### Section 1 — Data Loading
Loads `train.csv` (165,226 user–recipe interactions) and `metadata.csv` (231,637 recipes). Prints shapes, column types, null counts, and duplicate checks for both files.

### Section 2 — Exploratory Data Analysis
Explores the structure of the data before any modelling:

- **Rating distribution** — reveals extreme positive skew (73% of ratings are 6/6)
- **Unique users and recipes** — 11,346 users, 62,517 recipes
- **Ratings per user** — mean 14.6, median 16; most users are highly active
- **Ratings per recipe** — mean 2.6, median 1; long-tail distribution with most recipes barely rated
- **Matrix sparsity** — 99.97% of the user×recipe matrix is empty, which is the key challenge for collaborative filtering

### Section 3 — Train / Validation / Test Split
Performs a random 70/15/15 split (`random_state=42`). The validation set is used only for hyperparameter tuning; the test set is held out and used once for final evaluation. Cold-start analysis reveals 6,751+ recipes in the val/test sets have zero training ratings.

### Section 4 — Content-Based Recommender
Builds a TF-IDF representation (5,000 features, `sublinear_tf=True`, English stop-words removed) from recipe tags, ingredients, and descriptions. Predictions are computed as a cosine-similarity-weighted average of the K most similar recipes a user has already rated.

- Includes 3 sample recommendation lists for real users
- Evaluated on the validation set: RMSE 1.1118, MAE 0.6619

### Section 5 — Collaborative Filtering (User-Based KNNBasic)
Implements User-Based CF using `KNNBasic` from scikit-surprise with cosine similarity and k=20 neighbours. Due to the extreme matrix sparsity, 97.1% of validation predictions fall back to the global mean — the model is effectively non-personalised.

### Section 6 — Matrix Factorization (SVD)
Uses biased SVD from scikit-surprise. Hyperparameter tuning sweeps `n_factors ∈ {5, 20, 50}` and `n_epochs ∈ {10, 20}` on the validation set. Best configuration: `n_factors=5, n_epochs=20` (Val RMSE 1.0680). Final model is re-fit on the full training set.

### Section 7 — Evaluation on Test Set
Evaluates all three models on the held-out test set for the first and only time. Reports RMSE, MAE, Recall, Precision, and F1, plus a side-by-side comparison bar chart.

### Section 8 — Conclusion
Markdown-only section covering model comparison, strengths and weaknesses of each approach, the rationale for SVD's superior performance, and a detailed discussion of limitations (rating skew, item sparsity, unused review text, TF-IDF semantic gap) and what could be improved with more time.

---

## How to Run

From the project root:

```bash
conda activate rec_system
jupyter notebook notebook/notebook.ipynb
```

Then select **Kernel → Restart & Run All** to execute every cell cleanly from scratch. Expected total runtime: ~3–5 minutes (the TF-IDF evaluation loop and SVD tuning grid are the slowest steps).

---

## Notes

- `random_state=42` is set everywhere randomness is involved — results are fully reproducible.
- Data paths in the notebook (`../data/train.csv`, `../data/metadata.csv`) are relative to the `notebook/` directory. Do not move the notebook or the `data/` folder independently.
- All code follows the style and complexity level of the course practice notebooks: no custom classes, no lambda functions, explicit for-loops over comprehensions where clarity matters.
