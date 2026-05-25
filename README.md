# Recipe Recommendation System

A recipe recommendation system built on the [food.com](https://www.food.com/) dataset, implementing and comparing three recommender approaches: Content-Based Filtering, User-Based Collaborative Filtering, and Matrix Factorization (SVD).

**Course:** Recommendation Tools — IESEG School of Management  
**Instructor:** Fernando DIAZGONZALEZ  
**Author:** Duc Manh NGUYEN

---

## Project Overview

The goal of this project is to predict how a given user would rate a recipe they have not yet tried, and to use those predictions to recommend relevant recipes. Three models are trained and evaluated on a held-out test set:

- **Content-Based Filtering** — represents each recipe as a TF-IDF vector of its tags, ingredients, and description, then recommends recipes that are most similar to ones the user has already liked.
- **User-Based Collaborative Filtering** — finds users with similar rating histories and recommends items that those neighbours rated highly (implemented with `KNNBasic` from scikit-surprise, cosine similarity, k=20).
- **Matrix Factorization (SVD)** — decomposes the user-item rating matrix into latent factors to capture underlying preference patterns; more robust to the extreme data sparsity present in this dataset.

---

## Repository Structure

```
recommendation_project/
├── README.md                  ← you are here
├── .gitignore
├── requirements.txt           ← Python dependencies
├── notebook/
│   ├── notebook.ipynb         ← full analysis: EDA → models → evaluation
│   └── README.md              ← notebook guide
└── data/
    ├── train.csv              ← user–recipe interactions (165,226 rows)
    ├── metadata.csv           ← recipe metadata (231,637 rows)
    └── README.md              ← data dictionary
```

---

## Key Results

All models were evaluated on a held-out test set (24,784 interactions, 15% of the data). Ratings are on a 1–6 scale. Classification metrics use a threshold of 4 (≥ 4 = "relevant").

| Metric | Content-Based | User-Based CF | SVD (k=5) |
|---|:---:|:---:|:---:|
| **RMSE** | 1.0907 | 1.0877 | **1.0518** |
| **MAE** | **0.6491** | 0.7253 | 0.6553 |
| Recall | 0.9852 | 0.9999 | 0.9987 |
| Precision | **0.9483** | 0.9437 | 0.9453 |
| F1 | 0.9664 | 0.9710 | **0.9712** |

**SVD achieves the best RMSE and F1**, making it the strongest overall recommender. The content-based model achieves the best MAE and Precision. User-Based CF degrades almost entirely to a global-mean baseline due to the 99.97% matrix sparsity — 97% of its predictions fall back to the global mean.

> **Note on inflated classification metrics:** 73% of all ratings in the dataset are the maximum (6/6), so any model predicting values near the global mean (~5.5) will naturally achieve high Recall and F1. These figures reflect rating skew more than model quality; RMSE is the more reliable comparison metric.

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/recommendation_project.git
cd recommendation_project
```

### 2. Set up the environment

This project uses a Conda environment with Python 3.10.

```bash
conda create -n rec_system python=3.10 -y
conda activate rec_system
pip install -r requirements.txt
```

### 3. Launch the notebook

```bash
jupyter notebook notebook/notebook.ipynb
```

Run all cells from top to bottom (`Kernel → Restart & Run All`). The notebook is self-contained and will produce all EDA plots, model outputs, and the final comparison table.

---

## Dependencies

| Package | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy <2.0` | Numerical operations |
| `scikit-learn` | TF-IDF vectorisation, cosine similarity, train/test splitting |
| `matplotlib` | EDA and results visualisation |
| `scikit-surprise` | KNNBasic (CF) and SVD (Matrix Factorization) |

See `requirements.txt` for pinned versions.

---

## Data

| File | Rows | Description |
|---|---:|---|
| `data/train.csv` | 165,226 | User–recipe interactions with a 1–6 star rating |
| `data/metadata.csv` | 231,637 | Recipe metadata: name, tags, ingredients, description, nutrition, steps |

See `data/README.md` for a full column-level data dictionary.
