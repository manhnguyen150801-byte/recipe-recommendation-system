# Data Dictionary

The dataset is derived from [food.com](https://www.food.com/) and contains user ratings and recipe metadata. Both files are required to run the notebook.

> **Note:** The raw data files (`train.csv`, `metadata.csv`) are not tracked in this repository because of their size. Download them and place them in this `data/` folder before running the notebook.

---

## train.csv — User–Recipe Interactions

165,226 rows · 5 columns

| Column | Type | Description |
|---|---|---|
| `user_id` | string | Unique user identifier (e.g. `U9240752`) |
| `recipe_id` | string | Unique recipe identifier (e.g. `R6574412`) |
| `date` | string | Date the rating was submitted (`YYYY-MM-DD`) |
| `rating` | int | Star rating from 1 (worst) to 6 (best) |
| `review` | string | Free-text review written by the user |

**Key statistics:**
- 11,346 unique users
- 62,517 unique recipes
- Ratings heavily skewed: 73% of all ratings are 6/6
- No null values, no duplicates

---

## metadata.csv — Recipe Metadata

231,637 rows · 12 columns

| Column | Type | Description |
|---|---|---|
| `name` | string | Recipe name (1 null) |
| `id` | string | Recipe identifier — matches `recipe_id` in `train.csv` |
| `minutes` | int | Estimated preparation time in minutes |
| `contributor_id` | string | User who submitted the recipe |
| `submitted` | string | Date the recipe was submitted (`YYYY-MM-DD`) |
| `tags` | string (list) | List of descriptive tags (e.g. `['weeknight', 'quick', ...]`) |
| `nutrition` | string (list) | Nutritional values: `[calories, total_fat, sugar, sodium, protein, saturated_fat, carbohydrates]` |
| `n_steps` | int | Number of preparation steps |
| `steps` | string (list) | Step-by-step cooking instructions |
| `description` | string | Free-text recipe description (4,979 nulls) |
| `ingredients` | string (list) | List of ingredients (e.g. `['flour', 'sugar', ...]`) |
| `n_ingredients` | int | Number of ingredients |

**Key statistics:**
- 100% of recipes appearing in `train.csv` are present in `metadata.csv`
- `tags`, `ingredients`, and `description` are used as the content signal for the content-based model
- `nutrition` and `steps` are not used in modelling but are available for future work
