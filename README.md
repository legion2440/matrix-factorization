# MovieLens 1M Matrix Factorization

This project builds a reproducible movie recommender on the official [MovieLens 1M dataset](https://grouplens.org/datasets/movielens/1m/). It compares truncated SVD from `scipy.sparse.linalg.svds` with a locally implemented biased probabilistic matrix factorization model trained by shuffled SGD. Generated artifacts power an interactive Streamlit application without retraining.

## Installation on Windows with Git Bash

```bash
cd /d/TSchool/matrix-factorization
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements.txt
```

Download data and run the complete deterministic workflow:

```bash
python -m scripts.download_data
python -m scripts.run_pipeline
```

Other commands:

```bash
python -m pytest -q
python -m scripts.validate_project
python -m jupyter nbconvert --to notebook --execute Movie_Recommender_System.ipynb --output Movie_Recommender_System.ipynb
python -m streamlit run app.py
```

## Project structure

- `data/`: raw MovieLens `ratings.dat`, `users.dat`, and `movies.dat`.
- `processed/`: exact split CSVs, normalized training matrix, and stable ID mappings.
- `models/`: SVD and local PMF implementations.
- `utils/`: parsing, validation, splitting, matrix, metric, recommendation, and artifact helpers.
- `reports/`: model arrays, factors, metrics, tuning results, plots, and example recommendations.
- `scripts/`: data download, complete pipeline, validation, and bounded Streamlit smoke test.
- `tests/`: fast synthetic unit and integration tests.
- `Movie_Recommender_System.ipynb`: executed analysis using reusable production modules.
- `app.py`: artifact-backed Streamlit dashboard.

## Methodology

The original observed rating rows are split independently per user with `random_state=42`. Allocation is approximately 70% train, 15% validation, and 15% test, with at least one validation and test interaction for eligible users. Every user retains training history. Any held-out movie absent from train is deterministically moved to train. Partitions are non-overlapping and both models use the identical rows. Hyperparameters are chosen only by validation RMSE; test is evaluated exactly once after retraining on train plus validation.

For SVD, each user's mean is calculated from observed training ratings and subtracted only from those observations. A regularized item correction is then estimated from the observed user-centered residuals, and `svds` factorizes the remaining sparse residual matrix. Unobserved entries stay conceptually missing and appear as sparse zeros. Singular values are reordered descending before reconstruction; user means and item corrections are restored, and predictions are clipped to `[1, 5]`.

PMF predicts `global_mean + user_bias + item_bias + dot(user_factors, item_factors)`. The local implementation uses seeded initialization, shuffled rating-level SGD, separate factor and bias regularization, validation-only early stopping, best-checkpoint restoration, and finite-value checks.

MSE is the mean squared test-rating error; RMSE is its square root. PMF improvement is `(SVD_RMSE - PMF_RMSE) / SVD_RMSE * 100`.

## Evaluated users and recommendations

Overall metrics use every test rating. Three example users are selected before recommendation inspection: the users nearest the 25th, 50th, and 75th percentiles of training interaction count. Recommendations exclude every movie in each user's full known MovieLens history and use movie ID as a deterministic tie-breaker.

## Final generated metrics

The deterministic full run produced:

| Model | Test MSE | Test RMSE |
|---|---:|---:|
| SVD | 0.793518 | 0.890796 |
| PMF | 0.715559 | 0.845907 |

PMF improves RMSE over SVD by **5.039%**. All assignment targets are met.

Validation selected SVD rank 20 with item-bias regularization 5.0. PMF validation selected 96 factors, learning rate 0.006, factor regularization 0.06, bias regularization 0.02, and 45 epochs. The exact split contains 705,806 train rows, 147,201 validation rows, and 147,202 untouched test rows.

The automatically selected showcase users are user 91 (32 train ratings, nearest the 25th percentile), user 40 (68, median), and user 1186 (146, nearest the 75th percentile). Authoritative machine-readable values remain in `reports/model_metrics.json` and `reports/evaluated_users.json`.

## Limitations

The models use collaborative ratings only. They do not solve new-user cold start, learn semantic genre preferences directly, optimize ranking metrics, or model changing taste over time. Unrated catalog items are omitted from model mappings because they contain no collaborative signal.
