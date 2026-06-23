# MovieLens 1M Matrix Factorization

This project builds a reproducible movie recommender on the official [MovieLens 1M dataset](https://grouplens.org/datasets/movielens/1m/). It compares truncated SVD from `scipy.sparse.linalg.svds` with a locally implemented biased probabilistic matrix factorization model trained by shuffled SGD. Generated artifacts power an interactive Streamlit application without retraining.

## Clean-clone run order on Windows with Git Bash

1. Install dependencies:

   ```bash
   cd /d/TSchool/matrix-factorization
   python -m venv .venv
   source .venv/Scripts/activate
   python -m pip install -r requirements.txt
   ```

2. Generate data, models, reports, and serving artifacts:

   ```bash
   python -m scripts.run_pipeline
   ```

3. Validate the generated project:

   ```bash
   python -m scripts.validate_project
   ```

4. Launch the application:

   ```bash
   python -m streamlit run app.py
   ```

`run_pipeline` downloads MovieLens 1M when needed and generates all ignored serving
artifacts. Tests and notebook execution can be run independently:

```bash
python -m pytest -q
python -m scripts.validate_project
python -m jupyter nbconvert --to notebook --execute Movie_Recommender_System.ipynb --output Movie_Recommender_System.ipynb
python -m streamlit run app.py
```

## Project structure

- `data/`: raw MovieLens `ratings.dat`, `users.dat`, and `movies.dat`.
- `processed/`: generated split CSVs and mappings. The large normalized
  `user_item_matrix.csv` is generated locally and ignored by Git.
- `models/`: SVD and local PMF implementations.
- `utils/`: parsing, validation, splitting, matrix, metric, recommendation, and artifact helpers.
- `reports/`: generated model factors, metrics, tuning results, plots, and example
  recommendations. The large `svd_predictions.npy` serving matrix is generated
  locally and ignored by Git.
- `scripts/`: data download, complete pipeline, validation, and bounded Streamlit smoke test.
- `tests/`: fast synthetic unit and integration tests.
- `Movie_Recommender_System.ipynb`: executed analysis using reusable production modules.
- `app.py`: artifact-backed Streamlit dashboard.

## Methodology

The original observed rating rows are split independently per user with `random_state=42`. Allocation is approximately 70% train, 15% validation, and 15% test, with at least one validation and test interaction for eligible users. Every user retains training history. Any held-out movie absent from train is deterministically moved to train. Partitions are non-overlapping and both models use the identical rows. Hyperparameters are chosen only by validation RMSE; test is evaluated exactly once after retraining on train plus validation.

For SVD, each user's mean is calculated from observed training ratings and subtracted only from those observations. A regularized item correction is then estimated from the observed user-centered residuals, and `svds` factorizes the remaining sparse residual matrix. Unobserved entries stay conceptually missing and appear as sparse zeros. Singular values are reordered descending before reconstruction; user means and item corrections are restored.

`reports/svd_predictions.npy` stores raw, unclipped SVD prediction scores so
recommendation ordering remains meaningful above 5 and below 1. It is not stored
in Git because of its size. `processed/user_item_matrix.csv` is also generated and
ignored for the same reason. Displayed ratings and all MSE/RMSE calculations clip
model predictions to the valid `[1, 5]` rating range.

PMF predicts `global_mean + user_bias + item_bias + dot(user_factors, item_factors)`. The local implementation uses seeded initialization, shuffled rating-level SGD, separate factor and bias regularization, validation-only early stopping, best-checkpoint restoration, and finite-value checks. The bounded validation search jointly evaluates 96, 112, and 128 factors with factor regularization 0.05, 0.06, and 0.07 at learning rate 0.006. Each configuration may run up to 70 epochs with patience 8. Test ratings are not used for this selection.

MSE is the mean squared test-rating error; RMSE is its square root. PMF improvement is `(SVD_RMSE - PMF_RMSE) / SVD_RMSE * 100`.

## Evaluated users and recommendations

Overall metrics use every test rating. Three example users are selected before recommendation inspection: the users nearest the 25th, 50th, and 75th percentiles of training interaction count. Recommendations exclude every movie in each user's full known MovieLens history. They rank by raw model score descending and use movie ID ascending only when raw scores are equal. Tables expose both the raw `ranking_score` and its clipped `predicted_rating`.

## Final generated metrics

The deterministic full run produced:

| Model | Test MSE | Test RMSE |
|---|---:|---:|
| SVD | 0.793518 | 0.890796 |
| PMF | 0.712165 | 0.843899 |

PMF improves RMSE over SVD by **5.265%**. All assignment targets are met.

Validation selected SVD rank 20 with item-bias regularization 5.0. PMF validation selected 128 factors, learning rate 0.006, factor regularization 0.06, bias regularization 0.02, and epoch 53, with validation RMSE 0.849353. Early stopping completed after 61 tuning epochs. The selected epoch is below the 70-epoch boundary, but the selected factor count is the search maximum of 128, so the factor boundary remains open and is reported rather than expanded automatically. Final PMF training uses train plus validation for exactly 53 epochs without a holdout. The exact split contains 705,806 train rows, 147,201 validation rows, and 147,202 untouched test rows.

The automatically selected showcase users are user 91 (32 train ratings, nearest the 25th percentile), user 40 (68, median), and user 1186 (146, nearest the 75th percentile). Authoritative machine-readable values remain in `reports/model_metrics.json` and `reports/evaluated_users.json`.

## Limitations

The models use collaborative ratings only. They do not solve new-user cold start, learn semantic genre preferences directly, optimize ranking metrics, or model changing taste over time. Unrated catalog items are omitted from model mappings because they contain no collaborative signal.
