# MovieLens 1M Matrix Factorization

This project builds a reproducible MovieLens 1M recommender and compares three local models on the same deterministic split:

- Baseline collaborative filtering: regularized bias-only model, `global_mean + user_bias + item_bias`.
- SVD: truncated sparse residual factorization with regularized item residual bias.
- PMF: locally implemented biased matrix factorization trained by seeded shuffled SGD.

Generated artifacts power the notebook and Streamlit dashboard without retraining at display time.

## Clean-Clone Order

Use Windows Git Bash from the project root:

```bash
cd /d/TSchool/matrix-factorization
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements.txt
python -m scripts.run_pipeline
python -m pytest -q
python -m compileall models utils scripts app.py
python -m scripts.validate_project
python -m jupyter nbconvert --to notebook --execute Movie_Recommender_System.ipynb --output Movie_Recommender_System.ipynb
python -m scripts.validate_project
bash scripts/smoke_streamlit.sh
```

For manual dashboard use after artifacts exist:

```bash
python -m streamlit run app.py
```

The pipeline downloads MovieLens 1M only when raw files are missing.

## Project Structure

- `data/`: raw MovieLens `ratings.dat`, `users.dat`, and `movies.dat`.
- `processed/`: generated train/validation/test splits, mappings, and normalized matrix CSV.
- `models/`: local `BaselineCFModel`, `SVDModel`, and `PMFModel`.
- `utils/`: data loading, splitting, matrix construction, metrics, recommendation, artifact, and interpretability helpers.
- `reports/`: generated metrics, tuning diagnostics, factors, plots, recommendations, and explanations.
- `scripts/`: data download, full pipeline, validation, notebook builder, and bounded Streamlit smoke test.
- `tests/`: fast synthetic behavior tests.
- `Movie_Recommender_System.ipynb`: executed artifact-backed audit report.
- `app.py`: artifact-backed Streamlit dashboard.

## Data Split

The original rating rows are split deterministically per user with `random_state=42`. The target ratio is 70% train, 15% validation, and 15% test. Every user keeps training history, and held-out movies absent from train are deterministically moved into train so validation/test rows are covered by mappings.

Actual split counts:

| Split | Rows |
|---|---:|
| Train | 705,806 |
| Validation | 147,201 |
| Test | 147,202 |

Validation is used for hyperparameter and stopping decisions. Test rows are evaluated once after final refit on train plus validation.

## Benchmark CF

The benchmark collaborative filtering model is implemented in `models/baseline_cf.py` without Surprise, scikit-learn, or a recommender library. It learns regularized user and item biases only from observed training ratings:

```text
prediction = global_mean + user_bias + item_bias
```

Regularization is selected on validation from `[1, 2, 5, 10, 20, 40, 80]`. The selected value is 2.0 for both user and item biases. Final baseline refit uses train plus validation, then evaluates the untouched test rows.

## SVD Methodology

SVD user-centers observed training ratings, estimates a regularized item residual bias from observed residuals, factorizes the sparse residual matrix with `scipy.sparse.linalg.svds`, then restores user means and item corrections. `reports/svd_predictions.npy` stores raw, unclipped scores for recommendation ranking. Displayed ratings and RMSE/MSE use clipping to `[1, 5]`.

Selected SVD parameters:

- factors: 20
- item-bias regularization: 5.0

## PMF Methodology

PMF predicts:

```text
global_mean + user_bias + item_bias + dot(user_factors, item_factors)
```

The local SGD implementation uses seeded initialization, deterministic per-epoch shuffling from `random_state=42`, separate factor and bias regularization, validation-only early stopping, best-checkpoint restoration, and finite-value checks.

The stable PMF grid is intentionally preserved:

- factors: 96, 112, 128
- learning rate: 0.006
- factor regularization: 0.05, 0.06, 0.07
- bias regularization: 0.02
- max tuning epochs: 70
- patience: 8

Selected PMF config:

- factors: 128
- learning rate: 0.006
- factor regularization: 0.06
- bias regularization: 0.02
- selected epoch: 53
- validation RMSE: 0.849353

## Overfitting Protection

For the selected PMF config, validation RMSE reaches its minimum at epoch 53. With patience 8, tuning runs through epoch 61 and then stops. The final PMF refit uses exactly 53 epochs on train plus validation and does not use a validation holdout or the test set for stopping.

The selected epoch is not on the 70-epoch boundary. The selected factor count is at the searched factor boundary of 128, which is reported as a limitation rather than retuned here.

## Final Metrics

| Model | Test MSE | Test RMSE |
|---|---:|---:|
| Baseline CF | 0.824119 | 0.907810 |
| SVD | 0.793518 | 0.890796 |
| PMF | 0.712165 | 0.843899 |

SVD improves over baseline by 1.874% RMSE. PMF improves over baseline by 7.040% RMSE and over SVD by 5.265% RMSE.

Acceptance checks:

```text
SVD_RMSE < Baseline_CF_RMSE
PMF_RMSE < Baseline_CF_RMSE
```

Both pass.

## Interpretability

Global PMF factor interpretation uses saved final-refit item factors. The report selects the top five factors by item-loading variance and lists the highest positive and negative movies for each factor. It also aggregates genres by factor polarity and renders a latent-factor heatmap.

Important limitation: the sign of a latent factor is arbitrary. Factor meaning is inferred descriptively from movies and genres on both poles; it is not objective ground truth.

Similarity analysis computes cosine similarity between PMF item-factor vectors. Anchor movies are selected deterministically from popular mapped movies while encouraging genre diversity. Self-matches are excluded, and ties sort by similarity descending then movie ID ascending.

## Three Audit Users

The split is interaction-level, so selected users can have train, validation, and test rows. The two training-profile users are selected from users with sufficient train/test support near lower and upper quartiles of PMF per-user test RMSE. The third user is a separate deterministic test case.

| Role | User ID | Train | Validation | Test | SVD RMSE | PMF RMSE |
|---|---:|---:|---:|---:|---:|---:|
| `train_profile_accurate` | 3233 | 113 | 24 | 24 | 0.719702 | 0.692820 |
| `train_profile_less_accurate` | 119 | 75 | 15 | 15 | 0.850408 | 0.940119 |
| `test_case` | 133 | 120 | 25 | 25 | 1.094226 | 1.008767 |

The accurate profile has lower PMF per-user test RMSE than the less-accurate profile. In the notebook, their train support, rating distributions, high-rating share, genre entropy, movie popularity, recommendations, and local explanation examples are compared. The available statistics explain the difference partially; the observed error gap is not treated as a universal profile rule.

## Local Explanations

Each audit user has:

- `reports/user_<id>_recommendations.csv`
- `reports/user_<id>_explanations.csv`
- `reports/user_<id>_explanation.png`

The explanation CSV reconstructs each raw PMF recommendation score as:

```text
global_mean
+ user_bias
+ item_bias
+ sum(user_factor[k] * item_factor[k])
```

The validator enforces reconstruction error `<= 1e-5`. Each row also includes the top three latent factor contributions and the nearest highly rated known movie in PMF item-factor space, with cosine similarity and shared genres.

## Streamlit Dashboard

`app.py` reads saved artifacts only. It does not train or tune models at startup.

Dashboard sections:

1. Recommendations
2. Why recommended
3. Model evaluation
4. Global latent factors

The dashboard includes manual user ID input, so invalid or unknown IDs can be audited. Invalid input displays `st.error` and does not raise a traceback.

## Generated Artifacts

Core artifacts:

- `reports/baseline_tuning.json`
- `reports/model_metrics.json`
- `reports/rmse_comparison.png`
- `reports/predicted_vs_actual.png`
- `reports/pmf_convergence.png`
- `reports/user_comparison.png`
- `reports/top_recommendations.png`
- `reports/svd_predictions.npy`
- `reports/svd_metadata.json`
- `reports/pmf_tuning.json`
- `reports/pmf_factors/`

Interpretability and audit artifacts:

- `reports/pmf_factor_interpretation.csv`
- `reports/pmf_factor_genre_profiles.csv`
- `reports/pmf_latent_factor_heatmap.png`
- `reports/pmf_movie_similarities.csv`
- `reports/evaluated_users.json`
- `reports/user_3233_recommendations.csv`
- `reports/user_3233_explanations.csv`
- `reports/user_3233_explanation.png`
- `reports/user_119_recommendations.csv`
- `reports/user_119_explanations.csv`
- `reports/user_119_explanation.png`
- `reports/user_133_recommendations.csv`
- `reports/user_133_explanations.csv`
- `reports/user_133_explanation.png`

## Validation

`scripts/validate_project.py` checks required paths, RMSE targets, benchmark-vs-MF acceptance, PMF grid diagnostics, final refit metadata, raw SVD artifact behavior, mappings, notebook error outputs, app import, recommendation ordering, factor interpretation, similarity sorting, audit-user roles/support, and PMF explanation decomposition.

Synthetic tests cover baseline CF behavior, interpretability helpers, user selection, recommendation ordering, split behavior, model behavior, metrics, and validator helper failures.

## Limitations

The models use collaborative ratings only. They do not solve cold start for new users or unrated movies, optimize ranking metrics directly, or model changing taste over time. PMF latent factors are descriptive analysis tools, not proven semantic dimensions. The selected PMF factor count is at the searched boundary, and a wider future search could be informative, but this audit pass preserves the stable tuning setup.
