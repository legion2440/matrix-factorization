# MovieLens 1M Matrix Factorization

This project builds a reproducible MovieLens 1M recommender with four local models:

- Bias baseline: `global_mean + user_bias + item_bias`.
- Item-kNN: residualized neighborhood collaborative filtering with shrunk cosine similarity.
- SVD: truncated sparse residual factorization with a regularized item residual bias.
- PMF: biased matrix factorization trained by locally implemented seeded SGD.

Generated artifacts power the notebook and Streamlit dashboard without retraining at display time.

## Clean-Clone Order

Use Windows Git Bash from the project root:

```bash
cd /d/TSchool/matrix-factorization
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements.txt
python -m scripts.run_pipeline
python -m scripts.build_analysis_notebook
python -m jupyter nbconvert --to notebook --execute Movie_Recommender_System.ipynb --output Movie_Recommender_System.ipynb
python -m pytest -q
python -m compileall models utils scripts app.py
python -m scripts.validate_project
bash scripts/smoke_streamlit.sh
```

The pipeline downloads MovieLens 1M only when the raw files are missing.

## Project Structure

- `data/`: raw MovieLens files.
- `processed/`: rating-prediction splits, temporal ranking data, mappings, and normalized matrix data.
- `models/`: local BiasBaseline, ItemKNN, SVD, and PMF implementations.
- `utils/`: data, evaluation, recommendation, artifact, and interpretation helpers.
- `reports/`: generated metrics, plots, recommendations, and explanations.
- `scripts/`: pipeline, validation, notebook builder, and bounded Streamlit smoke test.
- `tests/`: synthetic model, protocol, validator, and Streamlit behavior tests.
- `Movie_Recommender_System.ipynb`: executed artifact-backed analysis.
- `app.py`: artifact-backed Streamlit dashboard.

## Rating Prediction Evaluation

The authoritative pointwise rating protocol uses the unchanged deterministic 70/15/15 interaction split with `random_state=42`.

- Hyperparameters and stopping decisions use validation rows only.
- BiasBaseline, ItemKNN, SVD, and PMF are refit on train plus validation.
- MSE and RMSE are computed once on the same untouched test rows.
- Predictions are clipped to `[1, 5]` only for rating evaluation and display.
- Raw scores remain available for recommendation ordering.

RMSE measures pointwise rating-prediction error. It does not measure Top-K recommendation accuracy.

### Bias Baseline

`models/bias_baseline.py` implements the regularized ablation:

```text
prediction = global_mean + user_bias + item_bias
```

User and item regularization are selected from `[1, 2, 5, 10, 20, 40, 80]` using validation RMSE. The selected value is data-driven and stored in `reports/bias_baseline_tuning.json`.

### Item-kNN

`models/item_knn.py` is the classical neighborhood collaborative-filtering reference. It first fits the selected bias baseline and computes observed residuals:

```text
residual_ui = rating_ui - bias_baseline_ui
```

Item similarity is shrunk cosine:

```text
similarity_ij =
    cosine(residual_vector_i, residual_vector_j)
    * common_users_ij / (common_users_ij + shrinkage)
```

Pairs require at least three common users. A prediction adds a signed residual correction over the globally selected top-k item neighbors that the user rated:

```text
bias_baseline_ui
+ sum(similarity_ij * residual_uj) / sum(abs(similarity_ij))
```

The grid is `k in [20, 40, 80]` and `shrinkage in [10, 50, 100]`. Neighbor ordering is deterministic: absolute similarity descending, signed similarity descending, then movie ID ascending. Missing usable neighbors fall back exactly to the bias prediction.

### SVD

SVD user-centers observed ratings, estimates a regularized item residual bias, and factorizes the sparse residual matrix with `scipy.sparse.linalg.svds`.

The preserved selected parameters are:

- factors: 20
- item-bias regularization: 5.0
- random state: 42

### PMF

PMF predicts:

```text
global_mean + user_bias + item_bias + dot(user_factors, item_factors)
```

The preserved selected configuration is:

- factors: 128
- learning rate: 0.006
- factor regularization: 0.06
- bias regularization: 0.02
- selected epoch: 53
- random state: 42

Tuning uses validation-only early stopping. The final refit uses exactly the selected epoch count on train plus validation.

Actual test metrics and pairwise booleans are stored in `reports/model_metrics.json`. No model is declared better than ItemKNN unless the generated RMSE values prove it.

Current generated rating results:

| Model | Test MSE | Test RMSE |
|---|---:|---:|
| BiasBaseline | 0.824119 | 0.907810 |
| ItemKNN | 0.737614 | 0.858845 |
| SVD | 0.793518 | 0.890796 |
| PMF | 0.712165 | 0.843899 |

ItemKNN improves on BiasBaseline by 5.394% RMSE. SVD does not beat ItemKNN in this run; its RMSE is 3.720% worse. PMF improves on ItemKNN by 1.740% and on SVD by 5.265%.

## Top-K Recommendation Evaluation

Top-K quality uses a separate temporal leave-one-positive-out protocol. It does not reuse the 70/15/15 test rows as ranking positives.

For each eligible user:

1. Select the latest interaction with rating at least 4.0.
2. Break latest-timestamp ties by movie ID ascending.
3. Keep only history with `timestamp < target_timestamp`.
4. Require at least 20 prior interactions.
5. Require at least 10 ranking-training interactions for the target movie.

Separate frozen copies of all four models are trained on `processed/ranking_train_ratings.csv`. SVD and PMF are not retuned on ranking targets.

The candidate set is the full ranking-training-supported catalog minus the user's temporal-prefix history. No sampled negatives are used. The held-out target remains a candidate.

The protocol reports:

- HitRate@5 and HitRate@10
- NDCG@5 and NDCG@10
- MRR@5 and MRR@10
- mean and median target rank

This evaluation measures next-positive recovery under temporal leave-one-positive-out. It asks whether one known future positive is ranked highly among unseen candidates. Unknown catalog items are not observed negatives, so the protocol does not prove that every other unseen movie is irrelevant.

Generated ranking artifacts:

- `processed/ranking_train_ratings.csv`
- `processed/ranking_targets.csv`
- `reports/ranking_protocol.json`
- `reports/ranking_metrics.json`
- `reports/ranking_results.csv`
- `reports/ranking_comparison.png`

The current protocol evaluates 5,767 eligible users. Generated ranking results:

| Model | HitRate@5 | HitRate@10 | NDCG@5 | NDCG@10 | MRR@5 | MRR@10 | Mean rank | Median rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BiasBaseline | 0.004682 | 0.013178 | 0.002093 | 0.004714 | 0.001283 | 0.002291 | 1148.59 | 934 |
| ItemKNN | 0.002601 | 0.005375 | 0.001599 | 0.002502 | 0.001269 | 0.001645 | 1084.28 | 845 |
| SVD | 0.019421 | 0.030692 | 0.011453 | 0.015065 | 0.008841 | 0.010313 | 1097.46 | 796 |
| PMF | 0.014392 | 0.026704 | 0.009064 | 0.012976 | 0.007338 | 0.008912 | 975.60 | 696 |

SVD has the strongest HitRate@10, NDCG@10, and MRR@10 in this run. PMF has the best mean and median target rank. These outcomes are reported independently from rating RMSE.

## User Case Studies

The persisted roles remain:

- `train_profile_accurate`
- `train_profile_less_accurate`
- `test_case`

Their meaning is now based on temporal ranking outcomes:

- The accurate profile is a non-extreme PMF Hit@10 near the median hit rank.
- The less-accurate profile is a PMF miss near the median miss rank.
- The test case is a distinct user near the overall median PMF target rank.

Per-user SVD and PMF test RMSE remain secondary diagnostics. They do not determine the hit/miss labels.

Current selected cases:

| Role | User | Target | PMF rank | PMF Hit@10 |
|---|---:|---|---:|---:|
| `train_profile_accurate` | 2739 | Sixth Sense, The (1999) | 6 | true |
| `train_profile_less_accurate` | 2505 | Santa Clause, The (1994) | 736 | false |
| `test_case` | 2210 | Contender, The (2000) | 696 | false |

Each selected user has production recommendation explanations and a target-specific ranking case:

- `reports/user_<id>_recommendations.csv`
- `reports/user_<id>_explanations.csv`
- `reports/user_<id>_explanation.png`
- `reports/user_<id>_ranking_case.csv`
- `reports/user_<id>_ranking_case.png`

The ranking case reconstructs the ranking-copy PMF target score within `1e-5`, records all four target ranks and scores, and identifies the nearest highly rated movie from the strict temporal prefix.

## Streamlit Dashboard

`app.py` reads saved artifacts only. It does not train or tune models.

The evaluation-profile selectbox uses an explicit key and an `on_change` callback that writes the selected user ID into `st.session_state["user_id_input"]`. The keyed manual input remains editable. Invalid text and unknown IDs produce `st.error` without a traceback.

The Model Evaluation tab separates:

- Rating prediction: four-model MSE/RMSE and the RMSE plot.
- Top-K next-positive recovery: HitRate@10, NDCG@10, MRR@10, protocol context, and the selected ranking case.

## Validation

`scripts/validate_project.py` checks:

- required naming and stale identifier removal;
- the unchanged deterministic rating split;
- BiasBaseline and ItemKNN tuning/refit metadata;
- honest RMSE comparison fields;
- SVD/PMF artifacts and recommendation ordering;
- temporal prefix, target support, full-catalog candidates, ranks, and exact rank-derived metrics;
- ranking-based case selection and PMF target decomposition;
- notebook execution outputs and Streamlit import.

Synthetic tests cover BiasBaseline, ItemKNN, ranking protocol construction, full-catalog ranking math, case selection, validator failures, and Streamlit session-state behavior.

## Limitations

The models use collaborative ratings only and do not solve cold start. RMSE and ranking metrics answer different questions. The temporal protocol holds out one known future positive and does not observe true negatives. PMF latent factors are descriptive rather than proven semantic dimensions. The selected PMF factor count remains at the searched boundary.
