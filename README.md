# Matrix Factorization для MovieLens 1M

Рекомендательная система на MovieLens 1M с двумя отдельными задачами оценки:

- **rating prediction** - точность предсказания числовой оценки;
- **Top-K ranking** - положение будущего положительного объекта среди непросмотренных кандидатов.

В проекте реализованы четыре локальные модели:

- `BiasBaseline` - регуляризованный baseline из глобального среднего, user bias и item bias;
- `SVD` - усечённая факторизация разреженной residual-матрицы;
- `PMF` - вероятностная матричная факторизация с локально реализованным SGD;
- `ItemKNN` - neighborhood collaborative filtering по residual ratings.

Notebook и Streamlit используют сохранённые артефакты и не переобучают модели при отображении.

## 📋 Содержание

- [Запуск](#-запуск)
- [Данные и лицензия](#-данные-и-лицензия)
- [Структура проекта](#-структура-проекта)
- [EDA](#-eda)
- [Rating prediction](#-rating-prediction)
- [Модели](#-модели)
- [MSE и RMSE](#-mse-и-rmse)
- [Benchmark для RMSE](#-benchmark-для-rmse)
- [Top-K ranking](#-top-k-ranking)
- [Rating accuracy и ranking quality](#-rating-accuracy-и-ranking-quality)
- [Showcase-пользователи](#-showcase-пользователи)
- [Интерпретация](#-интерпретация)
- [Streamlit](#-streamlit)
- [Проверка проекта](#-проверка-проекта)
- [Ограничения](#-ограничения)
- [🧑‍💻 Автор](#-автор)

## 🚀 Запуск

Среда разработки - Windows и Git Bash. Путь `/d/...` соответствует `D:\...`.

```bash
git clone https://01.tomorrow-school.ai/git/nyestaye/matrix-factorization
cd matrix-factorization

python -m venv .venv
source .venv/Scripts/activate

python -m pip install -r requirements.txt
python -m scripts.run_pipeline
python -m scripts.build_analysis_notebook
python -m jupyter nbconvert \
  --to notebook \
  --execute Movie_Recommender_System.ipynb \
  --output Movie_Recommender_System.ipynb
```

Проверка:

```bash
python -m pytest -q
python -m compileall models utils scripts app.py
python -m scripts.validate_project
bash scripts/smoke_streamlit.sh
```

Запуск интерфейса:

```bash
python -m streamlit run app.py
```

## 📦 Данные и лицензия

Проект использует стабильный набор [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/).

Исходные файлы не хранятся в репозитории:

```text
data/ratings.dat
data/users.dat
data/movies.dat
```

Условия MovieLens не разрешают публичное перераспространение набора без отдельного разрешения. При отсутствии файлов pipeline загружает официальный архив GroupLens и распаковывает его локально.

```bash
python -m scripts.run_pipeline
```

Крупные воспроизводимые артефакты также не хранятся в Git:

```text
processed/user_item_matrix.csv
reports/svd_predictions.npy
```

Они создаются pipeline и проверяются validator после выполнения полного процесса.

Цитирование набора:

> F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems, 5(4), Article 19. https://doi.org/10.1145/2827872

## 📁 Структура проекта

```text
matrix-factorization/
├── audit/                             # официальный аудит-лист 01-edu
│   └── README.md   
├── data/                              # Локальные raw-файлы MovieLens
├── processed/                         # Splits, mappings и ranking data
├── models/                            # BiasBaseline, ItemKNN, SVD, PMF
├── utils/                             # Метрики, ranking, explanations, plots
├── reports/                           # Сгенерированные отчёты и графики
├── scripts/                           # Pipeline, notebook builder, validator
├── tests/                             # Unit и protocol tests
├── Movie_Recommender_System.ipynb     # Выполненный аналитический notebook
├── app.py                             # Streamlit dashboard
├── README.md
└── requirements.txt
```

## 🔎 EDA

EDA считается непосредственно из локальных raw-файлов MovieLens и используется только для описания данных.

Основные свойства:

- `1,000,209` оценок;
- `6,040` пользователей;
- `3,706` фильмов;
- разреженность user-item матрицы около `0.957`;
- медианная активность пользователя - `96` оценок;
- медианная популярность фильма - `123` оценки;
- распределение оценок смещено к высоким значениям;
- активность пользователей неоднородна;
- популярность фильмов имеет длинный хвост.

Дополнительные срезы:

- распределение оценок во времени;
- покрытие фильмов и оценок по жанрам;
- пол, возрастные группы и профессии пользователей.

Фильмы могут относиться к нескольким жанрам, поэтому после `explode` жанровые количества не являются взаимоисключающими долями.

Жанры и демография не используются как признаки моделей:

- жанры применяются только для post-hoc интерпретации, genre entropy и latent factor profiles;
- возраст, пол и профессия показываются только как контекст набора;
- модели остаются collaborative-only.

Полный raw dataset используется только для descriptive EDA. Выбор гиперпараметров, stopping decisions и сравнение моделей выполняются по заранее заданным train и validation splits. Test split не используется до финальной оценки.

## 🎯 Rating prediction

Основной протокол использует детерминированный interaction split:

```text
70% train
15% validation
15% test
random_state = 42
```

Правила:

- split выполняется внутри истории каждого пользователя;
- validation используется для выбора гиперпараметров и early stopping;
- финальные модели переобучаются на `train + validation`;
- все модели оцениваются на одинаковых test rows;
- test не используется для выбора модели;
- predictions ограничиваются диапазоном `[1, 5]` только для rating metrics и отображения;
- raw scores сохраняются для recommendation ordering.

## 🧠 Модели

### BiasBaseline

```text
prediction = global_mean + user_bias + item_bias
```

Регуляризованный baseline учитывает:
- общий уровень оценок;
- склонность пользователя ставить выше или ниже среднего;
- систематическое смещение конкретного фильма.

Такая постановка соответствует стандартным baseline estimates в collaborative filtering, включая формулировки Yehuda Koren и алгоритм [`BaselineOnly`](https://surprise.readthedocs.io/en/stable/basic_algorithms.html) в Surprise.

Регуляризация user и item biases выбирается по validation RMSE.

### SVD

SVD:
- центрирует наблюдаемые оценки по пользователям;
- оценивает регуляризованный item residual bias;
- факторизует sparse residual matrix через `scipy.sparse.linalg.svds`;
- сохраняет raw predictions для ranking.

Выбранная конфигурация:

```text
n_factors = 20
item_bias_regularization = 5.0
random_state = 42
```

SVD выполняется прямым truncated decomposition и не имеет epoch-based learning curve. Графики `SVD rank tuning` показывают validation MSE и RMSE для проверенных значений rank и фиксируют выбор `rank = 20`.

### PMF

PMF предсказывает:

```text
global_mean
+ user_bias
+ item_bias
+ dot(user_factors, item_factors)
```

Реализация включает:
- локальный seeded SGD;
- детерминированное перемешивание;
- отдельную регуляризацию biases и factors;
- validation-only early stopping;
- восстановление лучшего checkpoint;
- финальный refit на `train + validation`.

Выбранная конфигурация:

```text
n_factors = 128
learning_rate = 0.006
factor_regularization = 0.06
bias_regularization = 0.02
selected_epoch = 53
random_state = 42
```

PMF convergence показывается отдельно в MSE и RMSE с train и validation curves.

### ItemKNN

`ItemKNN` служит дополнительным сильным neighborhood-CF reference.

Алгоритм:
1. Обучает выбранный `BiasBaseline`.
2. Вычитает baseline predictions из наблюдаемых ratings.
3. Считает item-item cosine similarity по residual-векторам.
4. Применяет significance shrinkage.
5. Требует минимум три общих пользователя.
6. Использует signed similarity в numerator и absolute similarity в denominator.
7. При отсутствии подходящих соседей возвращается к bias prediction.

Параметры выбираются по validation RMSE:

```text
k ∈ {20, 40, 80}
shrinkage ∈ {10, 50, 100}
```

ItemKNN не является порогом задания. Его задачи:
- проверить, превосходит ли PMF не только простой bias baseline, но и настроенный классический CF;
- показать, что rating accuracy и Top-K ranking могут давать разный порядок моделей.


## 📐 MSE и RMSE

MSE усредняет квадраты ошибок:

```text
MSE = mean((actual - predicted)²)
```

RMSE является корнем из MSE и выражен в исходной шкале рейтингов:

```text
RMSE = sqrt(MSE)
```

Для обеих метрик меньшее значение означает меньшую ошибку.

Финальные результаты на test split:

| Модель       | Test MSE | Test RMSE |
|--------------|---------:|----------:|
| PMF          | 0.712165 | 0.843899  |
| ItemKNN      | 0.737614 | 0.858845  |
| SVD          | 0.793518 | 0.890796  |
| BiasBaseline | 0.824119 | 0.907810  |

Требования задания выполнены:

```text
SVD RMSE <= 0.90
PMF RMSE <= 0.85
PMF improvement over SVD >= 5%
```

Фактическое улучшение PMF относительно SVD - `5.265%`.

## 📏 Benchmark для RMSE

Benchmark для audit-проверки rating prediction - `BiasBaseline`.

Это стандартный регуляризованный bias-based collaborative predictor:

```text
global_mean + user_bias + item_bias
```

Обе обязательные модели матричной факторизации превосходят его:
- SVD улучшает RMSE на `1.874%`;
- PMF улучшает RMSE на `7.040%`.


`ItemKNN` используется отдельно как более сильный neighborhood-CF sanity check, а не как acceptance threshold:
- PMF лучше ItemKNN по RMSE на `1.740%`;
- SVD хуже ItemKNN по RMSE на `3.720%`.

Один из возможных методологических факторов состоит в том, что truncated `svds` аппроксимирует разреженную residual-матрицу, где ненаблюдаемые residual entries представлены нулями, тогда как SGD-based PMF оптимизирует ошибку только по наблюдаемым ratings. Это возможное объяснение, а не установленный причинный механизм.

## 🏁 Top-K ranking

Ranking оценивается отдельно от rating prediction.

Протокол:

```text
temporal leave-one-positive-out
```

Для каждого подходящего пользователя:
1. Выбирается последнее взаимодействие с rating `>= 4.0`.
2. При одинаковом timestamp используется movie ID ascending.
3. В ranking history остаются только строки с `timestamp < target_timestamp`.
4. Требуется минимум `20` prior interactions.
5. Target movie должен иметь минимум `10` interactions в ranking-training data.
6. Candidate set содержит весь поддерживаемый каталог за вычетом strict temporal history пользователя.
7. Sampled negatives не используются.
8. Target остаётся в candidate set.
9. Tie-break при одинаковом score - movie ID ascending.

SVD и PMF для ranking обучаются отдельными frozen copies. Ranking targets не используются для tuning.

Текущий протокол оценивает `5,767` пользователей.

| Модель       | HitRate@5 | HitRate@10 | NDCG@10  | MRR@10   | Mean rank | Median rank |
|--------------|----------:|-----------:|---------:|---------:|----------:|------------:|
| BiasBaseline | 0.004682  | 0.013178   | 0.004714 | 0.002291 | 1148.59   | 934         |
| ItemKNN      | 0.002601  | 0.005375   | 0.002502 | 0.001645 | 1084.28   | 845         |
| SVD          | 0.019421  | 0.030692   | 0.015065 | 0.010313 | 1097.46   | 796         |
| PMF          | 0.014392  | 0.026704   | 0.012976 | 0.008912 | 975.60    | 696         |

Unknown unseen movies не считаются observed negatives. Протокол измеряет, насколько высоко модель поднимает один известный будущий positive item.

## 🔄 Rating accuracy и ranking quality

Главный результат проекта - порядок моделей зависит от целевой функции.

По RMSE:

```text
PMF > ItemKNN > SVD > BiasBaseline
```

Здесь знак `>` означает лучшее качество, то есть меньший RMSE.

По HitRate@10:

```text
SVD > PMF > BiasBaseline > ItemKNN
```

Наиболее показательные изменения:

```text
ItemKNN: #2 по RMSE -> #4 по HitRate@10
SVD:     #3 по RMSE -> #1 по HitRate@10
```

ItemKNN особенно важен для этой демонстрации: он силён при предсказании rating, но хуже остальных возвращает target в первые десять позиций.

PMF:
- лучший по rating MSE и RMSE;
- имеет лучший mean и median target rank;
- лучше SVD на большинстве распределения target ranks;
- имеет более лёгкий deep tail.

SVD:
- лучший по HitRate@5 и HitRate@10;
- сильнее в extreme head списка;
- уступает PMF по typical rank и глубокой части распределения.

Текущие артефакты фиксируют это расхождение, но не доказывают его причинный механизм.

## 👤 Showcase-пользователи

Сохраняются три роли:

| Роль                          | User | Target                   | PMF rank | PMF Hit@10 |
|-------------------------------|-----:|--------------------------|---------:|-----------:|
| `train_profile_accurate`      | 2739 | Sixth Sense, The (1999)  | 6        | true       |
| `train_profile_less_accurate` | 2505 | Santa Clause, The (1994) | 736      | false      |
| `test_case`                   | 2210 | Contender, The (2000)    | 696      | false      |

Роли определяются результатом temporal ranking, а не per-user RMSE:

- `accurate` - PMF Hit@10;
- `less_accurate` - PMF miss;
- `test_case` - отдельный пользователь около медианного PMF target rank.

`test_case` не означает cold-start пользователя. Split выполняется по interactions, поэтому пользователь 2210 присутствует в train history, а оценивается его held-out future interaction. Cold start не входит в scope проекта.

Из-за различия целей per-user RMSE может выглядеть контринтуитивно:

- у пользователя 2739 PMF попадает в Top-10, хотя его per-user RMSE хуже SVD;
- у пользователя 2505 SVD имеет низкий per-user RMSE, но PMF не поднимает target в Top-10.

Для каждого showcase-пользователя создаются:

```text
reports/user_<id>_recommendations.csv
reports/user_<id>_explanations.csv
reports/user_<id>_explanation.png
reports/user_<id>_ranking_case.csv
reports/user_<id>_ranking_case.png
```

Pipeline удаляет устаревшие `user_<id>_*` и сохраняет артефакты только для пользователей из `reports/evaluated_users.json`.

## 🧩 Интерпретация

Глобальная интерпретация PMF включает:

- item factors с наибольшей дисперсией;
- фильмы на положительном и отрицательном полюсах факторов;
- genre profiles;
- heatmap latent factors;
- cosine similarity между item-factor vectors.

Локальное объяснение PMF раскладывает score на:

```text
global mean
+ user bias
+ item bias
+ latent dot product
```

Latent factors и локальные decompositions являются описательными. Они не доказывают причинные предпочтения пользователя.

## 🖥️ Streamlit

`app.py` читает сохранённые артефакты и не обучает модели.

Интерфейс показывает:
- MSE и RMSE четырёх моделей;
- PMF convergence;
- SVD rank tuning;
- Top-K ranking metrics;
- showcase cases;
- рекомендации и локальные explanations.

Ручной ввод user ID остаётся редактируемым. Некорректный текст и неизвестный ID обрабатываются через `st.error` без traceback.

## 🧪 Проверка проекта

Pipeline и validator проверяют:
- детерминированный rating split;
- validation-only tuning и final refit;
- согласованность MSE и RMSE;
- обязательные SVD и PMF artifacts;
- runtime-generated large artifacts;
- temporal ranking protocol;
- candidate sets и target ranks;
- deterministic tie-breaking;
- PMF score decomposition;
- полный набор showcase artifacts;
- отсутствие orphan `user_<id>_*`;
- выполненный notebook без error outputs;
- импорт Streamlit app.

Основные команды:

```bash
python -m pytest -q
python -m compileall models utils scripts app.py
python -m scripts.validate_project
bash scripts/smoke_streamlit.sh
```

## ⚠️ Ограничения
- Модели используют только collaborative ratings.
- Cold start не рассматривается.
- Rating prediction и Top-K ranking отвечают на разные вопросы.
- Temporal ranking держит один известный future positive и не содержит истинных negative labels.
- Отсутствие interaction не означает, что фильм пользователю не нравится.
- Latent factors интерпретируются post-hoc и не имеют гарантированного семантического значения.
- Причины различий между RMSE и ranking требуют отдельной диагностики.
- Выбор production model зависит от целевой функции: PMF сильнее для rating accuracy и typical rank, SVD - для extreme-head retrieval.

## 🧑‍💻 Автор
- Nazar Yestayev (@nyestaye)