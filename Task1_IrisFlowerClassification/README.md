# 🌸 Iris Flower Classification — CodeAlpha Data Science Internship (Task 1)

An end-to-end Machine Learning project that classifies Iris flowers into one of
three species — **Setosa**, **Versicolor**, or **Virginica** — based on four
simple physical measurements, complete with data cleaning, exploratory data
analysis, multi-model comparison, and an interactive Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📌 1. Project Objective (in Simple English)

We want to teach a computer to look at four measurements of an Iris flower
(sepal length, sepal width, petal length, petal width) and correctly guess
which of three species it belongs to — without a human botanist needing to
look at it. This is a classic "hello world" project in machine learning
because the data is small, clean, and the patterns are genuinely learnable.

## 📌 2. Problem Statement

Given a flower's:
- Sepal length (cm)
- Sepal width (cm)
- Petal length (cm)
- Petal width (cm)

...predict its species: **Setosa**, **Versicolor**, or **Virginica**.

This is a **multi-class classification** problem — the model must choose
exactly one label out of three possible categories.

## 📌 3. Business Impact

While the Iris dataset itself is a teaching example, this exact pattern —
"predict a category from a small set of measurements" — has direct real-world
value:

- **Agriculture/Botany:** automated species/variety identification from field
  measurements, reducing dependence on expert manual classification.
- **Quality control:** any manufacturing or agricultural setting that sorts
  physical items into categories based on measurements (e.g. grading produce,
  sorting seeds) can use the same modeling approach.
- **Education & prototyping:** this project demonstrates the full ML
  lifecycle — cleaning, EDA, modeling, evaluation, deployment — in a way
  that generalizes directly to larger, messier, higher-stakes business
  datasets (churn prediction, fraud detection, medical diagnosis, etc.).
  The skills demonstrated here are the same skills used in production ML
  systems.

## 📌 4. Dataset

**Source:** `data/Iris_raw_kaggle.csv` — the
[Kaggle "Iris Species" dataset](https://www.kaggle.com/datasets/uciml/iris),
provided by the project author. This is the same classic Iris data
originally collected by botanist **Edgar Anderson** and made famous by
statistician **Ronald Fisher** in his 1936 paper *"The Use of Multiple
Measurements in Taxonomic Problems,"* also available on the
[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/53/iris).

The raw file has columns `Id, SepalLengthCm, SepalWidthCm, PetalLengthCm,
PetalWidthCm, Species` with labels like `Iris-setosa`.
`src/generate_dataset.py` loads this file, drops the `Id` column, renames
columns to snake_case, and simplifies species labels (`Iris-setosa` →
`setosa`), saving the standardized result to `data/iris.csv`. If the raw
file is ever missing, the script automatically falls back to loading the
same dataset from scikit-learn, so the pipeline never breaks.

**150 rows (raw), 3 species, 50 samples each (balanced dataset).**

| Column             | Type    | Description                                          |
|--------------------|---------|-------------------------------------------------------|
| `sepal_length_cm`  | float   | Length of the flower's sepal (outer petal-like part), in cm |
| `sepal_width_cm`   | float   | Width of the sepal, in cm                             |
| `petal_length_cm`  | float   | Length of the flower's petal, in cm                   |
| `petal_width_cm`   | float   | Width of the petal, in cm                              |
| `species_id`       | int     | Numeric label: 0 = setosa, 1 = versicolor, 2 = virginica |
| `species`          | string  | Human-readable species name (the prediction target)    |

After feature engineering, four more columns are added:
`sepal_area_cm2`, `petal_area_cm2`, `sepal_aspect_ratio`, `petal_aspect_ratio`.

## 📌 5. Project Folder Structure

```
CodeAlpha_IrisFlowerClassification/
│
├── app/
│   └── app.py                  # Streamlit dashboard (4 tabs: Dashboard, EDA, Predict, Insights)
│
├── src/
│   ├── generate_dataset.py     # Loads iris from sklearn -> saves data/iris.csv
│   ├── preprocessing.py        # Missing values, duplicates, outlier handling
│   ├── eda.py                  # Generates all EDA charts
│   ├── feature_engineering.py  # Adds engineered features
│   ├── train_models.py         # Trains 7 models, compares, saves the best
│   └── run_pipeline.py         # Runs the entire pipeline end-to-end
│
├── data/
│   ├── Iris_raw_kaggle.csv      # Original uploaded Kaggle-format file (source of truth)
│   ├── iris.csv                 # Standardized raw dataset
│   ├── iris_cleaned.csv         # After cleaning
│   └── iris_features.csv        # After feature engineering
│
├── models/
│   ├── best_model.pkl           # Trained, saved best model (SVM pipeline)
│   ├── label_encoder.pkl        # Encodes species names <-> numbers
│   └── model_metadata.json      # Which model won, its accuracy, feature list
│
├── assets/charts/                # All PNG charts generated by eda.py & train_models.py
├── docs/                          # Reports, model comparison table, classification report
│
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

## 📌 6. How the Pipeline Works (Source Code Explained)

Run everything with a single command (see **Installation** below), or read
through `src/` in this order to understand it step by step:

1. **`generate_dataset.py`** — loads the Iris dataset from scikit-learn and
   saves it as a real CSV file (`data/iris.csv`), so the project works off
   an actual file like a real-world dataset would.
2. **`preprocessing.py`** — checks for missing values (none found), removes
   1 duplicate row, and detects/caps outliers in `sepal_width_cm` using the
   IQR (Interquartile Range) method. Saves `data/iris_cleaned.csv`.
3. **`eda.py`** — generates 8 professional charts (distribution, correlation
   heatmap, pairplot, boxplots, histograms, scatter, violin plots) to
   `assets/charts/`.
4. **`feature_engineering.py`** — adds 4 new engineered columns (sepal/petal
   area and aspect ratio) to help tree-based models. Saves
   `data/iris_features.csv`.
5. **`train_models.py`** — trains **7 different classification models**
   (Logistic Regression, KNN, SVM, Decision Tree, Random Forest, Naive Bayes,
   Gradient Boosting), evaluates each with accuracy/precision/recall/F1 and
   5-fold cross-validation, picks the winner, and saves it to `models/`.
6. **`app/app.py`** — the Streamlit dashboard that ties it all together with
   filters, KPIs, interactive charts, and a live prediction tool.

Every function in every file has detailed docstrings and inline comments
explaining *what* it does and *why*, written for beginners.

## 📌 7–11. Data Cleaning, EDA, and Feature Engineering — Results

- **Missing values found:** 0
- **Duplicate rows found & removed:** 3 (150 → 147 rows remained). These
  appear only after dropping the `Id` column, since a few rows share
  identical measurements and species once the row-number column is removed.
- **Outliers found:** 4 mild outliers in `sepal_width_cm`, capped using the
  IQR method (Winsorization) rather than deleted, to preserve data.
- **Key EDA finding:** `petal_length_cm` and `petal_width_cm` are by far the
  most powerful predictors — a simple scatter plot of these two features
  almost perfectly separates all three species. Setosa is linearly separable
  from the other two species; Versicolor and Virginica have some overlap.
- **Correlation:** Petal length and petal width are very strongly correlated
  (~0.96), and both are strongly correlated with sepal length. Sepal width is
  only weakly correlated with the other features.

See `assets/charts/` for all charts and `docs/eda_summary_stats.csv` for full
descriptive statistics by species.

## 📌 12–14. Machine Learning Models — Comparison & Results

Seven models were trained and evaluated on an 80/20 train-test split
(stratified by species) plus 5-fold cross-validation:

| Model                   | Test Accuracy | Precision (macro) | Recall (macro) | F1 Score (macro) | CV Mean Accuracy |
|--------------------------|:---:|:---:|:---:|:---:|:---:|
| **Support Vector Machine** ⭐ | **0.967** | 0.970 | 0.967 | 0.967 | 0.966 |
| Gradient Boosting        | 0.967 | 0.970 | 0.967 | 0.967 | 0.966 |
| Random Forest            | 0.967 | 0.970 | 0.967 | 0.967 | 0.957 |
| Logistic Regression      | 0.933 | 0.933 | 0.933 | 0.933 | 0.957 |
| K-Nearest Neighbors      | 0.933 | 0.944 | 0.933 | 0.933 | 0.940 |
| Decision Tree            | 0.933 | 0.933 | 0.933 | 0.933 | 0.940 |
| Naive Bayes              | 0.933 | 0.933 | 0.933 | 0.933 | 0.957 |

*(Exact numbers are reproducible — re-run `python src/run_pipeline.py` and see
`docs/model_comparison_results.csv`.)*

### 🏆 Best Model: Support Vector Machine (RBF kernel)

**Why SVM was selected:**
- Tied for the highest test-set accuracy (96.7%) along with Gradient
  Boosting and Random Forest — but SVM was ranked first as the simplest,
  fastest model among the three ties (a single margin-based classifier
  vs. large tree ensembles), and it matched Gradient Boosting for the best
  cross-validation accuracy (96.6% mean).
- Only **1 misclassification** out of 30 test samples (see the confusion
  matrix), and that single error was between Versicolor and Virginica — the
  only two species with any real overlap in the data.
- SVMs work especially well on small, clean, low-dimensional datasets like
  this one, where finding an optimal separating boundary matters more than
  having a very flexible/complex model (which would risk overfitting on
  only ~150 samples), and it trains/predicts far faster than the ensemble
  alternatives while matching their accuracy.

The trained model is saved at `models/best_model.pkl` (a full scikit-learn
`Pipeline` including the feature scaler, so raw measurements can be fed
directly to it).

## 📌 16–18. Streamlit Dashboard

The app (`app/app.py`) has 4 tabs:

1. **📊 Dashboard** — KPIs (sample count, species count, avg petal length,
   model accuracy), sidebar filters (species multiselect, sepal/petal length
   range sliders), and 4 interactive Plotly charts that respond to filters.
2. **🔬 EDA Explorer** — pick any feature to see boxplots/violin plots by
   species, a correlation heatmap, and a 3D scatter plot.
3. **🤖 Predict** — move sliders to enter new flower measurements and get a
   live species prediction with a confidence bar chart, powered by the saved
   `best_model.pkl`.
4. **💡 Insights** — key findings, the full model comparison table, and
   business recommendations.

## 📌 19. Installation & Usage

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/CodeAlpha_IrisFlowerClassification.git
cd CodeAlpha_IrisFlowerClassification

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the full data + ML pipeline (generates data, charts, and the trained model)
python src/run_pipeline.py

# 5. Launch the Streamlit dashboard
streamlit run app/app.py
```

The app will open automatically at `http://localhost:8501`.

> **Note:** Steps 4 and 5 are independent — `models/best_model.pkl` is
> already included in this repo, so you can skip straight to step 5 if you
> just want to see the dashboard.

## 📌 23. Uploading to GitHub

```bash
git init
git add .
git commit -m "Initial commit: Iris Flower Classification project"
git branch -M main
git remote add origin https://github.com/<your-username>/CodeAlpha_IrisFlowerClassification.git
git push -u origin main
```

## 📌 24. Deploying the Streamlit App (Free, via Streamlit Community Cloud)

1. Push this project to a **public** GitHub repository (see commands above).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with
   GitHub.
3. Click **"New app"**, select your repository, branch (`main`), and set the
   main file path to `app/app.py`.
4. Click **Deploy**. Streamlit Cloud will install `requirements.txt`
   automatically and give you a public URL (e.g.
   `https://your-app-name.streamlit.app`) to share on LinkedIn/your resume.

## 🛠️ Tech Stack

- **Python 3.10+**
- **pandas / numpy** — data manipulation
- **scikit-learn** — machine learning models & evaluation
- **matplotlib / seaborn** — static EDA charts
- **plotly** — interactive dashboard charts
- **Streamlit** — web application framework
- **joblib** — model serialization

## 📚 References & Documentation

- Fisher, R.A. (1936). *The Use of Multiple Measurements in Taxonomic
  Problems.* Annals of Eugenics, 7(2), 179–188.
- [UCI Machine Learning Repository — Iris Dataset](https://archive.ics.uci.edu/dataset/53/iris)
- [Kaggle — Iris Species Dataset](https://www.kaggle.com/datasets/uciml/iris)
- [scikit-learn documentation](https://scikit-learn.org/stable/documentation.html)
- [scikit-learn: `load_iris`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html)
- [Streamlit documentation](https://docs.streamlit.io/)
- [Plotly Python documentation](https://plotly.com/python/)
- [pandas documentation](https://pandas.pydata.org/docs/)

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for
details.

---
*Built as part of the CodeAlpha Data Science Internship — Task 1.*
