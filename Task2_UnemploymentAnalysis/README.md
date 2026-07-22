# 📉 Unemployment Analysis in India — CodeAlpha Data Science Internship (Task 2)

An end-to-end data analysis and machine learning project exploring how the
COVID-19 lockdown affected unemployment across Indian states, with a
regression model that predicts unemployment rate and an interactive
Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📌 1. Project Objective (in Simple English)

We want to understand how the COVID-19 lockdown affected jobs in India —
which states were hit hardest, how urban and rural areas differed, and
whether we can predict the unemployment rate from other known economic
indicators.

## 📌 2. Problem Statement

Given monthly unemployment data for Indian states (unemployment rate,
number of people employed, labour force participation rate, urban/rural
split), analyze how these numbers changed before and during the COVID-19
lockdown, and build a model to predict the unemployment rate from related
indicators.

## 📌 3. Business Impact

- **Policy targeting:** identifying which states and periods saw the
  sharpest unemployment spikes helps direct relief programs where they're
  needed most, rather than applying one-size-fits-all national policy.
- **Economic early-warning:** a working unemployment-rate predictor can
  help estimate the likely effect of future disruptions (new lockdowns,
  economic shocks) before official data catches up.
- **General applicability:** the same "clean → analyze → model → predict"
  workflow used here applies directly to other economic and social
  indicator datasets in government, NGO, or corporate analytics settings.

## 📌 4. Dataset

**Source:** ["Unemployment in India" dataset on Kaggle](https://www.kaggle.com/datasets/gokulrajkmv/unemployment-in-india),
compiled from the Centre for Monitoring Indian Economy (CMIE)
(https://unemploymentinindia.cmie.com/).

The dataset ships as **two CSV files**, which this project cleans and
merges into one:

| File | Coverage | Adds |
|---|---|---|
| `Unemployment in India.csv` | May 2019 – Jun 2020 | Rural/Urban area split |
| `Unemployment_Rate_upto_11_2020.csv` | Jan – Oct 2020 | Zone (region group) + longitude/latitude |

**Columns (standardized names used throughout this project):**

| Column | Description |
|---|---|
| `state` | Indian state or union territory |
| `date` | Date the observation was recorded (monthly) |
| `frequency` | Measurement frequency (Monthly) |
| `unemployment_rate` | % of the labour force that is unemployed |
| `employed_count` | Estimated number of people employed |
| `labour_participation_rate` | % of eligible population actively in the labour force |
| `area` | Rural / Urban (only present in file 1) |
| `zone` | Geographic zone e.g. South, North, East (only present in file 2) |
| `longitude`, `latitude` | State coordinates (only present in file 2) |

After cleaning: **1,007 rows** covering **28 states**, May 2019 – October 2020.

## 📌 5. Project Folder Structure

```
Task2_UnemploymentAnalysis/
│
├── app/
│   └── app.py                  # Streamlit dashboard (Dashboard, COVID Impact, Predict, Insights)
│
├── src/
│   ├── preprocessing.py        # Loads, cleans & merges both raw CSVs
│   ├── eda.py                  # Generates all EDA charts
│   ├── feature_engineering.py  # Adds lockdown flag, state encoding
│   ├── train_models.py         # Trains 5 regression models, compares, saves the best
│   └── run_pipeline.py         # Runs the entire pipeline end-to-end
│
├── data/
│   ├── Unemployment_in_India_raw.csv
│   ├── Unemployment_Rate_upto_11_2020_raw.csv
│   ├── unemployment_cleaned.csv
│   └── unemployment_features.csv
│
├── models/
│   ├── best_model.pkl
│   └── model_metadata.json
│
├── assets/charts/                # 11 PNG charts from EDA & model evaluation
├── docs/                          # Reports, model comparison, summary stats
│
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

## 📌 6. How the Pipeline Works

Run everything with one command (see Installation below), or read `src/`
in this order:

1. **`preprocessing.py`** — loads both raw CSVs, strips whitespace from
   column names/values, standardizes column names, drops 28 fully-blank
   junk rows, parses dates, removes duplicates, and caps extreme outliers
   in `unemployment_rate` and `labour_participation_rate` using a widened
   IQR method (3.0×IQR instead of the standard 1.5×, since COVID-era
   spikes are real signal, not data errors). Merges both files into
   `data/unemployment_cleaned.csv`.
2. **`eda.py`** — generates 8 charts: national trend, top-10 states,
   urban vs rural, COVID before/after boxplot, zone comparison,
   correlation heatmap, state×month heatmap, labour participation trend.
3. **`feature_engineering.py`** — adds a binary lockdown-period flag,
   fills missing Area values, and target-encodes each state by its
   historical average unemployment rate.
4. **`train_models.py`** — trains **5 regression models** (Linear
   Regression, Ridge, Decision Tree, Random Forest, Gradient Boosting),
   evaluates with MAE/RMSE/R² and 5-fold cross-validation, saves the best.
5. **`app/app.py`** — the Streamlit dashboard tying it all together.

## 📌 7–11. Data Cleaning, EDA & Feature Engineering — Results

- **Missing values:** 28 fully-blank trailing rows in file 1, dropped.
- **Duplicates:** 0 found after merging (checked on state+date+area).
- **Outliers:** 11 extreme values capped in `unemployment_rate`, 14 in
  `labour_participation_rate`, using a widened IQR threshold to preserve
  genuine COVID-era spikes while still catching data entry errors.
- **Key EDA finding — the headline number:** national average
  unemployment rate rose from **9.48%** before the 25 March 2020 lockdown
  to **14.92%** during/after it — a **57.3% relative increase**, peaking
  above 23% in June 2020 before gradually recovering.
- Urban and rural areas were both affected, with some differences in
  timing and magnitude (see dashboard chart).
- Labour force participation also dropped during lockdown, meaning the
  unemployment spike likely *understates* the true economic disruption.

## 📌 12–14. Machine Learning Models — Comparison & Results

Five regression models were trained on an 80/20 split with 5-fold
cross-validation, predicting `unemployment_rate` from labour
participation rate, month, year, lockdown-period flag, and each state's
historical average unemployment rate:

| Model | R² Score | MAE | RMSE | CV Mean R² |
|---|:---:|:---:|:---:|:---:|
| **Random Forest** ⭐ | **0.703** | 3.42 | 5.06 | 0.674 |
| Decision Tree | 0.662 | 3.61 | 5.40 | 0.569 |
| Gradient Boosting | 0.607 | 3.91 | 5.82 | 0.581 |
| Ridge Regression | 0.469 | 4.62 | 6.77 | 0.435 |
| Linear Regression | 0.469 | 4.62 | 6.77 | 0.435 |

*(Reproducible — re-run `python src/run_pipeline.py`; see `docs/model_comparison_results.csv`.)*

### 🏆 Best Model: Random Forest Regressor

**Why Random Forest was selected:**
- Highest R² score (0.70) — explains 70% of the variance in unemployment
  rate, far ahead of the linear models (~0.47), showing the relationship
  between these indicators and unemployment is meaningfully non-linear.
- Best cross-validation performance (0.674 mean R²), confirming it
  generalizes rather than overfitting to one split.
- Handles the mix of a categorical-like encoded feature (state average)
  and continuous/temporal features naturally, without needing extensive
  preprocessing.

**Honest limitation:** with only 5 features and ~1,000 rows spanning a
single unprecedented economic shock, R²=0.70 is a solid but not perfect
result — unemployment is affected by many factors outside this dataset
(sector mix, local policy, informal economy size). See Future
Improvements for how this could be strengthened.

## 📌 16–18. Streamlit Dashboard

The app (`app/app.py`) has 4 tabs:

1. **📊 Dashboard** — KPIs, sidebar filters (state, date range, area),
   time series, top-10 states, urban vs rural, labour participation trend.
2. **🦠 COVID Impact** — before/after lockdown comparison metrics,
   boxplot, zone-wise comparison, full state×month heatmap.
3. **🤖 Predict** — select a state and indicators to get a live
   unemployment rate prediction from the saved Random Forest model.
4. **💡 Insights** — key findings, full model comparison table, and
   policy recommendations.

## 📌 19. Installation & Usage

```bash
git clone https://github.com/<your-username>/codealpha_tasks.git
cd codealpha_tasks/Task2_UnemploymentAnalysis

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

python src/run_pipeline.py      # regenerates data, charts, and the trained model

streamlit run app/app.py
```

The app opens at `http://localhost:8501`. Steps 4-5 can be skipped since
`models/best_model.pkl` is already included in this repo.

## 📌 24. Deploying the Streamlit App

1. Push this project to your GitHub repo (see main repo README for
   general Git/GitHub instructions).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with
   GitHub.
3. Click **New app**, select the repo, set the main file path to
   `Task2_UnemploymentAnalysis/app/app.py`.
4. Click **Deploy**.

## 🛠️ Tech Stack

Python, pandas, numpy, scikit-learn, matplotlib, seaborn, plotly,
Streamlit, joblib.

## 📚 References & Documentation

- [Kaggle — Unemployment in India dataset](https://www.kaggle.com/datasets/gokulrajkmv/unemployment-in-india)
- [CMIE — Centre for Monitoring Indian Economy](https://unemploymentinindia.cmie.com/)
- [scikit-learn documentation](https://scikit-learn.org/stable/documentation.html)
- [Streamlit documentation](https://docs.streamlit.io/)
- [Plotly Python documentation](https://plotly.com/python/)
- [pandas documentation](https://pandas.pydata.org/docs/)

## 📄 License

MIT License — see [LICENSE](LICENSE).

---
*Built as part of the CodeAlpha Data Science Internship — Task 2.*
