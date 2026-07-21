# Project Report: Iris Flower Classification

**Prepared for:** CodeAlpha Data Science Internship — Task 1
**Author:** [Your Name]
**Date:** July 2026

---

## 1. Executive Summary

This project builds a complete, end-to-end machine learning system to
classify Iris flowers into one of three species (Setosa, Versicolor,
Virginica) using four physical measurements, trained on the
user-provided Kaggle "Iris Species" dataset. Seven classification
algorithms were trained and rigorously compared; a **Support Vector
Machine** was selected as the final model, achieving **96.7% test
accuracy** and **96.6% cross-validated accuracy** (tied with Gradient
Boosting and Random Forest, with SVM chosen as the simplest and fastest
of the three). The model was deployed in an interactive Streamlit web
dashboard supporting data exploration, filtering, and live predictions.

## 2. Objective

To automatically classify an Iris flower's species from four measurements
(sepal length, sepal width, petal length, petal width), removing the need
for manual/expert visual identification, and to demonstrate a full,
professional data science workflow from raw data to a deployed application.

## 3. Data Understanding

The dataset (`data/Iris_raw_kaggle.csv`, the Kaggle "Iris Species" dataset)
contains 150 samples, 50 per species, with four continuous numeric
features plus a row-identifier column that was dropped during
standardization. It is a classic benchmark dataset in statistics and
machine learning, chosen for this task because it is small enough to fully
understand yet rich enough to demonstrate every step of a real ML pipeline.

## 4. Data Cleaning & Preprocessing

- **Missing values:** None found across any column.
- **Duplicates:** 3 exact duplicate rows were identified and removed
  (150 → 147 rows). These only appear once the `Id` column is dropped,
  since a few rows share identical measurements and species.
- **Outliers:** Using the IQR (Interquartile Range) method, 4 mild outliers
  were detected in `sepal_width_cm`. These were **capped** (Winsorized) to
  the IQR bounds rather than deleted, preserving sample size while reducing
  the influence of extreme values.

## 5. Exploratory Data Analysis — Key Findings

- Petal measurements (length & width) show dramatically stronger separation
  between species than sepal measurements.
- Setosa is linearly separable from the other two species on petal length
  alone — no model in this project ever misclassifies a Setosa sample.
- Versicolor and Virginica overlap somewhat in petal length/width, which is
  the primary source of classification error across all models tested.
- Petal length and petal width are highly correlated (r ≈ 0.96); sepal
  width is comparatively uncorrelated with the other features and is the
  weakest standalone predictor.

Charts supporting these findings: `assets/charts/01` through `07`.

## 6. Feature Engineering

Four engineered features were added to support tree-based models and
demonstrate domain-driven feature creation:
- `sepal_area_cm2`, `petal_area_cm2` (approximate area, length × width)
- `sepal_aspect_ratio`, `petal_aspect_ratio` (length ÷ width, shape proxy)

The final models were trained on the four original measurements (the
engineered features were explored but did not materially improve
performance over the raw measurements for this dataset — a useful finding
in itself, and consistent with how well-separated the raw features already
are).

## 7. Modeling & Evaluation

Seven models were trained on an 80/20 stratified train-test split, with
5-fold stratified cross-validation for robustness:

| Model | Test Accuracy | CV Mean Accuracy |
|---|---|---|
| **Support Vector Machine** | **0.967** | 0.966 |
| Gradient Boosting | 0.967 | 0.966 |
| Random Forest | 0.967 | 0.957 |
| Logistic Regression | 0.933 | 0.957 |
| K-Nearest Neighbors | 0.933 | 0.940 |
| Decision Tree | 0.933 | 0.940 |
| Naive Bayes | 0.933 | 0.957 |

**Selected model: Support Vector Machine (RBF kernel).** It tied for the
highest test accuracy and cross-validation performance with Gradient
Boosting and Random Forest, but was chosen as the final model for being
the simplest and fastest of the three top performers (a single margin-based
classifier vs. large tree ensembles), while making only one
misclassification on the 30-sample test set — between Versicolor and
Virginica, the only pair of species with genuine measurement overlap.
Full metrics are in `docs/model_comparison_results.csv` and
`docs/classification_report_best_model.txt`.

## 8. Business Recommendations

1. Prioritize **petal measurements** in any future data collection —
   they carry nearly all the predictive signal.
2. To further reduce the small remaining error rate, collect more labeled
   samples near the Versicolor/Virginica boundary.
3. The final model is lightweight (4 input features, sub-millisecond
   inference) and suitable for real-time or embedded/mobile deployment.

## 9. Deliverables

- Cleaned dataset (`data/iris_cleaned.csv`) and feature-engineered dataset
  (`data/iris_features.csv`)
- 10 EDA and evaluation charts (`assets/charts/`)
- Trained model + encoder + metadata (`models/`)
- Interactive Streamlit dashboard (`app/app.py`)
- Full source code, documentation, and this report

## 10. Conclusion

The project successfully delivers a clean, reproducible, well-documented
machine learning pipeline achieving strong classification performance
(96.7% accuracy) on the Iris dataset, packaged into a usable, interactive
web application — demonstrating the complete data science lifecycle from
raw data to deployment.
