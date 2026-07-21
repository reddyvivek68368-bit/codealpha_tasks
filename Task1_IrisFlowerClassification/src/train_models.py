"""
train_models.py
------------------
Purpose:
    Train MULTIPLE machine learning classification models on the Iris
    dataset, evaluate each one with proper metrics, compare them,
    select the best one, and save it to disk for the Streamlit app.

Models trained:
    1. Logistic Regression
    2. K-Nearest Neighbors (KNN)
    3. Support Vector Machine (SVM)
    4. Decision Tree
    5. Random Forest
    6. Naive Bayes
    7. Gradient Boosting

Why train so many models?
    There's no way to know in advance which algorithm will work best
    for a given dataset. Best practice is to try several, evaluate
    them fairly on the SAME train/test split, and pick the winner
    based on objective metrics -- not guessing.
"""

import pandas as pd
import numpy as np
import json
import os
import time
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_STATE = 42
FEATURE_COLUMNS = [
    "sepal_length_cm",
    "sepal_width_cm",
    "petal_length_cm",
    "petal_width_cm",
]
TARGET_COLUMN = "species"


def load_data(path: str = "data/iris_features.csv"):
    return pd.read_csv(path)


def get_model_zoo() -> dict:
    """
    Returns a dictionary of {model_name: sklearn Pipeline}.
    Each pipeline first scales the features with StandardScaler
    (mean=0, std=1) then feeds them into the classifier.
    Scaling matters most for distance-based models (KNN, SVM) and
    doesn't hurt tree-based models, so we apply it consistently
    to all models for a fair, simple comparison.
    """
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Support Vector Machine": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=5, random_state=RANDOM_STATE),
        "Naive Bayes": GaussianNB(),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, random_state=RANDOM_STATE),
    }

    pipelines = {}
    for name, clf in models.items():
        pipelines[name] = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", clf),
        ])
    return pipelines


def train_and_evaluate_all(df: pd.DataFrame):
    """
    Splits data, trains every model in the zoo, evaluates each on
    the held-out test set AND with 5-fold cross-validation, and
    returns a results DataFrame plus the fitted pipelines.
    """
    X = df[FEATURE_COLUMNS]
    y_raw = df[TARGET_COLUMN]

    # Encode species names (setosa/versicolor/virginica) into numbers (0/1/2)
    # because scikit-learn models work with numeric labels internally.
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    # Split into training (80%) and testing (20%) sets.
    # stratify=y ensures each species is proportionally represented
    # in both the train and test sets (important for small datasets).
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    print(f"Training set size: {X_train.shape[0]} rows")
    print(f"Test set size: {X_test.shape[0]} rows")

    pipelines = get_model_zoo()
    results = []
    fitted_models = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for name, pipeline in pipelines.items():
        start = time.time()
        pipeline.fit(X_train, y_train)
        train_time = time.time() - start

        y_pred = pipeline.predict(X_test)

        # 5-fold cross-validation on the TRAINING data for a more
        # robust estimate of generalization performance
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy")

        metrics = {
            "Model": name,
            "Test Accuracy": accuracy_score(y_test, y_pred),
            "Precision (macro)": precision_score(y_test, y_pred, average="macro"),
            "Recall (macro)": recall_score(y_test, y_pred, average="macro"),
            "F1 Score (macro)": f1_score(y_test, y_pred, average="macro"),
            "CV Mean Accuracy": cv_scores.mean(),
            "CV Std": cv_scores.std(),
            "Train Time (s)": round(train_time, 4),
        }
        results.append(metrics)
        fitted_models[name] = pipeline

        print(f"✅ {name}: Test Accuracy = {metrics['Test Accuracy']:.4f} | "
              f"CV Accuracy = {metrics['CV Mean Accuracy']:.4f} (+/- {metrics['CV Std']:.4f})")

    results_df = pd.DataFrame(results).sort_values("Test Accuracy", ascending=False).reset_index(drop=True)

    return results_df, fitted_models, label_encoder, (X_train, X_test, y_train, y_test)


def plot_model_comparison(results_df: pd.DataFrame, out_path: str = "assets/charts/08_model_comparison.png"):
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_df = results_df.sort_values("Test Accuracy")
    sns.barplot(data=plot_df, y="Model", x="Test Accuracy", hue="Model", legend=False, ax=ax, palette="viridis")
    ax.set_title("Model Comparison — Test Accuracy", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 1.05)
    for i, v in enumerate(plot_df["Test Accuracy"]):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontweight="bold")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved chart: {out_path}")


def plot_confusion_matrix(model, X_test, y_test, label_encoder, model_name: str,
                           out_path: str = "assets/charts/09_confusion_matrix_best_model.png"):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_
    )
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted Species")
    ax.set_ylabel("Actual Species")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved chart: {out_path}")


def plot_feature_importance(model, model_name: str, out_path: str = "assets/charts/10_feature_importance.png"):
    """Only works for tree-based models that expose feature_importances_."""
    classifier = model.named_steps["classifier"]
    if not hasattr(classifier, "feature_importances_"):
        print(f"{model_name} does not support feature importance plotting. Skipping.")
        return

    importances = classifier.feature_importances_
    feat_df = pd.DataFrame({
        "Feature": FEATURE_COLUMNS,
        "Importance": importances
    }).sort_values("Importance", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=feat_df, x="Importance", y="Feature", hue="Feature", legend=False, ax=ax, palette="mako")
    ax.set_title(f"Feature Importance — {model_name}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved chart: {out_path}")


def save_best_model(fitted_models: dict, results_df: pd.DataFrame, label_encoder,
                     model_dir: str = "models"):
    os.makedirs(model_dir, exist_ok=True)

    best_model_name = results_df.iloc[0]["Model"]
    best_model = fitted_models[best_model_name]

    joblib.dump(best_model, os.path.join(model_dir, "best_model.pkl"))
    joblib.dump(label_encoder, os.path.join(model_dir, "label_encoder.pkl"))

    metadata = {
        "best_model_name": best_model_name,
        "test_accuracy": float(results_df.iloc[0]["Test Accuracy"]),
        "cv_mean_accuracy": float(results_df.iloc[0]["CV Mean Accuracy"]),
        "feature_columns": FEATURE_COLUMNS,
        "target_classes": list(label_encoder.classes_),
    }
    with open(os.path.join(model_dir, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"\n🏆 Best model: {best_model_name}")
    print(f"✅ Saved to {model_dir}/best_model.pkl")
    print(f"✅ Saved label encoder to {model_dir}/label_encoder.pkl")
    print(f"✅ Saved metadata to {model_dir}/model_metadata.json")

    return best_model_name, best_model


def run_full_training_pipeline(data_path: str = "data/iris_features.csv"):
    df = load_data(data_path)
    results_df, fitted_models, label_encoder, splits = train_and_evaluate_all(df)
    X_train, X_test, y_train, y_test = splits

    print("\n" + "=" * 70)
    print("MODEL COMPARISON TABLE")
    print("=" * 70)
    print(results_df.to_string(index=False))

    results_df.to_csv("docs/model_comparison_results.csv", index=False)

    plot_model_comparison(results_df)

    best_model_name, best_model = save_best_model(fitted_models, results_df, label_encoder)

    plot_confusion_matrix(best_model, X_test, y_test, label_encoder, best_model_name)
    plot_feature_importance(best_model, best_model_name)

    # Also try feature importance on Random Forest specifically (for the report)
    if "Random Forest" in fitted_models:
        plot_feature_importance(
            fitted_models["Random Forest"], "Random Forest",
            out_path="assets/charts/10b_feature_importance_rf.png"
        )

    # Save a full classification report (text) for the best model
    y_pred = best_model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
    with open("docs/classification_report_best_model.txt", "w") as f:
        f.write(f"Best Model: {best_model_name}\n\n")
        f.write(report)
    print(f"\n✅ Classification report saved to docs/classification_report_best_model.txt")

    return results_df, best_model_name, best_model


if __name__ == "__main__":
    run_full_training_pipeline()
