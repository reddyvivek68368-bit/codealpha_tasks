"""
train_models.py
------------------
Purpose:
    Train MULTIPLE regression models to predict the unemployment
    rate from other known indicators (labour participation rate,
    month/year, lockdown period, state's historical average), compare
    them, select the best, and save it.

Why regression (not classification)?
    Unemployment rate is a continuous number (e.g. 7.32%), not a
    category -- so this is a regression problem, unlike Task 1's
    species classification.

Models trained:
    1. Linear Regression
    2. Ridge Regression
    3. Decision Tree Regressor
    4. Random Forest Regressor
    5. Gradient Boosting Regressor
"""

import pandas as pd
import numpy as np
import json
import os
import time
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_STATE = 42
FEATURE_COLUMNS = [
    "labour_participation_rate",
    "month",
    "year",
    "is_lockdown_period",
    "state_avg_unemployment",
]
TARGET_COLUMN = "unemployment_rate"


def load_data(path: str = "data/unemployment_features.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def get_model_zoo() -> dict:
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=8, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=150, random_state=RANDOM_STATE),
    }
    pipelines = {}
    for name, model in models.items():
        pipelines[name] = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", model),
        ])
    return pipelines


def train_and_evaluate_all(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"Training set size: {X_train.shape[0]} rows")
    print(f"Test set size: {X_test.shape[0]} rows")

    pipelines = get_model_zoo()
    results = []
    fitted_models = {}
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for name, pipeline in pipelines.items():
        start = time.time()
        pipeline.fit(X_train, y_train)
        train_time = time.time() - start

        y_pred = pipeline.predict(X_test)

        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="r2")

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        metrics = {
            "Model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2 Score": r2,
            "CV Mean R2": cv_scores.mean(),
            "CV Std": cv_scores.std(),
            "Train Time (s)": round(train_time, 4),
        }
        results.append(metrics)
        fitted_models[name] = pipeline

        print(f"✅ {name}: R2 = {r2:.4f} | MAE = {mae:.3f} | RMSE = {rmse:.3f} | "
              f"CV R2 = {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    results_df = pd.DataFrame(results).sort_values("R2 Score", ascending=False).reset_index(drop=True)
    return results_df, fitted_models, (X_train, X_test, y_train, y_test)


def plot_model_comparison(results_df: pd.DataFrame, out_path: str = "assets/charts/09_model_comparison.png"):
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_df = results_df.sort_values("R2 Score")
    sns.barplot(data=plot_df, y="Model", x="R2 Score", hue="Model", legend=False, ax=ax, palette="viridis")
    ax.set_title("Model Comparison — R² Score (higher is better)", fontsize=14, fontweight="bold")
    for i, v in enumerate(plot_df["R2 Score"]):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontweight="bold")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved chart: {out_path}")


def plot_actual_vs_predicted(model, X_test, y_test, model_name: str,
                              out_path: str = "assets/charts/10_actual_vs_predicted.png"):
    y_pred = model.predict(X_test)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_test, y_pred, alpha=0.5, color="#3498db")
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", label="Perfect Prediction")
    ax.set_xlabel("Actual Unemployment Rate (%)")
    ax.set_ylabel("Predicted Unemployment Rate (%)")
    ax.set_title(f"Actual vs Predicted — {model_name}", fontsize=14, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved chart: {out_path}")


def plot_feature_importance(model, model_name: str, out_path: str = "assets/charts/11_feature_importance.png"):
    regressor = model.named_steps["regressor"]
    if hasattr(regressor, "feature_importances_"):
        importances = regressor.feature_importances_
    elif hasattr(regressor, "coef_"):
        importances = np.abs(regressor.coef_)
    else:
        print(f"{model_name} has no importance/coefficients to plot. Skipping.")
        return

    feat_df = pd.DataFrame({
        "Feature": FEATURE_COLUMNS,
        "Importance": importances,
    }).sort_values("Importance", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=feat_df, x="Importance", y="Feature", hue="Feature", legend=False, ax=ax, palette="mako")
    ax.set_title(f"Feature Importance — {model_name}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved chart: {out_path}")


def save_best_model(fitted_models: dict, results_df: pd.DataFrame, model_dir: str = "models"):
    os.makedirs(model_dir, exist_ok=True)

    best_model_name = results_df.iloc[0]["Model"]
    best_model = fitted_models[best_model_name]

    joblib.dump(best_model, os.path.join(model_dir, "best_model.pkl"))

    metadata = {
        "best_model_name": best_model_name,
        "r2_score": float(results_df.iloc[0]["R2 Score"]),
        "mae": float(results_df.iloc[0]["MAE"]),
        "rmse": float(results_df.iloc[0]["RMSE"]),
        "cv_mean_r2": float(results_df.iloc[0]["CV Mean R2"]),
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
    }
    with open(os.path.join(model_dir, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"\n🏆 Best model: {best_model_name}")
    print(f"✅ Saved to {model_dir}/best_model.pkl")
    print(f"✅ Saved metadata to {model_dir}/model_metadata.json")

    return best_model_name, best_model


def run_full_training_pipeline(data_path: str = "data/unemployment_features.csv"):
    df = load_data(data_path)
    results_df, fitted_models, splits = train_and_evaluate_all(df)
    X_train, X_test, y_train, y_test = splits

    print("\n" + "=" * 70)
    print("MODEL COMPARISON TABLE")
    print("=" * 70)
    print(results_df.to_string(index=False))

    os.makedirs("docs", exist_ok=True)
    results_df.to_csv("docs/model_comparison_results.csv", index=False)

    plot_model_comparison(results_df)

    best_model_name, best_model = save_best_model(fitted_models, results_df)

    plot_actual_vs_predicted(best_model, X_test, y_test, best_model_name)
    plot_feature_importance(best_model, best_model_name)

    if "Random Forest" in fitted_models:
        plot_feature_importance(
            fitted_models["Random Forest"], "Random Forest",
            out_path="assets/charts/11b_feature_importance_rf.png"
        )

    return results_df, best_model_name, best_model


if __name__ == "__main__":
    run_full_training_pipeline()
