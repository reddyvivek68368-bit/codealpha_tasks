"""
eda.py
-------
Purpose:
    Perform Exploratory Data Analysis (EDA) on the cleaned Iris dataset
    and save professional charts to assets/charts/ for use in the
    report, README, and Streamlit dashboard.

What is EDA?
    EDA means looking at the data BEFORE modeling to understand its
    shape, spot patterns, and check assumptions. Good EDA often reveals
    which features will be useful for the model.
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # use a non-interactive backend (no GUI needed, just save files)
import matplotlib.pyplot as plt
import seaborn as sns
import os

# A consistent, professional visual style for every chart in this project
sns.set_theme(style="whitegrid", palette="Set2")
FIGSIZE = (8, 6)
CHART_DIR = "assets/charts"


def load_clean_data(path: str = "data/iris_cleaned.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def save_fig(fig, name: str):
    os.makedirs(CHART_DIR, exist_ok=True)
    path = os.path.join(CHART_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved chart: {path}")


def chart_species_distribution(df: pd.DataFrame):
    """Bar chart: how many samples per species (checks class balance)."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    counts = df["species"].value_counts()
    sns.barplot(x=counts.index, y=counts.values, ax=ax, hue=counts.index, legend=False)
    ax.set_title("Number of Samples per Species", fontsize=14, fontweight="bold")
    ax.set_xlabel("Species")
    ax.set_ylabel("Count")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 0.5, str(v), ha="center", fontweight="bold")
    save_fig(fig, "01_species_distribution.png")


def chart_correlation_heatmap(df: pd.DataFrame):
    """Heatmap showing how strongly each numeric feature relates to the others."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    numeric_df = df.drop(columns=["species_id", "species"])
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, square=True, linewidths=0.5)
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    save_fig(fig, "02_correlation_heatmap.png")


def chart_pairplot(df: pd.DataFrame):
    """Pairwise scatterplots of all features, colored by species."""
    plot_df = df.drop(columns=["species_id"])
    g = sns.pairplot(plot_df, hue="species", palette="Set2", diag_kind="hist", height=2.2)
    g.fig.suptitle("Pairwise Feature Relationships by Species", y=1.02, fontsize=14, fontweight="bold")
    g.savefig(os.path.join(CHART_DIR, "03_pairplot.png"), dpi=150, bbox_inches="tight")
    plt.close(g.fig)
    print(f"Saved chart: {CHART_DIR}/03_pairplot.png")


def chart_boxplots(df: pd.DataFrame):
    """Boxplots of each feature split by species (shows spread + outliers)."""
    features = ["sepal_length_cm", "sepal_width_cm", "petal_length_cm", "petal_width_cm"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for i, feature in enumerate(features):
        sns.boxplot(data=df, x="species", y=feature, ax=axes[i], hue="species", legend=False)
        axes[i].set_title(f"{feature.replace('_', ' ').title()} by Species", fontweight="bold")
    fig.suptitle("Feature Distributions by Species (Boxplots)", fontsize=15, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "04_boxplots_by_species.png")


def chart_histograms(df: pd.DataFrame):
    """Histograms with KDE curves for each feature (overall distribution shape)."""
    features = ["sepal_length_cm", "sepal_width_cm", "petal_length_cm", "petal_width_cm"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for i, feature in enumerate(features):
        sns.histplot(data=df, x=feature, hue="species", kde=True, ax=axes[i], element="step")
        axes[i].set_title(f"Distribution of {feature.replace('_', ' ').title()}", fontweight="bold")
    fig.suptitle("Feature Distributions (Histograms + KDE)", fontsize=15, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "05_histograms.png")


def chart_petal_scatter(df: pd.DataFrame):
    """
    Single most important chart for Iris: petal length vs petal width.
    This one chart alone almost perfectly separates the 3 species.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.scatterplot(
        data=df, x="petal_length_cm", y="petal_width_cm",
        hue="species", style="species", s=100, ax=ax
    )
    ax.set_title("Petal Length vs Petal Width (Best Species Separator)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Petal Length (cm)")
    ax.set_ylabel("Petal Width (cm)")
    save_fig(fig, "06_petal_scatter.png")


def chart_violin(df: pd.DataFrame):
    """Violin plots: like boxplots but also show the shape of the distribution."""
    features = ["sepal_length_cm", "sepal_width_cm", "petal_length_cm", "petal_width_cm"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for i, feature in enumerate(features):
        sns.violinplot(data=df, x="species", y=feature, ax=axes[i], hue="species", legend=False)
        axes[i].set_title(f"{feature.replace('_', ' ').title()} Distribution Shape", fontweight="bold")
    fig.suptitle("Feature Distributions by Species (Violin Plots)", fontsize=15, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "07_violin_plots.png")


def generate_summary_stats(df: pd.DataFrame, out_path: str = "docs/eda_summary_stats.csv"):
    """Saves descriptive statistics (mean, std, min, max, etc.) grouped by species."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    summary = df.groupby("species").describe().T
    summary.to_csv(out_path)
    print(f"Saved summary statistics: {out_path}")
    return summary


def run_full_eda(clean_path: str = "data/iris_cleaned.csv"):
    df = load_clean_data(clean_path)
    print(f"Loaded cleaned data: {df.shape}")

    chart_species_distribution(df)
    chart_correlation_heatmap(df)
    chart_pairplot(df)
    chart_boxplots(df)
    chart_histograms(df)
    chart_petal_scatter(df)
    chart_violin(df)
    generate_summary_stats(df)

    print("\n✅ EDA complete. All charts saved to assets/charts/")


if __name__ == "__main__":
    run_full_eda()
