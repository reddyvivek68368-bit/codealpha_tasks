"""
eda.py
-------
Purpose:
    Exploratory Data Analysis on the cleaned unemployment dataset,
    focused on understanding how COVID-19 and the 2020 lockdown
    affected unemployment across Indian states.
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os

sns.set_theme(style="whitegrid", palette="Set2")
CHART_DIR = "assets/charts"

# India's national lockdown began March 25, 2020 -- used to mark
# the "before vs during COVID" split point on time series charts
LOCKDOWN_DATE = pd.Timestamp("2020-03-25")


def load_clean_data(path: str = "data/unemployment_cleaned.csv") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def save_fig(fig, name: str):
    os.makedirs(CHART_DIR, exist_ok=True)
    path = os.path.join(CHART_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved chart: {path}")


def chart_national_trend(df: pd.DataFrame):
    """National average unemployment rate over time, with lockdown marked."""
    monthly = df.groupby("date")["unemployment_rate"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(monthly["date"], monthly["unemployment_rate"], marker="o", linewidth=2, color="#e74c3c")
    ax.axvline(LOCKDOWN_DATE, color="black", linestyle="--", alpha=0.7, label="National Lockdown (25 Mar 2020)")
    ax.set_title("India's National Average Unemployment Rate Over Time", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Unemployment Rate (%)")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    save_fig(fig, "01_national_trend.png")


def chart_top_states(df: pd.DataFrame, n: int = 10):
    """Top N states by average unemployment rate over the full period."""
    avg_by_state = df.groupby("state")["unemployment_rate"].mean().sort_values(ascending=False).head(n)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.barplot(x=avg_by_state.values, y=avg_by_state.index, hue=avg_by_state.index, legend=False, ax=ax, palette="Reds_r")
    ax.set_title(f"Top {n} States by Average Unemployment Rate", fontsize=14, fontweight="bold")
    ax.set_xlabel("Average Unemployment Rate (%)")
    for i, v in enumerate(avg_by_state.values):
        ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontweight="bold")
    save_fig(fig, "02_top10_states.png")


def chart_urban_vs_rural(df: pd.DataFrame):
    """Compares unemployment trends between Urban and Rural areas over time."""
    area_df = df.dropna(subset=["area"])
    monthly = area_df.groupby(["date", "area"])["unemployment_rate"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(11, 6))
    for area, color in [("Urban", "#3498db"), ("Rural", "#2ecc71")]:
        subset = monthly[monthly["area"] == area]
        ax.plot(subset["date"], subset["unemployment_rate"], marker="o", label=area, linewidth=2, color=color)
    ax.axvline(LOCKDOWN_DATE, color="black", linestyle="--", alpha=0.6, label="Lockdown Start")
    ax.set_title("Urban vs Rural Unemployment Rate Over Time", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Unemployment Rate (%)")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    save_fig(fig, "03_urban_vs_rural.png")


def chart_covid_impact_boxplot(df: pd.DataFrame):
    """Boxplot comparing unemployment rate distribution before vs during/after COVID lockdown."""
    df = df.copy()
    df["period"] = df["date"].apply(lambda d: "Before Lockdown" if d < LOCKDOWN_DATE else "During/After Lockdown")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=df, x="period", y="unemployment_rate", hue="period", legend=False, ax=ax, palette=["#95a5a6", "#e74c3c"])
    ax.set_title("Unemployment Rate: Before vs During/After Lockdown", fontsize=14, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Unemployment Rate (%)")
    save_fig(fig, "04_covid_impact_boxplot.png")


def chart_zone_comparison(df: pd.DataFrame):
    """Average unemployment rate by geographic zone (South, North, East, West, etc.) -- only available in file2 data."""
    zone_df = df.dropna(subset=["zone"])
    if zone_df.empty:
        print("No zone data available, skipping zone chart.")
        return
    avg_by_zone = zone_df.groupby("zone")["unemployment_rate"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=avg_by_zone.index, y=avg_by_zone.values, hue=avg_by_zone.index, legend=False, ax=ax, palette="mako")
    ax.set_title("Average Unemployment Rate by Zone (2020 data)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Average Unemployment Rate (%)")
    ax.set_xlabel("Zone")
    save_fig(fig, "05_zone_comparison.png")


def chart_correlation_heatmap(df: pd.DataFrame):
    numeric_cols = ["unemployment_rate", "employed_count", "labour_participation_rate"]
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, square=True, linewidths=0.5)
    ax.set_title("Correlation Between Key Indicators", fontsize=14, fontweight="bold")
    save_fig(fig, "06_correlation_heatmap.png")


def chart_monthly_heatmap(df: pd.DataFrame):
    """Heatmap: state (rows) x month (columns) unemployment rate."""
    pivot = df.pivot_table(index="state", columns="date", values="unemployment_rate", aggfunc="mean")
    pivot.columns = [d.strftime("%b-%y") for d in pivot.columns]
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(pivot, cmap="YlOrRd", ax=ax, linewidths=0.3, cbar_kws={"label": "Unemployment Rate (%)"})
    ax.set_title("Unemployment Rate Heatmap: State x Month", fontsize=14, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("State")
    fig.tight_layout()
    save_fig(fig, "07_state_month_heatmap.png")


def chart_labour_participation_trend(df: pd.DataFrame):
    monthly = df.groupby("date")["labour_participation_rate"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(monthly["date"], monthly["labour_participation_rate"], marker="o", linewidth=2, color="#9b59b6")
    ax.axvline(LOCKDOWN_DATE, color="black", linestyle="--", alpha=0.7, label="National Lockdown")
    ax.set_title("National Average Labour Participation Rate Over Time", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Labour Participation Rate (%)")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    save_fig(fig, "08_labour_participation_trend.png")


def generate_summary_stats(df: pd.DataFrame, out_path: str = "docs/eda_summary_stats.csv"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    summary = df.groupby("state")[["unemployment_rate", "employed_count", "labour_participation_rate"]].describe().T
    summary.to_csv(out_path)
    print(f"Saved summary statistics: {out_path}")
    return summary


def run_full_eda(clean_path: str = "data/unemployment_cleaned.csv"):
    df = load_clean_data(clean_path)
    print(f"Loaded cleaned data: {df.shape}")

    chart_national_trend(df)
    chart_top_states(df)
    chart_urban_vs_rural(df)
    chart_covid_impact_boxplot(df)
    chart_zone_comparison(df)
    chart_correlation_heatmap(df)
    chart_monthly_heatmap(df)
    chart_labour_participation_trend(df)
    generate_summary_stats(df)

    # Print key COVID impact numbers for the report
    before = df[df["date"] < LOCKDOWN_DATE]["unemployment_rate"].mean()
    during = df[df["date"] >= LOCKDOWN_DATE]["unemployment_rate"].mean()
    print(f"\nAverage unemployment BEFORE lockdown: {before:.2f}%")
    print(f"Average unemployment DURING/AFTER lockdown: {during:.2f}%")
    print(f"Increase: {during - before:.2f} percentage points ({(during/before - 1)*100:.1f}% relative increase)")

    print("\n✅ EDA complete. All charts saved to assets/charts/")


if __name__ == "__main__":
    run_full_eda()
