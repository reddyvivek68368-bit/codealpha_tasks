"""
feature_engineering.py
------------------------
Purpose:
    Create features needed to model/predict the unemployment rate:
    encode categorical fields, add a lockdown-period flag, and add
    month/year as numeric features.
"""

import pandas as pd

LOCKDOWN_DATE = pd.Timestamp("2020-03-25")


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Binary flag: was this observation during/after the national lockdown?
    df["is_lockdown_period"] = (df["date"] >= LOCKDOWN_DATE).astype(int)

    # Fill missing 'area' (file2 rows have no Rural/Urban split) with
    # "Combined" so it can still be used as a category rather than dropped
    df["area_filled"] = df["area"].fillna("Combined")

    # Simple numeric encoding for state (needed for regression models).
    # We use each state's mean historical unemployment rate as its
    # encoded value ("target encoding") -- this captures each state's
    # typical unemployment level as a single useful number instead of
    # creating 28 separate one-hot columns.
    state_avg = df.groupby("state")["unemployment_rate"].transform("mean")
    df["state_avg_unemployment"] = state_avg

    return df


def run_feature_engineering(
    clean_path: str = "data/unemployment_cleaned.csv",
    out_path: str = "data/unemployment_features.csv",
) -> pd.DataFrame:
    df = pd.read_csv(clean_path, parse_dates=["date"])
    df = add_engineered_features(df)
    df.to_csv(out_path, index=False)
    print(f"✅ Feature-engineered dataset saved to: {out_path}")
    print("New columns: is_lockdown_period, area_filled, state_avg_unemployment")
    print(f"Final shape: {df.shape}")
    return df


if __name__ == "__main__":
    run_feature_engineering()
