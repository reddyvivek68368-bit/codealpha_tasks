"""
preprocessing.py
------------------
Purpose:
    Load the two raw "Unemployment in India" CSV files, clean them
    (fix column names, handle missing values/duplicates/outliers,
    parse dates), and merge them into one unified, analysis-ready
    dataset.

Why two files?
    The Kaggle dataset ships as two files:
      1. "Unemployment in India.csv" -- covers May 2019 to June 2020,
         split by Rural/Urban Area, no geographic coordinates.
      2. "Unemployment_Rate_upto_11_2020.csv" -- covers Jan 2020 to
         Oct 2020, includes a Region/Zone (e.g. "South", "North") and
         longitude/latitude for each state, but no Rural/Urban split.
    We standardize both to a common schema and combine them so the
    rest of the project (EDA, modeling, dashboard) has one clean,
    consistent dataset covering the full time range.
"""

import pandas as pd
import numpy as np
import os


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    The raw CSVs have inconsistent column names with leading/trailing
    spaces (e.g. " Date", " Estimated Unemployment Rate (%)"). This
    strips whitespace and standardizes names to snake_case so the
    rest of the code can reference them reliably.
    """
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df


def load_and_clean_file1(path: str) -> pd.DataFrame:
    """
    Loads 'Unemployment in India.csv' (Region/Area version).
    """
    df = pd.read_csv(path)
    df = _clean_columns(df)

    rename_map = {
        "Region": "state",
        "Date": "date",
        "Frequency": "frequency",
        "Estimated Unemployment Rate (%)": "unemployment_rate",
        "Estimated Employed": "employed_count",
        "Estimated Labour Participation Rate (%)": "labour_participation_rate",
        "Area": "area",
    }
    df = df.rename(columns=rename_map)

    # Strip whitespace from every text (string) column -- the raw
    # file has values like " 31-05-2019" with a leading space
    for col in df.select_dtypes(include=["object", "str"]).columns:
        df[col] = df[col].str.strip()

    df["source_file"] = "file1_2019_2020"
    df["zone"] = np.nan
    df["longitude"] = np.nan
    df["latitude"] = np.nan

    return df


def load_and_clean_file2(path: str) -> pd.DataFrame:
    """
    Loads 'Unemployment_Rate_upto_11_2020.csv' (Zone/coordinates version).
    Note: this file has TWO columns both originally named "Region" --
    pandas auto-renames the second one to "Region.1" on read, which
    is actually the Zone (e.g. South, North, East, West).
    """
    df = pd.read_csv(path)
    df = _clean_columns(df)

    rename_map = {
        "Region": "state",
        "Date": "date",
        "Frequency": "frequency",
        "Estimated Unemployment Rate (%)": "unemployment_rate",
        "Estimated Employed": "employed_count",
        "Estimated Labour Participation Rate (%)": "labour_participation_rate",
        "Region.1": "zone",
        "longitude": "longitude",
        "latitude": "latitude",
    }
    df = df.rename(columns=rename_map)

    for col in df.select_dtypes(include=["object", "str"]).columns:
        df[col] = df[col].str.strip()

    df["source_file"] = "file2_2020_extended"
    df["area"] = np.nan  # this file has no Rural/Urban split

    return df


def check_missing_values(df: pd.DataFrame) -> pd.Series:
    return df.isnull().sum()


def handle_missing_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    The raw file1 CSV has 28 fully-blank trailing rows (every column
    is NaN) -- these are junk rows from the source file, not real
    missing data points, so we drop rows where the core measurement
    columns (state, date, unemployment_rate) are all missing.
    """
    df = df.copy()
    before = len(df)
    df = df.dropna(subset=["state", "date", "unemployment_rate"], how="all")
    after = len(df)
    print(f"Dropped {before - after} fully-blank rows ({before} -> {after})")
    return df


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Converts the date column (format DD-MM-YYYY) into a real datetime."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y", errors="coerce")
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["month_name"] = df["date"].dt.strftime("%b")
    return df


def check_duplicates(df: pd.DataFrame) -> int:
    return df.duplicated(subset=["state", "date", "area"]).sum()


def handle_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=["state", "date", "area"], keep="first").reset_index(drop=True)
    after = len(df)
    print(f"Removed {before - after} duplicate rows ({before} -> {after})")
    return df


def detect_and_cap_outliers(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Caps extreme outliers in a numeric column using the IQR method,
    same approach as Task 1. Unemployment rate genuinely spiked very
    high during COVID lockdown (April-May 2020), so we use a wider
    3.0 * IQR threshold (instead of the standard 1.5) to avoid
    flattening real, meaningful pandemic-era spikes as if they were
    data errors.
    """
    df = df.copy()
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 3.0 * IQR
    upper_bound = Q3 + 3.0 * IQR

    n_outliers = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
    if n_outliers > 0:
        df[column] = df[column].clip(lower=max(lower_bound, 0), upper=upper_bound)
        print(f"Capped {n_outliers} extreme outlier(s) in '{column}' to range "
              f"[{max(lower_bound, 0):.2f}, {upper_bound:.2f}]")
    else:
        print(f"No extreme outliers found in '{column}'")
    return df


def merge_and_clean(
    file1_path: str = "data/Unemployment_in_India_raw.csv",
    file2_path: str = "data/Unemployment_Rate_upto_11_2020_raw.csv",
    output_path: str = "data/unemployment_cleaned.csv",
) -> pd.DataFrame:
    """
    Full pipeline: load both files, clean each, merge them into one
    combined dataset, and save the result.
    """
    print("=" * 60)
    print("STEP 1: Loading and standardizing both raw files")
    print("=" * 60)
    df1 = load_and_clean_file1(file1_path)
    df2 = load_and_clean_file2(file2_path)
    print(f"File 1 shape: {df1.shape}")
    print(f"File 2 shape: {df2.shape}")

    common_cols = [
        "state", "date", "frequency", "unemployment_rate", "employed_count",
        "labour_participation_rate", "area", "zone", "longitude", "latitude",
        "source_file",
    ]
    df1 = df1[common_cols]
    df2 = df2[common_cols]

    print("\n" + "=" * 60)
    print("STEP 2: Merging both files into one dataset")
    print("=" * 60)
    df = pd.concat([df1, df2], ignore_index=True)
    print(f"Combined shape: {df.shape}")

    print("\n" + "=" * 60)
    print("STEP 3: Checking & handling missing rows")
    print("=" * 60)
    print(check_missing_values(df))
    df = handle_missing_rows(df)

    print("\n" + "=" * 60)
    print("STEP 4: Parsing dates")
    print("=" * 60)
    df = parse_dates(df)
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    print("\n" + "=" * 60)
    print("STEP 5: Checking & handling duplicates")
    print("=" * 60)
    n_dupes = check_duplicates(df)
    print(f"Duplicate rows found: {n_dupes}")
    if n_dupes > 0:
        df = handle_duplicates(df)

    print("\n" + "=" * 60)
    print("STEP 6: Handling outliers in key numeric columns")
    print("=" * 60)
    df = detect_and_cap_outliers(df, "unemployment_rate")
    df = detect_and_cap_outliers(df, "labour_participation_rate")

    print("\n" + "=" * 60)
    print("STEP 7: Saving cleaned & merged dataset")
    print("=" * 60)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Cleaned dataset saved to: {output_path}")
    print(f"Final shape: {df.shape}")

    return df


if __name__ == "__main__":
    merge_and_clean()
