"""
preprocessing.py
------------------
Purpose:
    Load the raw iris.csv, clean it (handle missing values, duplicates,
    outliers), and produce a cleaned dataset ready for EDA and modeling.

Why this matters:
    Even though the classic Iris dataset is famous for being very clean,
    a real-world data science workflow ALWAYS includes a cleaning step.
    We build this step properly and generically so the same code would
    work correctly even if the data had real-world messiness (which is
    common in almost every other dataset you'll work with).
"""

import pandas as pd
import numpy as np


def load_raw_data(path: str = "data/iris.csv") -> pd.DataFrame:
    """Reads the raw CSV file into a pandas DataFrame."""
    return pd.read_csv(path)


def check_missing_values(df: pd.DataFrame) -> pd.Series:
    """
    Returns a count of missing (NaN) values per column.
    isnull() marks every missing cell as True (1) and every filled
    cell as False (0). sum() then adds those up per column.
    """
    return df.isnull().sum()


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fills missing numeric values with the column's MEDIAN.
    We use median (not mean) because it is more robust to outliers --
    a few extreme values won't drag the median far away like they
    would the mean.
    """
    df = df.copy()  # work on a copy so we never silently modify the original
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value)
            print(f"Filled {df[col].isnull().sum()} missing values in '{col}' with median={median_value}")

    return df


def check_duplicates(df: pd.DataFrame) -> int:
    """Returns the number of fully duplicated rows in the DataFrame."""
    return df.duplicated().sum()


def handle_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes exact duplicate rows.
    keep='first' means: if a row appears more than once, keep only
    the first occurrence and drop the rest.
    """
    before = len(df)
    df = df.drop_duplicates(keep="first").reset_index(drop=True)
    after = len(df)
    print(f"Removed {before - after} duplicate rows ({before} -> {after})")
    return df


def detect_outliers_iqr(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Detects outliers in a single numeric column using the IQR
    (Interquartile Range) method -- a standard statistical technique.

    Q1 = 25th percentile, Q3 = 75th percentile
    IQR = Q3 - Q1
    A value is considered an outlier if it falls below (Q1 - 1.5*IQR)
    or above (Q3 + 1.5*IQR).

    Returns the subset of rows that ARE outliers for that column.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers


def handle_outliers(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Caps (clips) outliers to the IQR bounds instead of deleting rows.
    We CAP rather than DELETE because:
      1) The Iris dataset is small (150 rows) -- deleting rows loses
         valuable data.
      2) Capping preserves the row (and its other correct measurements)
         while pulling only the extreme value back to a reasonable range.
    This is called "Winsorization".
    """
    df = df.copy()
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        n_outliers = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
        if n_outliers > 0:
            df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
            print(f"Capped {n_outliers} outlier(s) in '{column}' to range [{lower_bound:.2f}, {upper_bound:.2f}]")

    return df


def run_full_cleaning_pipeline(
    raw_path: str = "data/iris.csv",
    clean_path: str = "data/iris_cleaned.csv",
) -> pd.DataFrame:
    """
    Runs the complete cleaning pipeline end to end:
    1. Load raw data
    2. Report + handle missing values
    3. Report + handle duplicates
    4. Report + handle outliers (numeric feature columns only)
    5. Save the cleaned dataset
    """
    print("=" * 60)
    print("STEP 1: Loading raw data")
    print("=" * 60)
    df = load_raw_data(raw_path)
    print(f"Raw shape: {df.shape}")

    print("\n" + "=" * 60)
    print("STEP 2: Checking missing values")
    print("=" * 60)
    print(check_missing_values(df))
    df = handle_missing_values(df)

    print("\n" + "=" * 60)
    print("STEP 3: Checking duplicates")
    print("=" * 60)
    n_dupes = check_duplicates(df)
    print(f"Duplicate rows found: {n_dupes}")
    if n_dupes > 0:
        df = handle_duplicates(df)

    print("\n" + "=" * 60)
    print("STEP 4: Checking & handling outliers (IQR method)")
    print("=" * 60)
    feature_cols = [
        "sepal_length_cm",
        "sepal_width_cm",
        "petal_length_cm",
        "petal_width_cm",
    ]
    for col in feature_cols:
        outliers = detect_outliers_iqr(df, col)
        print(f"{col}: {len(outliers)} outlier(s) detected")

    df = handle_outliers(df, feature_cols)

    print("\n" + "=" * 60)
    print("STEP 5: Saving cleaned dataset")
    print("=" * 60)
    df.to_csv(clean_path, index=False)
    print(f"✅ Cleaned dataset saved to: {clean_path}")
    print(f"Final shape: {df.shape}")

    return df


if __name__ == "__main__":
    run_full_cleaning_pipeline()
