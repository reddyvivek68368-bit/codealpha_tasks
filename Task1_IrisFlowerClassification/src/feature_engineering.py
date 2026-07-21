"""
feature_engineering.py
------------------------
Purpose:
    Create additional engineered features from the 4 raw measurements.
    Even though the original 4 features already work very well for
    Iris classification, we add a few engineered features to:
      1) Demonstrate the feature engineering step properly.
      2) Squeeze out a little extra model performance.
      3) Show how domain knowledge (basic geometry of a petal/sepal)
         can be turned into new, useful columns.
"""

import pandas as pd


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds 4 new engineered features based on simple geometry/ratios:

    1. sepal_area_cm2  = sepal_length * sepal_width
       (approximates the area of the sepal, treating it like a rectangle)
    2. petal_area_cm2  = petal_length * petal_width
       (approximates the area of the petal)
    3. sepal_aspect_ratio = sepal_length / sepal_width
       (captures the "shape" of the sepal -- long & thin vs short & wide)
    4. petal_aspect_ratio = petal_length / petal_width
       (captures the "shape" of the petal)

    These ratio/area features often help tree-based models split the
    data more effectively than raw length/width alone.
    """
    df = df.copy()

    df["sepal_area_cm2"] = df["sepal_length_cm"] * df["sepal_width_cm"]
    df["petal_area_cm2"] = df["petal_length_cm"] * df["petal_width_cm"]
    df["sepal_aspect_ratio"] = df["sepal_length_cm"] / df["sepal_width_cm"]
    df["petal_aspect_ratio"] = df["petal_length_cm"] / df["petal_width_cm"]

    return df


def run_feature_engineering(
    clean_path: str = "data/iris_cleaned.csv",
    out_path: str = "data/iris_features.csv",
) -> pd.DataFrame:
    df = pd.read_csv(clean_path)
    df = add_engineered_features(df)
    df.to_csv(out_path, index=False)
    print(f"✅ Feature-engineered dataset saved to: {out_path}")
    print(f"New columns added: sepal_area_cm2, petal_area_cm2, sepal_aspect_ratio, petal_aspect_ratio")
    print(f"Final shape: {df.shape}")
    return df


if __name__ == "__main__":
    run_feature_engineering()
