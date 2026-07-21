"""
generate_dataset.py
--------------------
Purpose:
    Loads the project's raw Iris dataset and standardizes it into the
    column format used throughout the rest of the pipeline, saving the
    result to data/iris.csv.

Data source:
    This project uses the user-provided dataset
    (data/Iris_raw_kaggle.csv) -- the well-known Kaggle "Iris Species"
    dataset (https://www.kaggle.com/datasets/uciml/iris), which itself
    is the same classic Iris data originally collected by botanist
    Edgar Anderson and published by statistician Ronald Fisher in 1936.

    Its raw columns look like this:
        Id, SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm, Species
    with Species values like "Iris-setosa", "Iris-versicolor", "Iris-virginica".

    We standardize this into the column format the rest of this project
    (preprocessing.py, eda.py, train_models.py, app.py) expects:
        sepal_length_cm, sepal_width_cm, petal_length_cm, petal_width_cm,
        species_id, species
    with species values "setosa", "versicolor", "virginica".

    If the raw Kaggle file isn't found (e.g. someone clones this repo
    without data/Iris_raw_kaggle.csv), we fall back to loading the same
    dataset from scikit-learn, so the pipeline never breaks.
"""

import pandas as pd
import os


RAW_KAGGLE_PATH = "data/Iris_raw_kaggle.csv"


def _load_from_kaggle_csv(path: str = RAW_KAGGLE_PATH) -> pd.DataFrame:
    """Loads and standardizes the user-provided Kaggle-format Iris CSV."""
    raw = pd.read_csv(path)

    # Drop the "Id" column -- it's just a row number, not a real feature
    if "Id" in raw.columns:
        raw = raw.drop(columns=["Id"])

    # Rename CamelCase columns to our standard snake_case_cm format
    rename_map = {
        "SepalLengthCm": "sepal_length_cm",
        "SepalWidthCm": "sepal_width_cm",
        "PetalLengthCm": "petal_length_cm",
        "PetalWidthCm": "petal_width_cm",
        "Species": "species_raw",
    }
    raw = raw.rename(columns=rename_map)

    # Clean up species labels: "Iris-setosa" -> "setosa"
    raw["species"] = raw["species_raw"].str.replace("Iris-", "", regex=False).str.lower()
    raw = raw.drop(columns=["species_raw"])

    # Add a numeric species_id column (0/1/2), matching scikit-learn's
    # convention: 0 = setosa, 1 = versicolor, 2 = virginica
    species_order = ["setosa", "versicolor", "virginica"]
    species_to_id = {name: i for i, name in enumerate(species_order)}
    raw["species_id"] = raw["species"].map(species_to_id)

    # Reorder columns to match the rest of the project
    df = raw[[
        "sepal_length_cm", "sepal_width_cm", "petal_length_cm",
        "petal_width_cm", "species_id", "species",
    ]]

    return df


def _load_from_sklearn() -> pd.DataFrame:
    """Fallback loader: builds the same dataset from scikit-learn."""
    from sklearn.datasets import load_iris

    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df.columns = [col.replace(" (cm)", "").replace(" ", "_") + "_cm" for col in df.columns]
    df["species_id"] = iris.target
    species_map = {i: name for i, name in enumerate(iris.target_names)}
    df["species"] = df["species_id"].map(species_map)
    return df


def generate_iris_csv(output_path: str = "data/iris.csv") -> pd.DataFrame:
    """
    Loads the raw Iris data (from the user's uploaded Kaggle CSV if
    present, otherwise falls back to scikit-learn), standardizes the
    columns, and saves it to output_path.
    """
    if os.path.exists(RAW_KAGGLE_PATH):
        print(f"Loading dataset from user-provided file: {RAW_KAGGLE_PATH}")
        df = _load_from_kaggle_csv(RAW_KAGGLE_PATH)
    else:
        print(f"'{RAW_KAGGLE_PATH}' not found -- falling back to scikit-learn's built-in Iris dataset")
        df = _load_from_sklearn()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"✅ Standardized dataset saved to: {output_path}")
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")

    return df


if __name__ == "__main__":
    generate_iris_csv("data/iris.csv")
