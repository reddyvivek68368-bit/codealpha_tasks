"""
run_pipeline.py
------------------
Purpose:
    Runs the ENTIRE project pipeline end-to-end, in the correct order,
    with one single command:

        python src/run_pipeline.py

    This is the "one button" script that takes you from raw data all
    the way to a saved, ready-to-use trained model.

Pipeline order:
    1. generate_dataset.py   -> creates data/iris.csv
    2. preprocessing.py      -> cleans it into data/iris_cleaned.csv
    3. eda.py                -> creates all charts in assets/charts/
    4. feature_engineering.py-> creates data/iris_features.csv
    5. train_models.py       -> trains, compares, and saves the best model
"""

from generate_dataset import generate_iris_csv
from preprocessing import run_full_cleaning_pipeline
from eda import run_full_eda
from feature_engineering import run_feature_engineering
from train_models import run_full_training_pipeline


def main():
    print("\n" + "#" * 70)
    print("# STEP 1/5: GENERATING RAW DATASET")
    print("#" * 70)
    generate_iris_csv("data/iris.csv")

    print("\n" + "#" * 70)
    print("# STEP 2/5: CLEANING & PREPROCESSING")
    print("#" * 70)
    run_full_cleaning_pipeline("data/iris.csv", "data/iris_cleaned.csv")

    print("\n" + "#" * 70)
    print("# STEP 3/5: EXPLORATORY DATA ANALYSIS")
    print("#" * 70)
    run_full_eda("data/iris_cleaned.csv")

    print("\n" + "#" * 70)
    print("# STEP 4/5: FEATURE ENGINEERING")
    print("#" * 70)
    run_feature_engineering("data/iris_cleaned.csv", "data/iris_features.csv")

    print("\n" + "#" * 70)
    print("# STEP 5/5: MODEL TRAINING, COMPARISON & SAVING")
    print("#" * 70)
    run_full_training_pipeline("data/iris_features.csv")

    print("\n" + "#" * 70)
    print("# ✅ PIPELINE COMPLETE — run `streamlit run app/app.py` to launch the dashboard")
    print("#" * 70)


if __name__ == "__main__":
    main()
