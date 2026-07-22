"""
run_pipeline.py
------------------
Runs the entire Task 2 pipeline end-to-end with one command:
    python src/run_pipeline.py
"""

from preprocessing import merge_and_clean
from eda import run_full_eda
from feature_engineering import run_feature_engineering
from train_models import run_full_training_pipeline


def main():
    print("\n" + "#" * 70)
    print("# STEP 1/4: CLEANING & MERGING RAW DATA")
    print("#" * 70)
    merge_and_clean(
        "data/Unemployment_in_India_raw.csv",
        "data/Unemployment_Rate_upto_11_2020_raw.csv",
        "data/unemployment_cleaned.csv",
    )

    print("\n" + "#" * 70)
    print("# STEP 2/4: EXPLORATORY DATA ANALYSIS")
    print("#" * 70)
    run_full_eda("data/unemployment_cleaned.csv")

    print("\n" + "#" * 70)
    print("# STEP 3/4: FEATURE ENGINEERING")
    print("#" * 70)
    run_feature_engineering("data/unemployment_cleaned.csv", "data/unemployment_features.csv")

    print("\n" + "#" * 70)
    print("# STEP 4/4: MODEL TRAINING, COMPARISON & SAVING")
    print("#" * 70)
    run_full_training_pipeline("data/unemployment_features.csv")

    print("\n" + "#" * 70)
    print("# ✅ PIPELINE COMPLETE — run `streamlit run app/app.py` to launch the dashboard")
    print("#" * 70)


if __name__ == "__main__":
    main()
