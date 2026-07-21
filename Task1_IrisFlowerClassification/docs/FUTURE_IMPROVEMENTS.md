# Future Improvements

1. **Hyperparameter tuning** — use `GridSearchCV` or `RandomizedSearchCV`
   (or `Optuna`) to tune SVM's `C` and `gamma` parameters instead of using
   defaults, likely squeezing out a bit more accuracy.

2. **Model explainability** — add SHAP (SHapley Additive exPlanations)
   values to the Streamlit app so users can see *why* the model made a
   given prediction, not just the prediction itself.

3. **Larger, more diverse dataset** — the classic Iris dataset has only
   150 samples from a single 1936 study. A larger, modern, geographically
   diverse dataset would make the model more robust and realistic.

4. **Hyperparameter-tuned ensemble/stacking model** — combine multiple
   models (e.g. SVM + Logistic Regression + Random Forest) via a voting or
   stacking classifier to potentially push accuracy even higher.

5. **Automated retraining pipeline** — set up a CI/CD pipeline (e.g. GitHub
   Actions) to automatically retrain and validate the model whenever new
   data is added, with automated tests checking accuracy doesn't regress.

6. **Authentication & usage logging** — for a production deployment, add
   basic authentication and log predictions (with user consent) to monitor
   real-world model performance over time (data drift detection).

7. **Mobile-friendly / offline version** — export the model to ONNX or
   TensorFlow Lite format for use in a mobile field-data-collection app
   for botanists working without internet access.

8. **Unit tests** — add `pytest` tests for the preprocessing, feature
   engineering, and training functions to catch regressions automatically.

9. **API endpoint** — wrap the model in a lightweight FastAPI/Flask REST
   API so other applications (not just the Streamlit dashboard) can
   request predictions programmatically.

10. **Multi-language support** — internationalize the Streamlit dashboard
    for non-English-speaking users.
