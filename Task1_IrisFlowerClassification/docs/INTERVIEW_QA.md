# Interview Questions & Answers — Iris Flower Classification Project

### Q1: Why is the Iris dataset commonly used to learn machine learning?
It's small (150 rows), has no missing values, is perfectly balanced across
3 classes, and has clear, learnable patterns — making it ideal for
practicing the full ML workflow without data-quality distractions.

### Q2: What type of machine learning problem is this?
Supervised **multi-class classification** — the model learns from labeled
examples (measurements → known species) and predicts one of three discrete
categories for new data.

### Q3: Why did you scale the features before training?
Distance-based and margin-based algorithms (KNN, SVM, Logistic Regression)
are sensitive to feature scale — a feature measured in larger numbers can
dominate the distance calculation even if it's not more informative.
`StandardScaler` transforms every feature to mean=0, std=1, putting them
on equal footing. Tree-based models don't need this, but scaling doesn't
hurt them, so we applied it uniformly via a `Pipeline` for simplicity and
to prevent data leakage (the scaler is fit only on training data).

### Q4: How did you avoid data leakage during preprocessing?
By wrapping the scaler and classifier together in a single scikit-learn
`Pipeline` and fitting that pipeline only on the training split. This
guarantees the scaler's mean/std are learned from training data alone, and
are then applied (not re-fit) to the test data.

### Q5: Why did you use stratified train-test split and stratified cross-validation?
Stratification ensures each split preserves the original class proportions
(50/50/50). Without it, a random split on a small dataset could
accidentally leave one species under-represented in the test set, making
the evaluation unreliable.

### Q6: Why compare 7 different models instead of just picking one?
No single algorithm is best for every dataset (the "no free lunch"
theorem). Comparing several — spanning linear (Logistic Regression),
distance-based (KNN), margin-based (SVM), and tree-based (Decision Tree,
Random Forest, Gradient Boosting) approaches — lets the data itself decide
which inductive bias fits best, rather than guessing.

### Q7: Why did SVM win, and what does that tell you about the data?
SVM (with an RBF kernel) finds a boundary that maximizes the margin between
classes. It won because the classes are almost linearly/smoothly separable
in feature space, with only Versicolor and Virginica overlapping slightly
— exactly the kind of geometry SVMs handle very well, especially on small,
low-noise datasets where more flexible models (like Gradient Boosting) can
overfit.

### Q8: What metrics did you use besides accuracy, and why?
Precision, recall, and F1-score (macro-averaged across the 3 classes), plus
5-fold cross-validation accuracy. Accuracy alone can be misleading if
classes were imbalanced (they aren't here, but it's still good practice);
macro-averaging treats every class equally regardless of size, and
cross-validation checks that performance isn't just a lucky single split.

### Q9: How would this approach change with a much larger or messier dataset?
I'd add more rigorous missing-value imputation strategies, feature
selection to control dimensionality, hyperparameter tuning (GridSearchCV /
RandomizedSearchCV) rather than default parameters, and likely move to
more scalable models (e.g. XGBoost/LightGBM) and a proper experiment
tracking tool (e.g. MLflow) given more models/iterations to manage.

### Q10: How did you handle outliers, and why not just delete them?
Used the IQR method to detect values beyond 1.5×IQR from Q1/Q3, then
**capped** (Winsorized) them to the boundary rather than deleting the rows.
With only 150 samples, deleting rows wastes data; capping keeps the sample
(and its other correct measurements) while limiting the extreme value's
influence.

### Q11: Why build a Streamlit app instead of just a Jupyter notebook?
A notebook is great for exploration, but a deployed app makes the model
usable by non-technical stakeholders (e.g. a botanist could enter
measurements and get an instant prediction), and demonstrates the ability
to ship a complete product, not just an analysis.

### Q12: What would you improve if you had more time?
See `docs/FUTURE_IMPROVEMENTS.md` — hyperparameter tuning, a larger/more
diverse dataset, model explainability (SHAP values), authentication/logging
for the app, and automated retraining pipelines.
