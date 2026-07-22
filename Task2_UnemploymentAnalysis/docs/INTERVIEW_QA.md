# Interview Questions & Answers — Unemployment Analysis Project

### Q1: Why did you combine two separate CSV files instead of using just one?
Each file covers a different, only partially-overlapping time range and
has different extra columns (one has Urban/Rural, the other has Zone and
coordinates). Combining them gives a longer, more complete time series
(May 2019–October 2020) than either file alone, which is essential for
seeing the full before/after picture of the lockdown's impact.

### Q2: How did you handle the two columns that were both named "Region" in the second file?
Pandas automatically renames a duplicate column name by appending ".1"
on read, so the second "Region" column became "Region.1". I renamed it
explicitly to "zone" since it actually represents a broader geographic
grouping (e.g. South, North, East) rather than the state name.

### Q3: Why did you use a wider outlier threshold (3×IQR) instead of the standard 1.5×IQR?
The standard 1.5×IQR threshold is tuned for detecting genuine data
errors in relatively stable data. But this dataset contains a real,
extreme, well-documented event (the COVID lockdown) that legitimately
pushed unemployment rates far outside their normal range. Using the
standard threshold would have flagged and flattened real pandemic-era
spikes as if they were mistakes. Widening the threshold to 3×IQR let me
still catch genuine data entry errors while preserving the real signal.

### Q4: Why is this a regression problem rather than classification?
Unemployment rate is a continuous percentage (e.g. 7.32%), not a
category — so the goal is to predict a number, which is what regression
models do, unlike Task 1's species classification which predicted a
category.

### Q5: Why did Random Forest outperform the linear models by such a wide margin?
An R² gap of roughly 0.47 (linear) vs 0.70 (Random Forest) suggests the
relationship between the features and unemployment rate is meaningfully
non-linear — for example, the effect of the lockdown flag likely
interacts differently with different states' baseline unemployment
levels, which tree-based models can capture naturally but linear models
cannot without manually engineering interaction terms.

### Q6: How did you encode the "state" feature for the regression models?
I used target encoding: each state's historical average unemployment
rate becomes its numeric feature value. This is simpler than one-hot
encoding 28 separate state columns and captures each state's typical
unemployment level directly as useful signal, though it does carry some
risk of leakage if not handled carefully with cross-validation (which is
why 5-fold CV was used to sanity-check performance).

### Q7: What does an R² of 0.70 actually mean, and is that "good"?
R² of 0.70 means the model explains about 70% of the variance in
unemployment rate using just 5 features. That's a solid result for a
real-world economic dataset with only ~1,000 rows covering a single
unprecedented shock, though it's honestly not high enough for precise
forecasting — it's better suited for explanatory analysis than for
making high-stakes predictions.

### Q8: What's the difference between unemployment rate and labour force participation rate, and why do both matter?
Unemployment rate measures the share of people *actively in the labour
force* who don't have jobs. Labour force participation rate measures
what share of the *eligible population* is even trying to work. If
participation drops (people give up looking for work), unemployment rate
can look artificially better even though the true jobs situation hasn't
improved — which is exactly a pattern seen in this dataset during the
lockdown period.

### Q9: How would you validate that the lockdown, not some other factor, actually caused the unemployment spike?
This analysis shows correlation (unemployment rose right after the
lockdown date), not proven causation. To strengthen the causal claim,
I'd want to compare against a counterfactual (e.g. seasonal unemployment
patterns from prior years without a lockdown) or look at whether states
that entered lockdown at different times saw their spikes shift
accordingly — which would be a stronger sign that the lockdown itself,
not some unrelated trend, drove the change.

### Q10: What would you improve given more time/data?
See `docs/FUTURE_IMPROVEMENTS.md` — sector-level breakdowns, longer
time coverage (data collection stopped at the end of 2020), external
economic indicators (GDP, sector output), and proper time-series
forecasting methods (e.g. ARIMA/Prophet) rather than treating each row
as independent.
