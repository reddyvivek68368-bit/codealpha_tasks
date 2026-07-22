# Project Report: Unemployment Analysis in India

**Prepared for:** CodeAlpha Data Science Internship — Task 2
**Author:** [Your Name]
**Date:** July 2026

---

## 1. Executive Summary

This project analyzes unemployment trends across Indian states using
CMIE data covering May 2019 through October 2020, with a specific focus
on the impact of the COVID-19 lockdown. A Random Forest regression model
was trained to predict unemployment rate from labour participation rate,
time, lockdown status, and state-level historical averages, achieving an
R² of 0.70. The analysis found the national average unemployment rate
rose from 9.48% to 14.92% (a 57.3% relative increase) following the
March 2020 lockdown.

## 2. Objective

To understand how COVID-19 and the resulting lockdown affected
unemployment in India — nationally, by state, and by urban/rural area —
and to build a model capable of estimating unemployment rate from related
economic indicators.

## 3. Data Understanding

Two CMIE-sourced CSV files were combined: one covering May 2019–June 2020
with an Urban/Rural split, and one covering January–October 2020 with
zone and geographic coordinate data. After merging and cleaning, the
final dataset contains 1,007 monthly observations across 28 states.

## 4. Data Cleaning & Preprocessing

- **Missing values:** 28 fully-blank trailing rows (a known artifact of
  the source file) were identified and dropped.
- **Column inconsistencies:** raw column names and string values had
  inconsistent leading/trailing whitespace; all were standardized.
- **Duplicates:** none found after merging (checked on state + date +
  area).
- **Outliers:** using a widened IQR threshold (3.0×IQR rather than the
  standard 1.5×), 11 extreme values in unemployment rate and 14 in
  labour participation rate were capped. The wider threshold was
  deliberately chosen so that genuine, large COVID-era spikes were
  preserved as real signal rather than flattened as if they were
  measurement errors.

## 5. Exploratory Data Analysis — Key Findings

- **National unemployment rate rose from 9.48% (pre-lockdown) to 14.92%
  (during/after lockdown)** — a 57.3% relative increase, peaking above
  23% in June 2020 before gradually declining through October 2020.
- The impact was uneven across states — the Top-10-states chart and the
  state×month heatmap both show some states experiencing far sharper and
  more prolonged spikes than others.
- Labour force participation also declined during the lockdown period,
  suggesting the unemployment figures may understate the true scale of
  economic disruption (people who stopped looking for work aren't
  counted as "unemployed" in the standard definition).
- Urban and rural areas were both affected, with differences in timing
  and severity visible in the Urban vs Rural comparison chart.

Charts supporting these findings: `assets/charts/01` through `08`.

## 6. Feature Engineering

Three features were engineered: a binary `is_lockdown_period` flag
(based on the 25 March 2020 lockdown date), a filled `area_filled`
category (labeling file-2-only rows as "Combined" since they lack an
Urban/Rural split), and a `state_avg_unemployment` feature (each state's
historical mean unemployment rate, used as a simple, effective numeric
encoding of state identity for the regression models).

## 7. Modeling & Evaluation

Five regression models were trained and compared on an 80/20 split with
5-fold cross-validation:

| Model | R² Score | MAE | CV Mean R² |
|---|---|---|---|
| **Random Forest** | **0.703** | 3.42 | 0.674 |
| Decision Tree | 0.662 | 3.61 | 0.569 |
| Gradient Boosting | 0.607 | 3.91 | 0.581 |
| Ridge Regression | 0.469 | 4.62 | 0.435 |
| Linear Regression | 0.469 | 4.62 | 0.435 |

**Selected model: Random Forest Regressor.** It achieved the best R² and
cross-validated performance, roughly 50% better than the linear models —
indicating meaningful non-linear relationships between the features and
unemployment rate that linear models cannot capture. An MAE of 3.42
percentage points means predictions are, on average, within about 3.4
points of the true unemployment rate.

**Limitation:** with a relatively small feature set covering a single,
unprecedented economic shock, this model is a solid analytical tool but
should not be treated as a precise forecasting instrument for future,
different economic conditions.

## 8. Business Recommendations

1. Target relief programs and job-creation initiatives at the states and
   time periods showing the sharpest unemployment increases rather than
   applying uniform national policy.
2. Track labour force participation alongside unemployment rate, since a
   drop in participation can mask the true scale of economic distress.
3. Build economic contingency triggers into future public health crisis
   response plans, given how quickly and sharply unemployment responded
   to the lockdown.

## 9. Deliverables

- Cleaned & merged dataset, feature-engineered dataset
- 11 EDA and evaluation charts
- Trained regression model + metadata
- Interactive Streamlit dashboard with COVID-impact analysis and live
  prediction tool
- Full source code, documentation, and this report

## 10. Conclusion

This project delivers a complete, reproducible analysis of how COVID-19
affected unemployment in India, quantifying the impact precisely (a
57.3% relative increase) and building a working predictive model (R² =
0.70), packaged into an interactive dashboard for further exploration.
