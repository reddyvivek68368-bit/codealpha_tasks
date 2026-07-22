# Future Improvements

1. **Extend the time range** — the dataset stops in October 2020; adding
   more recent CMIE data would show the full recovery trajectory and
   any subsequent COVID waves' impact.

2. **Sector-level breakdown** — knowing which industries (manufacturing,
   services, agriculture, informal sector) lost the most jobs would let
   recommendations target specific sectors, not just states.

3. **Proper time-series forecasting** — the current model treats each
   row independently; using ARIMA, Prophet, or an LSTM that respects
   the time-ordering and seasonality of the data would likely produce
   more accurate and honest forward-looking forecasts.

4. **External economic indicators** — incorporating GDP growth, state
   government stimulus spending, or COVID case counts per state as
   features could substantially improve model R² and explanatory power.

5. **Causal analysis** — using techniques like difference-in-differences
   (comparing states that locked down at different times) to move from
   correlational findings to more defensible causal claims about the
   lockdown's specific effect.

6. **Hyperparameter tuning** — GridSearchCV/RandomizedSearchCV on the
   Random Forest's tree depth, number of estimators, and min-samples
   parameters to likely improve R² further.

7. **Confidence intervals on predictions** — using quantile regression
   or a model like LightGBM with prediction intervals, so the dashboard
   can show a plausible range, not just a single point estimate.

8. **Geospatial visualization** — using the longitude/latitude data
   (available in file 2) to build an actual map of India showing
   unemployment rate by state, rather than just a bar chart.

9. **Automated data refresh** — a scheduled pipeline that pulls updated
   CMIE data automatically and retrains the model periodically.

10. **Unit tests** — pytest coverage for the cleaning and feature
    engineering functions to catch regressions as the pipeline evolves.
