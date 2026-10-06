# Interview Questions and Project Talking Points

## 60-second project explanation

I developed an end-to-end machine learning application to estimate gross heat rate from thermal plant operating parameters. I created a reproducible preprocessing and training pipeline, compared Linear Regression with Random Forest, evaluated performance using MAE, RMSE and R-squared, and deployed the final model through a Streamlit dashboard. The demonstration data is synthetic, while the architecture is designed so that approved plant data can replace it later.

## Questions

### Why predict heat rate?
Heat rate connects fuel energy input with electrical output. Predicting expected heat rate helps identify deviations that may warrant engineering investigation.

### Why use a baseline model?
A simple baseline shows whether a more complex model creates meaningful improvement and makes the modelling decision defensible.

### Why Random Forest?
It can capture nonlinear relationships and interactions, requires limited scaling, and offers feature importance. Its disadvantages include lower interpretability than linear models and weak extrapolation beyond the training range.

### Why MAE and RMSE?
MAE is easy to interpret in target units. RMSE penalizes larger errors more heavily. Reporting both gives a more balanced assessment.

### How did you prevent data leakage?
The train-test split occurs before fitting preprocessing and the model. Any transformation that learns from data should be placed inside the pipeline.

### What would you change with time-series plant data?
I would use chronological splitting rather than random splitting, engineer lag and rolling-window features, validate across operating periods, and monitor concept drift.

### How would you productionize it?
I would add a validated ingestion layer, experiment tracking, model registry, API or batch scoring, data-drift monitoring, role-based access and retraining controls.

### What are the model limitations?
The included sample is synthetic; predictions cannot be treated as operational recommendations. Real deployment requires approved historical data, instrument-quality checks, time-aware validation, domain review and ongoing monitoring.
