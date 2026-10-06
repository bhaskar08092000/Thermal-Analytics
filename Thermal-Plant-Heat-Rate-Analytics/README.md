# Thermal Plant Heat Rate Prediction and Analytics

An end-to-end industrial data science project that predicts **gross heat rate** using thermal power plant operating parameters and translates model results into actionable performance insights.

> **Portfolio note:** The included dataset is synthetic and intended only for learning and demonstration. It does not represent confidential plant data.

## Business problem

Heat rate is a major indicator of thermal plant efficiency. A higher-than-expected heat rate can indicate efficiency deterioration, increased fuel cost, poor condenser performance, suboptimal steam parameters, or higher auxiliary consumption.

This project answers:

- What heat rate should be expected under current operating conditions?
- Which operating parameters influence predicted heat rate?
- How accurately can machine learning estimate heat rate?
- How can operators explore performance through an interactive dashboard?

## Project highlights

- Automated validation and preprocessing pipeline
- Exploratory data analysis
- Linear Regression baseline
- Random Forest regression model
- MAE, RMSE and R-squared evaluation
- Feature importance analysis
- Streamlit prediction and analytics dashboard
- Reproducible training script
- Unit tests and interview discussion notes

## Technology stack

- Python, Pandas, NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Streamlit and Plotly
- Joblib
- Pytest

## Project structure

```text
Thermal-Plant-Heat-Rate-Analytics/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
├── data/
│   └── sample_plant_data.csv
├── models/
├── screenshots/
│   └── README.md
├── test_pipeline.py
└── interview_questions.md
```

## Input features

- Unit load
- Coal gross calorific value
- Main steam temperature and pressure
- Reheat steam temperature
- Condenser vacuum
- Auxiliary power consumption
- Boiler efficiency
- Ambient temperature

## Target

`gross_heat_rate_kcal_kwh`

## Quick start

```bash
git clone https://github.com/YOUR-USERNAME/Thermal-Plant-Heat-Rate-Analytics.git
cd Thermal-Plant-Heat-Rate-Analytics
python -m venv .venv
```

Windows:

```bash
.venv\Scriptsctivate
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

Linux or macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## Model evaluation

Run `python train_model.py`. The script prints baseline and Random Forest metrics, then saves:

- `models/heat_rate_model.joblib`
- `models/metrics.json`
- `models/feature_importance.csv`

Do not publish a claimed accuracy before running the pipeline on the final dataset. Update this section using the generated metrics.

## Streamlit application

The application provides:

1. KPI summary
2. Actual-versus-predicted visualization
3. Feature-importance chart
4. Interactive single-point prediction
5. Operational interpretation of predicted heat rate

## Suggested Power BI extension

Connect `data/sample_plant_data.csv` and build:

- KPI cards: average heat rate, load, boiler efficiency, auxiliary power
- Heat rate versus load trend
- GCV versus heat rate scatter plot
- Monthly or shift-wise performance trend when timestamp data is available
- Deviation from expected heat rate
- Parameter drill-through page

## Responsible use

This model is an analytical aid, not an automatic plant-control system. Predictions should be reviewed by qualified plant personnel before operational action.

## Future improvements

- Replace synthetic data with approved anonymized plant data
- Add timestamps, unit identity, shift and equipment status
- Add XGBoost and hyperparameter optimization
- Track experiments with MLflow
- Add SHAP explanations and drift monitoring
- Deploy with Docker or Streamlit Community Cloud
- Integrate predicted-versus-actual deviation with Power BI

## Author

**Bhaskar Kewalramani**  
Senior Engineer | Thermal Power Analytics | Python | Machine Learning | Power BI

Add your LinkedIn URL and GitHub profile URL here.

## License

MIT License. See `LICENSE`.
