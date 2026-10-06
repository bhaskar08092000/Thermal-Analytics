from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data"/"sample_plant_data.csv"
OUT=ROOT/"models"
TARGET="gross_heat_rate_kcal_kwh"
FEATURES=["load_mw","coal_gcv_kcal_kg","main_steam_temp_c","main_steam_pressure_mpa","reheat_temp_c","condenser_vacuum_bar","aux_power_pct","boiler_efficiency_pct","ambient_temp_c"]

def evaluate(name, model, X_test, y_test):
    p=model.predict(X_test)
    return {"model":name,"MAE":round(mean_absolute_error(y_test,p),3),"RMSE":round(mean_squared_error(y_test,p)**0.5,3),"R2":round(r2_score(y_test,p),4)}

def main():
    df=pd.read_csv(DATA)
    missing=set(FEATURES+[TARGET])-set(df.columns)
    if missing: raise ValueError(f"Missing columns: {sorted(missing)}")
    X,y=df[FEATURES],df[TARGET]
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42)
    baseline=Pipeline([("imputer",SimpleImputer(strategy="median")),("model",LinearRegression())])
    forest=Pipeline([("imputer",SimpleImputer(strategy="median")),("model",RandomForestRegressor(n_estimators=350,min_samples_leaf=2,random_state=42,n_jobs=-1))])
    baseline.fit(X_train,y_train); forest.fit(X_train,y_train)
    results=[evaluate("Linear Regression",baseline,X_test,y_test),evaluate("Random Forest",forest,X_test,y_test)]
    OUT.mkdir(exist_ok=True)
    final_model=forest.named_steps["model"]
    joblib.dump(forest,OUT/"heat_rate_model.joblib")
    pd.DataFrame({"feature":FEATURES,"importance":final_model.feature_importances_}).sort_values("importance",ascending=False).to_csv(OUT/"feature_importance.csv",index=False)
    (OUT/"metrics.json").write_text(json.dumps(results,indent=2))
    print(pd.DataFrame(results).to_string(index=False))
    print(f"Saved model to {OUT/'heat_rate_model.joblib'}")
if __name__=="__main__": main()
