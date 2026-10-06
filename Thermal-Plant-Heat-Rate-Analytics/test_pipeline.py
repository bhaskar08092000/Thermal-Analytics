import pandas as pd
from train_model import DATA, FEATURES, TARGET

def test_dataset_schema():
    df=pd.read_csv(DATA)
    assert set(FEATURES+[TARGET]).issubset(df.columns)
    assert len(df)>=100
    assert df[FEATURES+[TARGET]].notna().all().all()

def test_ranges():
    df=pd.read_csv(DATA)
    assert df["load_mw"].between(0,1000).all()
    assert df["boiler_efficiency_pct"].between(0,100).all()
    assert (df[TARGET]>0).all()
