from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "demand.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA, parse_dates=["date"])
series = (
    df.groupby("date", as_index=False)["demand"].sum()
      .sort_values("date")
      .set_index("date")["demand"]
)

split = int(len(series) * 0.8)
train, test = series.iloc[:split], series.iloc[split:]

baseline_pred = pd.Series(train.iloc[-7:].mean(), index=test.index)

sarima = SARIMAX(train, order=(1,1,1), seasonal_order=(1,0,1,7),
                 enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
sarima_pred = sarima.forecast(len(test))
sarima_pred.index = test.index

tmp = series.to_frame("y")
for lag in [1, 7, 14, 28]:
    tmp[f"lag_{lag}"] = tmp["y"].shift(lag)
tmp["dow"] = tmp.index.dayofweek
tmp["month"] = tmp.index.month
tmp = tmp.dropna()

train_ml = tmp.loc[tmp.index <= train.index[-1]]
test_ml = tmp.loc[tmp.index > train.index[-1]]
features = [c for c in tmp.columns if c != "y"]

model = GradientBoostingRegressor(random_state=42)
model.fit(train_ml[features], train_ml["y"])
gb_pred = pd.Series(model.predict(test_ml[features]), index=test_ml.index)

eval_test = test.loc[gb_pred.index]
baseline_eval = baseline_pred.loc[eval_test.index]
sarima_eval = sarima_pred.loc[eval_test.index]

def metrics(name, y, p):
    mae = mean_absolute_error(y, p)
    rmse = mean_squared_error(y, p) ** 0.5
    mape = (np.abs((y-p)/y).replace([np.inf,-np.inf], np.nan).dropna()).mean()*100
    return {"model": name, "MAE": round(mae,2), "RMSE": round(rmse,2), "MAPE_pct": round(mape,2)}

results = pd.DataFrame([
    metrics("7-day baseline", eval_test, baseline_eval),
    metrics("SARIMA", eval_test, sarima_eval),
    metrics("Gradient Boosting", eval_test, gb_pred),
])
results.to_csv(OUT/"forecast_metrics.csv", index=False)
print(results)

plt.figure(figsize=(10,5))
plt.plot(eval_test.index, eval_test.values, label="Actual")
plt.plot(sarima_eval.index, sarima_eval.values, label="SARIMA")
plt.plot(gb_pred.index, gb_pred.values, label="Gradient Boosting")
plt.legend()
plt.title("Demand Forecast: Actual vs Predictions")
plt.xlabel("Date")
plt.ylabel("Units")
plt.tight_layout()
plt.savefig(OUT/"forecast_comparison.png", dpi=180)
print(f"Saved outputs to {OUT}")
