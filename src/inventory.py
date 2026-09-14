from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "demand.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA, parse_dates=["date"])
recent = df[df["date"] >= df["date"].max() - pd.Timedelta(days=90)]

summary = recent.groupby(["sku","node"])["demand"].agg(["mean","std"]).reset_index()
summary["lead_time_days"] = 7
service_z = 1.65
summary["safety_stock"] = (service_z * summary["std"] * np.sqrt(summary["lead_time_days"])).round().astype(int)
summary["reorder_point"] = (summary["mean"] * summary["lead_time_days"] + summary["safety_stock"]).round().astype(int)
summary = summary.rename(columns={"mean":"avg_daily_demand","std":"demand_std"})
summary.to_csv(OUT/"inventory_recommendations.csv", index=False)
print(summary.head(12).to_string(index=False))
