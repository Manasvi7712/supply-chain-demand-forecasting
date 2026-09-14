from pathlib import Path
import numpy as np
import pandas as pd

np.random.seed(42)
out = Path(__file__).resolve().parents[1] / "data" / "demand.csv"

dates = pd.date_range("2024-01-01", periods=365, freq="D")
skus = [f"SKU_{i:03d}" for i in range(1, 9)]
nodes = [f"DC_{i:02d}" for i in range(1, 7)]

rows = []
for sku_i, sku in enumerate(skus):
    base = 35 + 7 * sku_i
    for node_i, node in enumerate(nodes):
        node_factor = 0.85 + node_i * 0.06
        trend = np.linspace(0, 8 + sku_i, len(dates))
        weekly = 8 * np.sin(np.arange(len(dates)) * 2*np.pi/7)
        annual = 5 * np.sin(np.arange(len(dates)) * 2*np.pi/365)
        noise = np.random.normal(0, 5, len(dates))
        demand = np.maximum(0, base*node_factor + trend + weekly + annual + noise).round().astype(int)
        for d, q in zip(dates, demand):
            rows.append((d, sku, node, int(q)))

df = pd.DataFrame(rows, columns=["date", "sku", "node", "demand"])
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
print(f"Wrote {len(df):,} rows to {out}")
