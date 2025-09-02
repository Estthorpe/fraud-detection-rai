import pandas as pd
import json

# Load parquet
df = pd.read_parquet("data/processed/transactions.parquet")
print("Shape:", df.shape)
print("First 5 cols:", df.columns[:5].tolist())
print("Fraud ratio:", df["Class"].mean())

# Peek readiness report
with open("reports/readiness_report.json") as f:
    rep = json.load(f)
print("\nReadiness Report (truncated):")
print(json.dumps(rep, indent=2)[:800])
