# scripts/export_confusion.py
import argparse, json
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="model save-name, e.g., lightgbm_smote")
    ap.add_argument("--which", choices=["fpr_target","best_f1"], default="fpr_target")
    args = ap.parse_args()

    t = json.load(open(f"reports/thresholds_{args.name}.json"))
    d = t[args.which]["confusion"]
    out = Path(f"reports/confusion_{args.name}_{args.which}.csv")
    pd.DataFrame([d]).to_csv(out, index=False)
    print("Saved", out)

if __name__ == "__main__":
    main()
