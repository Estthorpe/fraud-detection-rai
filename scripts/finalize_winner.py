# scripts/finalize_winner.py
import argparse, json
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="winner model, e.g., lightgbm_smote")
    ap.add_argument("--reason", default="Highest AUPRC with recall at FPR target")
    args = ap.parse_args()

    path = Path("reports/final_model.json")
    payload = {"winner": args.name, "reason": args.reason}
    path.write_text(json.dumps(payload, indent=2))
    print("Saved", path)

if __name__ == "__main__":
    main()
