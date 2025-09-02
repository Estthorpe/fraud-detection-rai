from __future__ import annotations
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import utils
from utils.config import load_config
from utils.paths import p, ensure_dir
from utils.io import write_parquet
from utils.logging import get_logger

log = get_logger("prepare_data")

EXPECTED_COLS = {"Time"} | {f"V{i}" for i in range(1, 29)} | {"Amount", "Class"}

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Clean + validate credit card fraud dataset")
    ap.add_argument("--drop-time", action="store_true",
                    help="Drop 'Time' to avoid temporal leakage influence in models.")
    ap.add_argument("--output", type=str, default="transactions.parquet",
                    help="Output parquet filename (under data/processed)")
    return ap.parse_args()



def schema_checks(df: pd.DataFrame) -> dict:
    cols = set(df.columns)
    missing = sorted(list(EXPECTED_COLS - cols))
    extra = sorted(list(cols - EXPECTED_COLS))
    report = {
        "n_rows": len(df),
        "n_cols": df.shape[1],
        "has_all_expected": len(missing) == 0,
        "missing_columns": missing,
        "extra_columns": extra,
    }
    return report

def null_duplicate_class_checks(df: pd.DataFrame, target: str) -> dict:
    nulls = df.isna().sum().to_dict()
    dup_count = int(df.duplicated().sum())
    class_counts = df[target].value_counts(dropna=False).to_dict()
    class_ratio = {str(k): float(v) / len(df) for k, v in class_counts.items()}
    return {
        "nulls": nulls,
        "duplicate_rows": dup_count,
        "class_counts": class_counts,
        "class_ratio": class_ratio,
    }

def stratified_split_sanity(df: pd.DataFrame, target: str, test_size: float, random_state: int) -> dict:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    y = df[target].values
    for train_idx, test_idx in splitter.split(df, y):
        y_train = y[train_idx]
        y_test = y[test_idx]
        c_train = pd.Series(y_train).value_counts(normalize=True).to_dict()
        c_test = pd.Series(y_test).value_counts(normalize=True).to_dict()
        return {
            "test_size": test_size,
            "class_ratio_train": {str(k): float(v) for k, v in c_train.items()},
            "class_ratio_test": {str(k): float(v) for k, v in c_test.items()},
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
        }
    return {}

def main() -> None:
    args = parse_args()
    cfg = load_config()

    raw_dir = p(cfg["paths"]["data_raw"])
    processed_dir = p(cfg["paths"]["data_processed"])
    reports_dir = p(cfg["paths"]["reports_dir"])
    ensure_dir(processed_dir)
    ensure_dir(reports_dir)

    csv_path = raw_dir / "creditcard.csv"
    if not csv_path.exists():
        log.error("Raw file not found at %s. Run scripts/download_kaggle.py first.", csv_path)
        raise SystemExit(1)

    log.info("Loading %s ...", csv_path)
    df = pd.read_csv(csv_path)

    # Schema checks
    sch = schema_checks(df)
    if not sch["has_all_expected"]:
        log.warning("Missing expected columns: %s", sch["missing_columns"])
    if sch["extra_columns"]:
        log.info("Extra columns present: %s", sch["extra_columns"])

    # Nulls/duplicates/class ratio
    qc = null_duplicate_class_checks(df, target=cfg["data"]["target"])

    # Drop duplicate full rows (safe)
    if qc["duplicate_rows"] > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        log.info("Dropped %d duplicate rows.", qc["duplicate_rows"])

    # Leakage safeguard notes:
    # - 'Class' is target; no target-derived features present.
    # - 'Time' is seconds from first tx; it can encode sequence/shift information.
    #   For baseline comparability and to avoid accidental temporal leakage patterns,
    #   we default to dropping 'Time' when --drop-time is passed (recommended).
    if args.drop_time and "Time" in df.columns:
        df = df.drop(columns=["Time"])
        log.info("Dropped 'Time' column to reduce temporal leakage influence.")

    # Ensure numeric dtypes (all features in this dataset are numeric)
    df = df.apply(pd.to_numeric, errors="coerce")
    if df.isna().sum().sum() > 0:
        # If any coercion created NaNs (shouldn't happen for this dataset), drop safely
        log.warning("Found NaNs after numeric coercion; dropping rows with NaNs.")
        df = df.dropna().reset_index(drop=True)

    # Save canonical parquet
    out_path = processed_dir / args.output
    write_parquet(df, out_path)
    log.info("Saved canonical dataset: %s (rows=%d, cols=%d)", out_path, len(df), df.shape[1])

    # Stratified split sanity (we do not save splits here; training will do it)
    split_cfg = cfg["split"]
    split_report = stratified_split_sanity(
        df, target=cfg["data"]["target"],
        test_size=float(split_cfg["test_size"]),
        random_state=int(split_cfg["random_state"])
    )

    # Persist a readiness report
    readiness = {
        "schema": sch,
        "qc": qc,
        "stratified_split_check": split_report,
        "leakage_safeguards": {
            "dropped_time": bool(args.drop_time),
            "note": "Time dropped to avoid temporal leakage influence; model will not see sequence index."
        }
    }
    readiness_path = reports_dir / "readiness_report.json"
    with open(readiness_path, "w", encoding="utf-8") as f:
        json.dump(readiness, f, indent=2)
    log.info("Wrote readiness report to %s", readiness_path)

if __name__ == "__main__":
    main()