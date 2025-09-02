from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from utils.config import load_config, env
from utils.paths import p, ensure_dir
from utils.io import write_csv, save_json, save_pickle
from utils.metrics import (
    compute_core_metrics,
    recall_at_topk,
    threshold_for_fpr,
    threshold_for_best_f1,
    confusion_at_threshold,
)
from utils.logging import get_logger

log = get_logger("train")

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train models")
    ap.add_argument("--model", choices=["baseline"], default="baseline")
    ap.add_argument("--save-split", action="store_true", help="Persist train/test indices for reproducibility")
    ap.add_argument("--fpr-target", type=float, default=None, help="Constraint for threshold selection, e.g., 0.01")
    ap.add_argument("--topk-frac", type=float, default=0.01, help="Top-k fraction for Recall@Topk")
    return ap.parse_args()

def maybe_log_mlflow(params: dict, metrics: dict, artifacts: dict) -> None:
    try:
        import mlflow
        cfg = load_config()
        if cfg.get("mlflow", {}).get("enabled", False):
            from numbers import Number

            tracking_uri = env(cfg["mlflow"]["tracking_uri_env"], "file:mlflow")
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

            # keep only numeric metrics
            numeric_metrics = {k: float(v) for k, v in metrics.items() if isinstance(v, Number)}

            with mlflow.start_run():
                for k, v in params.items():
                    mlflow.log_param(k, v)
                for k, v in metrics.items():
                    mlflow.log_metric(k, float(v))
                for name, path in artifacts.items():
                    pth = Path(path)
                    if pth.exists():
                        mlflow.log_artifact(str(path), artifact_path=name)
    except Exception as e:
        log.warning("MLflow logging skipped (%s)", e)

def main() -> None:
    args = parse_args()
    cfg = load_config()

    processed = p(cfg["paths"]["data_processed"]) / "transactions.parquet"
    reports_dir = p(cfg["paths"]["reports_dir"])
    models_dir = p(cfg["paths"]["models_dir"])
    ensure_dir(reports_dir)
    ensure_dir(models_dir)

    df = pd.read_parquet(processed)
    target = cfg["data"]["target"]
    y = df[target].values.astype(int)
    X = df.drop(columns=[target])

    num_cols = list(X.columns)

    #Stratified split for reproducibility
    test_size = float(cfg["split"]["test_size"])
    random_state = int(cfg["split"]["random_state"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    if args.save_split:
    #Persist  row indices masks (useful to align with future models)
        idx = np.arange(len(df))
        train_mask = np.isin(idx, X_train.index.values)
        test_mask = np.isin(idx, X_test.index.values)
        write_csv(
            pd.DataFrame({"train_mask": train_mask.astype(int), "test_mask": test_mask.astype(int)}),
                reports_dir / "split_indices.csv"
    )
    

# Baseline pipeline: scale -> logistic regression (class_weight balanced)
    scaler = StandardScaler()
    logreg = LogisticRegression(
        C=float(cfg["baseline"]["logreg"]["C"]),
        class_weight=cfg["baseline"]["logreg"]["class_weight"],
        penalty=cfg["baseline"]["logreg"]["penalty"],
        solver=cfg["baseline"]["logreg"]["solver"],
        max_iter=int(cfg["baseline"]["logreg"]["max_iter"]),
    )
    pre = ColumnTransformer(transformers=[("num", scaler, num_cols)], remainder="drop")
    pipe = Pipeline(steps=[("pre", pre), ("clf", logreg)])

    log.info("Training baseline Logistic Regression…")
    pipe.fit(X_train, y_train)

    p_test = pipe.predict_proba(X_test)[:, 1]

    #core metrics

    core =  compute_core_metrics(y_test, p_test)
    r_topk = recall_at_topk(y_test, p_test, frac=float(args.topk_frac))

    thr_reports = {}
    tr_f1 = threshold_for_best_f1(y_test, p_test)
    thr_reports["best_f1"] = tr_f1.__dict__
    thr_reports["best_f1"]["confusion"] = confusion_at_threshold(y_test, p_test, threshold=tr_f1.threshold)

    if args.fpr_target is not None:
        tr_fpr = threshold_for_fpr(y_test, p_test, fpr_target=float(args.fpr_target))
        thr_reports["fpr_target"] = tr_fpr.__dict__
        thr_reports["fpr_target"]["confusion"] = confusion_at_threshold(y_test, p_test, threshold=tr_fpr.threshold)

    # Persist artifacts
    save_pickle(pipe, models_dir / "baseline_pipeline.pkl")
    metrics = {
        **core,
        "recall_at_topk_frac": float(r_topk),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        }
    
    save_json(metrics, reports_dir / "metrics_baseline.json")
    save_json(thr_reports, reports_dir / "thresholds_baseline.json")

    # Flat CSV for Power BI later
    rows = []
    for name, rep in thr_reports.items():
        row = {"strategy": name, **{k: v for k, v in rep.items() if k != "confusion"}}
        row.update({f"cm_{k}": v for k, v in rep.get("confusion", {}).items()})
        rows.append(row)
    if rows:
        pd.DataFrame(rows).to_csv(reports_dir / "threshold_eval_baseline.csv", index=False)

    # Optional MLflow logging
    mlflow_params = {
        "model": "logreg",
        "scaler": "standard",
        "class_weight": cfg["baseline"]["logreg"]["class_weight"],
        "test_size": test_size,
        "random_state": random_state,
        "topk_frac": float(args.topk_frac),
        "fpr_target": float(args.fpr_target) if args.fpr_target is not None else None,
    }
    mlflow_metrics = metrics.copy()
    if "fpr_target" in thr_reports:
        for k, v in thr_reports["fpr_target"].items():
            if isinstance(v, (int, float)):
                mlflow_metrics[f"thr_fpr_{k}"] = float(v)
    for k, v in thr_reports["best_f1"].items():
        if isinstance(v, (int, float)):
            mlflow_metrics[f"thr_f1_{k}"] = float(v)
    maybe_log_mlflow(mlflow_params, mlflow_metrics, {
        "reports": str(reports_dir),
        "models": str(models_dir),
    })

    log.info(
        "Baseline complete. AUPRC=%.6f ROC-AUC=%.6f Recall@Top%.2f%%=%.4f",
        metrics["auprc"], metrics["roc_auc"], float(args.topk_frac)*100, metrics["recall_at_topk_frac"]
    )
    log.info(
        "Artifacts saved: %s | %s | %s",
        models_dir / "baseline_pipeline.pkl",
        reports_dir / "metrics_baseline.json",
        reports_dir / "thresholds_baseline.json"
    )
 

if __name__ == "__main__":
    main()
