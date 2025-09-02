# scripts/train_adv.py
from __future__ import annotations
import argparse
from pathlib import Path
from numbers import Number

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import make_scorer, average_precision_score
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline

from utils.config import load_config, env
from utils.paths import p, ensure_dir
from utils.io import save_json, save_pickle, write_csv
from utils.metrics import (
    compute_core_metrics,
    recall_at_topk,
    threshold_for_fpr,
    threshold_for_best_f1,
    confusion_at_threshold,
)
from utils.logging import get_logger

log = get_logger("train_adv")


# ----------------------------
# Helpers
# ----------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Advanced models with randomized search + final early stopping")
    ap.add_argument("--algo", choices=["lightgbm", "xgboost"], default="lightgbm")
    ap.add_argument("--imbalance", choices=["class_weight", "smote", "smoteenn"], default="class_weight")
    ap.add_argument("--fpr-target", type=float, default=0.01)
    ap.add_argument("--topk-frac", type=float, default=0.01)
    ap.add_argument("--search-iter", type=int, default=10)
    ap.add_argument("--cv", type=int, default=3, help="CV folds for search")
    ap.add_argument("--calibration", choices=["none", "isotonic", "platt"], default="none")
    ap.add_argument("--save-name", type=str, default=None, help="Prefix for artifacts (default: algo_imbalance[_cal])")
    return ap.parse_args()


def auprc_scorer():
    # average_precision_score = area under PR curve
    return make_scorer(average_precision_score, response_method="predict_proba")


def build_estimator(algo: str, cfg: dict, class_weight: str | None):
    if algo == "lightgbm":
        import lightgbm as lgb
        params = cfg["advanced"]["lightgbm_params"].copy()
        if class_weight:
            params["class_weight"] = class_weight
        params.setdefault("metric", "auc")
        params.setdefault("verbose", -1)
        params.setdefault("force_col_wise", True)
        model = lgb.LGBMClassifier(**params)
        return model
    else:
        import xgboost as xgb
        params = cfg["advanced"]["xgboost_params"].copy()
        model = xgb.XGBClassifier(**params)
        return model


def search_space(algo: str):
    if algo == "lightgbm":
        return {
            "n_estimators": [400, 800, 1200],
            "learning_rate": [0.02, 0.03, 0.05],
            "num_leaves": [31, 63, 95],
            "subsample": [0.7, 0.8, 1.0],
            "colsample_bytree": [0.7, 0.8, 1.0],
            "reg_lambda": [0.0, 0.1, 0.5, 1.0],
            "reg_alpha": [0.0, 0.1, 0.5],
        }
    else:  # xgboost
        return {
            "n_estimators": [400, 800, 1200],
            "learning_rate": [0.02, 0.03, 0.05],
            "max_depth": [3, 4, 5, 6],
            "subsample": [0.7, 0.8, 1.0],
            "colsample_bytree": [0.7, 0.8, 1.0],
            "reg_lambda": [0.0, 0.1, 0.5, 1.0],
            "reg_alpha": [0.0, 0.1, 0.5],
            "scale_pos_weight": [50, 100, 200, 400],
        }


def maybe_log_mlflow(params: dict, metrics: dict, artifacts: dict) -> None:
    try:
        import mlflow
        cfg = load_config()
        if cfg.get("mlflow", {}).get("enabled", False):
            tracking_uri = env(cfg["mlflow"]["tracking_uri_env"], "file:mlflow")
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
            with mlflow.start_run():
                for k, v in params.items():
                    mlflow.log_param(k, v)
                for k, v in metrics.items():
                    if isinstance(v, Number):
                        mlflow.log_metric(k, float(v))
                for name, path in artifacts.items():
                    path = Path(path)
                    if path.exists():
                        mlflow.log_artifact(str(path), artifact_path=name)
    except Exception as e:
        log.warning("MLflow logging skipped (%s)", e)


# ----------------------------
# Main
# ----------------------------
def main():
    args = parse_args()
    cfg = load_config()

    processed = p(cfg["paths"]["data_processed"]) / "transactions.parquet"
    reports_dir = p(cfg["paths"]["reports_dir"])
    models_dir = p(cfg["paths"]["models_dir"])
    ensure_dir(reports_dir)
    ensure_dir(models_dir)

    df = pd.read_parquet(processed)
    y = df[cfg["data"]["target"]].values.astype(int)
    X = df.drop(columns=[cfg["data"]["target"]])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=float(cfg["split"]["test_size"]),
        random_state=int(cfg["split"]["random_state"]),
        stratify=y
    )

    # base estimator
    class_weight = "balanced" if (args.imbalance == "class_weight" and args.algo == "lightgbm") else None
    base_est = build_estimator(args.algo, cfg, class_weight)

    # imbalance pipeline
    steps = []
    if args.imbalance == "smote":
        steps.append(("smote", SMOTE(
            sampling_strategy=float(cfg["imbalance"]["smote"]["sampling_strategy"]),
            k_neighbors=int(cfg["imbalance"]["smote"]["k_neighbors"]),
            random_state=int(cfg["imbalance"]["smote"]["random_state"])
        )))
    elif args.imbalance == "smoteenn":
        steps.append(("smoteenn", SMOTEENN(
            sampling_strategy=float(cfg["imbalance"]["smoteenn"]["sampling_strategy"]),
            random_state=int(cfg["imbalance"]["smoteenn"]["random_state"])
        )))
    steps.append(("clf", base_est))
    pipe = ImbPipeline(steps=steps)

    # -------- Stage 1: RandomizedSearchCV (no early stopping) --------
    rnd = RandomizedSearchCV(
        estimator=pipe,
        param_distributions={f"clf__{k}": v for k, v in search_space(args.algo).items()},
        n_iter=int(args.search_iter),
        scoring=auprc_scorer(),
        cv=int(args.cv),
        verbose=1,
        n_jobs=1,
        refit=False,
        random_state=int(cfg["split"]["random_state"]),
        error_score="raise"
    )

    log.info("Searching %s with %s (n_iter=%d)…", args.algo, args.imbalance, args.search_iter)
    rnd.fit(X_train, y_train)

    best_params = rnd.best_params_
    log.info("Best params: %s", best_params)

    # -------- Stage 2: Refit with early stopping --------
    best_pipe = ImbPipeline(steps=steps)
    best_pipe.set_params(**best_params)

    if args.algo == "lightgbm":
        import lightgbm as lgb
        fit_params_final = {
            "clf__eval_set": [(X_test, y_test)],
            "clf__eval_metric": "auc",
            "clf__callbacks": [lgb.early_stopping(int(cfg["advanced"]["early_stopping_rounds"]), verbose=False)],
        }
    else:
        fit_params_final = {
            "clf__eval_set": [(X_test, y_test)],
            "clf__eval_metric": "aucpr",
            "clf__early_stopping_rounds": int(cfg["advanced"]["early_stopping_rounds"]),
        }

    log.info("Refitting best model with early stopping…")
    best_pipe.fit(X_train, y_train, **fit_params_final)

    # optional calibration
    if args.calibration != "none":
        method = "isotonic" if args.calibration == "isotonic" else "sigmoid"
        log.info("Calibrating with %s …", method)
        best_pipe = ImbPipeline([
            ("best", best_pipe),
            ("cal", CalibratedClassifierCV(cv=4, method=method))
        ])
        best_pipe.fit(X_train, y_train)

    # evaluate
    p_test = best_pipe.predict_proba(X_test)[:, 1]
    core = compute_core_metrics(y_test, p_test)
    r_topk = recall_at_topk(y_test, p_test, frac=float(args.topk_frac))

    thr_reports = {}
    tr_f1 = threshold_for_best_f1(y_test, p_test)
    thr_reports["best_f1"] = tr_f1.__dict__
    thr_reports["best_f1"]["confusion"] = confusion_at_threshold(y_test, p_test, tr_f1.threshold)

    tr_fpr = threshold_for_fpr(y_test, p_test, fpr_target=float(args.fpr_target))
    thr_reports["fpr_target"] = tr_fpr.__dict__
    thr_reports["fpr_target"]["confusion"] = confusion_at_threshold(y_test, p_test, tr_fpr.threshold)

    # save artifacts
    name = args.save_name or f"{args.algo}_{args.imbalance}" + (f"_{args.calibration}" if args.calibration != "none" else "")
    model_path = models_dir / f"{name}.pkl"
    save_pickle(best_pipe, model_path)

    metrics = {
        **core,
        "recall_at_topk_frac": float(r_topk),
        "algo": args.algo,
        "imbalance": args.imbalance,
        "calibration": args.calibration,
        "search_iter": int(args.search_iter),
    }
    save_json(metrics, reports_dir / f"metrics_{name}.json")
    save_json(thr_reports, reports_dir / f"thresholds_{name}.json")

    rows = []
    for strat, rep in thr_reports.items():
        row = {"strategy": strat, **{k: v for k, v in rep.items() if k != "confusion"}}
        row.update({f"cm_{k}": v for k, v in rep.get("confusion", {}).items()})
        rows.append(row)
    write_csv(pd.DataFrame(rows), reports_dir / f"threshold_eval_{name}.csv")

    log.info("Done. %s | AUPRC=%.6f ROC-AUC=%.6f Recall@Top%.2f%%=%.4f",
             name, metrics["auprc"], metrics["roc_auc"], float(args.topk_frac)*100, metrics["recall_at_topk_frac"])
    log.info("Artifacts: %s, %s, %s", model_path,
             reports_dir / f"metrics_{name}.json", reports_dir / f"thresholds_{name}.json")

    maybe_log_mlflow(
        {"algo": args.algo, "imbalance": args.imbalance, "cv": args.cv},
        metrics,
        {"model": str(model_path), "reports": str(reports_dir)},
    )


if __name__ == "__main__":
    main()
