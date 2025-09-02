from __future__ import annotations
from pathlib import Path
import json 
import glob
import pandas as pd
import datetime as dt

REPORTS = Path("reports")

def _load_metrics(name: str) -> dict:
    with open(REPORTS / f"metrics_{name}.json", "r") as f:
        return json.load(f)

def _load_thresholds(name: str) -> dict:
    with open(REPORTS / f"thresholds_{name}.json", "r") as f:
        return json.load(f)

def _safe(v, default=None):
    return v if v is not None else default

def collect_rows() -> pd.DataFrame:
    rows = []
    for mp in glob.glob(str(REPORTS / "metrics_*.json")):
        name = Path(mp).stem.replace("metrics_", "")
        # thresholds file may be missing for partial runs; skip if not present
        thr_path = REPORTS / f"thresholds_{name}.json"
        if not thr_path.exists():
            continue

        m = _load_metrics(name)
        t = _load_thresholds(name)
        fpr = t.get("fpr_target", {})
        f1  = t.get("best_f1", {})

        row = {
            "name": name,
            # model meta (if present in metrics)
            "algo": m.get("algo"),
            "imbalance": m.get("imbalance"),
            "calibration": m.get("calibration"),
            "search_iter": m.get("search_iter"),

            # overall metrics
            "auprc": m.get("auprc"),
            "roc_auc": m.get("roc_auc"),
            "brier": m.get("brier", None),
            "recall_at_topk_frac": m.get("recall_at_topk_frac"),

            # FPR-target operating point
            "fpr_target": _safe(fpr.get("fpr")),
            "recall_at_fpr_target": _safe(fpr.get("recall")),
            "precision_at_fpr_target": _safe(fpr.get("precision")),
            "threshold_at_fpr_target": _safe(fpr.get("threshold")),
            "cm_tn_fpr_target": _safe(fpr.get("confusion", {}).get("tn")),
            "cm_fp_fpr_target": _safe(fpr.get("confusion", {}).get("fp")),
            "cm_fn_fpr_target": _safe(fpr.get("confusion", {}).get("fn")),
            "cm_tp_fpr_target": _safe(fpr.get("confusion", {}).get("tp")),

            # Best-F1 operating point (handy for comparison)
            "f1_best": _safe(f1.get("f1")),
            "recall_best_f1": _safe(f1.get("recall")),
            "precision_best_f1": _safe(f1.get("precision")),
            "threshold_best_f1": _safe(f1.get("threshold")),
            "cm_tn_best_f1": _safe(f1.get("confusion", {}).get("tn")),
            "cm_fp_best_f1": _safe(f1.get("confusion", {}).get("fp")),
            "cm_fn_best_f1": _safe(f1.get("confusion", {}).get("fn")),
            "cm_tp_best_f1": _safe(f1.get("confusion", {}).get("tp")),
        }
        rows.append(row)

    if not rows:
        raise SystemExit("No metrics/thresholds pairs found in reports/. Run training first.")

    df = pd.DataFrame(rows)
    # nice ordering: by AUPRC desc then recall at FPR target desc
    df = df.sort_values(by=["auprc", "recall_at_fpr_target"], ascending=[False, False]).reset_index(drop=True)
    return df

def write_csv_and_md(df: pd.DataFrame):
    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    csv_path = REPORTS / "variant_comparison.csv"
    df.to_csv(csv_path, index=False)

    # small markdown table (top 8 rows for readability)
    md_path = REPORTS / "model_report.md"
    show = df[[
        "name","algo","imbalance","calibration","auprc","roc_auc",
        "recall_at_topk_frac","fpr_target","recall_at_fpr_target","threshold_at_fpr_target"
    ]].copy()
    show.rename(columns={
        "recall_at_topk_frac":"recall@topk",
        "fpr_target":"FPR@target",
        "recall_at_fpr_target":"recall@FPR",
        "threshold_at_fpr_target":"thr@FPR"
    }, inplace=True)
    show["recall@topk"] = show["recall@topk"].map(lambda x: f"{x:.4f}" if pd.notnull(x) else "")
    show["auprc"] = show["auprc"].map(lambda x: f"{x:.6f}" if pd.notnull(x) else "")
    show["roc_auc"] = show["roc_auc"].map(lambda x: f"{x:.6f}" if pd.notnull(x) else "")
    show["FPR@target"] = show["FPR@target"].map(lambda x: f"{x:.4f}" if pd.notnull(x) else "")
    show["recall@FPR"] = show["recall@FPR"].map(lambda x: f"{x:.4f}" if pd.notnull(x) else "")
    show["thr@FPR"] = show["thr@FPR"].map(lambda x: f"{x:.6f}" if pd.notnull(x) else "")

    top = show.head(8)

    lines = []
    lines.append(f"# Model Comparison Report\n")
    lines.append(f"_Generated: {ts}_\n")
    lines.append(f"**Primary objective**: Maximize AUPRC, and select operating threshold with FPR ≤ target.\n")
    lines.append(f"\n## Summary Table\n")
    lines.append(top.to_markdown(index=False))
    lines.append("\n## Notes\n")
    lines.append("- AUPRC (area under PR) is the primary metric on imbalanced data.")
    lines.append("- `recall@FPR` is measured at the configured FPR target (e.g., 1%).")
    lines.append("- `recall@topk` uses the configured top-k fraction (e.g., top 1% most suspicious transactions).")
    lines.append("- See `variant_comparison.csv` for the full dataset and confusion matrices.\n")

    md = "\n".join(lines)
    md_path.write_text(md, encoding="utf-8")
    print(f"✔ Wrote: {csv_path}")
    print(f"✔ Wrote: {md_path}")

def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    df = collect_rows()
    write_csv_and_md(df)

if __name__ == "__main__":
    main()



