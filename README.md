## Model Variants & Selection

We trained multiple advanced models with a laptop-friendly search and early stopping. Primary objective: **maximize AUPRC** and choose an operating point with **FPR ≤ 1%**.

Artifacts:
- Consolidated CSV: `reports/variant_comparison.csv`
- Human summary: `reports/model_report.md`
- Final decision: `reports/final_model.json`

**Winner:** `lightgbm_smote` — best AUPRC (≈0.82) and Recall at FPR≤1% ≈ 0.863.  
**Fallback:** `lightgbm_class_weight` — simpler to maintain with similar recall at the constraint.


