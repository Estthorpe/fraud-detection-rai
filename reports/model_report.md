# Model Comparison Report

_Generated: 2025-09-02 19:44:31_

**Primary objective**: Maximize AUPRC, and select operating threshold with FPR ≤ target.


## Summary Table

| name                  | algo     | imbalance    | calibration   |    auprc |   roc_auc |   recall@topk |   FPR@target |   recall@FPR |   thr@FPR |
|:----------------------|:---------|:-------------|:--------------|---------:|----------:|--------------:|-------------:|-------------:|----------:|
| lightgbm_smote        | lightgbm | smote        | none          | 0.821734 |  0.979124 |        0.8632 |       0.0095 |       0.8632 |  0.009003 |
| lightgbm_smoteenn     | lightgbm | smoteenn     | none          | 0.727039 |  0.974659 |        0.8526 |       0.0031 |       0.8526 |  0.144071 |
| lightgbm_class_weight | lightgbm | class_weight | none          | 0.709514 |  0.97851  |        0.8632 |       0.0077 |       0.8632 |  0.230376 |
| baseline              |          |              |               | 0.675274 |  0.964805 |        0.8632 |       0.0092 |       0.8632 |  0.745395 |

## Notes

- AUPRC (area under PR) is the primary metric on imbalanced data.
- `recall@FPR` is measured at the configured FPR target (e.g., 1%).
- `recall@topk` uses the configured top-k fraction (e.g., top 1% most suspicious transactions).
- See `variant_comparison.csv` for the full dataset and confusion matrices.
