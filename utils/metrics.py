from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    f1_score,
    precision_score,
    recall_score,
    brier_score_loss,
    confusion_matrix,
)

@dataclass
class ThresholdReport:
    threshold: float
    fpr: float
    tpr: float
    precision: float
    recall: float
    f1: float
    support_pos: int
    support_neg: int


def compute_core_metrics(y_true: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    return {
        "auprc": float(average_precision_score(y_true, p)),
        "roc_auc": float(roc_auc_score(y_true, p)),
        "brier": float(brier_score_loss(y_true, p)),
        "pos_rate": float(np.mean(y_true)),
    }


def recall_at_topk(y_true: np.ndarray, p: np.ndarray, frac: float) -> float:
    n = len(p)
    k = max(1, int(np.ceil(frac * n)))
    idx = np.argsort(p)[::-1][:k]
    return float(np.sum(y_true[idx]) / np.sum(y_true)) if np.sum(y_true) > 0 else 0.0

def threshold_for_fpr(y_true: np.ndarray, p: np.ndarray, fpr_target: float) -> ThresholdReport:
    fpr, tpr, thr = roc_curve(y_true, p)
    mask = fpr <= fpr_target
    i = np.where(mask)[0][-1] if np.any(mask) else int(np.argmin(fpr))
    th = float(thr[i])
    y_pred = (p >= th).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr_sel = fp / (fp + tn + 1e-12)
    tpr_sel = tp / (tp + fn + 1e-12)
    return ThresholdReport(
        threshold=th, fpr=float(fpr_sel), tpr=float(tpr_sel),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        support_pos=int(np.sum(y_true == 1)),
        support_neg=int(np.sum(y_true == 0)),
    )

def threshold_for_best_f1(y_true: np.ndarray, p: np.ndarray) -> ThresholdReport:
    prec, rec, thr = precision_recall_curve(y_true, p)
    f1s = 2 * (prec * rec) / (prec + rec + 1e-12)
    j = int(np.nanargmax(f1s))
    th = float(thr[j]) if j < len(thr) else 0.5
    y_pred = (p >= th).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr_sel = fp / (fp + tn + 1e-12)
    tpr_sel = tp / (tp + fn + 1e-12)
    return ThresholdReport(
        threshold=th, fpr=float(fpr_sel), tpr=float(tpr_sel),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        support_pos=int(np.sum(y_true == 1)),
        support_neg=int(np.sum(y_true == 0)),
    )

def confusion_at_threshold(y_true: np.ndarray, p: np.ndarray, threshold: float) -> Dict[str, int]:
    y_pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}