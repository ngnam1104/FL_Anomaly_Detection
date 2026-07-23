"""Anomaly metrics and the joint learning/physical objective."""

from __future__ import annotations

import numpy as np


def anomaly_threshold(errors: np.ndarray, percentile: float = 99.0) -> float:
    return float(np.percentile(errors, percentile)) if len(errors) else 0.0


def _scores(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return float(f1), float(precision), float(recall)


def point_adjust_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    adjusted = y_pred.copy()
    start = None
    for index, label in enumerate(np.append(y_true, 0)):
        if label == 1 and start is None:
            start = index
        elif label == 0 and start is not None:
            if np.any(y_pred[start:index] == 1):
                adjusted[start:index] = 1
            start = None
    return adjusted


def anomaly_metrics(y_true: np.ndarray, errors: np.ndarray, threshold: float) -> dict:
    prediction = (errors > threshold).astype(np.int64)
    f1, precision, recall = _scores(y_true, prediction)
    pa_f1, pa_precision, pa_recall = _scores(
        y_true, point_adjust_predictions(y_true, prediction)
    )
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "pa_f1": pa_f1,
        "pa_precision": pa_precision,
        "pa_recall": pa_recall,
    }


def joint_objective(
    loss: float,
    energy: float,
    latency: float,
    lambda_e: float,
    lambda_tau: float,
) -> float:
    return float(loss + lambda_e * energy + lambda_tau * latency)
