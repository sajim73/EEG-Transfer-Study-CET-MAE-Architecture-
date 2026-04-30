import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_recall_fscore_support


def compute_classification_metrics(y_true, y_pred, average_macro: str = "macro", average_weighted: str = "weighted"):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average=average_macro, zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average=average_weighted, zero_division=0)),
        "n_samples": int(len(y_true)),
        "per_class": [
            {
                "class_index": int(i),
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(len(precision))
        ],
    }


def save_classification_outputs(output_dir, split_name, y_true, y_pred, probs=None, sample_ids=None, subjects=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    metrics = compute_classification_metrics(y_true, y_pred)
    metrics["split"] = split_name

    pd.DataFrame([{k: v for k, v in metrics.items() if k != "per_class"}]).to_csv(output_dir / f"{split_name}_metrics.csv", index=False)
    pd.DataFrame(metrics["per_class"]).to_csv(output_dir / f"{split_name}_per_class_metrics.csv", index=False)
    pd.DataFrame(confusion_matrix(y_true, y_pred)).to_csv(output_dir / f"{split_name}_confusion_matrix.csv", index=False)

    pred_df = pd.DataFrame({
        "sample_id": sample_ids if sample_ids is not None else np.arange(len(y_true)),
        "y_true": y_true,
        "y_pred": y_pred,
    })
    if subjects is not None:
        pred_df["subject"] = np.asarray(subjects)
    if probs is not None:
        probs = np.asarray(probs)
        if probs.ndim == 2:
            for i in range(probs.shape[1]):
                pred_df[f"prob_{i}"] = probs[:, i]
    pred_df.to_csv(output_dir / f"{split_name}_predictions.csv", index=False)

    with open(output_dir / f"{split_name}_classification_report.json", "w") as f:
        json.dump(classification_report(y_true, y_pred, output_dict=True, zero_division=0), f, indent=2)

    return metrics
