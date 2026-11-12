"""
Model Evaluation and Asymmetric Metrics Module
Module: src/evaluation.py
Lead: Ipsita Priyadarshini (250301230015) & Prasun Kumar Biswas (250301230010)
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support
)

logger = logging.getLogger("UAV_Pipeline.Evaluation")


class UAVModelEvaluator:
    """
    Evaluates fault classifiers prioritizing Recall for safety-critical states
    (Motor Failure & Propeller Imbalance) and applies custom Cost Matrices.
    """

    TARGET_CLASSES = [
        "Normal Flight",
        "Propeller Imbalance",
        "Battery Degradation",
        "Motor Failure"
    ]

    # Asymmetric cost penalties: Missed Motor Failure is catastrophic
    COST_MATRIX = np.array([
        [0,   5,   5,   5],     # True: Normal (cost for false alarm = 5)
        [50,  0,  20,  20],     # True: Propeller Imbalance (missed = 50)
        [50, 20,   0,  20],     # True: Battery Degradation (missed = 50)
        [500, 50, 50,   0]      # True: Motor Failure (missed = 500)
    ])

    def evaluate_predictions(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Computes comprehensive classification metrics and cost penalties."""
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=[0, 1, 2, 3], zero_division=0
        )

        # Calculate operational cost penalty
        total_cost = int(np.sum(cm * self.COST_MATRIX))

        metrics = {
            "Macro_F1": float(macro_f1),
            "Operational_Cost": total_cost,
            "Confusion_Matrix": cm.tolist(),
            "Per_Class": {}
        }

        for i, class_name in enumerate(self.TARGET_CLASSES):
            metrics["Per_Class"][class_name] = {
                "Precision": float(precision[i]),
                "Recall": float(recall[i]),
                "F1": float(f1[i]),
                "Support": int(support[i])
            }

        logger.info(f"Evaluation: Macro-F1 = {macro_f1:.4f}, Total Cost = {total_cost}")
        logger.info(f"Motor Failure Recall = {metrics['Per_Class']['Motor Failure']['Recall']:.4f}")
        return metrics

    def apply_asymmetric_thresholds(
        self,
        proba_matrix: np.ndarray,
        motor_fail_thresh: float = 0.25,
        prop_fail_thresh: float = 0.30,
        batt_degrade_thresh: float = 0.30
    ) -> np.ndarray:
        """
        Adjusts decision boundaries to favor catching critical faults
        even when probability is below standard 0.50.
        Priority order: Motor Failure > Propeller Imbalance > Battery Degradation > argmax.
        """
        # Start with default argmax predictions
        preds = np.argmax(proba_matrix, axis=1).astype(int)

        # Apply thresholds in ascending priority (later overwrites earlier)
        if proba_matrix.shape[1] > 2:
            batt_mask = proba_matrix[:, 2] >= batt_degrade_thresh
            preds[batt_mask] = 2

        if proba_matrix.shape[1] > 1:
            prop_mask = proba_matrix[:, 1] >= prop_fail_thresh
            preds[prop_mask] = 1

        # Motor Failure: highest priority — applied last to override all
        if proba_matrix.shape[1] > 3:
            motor_mask = proba_matrix[:, 3] >= motor_fail_thresh
            preds[motor_mask] = 3

        return preds
