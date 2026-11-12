"""
Model Training and Validation Module
Module: src/model_training.py
Lead: Prasun Kumar Biswas (250301230010) & Ipsita Priyadarshini (250301230015)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
import lightgbm as lgb

logger = logging.getLogger("UAV_Pipeline.ModelTraining")


class UAVModelTrainer:
    """
    Implements Grouped cross-validation (by Flight_ID) and trains
    class-weighted Random Forest and LightGBM models.
    """

    def __init__(self, n_splits: int = 5, random_state: int = 42):
        self.n_splits = n_splits
        self.random_state = random_state
        self.best_model = None
        self.best_model_name = ""

    def evaluate_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray
    ) -> Dict[str, float]:
        """Runs StratifiedGroupKFold to prevent flight-level data leakage."""
        sgkf = StratifiedGroupKFold(n_splits=self.n_splits)

        rf_scores = []
        lgb_scores = []

        logger.info(f"Running {self.n_splits}-fold StratifiedGroupKFold across {len(np.unique(groups))} distinct flights...")

        for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=groups)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # 1. Random Forest with balanced class weighting
            rf = RandomForestClassifier(
                n_estimators=100,
                class_weight="balanced_subsample",
                random_state=self.random_state,
                n_jobs=-1
            )
            rf.fit(X_train, y_train)
            rf_scores.append(rf.score(X_val, y_val))

            # 2. LightGBM with balanced class weighting and regularization
            lgbm = lgb.LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                class_weight="balanced",
                max_depth=8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                verbose=-1
            )
            # Filter eval set to only classes present in y_train to prevent unseen label ValueError
            val_mask = np.isin(y_val, np.unique(y_train))
            if np.all(val_mask):
                val_X, val_y = X_val, y_val
            elif np.any(val_mask):
                val_X, val_y = X_val[val_mask], y_val[val_mask]
            else:
                val_X, val_y = None, None

            if val_X is not None:
                lgbm.fit(
                    X_train, y_train,
                    eval_X=val_X,
                    eval_y=val_y,
                    callbacks=[lgb.early_stopping(20, verbose=False)]
                )
            else:
                lgbm.fit(X_train, y_train)

            lgb_scores.append(lgbm.score(X_val, y_val))

        results = {
            "Random_Forest_Mean_Acc": float(np.mean(rf_scores)),
            "Random_Forest_Std_Acc": float(np.std(rf_scores)),
            "LightGBM_Mean_Acc": float(np.mean(lgb_scores)),
            "LightGBM_Std_Acc": float(np.std(lgb_scores))
        }

        logger.info(f"Grouped CV Results: RF={results['Random_Forest_Mean_Acc']:.4f}, LGBM={results['LightGBM_Mean_Acc']:.4f}")
        return results

    def train_final_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_type: str = "lightgbm",
        feature_names: list = None
    ):
        """Trains final model on the full dataset with balanced class weights."""
        if model_type.lower() == "lightgbm":
            model = lgb.LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                class_weight="balanced",
                max_depth=8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                verbose=-1
            )
            self.best_model_name = "LightGBM"
        else:
            model = RandomForestClassifier(
                n_estimators=150,
                class_weight="balanced_subsample",
                random_state=self.random_state,
                n_jobs=-1
            )
            self.best_model_name = "RandomForest"

        model.fit(X, y)
        self.best_model = model
        logger.info(f"Trained final production model: {self.best_model_name}")

        # Log feature importances for model interpretability
        self.feature_importances_ = {}
        if hasattr(model, 'feature_importances_') and feature_names:
            importances = model.feature_importances_
            sorted_idx = np.argsort(importances)[::-1]
            logger.info("Top-10 feature importances:")
            for rank, idx in enumerate(sorted_idx[:min(10, len(feature_names))]):
                logger.info(f"  {rank+1}. {feature_names[idx]}: {importances[idx]}")
            self.feature_importances_ = {
                feature_names[i]: float(importances[i]) for i in sorted_idx
            }

        return model

    def save_pipeline(self, output_dir: str = "models", filename: str = "uav_fault_model.joblib"):
        """Saves trained model using joblib and persists feature importance mapping."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        target = out_path / filename

        joblib.dump(self.best_model, target)
        logger.info(f"Saved model artifact to: {target}")

        if hasattr(self, "feature_importances_") and self.feature_importances_:
            fi_target = out_path / "feature_importance.json"
            with open(fi_target, "w") as f:
                json.dump(self.feature_importances_, f, indent=2)
            logger.info(f"Saved feature importance artifact to: {fi_target}")

        return target
