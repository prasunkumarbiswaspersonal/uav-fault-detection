"""
UAV Fault Detection System: End-to-End Orchestrator
Execution Entry Point: main.py
Coordinates Milestones 1 through 3.
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
import joblib
import numpy as np
import pandas as pd

from src.ingestion import UAVDataIngestionPipeline
from src.cleaning import TelemetryCleaner
from src.feature_engineering import TelemetryFeatureEngineer
from src.model_training import UAVModelTrainer
from src.evaluation import UAVModelEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("UAV_System")


def generate_synthetic_benchmark(target_dir: Path):
    """
    Generates synthetic flight telemetry to validate the pipeline
    if datasets/ directory does not yet contain pre-processed test runs.
    """
    np.random.seed(42)
    target_dir.mkdir(parents=True, exist_ok=True)
    flights = [
        ("Flight_01_Normal_Cruise", "Normal Flight", 400),
        ("Flight_02_Normal_Hover", "Normal Flight", 400),
        ("Flight_03_Propeller_Imbalance_Fault", "Propeller Imbalance", 350),
        ("Flight_04_Battery_Degradation_Fault", "Battery Degradation", 350),
        ("Flight_05_Motor_Failure_Fault", "Motor Failure", 350),
    ]

    for flight_name, fault_type, n in flights:
        t = np.linspace(0, 8, n)
        if fault_type == "Normal Flight":
            v = 22.2 - (0.02 * t) + np.random.normal(0, 0.02, n)
            i = 15.0 + np.random.normal(0, 0.4, n)
            temp = 40.0 + (0.2 * t) + np.random.normal(0, 0.1, n)
            vib = 9.81 + np.sin(20 * t) * 0.8 + np.random.normal(0, 0.15, n)
        elif fault_type == "Propeller Imbalance":
            v = 22.1 - (0.03 * t) + np.random.normal(0, 0.03, n)
            i = 18.5 + np.random.normal(0, 0.6, n)
            temp = 44.0 + (0.3 * t) + np.random.normal(0, 0.15, n)
            # Distinct high-frequency harmonic vibration resonance
            vib = 9.81 + np.sin(120 * t) * 4.5 + np.random.normal(0, 0.4, n)
        elif fault_type == "Battery Degradation":
            # Rapid non-linear voltage collapse
            v = 22.0 - (0.25 * t) - (0.05 * t**2) + np.random.normal(0, 0.05, n)
            i = 14.0 + np.random.normal(0, 0.5, n)
            temp = 48.0 + (0.6 * t) + np.random.normal(0, 0.2, n)
            vib = 9.81 + np.sin(25 * t) * 0.9 + np.random.normal(0, 0.2, n)
        else:  # Motor Failure
            # Throttle compensation surge then drop, thermal spike
            v = 21.8 - (0.04 * t) + np.random.normal(0, 0.03, n)
            i = 28.0 + np.random.normal(0, 1.2, n)
            temp = 62.0 + (1.2 * t) + np.random.normal(0, 0.3, n)
            vib = 9.81 + np.sin(60 * t) * 3.0 + np.random.normal(0, 0.5, n)

        df = pd.DataFrame({
            "timestamp": t,
            "voltage": v,
            "current": i,
            "temperature": temp,
            "vibration": vib,
            "fault_status": fault_type
        })
        # Save as CSV
        df.to_csv(target_dir / f"{flight_name}.csv", index=False)
    logger.info(f"Generated synthetic benchmark flights under: {target_dir}")


def main():
    logger.info("================================================================================")
    logger.info("UAV FAULT DETECTION SYSTEM - PRODUCTION PIPELINE EXECUTION")
    logger.info("================================================================================")

    data_dir = Path("datasets")
    if not data_dir.exists() or not any(data_dir.rglob("*.*")):
        logger.info("Datasets directory empty or missing. Creating sample telemetry...")
        generate_synthetic_benchmark(data_dir)

    # 1. Ingestion — scan datasets/ directory for flight logs
    logger.info("[MILESTONE 1 - TASK 1] Ingesting flight logs from datasets/ directory...")
    all_frames = []

    data_sources = [
        data_dir,
    ]

    for src_dir in data_sources:
        if src_dir.exists():
            try:
                pipeline = UAVDataIngestionPipeline(root_dir=src_dir)
                df = pipeline.ingest_all()
                all_frames.append(df)
                logger.info(f"  Loaded {len(df):,} records from '{src_dir}'")
            except Exception as e:
                logger.warning(f"  Skipping data source '{src_dir}': {e}")

    if not all_frames:
        raise RuntimeError("No valid telemetry data could be loaded from any source.")

    master_df = pd.concat(all_frames, axis=0, ignore_index=True)
    logger.info(f"Master DataFrame: {len(master_df):,} total records from {len(all_frames)} source(s)")

    # 2. Cleaning
    logger.info("[MILESTONE 1 - TASK 2] Standardizing schema and filtering sensor noise...")
    cleaner = TelemetryCleaner(outlier_method="iqr", outlier_threshold=3.0)
    cleaned_df = cleaner.clean(master_df)

    # 3. Feature Engineering & FFT Extraction
    logger.info("[MILESTONE 1 - TASK 3] Computing rolling statistics and Vibration FFT harmonics...")
    fe = TelemetryFeatureEngineer(sampling_rate_hz=50, window_sec=1.0)

    # Anchor StandardScaler to a verified Normal Flight baseline
    healthy_baseline = cleaned_df[cleaned_df["Fault_Status"] == "Normal Flight"]
    if not healthy_baseline.empty:
        fe.fit_calibration_baseline(healthy_baseline)

    X, y, feature_names = fe.transform(cleaned_df)
    groups = cleaned_df["Flight_ID"].values

    logger.info(f"Feature Matrix Shape: {X.shape}")
    logger.info(f"Features: {feature_names}")
    logger.info(f"Target Distribution: {dict(zip(fe.TARGET_CLASSES, np.bincount(y, minlength=4)))}")

    # 4. Held-out flight-level train/test split
    logger.info("[MILESTONE 2] Splitting data by flight groups into train/test sets...")
    unique_flights = np.unique(groups)
    n_flights = len(unique_flights)

    if n_flights >= 4:
        from sklearn.model_selection import StratifiedGroupKFold
        sgkf = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
        chosen_split = None
        for tr_idx, te_idx in sgkf.split(X, y, groups=groups):
            if len(np.unique(y[tr_idx])) == 4 and len(np.unique(y[te_idx])) == 4:
                chosen_split = (tr_idx, te_idx)
                break
        if chosen_split is None:
            chosen_split = next(sgkf.split(X, y, groups=groups))
        train_idx, test_idx = chosen_split
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train = groups[train_idx]
        logger.info(
            f"Train: {len(train_idx):,} samples ({len(np.unique(groups_train))} flights) | "
            f"Test: {len(test_idx):,} samples ({len(np.unique(groups[test_idx]))} flights)"
        )
    else:
        logger.warning("Too few flights for a held-out split. Using all data for train and test.")
        X_train, X_test = X, X
        y_train, y_test = y, y
        groups_train = groups

    # 5. Model Training & Cross-Validation (on training flights only)
    logger.info("[MILESTONE 2] Running StratifiedGroupKFold validation across training flights...")
    trainer = UAVModelTrainer(n_splits=min(3, len(np.unique(groups_train))))
    cv_results = trainer.evaluate_cv(X_train, y_train, groups=groups_train)
    logger.info(f"Grouped CV Performance: {cv_results}")

    logger.info("[MILESTONE 2] Training LightGBM on training flights...")
    trained_model = trainer.train_final_model(X_train, y_train, model_type="lightgbm", feature_names=feature_names)

    # 6. Asymmetric Evaluation on HELD-OUT test flights
    logger.info("[MILESTONE 3] Evaluating on HELD-OUT test flights (unbiased estimate)...")
    evaluator = UAVModelEvaluator()
    probas = trained_model.predict_proba(X_test)
    asym_preds = evaluator.apply_asymmetric_thresholds(probas, motor_fail_thresh=0.25)
    metrics = evaluator.evaluate_predictions(y_test, asym_preds)

    logger.info("--- EVALUATION METRICS REPORT (HELD-OUT TEST SET) ---")
    logger.info(f"Macro-F1 Score: {metrics['Macro_F1']:.4f}")
    logger.info(f"Total Operational Cost Penalty: {metrics['Operational_Cost']}")
    for cls_name, vals in metrics["Per_Class"].items():
        logger.info(f"  {cls_name:<20} | Recall: {vals['Recall']:.4f} | Prec: {vals['Precision']:.4f} | F1: {vals['F1']:.4f}")

    # Save evaluation metrics to disk for experiment tracking
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "timestamp": datetime.now().isoformat(),
        "cv_results": cv_results,
        "test_metrics": metrics,
        "n_train_samples": int(len(X_train)),
        "n_test_samples": int(len(X_test)),
        "n_flights": int(n_flights),
        "n_features": len(feature_names),
        "feature_names": list(feature_names),
    }
    with open(models_dir / "evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Saved evaluation report to: {models_dir / 'evaluation_report.json'}")

    # 7. Retrain on ALL data for production deployment
    logger.info("[PRODUCTION] Retraining final model on ALL data for deployment...")
    final_model = trainer.train_final_model(X, y, model_type="lightgbm", feature_names=feature_names)
    trainer.save_pipeline(output_dir=str(models_dir), filename="uav_fault_model.joblib")
    joblib.dump(fe, models_dir / "feature_engineer.joblib")
    joblib.dump(cleaner, models_dir / "cleaner.joblib")
    logger.info(f"Exported model and preprocessing artifacts to: {models_dir}")
    logger.info("================================================================================")
    logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("================================================================================")


if __name__ == "__main__":
    main()
