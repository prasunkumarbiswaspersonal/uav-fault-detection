"""
Unit Tests for UAV Fault Detection Pipeline
Run: pytest tests/ -v
"""
import numpy as np
import pandas as pd
import pytest

from pathlib import Path

from src.ingestion import UAVDataIngestionPipeline
from src.cleaning import TelemetryCleaner
from src.feature_engineering import TelemetryFeatureEngineer
from src.model_training import UAVModelTrainer
from src.evaluation import UAVModelEvaluator


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_flight(flight_id="test_01", fault_type="Normal Flight", n=100, seed=42):
    """Creates a single synthetic flight DataFrame with canonical raw column names."""
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 4, n)
    return pd.DataFrame({
        "timestamp": t,
        "voltage": 22.0 + rng.normal(0, 0.1, n),
        "current": 15.0 + rng.normal(0, 0.5, n),
        "temperature": 40.0 + rng.normal(0, 0.2, n),
        "vibration": 9.81 + rng.normal(0, 0.3, n),
        "fault_status": fault_type,
        "Flight_ID": flight_id,
        "Source_File": f"{flight_id}.csv",
    })


def _make_multi_flight_dataset():
    """Creates a 4-flight dataset covering all fault classes."""
    flights = [
        ("f1_healthy", "Normal Flight", 100, 1),
        ("f2_healthy", "Normal Flight", 100, 2),
        ("f3_prop", "Propeller Imbalance", 80, 3),
        ("f4_motor", "Motor Failure", 80, 4),
    ]
    frames = [_make_flight(fid, fault, n, seed) for fid, fault, n, seed in flights]
    return pd.concat(frames, ignore_index=True)


# ── Cleaning Tests ────────────────────────────────────────────────────────────

class TestCleaning:
    def test_column_standardization(self):
        """Fuzzy column aliases are mapped to canonical names."""
        cleaner = TelemetryCleaner()
        df = pd.DataFrame({
            "vbat": [22.0, 22.1],
            "curr": [15.0, 15.1],
            "temp": [40.0, 40.1],
            "vib": [9.8, 9.9],
            "Flight_ID": ["f1", "f1"],
        })
        result = cleaner.standardize_columns(df)
        for expected in ["Battery_Voltage", "Current", "Temperature", "Vibration"]:
            assert expected in result.columns, f"Missing canonical column: {expected}"

    def test_missing_column_raises(self):
        """Missing essential channels raise a clear KeyError."""
        cleaner = TelemetryCleaner()
        df = pd.DataFrame({"voltage": [22.0], "current": [15.0]})  # missing temp & vib
        with pytest.raises(KeyError, match="Missing essential telemetry"):
            cleaner.standardize_columns(df)

    def test_outlier_clipping_per_flight(self):
        """Per-flight clipping preserves inter-flight fault signatures."""
        cleaner = TelemetryCleaner(outlier_method="iqr", outlier_threshold=1.5)
        df1 = _make_flight("f1", "Normal Flight", 50, seed=10)
        df2 = _make_flight("f2", "Motor Failure", 50, seed=20)
        df2["current"] = 28.0 + np.random.RandomState(20).normal(0, 0.5, 50)

        combined = pd.concat([df1, df2], ignore_index=True)
        cleaned = cleaner.clean(combined)

        motor_current = cleaned[cleaned["Flight_ID"] == "f2"]["Current"]
        assert motor_current.mean() > 25.0, (
            "Per-flight clipping should preserve Motor Failure's high current"
        )

    def test_missing_data_interpolation(self):
        """NaN gaps are filled by linear interpolation."""
        cleaner = TelemetryCleaner()
        df = _make_flight()
        df.loc[10:15, "voltage"] = np.nan
        cleaned = cleaner.clean(df)
        assert not cleaned["Battery_Voltage"].isna().any()

    def test_clean_returns_core_columns(self):
        """Cleaned DataFrame always has the 4 core sensor channels."""
        cleaner = TelemetryCleaner()
        cleaned = cleaner.clean(_make_flight())
        for col in ["Battery_Voltage", "Current", "Temperature", "Vibration"]:
            assert col in cleaned.columns


# ── Feature Engineering Tests ─────────────────────────────────────────────────

class TestFeatureEngineering:
    def test_output_shape(self):
        """Transform output has (n_samples, n_features) and matching label vector."""
        fe = TelemetryFeatureEngineer(sampling_rate_hz=50, window_sec=0.5)
        cleaned = TelemetryCleaner().clean(_make_flight())
        X, y, feature_names = fe.transform(cleaned)
        assert X.shape[0] == len(cleaned)
        assert X.shape[1] == len(feature_names)
        assert len(y) == len(cleaned)

    def test_fft_features_present(self):
        """FFT-derived spectral features are extracted."""
        fe = TelemetryFeatureEngineer(sampling_rate_hz=50, window_sec=0.5)
        cleaned = TelemetryCleaner().clean(_make_flight())
        _, _, feature_names = fe.transform(cleaned)
        fft_features = [f for f in feature_names if "FFT" in f or "Spectral" in f or "Harmonic" in f]
        assert len(fft_features) >= 3, f"Expected ≥3 FFT features, got: {fft_features}"

    def test_target_encoding_four_classes(self):
        """All 4 canonical fault classes are registered."""
        fe = TelemetryFeatureEngineer()
        assert len(fe.TARGET_CLASSES) == 4
        assert "Normal Flight" in fe.TARGET_CLASSES
        assert "Motor Failure" in fe.TARGET_CLASSES

    def test_no_nan_in_features(self):
        """Feature matrix contains no NaN values."""
        fe = TelemetryFeatureEngineer(sampling_rate_hz=50, window_sec=0.5)
        cleaned = TelemetryCleaner().clean(_make_flight())
        X, _, _ = fe.transform(cleaned)
        assert not np.isnan(X).any(), "Feature matrix should be NaN-free"


# ── Model Training Tests ─────────────────────────────────────────────────────

class TestModelTraining:
    def test_cv_returns_metrics(self):
        """Cross-validation runs and returns accuracy metrics for both models."""
        fe = TelemetryFeatureEngineer(sampling_rate_hz=50, window_sec=0.5)
        cleaner = TelemetryCleaner()

        df = _make_multi_flight_dataset()
        cleaned = cleaner.clean(df)
        X, y, names = fe.transform(cleaned)
        groups = cleaned["Flight_ID"].values

        trainer = UAVModelTrainer(n_splits=2)
        results = trainer.evaluate_cv(X, y, groups=groups)

        assert "LightGBM_Mean_Acc" in results
        assert "Random_Forest_Mean_Acc" in results
        assert 0.0 <= results["LightGBM_Mean_Acc"] <= 1.0

    def test_train_final_model_predict_proba(self):
        """Final model supports predict_proba with correct output shape."""
        rng = np.random.RandomState(42)
        X = rng.randn(200, 10)
        y = np.array([0] * 50 + [1] * 50 + [2] * 50 + [3] * 50)

        trainer = UAVModelTrainer()
        model = trainer.train_final_model(X, y, model_type="lightgbm")
        assert hasattr(model, "predict_proba")
        probs = model.predict_proba(X[:5])
        assert probs.shape == (5, 4)

    def test_feature_importance_logging(self):
        """Feature importance is available after training LightGBM."""
        rng = np.random.RandomState(42)
        X = rng.randn(200, 5)
        y = np.array([0] * 50 + [1] * 50 + [2] * 50 + [3] * 50)

        trainer = UAVModelTrainer()
        model = trainer.train_final_model(
            X, y, model_type="lightgbm",
            feature_names=["feat_a", "feat_b", "feat_c", "feat_d", "feat_e"]
        )
        assert hasattr(model, "feature_importances_")
        assert len(model.feature_importances_) == 5


# ── Evaluation Tests ──────────────────────────────────────────────────────────

class TestEvaluation:
    def test_motor_failure_low_threshold(self):
        """Motor Failure is triggered even at low probability (>= 0.25)."""
        evaluator = UAVModelEvaluator()
        proba = np.array([[0.4, 0.1, 0.2, 0.3]])
        preds = evaluator.apply_asymmetric_thresholds(proba, motor_fail_thresh=0.25)
        assert preds[0] == 3, "Motor Failure should trigger at P=0.30 >= thresh=0.25"

    def test_motor_overrides_propeller(self):
        """Motor Failure (highest priority) overrides Propeller Imbalance."""
        evaluator = UAVModelEvaluator()
        proba = np.array([[0.1, 0.35, 0.2, 0.35]])
        preds = evaluator.apply_asymmetric_thresholds(proba, motor_fail_thresh=0.25)
        assert preds[0] == 3, "Motor Failure should override Propeller Imbalance"

    def test_battery_degradation_threshold(self):
        """Battery Degradation is caught by the new dedicated threshold."""
        evaluator = UAVModelEvaluator()
        proba = np.array([[0.3, 0.1, 0.35, 0.15]])
        preds = evaluator.apply_asymmetric_thresholds(
            proba, motor_fail_thresh=0.25, batt_degrade_thresh=0.30
        )
        assert preds[0] == 2, "Battery Degradation should trigger at P=0.35 >= thresh=0.30"

    def test_cost_matrix_shape(self):
        """Cost matrix is 4×4."""
        evaluator = UAVModelEvaluator()
        assert evaluator.COST_MATRIX.shape == (4, 4)

    def test_evaluation_metrics_keys(self):
        """evaluate_predictions returns all expected metric keys."""
        evaluator = UAVModelEvaluator()
        y_true = np.array([0, 1, 2, 3, 0, 1])
        y_pred = np.array([0, 1, 2, 3, 0, 0])
        metrics = evaluator.evaluate_predictions(y_true, y_pred)
        assert "Macro_F1" in metrics
        assert "Operational_Cost" in metrics
        assert "Per_Class" in metrics
        assert len(metrics["Per_Class"]) == 4

    def test_vectorized_matches_large_input(self):
        """Vectorized thresholds handle batched input correctly."""
        evaluator = UAVModelEvaluator()
        rng = np.random.RandomState(42)
        proba = rng.dirichlet([1, 1, 1, 1], size=1000)
        preds = evaluator.apply_asymmetric_thresholds(proba)
        assert preds.shape == (1000,)
        assert set(preds).issubset({0, 1, 2, 3})


# ── Ingestion & Real Dataset Tests ──────────────────────────────────────────

class TestIngestion:
    def test_sync_ardupilot_mat(self):
        """ArduPilot .mat file synchronizes into canonical schema at 50 Hz."""
        mat_path = Path("FAULT_M1_10.mat")
        if not mat_path.exists():
            pytest.skip("FAULT_M1_10.mat not present in root directory")
        pipeline = UAVDataIngestionPipeline(Path("."))
        df = pipeline._sync_ardupilot_mat(mat_path, fs=50)
        assert len(df) > 1000
        for col in ["Battery_Voltage", "Current", "Temperature", "Vibration", "Fault_Status"]:
            assert col in df.columns
        assert df["Fault_Status"].iloc[0] == "Motor Failure"

    def test_sync_droneprop_mat(self):
        """DronePropB ground test .mat file synchronizes into canonical schema at 50 Hz."""
        mat_path = Path("DronePropB Ground Testing Dataset/F1_SV1_SP1_C1.mat")
        if not mat_path.exists():
            pytest.skip("DronePropB file not present")
        pipeline = UAVDataIngestionPipeline(Path("."))
        df = pipeline._sync_droneprop_mat(mat_path, fs=50)
        assert len(df) > 1000
        for col in ["Battery_Voltage", "Current", "Temperature", "Vibration", "Fault_Status"]:
            assert col in df.columns
        assert df["Fault_Status"].iloc[0] == "Propeller Imbalance"

    def test_scan_files_discovers_real_flights(self):
        """scan_files discovers all real flight CSVs and excludes legacy synthetic files."""
        pipeline = UAVDataIngestionPipeline(Path("datasets"))
        files = pipeline.scan_files()
        assert len(files) >= 10
        assert all(not f.name.lower().startswith("flight_0") for f in files)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
