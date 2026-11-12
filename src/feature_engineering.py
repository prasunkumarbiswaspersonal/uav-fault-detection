"""
Task 3: Preprocessing and Feature Engineering (with Vibration FFT & Baseline Calibration)
Module: src/feature_engineering.py
Lead: Milan Mandal (250301230006) & Prasun Kumar Biswas (250301230010)
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("UAV_Pipeline.FeatureEngineering")


class ExplicitLabelEncoder:
    """
    Deterministic label encoder preserving canonical class order:
      0: Normal Flight
      1: Propeller Imbalance
      2: Battery Degradation
      3: Motor Failure
    Note: sklearn's default LabelEncoder sorts classes alphabetically,
    which scrambles class indices across models, evaluation, and dashboard.
    """
    def __init__(self, classes: List[str]):
        self.classes_ = np.array(classes)
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.idx_to_class = {i: c for i, c in enumerate(classes)}

    def fit(self, y=None):
        return self

    def transform(self, y) -> np.ndarray:
        if isinstance(y, (pd.Series, pd.Index)):
            return y.map(lambda val: self.class_to_idx.get(str(val), 0)).values.astype(int)
        return np.array([self.class_to_idx.get(str(val), 0) for val in y], dtype=int)

    def inverse_transform(self, y) -> np.ndarray:
        return np.array([self.idx_to_class.get(int(val), "Normal Flight") for val in y])


class TelemetryFeatureEngineer:
    """
    Computes time-series rolling statistics, Fast Fourier Transform (FFT)
    vibration harmonics, applies StandardScaler anchored to healthy calibration,
    and encodes multi-class fault targets.
    """

    TARGET_CLASSES = [
        "Normal Flight",
        "Propeller Imbalance",
        "Battery Degradation",
        "Motor Failure"
    ]

    def __init__(self, sampling_rate_hz: int = 50, window_sec: float = 1.0):
        self.sampling_rate = sampling_rate_hz
        self.window_size = max(5, int(sampling_rate_hz * window_sec))
        self.scaler = StandardScaler()
        self.label_encoder = ExplicitLabelEncoder(self.TARGET_CLASSES)
        self.feature_columns: List[str] = []
        self.is_scaler_fitted = False

    def _extract_fft_window_features(self, arr: np.ndarray) -> Tuple[float, float, float, float]:
        """
        Extracts spectral features from a 1D vibration buffer using FFT:
        1. Peak Harmonic Frequency (Hz)
        2. Peak Spectral Power
        3. Spectral Centroid (Hz)
        4. Harmonic Energy Concentration Ratio
        """
        if len(arr) < 4:
            return 0.0, 0.0, 0.0, 0.0

        n = len(arr)
        # Compute real Fast Fourier Transform
        fft_vals = np.abs(np.fft.rfft(arr))
        freqs = np.fft.rfftfreq(n, d=1.0 / self.sampling_rate)

        # Exclude DC offset (0 Hz)
        if len(fft_vals) > 1:
            ac_fft = fft_vals[1:]
            ac_freqs = freqs[1:]
            max_idx = np.argmax(ac_fft)
            peak_freq = float(ac_freqs[max_idx])
            peak_power = float(ac_fft[max_idx] ** 2)

            total_power = np.sum(fft_vals ** 2) + 1e-9
            # Spectral centroid (center of mass)
            centroid = float(np.sum(freqs * fft_vals) / (np.sum(fft_vals) + 1e-9))

            # Ratio of top harmonic power to broad spectrum
            harmonic_ratio = float(peak_power / total_power)
            return peak_freq, peak_power, centroid, harmonic_ratio

        return 0.0, 0.0, 0.0, 0.0

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computes rolling time-series and FFT spectral features per flight session."""
        feat_df = df.copy()
        group_col = "Flight_ID" if "Flight_ID" in feat_df.columns else None

        def _transform_group(group: pd.DataFrame) -> pd.DataFrame:
            # 1. 1-second rolling mean for Voltage (sag detection)
            group["Voltage_Roll_Mean"] = (
                group["Battery_Voltage"]
                .rolling(window=self.window_size, min_periods=1)
                .mean()
            )

            # 2. Rolling variance & RMS for high-frequency Vibration
            group["Vibration_Roll_Var"] = (
                group["Vibration"]
                .rolling(window=self.window_size, min_periods=1)
                .var()
                .fillna(0.0)
            )
            group["Vibration_Roll_RMS"] = np.sqrt(
                (group["Vibration"] ** 2)
                .rolling(window=self.window_size, min_periods=1)
                .mean()
            )

            # 3. Dynamic current and thermal rise
            group["Current_Roll_Mean"] = (
                group["Current"]
                .rolling(window=self.window_size, min_periods=1)
                .mean()
            )
            group["Temperature_Roll_Mean"] = (
                group["Temperature"]
                .rolling(window=self.window_size, min_periods=1)
                .mean()
            )

            # 4. Instantaneous electrical power: P = V * I (Watts)
            group["Electrical_Power"] = group["Battery_Voltage"] * group["Current"]

            # 5. Fast Fourier Transform (FFT) over sliding vibration window
            vib_values = group["Vibration"].values
            n_samples = len(vib_values)

            peak_freqs = np.zeros(n_samples)
            peak_powers = np.zeros(n_samples)
            centroids = np.zeros(n_samples)
            harmonic_ratios = np.zeros(n_samples)

            # Warm-up: variable-length windows for first (window_size - 1) samples
            warmup_end = min(self.window_size - 1, n_samples)
            for i in range(warmup_end):
                window_slice = vib_values[:i + 1]
                pf, pp, sc, hr = self._extract_fft_window_features(window_slice)
                peak_freqs[i] = pf
                peak_powers[i] = pp
                centroids[i] = sc
                harmonic_ratios[i] = hr

            # Vectorized batch FFT for all full-sized windows
            if n_samples >= self.window_size:
                windows = np.lib.stride_tricks.sliding_window_view(
                    vib_values, self.window_size
                )
                fft_vals = np.abs(np.fft.rfft(windows, axis=1))
                freqs = np.fft.rfftfreq(self.window_size, d=1.0 / self.sampling_rate)

                # Exclude DC component (index 0)
                ac_fft = fft_vals[:, 1:]
                ac_freqs = freqs[1:]

                max_indices = np.argmax(ac_fft, axis=1)
                peak_freqs[warmup_end:] = ac_freqs[max_indices]
                peak_powers[warmup_end:] = ac_fft[np.arange(len(ac_fft)), max_indices] ** 2

                total_power = np.sum(fft_vals ** 2, axis=1) + 1e-9
                fft_sum = np.sum(fft_vals, axis=1) + 1e-9
                centroids[warmup_end:] = np.sum(freqs * fft_vals, axis=1) / fft_sum
                harmonic_ratios[warmup_end:] = peak_powers[warmup_end:] / total_power

            group["Vibration_FFT_Peak_Freq"] = peak_freqs
            group["Vibration_FFT_Peak_Power"] = peak_powers
            group["Vibration_Spectral_Centroid"] = centroids
            group["Vibration_Harmonic_Energy_Ratio"] = harmonic_ratios

            return group

        if group_col and group_col in feat_df.columns:
            transformed_groups = []
            for _, group in feat_df.groupby(group_col, sort=False):
                transformed_groups.append(_transform_group(group.copy()))
            feat_df = pd.concat(transformed_groups, axis=0, ignore_index=True)
        else:
            feat_df = _transform_group(feat_df)

        return feat_df

    def encode_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """Encodes Fault_Status to integer multi-class target [0, 1, 2, 3]."""
        encoded_df = df.copy()

        def _map_to_canonical(val: str) -> str:
            clean_str = str(val).strip().lower()
            if "prop" in clean_str or "imbalance" in clean_str or "broken" in clean_str:
                return "Propeller Imbalance"
            if "batt" in clean_str or "degradation" in clean_str:
                return "Battery Degradation"
            if "motor" in clean_str or "engine" in clean_str or "esc" in clean_str:
                return "Motor Failure"
            return "Normal Flight"

        if "Fault_Status" in encoded_df.columns:
            encoded_df["Fault_Status"] = encoded_df["Fault_Status"].apply(_map_to_canonical)
            y = self.label_encoder.transform(encoded_df["Fault_Status"])
        else:
            y = np.zeros(len(encoded_df), dtype=int)

        encoded_df["Target_Class"] = y
        return encoded_df, y

    def fit_calibration_baseline(self, df_healthy: pd.DataFrame):
        """
        Fits StandardScaler strictly on a verified 'Smooth Flight' baseline
        to establish nominal reference values (mean=0, std=1 for healthy state).
        """
        featured_df = self.compute_features(df_healthy)
        meta_cols = {"Fault_Status", "Target_Class", "Flight_ID", "Source_File", "Timestamp"}
        numeric_cols = [c for c in featured_df.columns if c not in meta_cols and np.issubdtype(featured_df[c].dtype, np.number)]
        self.feature_columns = numeric_cols

        X_healthy = featured_df[numeric_cols].values
        self.scaler.fit(X_healthy)
        self.is_scaler_fitted = True
        logger.info(f"StandardScaler anchored to healthy calibration baseline over {len(numeric_cols)} features.")

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Extracts features, encodes labels, and standardizes against the baseline."""
        featured_df = self.compute_features(df)
        featured_df, y = self.encode_target(featured_df)

        meta_cols = {"Fault_Status", "Target_Class", "Flight_ID", "Source_File", "Timestamp"}
        if not self.feature_columns:
            self.feature_columns = [c for c in featured_df.columns if c not in meta_cols and np.issubdtype(featured_df[c].dtype, np.number)]

        X_raw = featured_df[self.feature_columns].values

        if not self.is_scaler_fitted:
            X_scaled = self.scaler.fit_transform(X_raw)
            self.is_scaler_fitted = True
        else:
            X_scaled = self.scaler.transform(X_raw)

        return X_scaled, y, self.feature_columns
