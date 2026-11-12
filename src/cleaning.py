"""
Task 2: Data Cleaning and Standardization
Module: src/cleaning.py
Lead: Omm Shree Hota (250301230002) & Manideepa Mohapatra (250301230001)
"""

import re
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("UAV_Pipeline.Cleaning")


class TelemetryCleaner:
    """
    Standardizes raw telemetry to canonical channels:
    - Battery_Voltage, Current, Temperature, Vibration
    Applies time-series interpolation and statistical outlier clipping.
    """

    CANONICAL_ALIASES: Dict[str, List[str]] = {
        "Battery_Voltage": [
            "battery_voltage", "voltage", "volt", "vbat", "bat_v", "batt_v",
            "u_bat", "volt_v", "battery_v", "v_cell", "voltage_v"
        ],
        "Current": [
            "current", "curr", "current_a", "i_bat", "batt_curr", "battery_current",
            "curr_a", "amps", "amperage", "current_amps"
        ],
        "Temperature": [
            "temperature", "temp", "esc_temp", "motor_temp", "t_degc", "temp_c",
            "imu_temp", "board_temp", "esc_temperature", "batt_temp"
        ],
        "Vibration": [
            "vibration", "vib", "vib_z", "acc_z", "accel_z", "acc_mag", "vibration_z",
            "vibe_z", "imu_acc_z", "vib_level", "acc_norm", "raw_vib"
        ],
        "Timestamp": [
            "timestamp", "time", "t_sec", "time_s", "t_ms", "time_usec", "time_us", "seconds"
        ],
        "Fault_Status": [
            "fault_status", "fault", "state", "label", "condition", "target", "class"
        ]
    }

    def __init__(self, outlier_method: str = "iqr", outlier_threshold: float = 3.0):
        self.outlier_method = outlier_method.lower()
        self.outlier_threshold = outlier_threshold
        self.core_features = ["Battery_Voltage", "Current", "Temperature", "Vibration"]

    def _resolve_column_name(self, col: str) -> Optional[str]:
        """Resolves raw column names to canonical schema using regex."""
        normalized = col.strip().lower().replace(" ", "_").replace(".", "_")
        for target, aliases in self.CANONICAL_ALIASES.items():
            if normalized == target.lower():
                return target
            for alias in aliases:
                if re.fullmatch(alias, normalized) or f"_{alias}_" in f"_{normalized}_":
                    return target
        return None

    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Renames recognized sensor columns and coalesces multiple aliases into core channels."""
        standardized_df = pd.DataFrame(index=df.index)

        # Group input columns by canonical target
        canonical_groups: Dict[str, List[str]] = {}
        unmapped_cols = []
        for col in df.columns:
            canonical = self._resolve_column_name(col)
            if canonical:
                canonical_groups.setdefault(canonical, []).append(col)
            else:
                unmapped_cols.append(col)

        # Coalesce columns resolving to the same canonical name (combine_first)
        for canonical, cols in canonical_groups.items():
            coalesced = df[cols[0]].copy()
            for other_col in cols[1:]:
                coalesced = coalesced.combine_first(df[other_col])
            standardized_df[canonical] = coalesced

        for col in unmapped_cols:
            standardized_df[col] = df[col]

        # Check required telemetry parameters
        missing = [ch for ch in self.core_features if ch not in standardized_df.columns]
        if missing:
            raise KeyError(
                f"Missing essential telemetry channel(s): {missing}. "
                f"Available channels in raw dataset: {list(df.columns)}"
            )

        metadata_cols = [c for c in ["Timestamp", "Fault_Status", "Flight_ID", "Source_File"] if c in standardized_df.columns]
        keep_cols = list(dict.fromkeys(self.core_features + metadata_cols))
        return standardized_df[keep_cols]

    def handle_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Interpolates missing sensor values grouped by Flight_ID."""
        cleaned_df = df.copy()

        sort_keys = [c for c in ["Flight_ID", "Timestamp"] if c in cleaned_df.columns]
        if sort_keys:
            cleaned_df = cleaned_df.sort_values(by=sort_keys).reset_index(drop=True)

        group_col = "Flight_ID" if "Flight_ID" in cleaned_df.columns else None

        if group_col:
            for feature in self.core_features:
                if feature in cleaned_df.columns:
                    cleaned_df[feature] = (
                        cleaned_df.groupby(group_col)[feature]
                        .transform(lambda s: s.interpolate(method="linear", limit_direction="both").ffill().bfill())
                    )
        else:
            for feature in self.core_features:
                if feature in cleaned_df.columns:
                    cleaned_df[feature] = (
                        cleaned_df[feature]
                        .interpolate(method="linear", limit_direction="both")
                        .ffill()
                        .bfill()
                    )

        cleaned_df = cleaned_df.dropna(subset=self.core_features).reset_index(drop=True)
        return cleaned_df

    def _clip_series(self, series: pd.Series) -> pd.Series:
        """Clips a single series using the configured outlier method."""
        if self.outlier_method == "iqr":
            q25, q75 = series.quantile(0.25), series.quantile(0.75)
            iqr = q75 - q25
            lower_bound = q25 - (self.outlier_threshold * iqr)
            upper_bound = q75 + (self.outlier_threshold * iqr)
        elif self.outlier_method == "zscore":
            mean, std = series.mean(), series.std()
            if std == 0 or np.isnan(std):
                return series
            lower_bound = mean - (self.outlier_threshold * std)
            upper_bound = mean + (self.outlier_threshold * std)
        else:
            raise ValueError(f"Unknown outlier method: {self.outlier_method}")
        return series.clip(lower=lower_bound, upper=upper_bound)

    def remove_sensor_noise_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clips extreme sensor spikes per flight using IQR/Z-score bounds (Winsorization)."""
        filtered_df = df.copy()
        group_col = "Flight_ID" if "Flight_ID" in filtered_df.columns else None

        for col in self.core_features:
            if group_col:
                filtered_df[col] = filtered_df.groupby(group_col)[col].transform(
                    self._clip_series
                )
            else:
                filtered_df[col] = self._clip_series(filtered_df[col])

        return filtered_df

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Executes full cleaning pipeline."""
        df_std = self.standardize_columns(df)
        df_imputed = self.handle_missing_data(df_std)
        df_cleaned = self.remove_sensor_noise_outliers(df_imputed)
        return df_cleaned
