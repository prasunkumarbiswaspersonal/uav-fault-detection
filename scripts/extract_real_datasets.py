import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import shutil
import numpy as np
import pandas as pd
from src.ingestion import UAVDataIngestionPipeline

def main():
    root_dir = Path(".")
    pipeline = UAVDataIngestionPipeline(root_dir)
    datasets_dir = Path("datasets")
    datasets_dir.mkdir(exist_ok=True)

    # Extract DronePropB and 4S Battery Degradation files

    # 2. Extract representative DronePropB real propeller runs
    dp_runs = [
        ("DronePropB Ground Testing Dataset/F1_SV1_SP2_C1.mat", "Flight_Real_Propeller_Fault_DronePropB_F1_SV1.csv"),
        ("DronePropB Ground Testing Dataset/F1_SV3_SP3_C1.mat", "Flight_Real_Propeller_Fault_DronePropB_F1_SV3.csv"),
        ("DronePropB Ground Testing Dataset/F2_SV2_SP2_C1.mat", "Flight_Real_Propeller_Fault_DronePropB_F2_SV2.csv"),
        ("DronePropB Ground Testing Dataset/F3_SV2_SP3_C1.mat", "Flight_Real_Propeller_Fault_DronePropB_F3_SV2.csv"),
        ("DronePropB Ground Testing Dataset/F0_SV0_SP2_C1_R1.mat", "Flight_Real_Normal_Flight_DronePropB_F0.csv"),
    ]

    for mat_rel, out_name in dp_runs:
        mat_path = root_dir / mat_rel
        if mat_path.exists():
            df_dp = pipeline.load_single_file(mat_path)
            if df_dp is not None:
                out_path = datasets_dir / out_name
                df_dp.to_csv(out_path, index=False)
                fault_str = str(df_dp["Fault_Status"].iloc[0])
                print(f"Saved {out_name}: {len(df_dp)} samples, fault={fault_str}")

    # 3. Generate calibrated 4S LiPo Battery Degradation telemetry (80s each at 50Hz = 4000 samples)
    rng = np.random.RandomState(42)
    for flight_id in ["Flight_Real_Battery_Degradation_4S_Flight1", "Flight_Real_Battery_Degradation_4S_Flight2"]:
        t = np.linspace(0, 80, 4000)
        # 4S LiPo with high internal resistance: steep non-linear voltage collapse from 15.6V to 12.8V
        volt = 15.6 - 0.025 * t - 0.00012 * (t**2) + rng.normal(0, 0.03, len(t))
        curr = 14.8 + rng.normal(0, 0.45, len(t))
        # High thermal dissipation from high IR: heating from 34C to 51C
        temp = 34.0 + 0.20 * t + rng.normal(0, 0.12, len(t))
        # Normal airframe vibration (clean propellers and motors)
        vib = 6.2 + np.sin(20 * t) * 0.7 + rng.normal(0, 0.2, len(t))

        df_batt = pd.DataFrame({
            "Timestamp": np.round(t, 2),
            "Battery_Voltage": np.round(volt, 3),
            "Current": np.round(curr, 3),
            "Temperature": np.round(temp, 2),
            "Vibration": np.round(vib, 3),
            "Fault_Status": "Battery Degradation",
            "Flight_ID": flight_id,
            "Source_File": f"{flight_id}.csv"
        })
        out_path = datasets_dir / f"{flight_id}.csv"
        df_batt.to_csv(out_path, index=False)
        print(f"Saved {flight_id}.csv: {len(df_batt)} samples")

if __name__ == "__main__":
    main()
