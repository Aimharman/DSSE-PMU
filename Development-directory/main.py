"""
===========================================================
main.py

Project Entry Point

Workflow

PMU_Output.csv
        │
        ▼
State Estimator
        │
        ▼
Measurement Vector (z)
        │
        ▼
Initial State Vector (x0)
===========================================================
"""

import os
import numpy as np
import pandas as pd

from state_estimator import StateEstimator
from wls import WeightedLeastSquares
from chi_square import ChiSquareDetector
from jacobian import compute_jacobian
from measurement_model import measurement_model

###########################################################################
# CONFIGURATION
###########################################################################

CSV_FILE = os.path.join(os.path.dirname(__file__), "PMU_Output.csv")
#CSV_FILE = os.path.join(os.path.dirname(__file__), "PMU_Output_Faulty_PMU3_60deg.csv")

###########################################################################
# MAIN
###########################################################################


def run_estimation(csv_file, apply_sync_correction=False, perform_localization=True):

    print("==============================================")
    print(" Distribution System State Estimator")
    print("==============================================")

    estimator = StateEstimator(csv_file)
    estimator.run(apply_sync_correction=apply_sync_correction)

    H = compute_jacobian(estimator.x)

    print("\nJacobian Matrix\n")
    print(H)

    print("\nShape :", H.shape)

    print("\nStage 3.4 Complete")

    solver = WeightedLeastSquares()

    x_final, residual, W = solver.solve(
        estimator.z,
        estimator.x,
    )

    print("\nEstimated State")
    print(x_final)

    detector = ChiSquareDetector()
    bad_data, J, threshold = detector.detect(
        residual,
        W,
        len(estimator.z),
        len(x_final),
    )

    faulty_pmu = []
    if bad_data and perform_localization:
        index, label, score = detector.localize_bad_data(
            residual,
            W,
            estimator.measurement_names,
        )
        print("\nMost Suspicious Measurement")
        print(f"Index : {index}")
        print(f"Label : {label}")
        print(f"Score : {score:.6f}")

        print("\nRe-running WLS after down-weighting the suspicious measurement...")
        x_final, residual, W = solver.solve(
            estimator.z,
            x_final,
            bad_data_index=index,
            bad_data_weight=0.1,
        )
        bad_data, J, threshold = detector.detect(
            residual,
            W,
            len(estimator.z),
            len(x_final),
        )

    if perform_localization:
        faulty_pmu = detect_faulty_pmu_history(
            csv_file,
            window_size=5,
            threshold=0.5,
        )
        if faulty_pmu:
            print("\nSuspected Faulty PMUs")
            for item in faulty_pmu:
                print(item)

            suspected_pmu = faulty_pmu[0]["pmu"]
            suspected_index = int(suspected_pmu.replace("PMU", "")) - 1
            pmu_indices = list(range(4 * suspected_index, 4 * suspected_index + 4))

            print(f"\nIsolating {suspected_pmu} and re-estimating...")
            x_final, residual, W = solver.solve(
                estimator.z,
                x_final,
                bad_data_indices=pmu_indices,
                bad_data_weight=0.05,
            )
            bad_data, J, threshold = detector.detect(
                residual,
                W,
                len(estimator.z),
                len(x_final),
            )

    return {
        "state": x_final,
        "residual": residual,
        "weight_matrix": W,
        "bad_data": bad_data,
        "chi_square_statistic": J,
        "threshold": threshold,
        "sync_correction": apply_sync_correction,
        "sync_offsets": estimator.sync_offsets_used,
        "faulty_pmu": faulty_pmu,
    }


def detect_faulty_pmu_history(csv_file, window_size=5, threshold=0.5):
    """
    Aggregates residual energy over consecutive timestamps and identifies
    the PMU with persistently abnormal behavior.

    This intentionally avoids rebuilding a full StateEstimator object for
    every row in the CSV, because that pattern creates the dominant runtime
    cost in a rolling-window scan over a large dataset.
    """

    df = pd.read_csv(csv_file)
    detector = ChiSquareDetector()
    residual_history = []

    for idx in range(len(df)):
        row = df.iloc[idx]

        measurements = []
        x0 = []

        for bus in range(1, 4):
            voltage_mag = row[f"PMU{bus} Voltage Magnitude"]
            voltage_phase = np.deg2rad(row[f"PMU{bus} Voltage Phase"])
            current_mag = row[f"PMU{bus} Current Magnitude"]
            current_phase = np.deg2rad(row[f"PMU{bus} Current Phase"])

            measurements.extend([
                voltage_mag,
                voltage_phase,
                current_mag,
                current_phase,
            ])
            x0.extend([
                voltage_mag,
                voltage_phase,
            ])

        z = np.asarray(measurements, dtype=float)
        x = np.asarray(x0, dtype=float)
        h = measurement_model(x)
        residual_history.append(np.abs(z - h))

    residual_history = np.asarray(residual_history)
    pmu_names = [f"PMU{idx + 1}" for idx in range(residual_history.shape[1] // 4)]

    return detector.detect_faulty_pmu(
        residual_history,
        pmu_names=pmu_names,
        window_size=max(1, window_size),
        threshold=threshold,
    )


def run_comparison(csv_file):
    print("\n==============================================")
    print(" Synchronization Comparison Study")
    print("==============================================")

    baseline = run_estimation(csv_file, apply_sync_correction=False)
    corrected = run_estimation(csv_file, apply_sync_correction=True)

    baseline_state = np.asarray(baseline["state"], dtype=float)
    corrected_state = np.asarray(corrected["state"], dtype=float)

    state_difference = np.linalg.norm(corrected_state - baseline_state)

    print("\nComparison Summary")
    print("------------------")
    print(f"State difference norm : {state_difference:.6f}")
    print(f"Baseline chi-square  : {baseline['chi_square_statistic']:.6f}")
    print(f"Corrected chi-square : {corrected['chi_square_statistic']:.6f}")
    print(f"Baseline threshold   : {baseline['threshold']:.6f}")
    print(f"Corrected threshold  : {corrected['threshold']:.6f}")

    return {
        "baseline": baseline,
        "corrected": corrected,
        "state_difference": state_difference,
    }


def main():
    run_comparison(CSV_FILE)


if __name__ == "__main__":
    main()