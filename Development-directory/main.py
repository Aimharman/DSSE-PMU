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

from state_estimator import StateEstimator
from wls import WeightedLeastSquares
from chi_square import ChiSquareDetector
from jacobian import compute_jacobian

###########################################################################
# CONFIGURATION
###########################################################################

CSV_FILE = os.path.join(os.path.dirname(__file__), "PMU_Output.csv")

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

    if perform_localization:
        index, label, score = detector.localize_bad_data(
            residual,
            W,
            estimator.measurement_names,
        )
        print("\nMost Suspicious Measurement")
        print(f"Index : {index}")
        print(f"Label : {label}")
        print(f"Score : {score:.6f}")

        if bad_data:
            print("\nRe-running WLS after down-weighting the suspicious measurement...")
            x_final, residual, W = solver.solve(
                estimator.z,
                estimator.x,
                bad_data_index=index,
                bad_data_weight=0.1,
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
    }


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