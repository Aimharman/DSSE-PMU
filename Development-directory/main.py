"""
===========================================================
main.py
===========================================================

Cyber Resilient PDC Automation Technique for Faulty PMU Detection

Automatic PDC-style monitoring:

PMU_Output.csv
      |
      v
Automatic time/sample scan
      |
      v
WLS state estimation
      |
      v
Chi-square validation
      |
      +---- PASS ----> Record healthy snapshot
      |
      +---- FAIL ----> Faulty PMU localization
                            |
                            v
                       PMU isolation
                            |
                            v
                      WLS re-estimation
                            |
                            v
                     Corrected chi-square
                            |
                            v
                       Record event

No manually selected simulation time is required.
===========================================================
"""

import contextlib
import io
import os

import numpy as np
import pandas as pd

from state_estimator import StateEstimator
from wls import WeightedLeastSquares
from chi_square import ChiSquareDetector
from jacobian import compute_jacobian


###########################################################################
# CONFIGURATION
###########################################################################

CSV_FILE = os.path.join(
    os.path.dirname(__file__),
    "PMU_Output.csv",
)

# PMU simulator rate.
ADC_RATE = 1000.0

# PDC evaluation rate.
# 50 evaluations/sec = one DSSE evaluation every 20 PMU samples.
PDC_SCAN_RATE_HZ = 50.0

SCAN_INTERVAL_SAMPLES = max(
    1,
    int(round(ADC_RATE / PDC_SCAN_RATE_HZ)),
)

# Keep console output compact during automatic scanning.
PRINT_EACH_SNAPSHOT = False

# Save the complete automatic-monitoring result table.
RESULTS_FILE = os.path.join(
    os.path.dirname(__file__),
    "PDC_Detection_Results.csv",
)


###########################################################################
# HELPERS
###########################################################################

def _valid_measurement_rows(df):
    """
    Return indices where all 12 PMU measurements needed by DSSE
    are available.

    The simulator can contain initial NaN DFT values while the first
    complete cycle is being accumulated. Those rows are not evaluated.
    """

    columns = []

    for bus in range(1, 4):
        columns.extend([
            f"PMU{bus} Voltage Magnitude",
            f"PMU{bus} Voltage Phase",
            f"PMU{bus} Current Magnitude",
            f"PMU{bus} Current Phase",
        ])

    mask = df[columns].notna().all(axis=1)

    return df.index[mask].to_numpy()


def _actual_fault_pmU(row):
    """
    Read simulator ground truth only for evaluation/reporting.

    This value is NEVER used by WLS, chi-square detection,
    localization, or isolation.
    """

    faulty = []

    for bus in range(1, 4):
        column = f"PMU{bus} Bad Data"

        if column in row.index:
            value = row[column]

            if bool(value):
                faulty.append(f"PMU{bus}")

    if not faulty:
        return ""

    return ",".join(faulty)


def _run_one_snapshot(
    csv_file,
    sample_index,
    apply_sync_correction=False,
    perform_localization=True,
):
    """
    Run one complete DSSE cycle for one CSV snapshot.

    WLS prints are suppressed here because automatic monitoring can
    process hundreds of snapshots. Only fault events are printed by
    the outer scan.
    """

    estimator = StateEstimator(
        csv_file,
        sample_index=sample_index,
    )

    estimator.run(
        apply_sync_correction=apply_sync_correction,
        verbose=False,
    )

    solver = WeightedLeastSquares()

    # Suppress detailed WLS iteration output during automatic scanning.
    with contextlib.redirect_stdout(io.StringIO()):

        (
            x_final,
            residual,
            W,
            active_indices,
        ) = solver.solve(
            estimator.z,
            estimator.x,
        )

    detector = ChiSquareDetector()

    (
        initial_bad_data,
        initial_J,
        initial_threshold,
    ) = detector.detect(
        residual,
        W,
        len(active_indices),
        len(x_final),
    )

    faulty_pmu = ""
    isolated_indices = []
    corrected_J = initial_J
    corrected_threshold = initial_threshold
    corrected_bad_data = initial_bad_data
    corrected_active_count = len(active_indices)

    if initial_bad_data and perform_localization:

        (
            pmu_name,
            pmu_indices,
            pmu_score,
        ) = detector.localize_faulty_pmu(
            residual,
            W,
            estimator.measurement_names,
        )

        faulty_pmu = pmu_name
        isolated_indices = list(pmu_indices)

        with contextlib.redirect_stdout(io.StringIO()):

            (
                x_final,
                residual,
                W,
                active_indices,
            ) = solver.solve(
                estimator.z,
                x_final,
                bad_data_indices=pmu_indices,
                bad_data_weight=0.0,
            )

        (
            corrected_bad_data,
            corrected_J,
            corrected_threshold,
        ) = detector.detect(
            residual,
            W,
            len(active_indices),
            len(x_final),
        )

        corrected_active_count = len(active_indices)

    row = estimator.df.iloc[sample_index]

    return {
        "sample_index": int(sample_index),
        "time_s": float(estimator.selected_timestamp),
        "sync_corrected": bool(apply_sync_correction),

        "initial_chi_square": float(initial_J),
        "initial_threshold": float(initial_threshold),
        "initial_status": "FAIL" if initial_bad_data else "PASS",

        "detected_pmu": faulty_pmu,
        "pmu_isolated": ",".join(isolated_indices.__str__().strip("[]").split(", ")),

        "corrected_chi_square": float(corrected_J),
        "corrected_threshold": float(corrected_threshold),
        "corrected_status": (
            "FAIL" if corrected_bad_data else "PASS"
        ),
        "active_measurements": int(corrected_active_count),

        # Ground truth is recorded for evaluation only.
        "actual_faulty_pmu": _actual_fault_pmU(row),
    }


###########################################################################
# AUTOMATIC PDC SCAN
###########################################################################

def run_automatic_scan(
    csv_file,
    apply_sync_correction=False,
):
    """
    Scan the complete PMU simulation automatically.

    The scanner evaluates the CSV at the configured PDC rate and
    performs the complete WLS -> chi-square -> localization ->
    isolation -> re-estimation chain whenever required.
    """

    df = pd.read_csv(csv_file)

    valid_indices = _valid_measurement_rows(df)

    if len(valid_indices) == 0:
        raise RuntimeError(
            "No complete PMU measurement rows were found in the CSV."
        )

    first_valid = int(valid_indices[0])
    last_valid = int(valid_indices[-1])

    scan_indices = np.arange(
        first_valid,
        last_valid + 1,
        SCAN_INTERVAL_SAMPLES,
        dtype=int,
    )

    # Ensure the final valid sample is included.
    if scan_indices[-1] != last_valid:
        scan_indices = np.append(scan_indices, last_valid)

    print("\n==============================================")
    print(" Automatic PDC Monitoring")
    print("==============================================")
    print(f"CSV samples             : {len(df)}")
    print(f"Valid DSSE samples      : {len(valid_indices)}")
    print(f"PDC scan rate           : {PDC_SCAN_RATE_HZ:.1f} Hz")
    print(f"Scan interval           : {SCAN_INTERVAL_SAMPLES} samples")
    print(f"First evaluated sample  : {first_valid}")
    print(f"Last evaluated sample   : {last_valid}")
    print(f"Evaluation points       : {len(scan_indices)}")
    print("==============================================")

    results = []

    previous_detected = None
    event_count = 0

    for sample_index in scan_indices:

        result = _run_one_snapshot(
            csv_file,
            int(sample_index),
            apply_sync_correction=apply_sync_correction,
            perform_localization=True,
        )

        results.append(result)

        detected = result["detected_pmu"]

        if detected:
            # Print only the beginning of a new detected event.
            if detected != previous_detected:
                event_count += 1

                print(
                    f"\n[FAULT EVENT {event_count}] "
                    f"t={result['time_s']:.3f} s | "
                    f"Detected={detected} | "
                    f"Actual={result['actual_faulty_pmu'] or 'None'} | "
                    f"chi2={result['initial_chi_square']:.4f} | "
                    f"corrected={result['corrected_chi_square']:.4f}"
                )

            previous_detected = detected

        else:
            previous_detected = None

    result_df = pd.DataFrame(results)
    result_df.to_csv(RESULTS_FILE, index=False)

    #######################################################################
    # Summary
    #######################################################################

    detected_rows = result_df[
        result_df["detected_pmu"].astype(str).str.len() > 0
    ]

    print("\n==============================================")
    print(" Automatic Monitoring Summary")
    print("==============================================")
    print(f"Evaluation points       : {len(result_df)}")
    print(f"Detected fault points   : {len(detected_rows)}")
    print(f"Fault events            : {event_count}")

    for pmu in ["PMU1", "PMU2", "PMU3"]:
        count = int(
            (result_df["detected_pmu"] == pmu).sum()
        )
        actual = int(
            result_df["actual_faulty_pmu"].astype(str)
            .str.contains(pmu, regex=False)
            .sum()
        )

        print(
            f"{pmu}: detected={count}, "
            f"ground_truth_points={actual}"
        )

    print(f"\nResults CSV : {RESULTS_FILE}")
    print("==============================================")

    return result_df


###########################################################################
# MAIN
###########################################################################

def main():
    run_automatic_scan(
        CSV_FILE,
        apply_sync_correction=False,
    )


if __name__ == "__main__":
    main()