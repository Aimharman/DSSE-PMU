"""
===========================================================
main.py

Cyber Resilient PDC Automation Technique for Faulty PMU Detection

Automatic PDC monitoring:

PMU_Output.csv
      |
      v
Automatic time/sample scan
      |
      v
WLS state estimation
      |
      v
Global chi-square validation
      |
      +---- PASS ------------------> Healthy snapshot
      |
      +---- FAIL
             |
             v
      PMU residual-energy screening
             |
       concentrated?
          /       \
        NO         YES
        |           |
   model/noise     candidate PMU
                    |
                    v
             isolate candidate
                    |
                    v
              WLS re-estimation
                    |
                    v
             corrected chi-square
                    |
          correction successful?
             /             \
           NO               YES
           |                 |
       reject candidate   CONFIRMED
                           FAULT
                           |
                           v
                       record event

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


###########################################################################
# CONFIGURATION
###########################################################################

CSV_FILE = os.path.join(
    os.path.dirname(__file__),
    "PMU_Output.csv",
)

ADC_RATE = 1000.0
PDC_SCAN_RATE_HZ = 50.0

SCAN_INTERVAL_SAMPLES = max(
    1,
    int(round(ADC_RATE / PDC_SCAN_RATE_HZ)),
)

RESULTS_FILE = os.path.join(
    os.path.dirname(__file__),
    "PDC_Detection_Results.csv",
)

# PMU-level screening.
PMU_SHARE_THRESHOLD = 0.55

# After isolating a candidate PMU, the residual must fall
# below the reduced chi-square threshold.
MIN_CHI_SQUARE_REDUCTION = 0.50


###########################################################################
# HELPERS
###########################################################################

def _valid_measurement_rows(df):
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


def _actual_fault_pmu(row):
    """
    Ground truth is used ONLY for final evaluation/reporting.
    It never enters WLS, chi-square screening, localization,
    or isolation.
    """

    faulty = []

    for bus in range(1, 4):
        column = f"PMU{bus} Bad Data"

        if column in row.index:
            try:
                value = bool(row[column])
            except (TypeError, ValueError):
                value = False

            if value:
                faulty.append(f"PMU{bus}")

    return ",".join(faulty)


def _pmu_indices(pmu_name):
    number = int(
        str(pmu_name).replace("PMU", "")
    )

    start = 4 * (number - 1)

    return list(
        range(start, start + 4)
    )


def _run_wls(
    solver,
    z,
    x0,
    **kwargs,
):
    with contextlib.redirect_stdout(
        io.StringIO()
    ):
        return solver.solve(
            z,
            x0,
            **kwargs,
        )


###########################################################################
# ONE AUTOMATIC PDC SNAPSHOT
###########################################################################

def _run_one_snapshot(
    csv_file,
    sample_index,
    apply_sync_correction=False,
):
    estimator = StateEstimator(
        csv_file,
        sample_index=sample_index,
    )

    estimator.run(
        apply_sync_correction=apply_sync_correction,
        verbose=False,
    )

    solver = WeightedLeastSquares()
    detector = ChiSquareDetector(
        pmu_share_threshold=PMU_SHARE_THRESHOLD
    )

    # ---------------------------------------------------------
    # Baseline WLS
    # ---------------------------------------------------------

    (
        x_final,
        residual,
        W,
        active_indices,
    ) = _run_wls(
        solver,
        estimator.z,
        estimator.x,
    )

    (
        initial_bad_data,
        initial_J,
        initial_threshold,
    ) = detector.detect(
        residual,
        W,
        len(active_indices),
        len(x_final),
        verbose=False,
    )

    # ---------------------------------------------------------
    # PMU screening
    # ---------------------------------------------------------

    screening = detector.screen_fault_candidate(
        residual,
        W,
        initial_J,
        initial_threshold,
        estimator.measurement_names,
    )

    detected_pmu = ""
    isolated_indices = []
    corrected_J = initial_J
    corrected_threshold = initial_threshold
    corrected_bad_data = initial_bad_data
    corrected_active_count = len(active_indices)

    best_reduction = 0.0
    best_candidate = ""
    best_screen_energy = screening["energy"]
    best_screen_share = screening["share"]

    # ---------------------------------------------------------
    # Candidate PMU isolation
    #
    # Test PMUs in descending residual-energy order.
    # This avoids permanently trusting a single PMU ranking.
    # ---------------------------------------------------------

    if initial_bad_data and screening["candidate"]:

        candidates = screening["pmu_data"]

        for candidate in candidates:

            pmu_name = candidate["pmu"]
            pmu_indices = candidate["indices"]

            (
                x_test,
                residual_test,
                W_test,
                active_test,
            ) = _run_wls(
                solver,
                estimator.z,
                x_final,
                bad_data_indices=pmu_indices,
                bad_data_weight=0.0,
            )

            (
                test_bad_data,
                test_J,
                test_threshold,
            ) = detector.detect(
                residual_test,
                W_test,
                len(active_test),
                len(x_test),
                verbose=False,
            )

            if initial_J > 0.0:
                reduction = (
                    initial_J - test_J
                ) / initial_J
            else:
                reduction = 0.0

            # A valid isolation must:
            # 1. remove the global chi-square failure,
            # 2. produce a substantial residual reduction.
            valid_isolation = (
                not test_bad_data
                and reduction >= MIN_CHI_SQUARE_REDUCTION
            )

            if valid_isolation:
                x_final = x_test
                residual = residual_test
                W = W_test
                active_indices = active_test

                detected_pmu = pmu_name
                isolated_indices = list(
                    pmu_indices
                )

                corrected_J = test_J
                corrected_threshold = test_threshold
                corrected_bad_data = test_bad_data
                corrected_active_count = len(
                    active_test
                )

                best_reduction = reduction
                best_candidate = pmu_name

                break

            # Keep the strongest attempted correction
            # for diagnostics even if it did not pass.
            if reduction > best_reduction:
                best_reduction = reduction
                best_candidate = pmu_name
                corrected_J = test_J
                corrected_threshold = test_threshold

    row = estimator.df.iloc[sample_index]

    return {
        "sample_index": int(sample_index),
        "time_s": float(
            estimator.selected_timestamp
        ),
        "sync_corrected": bool(
            apply_sync_correction
        ),

        "initial_chi_square": float(initial_J),
        "initial_threshold": float(
            initial_threshold
        ),
        "initial_status": (
            "FAIL"
            if initial_bad_data
            else "PASS"
        ),

        "pmu_screen_candidate": bool(
            screening["candidate"]
        ),
        "pmu_screened": screening["pmu"],
        "pmu_screen_energy": float(
            screening["energy"]
        ),
        "pmu_screen_share": float(
            screening["share"]
        ),
        "pmu_screen_threshold": float(
            screening["threshold"]
        ),

        "detected_pmu": detected_pmu,
        "pmu_isolated": ",".join(
            str(x)
            for x in isolated_indices
        ),

        "corrected_chi_square": float(
            corrected_J
        ),
        "corrected_threshold": float(
            corrected_threshold
        ),
        "corrected_status": (
            "FAIL"
            if corrected_bad_data
            else "PASS"
        ),
        "chi_square_reduction": float(
            best_reduction
        ),
        "correction_candidate": best_candidate,
        "active_measurements": int(
            corrected_active_count
        ),

        # Ground truth is strictly for evaluation.
        "actual_faulty_pmu": _actual_fault_pmu(
            row
        ),
    }


###########################################################################
# AUTOMATIC PDC SCAN
###########################################################################

def run_automatic_scan(
    csv_file,
    apply_sync_correction=False,
):
    df = pd.read_csv(csv_file)

    valid_indices = _valid_measurement_rows(
        df
    )

    if len(valid_indices) == 0:
        raise RuntimeError(
            "No complete PMU measurement rows "
            "were found in the CSV."
        )

    first_valid = int(
        valid_indices[0]
    )
    last_valid = int(
        valid_indices[-1]
    )

    scan_indices = np.arange(
        first_valid,
        last_valid + 1,
        SCAN_INTERVAL_SAMPLES,
        dtype=int,
    )

    if scan_indices[-1] != last_valid:
        scan_indices = np.append(
            scan_indices,
            last_valid,
        )

    print("\n==============================================")
    print(" Automatic PDC Monitoring")
    print("==============================================")
    print(
        f"CSV samples             : {len(df)}"
    )
    print(
        f"Valid DSSE samples      : "
        f"{len(valid_indices)}"
    )
    print(
        f"PDC scan rate           : "
        f"{PDC_SCAN_RATE_HZ:.1f} Hz"
    )
    print(
        f"Scan interval           : "
        f"{SCAN_INTERVAL_SAMPLES} samples"
    )
    print(
        f"First evaluated sample  : "
        f"{first_valid}"
    )
    print(
        f"Last evaluated sample   : "
        f"{last_valid}"
    )
    print(
        f"Evaluation points       : "
        f"{len(scan_indices)}"
    )
    print("==============================================")

    results = []
    previous_detected = None
    event_count = 0

    for sample_index in scan_indices:

        result = _run_one_snapshot(
            csv_file,
            int(sample_index),
            apply_sync_correction=apply_sync_correction,
        )

        results.append(result)

        detected = result["detected_pmu"]

        if detected:

            if detected != previous_detected:
                event_count += 1

                print(
                    f"\n[FAULT EVENT {event_count}] "
                    f"t={result['time_s']:.3f} s | "
                    f"Detected={detected} | "
                    f"Actual="
                    f"{result['actual_faulty_pmu'] or 'None'} | "
                    f"chi2="
                    f"{result['initial_chi_square']:.4f} | "
                    f"PMU-share="
                    f"{result['pmu_screen_share']:.3f} | "
                    f"corrected="
                    f"{result['corrected_chi_square']:.4f} | "
                    f"reduction="
                    f"{result['chi_square_reduction']:.3f}"
                )

            previous_detected = detected

        else:
            previous_detected = None

    result_df = pd.DataFrame(results)

    result_df.to_csv(
        RESULTS_FILE,
        index=False,
    )

    #######################################################################
    # Summary
    #######################################################################

    detected_rows = result_df[
        result_df["detected_pmu"]
        .astype(str)
        .str.len() > 0
    ]

    candidate_rows = result_df[
        result_df["pmu_screen_candidate"]
    ]

    print("\n==============================================")
    print(" Automatic Monitoring Summary")
    print("==============================================")
    print(
        f"Evaluation points       : "
        f"{len(result_df)}"
    )
    print(
        f"Global chi-square fails : "
        f"{int((result_df['initial_status'] == 'FAIL').sum())}"
    )
    print(
        f"PMU candidates          : "
        f"{len(candidate_rows)}"
    )
    print(
        f"Confirmed fault points  : "
        f"{len(detected_rows)}"
    )
    print(
        f"Confirmed fault events  : "
        f"{event_count}"
    )

    for pmu in ["PMU1", "PMU2", "PMU3"]:

        detected_count = int(
            (
                result_df["detected_pmu"]
                == pmu
            ).sum()
        )

        actual_count = int(
            result_df[
                "actual_faulty_pmu"
            ]
            .astype(str)
            .str.contains(
                pmu,
                regex=False,
            )
            .sum()
        )

        true_positive = int(
            (
                (result_df["detected_pmu"] == pmu)
                &
                (
                    result_df["actual_faulty_pmu"]
                    .astype(str)
                    .str.contains(
                        pmu,
                        regex=False,
                    )
                )
            ).sum()
        )

        false_positive = int(
            (
                (result_df["detected_pmu"] == pmu)
                &
                ~(
                    result_df["actual_faulty_pmu"]
                    .astype(str)
                    .str.contains(
                        pmu,
                        regex=False,
                    )
                )
            ).sum()
        )

        print(
            f"{pmu}: "
            f"detected={detected_count}, "
            f"ground_truth={actual_count}, "
            f"TP={true_positive}, "
            f"FP={false_positive}"
        )

    print(
        f"\nResults CSV : {RESULTS_FILE}"
    )
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
