"""
Cyber Resilient PDC Automation Technique for Faulty PMU Detection

Automatic PDC scan with robust empirical chi-square calibration.

The detector deliberately separates:
    1. model/noise residuals present in healthy operation, and
    2. gross residuals produced by injected faulty-PMU data.

No simulation time is entered by the user and simulator ground truth is
never used to make a detection/isolation decision.
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(BASE_DIR, "PMU_Output.csv")
RESULTS_FILE = os.path.join(BASE_DIR, "PDC_Detection_Results.csv")

ADC_RATE = 1000.0
PDC_SCAN_RATE_HZ = 50.0
SCAN_INTERVAL_SAMPLES = max(1, int(round(ADC_RATE / PDC_SCAN_RATE_HZ)))

# PMU localization screening.
PMU_SHARE_THRESHOLD = 0.55

# Isolation must reduce the original J by at least this fraction.
MIN_CHI_SQUARE_REDUCTION = 0.50

# Robust calibration: threshold = median + K * 1.4826 * MAD.
ROBUST_SIGMA_MULTIPLIER = 6.0

# Group consecutive detections of the same PMU into one event.
EVENT_MAX_GAP_S = 0.06


###########################################################################
# CSV HELPERS
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
    return df.index[df[columns].notna().all(axis=1)].to_numpy()


def _truth_bool(value):
    """Convert CSV truth/flag values to a reliable boolean."""
    if pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "1.0", "yes", "y"}


def _actual_fault_pmu_window(df, sample_index, window_samples=SCAN_INTERVAL_SAMPLES):
    """
    Return simulator ground truth for the PDC snapshot, using the raw
    measurement window that contributes to that snapshot.

    IMPORTANT:
        This function is used ONLY for post-run evaluation.  It is never
        called by WLS, chi-square detection, PMU screening, or isolation.

    A random raw-sample fault does not necessarily occur on the exact CSV
    row selected by the PDC.  Therefore checking only row[sample_index] can
    produce false FP/FN statistics.  For a 1000-Hz ADC and 50-Hz PDC, the
    PMU reporting window contains 20 raw samples.

    A PMU is marked faulty for the PDC snapshot if ANY raw sample in that
    contributing window carries its simulator Bad Data flag.
    """
    start = max(0, int(sample_index) - int(window_samples) + 1)
    stop = min(len(df), int(sample_index) + 1)

    faulty = []
    for bus in range(1, 4):
        col = f"PMU{bus} Bad Data"
        if col not in df.columns:
            continue

        flags = df.iloc[start:stop][col]
        if flags.map(_truth_bool).any():
            faulty.append(f"PMU{bus}")

    return ",".join(faulty)


def _fault_window_description(df, sample_index, window_samples=SCAN_INTERVAL_SAMPLES):
    """Return raw-sample/time limits used only for audit/reporting."""
    start = max(0, int(sample_index) - int(window_samples) + 1)
    stop = min(len(df), int(sample_index) + 1)
    if stop <= start:
        return start, max(start, stop - 1), float("nan"), float("nan")

    times = pd.to_numeric(df.iloc[start:stop].iloc[:, 0], errors="coerce")
    return (
        start,
        stop - 1,
        float(times.iloc[0]),
        float(times.iloc[-1]),
    )


def _run_wls(solver, z, x0, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return solver.solve(z, x0, **kwargs)


###########################################################################
# BASELINE WLS PASS
###########################################################################

def collect_baseline_statistics(csv_file, scan_indices, apply_sync_correction=False):
    """Run WLS over the complete scan and collect J values.

    This pass is only used to establish the robust operating threshold.
    No simulator fault flag is inspected here.
    """
    solver = WeightedLeastSquares()
    detector = ChiSquareDetector(
        pmu_share_threshold=PMU_SHARE_THRESHOLD
    )

    snapshots = []
    statistics = []

    for sample_index in scan_indices:
        estimator = StateEstimator(csv_file, sample_index=int(sample_index))
        estimator.run(
            apply_sync_correction=apply_sync_correction,
            verbose=False,
        )

        x_final, residual, W, active_indices = _run_wls(
            solver,
            estimator.z,
            estimator.x,
        )

        J = detector.compute_statistic(residual, W)
        dof = detector.degrees_of_freedom(len(active_indices), len(x_final))

        snapshots.append({
            "sample_index": int(sample_index),
            "estimator": estimator,
            "x_final": x_final,
            "residual": residual,
            "W": W,
            "active_indices": active_indices,
            "dof": dof,
        })
        statistics.append(J)

    return snapshots, np.asarray(statistics, dtype=float), detector


###########################################################################
# ONE DETECTION / ISOLATION PASS
###########################################################################

def process_snapshot(snapshot, detector, csv_file, operational_threshold, truth_df=None):
    estimator = snapshot["estimator"]
    solver = WeightedLeastSquares()

    x_final = snapshot["x_final"]
    residual = snapshot["residual"]
    W = snapshot["W"]
    active_indices = snapshot["active_indices"]

    initial_bad, initial_J, _ = detector.detect(
        residual,
        W,
        len(active_indices),
        len(x_final),
        threshold_override=operational_threshold,
        verbose=False,
    )

    theoretical_threshold = detector.threshold(
        len(active_indices) - len(x_final)
    )

    screening = detector.screen_fault_candidate(
        residual,
        W,
        initial_J,
        operational_threshold,
        estimator.measurement_names,
    )

    detected_pmu = ""
    isolated_indices = []
    corrected_J = initial_J
    corrected_bad = initial_bad
    correction_candidate = ""
    best_reduction = 0.0
    best_candidate = ""
    corrected_active_count = len(active_indices)
    corrected_theoretical_threshold = theoretical_threshold

    # Only a global failure enters PMU localization.
    if initial_bad and screening["candidate"]:
        for candidate in screening["pmu_data"]:
            pmu_name = candidate["pmu"]
            pmu_indices = candidate["indices"]

            x_test, residual_test, W_test, active_test = _run_wls(
                solver,
                estimator.z,
                x_final,
                bad_data_indices=pmu_indices,
                bad_data_weight=0.0,
            )

            test_J = detector.compute_statistic(residual_test, W_test)
            test_dof = len(active_test) - len(x_test)
            test_theoretical = detector.threshold(test_dof)
            test_bad = bool(test_J > operational_threshold)

            reduction = ((initial_J - test_J) / initial_J) if initial_J > 0 else 0.0

            # A confirmed PMU must materially improve the fit and leave the
            # residual below the calibrated operating limit.
            valid = (
                not test_bad
                and reduction >= MIN_CHI_SQUARE_REDUCTION
            )

            if reduction > best_reduction:
                best_reduction = reduction
                best_candidate = pmu_name

            if valid:
                x_final = x_test
                residual = residual_test
                W = W_test
                active_indices = active_test
                detected_pmu = pmu_name
                isolated_indices = list(pmu_indices)
                corrected_J = test_J
                corrected_bad = test_bad
                correction_candidate = pmu_name
                corrected_active_count = len(active_test)
                corrected_theoretical_threshold = test_theoretical
                break

    # Ground truth is evaluated over the complete raw-sample window, not
    # merely at the selected PDC row.  This is especially important for
    # random injection, where a fault may occur between two PDC snapshots.
    csv_df = truth_df if truth_df is not None else pd.read_csv(csv_file)
    sample_index = int(snapshot["sample_index"])
    actual_fault = _actual_fault_pmu_window(
        csv_df, sample_index, SCAN_INTERVAL_SAMPLES
    )
    window_start, window_end, window_t0, window_t1 = _fault_window_description(
        csv_df, sample_index, SCAN_INTERVAL_SAMPLES
    )

    return {
        "sample_index": snapshot["sample_index"],
        "time_s": float(estimator.selected_timestamp),
        "initial_chi_square": float(initial_J),
        "theoretical_threshold": float(theoretical_threshold),
        "operating_threshold": float(operational_threshold),
        "initial_status": "FAIL" if initial_bad else "PASS",
        "pmu_screen_candidate": bool(screening["candidate"]),
        "pmu_screened": screening["pmu"],
        "pmu_screen_energy": float(screening["energy"]),
        "pmu_screen_share": float(screening["share"]),
        "pmu_screen_threshold": float(screening["threshold"]),
        "detected_pmu": detected_pmu,
        "pmu_isolated": ",".join(str(i) for i in isolated_indices),
        "corrected_chi_square": float(corrected_J),
        "corrected_theoretical_threshold": float(corrected_theoretical_threshold),
        "corrected_operating_threshold": float(operational_threshold),
        "corrected_status": "FAIL" if corrected_bad else "PASS",
        "chi_square_reduction": float(best_reduction),
        "correction_candidate": correction_candidate or best_candidate,
        "active_measurements": int(corrected_active_count),
        "actual_faulty_pmu": actual_fault,
        "truth_window_start": int(window_start),
        "truth_window_end": int(window_end),
        "truth_window_t0_s": float(window_t0),
        "truth_window_t1_s": float(window_t1),
    }


###########################################################################
# EVENT / PERFORMANCE METRICS
###########################################################################

def _event_count(df):
    events = 0
    last_pmu = ""
    last_time = None

    for _, row in df.iterrows():
        pmu = row["detected_pmu"]
        if not isinstance(pmu, str) or not pmu:
            continue
        t = float(row["time_s"])
        if pmu != last_pmu or last_time is None or (t - last_time) > EVENT_MAX_GAP_S:
            events += 1
        last_pmu = pmu
        last_time = t
    return events


def _print_metrics(df):
    print("\n==============================================")
    print(" Automatic Monitoring Summary")
    print("==============================================")
    print(f"Evaluation points       : {len(df)}")
    print(f"Global chi-square fails : {int((df['initial_status'] == 'FAIL').sum())}")
    print(f"PMU candidates          : {int(df['pmu_screen_candidate'].sum())}")
    print(f"Confirmed fault points  : {int((df['detected_pmu'] != '').sum())}")
    print(f"Confirmed fault events  : {_event_count(df)}")

    for pmu in ["PMU1", "PMU2", "PMU3"]:
        detected = df["detected_pmu"] == pmu
        actual = df["actual_faulty_pmu"].astype(str).str.contains(pmu, regex=False)
        tp = int((detected & actual).sum())
        fp = int((detected & ~actual).sum())
        fn = int((~detected & actual).sum())
        tn = int((~detected & ~actual).sum())
        actual_count = int(actual.sum())

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0

        print(
            f"{pmu}: ground_truth={actual_count}, TP={tp}, FP={fp}, "
            f"FN={fn}, TN={tn}, precision={precision:.3f}, recall={recall:.3f}"
        )

    print("==============================================")


###########################################################################
# AUTOMATIC SCAN
###########################################################################

def run_automatic_scan(csv_file, apply_sync_correction=False):
    df = pd.read_csv(csv_file)
    valid_indices = _valid_measurement_rows(df)
    if len(valid_indices) == 0:
        raise RuntimeError("No complete PMU measurement rows were found in the CSV.")

    first_valid = int(valid_indices[0])
    last_valid = int(valid_indices[-1])
    scan_indices = np.arange(first_valid, last_valid + 1, SCAN_INTERVAL_SAMPLES, dtype=int)
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

    # ---------------------------------------------------------
    # PASS 1: robust calibration
    # ---------------------------------------------------------
    snapshots, statistics, detector = collect_baseline_statistics(
        csv_file,
        scan_indices,
        apply_sync_correction=apply_sync_correction,
    )

    dof = snapshots[0]["dof"]
    theoretical = detector.threshold(dof)
    operational = detector.calibrate(statistics, dof)

    print("\n----------------------------------------------")
    print(" Robust Chi-Square Calibration")
    print("----------------------------------------------")
    print(f"Degrees of freedom      : {dof}")
    print(f"Theoretical threshold   : {theoretical:.6f}")
    print(f"Baseline median J       : {detector.baseline_median:.6f}")
    print(f"Baseline MAD            : {detector.baseline_mad:.6f}")
    print(f"MAD scale factor        : {1.4826:.4f}")
    print(f"Robust sigma multiplier : {ROBUST_SIGMA_MULTIPLIER:.1f}")
    print(f"Operating threshold     : {operational:.6f}")
    print(f"Truth window            : {SCAN_INTERVAL_SAMPLES} raw samples")
    print("Ground truth is used only for post-run evaluation.")
    print("----------------------------------------------")

    # ---------------------------------------------------------
    # PASS 2: detection + localization + isolation
    # ---------------------------------------------------------
    results = []
    last_printed_event = None

    for snapshot in snapshots:
        result = process_snapshot(
            snapshot,
            detector,
            csv_file,
            operational,
            truth_df=df,
        )
        results.append(result)

        pmu = result["detected_pmu"]
        if pmu:
            event_key = (pmu, round(result["time_s"] / EVENT_MAX_GAP_S))
            if event_key != last_printed_event:
                print(
                    f"\n[FAULT EVENT] t={result['time_s']:.3f} s | "
                    f"Detected={pmu} | "
                    f"Actual={result['actual_faulty_pmu'] or 'None'} | "
                    f"J={result['initial_chi_square']:.4f} | "
                    f"corrected={result['corrected_chi_square']:.4f} | "
                    f"reduction={result['chi_square_reduction']:.3f}"
                )
                last_printed_event = event_key

    result_df = pd.DataFrame(results)
    result_df.to_csv(RESULTS_FILE, index=False)
    _print_metrics(result_df)
    print(f"Results CSV : {RESULTS_FILE}")

    return result_df


def main():
    run_automatic_scan(CSV_FILE, apply_sync_correction=False)


if __name__ == "__main__":
    main()