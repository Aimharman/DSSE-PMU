"""
Neural active fault management controller demo.

This is intentionally lightweight and kept inside the Development directory.
It demonstrates how the baseline PMU estimator can feed a neural-style
controller layer that classifies the fault and maps it to a management action.

The current design is a practical prototype: synthetic fault examples are
built from the baseline measurement features, then a controller policy is
applied to each fault type.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np

from neural_active_fault_controller import NeuralActiveFaultController
from state_estimator import StateEstimator


def resolve_csv_path(csv_path: str) -> str:
    """Allow the demo to resolve the PMU CSV from the repo root when needed."""
    if os.path.exists(csv_path):
        return csv_path

    base_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(base_dir, ".."))
    fallback = os.path.join(repo_root, os.path.basename(csv_path))
    if os.path.exists(fallback):
        return fallback

    return csv_path


FAULT_CLASSES = [
    "NORMAL",
    "BAD_DATA",
    "SYNC_FAULT",
    "CLOCK_DRIFT",
    "TRANSIENT_FAULT",
]


def _healthy_feature_vector(csv_file: str, sample_index: int = 0) -> np.ndarray:
    """Collect a compact PMU feature vector from the healthy baseline."""
    estimator = StateEstimator(csv_file, sample_index=sample_index)
    estimator.run(apply_sync_correction=False, verbose=False)

    row = estimator.selected_row
    values = []

    for pmu in range(1, 4):
        v_mag = float(row[f"PMU{pmu} Voltage Magnitude"])
        v_phase = float(row[f"PMU{pmu} Voltage Phase"])
        i_mag = float(row[f"PMU{pmu} Current Magnitude"])
        i_phase = float(row[f"PMU{pmu} Current Phase"])

        values.extend([
            abs(v_mag - 1.0),
            abs(v_phase),
            abs(i_mag - 1.0),
            abs(i_phase),
        ])

    return np.asarray(values, dtype=float)


def build_synthetic_fault_dataset(csv_file: str, samples_per_class: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """Create a compact artificial dataset for fault-class demonstrations."""
    base = _healthy_feature_vector(csv_file, sample_index=0)
    rng = np.random.default_rng(7)

    X_list: List[np.ndarray] = []
    y_list: List[str] = []

    for fault_type in FAULT_CLASSES:
        for _ in range(samples_per_class):
            sample = base.copy() + rng.normal(0.0, 0.05, size=base.shape[0])

            if fault_type == "NORMAL":
                label = "NORMAL"
            elif fault_type == "BAD_DATA":
                sample[2] += 0.7
                sample[6] += 0.8
                label = "BAD_DATA"
            elif fault_type == "SYNC_FAULT":
                sample[5] += 20.0
                sample[9] += 22.0
                label = "SYNC_FAULT"
            elif fault_type == "CLOCK_DRIFT":
                sample[3] += 9.0
                sample[7] += 12.0
                label = "CLOCK_DRIFT"
            elif fault_type == "TRANSIENT_FAULT":
                sample[1] += 1.1
                sample[10] += 1.3
                label = "TRANSIENT_FAULT"
            else:
                label = "NORMAL"

            X_list.append(sample)
            y_list.append(label)

    X = np.vstack(X_list)
    y = np.asarray(y_list)
    return X, y


def preview_controller_policy() -> List[Dict[str, str]]:
    """Return the controller decisions for each supported fault class."""
    controller = NeuralActiveFaultController()
    rows = []
    for fault_type in FAULT_CLASSES:
        decision = controller.label_fault(fault_type, pmu_id=3)
        rows.append({
            "fault_type": fault_type,
            "action": decision["action"],
            "weight": str(decision["weight"]),
            "decision": decision["decision"],
        })
    return rows


def run_demo(csv_file: str = None) -> Dict[str, object]:
    """Run the controller design demo and return a lightweight summary."""
    if csv_file is None:
        csv_file = resolve_csv_path(os.path.join(os.path.dirname(__file__), "PMU_Output.csv"))

    X, y = build_synthetic_fault_dataset(csv_file, samples_per_class=12)
    controller = NeuralActiveFaultController()

    summary = {}
    for fault_type in FAULT_CLASSES:
        subset = y == fault_type
        summary[fault_type] = int(np.sum(subset))

    policy_rows = preview_controller_policy()

    print("\n==============================================")
    print(" Neural Active Fault Management Controller")
    print("==============================================")
    print("Fault dataset summary")
    for fault_type in FAULT_CLASSES:
        print(f"  {fault_type:15s} : {summary[fault_type]} samples")

    print("\nController action mapping")
    for row in policy_rows:
        print(f"  {row['fault_type']:15s} -> {row['action']:20s} | weight={row['weight']} | {row['decision']}")

    print("\nManagement interpretation")
    for fault_type in FAULT_CLASSES:
        print(f"  {fault_type:15s} -> {controller.explain_action(fault_type, 3)}")

    print("==============================================")

    return {
        "dataset_shape": X.shape,
        "fault_summary": summary,
        "controller_policy": policy_rows,
    }


if __name__ == "__main__":
    run_demo()
