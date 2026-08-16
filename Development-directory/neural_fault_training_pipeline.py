"""
Small neural training pipeline for the neural active fault-management controller.

This module keeps the implementation lightweight and fully runnable without
external ML packages. It generates a fault dataset from the baseline PMU CSV,
trains a small MLP using NumPy, and reports classification accuracy.

The goal is to provide a practical demonstration of the project progression:
    baseline PMU estimator -> feature extraction -> fault labels -> NN training -> action mapping.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


FAULT_LABELS = [
    "NORMAL",
    "BAD_DATA",
    "SYNC_FAULT",
    "CLOCK_DRIFT",
    "TRANSIENT_FAULT",
]


def resolve_csv_path(csv_path: str) -> str:
    """Use the Development directory CSV when present; otherwise fall back to repo root."""
    if os.path.exists(csv_path):
        return csv_path

    base_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(base_dir, ".."))
    fallback = os.path.join(repo_root, os.path.basename(csv_path))
    if os.path.exists(fallback):
        return fallback

    return csv_path


def load_reference_csv(csv_path: str) -> pd.DataFrame:
    """Load the project CSV and keep only valid measurement rows."""
    df = pd.read_csv(csv_path)
    valid_columns = []
    for pmu in range(1, 4):
        valid_columns.extend([
            f"PMU{pmu} Voltage Magnitude",
            f"PMU{pmu} Voltage Phase",
            f"PMU{pmu} Current Magnitude",
            f"PMU{pmu} Current Phase",
        ])
    valid = df[valid_columns].notna().all(axis=1)
    return df.loc[valid].reset_index(drop=True)


def get_feature_template(df: pd.DataFrame) -> np.ndarray:
    """Create a compact feature template from the healthy PMU data."""
    sample = df.iloc[0]
    features = []
    for pmu in range(1, 4):
        v_mag = float(sample[f"PMU{pmu} Voltage Magnitude"])
        v_phase = float(sample[f"PMU{pmu} Voltage Phase"])
        i_mag = float(sample[f"PMU{pmu} Current Magnitude"])
        i_phase = float(sample[f"PMU{pmu} Current Phase"])
        features.extend([
            v_mag,
            v_phase,
            i_mag,
            i_phase,
        ])
    return np.asarray(features, dtype=float)


def generate_synthetic_fault_dataset(
    csv_path: str,
    samples_per_class: int = 80,
    seed: int = 11,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a synthetic fault dataset using perturbations around healthy PMU data."""
    df = load_reference_csv(csv_path)
    base = get_feature_template(df)
    rng = np.random.default_rng(seed)

    X_list: List[np.ndarray] = []
    y_list: List[str] = []

    for fault_type in FAULT_LABELS:
        for _ in range(samples_per_class):
            noise = rng.normal(0.0, 0.08, size=base.shape[0])
            sample = base.copy() + noise

            if fault_type == "NORMAL":
                pass
            elif fault_type == "BAD_DATA":
                sample[2] += 0.6
                sample[6] += 0.9
                sample[10] += 0.7
            elif fault_type == "SYNC_FAULT":
                sample[1] += 12.0
                sample[5] += 15.0
                sample[9] += 10.0
            elif fault_type == "CLOCK_DRIFT":
                sample[3] += 9.0
                sample[7] += 10.0
                sample[11] += 8.0
            elif fault_type == "TRANSIENT_FAULT":
                sample[0] += 0.8
                sample[4] += 0.9
                sample[8] += 1.1
                sample[10] += 0.7
            else:
                pass

            X_list.append(sample)
            y_list.append(fault_type)

    X = np.vstack(X_list)
    y = np.asarray(y_list)
    return X, y


class SmallMLP:
    """A lightweight MLP trained with NumPy for presentation-ready demonstration."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0.0, 0.2, size=(input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.normal(0.0, 0.2, size=(hidden_dim, output_dim))
        self.b2 = np.zeros(output_dim)

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        shifted = x - np.max(x, axis=1, keepdims=True)
        exp_vals = np.exp(shifted)
        return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        z1 = X @ self.W1 + self.b1
        a1 = self.relu(z1)
        z2 = a1 @ self.W2 + self.b2
        probs = self.softmax(z2)
        return z1, a1, probs

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        _, _, probs = self.forward(X)
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 300, lr: float = 0.05):
        y_idx = np.asarray([FAULT_LABELS.index(label) for label in y], dtype=int)
        num_classes = len(FAULT_LABELS)
        y_onehot = np.eye(num_classes)[y_idx]

        for epoch in range(epochs):
            z1, a1, probs = self.forward(X)

            dZ2 = (probs - y_onehot) / X.shape[0]
            dW2 = a1.T @ dZ2
            db2 = np.sum(dZ2, axis=0)

            dA1 = dZ2 @ self.W2.T
            dZ1 = dA1 * (z1 > 0)
            dW1 = X.T @ dZ1
            db1 = np.sum(dZ1, axis=0)

            self.W2 -= lr * dW2
            self.b2 -= lr * db2
            self.W1 -= lr * dW1
            self.b1 -= lr * db1

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        pred = self.predict(X)
        true = np.asarray([FAULT_LABELS.index(label) for label in y], dtype=int)
        return float(np.mean(pred == true))


def train_and_evaluate(csv_path: str) -> Dict[str, float]:
    """Generate data, train a simple MLP, and report accuracy."""
    X, labels = generate_synthetic_fault_dataset(csv_path, samples_per_class=80, seed=12)

    label_to_idx = {label: idx for idx, label in enumerate(FAULT_LABELS)}
    y_idx = np.asarray([label_to_idx[label] for label in labels], dtype=int)

    rng = np.random.default_rng(5)
    perm = rng.permutation(len(X))
    X = X[perm]
    y_idx = y_idx[perm]

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y_idx[:split], y_idx[split:]

    model = SmallMLP(input_dim=X.shape[1], hidden_dim=16, output_dim=len(FAULT_LABELS), seed=7)
    model.train(X_train, [FAULT_LABELS[i] for i in y_train], epochs=400, lr=0.06)

    train_acc = model.accuracy(X_train, [FAULT_LABELS[i] for i in y_train])
    test_acc = model.accuracy(X_test, [FAULT_LABELS[i] for i in y_test])

    return {
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "n_classes": len(FAULT_LABELS),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = resolve_csv_path(os.path.join(base_dir, "PMU_Output.csv"))
    result = train_and_evaluate(csv_path)

    print("\n==============================================")
    print(" Neural Fault Training Pipeline")
    print("==============================================")
    print(f"Train samples : {result['n_train']}")
    print(f"Test samples  : {result['n_test']}")
    print(f"Classes       : {result['n_classes']}")
    print(f"Train acc     : {result['train_accuracy']:.4f}")
    print(f"Test acc      : {result['test_accuracy']:.4f}")
    print("==============================================")
    print("This is a lightweight proof-of-concept neural controller for the project.")
    print("The next step is to replace the synthetic generation with simulator-driven windows.")
    print("==============================================")


if __name__ == "__main__":
    main()
