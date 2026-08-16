"""
Chi-square validation for the Cyber Resilient PDC project.

Two thresholds are retained:

1. Theoretical chi-square threshold from the ideal statistical model.
2. Robust empirical operating threshold derived from the observed
   healthy/noise residual population.

The empirical threshold is NOT based on simulator ground-truth fault
labels. It is calculated from the chi-square statistics themselves using
median + 6 * 1.4826*MAD, with the theoretical threshold as a floor.
"""

import re
import numpy as np
from scipy.stats import chi2


class ChiSquareDetector:

    def __init__(
        self,
        confidence=0.95,
        pmu_measurement_count=4,
        pmu_share_threshold=0.55,
    ):
        self.confidence = confidence
        self.pmu_measurement_count = pmu_measurement_count
        self.pmu_share_threshold = pmu_share_threshold
        self.empirical_threshold = None
        self.baseline_median = None
        self.baseline_mad = None

    # ---------------------------------------------------------
    # Statistic
    # ---------------------------------------------------------

    def compute_statistic(self, residual, W):
        residual = np.asarray(residual, dtype=float).reshape(-1)

        if np.ndim(W) == 2:
            diagonal = np.diag(W)
        else:
            diagonal = np.asarray(W, dtype=float).reshape(-1)

        return float(np.sum((residual ** 2) * diagonal))

    # ---------------------------------------------------------
    # Degrees of freedom / theoretical threshold
    # ---------------------------------------------------------

    def degrees_of_freedom(self, num_measurements, num_states):
        return int(num_measurements - num_states)

    def threshold(self, dof):
        return float(chi2.ppf(self.confidence, dof))

    # ---------------------------------------------------------
    # Robust empirical threshold
    # ---------------------------------------------------------

    def calibrate(self, statistics, dof):
        """
        Establish the operating threshold from the complete scan.

        Fault values are gross outliers in this project, so the median and
        MAD are robust to their presence. Ground-truth PMU labels are not
        used here.
        """
        values = np.asarray(statistics, dtype=float)
        values = values[np.isfinite(values)]

        if len(values) == 0:
            raise ValueError("No valid chi-square statistics for calibration.")

        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        robust_sigma = 1.4826 * mad

        robust_limit = median + 6.0 * robust_sigma
        theoretical = self.threshold(dof)

        self.baseline_median = median
        self.baseline_mad = mad
        self.empirical_threshold = float(max(theoretical, robust_limit))

        return self.empirical_threshold

    def operating_threshold(self, dof):
        if self.empirical_threshold is not None:
            return float(self.empirical_threshold)
        return self.threshold(dof)

    # ---------------------------------------------------------
    # Detection
    # ---------------------------------------------------------

    def detect(
        self,
        residual,
        W,
        num_measurements,
        num_states,
        threshold_override=None,
        verbose=True,
    ):
        J = self.compute_statistic(residual, W)
        dof = self.degrees_of_freedom(num_measurements, num_states)
        theoretical_threshold = self.threshold(dof)
        threshold = (
            float(threshold_override)
            if threshold_override is not None
            else self.operating_threshold(dof)
        )

        bad_data = bool(J > threshold)

        if verbose:
            print("\n==============================================")
            print(" Chi-Square Bad Data Detection")
            print("==============================================")
            print(f"\nDegrees of Freedom       : {dof}")
            print(f"Confidence Level         : {self.confidence * 100:.1f}%")
            print(f"Theoretical Threshold    : {theoretical_threshold:.6f}")
            print(f"Operating Threshold      : {threshold:.6f}")
            print(f"\nChi-Square Statistic (J) : {J:.6f}")
            print("\nResult")
            if bad_data:
                print("✗ Measurement set FAILED")
                print("Bad data suspected.")
            else:
                print("✓ Measurement set PASSED")
                print("No bad data detected.")
            print("==============================================")

        return bad_data, J, threshold

    # ---------------------------------------------------------
    # PMU-level rolling-window detection
    # ---------------------------------------------------------

    def detect_faulty_pmu(self, residual_history, pmu_names=None, window_size=3, threshold=0.5):
        """
        Identify a persistently faulty PMU by accumulating PMU residual energy
        over a rolling window.
        """

        residual_history = np.asarray(residual_history, dtype=float)
        if residual_history.ndim == 1:
            residual_history = residual_history.reshape(1, -1)

        if residual_history.shape[1] % self.pmu_measurement_count != 0:
            raise ValueError("Residual history length must be a multiple of PMU measurements.")

        if pmu_names is None:
            pmu_names = [f"PMU{idx + 1}" for idx in range(residual_history.shape[1] // self.pmu_measurement_count)]

        num_pmus = residual_history.shape[1] // self.pmu_measurement_count
        results = []

        for pmu_index in range(num_pmus):
            start = pmu_index * self.pmu_measurement_count
            stop = start + self.pmu_measurement_count
            pmu_residuals = residual_history[:, start:stop]
            pmu_energy = np.linalg.norm(pmu_residuals, axis=1)

            if pmu_residuals.shape[0] < window_size:
                rolling_score = float(np.mean(pmu_energy))
            else:
                rolling_scores = []
                for pos in range(len(pmu_energy)):
                    window = pmu_energy[max(0, pos - window_size + 1):pos + 1]
                    rolling_scores.append(float(np.mean(window)))
                rolling_score = float(np.mean(rolling_scores))

            results.append({
                "pmu": pmu_names[pmu_index],
                "score": rolling_score,
                "window_score": float(np.max(pmu_energy)),
                "mean_energy": float(np.mean(pmu_energy)),
            })

        ranked = sorted(results, key=lambda item: item["score"], reverse=True)
        return [item for item in ranked if item["score"] >= threshold]

    # ---------------------------------------------------------
    # PMU contribution / localization
    # ---------------------------------------------------------

    def pmu_contributions(self, residual, W, measurement_names=None):
        residual = np.asarray(residual, dtype=float).reshape(-1)

        if np.ndim(W) == 2:
            diagonal = np.diag(W)
        else:
            diagonal = np.asarray(W, dtype=float).reshape(-1)

        diagonal = np.maximum(np.abs(diagonal), 1e-12)

        if measurement_names is None:
            measurement_names = [f"measurement_{i + 1}" for i in range(len(residual))]

        totals = {}
        indices = {}

        for idx, name in enumerate(measurement_names):
            match = re.search(r"PMU\s*\d+", str(name), flags=re.IGNORECASE)
            pmu = match.group(0).upper().replace(" ", "") if match else str(name).split()[0]
            totals.setdefault(pmu, 0.0)
            indices.setdefault(pmu, [])
            totals[pmu] += float(residual[idx] ** 2 * diagonal[idx])
            indices[pmu].append(idx)

        J = float(sum(totals.values()))
        ranked = []
        for pmu, energy in totals.items():
            ranked.append({
                "pmu": pmu,
                "energy": float(energy),
                "share": float(energy / J) if J > 0 else 0.0,
                "indices": indices[pmu],
            })

        ranked.sort(key=lambda item: item["energy"], reverse=True)
        return ranked

    def screen_fault_candidate(self, residual, W, global_J, global_threshold, measurement_names=None):
        pmu_data = self.pmu_contributions(residual, W, measurement_names)
        if not pmu_data:
            return {"candidate": False, "pmu": "", "energy": 0.0,
                    "share": 0.0, "threshold": 0.0, "pmu_data": []}

        top = pmu_data[0]
        pmu_threshold = float(chi2.ppf(self.confidence, self.pmu_measurement_count))
        candidate = (
            global_J > global_threshold
            and top["energy"] > pmu_threshold
            and top["share"] >= self.pmu_share_threshold
        )
        return {
            "candidate": bool(candidate),
            "pmu": top["pmu"],
            "energy": float(top["energy"]),
            "share": float(top["share"]),
            "threshold": pmu_threshold,
            "pmu_data": pmu_data,
        }

    def localize_faulty_pmu(self, residual, W, measurement_names=None):
        data = self.pmu_contributions(residual, W, measurement_names)
        if not data:
            return "", [], 0.0
        top = data[0]
        return top["pmu"], top["indices"], top["energy"]

    # Compatibility helpers
    def localize_bad_data(self, residual, W, measurement_names=None):
        residual = np.asarray(residual, dtype=float).reshape(-1)
        diagonal = np.diag(W) if np.ndim(W) == 2 else np.asarray(W).reshape(-1)
        diagonal = np.maximum(np.abs(diagonal), 1e-12)
        scores = np.abs(residual) * np.sqrt(diagonal)
        index = int(np.argmax(scores))
        label = (measurement_names[index] if measurement_names is not None and len(measurement_names) > index
                 else f"measurement_{index + 1}")
        return index, label, float(scores[index])