"""
===========================================================
chi_square.py

Chi-Square Bad Data Detection

Stage 3.8

Computes

        J = rᵀWr

and compares it with the theoretical
Chi-square threshold.

===========================================================
"""

import numpy as np

from scipy.stats import chi2


class ChiSquareDetector:

    def __init__(self, confidence=0.95):

        self.confidence = confidence

    ########################################################
    # Chi-Square Statistic
    ########################################################

    def compute_statistic(self, residual, W):
        """
        Computes

            J = rᵀWr
        """

        J = residual.T @ W @ residual

        return float(J)

    ########################################################
    # Degrees of Freedom
    ########################################################

    def degrees_of_freedom(self,
                           num_measurements,
                           num_states):

        """
        ν = m − n
        """

        return num_measurements - num_states

    ########################################################
    # Chi-Square Threshold
    ########################################################

    def threshold(self, dof):

        """
        Computes

            χ²(confidence, dof)
        """

        return chi2.ppf(self.confidence, dof)

    ########################################################
    # Bad-Data Localization
    ########################################################

    def localize_bad_data(self, residual, W, measurement_names=None):
        """
        Identify the most suspicious measurement using a simplified
        residual-based heuristic.

        For a diagonal weight matrix, the score is

            score_i = |r_i| * sqrt(W_ii)

        This is a simplified normalized-residual heuristic and is not
        the full statistically rigorous normalized residual test.
        Larger scores indicate measurements that are less consistent
        with the current estimate.
        """

        residual = np.asarray(residual, dtype=float).reshape(-1)

        if W.ndim == 2:
            diag_weights = np.diag(W)
        else:
            diag_weights = np.asarray(W, dtype=float).reshape(-1)

        diag_weights = np.maximum(np.abs(diag_weights), 1e-12)
        scores = np.abs(residual) * np.sqrt(diag_weights)

        index = int(np.argmax(scores))
        label = (
            measurement_names[index]
            if measurement_names is not None and len(measurement_names) > index
            else f"measurement_{index + 1}"
        )

        return index, label, float(scores[index])

    ########################################################
    # PMU-Level Faulty Detection
    ########################################################

    def detect_faulty_pmu(self, residual_history, pmu_names=None, window_size=3, threshold=0.5):
        """
        Detect a persistent faulty PMU by aggregating residual energy
        over a rolling window of consecutive timestamps.

        residual_history is expected to be shaped as
            (num_windows, num_measurements)
        or (num_timestamps, num_measurements), where each measurement
        group belongs to a PMU.
        """

        residual_history = np.asarray(residual_history, dtype=float)
        if residual_history.ndim == 1:
            residual_history = residual_history.reshape(1, -1)

        if residual_history.shape[1] % 4 != 0:
            raise ValueError("Residual history length must be a multiple of 4 for PMU groups.")

        if pmu_names is None:
            pmu_names = [f"PMU{idx + 1}" for idx in range(residual_history.shape[1] // 4)]

        n_measurements = residual_history.shape[1]
        n_pmus = n_measurements // 4

        scores = []
        for pmu_idx in range(n_pmus):
            pmu_slice = slice(4 * pmu_idx, 4 * pmu_idx + 4)
            pmu_residuals = residual_history[:, pmu_slice]
            pmu_energy = np.linalg.norm(pmu_residuals, axis=1)

            if pmu_residuals.shape[0] >= window_size:
                window = np.array([
                    np.mean(pmu_energy[max(0, t - window_size + 1):t + 1])
                    for t in range(pmu_residuals.shape[0])
                ])
            else:
                window = np.full(pmu_residuals.shape[0], np.mean(pmu_energy))

            score = float(np.mean(window))
            scores.append({
                "pmu": pmu_names[pmu_idx],
                "score": score,
                "window_score": float(np.max(window)),
                "mean_energy": float(np.mean(pmu_energy)),
            })

        ranked = sorted(scores, key=lambda x: x["score"], reverse=True)
        suspicious = [entry for entry in ranked if entry["score"] >= threshold]

        if not suspicious:
            return []

        return suspicious

    ########################################################
    # Detection
    ########################################################

    def detect(self,
               residual,
               W,
               num_measurements,
               num_states):

        J = self.compute_statistic(residual, W)

        dof = self.degrees_of_freedom(
            num_measurements,
            num_states
        )

        threshold = self.threshold(dof)

        print("\n==============================================")
        print(" Chi-Square Bad Data Detection")
        print("==============================================")

        print(f"\nDegrees of Freedom : {dof}")

        print(f"Confidence Level   : {self.confidence * 100:.1f}%")

        print(f"\nChi-Square Statistic (J)")
        print(J)

        print(f"\nCritical Threshold")
        print(threshold)

        if J <= threshold:

            print("\nResult")
            print("✓ Measurement set PASSED")
            print("No bad data detected.")

            bad_data = False

        else:

            print("\nResult")
            print("✗ Measurement set FAILED")
            print("Bad data detected.")

            bad_data = True

        print("==============================================")

        return bad_data, J, threshold