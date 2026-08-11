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
        Identify the most suspicious measurement using a simple
        normalized-residual score.

        For a diagonal weight matrix, the score is

            score_i = |r_i| * sqrt(W_ii)

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