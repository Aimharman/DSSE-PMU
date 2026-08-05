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