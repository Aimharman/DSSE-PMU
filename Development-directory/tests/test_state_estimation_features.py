import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from chi_square import ChiSquareDetector
from state_estimator import StateEstimator


class TestStateEstimationFeatures(unittest.TestCase):

    def test_localize_bad_data_returns_most_suspicious_measurement(self):
        detector = ChiSquareDetector()
        residual = np.array([0.1, 0.2, 3.0, -0.1])
        W = np.diag([1.0, 1.0, 1.0, 1.0])

        index, label, score = detector.localize_bad_data(
            residual,
            W,
            measurement_names=["m1", "m2", "m3", "m4"],
        )

        self.assertEqual(index, 2)
        self.assertEqual(label, "m3")
        self.assertGreater(score, 2.0)

    def test_sync_correction_changes_measurement_vector(self):
        csv_path = os.path.join(os.path.dirname(__file__), "..", "PMU_Output.csv")
        estimator = StateEstimator(csv_path)

        estimator.build_measurement_vector(apply_sync_correction=False)
        raw_vector = estimator.z.copy()

        estimator.build_measurement_vector(apply_sync_correction=True)
        corrected_vector = estimator.z.copy()

        self.assertFalse(np.allclose(raw_vector, corrected_vector))


if __name__ == "__main__":
    unittest.main()
