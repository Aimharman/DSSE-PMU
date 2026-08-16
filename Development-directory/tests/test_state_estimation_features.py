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

    def test_faulty_pmu_detection_identifies_suspicious_pmu(self):
        detector = ChiSquareDetector()
        residual_history = np.array([
            [0.1, 0.0, 0.1, 0.0, 0.2, 0.0, 0.2, 0.0, 1.0, 0.2, 1.2, 0.3],
            [0.2, 0.0, 0.2, 0.0, 0.3, 0.0, 0.3, 0.0, 1.2, 0.3, 1.3, 0.2],
            [0.1, 0.0, 0.1, 0.0, 0.2, 0.0, 0.2, 0.0, 1.5, 0.2, 1.0, 0.1],
        ])

        result = detector.detect_faulty_pmu(
            residual_history,
            pmu_names=["PMU1", "PMU2", "PMU3"],
            window_size=3,
            threshold=0.5,
        )

        self.assertEqual(result[0]["pmu"], "PMU3")
        self.assertGreater(result[0]["score"], 0.5)

    def test_faulty_pmu_grouping_identifies_pmu_from_single_residual_vector(self):
        detector = ChiSquareDetector()
        residual = np.array([
            0.1, 0.05, 0.2, 0.1,
            0.2, 0.1, 0.3, 0.2,
            2.5, 3.2, 2.8, 2.1,
        ])
        W = np.diag(np.ones(len(residual)))
        measurement_names = [
            "PMU1 Voltage Magnitude",
            "PMU1 Voltage Phase",
            "PMU1 Current Magnitude",
            "PMU1 Current Phase",
            "PMU2 Voltage Magnitude",
            "PMU2 Voltage Phase",
            "PMU2 Current Magnitude",
            "PMU2 Current Phase",
            "PMU3 Voltage Magnitude",
            "PMU3 Voltage Phase",
            "PMU3 Current Magnitude",
            "PMU3 Current Phase",
        ]

        pmu_name, pmu_indices, score = detector.localize_faulty_pmu(
            residual,
            W,
            measurement_names,
        )

        self.assertEqual(pmu_name, "PMU3")
        self.assertEqual(pmu_indices, [8, 9, 10, 11])
        self.assertGreater(score, 0.0)

    def test_neural_controller_maps_faults_to_management_actions(self):
        from neural_active_fault_controller import NeuralActiveFaultController

        controller = NeuralActiveFaultController()

        self.assertEqual(controller.class_to_action["NORMAL"], "ACCEPT")
        self.assertEqual(controller.class_to_action["BAD_DATA"], "DOWN_WEIGHT")
        self.assertEqual(controller.class_to_action["SYNC_FAULT"], "PHASE_COMPENSATE")
        self.assertEqual(controller.class_to_action["CLOCK_DRIFT"], "TIMING_CORRECTION")
        self.assertEqual(controller.class_to_action["TRANSIENT_FAULT"], "ISOLATE")


if __name__ == "__main__":
    unittest.main()
