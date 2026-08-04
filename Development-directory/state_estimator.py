"""
===========================================================
State Estimator
Stage 3.1

Reads PMU measurements and constructs

1. Measurement Vector (z)
2. Initial State Vector (x)

The actual WLS solver will be added in Stage 3.2.
===========================================================
"""

import numpy as np
import pandas as pd
from measurement_model import measurement_model
from network_model import NUM_BUSES



class StateEstimator:

    def __init__(self, csv_file):

        self.csv_file = csv_file

        self.df = pd.read_csv(csv_file)

        self.z = None

        self.x = None

        self.h = None

        self.residual = None


    ########################################################
    # Measurement Vector
    ########################################################

    def build_measurement_vector(self):

        """
        Measurement vector

        z =

        [
        PMU1 Magnitude
        PMU1 Phase
        PMU2 Magnitude
        PMU2 Phase
        PMU3 Magnitude
        PMU3 Phase
        ]
        """

        latest = self.df.iloc[-1]

        self.z = np.array([

            latest["DFT1 Magnitude"],
            latest["DFT1 Phase (deg)"],

            latest["DFT2 Magnitude"],
            latest["DFT2 Phase (deg)"],

            latest["DFT3 Magnitude"],
            latest["DFT3 Phase (deg)"]

        ])

        return self.z


    ########################################################
    # Initial State Vector
    ########################################################

    def initialize_state(self):

        """
        Initial Guess

        x =
        [
        V1 θ1
        V2 θ2
        V3 θ3
        ]
        """

        # self.x = np.array([

        #     1.0,
        #     0.0,

        #     1.0,
        #     0.0,

        #     1.0,
        #     0.0

        # ])

        self.x = np.zeros(2 * NUM_BUSES)

        for bus in range(NUM_BUSES):
            self.x[2 * bus] = 1.0      # Voltage magnitude
            self.x[2 * bus + 1] = 0.0  # Voltage angle


        return self.x

    ########################################################
    # Predicted Measurement
    ########################################################

    def predict_measurements(self):

        self.h = measurement_model(self.x)

        return self.h

    ########################################################
    # Residual Calculation
    ########################################################

    def compute_residual(self):

        """
        Residual

        r = z - h(x)
        """

        self.residual = self.z - self.h

        return self.residual

    ########################################################
    # Display
    ########################################################

    def summary(self):

        print("\n====================================")
        print(" State Estimator")
        print("====================================")

        print("\nMeasurement Vector (z)")

        print(self.z)

        print("\nInitial State Vector (x0)")

        print(self.x)

        print("\nPredicted Measurement h(x)\n")

        print(self.h)

        print("\nResidual r = z - h(x)\n")

        print(self.residual)

        print("====================================")

    ########################################################
    # Main Estimation Flow
    ########################################################

    def run(self):

        print("\nBuilding measurement vector...")
        self.build_measurement_vector()

        print("Initializing state vector...")
        self.initialize_state()

        print("Predicting measurements...")
        self.predict_measurements()

        print("Computing residual...")
        self.compute_residual()

        self.summary()