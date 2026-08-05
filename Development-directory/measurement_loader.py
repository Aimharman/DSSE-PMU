"""
=========================================================
measurement_loader.py

Loads PMU measurements from PMU_Output.csv

=========================================================
"""

import pandas as pd
import numpy as np


class MeasurementLoader:

    def __init__(self, filename):

        self.filename = filename

    def load_latest_measurement(self):

        df = pd.read_csv(self.filename)

        row = df.iloc[-1]

        z = []

        ###################################################
        # PMU 1
        ###################################################

        z.append(row["PMU1 Voltage Magnitude"])
        z.append(row["PMU1 Voltage Phase"])

        z.append(row["PMU1 Current Magnitude"])
        z.append(row["PMU1 Current Phase"])

        ###################################################
        # PMU 2
        ###################################################

        z.append(row["PMU2 Voltage Magnitude"])
        z.append(row["PMU2 Voltage Phase"])

        z.append(row["PMU2 Current Magnitude"])
        z.append(row["PMU2 Current Phase"])

        ###################################################
        # PMU 3
        ###################################################

        z.append(row["PMU3 Voltage Magnitude"])
        z.append(row["PMU3 Voltage Phase"])

        z.append(row["PMU3 Current Magnitude"])
        z.append(row["PMU3 Current Phase"])

        return np.array(z)