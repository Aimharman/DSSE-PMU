"""
===========================================================
measurement_model.py

Distribution System State Estimation
Stage 3.2

Measurement Model

Computes

        h(x)

from the state vector

        x = [V1 θ1 V2 θ2 ... VN θN]

using

        I = Ybus · V

The predicted measurement vector is

h(x) =

[
Vmag1
Vphase1
Imag1
Iphase1

Vmag2
Vphase2
Imag2
Iphase2

...

VmagN
VphaseN
ImagN
IphaseN
]

===========================================================
"""

import numpy as np

from network_model import NUM_BUSES, YBUS


def measurement_model(x):

    ########################################################
    # Convert State Vector to Complex Bus Voltages
    ########################################################

    V = np.zeros(NUM_BUSES, dtype=complex)

    for bus in range(NUM_BUSES):

        magnitude = x[2 * bus]

        angle_deg = x[2 * bus + 1]

        angle_rad = np.deg2rad(angle_deg)

        V[bus] = magnitude * np.exp(1j * angle_rad)

    ########################################################
    # Compute Bus Currents
    #
    # I = Ybus · V
    ########################################################

    I = YBUS @ V

    ########################################################
    # Construct Predicted Measurement Vector
    ########################################################

    h = []

    for bus in range(NUM_BUSES):

        ####################################################
        # Voltage
        ####################################################

        h.append(np.abs(V[bus]))

        h.append(np.degrees(np.angle(V[bus])))

        ####################################################
        # Current
        ####################################################

        h.append(np.abs(I[bus]))

        h.append(np.degrees(np.angle(I[bus])))

    return np.array(h)