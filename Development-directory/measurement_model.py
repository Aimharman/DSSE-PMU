"""
===========================================================
measurement_model.py

Measurement Model

Implements

    z = h(x)

For Stage 3.2 we use the simplest possible PMU model.

State Vector

x =

[V1 θ1 V2 θ2 V3 θ3]

Measurement Vector

h(x) =

[V1 θ1 V2 θ2 V3 θ3]

Later this can be replaced by a full power-system model.
===========================================================
"""

import numpy as np

from network_model import NUM_BUSES


def measurement_model(x):
    """
    Generic measurement model.

    Input:
        x = [V1 θ1 V2 θ2 ... VN θN]

    Output:
        h(x)

    Currently:

        PMU Magnitude = Bus Voltage
        PMU Phase     = Bus Angle

    This will later be replaced by the network equations.
    """

    h = []

    for bus in range(NUM_BUSES):

        voltage = x[2 * bus]

        angle = x[2 * bus + 1]

        h.append(voltage)

        h.append(angle)

    return np.array(h)