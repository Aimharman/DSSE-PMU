"""
===========================================================
load_profiles.py

Dynamic Distribution Load Profiles

Provides realistic load variations for the PMU simulator.

Load values are returned as multipliers.

1.0 = Nominal load

These profiles can later be replaced by
actual feeder load data.
===========================================================
"""

import numpy as np


########################################################
# Residential Load
########################################################

def residential_load(t):

    base = 0.70

    variation = 0.20 * np.sin(2 * np.pi * 0.02 * t)

    return base + variation


########################################################
# Commercial Load
########################################################

def commercial_load(t):

    base = 0.90

    variation = 0.10 * np.sin(
        2 * np.pi * 0.015 * t + np.pi / 4
    )

    return base + variation


########################################################
# Industrial Load
########################################################

def industrial_load(t):

    base = 1.10

    variation = 0.05 * np.sin(
        2 * np.pi * 0.01 * t
    )

    return base + variation


########################################################
# Load Event Scheduler
########################################################

def apply_events(t, loads):
    """
    Apply scheduled grid events.

    loads = [L1, L2, L3]
    """

    L1, L2, L3 = loads

    ####################################################
    # Motor Starting
    ####################################################

    if 2.0 <= t < 2.5:

        L3 *= 1.8

    ####################################################
    # Capacitor Switching
    ####################################################

    if 4.0 <= t < 5.0:

        L2 *= 0.90

    ####################################################
    # EV Charging
    ####################################################

    if 6.0 <= t < 8.0:

        L1 *= 1.35

    ####################################################
    # Load Shedding
    ####################################################

    if 8.0 <= t < 9.0:

        L3 *= 0.50

    return L1, L2, L3


########################################################
# Main Interface
########################################################

def get_loads(t):

    L1 = residential_load(t)

    L2 = commercial_load(t)

    L3 = industrial_load(t)

    return apply_events(
        t,
        (L1, L2, L3)
    )