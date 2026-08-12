###########################################################################
# PMU Simulator - CSV-First Refactored Version
#
# Flow:
#   1. Generate the complete PMU simulation data
#   2. Compute the one-cycle DFT values
#   3. Write the COMPLETE dataset to PMU_Output.csv
#   4. Read the completed CSV back from disk
#   5. Generate the waveform / DFT plots from the CSV
#
# There is NO live waveform plotting during simulation.
###########################################################################

import sys
from collections import deque

import numpy as np
import pandas as pd
import pyqtgraph as pg

from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QScrollBar
from PyQt6.QtCore import Qt, QTimer

from load_profiles import get_loads
from network_model import YBUS


###########################################################################
# DEBUG / LOGGING CONFIGURATION
###########################################################################

PRINT_PACKET_LOSS = False
PRINT_BAD_DATA = False
PRINT_SYNC_OFFSETS = True


###########################################################################
# USER CONFIGURATION
###########################################################################

FREQUENCY = 50                 # Hz
ODR = 1000                     # Samples/sec
SIMULATION_TIME = 10.0         # seconds
DISPLAY_WINDOW = 0.10          # seconds displayed on screen
PLOT_UPDATE_SAMPLES = 160       # redraw plot every 160 samples

CURRENT_AMPLITUDE = 10         # Peak Current (A)
VOLTAGE_AMPLITUDE = 325        # Peak Voltage (230 Vrms)
LINE_RESISTANCE = 0.30         # Ohm


###########################################################################
# MEASUREMENT CHALLENGE CONFIGURATION
###########################################################################

ENABLE_SYNC_ERROR = True
ENABLE_MEASUREMENT_NOISE = True
ENABLE_CLOCK_DRIFT = True
ENABLE_PACKET_LOSS = False
ENABLE_BAD_DATA = True


###########################################################################
# FIXED PMU SYNCHRONIZATION OFFSETS
###########################################################################

SYNC_STD_DEG = 1.0

PMU1_SYNC_OFFSET = np.random.normal(0, SYNC_STD_DEG)
PMU2_SYNC_OFFSET = np.random.normal(0, SYNC_STD_DEG)
PMU3_SYNC_OFFSET = np.random.normal(0, SYNC_STD_DEG)

print("\nPMU Synchronization Offsets")
print("----------------------------")

if PRINT_SYNC_OFFSETS:
    print(f"PMU1 : {PMU1_SYNC_OFFSET:.2f}°")
    print(f"PMU2 : {PMU2_SYNC_OFFSET:.2f}°")
    print(f"PMU3 : {PMU3_SYNC_OFFSET:.2f}°")


###########################################################################
# BAD DATA CONFIGURATION
###########################################################################

BAD_PMU = 1  # Periodic mode selects PMUs deterministically in round-robin order

MAG_NOISE_STD = 0.005
PHASE_NOISE_STD = 0.2

CLOCK_DRIFT_RATE = 0.02
PACKET_LOSS_PROB = 0.02

# Injection mode:
#   "periodic"
#   "random"
#   "faulty_pmu"

BAD_DATA_MODE = "periodic"  # Options: "periodic", "random", "faulty_pmu"

# Periodic / random mode
BAD_DATA_INTERVAL = 500          # Start a new periodic fault every 500 samples = 0.5 s
PERIODIC_FAULT_DURATION = 50      # Keep each fault active for 50 samples = 50 ms
BAD_DATA_PROB = 0.01              # Used only by random mode
BAD_PHASE_ERROR = 20.0
BAD_MAG_SCALE = 1.20

# Faulty PMU mode
FAULTY_PMU = 1
FAULT_START_TIME = 2.0
# main.py uses the latest CSV row, so keep the fault active through the final sample.
FAULT_END_TIME = SIMULATION_TIME
FAULT_PHASE_ERROR = 20.0
FAULT_MAG_SCALE = 1.20


###########################################################################
# DERIVED PARAMETERS
###########################################################################

OMEGA = 2 * np.pi * FREQUENCY
DT = 1.0 / ODR
TOTAL_SAMPLES = int(SIMULATION_TIME * ODR)
N = int(ODR / FREQUENCY)       # One-cycle DFT window

###########################################################################
# FAULT CONFIGURATION VALIDATION
###########################################################################

def validate_fault_configuration():
    """Validate the faulty-PMU experiment before starting simulation."""
    if not ENABLE_BAD_DATA or BAD_DATA_MODE.lower() != "faulty_pmu":
        return

    if not 1 <= FAULTY_PMU <= 3:
        raise ValueError("FAULTY_PMU must be 1, 2, or 3.")

    if FAULT_END_TIME <= FAULT_START_TIME:
        raise ValueError("FAULT_END_TIME must be greater than FAULT_START_TIME.")

    final_sample_time = (TOTAL_SAMPLES - 1) * DT

    if final_sample_time < FAULT_START_TIME:
        raise ValueError(
            "Fault starts after the simulation ends. "
            "Increase SIMULATION_TIME or move FAULT_START_TIME."
        )

    if final_sample_time >= FAULT_END_TIME:
        print("\nWARNING: Final CSV row is outside the faulty-PMU interval.")
        print(f"Final sample time : {final_sample_time:.6f} s")
        print(f"Fault interval    : {FAULT_START_TIME:.6f} "
              f"to {FAULT_END_TIME:.6f} s")



###########################################################################
# CSV FORMAT
###########################################################################

COLUMNS = [
    "Time (s)",

    "Voltage 1 (V)",
    "Current 1 (A)",
    "Voltage 2 (V)",
    "Current 2 (A)",
    "Voltage 3 (V)",
    "Current 3 (A)",

    "Peak (A)",
    "Signal Angle (deg)",
    "Delta t (s)",

    "PMU1 Voltage DFT Real",
    "PMU1 Voltage DFT Imag",
    "PMU1 Voltage Magnitude",
    "PMU1 Voltage Phase",
    "PMU1 Voltage DFT RMS",
    "PMU1 Current DFT Real",
    "PMU1 Current DFT Imag",
    "PMU1 Current Magnitude",
    "PMU1 Current Phase",
    "PMU1 Current DFT RMS",

    "PMU1 Sync Offset",
    "PMU1 Mag Noise",
    "PMU1 Phase Noise",
    "PMU1 Clock Drift",
    "PMU1 Packet Loss",
    "PMU1 Bad Data",

    "PMU2 Voltage DFT Real",
    "PMU2 Voltage DFT Imag",
    "PMU2 Voltage Magnitude",
    "PMU2 Voltage Phase",
    "PMU2 Voltage DFT RMS",
    "PMU2 Current DFT Real",
    "PMU2 Current DFT Imag",
    "PMU2 Current Magnitude",
    "PMU2 Current Phase",
    "PMU2 Current DFT RMS",

    "PMU2 Sync Offset",
    "PMU2 Mag Noise",
    "PMU2 Phase Noise",
    "PMU2 Clock Drift",
    "PMU2 Packet Loss",
    "PMU2 Bad Data",

    "PMU3 Voltage DFT Real",
    "PMU3 Voltage DFT Imag",
    "PMU3 Voltage Magnitude",
    "PMU3 Voltage Phase",
    "PMU3 Voltage DFT RMS",
    "PMU3 Current DFT Real",
    "PMU3 Current DFT Imag",
    "PMU3 Current Magnitude",
    "PMU3 Current Phase",
    "PMU3 Current DFT RMS",

    "PMU3 Sync Offset",
    "PMU3 Mag Noise",
    "PMU3 Phase Noise",
    "PMU3 Clock Drift",
    "PMU3 Packet Loss",
    "PMU3 Bad Data",
]


###########################################################################
# DFT
###########################################################################

def compute_dft(buffer):
    """
    Compute the fundamental-frequency DFT for one complete cycle.

    Returns:
        real, imag, magnitude, phase_deg, rms
    """
    x = np.asarray(buffer)

    X = np.sum(
        x * np.exp(-1j * 2 * np.pi * np.arange(N) / N)
    )

    real = np.real(X)
    imag = np.imag(X)

    # Convert DFT magnitude to peak amplitude.
    magnitude = (2 / N) * np.abs(X)

    phase = np.degrees(np.angle(X))

    rms = magnitude / np.sqrt(2)

    return real, imag, magnitude, phase, rms


###########################################################################
# MEASUREMENT CHALLENGES
###########################################################################

def apply_measurement_challenges(
    mag,
    phase,
    t,
    sync_offset,
    sample_index,
    pmu_id,
):
    measured_mag = mag
    measured_phase = phase

    metadata = {
        "sync_error": 0.0,
        "mag_noise": 0.0,
        "phase_noise": 0.0,
        "clock_drift": 0.0,
        "packet_loss": False,
        "bad_data": False,
    }

    # ---------------------------------------------------------------
    # 1. Synchronization error
    # ---------------------------------------------------------------

    if ENABLE_SYNC_ERROR:
        measured_phase += sync_offset
        metadata["sync_error"] = sync_offset

    # ---------------------------------------------------------------
    # 2. Measurement noise
    # ---------------------------------------------------------------

    if ENABLE_MEASUREMENT_NOISE:
        mag_noise = np.random.normal(0, MAG_NOISE_STD)
        phase_noise = np.random.normal(0, PHASE_NOISE_STD)

        measured_mag += mag_noise
        measured_phase += phase_noise

        metadata["mag_noise"] = mag_noise
        metadata["phase_noise"] = phase_noise

    # ---------------------------------------------------------------
    # 3. Clock drift
    # ---------------------------------------------------------------

    if ENABLE_CLOCK_DRIFT:
        drift = CLOCK_DRIFT_RATE * t
        measured_phase += drift
        metadata["clock_drift"] = drift

    # ---------------------------------------------------------------
    # 4. Packet loss
    # ---------------------------------------------------------------

    if ENABLE_PACKET_LOSS:
        if np.random.rand() < PACKET_LOSS_PROB:
            metadata["packet_loss"] = True

            if PRINT_PACKET_LOSS:
                print(
                    f"[Sample {sample_index}] "
                    f"PMU {pmu_id} packet lost"
                )

            return np.nan, np.nan, metadata

    # ---------------------------------------------------------------
    # 5. Bad data
    # ---------------------------------------------------------------

    if ENABLE_BAD_DATA:
        inject_bad_data = False

        # Faulty PMU injection
        if BAD_DATA_MODE.lower() == "faulty_pmu":
            if (
                FAULT_START_TIME <= t < FAULT_END_TIME
                and pmu_id == FAULTY_PMU
            ):
                metadata["bad_data"] = True

                measured_phase += FAULT_PHASE_ERROR
                measured_mag *= FAULT_MAG_SCALE

                if PRINT_BAD_DATA:
                    print(
                        f"[Sample {sample_index}] "
                        f"FAULTY PMU {FAULTY_PMU} "
                        f"t={t:.3f}s "
                        f"phase=+{FAULT_PHASE_ERROR:.2f}deg "
                        f"magnitude=x{FAULT_MAG_SCALE:.2f}"
                    )

        # Periodic injection
        #
        # A new fault starts every BAD_DATA_INTERVAL samples.
        # The faulty PMU is selected deterministically in round-robin order.
        # Each event remains active for PERIODIC_FAULT_DURATION samples.
        elif BAD_DATA_MODE.lower() == "periodic":
            if sample_index != 0 and BAD_DATA_INTERVAL > 0:
                event_number = sample_index // BAD_DATA_INTERVAL
                samples_into_event = sample_index % BAD_DATA_INTERVAL
                periodic_pmu = ((event_number - 1) % 3) + 1

                if (
                    samples_into_event < PERIODIC_FAULT_DURATION
                    and pmu_id == periodic_pmu
                ):
                    inject_bad_data = True

        # Random injection
        elif BAD_DATA_MODE.lower() == "random":
            if np.random.rand() < BAD_DATA_PROB:
                inject_bad_data = True

        # Inject gross error
        if inject_bad_data:
            metadata["bad_data"] = True

            if PRINT_BAD_DATA:
                print(
                    f"[Sample {sample_index}] "
                    f"Bad data injected"
                )

            if np.random.rand() < 0.5:
                measured_phase += BAD_PHASE_ERROR

                if PRINT_BAD_DATA:
                    print("   -> Phase corrupted")
            else:
                measured_mag *= BAD_MAG_SCALE

                if PRINT_BAD_DATA:
                    print("   -> Magnitude corrupted")

    return measured_mag, measured_phase, metadata


###########################################################################
# HELPER FUNCTIONS
###########################################################################

def nan_measurement_metadata():
    return {
        "sync_error": np.nan,
        "mag_noise": np.nan,
        "phase_noise": np.nan,
        "clock_drift": np.nan,
        "packet_loss": False,
        "bad_data": False,
    }


def initialize_dft_values():
    """
    Return NaN-initialized DFT values for a sample for which
    the one-cycle DFT window is not yet full.
    """
    return (np.nan, np.nan, np.nan, np.nan, np.nan)


###########################################################################
# COMPLETE SIMULATION
###########################################################################


###########################################################################
# LIVE CSV + LIVE PLOT SIMULATION
#
# Architecture:
#   QTimer
#      |
#      +--> generate one PMU sample
#      +--> append that sample to CSV
#      +--> update plot with accumulated data
#      +--> repeat
#
# The GUI remains responsive because simulation is driven by Qt.
###########################################################################

CSV_PATH = "PMU_Output.csv"

# Keep references to the GUI objects so they cannot be garbage collected.
app = None
main_window = None
win = None

sample_index = 0
csv_file = None
csv_writer = None

time_data = []
current_data = [[], [], []]
magnitude_data = [[], [], []]
phase_data = [[], [], []]

current_buffers = [
    deque(maxlen=N),
    deque(maxlen=N),
    deque(maxlen=N),
]

voltage_buffers = [
    deque(maxlen=N),
    deque(maxlen=N),
    deque(maxlen=N),
]

timer = None
scrollbar = None

curve_signal = []
curve_mag = []
curve_phase = []

paused = False

# Number of injected faulty samples for each PMU.
fault_sample_count = [0, 0, 0]


def nan_measurement_metadata():
    return {
        "sync_error": np.nan,
        "mag_noise": np.nan,
        "phase_noise": np.nan,
        "clock_drift": np.nan,
        "packet_loss": False,
        "bad_data": False,
    }


def initialize_dft_values():
    return (np.nan, np.nan, np.nan, np.nan, np.nan)


def append_csv_row(row):
    """
    Append exactly one completed PMU sample to the CSV.

    The CSV is therefore being built while the simulation is running.
    flush() makes the newly appended data visible on disk immediately.
    """
    global csv_writer

    csv_writer.writerow(row)
    csv_file.flush()


def setup_csv():
    """
    Create/overwrite the CSV and write its header before simulation starts.
    """
    global csv_file, csv_writer

    import csv

    csv_file = open(
        CSV_PATH,
        "w",
        newline="",
        buffering=1,
    )

    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(COLUMNS)
    csv_file.flush()


def finish_simulation():
    global csv_file

    timer.stop()

    if csv_file is not None:
        csv_file.flush()
        csv_file.close()
        csv_file = None

    print("----------------------------------------------")
    print("Simulation Complete")
    print(f"Samples generated : {sample_index}")
    print(f"CSV Saved         : {CSV_PATH}")
    print("Plot contains the same data accumulated into CSV.")

    if ENABLE_BAD_DATA and BAD_DATA_MODE.lower() == "periodic":
        print("\nPeriodic Fault Injection Summary")
        print("---------------------------------")
        print(f"Fault interval       : every {BAD_DATA_INTERVAL} samples "
              f"({BAD_DATA_INTERVAL * DT:.3f} s)")
        print(f"Fault duration       : {PERIODIC_FAULT_DURATION} samples "
              f"({PERIODIC_FAULT_DURATION * DT:.3f} s)")
        print("PMU selection        : deterministic round-robin")
        print("Sequence             : PMU1 -> PMU2 -> PMU3 -> repeat")
        print(f"Phase error          : +{BAD_PHASE_ERROR:.2f} deg")
        print(f"Magnitude scale      : x{BAD_MAG_SCALE:.2f}")
        print(f"PMU1 bad samples     : {fault_sample_count[0]}")
        print(f"PMU2 bad samples     : {fault_sample_count[1]}")
        print(f"PMU3 bad samples     : {fault_sample_count[2]}")

    if ENABLE_BAD_DATA and BAD_DATA_MODE.lower() == "faulty_pmu":
        final_sample_time = (TOTAL_SAMPLES - 1) * DT
        final_fault_active = (
            FAULT_START_TIME <= final_sample_time < FAULT_END_TIME
        )

        print("\nFault Injection Summary")
        print("------------------------")
        print(f"Faulty PMU        : PMU{FAULTY_PMU}")
        print(f"Fault interval    : {FAULT_START_TIME:.3f} "
              f"to {FAULT_END_TIME:.3f} s")
        print(f"Phase error       : +{FAULT_PHASE_ERROR:.2f} deg")
        print(f"Magnitude scale   : x{FAULT_MAG_SCALE:.2f}")
        print(f"PMU1 bad samples  : {fault_sample_count[0]}")
        print(f"PMU2 bad samples  : {fault_sample_count[1]}")
        print(f"PMU3 bad samples  : {fault_sample_count[2]}")
        print(f"Final sample time : {final_sample_time:.6f} s")
        print(f"Final row faulty  : {'YES' if final_fault_active else 'NO'}")

    print("----------------------------------------------")


def update_plot():
    """
    Update the already-created curves.

    For the normal 10 s / 1000 Hz case, updating 10,000 points every
    millisecond is unnecessary. Plot only every PLOT_UPDATE_SAMPLES samples.
    """
    if sample_index == 0:
        return

    curve_signal[0].setData(time_data, current_data[0])
    curve_signal[1].setData(time_data, current_data[1])
    curve_signal[2].setData(time_data, current_data[2])

    curve_mag[0].setData(time_data, magnitude_data[0])
    curve_mag[1].setData(time_data, magnitude_data[1])
    curve_mag[2].setData(time_data, magnitude_data[2])

    curve_phase[0].setData(time_data, phase_data[0])
    curve_phase[1].setData(time_data, phase_data[1])
    curve_phase[2].setData(time_data, phase_data[2])

    # Keep the display following the newest data.
    t = time_data[-1]

    if t < DISPLAY_WINDOW:
        left = 0
        right = DISPLAY_WINDOW
    else:
        left = t - DISPLAY_WINDOW
        right = t

    plot_signal.setXRange(left, right, padding=0)
    plot_mag.setXRange(left, right, padding=0)
    plot_phase.setXRange(left, right, padding=0)


def generate_one_sample():
    """
    Generate exactly one sample, append it to CSV, and update the plot.

    This is the key refactoring:
        simulation -> CSV append -> plot update
    happen in one Qt event-loop iteration.
    """
    global sample_index
    global BAD_PMU

    if sample_index >= TOTAL_SAMPLES:
        finish_simulation()
        return

    t = sample_index * DT

    #######################################################################
    # Dynamic load profiles
    #######################################################################

    LOAD1, LOAD2, LOAD3 = get_loads(t)

    #######################################################################
    # Complex bus voltages
    #######################################################################

    V = np.array([
        (1.00 - 0.02 * (LOAD1 - 1.0))
        * np.exp(1j * np.deg2rad(0.0)),

        (1.00 - 0.02 * (LOAD2 - 1.0))
        * np.exp(1j * np.deg2rad(-2.0)),

        (1.00 - 0.02 * (LOAD3 - 1.0))
        * np.exp(1j * np.deg2rad(-4.0)),
    ])

    #######################################################################
    # Network currents
    #######################################################################

    I = YBUS @ V

    V1_peak = np.abs(V[0]) * VOLTAGE_AMPLITUDE
    V2_peak = np.abs(V[1]) * VOLTAGE_AMPLITUDE
    V3_peak = np.abs(V[2]) * VOLTAGE_AMPLITUDE

    I1_peak = np.abs(I[0]) * CURRENT_AMPLITUDE
    I2_peak = np.abs(I[1]) * CURRENT_AMPLITUDE
    I3_peak = np.abs(I[2]) * CURRENT_AMPLITUDE

    #######################################################################
    # Waveforms
    #######################################################################

    voltage1 = V1_peak * np.sin(
        OMEGA * t + np.angle(V[0])
    )
    voltage2 = V2_peak * np.sin(
        OMEGA * t + np.angle(V[1])
    )
    voltage3 = V3_peak * np.sin(
        OMEGA * t + np.angle(V[2])
    )

    current1 = I1_peak * np.sin(
        OMEGA * t + np.angle(I[0])
    )
    current2 = I2_peak * np.sin(
        OMEGA * t + np.angle(I[1])
    )
    current3 = I3_peak * np.sin(
        OMEGA * t + np.angle(I[2])
    )

    signal_angle = (360.0 * FREQUENCY * t) % 360.0
    delta_t = np.nan if sample_index == 0 else DT

    #######################################################################
    # Periodic bad-PMU target
    #######################################################################

    if (
        BAD_DATA_MODE.lower() == "periodic"
        and sample_index != 0
        and sample_index % BAD_DATA_INTERVAL == 0
    ):
        event_number = sample_index // BAD_DATA_INTERVAL
        BAD_PMU = ((event_number - 1) % 3) + 1

    #######################################################################
    # DFT defaults
    #######################################################################

    real1, imag1, mag1, phase1, rms1 = initialize_dft_values()
    real2, imag2, mag2, phase2, rms2 = initialize_dft_values()
    real3, imag3, mag3, phase3, rms3 = initialize_dft_values()

    v_real1, v_imag1, v_mag1, v_phase1, v_rms1 = initialize_dft_values()
    v_real2, v_imag2, v_mag2, v_phase2, v_rms2 = initialize_dft_values()
    v_real3, v_imag3, v_mag3, v_phase3, v_rms3 = initialize_dft_values()

    meta1 = nan_measurement_metadata()
    meta2 = nan_measurement_metadata()
    meta3 = nan_measurement_metadata()

    v_meta1 = nan_measurement_metadata()
    v_meta2 = nan_measurement_metadata()
    v_meta3 = nan_measurement_metadata()

    #######################################################################
    # DFT buffers
    #######################################################################

    current_buffers[0].append(current1)
    current_buffers[1].append(current2)
    current_buffers[2].append(current3)

    voltage_buffers[0].append(voltage1)
    voltage_buffers[1].append(voltage2)
    voltage_buffers[2].append(voltage3)

    #######################################################################
    # PMU 1
    #######################################################################

    if len(voltage_buffers[0]) == N:
        (
            v_real1,
            v_imag1,
            v_mag1,
            v_phase1,
            v_rms1,
        ) = compute_dft(voltage_buffers[0])

        v_mag1, v_phase1, v_meta1 = apply_measurement_challenges(
            v_mag1,
            v_phase1,
            t,
            PMU1_SYNC_OFFSET,
            sample_index,
            1,
        )

    if len(current_buffers[0]) == N:
        (
            real1,
            imag1,
            mag1,
            phase1,
            rms1,
        ) = compute_dft(current_buffers[0])

        mag1, phase1, meta1 = apply_measurement_challenges(
            mag1,
            phase1,
            t,
            PMU1_SYNC_OFFSET,
            sample_index,
            1,
        )

        if meta1["packet_loss"]:
            real1 = np.nan
            imag1 = np.nan
            rms1 = np.nan

    #######################################################################
    # PMU 2
    #######################################################################

    if len(voltage_buffers[1]) == N:
        (
            v_real2,
            v_imag2,
            v_mag2,
            v_phase2,
            v_rms2,
        ) = compute_dft(voltage_buffers[1])

        v_mag2, v_phase2, v_meta2 = apply_measurement_challenges(
            v_mag2,
            v_phase2,
            t,
            PMU2_SYNC_OFFSET,
            sample_index,
            2,
        )

    if len(current_buffers[1]) == N:
        (
            real2,
            imag2,
            mag2,
            phase2,
            rms2,
        ) = compute_dft(current_buffers[1])

        mag2, phase2, meta2 = apply_measurement_challenges(
            mag2,
            phase2,
            t,
            PMU2_SYNC_OFFSET,
            sample_index,
            2,
        )

        if meta2["packet_loss"]:
            real2 = np.nan
            imag2 = np.nan
            rms2 = np.nan

    #######################################################################
    # PMU 3
    #######################################################################

    if len(voltage_buffers[2]) == N:
        (
            v_real3,
            v_imag3,
            v_mag3,
            v_phase3,
            v_rms3,
        ) = compute_dft(voltage_buffers[2])

        v_mag3, v_phase3, v_meta3 = apply_measurement_challenges(
            v_mag3,
            v_phase3,
            t,
            PMU3_SYNC_OFFSET,
            sample_index,
            3,
        )

    if len(current_buffers[2]) == N:
        (
            real3,
            imag3,
            mag3,
            phase3,
            rms3,
        ) = compute_dft(current_buffers[2])

        mag3, phase3, meta3 = apply_measurement_challenges(
            mag3,
            phase3,
            t,
            PMU3_SYNC_OFFSET,
            sample_index,
            3,
        )

        if meta3["packet_loss"]:
            real3 = np.nan
            imag3 = np.nan
            rms3 = np.nan

    #######################################################################
    # Fault statistics
    #######################################################################

    if meta1["bad_data"]:
        fault_sample_count[0] += 1
    if meta2["bad_data"]:
        fault_sample_count[1] += 1
    if meta3["bad_data"]:
        fault_sample_count[2] += 1

    #######################################################################
    # Per-unit values
    #######################################################################

    v_mag1_pu = v_mag1 / VOLTAGE_AMPLITUDE
    v_mag2_pu = v_mag2 / VOLTAGE_AMPLITUDE
    v_mag3_pu = v_mag3 / VOLTAGE_AMPLITUDE

    i_mag1_pu = mag1 / CURRENT_AMPLITUDE
    i_mag2_pu = mag2 / CURRENT_AMPLITUDE
    i_mag3_pu = mag3 / CURRENT_AMPLITUDE

    #######################################################################
    # Complete CSV row
    #######################################################################

    row = [
        t,

        voltage1,
        current1,
        voltage2,
        current2,
        voltage3,
        current3,

        CURRENT_AMPLITUDE,
        signal_angle,
        delta_t,

        v_real1,
        v_imag1,
        v_mag1_pu,
        v_phase1,
        v_rms1,

        real1,
        imag1,
        i_mag1_pu,
        phase1,
        rms1,

        meta1["sync_error"],
        meta1["mag_noise"],
        meta1["phase_noise"],
        meta1["clock_drift"],
        meta1["packet_loss"],
        meta1["bad_data"],

        v_real2,
        v_imag2,
        v_mag2_pu,
        v_phase2,
        v_rms2,

        real2,
        imag2,
        i_mag2_pu,
        phase2,
        rms2,

        meta2["sync_error"],
        meta2["mag_noise"],
        meta2["phase_noise"],
        meta2["clock_drift"],
        meta2["packet_loss"],
        meta2["bad_data"],

        v_real3,
        v_imag3,
        v_mag3_pu,
        v_phase3,
        v_rms3,

        real3,
        imag3,
        i_mag3_pu,
        phase3,
        rms3,

        meta3["sync_error"],
        meta3["mag_noise"],
        meta3["phase_noise"],
        meta3["clock_drift"],
        meta3["packet_loss"],
        meta3["bad_data"],
    ]

    #######################################################################
    # 1. Append row to CSV
    #######################################################################

    append_csv_row(row)

    #######################################################################
    # 2. Append same values to plot buffers
    #######################################################################

    time_data.append(t)

    current_data[0].append(current1)
    current_data[1].append(current2)
    current_data[2].append(current3)

    magnitude_data[0].append(i_mag1_pu)
    magnitude_data[1].append(i_mag2_pu)
    magnitude_data[2].append(i_mag3_pu)

    phase_data[0].append(phase1)
    phase_data[1].append(phase2)
    phase_data[2].append(phase3)

    sample_index += 1

    #######################################################################
    # 3. Update plot
    #######################################################################

    if (
        sample_index % PLOT_UPDATE_SAMPLES == 0
        or sample_index == TOTAL_SAMPLES
    ):
        update_plot()

    #######################################################################
    # 4. End simulation
    #######################################################################

    if sample_index >= TOTAL_SAMPLES:
        update_plot()
        finish_simulation()


def create_live_plot():
    global app, main_window, win, scrollbar, timer
    global plot_signal, plot_mag, plot_phase
    global curve_signal, curve_mag, curve_phase

    pg.setConfigOptions(antialias=False)

    app = QApplication(sys.argv)

    main_window = QWidget()
    layout = QVBoxLayout(main_window)

    win = pg.GraphicsLayoutWidget()
    win.setWindowTitle("PMU Simulator - Live CSV + Plot")

    #######################################################################
    # Current waveform
    #######################################################################

    plot_signal = win.addPlot(
        title="Current Waveform"
    )
    plot_signal.setLabel("left", "Current", units="A")
    plot_signal.setLabel("bottom", "Time", units="s")
    plot_signal.showGrid(x=True, y=True)
    plot_signal.addLegend(offset=(-10, 10))

    curve_signal = [
        plot_signal.plot(
            pen=pg.mkPen("y", width=2),
            name="PMU1",
        ),
        plot_signal.plot(
            pen=pg.mkPen("r", width=2),
            name="PMU2",
        ),
        plot_signal.plot(
            pen=pg.mkPen("g", width=2),
            name="PMU3",
        ),
    ]

    plot_signal.setYRange(
        -1.2 * CURRENT_AMPLITUDE,
        1.2 * CURRENT_AMPLITUDE,
    )

    #######################################################################
    # DFT magnitude
    #######################################################################

    win.nextRow()

    plot_mag = win.addPlot(
        title="One-Cycle DFT Magnitude"
    )
    plot_mag.setLabel("left", "Magnitude")
    plot_mag.setLabel("bottom", "Time", units="s")
    plot_mag.showGrid(x=True, y=True)
    plot_mag.addLegend(offset=(-10, 10))

    curve_mag = [
        plot_mag.plot(
            pen=pg.mkPen("y", width=2),
            name="PMU1",
        ),
        plot_mag.plot(
            pen=pg.mkPen("r", width=2),
            name="PMU2",
        ),
        plot_mag.plot(
            pen=pg.mkPen("g", width=2),
            name="PMU3",
        ),
    ]

    #######################################################################
    # DFT phase
    #######################################################################

    win.nextRow()

    plot_phase = win.addPlot(
        title="One-Cycle DFT Phase"
    )
    plot_phase.setLabel("left", "Phase", units="deg")
    plot_phase.setLabel("bottom", "Time", units="s")
    plot_phase.showGrid(x=True, y=True)
    plot_phase.addLegend(offset=(-10, 10))

    curve_phase = [
        plot_phase.plot(
            pen=pg.mkPen("y", width=2),
            name="PMU1",
        ),
        plot_phase.plot(
            pen=pg.mkPen("r", width=2),
            name="PMU2",
        ),
        plot_phase.plot(
            pen=pg.mkPen("g", width=2),
            name="PMU3",
        ),
    ]

    plot_phase.setYRange(-180, 180)

    #######################################################################
    # Scrollbar
    #######################################################################

    scrollbar = QScrollBar(Qt.Orientation.Horizontal)
    scrollbar.setMinimum(0)
    scrollbar.setMaximum(max(0, TOTAL_SAMPLES - 1))
    scrollbar.setSingleStep(1)
    scrollbar.setPageStep(100)

    def scroll_plot(value):
        left = value * DT
        right = left + DISPLAY_WINDOW

        plot_signal.setXRange(left, right, padding=0)
        plot_mag.setXRange(left, right, padding=0)
        plot_phase.setXRange(left, right, padding=0)

    scrollbar.valueChanged.connect(scroll_plot)

    layout.addWidget(win)
    layout.addWidget(scrollbar)

    main_window.resize(1300, 950)
    main_window.setWindowTitle("PMU Simulator - Live CSV + Plot")

    # Keep strong references.
    app.main_window = main_window
    app.graphics_widget = win
    app.scrollbar = scrollbar

    main_window.show()
    main_window.raise_()
    main_window.activateWindow()

    #######################################################################
    # Start simulation timer
    #######################################################################

    timer = QTimer()
    timer.timeout.connect(generate_one_sample)

    # QTimer interval is 1 ms for ODR=1000 Hz.
    # The simulation uses sample_index * DT for deterministic timestamps.
    timer.start(max(1, int(DT * 1000)))

    print("==============================================")
    print(" PMU Simulator - Live CSV + Live Plot")
    print("==============================================")
    print(f"Frequency           : {FREQUENCY} Hz")
    print(f"ODR                 : {ODR} Samples/sec")
    print(f"Simulation Time     : {SIMULATION_TIME} s")
    print(f"Samples/Cycle       : {N}")
    print(f"Total Samples       : {TOTAL_SAMPLES}")
    print("CSV                 : APPENDING LIVE")
    print("Plot                : UPDATING LIVE")
    print("==============================================")

    return app


if __name__ == "__main__":
    validate_fault_configuration()
    setup_csv()
    app = create_live_plot()
    sys.exit(app.exec())