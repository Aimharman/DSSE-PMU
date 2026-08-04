
###########################################################################
# PMU Simulator
# Part 1
# Imports, Configuration, GUI Initialization
###########################################################################

from importlib.metadata import metadata
import sys
import time
from collections import deque

import numpy as np
import pandas as pd

import pyqtgraph as pg

from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QScrollBar
)

from PyQt6.QtCore import (
    Qt,
    QTimer
)

###########################################################################
# DEBUG / LOGGING CONFIGURATION
###########################################################################

PRINT_PACKET_LOSS = False
PRINT_BAD_DATA = True
PRINT_SYNC_OFFSETS = True


###########################################################################
# USER CONFIGURATION
###########################################################################

FREQUENCY = 50                 # Hz
ODR = 1000                     # Samples/sec
#AMPLITUDE = 10                 # Peak Current
SIMULATION_TIME = 0.5           # seconds
DISPLAY_WINDOW = 0.10          # seconds displayed on screen

###########################################################################
# CURRENT SIGNAL
###########################################################################

CURRENT_AMPLITUDE = 10      # Peak Current (A)

###########################################################################
# VOLTAGE SIGNAL
###########################################################################

VOLTAGE_AMPLITUDE = 325     # Peak Voltage (230 Vrms)

###########################################################################
# MEASUREMENT CHALLENGE CONFIGURATION
###########################################################################

ENABLE_SYNC_ERROR = True
ENABLE_MEASUREMENT_NOISE = True
ENABLE_CLOCK_DRIFT = True
ENABLE_PACKET_LOSS = True
ENABLE_BAD_DATA = True

###########################################################################
# FIXED PMU SYNCHRONIZATION OFFSETS
###########################################################################

# Synchronization uncertainty
SYNC_STD_DEG = 1.0          # standard deviation (degrees)
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
# ACTIVE BAD PMU (changes only during periodic injections)
###########################################################################

BAD_PMU = np.random.randint(1, 4)

###########################################################################

# PMU measurement noise
MAG_NOISE_STD = 0.005       # 0.5%
PHASE_NOISE_STD = 0.2       # degrees

# Clock drift
CLOCK_DRIFT_RATE = 0.02     # degrees/second

# Packet loss probability
PACKET_LOSS_PROB = 0.02      # 2%

# Gross bad data
# BAD_DATA_PROB = 0.01         # 1%
# BAD_PHASE_ERROR = 20.0       # degrees
# BAD_MAG_SCALE = 1.20         # +20%

###########################################################################
# BAD DATA CONFIGURATION
###########################################################################

# Injection mode:
# "periodic" -> deterministic injection every BAD_DATA_INTERVAL samples
# "random"   -> probabilistic injection using BAD_DATA_PROB

BAD_DATA_MODE = "periodic"
BAD_DATA_INTERVAL = 500      # samples
BAD_DATA_PROB = 0.01         # used only in random mode
BAD_PHASE_ERROR = 20.0       # degrees
BAD_MAG_SCALE = 1.20         # +20%


###########################################################################
# DERIVED PARAMETERS
###########################################################################

OMEGA = 2 * np.pi * FREQUENCY
DT = 1.0 / ODR
TOTAL_SAMPLES = int(SIMULATION_TIME * ODR)
N = int(ODR / FREQUENCY)       # One-cycle DFT window (20 samples)

###########################################################################
# DATA STORAGE
###########################################################################

sample_index = 0

#--------------------------------------
# WAVEFORM STORAGE
#--------------------------------------

signal_time = []

#--------------------------------------
# VOLTAGE SIGNAL STORAGE
#--------------------------------------

voltage_signal1 = []
voltage_signal2 = []
voltage_signal3 = []

#--------------------------------------
# CURRENT SIGNAL STORAGE
#--------------------------------------

current_signal1 = []
current_signal2 = []
current_signal3 = []

#--------------------------------------
# VOLTAGE DFT STORAGE
#--------------------------------------

voltage_dft_time1 = []
voltage_dft_mag1 = []
voltage_dft_phase1 = []

voltage_dft_time2 = []
voltage_dft_mag2 = []
voltage_dft_phase2 = []

voltage_dft_time3 = []
voltage_dft_mag3 = []
voltage_dft_phase3 = []

#--------------------------------------
# CURRENT DFT STORAGE
#--------------------------------------

current_dft_time1 = []
current_dft_mag1 = []
current_dft_phase1 = []

current_dft_time2 = []
current_dft_mag2 = []
current_dft_phase2 = []

current_dft_time3 = []
current_dft_mag3 = []
current_dft_phase3 = []


###########################################################################
# CSV STORAGE
###########################################################################

csv_rows = []
# rows1 = []
# rows2 = []
# rows3 = []

columns = [

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

    "Voltage 1 DFT Real",
    "Voltage 1 DFT Imag",
    "Voltage 1 DFT Magnitude",
    "Voltage 1 DFT Phase (deg)",
    "Voltage 1 DFT RMS",
    "Current 1 DFT Real",
    "Current 1 DFT Imag",
    "Current 1 DFT Magnitude",
    "Current 1 DFT Phase (deg)",
    "Current 1 DFT RMS",

# PMU1 Challenge Metadata
    "PMU1 Sync Offset",
    "PMU1 Mag Noise",
    "PMU1 Phase Noise",
    "PMU1 Clock Drift",
    "PMU1 Packet Loss",
    "PMU1 Bad Data",

    "Voltage 2 DFT Real",
    "Voltage 2 DFT Imag",
    "Voltage 2 DFT Magnitude",
    "Voltage 2 DFT Phase (deg)",
    "Voltage 2 DFT RMS",
    "Current 2 DFT Real",
    "Current 2 DFT Imag",
    "Current 2 DFT Magnitude",
    "Current 2 DFT Phase (deg)",
    "Current 2 DFT RMS",


    "PMU2 Sync Offset",
    "PMU2 Mag Noise",
    "PMU2 Phase Noise",
    "PMU2 Clock Drift",
    "PMU2 Packet Loss",
    "PMU2 Bad Data",

    "Voltage 3 DFT Real",
    "Voltage 3 DFT Imag",
    "Voltage 3 DFT Magnitude",
    "Voltage 3 DFT Phase (deg)",
    "Voltage 3 DFT RMS",
    "Current 3 DFT Real",
    "Current 3 DFT Imag",
    "Current 3 DFT Magnitude",
    "Current 3 DFT Phase (deg)",
    "Current 3 DFT RMS",

    
    "PMU3 Sync Offset",
    "PMU3 Mag Noise",
    "PMU3 Phase Noise",
    "PMU3 Clock Drift",
    "PMU3 Packet Loss",
    "PMU3 Bad Data"

]

###########################################################################
# DFT BUFFER
###########################################################################

# dft_buffer1 = deque(maxlen=N)
# dft_buffer2 = deque(maxlen=N)
# dft_buffer3 = deque(maxlen=N)

###########################################################################
# CURRENT DFT BUFFERS
###########################################################################

current_dft_buffer1 = deque(maxlen=N)
current_dft_buffer2 = deque(maxlen=N)
current_dft_buffer3 = deque(maxlen=N)

###########################################################################
# VOLTAGE DFT BUFFERS
###########################################################################

voltage_dft_buffer1 = deque(maxlen=N)
voltage_dft_buffer2 = deque(maxlen=N)
voltage_dft_buffer3 = deque(maxlen=N)


###########################################################################
# QT APPLICATION
###########################################################################

app = QApplication(sys.argv)
scrollbar = QScrollBar(Qt.Orientation.Horizontal)
scrollbar.setMinimum(0)
scrollbar.setMaximum(TOTAL_SAMPLES)
scrollbar.setSingleStep(1)
scrollbar.setPageStep(100)
paused = False
button = QPushButton("Pause")

###########################################################################
# MAIN WINDOW
###########################################################################

win = pg.GraphicsLayoutWidget()

win.resize(1300, 900)

win.setWindowTitle(
    "PMU Simulator - Sliding One Cycle DFT"
)

###########################################################################
# PLOT 1 : SIGNAL
###########################################################################

plot_signal = win.addPlot(
    title="Current Waveform"
)

plot_signal.setLabel(
    'left',
    'Current',
    units='A'
)

plot_signal.setLabel(
    'bottom',
    'Time',
    units='s'
)

plot_signal.showGrid(
    x=True,
    y=True
)

legend = plot_signal.addLegend(offset=(-10, 10))

curve_signal1 = plot_signal.plot(
    pen=pg.mkPen('y', width=2),
    name="sin(ωt)"
)

curve_signal2 = plot_signal.plot(
    pen=pg.mkPen('r', width=2),
    name="sin(ω(t+0.001))"
)

curve_signal3 = plot_signal.plot(
    pen=pg.mkPen('g', width=2),
    name="sin(ωt+I₀e⁻ᵗ/τ)"
)

plot_signal.setYRange(
    -1.2 * CURRENT_AMPLITUDE,
    1.2 * CURRENT_AMPLITUDE
)

###########################################################################
# PLOT 2 : MAGNITUDE
###########################################################################

win.nextRow()

plot_mag = win.addPlot(
    title="Sliding One Cycle DFT Magnitude"
)

plot_mag.setLabel(
    'left',
    'Magnitude'
)

plot_mag.setLabel(
    'bottom',
    'Time',
    units='s'
)

plot_mag.showGrid(
    x=True,
    y=True
)

legend = plot_mag.addLegend(offset=(-10, 10))

curve_mag1 = plot_mag.plot(
    pen=pg.mkPen('y', width=2),
    name="sin(ωt)"
)

curve_mag2 = plot_mag.plot(
    pen=pg.mkPen('r', width=2),
    name="sin(ω(t+0.001))"
)

curve_mag3 = plot_mag.plot(
    pen=pg.mkPen('g', width=2),
    name="sin(ωt+I₀e⁻ᵗ/τ)"
)

###########################################################################
# PLOT 3 : PHASE
###########################################################################

win.nextRow()

plot_phase = win.addPlot(
    title="Sliding One Cycle DFT Phase"
)

plot_phase.setLabel(
    'left',
    'Phase',
    units='deg'
)

plot_phase.setLabel(
    'bottom',
    'Time',
    units='s'
)

plot_phase.showGrid(
    x=True,
    y=True
)

legend = plot_phase.addLegend(offset=(-10, 10))

curve_phase1 = plot_phase.plot(
    pen=pg.mkPen('y', width=2),
    name="sin(ωt)"
)

curve_phase2 = plot_phase.plot(
    pen=pg.mkPen('r', width=2),
    name="sin(ω(t+0.001))"
)

curve_phase3 = plot_phase.plot(
    pen=pg.mkPen('g', width=2),
    name="sin(ωt+I₀e⁻ᵗ/τ)"
)

plot_phase.setYRange(
    -180,
    180
)

###########################################################################
# SHOW WINDOW
###########################################################################

# win.show()

###########################################################################
# MAIN WINDOW
###########################################################################

main_window = QWidget()
layout = QVBoxLayout(main_window)
layout.addWidget(button)
layout.addWidget(win)
layout.addWidget(scrollbar)
main_window.resize(1300, 950)
main_window.setWindowTitle("PMU Simulator")
main_window.show()

###########################################################################
# TIMER
###########################################################################

timer = QTimer()

###########################################################################
# PAUSE BUTTON
###########################################################################

def toggle():
    global paused
    paused = not paused
    if paused:
        timer.stop()
        button.setText("Resume")
        print("Paused")
    else:
        timer.start(int(DT * 1000))
        button.setText("Pause")
        print("Running")

button.clicked.connect(toggle)

###########################################################################
# HORIZONTAL SCROLLBAR
###########################################################################

def scroll_plot(value):
    left = value * DT
    right = left + DISPLAY_WINDOW
    plot_signal.setXRange(left, right, padding=0)
    plot_mag.setXRange(left, right, padding=0)
    plot_phase.setXRange(left, right, padding=0)

scrollbar.valueChanged.connect(scroll_plot)

###########################################################################
# HIGH RESOLUTION START TIME
###########################################################################

start_time = time.perf_counter()

###########################################################################
# PMU Simulator
# Part 2
# Waveform Generation + Sliding One-Cycle DFT
###########################################################################

real = np.nan
imag = np.nan
magnitude = np.nan
phase = np.nan
rms = np.nan

def compute_dft(buffer):
    """
    Compute a single-cycle Discrete Fourier Transform (DFT) on the input buffer.
    
    Parameters:
    -----------
    buffer : deque
        A circular buffer containing N samples (one cycle of the waveform)
    
    Returns:
    --------
    tuple : (real, imag, mag, phase, rms)
        real : float - Real component of the DFT
        imag : float - Imaginary component of the DFT  
        mag  : float - Magnitude of the fundamental frequency component
        phase: float - Phase angle in degrees
        rms  : float - RMS value of the fundamental component
    """
    # Convert deque to numpy array for vectorized operations
    x = np.array(buffer)
    
    # Compute DFT at fundamental frequency (k=1)
    # X = Σ x[n] * e^(-j*2π*n/N) for n = 0 to N-1
    # Using Euler's formula: e^(-jθ) = cos(θ) - j*sin(θ)
    X = np.sum(x * np.exp(-1j * 2 * np.pi * np.arange(N) / N))

    # Extract real and imaginary components
    real = np.real(X)
    imag = np.imag(X)
    
    # Magnitude scaling: 2/N for fundamental component
    # This converts from DFT magnitude to peak amplitude
    mag = (2 / N) * np.abs(X)
    
    # Phase angle in degrees (atan2(imag, real))
    # Note: np.angle returns radians, convert to degrees
    phase = np.degrees(np.angle(X))
    
    # RMS value = peak / sqrt(2) for sinusoidal waveform
    rms = mag / np.sqrt(2)
    
    return real, imag, mag, phase, rms

###################################################################################
#####################<<< Section Boundary >>>######################################
###################################################################################

def apply_measurement_challenges(mag,phase,t,sync_offset,sample_index,pmu_id):

    measured_mag = mag
    measured_phase = phase

    metadata = {
        "sync_error":0.0,
        "mag_noise":0.0,
        "phase_noise":0.0,
        "clock_drift":0.0,
        "packet_loss":False,
        "bad_data":False
    }

    ###################################################
    # 1. Synchronization Error
    ###################################################

    if ENABLE_SYNC_ERROR:

        #sync_error = np.random.normal(0,SYNC_STD_DEG)
        #measured_phase += sync_error
        #metadata["sync_error"] = sync_error

        measured_phase += sync_offset
        metadata["sync_error"] = sync_offset

    ###################################################
    # 2. Measurement Noise
    ###################################################

    if ENABLE_MEASUREMENT_NOISE:

        mag_noise = np.random.normal(
            0,
            MAG_NOISE_STD
        )

        phase_noise = np.random.normal(
            0,
            PHASE_NOISE_STD
        )

        measured_mag += mag_noise
        measured_phase += phase_noise

        metadata["mag_noise"] = mag_noise
        metadata["phase_noise"] = phase_noise

    ###################################################
    # 3. Clock Drift
    ###################################################

    if ENABLE_CLOCK_DRIFT:
        drift = CLOCK_DRIFT_RATE * t
        measured_phase += drift
        metadata["clock_drift"] = drift

    ###################################################
    # 4. Packet Loss
    ###################################################

    if ENABLE_PACKET_LOSS:

        if np.random.rand() < PACKET_LOSS_PROB:
            metadata["packet_loss"] = True
            if PRINT_PACKET_LOSS:
                print(f"[Sample {sample_index}] "f"PMU {pmu_id} packet lost")
            return np.nan, np.nan, metadata


    ###################################################
    # 5. Bad Data
    ###################################################

    if ENABLE_BAD_DATA:

        inject_bad_data = False

        # ------------------------------------------------
        # Periodic Injection
        # ------------------------------------------------
        if BAD_DATA_MODE.lower() == "periodic":

            if (
                sample_index != 0 and
                sample_index % BAD_DATA_INTERVAL == 0
            ):
                if pmu_id == BAD_PMU:
                    inject_bad_data = True

        # ------------------------------------------------
        # Random Injection
        # ------------------------------------------------
        elif BAD_DATA_MODE.lower() == "random":

            if np.random.rand() < BAD_DATA_PROB:
                inject_bad_data = True

        # ------------------------------------------------
        # Inject Gross Error
        # ------------------------------------------------
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

###################################################################################
#####################<<< Section Boundary >>>######################################
###################################################################################

def update():
    """
    Main update function called by QTimer every DT seconds.
    Generates one sample, computes DFT, updates plots, and handles simulation control.
    """
    global sample_index

    # ---------------------------------------------------------------
    # Stop simulation and save data
    # ---------------------------------------------------------------
    if sample_index >= TOTAL_SAMPLES:
        # Stop the timer to prevent further updates
        timer.stop()
        
        # Convert collected data to DataFrame and save as CSV
        df = pd.DataFrame(csv_rows, columns=columns)
        df.to_csv("PMU_Output.csv", index=False)
        
        print("-----------------------------------------")
        print("Simulation Complete")
        print("Samples :", TOTAL_SAMPLES)
        print("CSV Saved : PMU_Output.csv")
        print("-----------------------------------------")
        
        # Exit the application cleanly
        app.quit()
        return


    ###########################################################################
    # Generate one sample at current time step
    ###########################################################################

    # Calculate current time in seconds
    t = sample_index * DT


    # SIGNAL 1: Pure sinusoidal waveform
    # Formula: I(t) = Im * sin(ωt)
    # Where: Im = amplitude (peak), ω = 2πf (angular frequency)
    # This represents an ideal, undistorted current signal
    current1 = CURRENT_AMPLITUDE * np.sin(OMEGA * t)
    # Voltage 1 : Ideal sinusoid
    voltage1 = VOLTAGE_AMPLITUDE * np.sin(OMEGA * t)

    # SIGNAL 2: Phase-shifted sinusoidal waveform
    # Formula: I(t) = Im * sin(ω(t + Δt))
    # Where: Δt = 0.001 seconds (1 ms time delay)
    # This represents a signal with a fixed time delay relative to signal 1
    # Useful for simulating propagation delays or phase shifts
    TIME_DELAY = 0.001      # seconds (1 millisecond)
    current2 = CURRENT_AMPLITUDE * np.sin(OMEGA * (t + TIME_DELAY))
    # Voltage 2 : Time delayed
    voltage2 = VOLTAGE_AMPLITUDE * np.sin(OMEGA * (t + TIME_DELAY))


    # SIGNAL 3: Sinusoidal with decaying phase modulation (exponential transient)
    # Formula: I(t) = Im * sin(ωt + I₀*e^(-t/τ))
    # Where: 
    #   I₀ = 45° initial phase offset (converted to radians)
    #   τ = 0.5 seconds time constant
    # This represents a phase transient that decays exponentially
    # Common in system faults, motor startup, or generator synchronization
    I0 = np.deg2rad(45)      # Initial phase offset (45° converted to radians)
    tau = 0.5                # Time constant in seconds (decay rate)
    phase_mod = I0 * np.exp(-t/tau)  # Exponentially decaying phase modulation
    current3 = CURRENT_AMPLITUDE * np.sin(OMEGA * t + phase_mod)
    # Voltage 3 : Exponential transient
    voltage3 = VOLTAGE_AMPLITUDE * np.sin(OMEGA * t + phase_mod)


    # Signal angle calculation (theoretical phase angle of an ideal signal)
    # This is the expected phase if the signal were pure sinusoidal
    # Useful for comparison with DFT-calculated phase
    signal_angle = (360.0 * FREQUENCY * t) % 360.0

    # Time step between samples (constant for fixed ODR)
    # First sample has no previous time reference (NaN)
    delta_t = np.nan if sample_index == 0 else DT

    meta1 = {
        "sync_error": np.nan,
        "mag_noise": np.nan,
        "phase_noise": np.nan,
        "clock_drift": np.nan,
        "packet_loss": False,
        "bad_data": False
    }

    meta2 = meta1.copy()
    meta3 = meta1.copy()

    # Initialize DFT values to NaN (will be updated when buffer is full)
    real1 = imag1 = mag1 = phase1 = rms1 = np.nan
    real2 = imag2 = mag2 = phase2 = rms2 = np.nan
    real3 = imag3 = mag3 = phase3 = rms3 = np.nan

    v_real1 = v_imag1 = v_mag1 = v_phase1 = v_rms1 = np.nan
    v_real2 = v_imag2 = v_mag2 = v_phase2 = v_rms2 = np.nan
    v_real3 = v_imag3 = v_mag3 = v_phase3 = v_rms3 = np.nan

    v_meta1 = meta1.copy()
    v_meta2 = meta2.copy()
    v_meta3 = meta3.copy()

    # ---------------------------------------------------------------
    # Store waveform data for plotting
    # ---------------------------------------------------------------
    
    signal_time.append(t)

    voltage_signal1.append(voltage1)
    voltage_signal2.append(voltage2)
    voltage_signal3.append(voltage3)

    current_signal1.append(current1)
    current_signal2.append(current2)
    current_signal3.append(current3)

    global BAD_PMU
    if (
        BAD_DATA_MODE.lower() == "periodic"
        and sample_index != 0
        and sample_index % BAD_DATA_INTERVAL == 0
    ):
        BAD_PMU = np.random.randint(1, 4)
        if PRINT_BAD_DATA:
            print(f"\n[Sample {sample_index}] Target PMU = {BAD_PMU}")

    # ---------------------------------------------------------------
    # Sliding DFT implementation
    # ---------------------------------------------------------------
    
    # Add new samples to circular buffers (FIFO - First In First Out)
    # When buffer reaches max length, oldest samples are automatically discarded
    # dft_buffer1.append(current1)
    # dft_buffer2.append(current2)
    # dft_buffer3.append(current3)

    # ---------------------------------------------------------------
    # Update Current DFT Buffers
    # ---------------------------------------------------------------

    current_dft_buffer1.append(current1)
    current_dft_buffer2.append(current2)
    current_dft_buffer3.append(current3)

    # ---------------------------------------------------------------
    # Update Voltage DFT Buffers
    # ---------------------------------------------------------------

    voltage_dft_buffer1.append(voltage1)
    voltage_dft_buffer2.append(voltage2)
    voltage_dft_buffer3.append(voltage3)

    # Compute DFT when buffer has collected one full cycle of samples
    # This ensures DFT is computed on exactly one cycle (N samples = 1/50 seconds)

    # ---------------------------------------------------------------
    # Voltage DFT
    # ---------------------------------------------------------------

    if len(voltage_dft_buffer1) == N:

        (
            v_real1,
            v_imag1,
            v_mag1,
            v_phase1,
            v_rms1
        ) = compute_dft(voltage_dft_buffer1)
        v_mag1, v_phase1, v_meta1 = apply_measurement_challenges(
            v_mag1,
            v_phase1,
            t,
            PMU1_SYNC_OFFSET,
            sample_index,
            1
        )
        voltage_dft_time1.append(t)
        voltage_dft_mag1.append(v_mag1)
        voltage_dft_phase1.append(v_phase1)

    if len(current_dft_buffer1) == N:
        # Compute DFT for Signal 1
        real1, imag1, mag1, phase1, rms1 = compute_dft(current_dft_buffer1)
        mag1, phase1, meta1 = apply_measurement_challenges(mag1,phase1,t,PMU1_SYNC_OFFSET,sample_index,1)
        if meta1["packet_loss"]:
            real1 = np.nan
            imag1 = np.nan
            rms1 = np.nan
        
        # Store DFT results for plotting
        current_dft_time1.append(t)
        current_dft_mag1.append(mag1)
        current_dft_phase1.append(phase1)

    if len(voltage_dft_buffer2) == N:
        (
            v_real2,
            v_imag2,
            v_mag2,
            v_phase2,
            v_rms2
        ) = compute_dft(voltage_dft_buffer2)
        v_mag2, v_phase2, v_meta2 = apply_measurement_challenges(
            v_mag2,
            v_phase2,
            t,
            PMU2_SYNC_OFFSET,
            sample_index,
            2
        )
        voltage_dft_time2.append(t)
        voltage_dft_mag2.append(v_mag2)
        voltage_dft_phase2.append(v_phase2)

    if len(current_dft_buffer2) == N:
        # Compute DFT for Signal 2
        real2, imag2, mag2, phase2, rms2 = compute_dft(current_dft_buffer2)
        mag2, phase2, meta2 = apply_measurement_challenges(mag2,phase2,t,PMU2_SYNC_OFFSET,sample_index,2)
        if meta2["packet_loss"]:
            real2 = np.nan
            imag2 = np.nan
            rms2 = np.nan
        # Store DFT results for plotting
        current_dft_time2.append(t)
        current_dft_mag2.append(mag2)
        current_dft_phase2.append(phase2)

    if len(voltage_dft_buffer3) == N:
        (
            v_real3,
            v_imag3,
            v_mag3,
            v_phase3,
            v_rms3
        ) = compute_dft(voltage_dft_buffer3)
        v_mag3, v_phase3, v_meta3 = apply_measurement_challenges(
            v_mag3,
            v_phase3,
            t,
            PMU3_SYNC_OFFSET,
            sample_index,
            3
        )
        voltage_dft_time3.append(t)
        voltage_dft_mag3.append(v_mag3)
        voltage_dft_phase3.append(v_phase3)

    if len(current_dft_buffer3) == N:
        # Compute DFT for Signal 3
        real3, imag3, mag3, phase3, rms3 = compute_dft(current_dft_buffer3)
        mag3, phase3, meta3 = apply_measurement_challenges(mag3,phase3,t,PMU3_SYNC_OFFSET,sample_index,3)
        if meta3["packet_loss"]:
            real3 = np.nan
            imag3 = np.nan
            rms3 = np.nan
        # Store DFT results for plotting
        current_dft_time3.append(t)
        current_dft_mag3.append(mag3)
        current_dft_phase3.append(phase3)

    # ---------------------------------------------------------------
    # CSV Data Storage
    # ---------------------------------------------------------------
    # Collect all data points into a single row for CSV export
    # This creates a comprehensive record of all signal and DFT values
    csv_rows.append([
        t,                      # Time stamp
        voltage1,               # Signal 1 raw value
        current1,               # Signal 1 raw value
        voltage2,               # Signal 2 raw value
        current2,               # Signal 2 raw value
        voltage3,               # Signal 3 raw value
        current3,               # Signal 3 raw value
        CURRENT_AMPLITUDE,      # Reference amplitude
        signal_angle,           # Theoretical angle
        delta_t,                # Time step
        v_real1, v_imag1, v_mag1, v_phase1, v_rms1,     # DFT results for Signal 1
        real1, imag1, mag1, phase1, rms1,               # DFT results for Signal 1
        meta1["sync_error"],
        meta1["mag_noise"],
        meta1["phase_noise"],
        meta1["clock_drift"],
        meta1["packet_loss"],
        meta1["bad_data"],
        v_real2, v_imag2, v_mag2, v_phase2, v_rms2,     # DFT results for Signal 2
        real2, imag2, mag2, phase2, rms2,  # DFT results for Signal 2
        meta2["sync_error"],
        meta2["mag_noise"],
        meta2["phase_noise"],
        meta2["clock_drift"],
        meta2["packet_loss"],
        meta2["bad_data"],
        v_real3, v_imag3, v_mag3, v_phase3, v_rms3,     # DFT results for Signal 3
        real3, imag3, mag3, phase3, rms3,   # DFT results for Signal 3
        meta3["sync_error"],
        meta3["mag_noise"],
        meta3["phase_noise"],
        meta3["clock_drift"],
        meta3["packet_loss"],
        meta3["bad_data"]
    ])

    # ---------------------------------------------------------------
    # Update Signal Plot (Waveform Display)
    # ---------------------------------------------------------------
    # Update all three signal curves with new data
    curve_signal1.setData(signal_time, current_signal1)
    curve_signal2.setData(signal_time, current_signal2)
    curve_signal3.setData(signal_time, current_signal3)

    # Auto-scroll the waveform display window
    # For first DISPLAY_WINDOW seconds: show from time 0
    # After that: follow the current time
    if t < DISPLAY_WINDOW:
        plot_signal.setXRange(0, DISPLAY_WINDOW, padding=0)
    else:
        plot_signal.setXRange(t - DISPLAY_WINDOW, t, padding=0)

    # ---------------------------------------------------------------
    # Update Magnitude Plot (DFT Magnitude Display)
    # ---------------------------------------------------------------
    # Only update magnitude plot once DFT data is available
    if len(current_dft_time3) > 0:
        curve_mag1.setData(current_dft_time1, current_dft_mag1)
        curve_mag2.setData(current_dft_time2, current_dft_mag2)
        curve_mag3.setData(current_dft_time3, current_dft_mag3)
        
        # Auto-scroll magnitude plot to match waveform
        if t < DISPLAY_WINDOW:
            plot_mag.setXRange(0, DISPLAY_WINDOW, padding=0)
        else:
            plot_mag.setXRange(t - DISPLAY_WINDOW, t, padding=0)

    # ---------------------------------------------------------------
    # Update Phase Plot (DFT Phase Display)
    # ---------------------------------------------------------------
    # Only update phase plot once DFT data is available
    if len(current_dft_time3) > 0:
        curve_phase1.setData(current_dft_time1, current_dft_phase1)
        curve_phase2.setData(current_dft_time2, current_dft_phase2)
        curve_phase3.setData(current_dft_time3, current_dft_phase3)

        # Auto-scroll phase plot to match waveform
        if t < DISPLAY_WINDOW:
            plot_phase.setXRange(0, DISPLAY_WINDOW, padding=0)
        else:
            plot_phase.setXRange(t - DISPLAY_WINDOW, t, padding=0)

    # ---------------------------------------------------------------
    # Advance to next sample
    # ---------------------------------------------------------------
    sample_index += 1
    
    # Update scrollbar position (block signals to avoid recursion)
    scrollbar.blockSignals(True)
    scrollbar.setValue(sample_index)
    scrollbar.blockSignals(False)


###########################################################################
# PMU Simulator
# Part 3
# Timer + Application Execution
###########################################################################

# Connect timer to update function
timer.timeout.connect(update)

# -------------------------------------------------------------------------
# Sampling Timer
#
# DT = 1/ODR
# For ODR = 1000 samples/sec
# DT = 1 ms
#
# Note:
# QTimer is not a hard real-time timer. It is adequate for visualization
# and demonstration purposes.
# -------------------------------------------------------------------------

timer.start(int(DT * 1000))

print("==============================================")
print(" PMU Simulator Started")
print("==============================================")
print(f"Frequency           : {FREQUENCY} Hz")
print(f"ODR                 : {ODR} Samples/sec")
print(f"Simulation Time     : {SIMULATION_TIME} s")
print(f"Samples/Cycle       : {N}")
print(f"Total Samples       : {TOTAL_SAMPLES}")
print(f"Display Window      : {DISPLAY_WINDOW*1000:.0f} ms")
print("==============================================")

# Start Qt Event Loop
sys.exit(app.exec())

#     The key additions to the annotations are:

#     Signal Generation Types - Clear explanation of each of the three signal types:

#         Pure sinusoidal (ideal)

#         Phase-shifted (time delay simulation)

#         Exponentially decaying phase modulation (transient simulation)

#     DFT Computation - Detailed breakdown of the DFT formula and scaling factors

#     Sliding Window Mechanism - Explanation of how the circular buffer maintains exactly one cycle of data

#     Plot Update Logic - How the three plots (waveform, magnitude, phase) are updated and auto-scrolled

#     Data Export - CSV storage format and data structure

#     Simulation Control - Start/stop logic and cleanup procedures

# The annotations now provide a comprehensive technical reference for anyone working with this PMU simulator code.

