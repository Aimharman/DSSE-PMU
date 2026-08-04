
###########################################################################
# PMU Simulator
# Part 1
# Imports, Configuration, GUI Initialization
###########################################################################

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
# USER CONFIGURATION
###########################################################################

FREQUENCY = 50                 # Hz
ODR = 1000                     # Samples/sec
AMPLITUDE = 10                 # Peak Current

SIMULATION_TIME = 10            # seconds

DISPLAY_WINDOW = 0.10          # seconds displayed on screen

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

signal_time = []
#signal_value = []

signal1 = []
signal2 = []
signal3 = []

dft_time1 = []
dft_mag1 = []
dft_phase1 = []

dft_time2 = []
dft_mag2 = []
dft_phase2 = []

dft_time3 = []
dft_mag3 = []
dft_phase3 = []


###########################################################################
# CSV STORAGE
###########################################################################

csv_rows = []
# rows1 = []
# rows2 = []
# rows3 = []

columns = [

    "Time (s)",

    "Signal 1 (A)",
    "Signal 2 (A)",
    "Signal 3 (A)",

    "Peak (A)",

    "Signal Angle (deg)",

    "Delta t (s)",

    "DFT1 Real",
    "DFT1 Imag",
    "DFT1 Magnitude",
    "DFT1 Phase (deg)",
    "DFT1 RMS",

    "DFT2 Real",
    "DFT2 Imag",
    "DFT2 Magnitude",
    "DFT2 Phase (deg)",
    "DFT2 RMS",

    "DFT3 Real",
    "DFT3 Imag",
    "DFT3 Magnitude",
    "DFT3 Phase (deg)",
    "DFT3 RMS"

]

###########################################################################
# DFT BUFFER
###########################################################################

dft_buffer1 = deque(maxlen=N)
dft_buffer2 = deque(maxlen=N)
dft_buffer3 = deque(maxlen=N)

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
    -1.2 * AMPLITUDE,
    1.2 * AMPLITUDE
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
    x = np.array(buffer)

    X = np.sum(
        x * np.exp(-1j * 2 * np.pi * np.arange(N) / N)
    )

    real = np.real(X)
    imag = np.imag(X)
    mag = (2 / N) * np.abs(X)
    phase = np.degrees(np.angle(X))
    # if len(dft_phase1):

    #     while phase - dft_phase1[-1] > 180:
    #         phase -= 360

    #     while phase - dft_phase1[-1] < -180:
    #         phase += 360

    rms = mag / np.sqrt(2)

    return real, imag, mag, phase, rms

def update():

    global sample_index

    # ---------------------------------------------------------------
    # Stop simulation
    # ---------------------------------------------------------------
    if sample_index >= TOTAL_SAMPLES:

        timer.stop()

        df = pd.DataFrame(csv_rows, columns=columns)

        df.to_csv("PMU_Output.csv", index=False)

        # df1 = pd.DataFrame(rows1, columns=columns)
        # df2 = pd.DataFrame(rows2, columns=columns)
        # df3 = pd.DataFrame(rows3, columns=columns)
        # print("Rows1 =", len(rows1))
        # print("Rows2 =", len(rows2))
        # print("Rows3 =", len(rows3))

        # with pd.ExcelWriter("PMU_Output.xlsx", engine="openpyxl") as writer:

        #     df1.to_excel(writer, sheet_name="Signal_1", index=False)

        #     df2.to_excel(writer, sheet_name="Signal_2", index=False)

        #     df3.to_excel(writer, sheet_name="Signal_3", index=False)

        print("-----------------------------------------")
        print("Simulation Complete")
        print("Samples :", TOTAL_SAMPLES)
        print("xlsx Saved : PMU_Output.xlsx")
        print("-----------------------------------------")

        app.quit()

        return

    # ---------------------------------------------------------------
    # Generate one sample
    # ---------------------------------------------------------------

    # >>>>>> Im sin wt <<<<<<<
    t = sample_index * DT
    current1 = AMPLITUDE * np.sin(OMEGA * t)


    # >>>>>> Im sin w*(t+0.001) <<<<<<<
    TIME_DELAY = 0.001      # seconds
    current2 = AMPLITUDE * np.sin(OMEGA * (t + TIME_DELAY))


    # >>>>>> Im sin (wt+(I0e^(-t/tau))) <<<<<<<
    I0 = np.deg2rad(45)      # Initial phase offset (radians)
    tau = 0.5                # seconds
    phase_mod = I0*np.exp(-t/tau)
    current3 = AMPLITUDE*np.sin(OMEGA*t + phase_mod)


    signal_angle = (360.0 * FREQUENCY * t) % 360.0

    delta_t = np.nan if sample_index == 0 else DT

    # Default DFT values until one complete cycle is available
    real1 = imag1 = mag1 = phase1 = rms1 = np.nan
    real2 = imag2 = mag2 = phase2 = rms2 = np.nan
    real3 = imag3 = mag3 = phase3 = rms3 = np.nan

    # ---------------------------------------------------------------
    # Store waveform
    # ---------------------------------------------------------------
    signal_time.append(t)
    signal1.append(current1)
    signal2.append(current2)
    signal3.append(current3)

    # ---------------------------------------------------------------
    # Sliding DFT
    # ---------------------------------------------------------------

    dft_buffer1.append(current1)
    dft_buffer2.append(current2)
    dft_buffer3.append(current3)

    if len(dft_buffer1) == N:
        real1, imag1, mag1, phase1, rms1 = compute_dft(dft_buffer1)
        # Store DFT results for plotting
        dft_time1.append(t)
        dft_mag1.append(mag1)
        dft_phase1.append(phase1)

    if len(dft_buffer2) == N:
        real2, imag2, mag2, phase2, rms2 = compute_dft(dft_buffer2)
        # Store DFT results for plotting
        dft_time2.append(t)
        dft_mag2.append(mag2)
        dft_phase2.append(phase2)

    if len(dft_buffer3) == N:
        real3, imag3, mag3, phase3, rms3 = compute_dft(dft_buffer3)
        # Store DFT results for plotting
        dft_time3.append(t)
        dft_mag3.append(mag3)
        dft_phase3.append(phase3)

        # # Store DFT results for plotting
        # dft_time.append(t)
        # dft_mag.append(magnitude)
        # dft_phase.append(phase)

    # ---------------------------------------------------------------
    # CSV Row
    # ---------------------------------------------------------------
    csv_rows.append([

        t,

        current1,
        current2,
        current3,

        AMPLITUDE,

        signal_angle,

        delta_t,

        real1,
        imag1,
        mag1,
        phase1,
        rms1,

        real2,
        imag2,
        mag2,
        phase2,
        rms2,

        real3,
        imag3,
        mag3,
        phase3,
        rms3

    ])

# Excel Implementation

    # rows1.append([
    #     t,
    #     current1,
    #     AMPLITUDE,
    #     signal_angle,
    #     delta_t,
    #     real1,
    #     imag1,
    #     mag1,
    #     phase1,
    #     rms1
    # ])

    # rows2.append([
    #     t,
    #     current2,
    #     AMPLITUDE,
    #     signal_angle,
    #     delta_t,
    #     real2,
    #     imag2,
    #     mag2,
    #     phase2,
    #     rms2
    # ])

    # rows3.append([
    #     t,
    #     current3,
    #     AMPLITUDE,
    #     signal_angle,
    #     delta_t,
    #     real3,
    #     imag3,
    #     mag3,
    #     phase3,
    #     rms3
    # ])

    # ---------------------------------------------------------------
    # Update Signal Plot
    # ---------------------------------------------------------------
 
    curve_signal1.setData(
        signal_time,
        signal1
    )

    curve_signal2.setData(
        signal_time,
        signal2
    )

    curve_signal3.setData(
        signal_time,
        signal3
    )

    # Auto-scroll waveform
    if t < DISPLAY_WINDOW:

        plot_signal.setXRange(
            0,
            DISPLAY_WINDOW,
            padding=0
        )

    else:

        plot_signal.setXRange(
            t - DISPLAY_WINDOW,
            t,
            padding=0
        )

    # ---------------------------------------------------------------
    # Update Magnitude Plot
    # ---------------------------------------------------------------
    if len(dft_time3) > 0:

        curve_mag1.setData(dft_time1, dft_mag1)
        curve_mag2.setData(dft_time2, dft_mag2)
        curve_mag3.setData(dft_time3, dft_mag3)
        
        if t < DISPLAY_WINDOW:

            plot_mag.setXRange(
                0,
                DISPLAY_WINDOW,
                padding=0
            )

        else:

            plot_mag.setXRange(
                t - DISPLAY_WINDOW,
                t,
                padding=0
            )

    # ---------------------------------------------------------------
    # Update Phase Plot
    # ---------------------------------------------------------------
    if len(dft_time3) > 0:

        curve_phase1.setData(dft_time1, dft_phase1)
        curve_phase2.setData(dft_time2, dft_phase2)
        curve_phase3.setData(dft_time3, dft_phase3)

        if t < DISPLAY_WINDOW:

            plot_phase.setXRange(
                0,
                DISPLAY_WINDOW,
                padding=0
            )

        else:

            plot_phase.setXRange(
                t - DISPLAY_WINDOW,
                t,
                padding=0
            )

    # ---------------------------------------------------------------
    # Next sample
    # ---------------------------------------------------------------
    sample_index += 1
    # scrollbar.setValue(sample_index)
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