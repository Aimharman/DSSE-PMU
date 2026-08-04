#!/usr/bin/env python3
###########################################################################
# PMU Simulator with User Input Interface
# Cross-platform Version (Windows/Linux/macOS)
###########################################################################

import sys
import time
import os
from collections import deque

# Determine if running as executable or script
if getattr(sys, 'frozen', False):
    # Running as compiled executable
    application_path = sys._MEIPASS
else:
    # Running as script
    application_path = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import pandas as pd
import pyqtgraph as pg

from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QScrollBar,
    QLabel,
    QLineEdit,
    QComboBox,
    QGroupBox,
    QGridLayout,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QFileDialog,
    QMessageBox,
    QTabWidget,
    QSplitter
)

from PyQt6.QtCore import (
    Qt,
    QTimer,
    pyqtSignal
)

###########################################################################
# MAIN WINDOW WITH USER INPUTS
###########################################################################

class PMUSimulatorGUI(QWidget):
    """Main GUI window for PMU Simulator with user configuration inputs"""
    
    # Custom signal to update plots from simulation thread
    update_plots_signal = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        
        # Initialize simulation parameters with defaults
        self.init_parameters()
        
        # Setup the user interface
        self.init_ui()
        
        # Initialize data storage
        self.init_data_storage()
        
        # Setup timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        
        # Simulation state
        self.paused = False
        self.running = False
        self.sample_index = 0
        
        # High resolution start time
        self.start_time = time.perf_counter()
        
    def init_parameters(self):
        """Initialize simulation parameters with default values"""
        self.params = {
            'frequency': 50.0,           # Hz
            'odr': 1000,                 # Samples/sec
            'amplitude': 10.0,           # Peak Current (A)
            'simulation_time': 0.2,      # seconds
            'display_window': 0.10,      # seconds
            'waveform1_type': 'Pure Sine',
            'waveform2_type': 'Phase Shifted',
            'waveform3_type': 'Exponential Decay',
            'time_delay': 0.001,         # seconds
            'phase_offset': 45.0,        # degrees
            'tau': 0.5,                  # seconds
            'save_csv': True,
            'csv_filename': 'PMU_Output.csv'
        }
        
        # Initialize waveform parameters
        for wf in [1, 2, 3]:
            self.params[f'wf{wf}_phase'] = 0.0
            self.params[f'wf{wf}_delay'] = 0.001
            self.params[f'wf{wf}_tau'] = 0.5
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("PMU Simulator - Configurable Waveform Generator")
        self.setGeometry(100, 100, 1400, 1000)
        
        # Main layout
        main_layout = QVBoxLayout()
        
        # Create tab widget for better organization
        self.tab_widget = QTabWidget()
        
        # Control tab
        control_tab = QWidget()
        control_layout = QVBoxLayout()
        control_layout.addWidget(self.create_control_panel())
        control_tab.setLayout(control_layout)
        self.tab_widget.addTab(control_tab, "Controls")
        
        # Plots tab
        plots_tab = QWidget()
        plots_layout = QVBoxLayout()
        
        # Create plots container
        self.plot_widget = pg.GraphicsLayoutWidget()
        self.setup_plots()
        plots_layout.addWidget(self.plot_widget, stretch=1)
        
        # Scrollbar for plots
        self.scrollbar = QScrollBar(Qt.Orientation.Horizontal)
        self.scrollbar.setMinimum(0)
        self.scrollbar.setMaximum(1000)
        self.scrollbar.setSingleStep(1)
        self.scrollbar.setPageStep(100)
        self.scrollbar.valueChanged.connect(self.scroll_plot)
        plots_layout.addWidget(self.scrollbar)
        
        plots_tab.setLayout(plots_layout)
        self.tab_widget.addTab(plots_tab, "Plots")
        
        main_layout.addWidget(self.tab_widget)
        
        # Status bar
        self.status_label = QLabel("Ready to start simulation")
        main_layout.addWidget(self.status_label)
        
        self.setLayout(main_layout)
        
    def create_control_panel(self):
        """Create the control panel with input widgets"""
        control_group = QGroupBox("Simulation Controls")
        layout = QGridLayout()
        
        row = 0
        
        # ----- WAVEFORM CONFIGURATION -----
        # Waveform 1
        layout.addWidget(QLabel("Waveform 1:"), row, 0)
        self.waveform1_combo = QComboBox()
        waveform_types = [
            'Pure Sine',
            'Phase Shifted',
            'Exponential Decay',
            'Damped Sine',
            'Frequency Sweep',
            'Harmonic Distortion',
            'Square Wave',
            'Custom'
        ]
        self.waveform1_combo.addItems(waveform_types)
        self.waveform1_combo.setCurrentText(self.params['waveform1_type'])
        self.waveform1_combo.currentTextChanged.connect(
            lambda t: self.update_waveform_params(1, t)
        )
        layout.addWidget(self.waveform1_combo, row, 1)
        
        # Waveform 1 parameters
        self.wf1_params = QHBoxLayout()
        self.create_waveform_param_widgets(self.wf1_params, 1)
        layout.addLayout(self.wf1_params, row, 2, 1, 2)
        row += 1
        
        # Waveform 2
        layout.addWidget(QLabel("Waveform 2:"), row, 0)
        self.waveform2_combo = QComboBox()
        self.waveform2_combo.addItems(waveform_types)
        self.waveform2_combo.setCurrentText(self.params['waveform2_type'])
        self.waveform2_combo.currentTextChanged.connect(
            lambda t: self.update_waveform_params(2, t)
        )
        layout.addWidget(self.waveform2_combo, row, 1)
        
        self.wf2_params = QHBoxLayout()
        self.create_waveform_param_widgets(self.wf2_params, 2)
        layout.addLayout(self.wf2_params, row, 2, 1, 2)
        row += 1
        
        # Waveform 3
        layout.addWidget(QLabel("Waveform 3:"), row, 0)
        self.waveform3_combo = QComboBox()
        self.waveform3_combo.addItems(waveform_types)
        self.waveform3_combo.setCurrentText(self.params['waveform3_type'])
        self.waveform3_combo.currentTextChanged.connect(
            lambda t: self.update_waveform_params(3, t)
        )
        layout.addWidget(self.waveform3_combo, row, 1)
        
        self.wf3_params = QHBoxLayout()
        self.create_waveform_param_widgets(self.wf3_params, 3)
        layout.addLayout(self.wf3_params, row, 2, 1, 2)
        row += 1
        
        # ----- GLOBAL PARAMETERS -----
        # Row for global parameters
        global_layout = QHBoxLayout()
        
        # Frequency
        global_layout.addWidget(QLabel("Frequency (Hz):"))
        self.freq_input = QDoubleSpinBox()
        self.freq_input.setRange(1, 1000)
        self.freq_input.setValue(self.params['frequency'])
        self.freq_input.setSingleStep(1)
        self.freq_input.valueChanged.connect(lambda v: self.update_global_param('frequency', v))
        global_layout.addWidget(self.freq_input)
        
        # ODR
        global_layout.addWidget(QLabel("ODR (sps):"))
        self.odr_input = QSpinBox()
        self.odr_input.setRange(100, 10000)
        self.odr_input.setValue(self.params['odr'])
        self.odr_input.setSingleStep(100)
        self.odr_input.valueChanged.connect(lambda v: self.update_global_param('odr', v))
        global_layout.addWidget(self.odr_input)
        
        # Amplitude
        global_layout.addWidget(QLabel("Amplitude (A):"))
        self.amp_input = QDoubleSpinBox()
        self.amp_input.setRange(0.1, 1000)
        self.amp_input.setValue(self.params['amplitude'])
        self.amp_input.setSingleStep(1)
        self.amp_input.valueChanged.connect(lambda v: self.update_global_param('amplitude', v))
        global_layout.addWidget(self.amp_input)
        
        # Simulation Time
        global_layout.addWidget(QLabel("Sim Time (s):"))
        self.sim_time_input = QDoubleSpinBox()
        self.sim_time_input.setRange(0.05, 10)
        self.sim_time_input.setValue(self.params['simulation_time'])
        self.sim_time_input.setSingleStep(0.05)
        self.sim_time_input.valueChanged.connect(lambda v: self.update_global_param('simulation_time', v))
        global_layout.addWidget(self.sim_time_input)
        
        # Display Window
        global_layout.addWidget(QLabel("Display (s):"))
        self.display_input = QDoubleSpinBox()
        self.display_input.setRange(0.02, 1)
        self.display_input.setValue(self.params['display_window'])
        self.display_input.setSingleStep(0.01)
        self.display_input.valueChanged.connect(lambda v: self.update_global_param('display_window', v))
        global_layout.addWidget(self.display_input)
        
        layout.addLayout(global_layout, row, 0, 1, 4)
        row += 1
        
        # ----- CONTROL BUTTONS -----
        button_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("▶ Start Simulation")
        self.start_btn.clicked.connect(self.start_simulation)
        self.start_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
        button_layout.addWidget(self.start_btn)
        
        self.pause_btn = QPushButton("⏸ Pause")
        self.pause_btn.clicked.connect(self.toggle_pause)
        self.pause_btn.setEnabled(False)
        button_layout.addWidget(self.pause_btn)
        
        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.clicked.connect(self.stop_simulation)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; font-weight: bold; }")
        button_layout.addWidget(self.stop_btn)
        
        button_layout.addStretch()
        
        self.save_btn = QPushButton("💾 Save Data")
        self.save_btn.clicked.connect(self.save_data)
        button_layout.addWidget(self.save_btn)
        
        self.clear_btn = QPushButton("🗑 Clear Plots")
        self.clear_btn.clicked.connect(self.clear_plots)
        button_layout.addWidget(self.clear_btn)
        
        self.export_btn = QPushButton("📤 Export Config")
        self.export_btn.clicked.connect(self.export_config)
        button_layout.addWidget(self.export_btn)
        
        layout.addLayout(button_layout, row, 0, 1, 4)
        row += 1
        
        # ----- OPTIONS -----
        options_layout = QHBoxLayout()
        
        self.save_csv_check = QCheckBox("Auto-save CSV")
        self.save_csv_check.setChecked(self.params['save_csv'])
        self.save_csv_check.stateChanged.connect(
            lambda s: self.update_global_param('save_csv', bool(s))
        )
        options_layout.addWidget(self.save_csv_check)
        
        self.show_phase_check = QCheckBox("Show Phase")
        self.show_phase_check.setChecked(True)
        self.show_phase_check.stateChanged.connect(self.toggle_phase_plot)
        options_layout.addWidget(self.show_phase_check)
        
        self.show_mag_check = QCheckBox("Show Magnitude")
        self.show_mag_check.setChecked(True)
        self.show_mag_check.stateChanged.connect(self.toggle_mag_plot)
        options_layout.addWidget(self.show_mag_check)
        
        layout.addLayout(options_layout, row, 0, 1, 4)
        
        control_group.setLayout(layout)
        return control_group
    
    def create_waveform_param_widgets(self, layout, wf_num):
        """Create parameter input widgets for a specific waveform"""
        # This stores references to widgets for later access
        if not hasattr(self, 'wf_param_widgets'):
            self.wf_param_widgets = {}
        
        # Phase offset
        phase_label = QLabel("Phase (°):")
        layout.addWidget(phase_label)
        phase_input = QDoubleSpinBox()
        phase_input.setRange(-360, 360)
        phase_input.setValue(0)
        phase_input.setSingleStep(5)
        phase_input.setPrefix("wf" + str(wf_num) + "_")
        phase_input.valueChanged.connect(
            lambda v, n=wf_num: self.update_waveform_param(n, 'phase', v)
        )
        layout.addWidget(phase_input)
        self.wf_param_widgets[f'wf{wf_num}_phase'] = phase_input
        
        # Time delay
        delay_label = QLabel("Delay (ms):")
        layout.addWidget(delay_label)
        delay_input = QDoubleSpinBox()
        delay_input.setRange(0, 100)
        delay_input.setValue(1)
        delay_input.setSingleStep(0.5)
        delay_input.valueChanged.connect(
            lambda v, n=wf_num: self.update_waveform_param(n, 'delay', v/1000)
        )
        layout.addWidget(delay_input)
        self.wf_param_widgets[f'wf{wf_num}_delay'] = delay_input
        
        # Tau (decay constant)
        tau_label = QLabel("Tau (s):")
        layout.addWidget(tau_label)
        tau_input = QDoubleSpinBox()
        tau_input.setRange(0.01, 10)
        tau_input.setValue(0.5)
        tau_input.setSingleStep(0.1)
        tau_input.valueChanged.connect(
            lambda v, n=wf_num: self.update_waveform_param(n, 'tau', v)
        )
        layout.addWidget(tau_input)
        self.wf_param_widgets[f'wf{wf_num}_tau'] = tau_input
        
        # Hide some parameters initially
        delay_label.hide()
        delay_input.hide()
        tau_label.hide()
        tau_input.hide()
        
    def update_waveform_params(self, wf_num, waveform_type):
        """Update parameter visibility based on waveform type"""
        param_key = f'waveform{wf_num}_type'
        self.params[param_key] = waveform_type
        
        # Determine which parameters to show
        show_delay = waveform_type in ['Phase Shifted', 'Damped Sine']
        show_tau = waveform_type in ['Exponential Decay', 'Damped Sine']
        
        # Update widget visibility
        delay_widget = self.wf_param_widgets.get(f'wf{wf_num}_delay')
        tau_widget = self.wf_param_widgets.get(f'wf{wf_num}_tau')
        
        if delay_widget:
            # Find and update label visibility
            parent_layout = delay_widget.parent().layout()
            if parent_layout:
                for i in range(parent_layout.count()):
                    widget = parent_layout.itemAt(i).widget()
                    if isinstance(widget, QLabel) and widget.text() == "Delay (ms):":
                        widget.setVisible(show_delay)
                delay_widget.setVisible(show_delay)
        
        if tau_widget:
            parent_layout = tau_widget.parent().layout()
            if parent_layout:
                for i in range(parent_layout.count()):
                    widget = parent_layout.itemAt(i).widget()
                    if isinstance(widget, QLabel) and widget.text() == "Tau (s):":
                        widget.setVisible(show_tau)
                tau_widget.setVisible(show_tau)
    
    def update_waveform_param(self, wf_num, param_name, value):
        """Update a specific waveform parameter"""
        param_key = f'wf{wf_num}_{param_name}'
        self.params[param_key] = value
    
    def update_global_param(self, param_name, value):
        """Update global simulation parameters"""
        self.params[param_name] = value
        
        # Recalculate derived parameters
        if param_name in ['frequency', 'odr']:
            self.calculate_derived_params()
    
    def calculate_derived_params(self):
        """Calculate derived parameters from user inputs"""
        self.params['omega'] = 2 * np.pi * self.params['frequency']
        self.params['dt'] = 1.0 / self.params['odr']
        self.params['total_samples'] = int(self.params['simulation_time'] * self.params['odr'])
        self.params['n'] = int(self.params['odr'] / self.params['frequency'])
        
        # Update scrollbar
        self.scrollbar.setMaximum(self.params['total_samples'])
    
    def setup_plots(self):
        """Initialize the three plots"""
        # Plot 1: Signal Waveform
        self.plot_signal = self.plot_widget.addPlot(title="Current Waveform")
        self.plot_signal.setLabel('left', 'Current', units='A')
        self.plot_signal.setLabel('bottom', 'Time', units='s')
        self.plot_signal.showGrid(x=True, y=True)
        legend = self.plot_signal.addLegend(offset=(-10, 10))
        
        self.curve_signal1 = self.plot_signal.plot(pen=pg.mkPen('y', width=2), name="Signal 1")
        self.curve_signal2 = self.plot_signal.plot(pen=pg.mkPen('r', width=2), name="Signal 2")
        self.curve_signal3 = self.plot_signal.plot(pen=pg.mkPen('g', width=2), name="Signal 3")
        self.plot_signal.setYRange(-12, 12)
        
        # Plot 2: Magnitude
        self.plot_widget.nextRow()
        self.plot_mag = self.plot_widget.addPlot(title="DFT Magnitude")
        self.plot_mag.setLabel('left', 'Magnitude')
        self.plot_mag.setLabel('bottom', 'Time', units='s')
        self.plot_mag.showGrid(x=True, y=True)
        legend = self.plot_mag.addLegend(offset=(-10, 10))
        
        self.curve_mag1 = self.plot_mag.plot(pen=pg.mkPen('y', width=2), name="Signal 1")
        self.curve_mag2 = self.plot_mag.plot(pen=pg.mkPen('r', width=2), name="Signal 2")
        self.curve_mag3 = self.plot_mag.plot(pen=pg.mkPen('g', width=2), name="Signal 3")
        
        # Plot 3: Phase
        self.plot_widget.nextRow()
        self.plot_phase = self.plot_widget.addPlot(title="DFT Phase")
        self.plot_phase.setLabel('left', 'Phase', units='deg')
        self.plot_phase.setLabel('bottom', 'Time', units='s')
        self.plot_phase.showGrid(x=True, y=True)
        legend = self.plot_phase.addLegend(offset=(-10, 10))
        
        self.curve_phase1 = self.plot_phase.plot(pen=pg.mkPen('y', width=2), name="Signal 1")
        self.curve_phase2 = self.plot_phase.plot(pen=pg.mkPen('r', width=2), name="Signal 2")
        self.curve_phase3 = self.plot_phase.plot(pen=pg.mkPen('g', width=2), name="Signal 3")
        self.plot_phase.setYRange(-180, 180)
    
    def init_data_storage(self):
        """Initialize data storage arrays"""
        self.signal_time = []
        self.signal1 = []
        self.signal2 = []
        self.signal3 = []
        
        self.dft_time1 = []
        self.dft_mag1 = []
        self.dft_phase1 = []
        
        self.dft_time2 = []
        self.dft_mag2 = []
        self.dft_phase2 = []
        
        self.dft_time3 = []
        self.dft_mag3 = []
        self.dft_phase3 = []
        
        self.csv_rows = []
        
        # DFT Buffers
        n = self.params.get('n', 20)
        self.dft_buffer1 = deque(maxlen=n)
        self.dft_buffer2 = deque(maxlen=n)
        self.dft_buffer3 = deque(maxlen=n)
        
        # Column definitions for CSV
        self.columns = [
            "Time (s)",
            "Signal 1 (A)", "Signal 2 (A)", "Signal 3 (A)",
            "Peak (A)", "Signal Angle (deg)", "Delta t (s)",
            "DFT1 Real", "DFT1 Imag", "DFT1 Magnitude", "DFT1 Phase (deg)", "DFT1 RMS",
            "DFT2 Real", "DFT2 Imag", "DFT2 Magnitude", "DFT2 Phase (deg)", "DFT2 RMS",
            "DFT3 Real", "DFT3 Imag", "DFT3 Magnitude", "DFT3 Phase (deg)", "DFT3 RMS"
        ]
    
    def generate_waveform(self, wf_type, t, wf_num):
        """Generate waveform based on selected type and parameters"""
        amp = self.params['amplitude']
        omega = self.params['omega']
        phase = self.params.get(f'wf{wf_num}_phase', 0)
        delay = self.params.get(f'wf{wf_num}_delay', 0)
        tau = self.params.get(f'wf{wf_num}_tau', 0.5)
        
        # Convert phase from degrees to radians
        phase_rad = np.deg2rad(phase)
        
        try:
            if wf_type == 'Pure Sine':
                return amp * np.sin(omega * t + phase_rad)
            
            elif wf_type == 'Phase Shifted':
                return amp * np.sin(omega * (t + delay) + phase_rad)
            
            elif wf_type == 'Exponential Decay':
                return amp * np.sin(omega * t + phase_rad) * np.exp(-t/tau)
            
            elif wf_type == 'Damped Sine':
                return amp * np.sin(omega * (t + delay) + phase_rad) * np.exp(-t/tau)
            
            elif wf_type == 'Frequency Sweep':
                sweep_rate = 5  # Hz/s
                freq = self.params['frequency'] + sweep_rate * t
                omega_sweep = 2 * np.pi * freq
                return amp * np.sin(omega_sweep * t + phase_rad)
            
            elif wf_type == 'Harmonic Distortion':
                h3_amp = 0.3 * amp
                return amp * np.sin(omega * t + phase_rad) + h3_amp * np.sin(3 * omega * t + phase_rad)
            
            elif wf_type == 'Square Wave':
                value = 0
                for n in range(1, 11, 2):
                    value += (4 * amp / (n * np.pi)) * np.sin(n * omega * t + phase_rad)
                return value
            
            else:  # Custom waveform
                return amp * np.sin(omega * t + phase_rad) * (1 + 0.5 * np.sin(5 * omega * t))
        
        except Exception as e:
            print(f"Error generating waveform {wf_num}: {e}")
            return 0
    
    def compute_dft(self, buffer):
        """Compute single-cycle DFT on buffer"""
        try:
            x = np.array(buffer)
            N = len(x)
            if N == 0:
                return np.nan, np.nan, np.nan, np.nan, np.nan
            
            X = np.sum(x * np.exp(-1j * 2 * np.pi * np.arange(N) / N))
            
            real = np.real(X)
            imag = np.imag(X)
            mag = (2 / N) * np.abs(X)
            phase = np.degrees(np.angle(X))
            rms = mag / np.sqrt(2)
            
            return real, imag, mag, phase, rms
        
        except Exception as e:
            print(f"DFT computation error: {e}")
            return np.nan, np.nan, np.nan, np.nan, np.nan
    
    def update(self):
        """Main update function called by timer"""
        if self.sample_index >= self.params['total_samples']:
            self.stop_simulation()
            return
        
        try:
            # Calculate time
            t = self.sample_index * self.params['dt']
            
            # Generate waveforms
            wf1_type = self.params['waveform1_type']
            wf2_type = self.params['waveform2_type']
            wf3_type = self.params['waveform3_type']
            
            current1 = self.generate_waveform(wf1_type, t, 1)
            current2 = self.generate_waveform(wf2_type, t, 2)
            current3 = self.generate_waveform(wf3_type, t, 3)
            
            # Store waveform data
            self.signal_time.append(t)
            self.signal1.append(current1)
            self.signal2.append(current2)
            self.signal3.append(current3)
            
            # Update DFT buffers
            self.dft_buffer1.append(current1)
            self.dft_buffer2.append(current2)
            self.dft_buffer3.append(current3)
            
            # Compute DFT when buffer is full
            real1 = imag1 = mag1 = phase1 = rms1 = np.nan
            real2 = imag2 = mag2 = phase2 = rms2 = np.nan
            real3 = imag3 = mag3 = phase3 = rms3 = np.nan
            
            n = self.params['n']
            
            if len(self.dft_buffer1) == n:
                real1, imag1, mag1, phase1, rms1 = self.compute_dft(self.dft_buffer1)
                self.dft_time1.append(t)
                self.dft_mag1.append(mag1)
                self.dft_phase1.append(phase1)
            
            if len(self.dft_buffer2) == n:
                real2, imag2, mag2, phase2, rms2 = self.compute_dft(self.dft_buffer2)
                self.dft_time2.append(t)
                self.dft_mag2.append(mag2)
                self.dft_phase2.append(phase2)
            
            if len(self.dft_buffer3) == n:
                real3, imag3, mag3, phase3, rms3 = self.compute_dft(self.dft_buffer3)
                self.dft_time3.append(t)
                self.dft_mag3.append(mag3)
                self.dft_phase3.append(phase3)
            
            # Signal angle
            signal_angle = (360.0 * self.params['frequency'] * t) % 360.0
            delta_t = np.nan if self.sample_index == 0 else self.params['dt']
            
            # Store CSV data
            self.csv_rows.append([
                t, current1, current2, current3,
                self.params['amplitude'], signal_angle, delta_t,
                real1, imag1, mag1, phase1, rms1,
                real2, imag2, mag2, phase2, rms2,
                real3, imag3, mag3, phase3, rms3
            ])
            
            # Update plots
            self.update_plots(t)
            
            # Update scrollbar
            self.sample_index += 1
            self.scrollbar.blockSignals(True)
            self.scrollbar.setValue(self.sample_index)
            self.scrollbar.blockSignals(False)
            
            # Update status
            if self.sample_index % 10 == 0:  # Update every 10 samples
                progress = (self.sample_index / self.params['total_samples']) * 100
                self.status_label.setText(
                    f"⏳ Running: {progress:.1f}% complete - "
                    f"Sample {self.sample_index}/{self.params['total_samples']}"
                )
        
        except Exception as e:
            print(f"Update error: {e}")
            self.stop_simulation()
    
    def update_plots(self, t):
        """Update all plots with new data"""
        try:
            # Update signal plot
            self.curve_signal1.setData(self.signal_time, self.signal1)
            self.curve_signal2.setData(self.signal_time, self.signal2)
            self.curve_signal3.setData(self.signal_time, self.signal3)
            
            # Auto-scroll signal plot
            display_window = self.params['display_window']
            if t < display_window:
                self.plot_signal.setXRange(0, display_window, padding=0)
            else:
                self.plot_signal.setXRange(t - display_window, t, padding=0)
            
            # Update magnitude plot
            if len(self.dft_time3) > 0:
                self.curve_mag1.setData(self.dft_time1, self.dft_mag1)
                self.curve_mag2.setData(self.dft_time2, self.dft_mag2)
                self.curve_mag3.setData(self.dft_time3, self.dft_mag3)
                
                if t < display_window:
                    self.plot_mag.setXRange(0, display_window, padding=0)
                else:
                    self.plot_mag.setXRange(t - display_window, t, padding=0)
            
            # Update phase plot
            if len(self.dft_time3) > 0:
                self.curve_phase1.setData(self.dft_time1, self.dft_phase1)
                self.curve_phase2.setData(self.dft_time2, self.dft_phase2)
                self.curve_phase3.setData(self.dft_time3, self.dft_phase3)
                
                if t < display_window:
                    self.plot_phase.setXRange(0, display_window, padding=0)
                else:
                    self.plot_phase.setXRange(t - display_window, t, padding=0)
        
        except Exception as e:
            print(f"Plot update error: {e}")
    
    def start_simulation(self):
        """Start the simulation"""
        if self.running:
            return
        
        try:
            # Recalculate derived parameters
            self.calculate_derived_params()
            
            # Reset data storage
            self.init_data_storage()
            
            # Reset sample index
            self.sample_index = 0
            
            # Reset DFT buffers with new N
            n = self.params['n']
            self.dft_buffer1 = deque(maxlen=n)
            self.dft_buffer2 = deque(maxlen=n)
            self.dft_buffer3 = deque(maxlen=n)
            
            # Update Y range
            amp = self.params['amplitude']
            self.plot_signal.setYRange(-1.2 * amp, 1.2 * amp)
            
            # Start timer
            dt_ms = self.params['dt'] * 1000
            self.timer.start(int(max(1, dt_ms)))  # Ensure at least 1ms
            
            # Update button states
            self.running = True
            self.start_btn.setEnabled(False)
            self.start_btn.setText("⏳ Running...")
            self.pause_btn.setEnabled(True)
            self.stop_btn.setEnabled(True)
            
            # Disable parameter inputs during simulation
            self.set_inputs_enabled(False)
            
            self.status_label.setText("▶ Simulation running...")
            print(f"Simulation started: {self.params['frequency']} Hz, {self.params['odr']} samples/sec")
            
            # Switch to plots tab
            self.tab_widget.setCurrentIndex(1)
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start simulation: {str(e)}")
            print(f"Start error: {e}")
    
    def toggle_pause(self):
        """Pause or resume simulation"""
        if not self.running:
            return
        
        self.paused = not self.paused
        
        if self.paused:
            self.timer.stop()
            self.pause_btn.setText("▶ Resume")
            self.status_label.setText("⏸ Paused")
            print("Simulation paused")
        else:
            dt_ms = self.params['dt'] * 1000
            self.timer.start(int(max(1, dt_ms)))
            self.pause_btn.setText("⏸ Pause")
            self.status_label.setText("▶ Resumed")
            print("Simulation resumed")
    
    def stop_simulation(self):
        """Stop the simulation"""
        self.timer.stop()
        self.running = False
        self.paused = False
        
        # Update button states
        self.start