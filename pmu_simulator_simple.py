#!/usr/bin/env python3
"""
PMU Simulator with User Input Interface
Simplified version for easy packaging
"""

import sys
import time
from collections import deque
import numpy as np
import pandas as pd
import pyqtgraph as pg

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QScrollBar, QLabel, QComboBox,
    QGroupBox, QGridLayout, QSpinBox, QDoubleSpinBox,
    QCheckBox, QFileDialog, QMessageBox
)
from PyQt6.QtCore import Qt, QTimer

# Constants
FREQUENCY = 50
ODR = 1000
AMPLITUDE = 10
SIMULATION_TIME = 0.2
DISPLAY_WINDOW = 0.10

class PMUSimulator(QWidget):
    def __init__(self):
        super().__init__()
        
        # Simulation parameters
        self.frequency = FREQUENCY
        self.odr = ODR
        self.amplitude = AMPLITUDE
        self.simulation_time = SIMULATION_TIME
        self.display_window = DISPLAY_WINDOW
        
        # Waveform types
        self.wf_types = ['Pure Sine', 'Phase Shifted', 'Exponential Decay', 
                        'Damped Sine', 'Frequency Sweep', 'Harmonic Distortion']
        self.wf1_type = 'Pure Sine'
        self.wf2_type = 'Phase Shifted'
        self.wf3_type = 'Exponential Decay'
        
        # Derived parameters
        self.omega = 2 * np.pi * self.frequency
        self.dt = 1.0 / self.odr
        self.total_samples = int(self.simulation_time * self.odr)
        self.n = int(self.odr / self.frequency)
        
        # State
        self.running = False
        self.paused = False
        self.sample_index = 0
        
        # Data storage
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
        
        # DFT buffers
        self.dft_buffer1 = deque(maxlen=self.n)
        self.dft_buffer2 = deque(maxlen=self.n)
        self.dft_buffer3 = deque(maxlen=self.n)
        
        # Setup UI
        self.init_ui()
        
        # Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
    
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("PMU Simulator")
        self.setGeometry(100, 100, 1200, 800)
        
        main_layout = QVBoxLayout()
        
        # Controls
        control_group = QGroupBox("Controls")
        layout = QGridLayout()
        
        # Waveform selections
        row = 0
        layout.addWidget(QLabel("Signal 1:"), row, 0)
        self.wf1_combo = QComboBox()
        self.wf1_combo.addItems(self.wf_types)
        self.wf1_combo.currentTextChanged.connect(lambda t: setattr(self, 'wf1_type', t))
        layout.addWidget(self.wf1_combo, row, 1)
        
        layout.addWidget(QLabel("Signal 2:"), row, 2)
        self.wf2_combo = QComboBox()
        self.wf2_combo.addItems(self.wf_types)
        self.wf2_combo.setCurrentIndex(1)
        self.wf2_combo.currentTextChanged.connect(lambda t: setattr(self, 'wf2_type', t))
        layout.addWidget(self.wf2_combo, row, 3)
        
        layout.addWidget(QLabel("Signal 3:"), row, 4)
        self.wf3_combo = QComboBox()
        self.wf3_combo.addItems(self.wf_types)
        self.wf3_combo.setCurrentIndex(2)
        self.wf3_combo.currentTextChanged.connect(lambda t: setattr(self, 'wf3_type', t))
        layout.addWidget(self.wf3_combo, row, 5)
        row += 1
        
        # Parameters
        param_layout = QHBoxLayout()
        
        param_layout.addWidget(QLabel("Freq (Hz):"))
        self.freq_input = QDoubleSpinBox()
        self.freq_input.setRange(1, 1000)
        self.freq_input.setValue(self.frequency)
        self.freq_input.valueChanged.connect(self.update_params)
        param_layout.addWidget(self.freq_input)
        
        param_layout.addWidget(QLabel("ODR:"))
        self.odr_input = QSpinBox()
        self.odr_input.setRange(100, 10000)
        self.odr_input.setValue(self.odr)
        self.odr_input.valueChanged.connect(self.update_params)
        param_layout.addWidget(self.odr_input)
        
        param_layout.addWidget(QLabel("Amplitude:"))
        self.amp_input = QDoubleSpinBox()
        self.amp_input.setRange(0.1, 1000)
        self.amp_input.setValue(self.amplitude)
        self.amp_input.valueChanged.connect(self.update_params)
        param_layout.addWidget(self.amp_input)
        
        param_layout.addWidget(QLabel("Sim Time (s):"))
        self.sim_time_input = QDoubleSpinBox()
        self.sim_time_input.setRange(0.05, 10)
        self.sim_time_input.setValue(self.simulation_time)
        self.sim_time_input.valueChanged.connect(self.update_params)
        param_layout.addWidget(self.sim_time_input)
        
        layout.addLayout(param_layout, row, 0, 1, 6)
        row += 1
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("▶ Start")
        self.start_btn.clicked.connect(self.start_simulation)
        btn_layout.addWidget(self.start_btn)
        
        self.pause_btn = QPushButton("⏸ Pause")
        self.pause_btn.clicked.connect(self.toggle_pause)
        self.pause_btn.setEnabled(False)
        btn_layout.addWidget(self.pause_btn)
        
        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.clicked.connect(self.stop_simulation)
        self.stop_btn.setEnabled(False)
        btn_layout.addWidget(self.stop_btn)
        
        self.save_btn = QPushButton("💾 Save")
        self.save_btn.clicked.connect(self.save_data)
        btn_layout.addWidget(self.save_btn)
        
        self.clear_btn = QPushButton("🗑 Clear")
        self.clear_btn.clicked.connect(self.clear_plots)
        btn_layout.addWidget(self.clear_btn)
        
        layout.addLayout(btn_layout, row, 0, 1, 6)
        
        control_group.setLayout(layout)
        main_layout.addWidget(control_group)
        
        # Plots
        self.plot_widget = pg.GraphicsLayoutWidget()
        self.setup_plots()
        main_layout.addWidget(self.plot_widget, stretch=1)
        
        # Scrollbar
        self.scrollbar = QScrollBar(Qt.Orientation.Horizontal)
        self.scrollbar.setMinimum(0)
        self.scrollbar.setMaximum(self.total_samples)
        self.scrollbar.valueChanged.connect(self.scroll_plot)
        main_layout.addWidget(self.scrollbar)
        
        # Status
        self.status_label = QLabel("Ready")
        main_layout.addWidget(self.status_label)
        
        self.setLayout(main_layout)
    
    def setup_plots(self):
        """Setup the three plots"""
        # Signal plot
        self.plot_signal = self.plot_widget.addPlot(title="Current Waveform")
        self.plot_signal.setLabel('left', 'Current', units='A')
        self.plot_signal.setLabel('bottom', 'Time', units='s')
        self.plot_signal.showGrid(x=True, y=True)
        legend = self.plot_signal.addLegend(offset=(-10, 10))
        
        self.curve_signal1 = self.plot_signal.plot(pen=pg.mkPen('y', width=2), name="Signal 1")
        self.curve_signal2 = self.plot_signal.plot(pen=pg.mkPen('r', width=2), name="Signal 2")
        self.curve_signal3 = self.plot_signal.plot(pen=pg.mkPen('g', width=2), name="Signal 3")
        self.plot_signal.setYRange(-15, 15)
        
        # Magnitude plot
        self.plot_widget.nextRow()
        self.plot_mag = self.plot_widget.addPlot(title="DFT Magnitude")
        self.plot_mag.setLabel('left', 'Magnitude')
        self.plot_mag.setLabel('bottom', 'Time', units='s')
        self.plot_mag.showGrid(x=True, y=True)
        legend = self.plot_mag.addLegend(offset=(-10, 10))
        
        self.curve_mag1 = self.plot_mag.plot(pen=pg.mkPen('y', width=2), name="Signal 1")
        self.curve_mag2 = self.plot_mag.plot(pen=pg.mkPen('r', width=2), name="Signal 2")
        self.curve_mag3 = self.plot_mag.plot(pen=pg.mkPen('g', width=2), name="Signal 3")
        
        # Phase plot
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
    
    def update_params(self):
        """Update simulation parameters from inputs"""
        self.frequency = self.freq_input.value()
        self.odr = self.odr_input.value()
        self.amplitude = self.amp_input.value()
        self.simulation_time = self.sim_time_input.value()
        
        self.omega = 2 * np.pi * self.frequency
        self.dt = 1.0 / self.odr
        self.total_samples = int(self.simulation_time * self.odr)
        self.n = int(self.odr / self.frequency)
        
        self.scrollbar.setMaximum(self.total_samples)
        self.plot_signal.setYRange(-1.2 * self.amplitude, 1.2 * self.amplitude)
    
    def generate_waveform(self, wf_type, t, wf_num):
        """Generate waveform based on type"""
        amp = self.amplitude
        omega = self.omega
        
        if wf_type == 'Pure Sine':
            return amp * np.sin(omega * t)
        elif wf_type == 'Phase Shifted':
            return amp * np.sin(omega * (t + 0.001))
        elif wf_type == 'Exponential Decay':
            return amp * np.sin(omega * t) * np.exp(-t/0.5)
        elif wf_type == 'Damped Sine':
            return amp * np.sin(omega * (t + 0.001)) * np.exp(-t/0.5)
        elif wf_type == 'Frequency Sweep':
            return amp * np.sin(2 * np.pi * (self.frequency + 5*t) * t)
        else:  # Harmonic Distortion
            return amp * np.sin(omega * t) + 0.3*amp * np.sin(3*omega*t)
    
    def compute_dft(self, buffer):
        """Compute single-cycle DFT"""
        x = np.array(buffer)
        N = len(x)
        if N == 0:
            return np.nan, np.nan, np.nan, np.nan, np.nan
        
        X = np.sum(x * np.exp(-1j * 2 * np.pi * np.arange(N) / N))
        real = np.real(X)
        imag = np.imag(X)
        mag = (2/N) * np.abs(X)
        phase = np.degrees(np.angle(X))
        rms = mag / np.sqrt(2)
        return real, imag, mag, phase, rms
    
    def update(self):
        """Main update function"""
        if self.sample_index >= self.total_samples:
            self.stop_simulation()
            return
        
        t = self.sample_index * self.dt
        
        # Generate signals
        current1 = self.generate_waveform(self.wf1_type, t, 1)
        current2 = self.generate_waveform(self.wf2_type, t, 2)
        current3 = self.generate_waveform(self.wf3_type, t, 3)
        
        # Store signals
        self.signal_time.append(t)
        self.signal1.append(current1)
        self.signal2.append(current2)
        self.signal3.append(current3)
        
        # DFT
        self.dft_buffer1.append(current1)
        self.dft_buffer2.append(current2)
        self.dft_buffer3.append(current3)
        
        if len(self.dft_buffer1) == self.n:
            r1, i1, m1, p1, _ = self.compute_dft(self.dft_buffer1)
            self.dft_time1.append(t)
            self.dft_mag1.append(m1)
            self.dft_phase1.append(p1)
        
        if len(self.dft_buffer2) == self.n:
            r2, i2, m2, p2, _ = self.compute_dft(self.dft_buffer2)
            self.dft_time2.append(t)
            self.dft_mag2.append(m2)
            self.dft_phase2.append(p2)
        
        if len(self.dft_buffer3) == self.n:
            r3, i3, m3, p3, _ = self.compute_dft(self.dft_buffer3)
            self.dft_time3.append(t)
            self.dft_mag3.append(m3)
            self.dft_phase3.append(p3)
        
        # Update plots
        self.update_plots(t)
        
        self.sample_index += 1
        self.scrollbar.blockSignals(True)
        self.scrollbar.setValue(self.sample_index)
        self.scrollbar.blockSignals(False)
        
        progress = (self.sample_index / self.total_samples) * 100
        self.status_label.setText(f"Running: {progress:.1f}%")
    
    def update_plots(self, t):
        """Update plots"""
        # Signal plot
        self.curve_signal1.setData(self.signal_time, self.signal1)
        self.curve_signal2.setData(self.signal_time, self.signal2)
        self.curve_signal3.setData(self.signal_time, self.signal3)
        
        if t < self.display_window:
            self.plot_signal.setXRange(0, self.display_window, padding=0)
        else:
            self.plot_signal.setXRange(t - self.display_window, t, padding=0)
        
        # Magnitude plot
        if len(self.dft_time3) > 0:
            self.curve_mag1.setData(self.dft_time1, self.dft_mag1)
            self.curve_mag2.setData(self.dft_time2, self.dft_mag2)
            self.curve_mag3.setData(self.dft_time3, self.dft_mag3)
            
            if t < self.display_window:
                self.plot_mag.setXRange(0, self.display_window, padding=0)
            else:
                self.plot_mag.setXRange(t - self.display_window, t, padding=0)
        
        # Phase plot
        if len(self.dft_time3) > 0:
            self.curve_phase1.setData(self.dft_time1, self.dft_phase1)
            self.curve_phase2.setData(self.dft_time2, self.dft_phase2)
            self.curve_phase3.setData(self.dft_time3, self.dft_phase3)
            
            if t < self.display_window:
                self.plot_phase.setXRange(0, self.display_window, padding=0)
            else:
                self.plot_phase.setXRange(t - self.display_window, t, padding=0)
    
    def start_simulation(self):
        """Start simulation"""
        if self.running:
            return
        
        self.update_params()
        
        # Reset data
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
        
        self.dft_buffer1 = deque(maxlen=self.n)
        self.dft_buffer2 = deque(maxlen=self.n)
        self.dft_buffer3 = deque(maxlen=self.n)
        self.sample_index = 0
        
        self.timer.start(int(self.dt * 1000))
        
        self.running = True
        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        
        self.status_label.setText("Running...")
    
    def toggle_pause(self):
        """Toggle pause"""
        if not self.running:
            return
        
        self.paused = not self.paused
        if self.paused:
            self.timer.stop()
            self.pause_btn.setText("▶ Resume")
            self.status_label.setText("Paused")
        else:
            self.timer.start(int(self.dt * 1000))
            self.pause_btn.setText("⏸ Pause")
            self.status_label.setText("Running")
    
    def stop_simulation(self):
        """Stop simulation"""
        self.timer.stop()
        self.running = False
        self.paused = False
        
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.pause_btn.setText("⏸ Pause")
        self.stop_btn.setEnabled(False)
        
        self.status_label.setText("Stopped")
    
    def save_data(self):
        """Save data to CSV"""
        if not self.signal_time:
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save CSV", "PMU_Output.csv", "CSV Files (*.csv)"
        )
        
        if file_path:
            columns = ["Time", "Signal1", "Signal2", "Signal3", 
                      "DFT1_Mag", "DFT2_Mag", "DFT3_Mag",
                      "DFT1_Phase", "DFT2_Phase", "DFT3_Phase"]
            
            # Create data arrays
            max_len = max(len(self.signal_time), len(self.dft_time1), len(self.dft_time2), len(self.dft_time3))
            
            data = {
                "Time": self.signal_time + [np.nan] * (max_len - len(self.signal_time)),
                "Signal1": self.signal1 + [np.nan] * (max_len - len(self.signal1)),
                "Signal2": self.signal2 + [np.nan] * (max_len - len(self.signal2)),
                "Signal3": self.signal3 + [np.nan] * (max_len - len(self.signal3)),
                "DFT1_Mag": self.dft_mag1 + [np.nan] * (max_len - len(self.dft_mag1)),
                "DFT2_Mag": self.dft_mag2 + [np.nan] * (max_len - len(self.dft_mag2)),
                "DFT3_Mag": self.dft_mag3 + [np.nan] * (max_len - len(self.dft_mag3)),
                "DFT1_Phase": self.dft_phase1 + [np.nan] * (max_len - len(self.dft_phase1)),
                "DFT2_Phase": self.dft_phase2 + [np.nan] * (max_len - len(self.dft_phase2)),
                "DFT3_Phase": self.dft_phase3 + [np.nan] * (max_len - len(self.dft_phase3)),
            }
            
            df = pd.DataFrame(data)
            df.to_csv(file_path, index=False)
            QMessageBox.information(self, "Success", f"Data saved to {file_path}")
    
    def clear_plots(self):
        """Clear all plots"""
        self.curve_signal1.clear()
        self.curve_signal2.clear()
        self.curve_signal3.clear()
        self.curve_mag1.clear()
        self.curve_mag2.clear()
        self.curve_mag3.clear()
        self.curve_phase1.clear()
        self.curve_phase2.clear()
        self.curve_phase3.clear()
        
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
    
    def scroll_plot(self, value):
        """Scroll plots"""
        left = value * self.dt
        right = left + self.display_window
        
        self.plot_signal.setXRange(left, right, padding=0)
        self.plot_mag.setXRange(left, right, padding=0)
        self.plot_phase.setXRange(left, right, padding=0)


def main():
    app = QApplication(sys.argv)
    window = PMUSimulator()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()