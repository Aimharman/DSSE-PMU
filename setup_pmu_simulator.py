"""
Setup script for creating executable
Run: python3 setup_pmu_simulator.py
"""

import os
import sys
import shutil
import platform
from PyInstaller.__main__ import run

def create_executable():
    """Create executable using PyInstaller"""
    
    # Clean previous builds
    if os.path.exists('dist'):
        shutil.rmtree('dist')
    if os.path.exists('build'):
        shutil.rmtree('build')
    
    # Detect operating system
    system = platform.system()
    print(f"Detected OS: {system}")
    
    # Base options for both platforms
    opts = [
        '--onefile',                    # Single executable file
        '--name=PMU_Simulator',         # Executable name
        '--hidden-import=PyQt6.sip',    # Required for PyQt6
        '--hidden-import=pyqtgraph',    # Required for pyqtgraph
        '--hidden-import=numpy',        # Required for numpy
        '--hidden-import=pandas',       # Required for pandas
        '--hidden-import=scipy',        # Required for scipy if used
        '--collect-all=pyqtgraph',      # Collect all pyqtgraph files
        '--add-data=.:.',               # Add current directory (correct syntax)
        'pmu_simulator.py'              # Main script file
    ]
    
    # Platform-specific options
    if system == 'Windows':
        opts.append('--windowed')       # No console window on Windows
        opts.append('--icon=pmu_icon.ico')  # Icon file (optional)
    elif system == 'Linux':
        # Linux options
        opts.append('--console')        # Keep console for debugging
        # Add Linux-specific hidden imports
        opts.append('--hidden-import=PyQt6.QtCore')
        opts.append('--hidden-import=PyQt6.QtWidgets')
        opts.append('--hidden-import=PyQt6.QtGui')
    
    # Run PyInstaller
    run(opts)
    
    print("\n" + "="*50)
    if system == 'Windows':
        print("Executable created successfully!")
        print(f"Location: {os.path.abspath('dist/PMU_Simulator.exe')}")
    else:
        print("Executable created successfully!")
        print(f"Location: {os.path.abspath('dist/PMU_Simulator')}")
        print("To run: ./dist/PMU_Simulator")
    print("="*50)

if __name__ == "__main__":
    create_executable()