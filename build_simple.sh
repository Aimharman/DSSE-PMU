#!/bin/bash
echo "========================================="
echo "Building PMU Simulator"
echo "========================================="

# Clean previous builds
rm -rf build dist *.spec

# Install required packages
pip3 install PyQt6 pyqtgraph numpy pandas

# Build with PyInstaller (using older version for compatibility)
pip3 install pyinstaller==6.10.0

# Create the executable
python3 -m PyInstaller --onefile \
    --name pmu_simulator \
    --hidden-import=PyQt6.sip \
    --hidden-import=pyqtgraph \
    --hidden-import=numpy \
    --hidden-import=pandas \
    pmu_simulator_simple.py

echo "========================================="
echo "Build complete!"
echo "Executable: ./dist/pmu_simulator"
echo "========================================="