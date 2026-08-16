#!/usr/bin/env bash
set -e

cd /home/ankesh/Documents/IIT_Jammu/DSSE-PMU/Development-directory

for c in normal sync clock_drift bad_data; do
  echo
  echo "===== $c ====="
  timeout 25s python3 run_fault_case.py "$c" 2>&1 | tail -n 60
done