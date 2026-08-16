#!/usr/bin/env python3
"""Run one PMU fault case without manually editing the simulator file.

Examples:
    python3 run_fault_case.py normal
    python3 run_fault_case.py sync
    python3 run_fault_case.py clock_drift
    python3 run_fault_case.py bad_data
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TEMPLATE = ROOT / "pmu_simulator_fault_refactored_timing_separated_voltage_window.py"

CASE_CONFIG = {
    "normal": [
        ("ENABLE_SYNC_FAULT = False", "ENABLE_SYNC_FAULT = False"),
        ("ENABLE_CLOCK_DRIFT = True", "ENABLE_CLOCK_DRIFT = False"),
        ("ENABLE_BAD_DATA = False", "ENABLE_BAD_DATA = False"),
    ],
    "sync": [
        ("ENABLE_SYNC_FAULT = False", "ENABLE_SYNC_FAULT = True"),
        ("ENABLE_CLOCK_DRIFT = True", "ENABLE_CLOCK_DRIFT = False"),
        ("ENABLE_BAD_DATA = False", "ENABLE_BAD_DATA = False"),
    ],
    "clock_drift": [
        ("ENABLE_SYNC_FAULT = False", "ENABLE_SYNC_FAULT = False"),
        ("ENABLE_CLOCK_DRIFT = True", "ENABLE_CLOCK_DRIFT = True"),
        ("ENABLE_BAD_DATA = False", "ENABLE_BAD_DATA = False"),
    ],
    "bad_data": [
        ("ENABLE_SYNC_FAULT = False", "ENABLE_SYNC_FAULT = False"),
        ("ENABLE_CLOCK_DRIFT = True", "ENABLE_CLOCK_DRIFT = False"),
        ("ENABLE_BAD_DATA = False", "ENABLE_BAD_DATA = True"),
        ('BAD_DATA_MODE = "random_event"', 'BAD_DATA_MODE = "faulty_pmu"'),
    ],
}


def patch_case(script_path: Path, case: str) -> None:
    text = script_path.read_text()
    if case not in CASE_CONFIG:
        raise ValueError(f"Unknown case: {case!r}")

    for old, new in CASE_CONFIG[case]:
        if old not in text:
            raise RuntimeError(f"Pattern not found in template: {old!r}")
        text = text.replace(old, new)

    script_path.write_text(text)


def run_case(case: str) -> int:
    if case not in CASE_CONFIG:
        print("Usage: python3 run_fault_case.py [normal|sync|clock_drift|bad_data]", file=sys.stderr)
        return 2

    temp_fd, temp_path = tempfile.mkstemp(prefix=f"{case}_case_", suffix=".py")
    os.close(temp_fd)
    temp_file = Path(temp_path)
    shutil.copy2(TEMPLATE, temp_file)
    patch_case(temp_file, case)

    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["PYTHONPATH"] = str(ROOT)

    proc = subprocess.run(
        [sys.executable, str(temp_file)],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
    )

    output = (proc.stdout or "") + (proc.stderr or "")
    lines = output.splitlines()
    print("\n".join(lines[-80:]))

    temp_file.unlink(missing_ok=True)
    return proc.returncode


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python3 run_fault_case.py [normal|sync|clock_drift|bad_data]", file=sys.stderr)
        return 2
    return run_case(sys.argv[1].strip().lower())


if __name__ == "__main__":
    raise SystemExit(main())
