"""
Neural active fault management controller.

This module provides the active-control layer for the PMU fault framework.
It is intentionally lightweight and implementation-friendly: the controller
maps a detected fault label and affected PMU into a management action that
can be applied to the WLS weighting layer.

The design follows the project direction:
    fault detection -> classification -> management decision -> estimator action
"""

from __future__ import annotations

from typing import Dict, List, Tuple


class NeuralActiveFaultController:
    """Map the detected fault class to a control action for the estimator."""

    def __init__(self):
        self.class_to_action = {
            "NORMAL": "ACCEPT",
            "BAD_DATA": "DOWN_WEIGHT",
            "SYNC_FAULT": "PHASE_COMPENSATE",
            "CLOCK_DRIFT": "TIMING_CORRECTION",
            "TRANSIENT_FAULT": "ISOLATE",
        }

        self.action_to_weight = {
            "ACCEPT": 1.0,
            "DOWN_WEIGHT": 0.1,
            "PHASE_COMPENSATE": 0.3,
            "TIMING_CORRECTION": 0.2,
            "ISOLATE": 0.0,
        }

        self.supported_classes = sorted(self.class_to_action.keys())

    def label_fault(self, fault_type: str, pmu_id: str | int | None = None) -> Dict[str, object]:
        """Return a structured controller decision for a detected fault."""
        fault_key = str(fault_type).upper().strip()
        if fault_key not in self.class_to_action:
            fault_key = "NORMAL"

        action = self.class_to_action[fault_key]
        weight = self.action_to_weight.get(action, 1.0)

        record = {
            "fault_type": fault_key,
            "pmu_id": pmu_id,
            "action": action,
            "weight": float(weight),
            "decision": f"{action} PMU{pmu_id}" if pmu_id is not None else action,
        }

        return record

    def get_measurement_weights(self, pmu_id: str | int | None, fault_type: str) -> Dict[str, float]:
        """Return per-PMU weight policy for a single detected PMU fault."""
        target_pmu = str(pmu_id).upper() if pmu_id is not None else "ALL"
        action = self.label_fault(fault_type, target_pmu)["action"]
        weight = self.action_to_weight.get(action, 1.0)

        return {
            "PMU1": 1.0,
            "PMU2": 1.0,
            "PMU3": 1.0,
            target_pmu: weight if target_pmu != "ALL" else 1.0,
        }

    def explain_action(self, fault_type: str, pmu_id: str | int | None = None) -> str:
        """Short human-readable description of the controller decision."""
        decision = self.label_fault(fault_type, pmu_id)
        action = decision["action"]

        mappings = {
            "ACCEPT": "accept the PMU measurements without modification.",
            "DOWN_WEIGHT": "down-weight the PMU measurements in the WLS solver.",
            "PHASE_COMPENSATE": "apply phase compensation and lower trust in the PMU output.",
            "TIMING_CORRECTION": "apply timing correction and reduce the PMU confidence in state estimation.",
            "ISOLATE": "remove the faulty PMU measurements from the estimation pass.",
        }

        return mappings.get(action, "keep the PMU in the estimation set.")

    def build_controller_policy(self, fault_type: str, affected_pmu: str | int) -> Dict[str, object]:
        """Create a full control policy for a specific detected PMU fault."""
        decision = self.label_fault(fault_type, affected_pmu)
        weight_policy = {
            "PMU1": 1.0,
            "PMU2": 1.0,
            "PMU3": 1.0,
            str(affected_pmu).upper(): decision["weight"],
        }

        return {
            "fault_type": decision["fault_type"],
            "affected_pmu": str(affected_pmu).upper(),
            "action": decision["action"],
            "weight_policy": weight_policy,
            "reason": self.explain_action(fault_type, affected_pmu),
        }
