"""
===========================================================
chi_square.py

Chi-Square Bad Data Detection
Stage 3.8 + PMU-level fault localization

The detector has two levels:

1. Global chi-square test:
       J = r^T W r

2. PMU-level screening:
       J_PMU = sum(r_i^2 W_ii)

A PMU is considered a fault candidate only when:
    - the global statistic exceeds the global threshold,
    - one PMU contributes enough normalized residual energy,
    - that PMU contributes a significant fraction of total residual energy.

The second stage prevents ordinary distributed model/noise residuals
from being immediately interpreted as a faulty PMU.
===========================================================
"""

import re
import numpy as np
from scipy.stats import chi2


class ChiSquareDetector:

    def __init__(
        self,
        confidence=0.95,
        pmu_measurement_count=4,
        pmu_share_threshold=0.55,
    ):
        self.confidence = confidence
        self.pmu_measurement_count = pmu_measurement_count
        self.pmu_share_threshold = pmu_share_threshold

    # ---------------------------------------------------------
    # Global statistic
    # ---------------------------------------------------------

    def compute_statistic(self, residual, W):
        residual = np.asarray(residual, dtype=float).reshape(-1)

        if W.ndim == 2:
            diagonal = np.diag(W)
        else:
            diagonal = np.asarray(W, dtype=float).reshape(-1)

        return float(np.sum((residual ** 2) * diagonal))

    # ---------------------------------------------------------
    # Degrees of freedom
    # ---------------------------------------------------------

    def degrees_of_freedom(self, num_measurements, num_states):
        return int(num_measurements - num_states)

    # ---------------------------------------------------------
    # Chi-square threshold
    # ---------------------------------------------------------

    def threshold(self, dof):
        return float(chi2.ppf(self.confidence, dof))

    # ---------------------------------------------------------
    # PMU residual-energy decomposition
    # ---------------------------------------------------------

    def pmu_contributions(self, residual, W, measurement_names=None):
        """
        Return normalized residual energy grouped by PMU.

        For each measurement:
            e_i = r_i^2 W_ii

        For each PMU:
            E_PMU = sum(e_i)

        The PMU share is:
            E_PMU / J
        """

        residual = np.asarray(residual, dtype=float).reshape(-1)

        if W.ndim == 2:
            diagonal = np.diag(W)
        else:
            diagonal = np.asarray(W, dtype=float).reshape(-1)

        diagonal = np.maximum(np.abs(diagonal), 1e-12)

        if measurement_names is None:
            measurement_names = [
                f"measurement_{i + 1}"
                for i in range(len(residual))
            ]

        totals = {}
        indices = {}

        for idx, name in enumerate(measurement_names):
            match = re.search(
                r"PMU\s*\d+",
                str(name),
                flags=re.IGNORECASE,
            )

            if match:
                pmu = match.group(0).upper().replace(" ", "")
            else:
                pmu = str(name).split()[0]

            totals.setdefault(pmu, 0.0)
            indices.setdefault(pmu, [])

            contribution = float(
                residual[idx] ** 2 * diagonal[idx]
            )

            totals[pmu] += contribution
            indices[pmu].append(idx)

        J = float(sum(totals.values()))

        ranked = []

        for pmu, energy in totals.items():
            ranked.append({
                "pmu": pmu,
                "energy": float(energy),
                "share": (
                    float(energy / J)
                    if J > 0.0
                    else 0.0
                ),
                "indices": indices[pmu],
            })

        ranked.sort(
            key=lambda item: item["energy"],
            reverse=True,
        )

        return ranked

    # ---------------------------------------------------------
    # PMU fault candidate screening
    # ---------------------------------------------------------

    def screen_fault_candidate(
        self,
        residual,
        W,
        global_J,
        global_threshold,
        measurement_names=None,
    ):
        """
        Decide whether a global chi-square failure has a
        concentrated PMU signature.

        The PMU-level reference uses the 95% chi-square
        threshold for 4 measurements.

        This is deliberately a screening stage, not a final
        isolation decision. The final decision is made only
        after re-estimation with the candidate PMU isolated.
        """

        pmu_data = self.pmu_contributions(
            residual,
            W,
            measurement_names,
        )

        if not pmu_data:
            return {
                "candidate": False,
                "pmu": "",
                "energy": 0.0,
                "share": 0.0,
                "threshold": 0.0,
                "pmu_data": [],
            }

        top = pmu_data[0]

        pmu_threshold = float(
            chi2.ppf(
                self.confidence,
                self.pmu_measurement_count,
            )
        )

        candidate = (
            global_J > global_threshold
            and top["energy"] > pmu_threshold
            and top["share"] >= self.pmu_share_threshold
        )

        return {
            "candidate": bool(candidate),
            "pmu": top["pmu"],
            "energy": float(top["energy"]),
            "share": float(top["share"]),
            "threshold": pmu_threshold,
            "pmu_data": pmu_data,
        }

    # ---------------------------------------------------------
    # PMU localization
    # ---------------------------------------------------------

    def localize_faulty_pmu(
        self,
        residual,
        W,
        measurement_names=None,
    ):
        """
        Rank PMUs by normalized residual energy.

        This replaces the previous sum of absolute normalized
        residuals. Since the global chi-square statistic itself is
        an energy quantity, PMU-level energy is the consistent metric.
        """

        data = self.pmu_contributions(
            residual,
            W,
            measurement_names,
        )

        if not data:
            return "", [], 0.0

        top = data[0]

        return (
            top["pmu"],
            top["indices"],
            top["energy"],
        )

    # ---------------------------------------------------------
    # Legacy single-measurement localization
    # ---------------------------------------------------------

    def localize_bad_data(
        self,
        residual,
        W,
        measurement_names=None,
    ):
        residual = np.asarray(
            residual,
            dtype=float,
        ).reshape(-1)

        if W.ndim == 2:
            diagonal = np.diag(W)
        else:
            diagonal = np.asarray(W).reshape(-1)

        diagonal = np.maximum(
            np.abs(diagonal),
            1e-12,
        )

        scores = np.abs(residual) * np.sqrt(diagonal)

        index = int(np.argmax(scores))

        label = (
            measurement_names[index]
            if measurement_names is not None
            and len(measurement_names) > index
            else f"measurement_{index + 1}"
        )

        return (
            index,
            label,
            float(scores[index]),
        )

    # ---------------------------------------------------------
    # Rolling PMU diagnostic retained for compatibility
    # ---------------------------------------------------------

    def detect_faulty_pmu(
        self,
        residual_history,
        pmu_names=None,
        window_size=3,
        threshold=0.5,
    ):
        residual_history = np.asarray(
            residual_history,
            dtype=float,
        )

        if residual_history.ndim == 1:
            residual_history = residual_history.reshape(1, -1)

        if residual_history.shape[1] % 4 != 0:
            raise ValueError(
                "Residual history length must be a multiple "
                "of 4 for PMU groups."
            )

        if pmu_names is None:
            pmu_names = [
                f"PMU{i + 1}"
                for i in range(
                    residual_history.shape[1] // 4
                )
            ]

        results = []

        for pmu_idx, pmu_name in enumerate(pmu_names):
            block = residual_history[
                :,
                4 * pmu_idx:4 * pmu_idx + 4,
            ]

            energy = np.linalg.norm(
                block,
                axis=1,
            )

            if len(energy) >= window_size:
                rolling = np.array([
                    np.mean(
                        energy[
                            max(0, t - window_size + 1):
                            t + 1
                        ]
                    )
                    for t in range(len(energy))
                ])
            else:
                rolling = np.full(
                    len(energy),
                    np.mean(energy),
                )

            results.append({
                "pmu": pmu_name,
                "score": float(np.mean(rolling)),
                "window_score": float(np.max(rolling)),
                "mean_energy": float(np.mean(energy)),
            })

        return [
            item
            for item in sorted(
                results,
                key=lambda x: x["score"],
                reverse=True,
            )
            if item["score"] >= threshold
        ]

    # ---------------------------------------------------------
    # Global detection
    # ---------------------------------------------------------

    def detect(
        self,
        residual,
        W,
        num_measurements,
        num_states,
        verbose=True,
    ):
        J = self.compute_statistic(
            residual,
            W,
        )

        dof = self.degrees_of_freedom(
            num_measurements,
            num_states,
        )

        threshold = self.threshold(dof)

        bad_data = bool(J > threshold)

        if verbose:
            print("\n==============================================")
            print(" Chi-Square Bad Data Detection")
            print("==============================================")
            print(f"\nDegrees of Freedom : {dof}")
            print(
                f"Confidence Level   : "
                f"{self.confidence * 100:.1f}%"
            )
            print("\nChi-Square Statistic (J)")
            print(J)
            print("\nCritical Threshold")
            print(threshold)

            if bad_data:
                print("\nResult")
                print("✗ Measurement set FAILED")
                print("Bad data detected.")
            else:
                print("\nResult")
                print("✓ Measurement set PASSED")
                print("No bad data detected.")

            print("==============================================")

        return (
            bad_data,
            J,
            threshold,
        )
