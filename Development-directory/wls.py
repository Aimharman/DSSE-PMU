"""
===========================================================
wls.py

Weighted Least Squares (WLS)

Stage 3.4

Performs one Gauss-Newton WLS iteration.

Implements

    r = z - h(x)

    G = HᵀWH

    g = HᵀWr

===========================================================
"""
"""
===========================================================
wls.py

Weighted Least Squares Solver

Implements the complete iterative
Gauss-Newton State Estimation.

===========================================================
"""

import numpy as np

from measurement_model import measurement_model
from jacobian import compute_jacobian
from network_model import NUM_BUSES

from measurement_model import (
    measurement_model,
    state_to_voltage,
    compute_currents
)

max_iterations = 250
tolerance = 1e-6

class WeightedLeastSquares:

    def __init__(self,
                 tolerance = tolerance,
                 max_iterations = max_iterations):

        self.tolerance = tolerance
        self.max_iterations = max_iterations

    ########################################################
    # Solve WLS
    ########################################################

    def solve(self, z, x0):

        x = x0.copy()

        ####################################################
        # Covariance Matrix
        ####################################################

        #R = np.eye(len(z))
        R = np.diag([
            1e-4, 1e-4, 1e-2, 1e-2,
            1e-4, 1e-4, 1e-2, 1e-2,
            1e-4, 1e-4, 1e-2, 1e-2
        ])
        ####################################################
        # Weight Matrix
        ####################################################

        W = np.linalg.inv(R)

        print("\n==============================================")
        print(" Weighted Least Squares")
        print("==============================================")

        for iteration in range(self.max_iterations):

            print(f"\nIteration {iteration+1}")

            ################################################
            # Measurement Prediction
            ################################################

            h = measurement_model(x)

            ################################################
            # Residual
            ################################################

            r = z - h

            ################################################
            # Jacobian
            ################################################

            H = compute_jacobian(x)

            ################################################
            # Gain Matrix
            ################################################

            G = H.T @ W @ H

            ################################################
            # Gradient
            ################################################

            g = H.T @ W @ r

            ################################################
            # State Correction
            ################################################

            try:

                dx = np.linalg.solve(G, g)

            except np.linalg.LinAlgError:

                print("Gain matrix is singular.")

                return x

            ################################################
            # Update State
            ################################################

            x = x + dx

            ################################################
            # Display
            ################################################

            print("\nResidual Norm")

            print(np.linalg.norm(r))

            print("\nCorrection Norm")

            print(np.linalg.norm(dx))

            ################################################
            # Convergence
            ################################################

            if np.linalg.norm(dx) < self.tolerance:

                print("\nConverged.")

                break

        ########################################################
        # Final Consistency Check
        ########################################################

        V = state_to_voltage(x)

        I = compute_currents(V)

        print("\n==============================================")
        print(" Current Measurement Consistency")
        print("==============================================")

        for bus in range(NUM_BUSES):

            idx = 4 * bus

            measured_mag = z[idx + 2]
            measured_ang = np.degrees(z[idx + 3])

            predicted_mag = np.abs(I[bus])
            predicted_ang = np.degrees(np.angle(I[bus]))

            print(f"\nBus {bus+1}")

            print(f"Measured Current Magnitude  : {measured_mag:.6f}")
            print(f"Predicted Current Magnitude : {predicted_mag:.6f}")

            print(f"Measured Current Angle      : {measured_ang:.6f} deg")
            print(f"Predicted Current Angle     : {predicted_ang:.6f} deg")
        print("\n==============================================")
        print(" Final Estimated State")
        print("==============================================")

        print(x)

        return x