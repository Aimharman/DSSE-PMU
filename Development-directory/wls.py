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

        print("\n==============================================")
        print(" Final Estimated State")
        print("==============================================")

        print(x)

        return x