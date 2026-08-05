"""
===========================================================
main.py

Project Entry Point

Workflow

PMU_Output.csv
        │
        ▼
State Estimator
        │
        ▼
Measurement Vector (z)
        │
        ▼
Initial State Vector (x0)
===========================================================
"""

from state_estimator import StateEstimator
from wls import WeightedLeastSquares

###########################################################################
# CONFIGURATION
###########################################################################

CSV_FILE = "PMU_Output.csv"

###########################################################################
# MAIN
###########################################################################

def main():

    print("==============================================")
    print(" Distribution System State Estimator")
    print("==============================================")

    estimator = StateEstimator(CSV_FILE)

    estimator.run()

    from jacobian import compute_jacobian

    H = compute_jacobian(estimator.x)

    print("\nJacobian Matrix\n")
    print(H)

    print("\nShape :", H.shape)

    print("\nStage 3.4 Complete")

    solver = WeightedLeastSquares()

    x_final = solver.solve(

        estimator.z,
        estimator.x

    )

    print("\nEstimated State")

    print(x_final)


###########################################################################

if __name__ == "__main__":
    main()