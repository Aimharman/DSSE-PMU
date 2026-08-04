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

    print("\nStage 3.1 Complete")


###########################################################################

if __name__ == "__main__":
    main()