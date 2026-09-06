"""Experiment 002 driver."""
from protocol import run_rototest, print_report


def main():
    result = run_rototest(seed=7)
    print("The Unknown World — Experiment 002 (feature-salience prior)\n")
    print_report(result)


if __name__ == "__main__":
    main()