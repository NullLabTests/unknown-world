"""Experiment driver: 003 (paired rototest v2) by default; `python3 run.py 002` reproduces the legacy run."""
import sys

from protocol import (
    paired_rototest,
    print_paired_report,
    print_report,
    run_rototest,
)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "002":
        result = run_rototest(seed=7)
        print("The Unknown World — Experiment 002 (feature-salience prior)\n")
        print_report(result)
        return
    result = paired_rototest(seed=7, n_worlds=5, n_seeds=16)
    print(
        "The Unknown World — Experiment 003 "
        "(measurement hardening: paired seeded rototest v2)\n"
    )
    print_paired_report(result)


if __name__ == "__main__":
    main()