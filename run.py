"""Experiment driver: 004 (change-aware salience) by default; `run.py 002`/`run.py 003` reproduce legacy runs."""
import sys

from protocol import (
    paired_rototest,
    print_004_report,
    print_paired_report,
    print_report,
    rototest_004,
    run_rototest,
)


def main():
    exp = sys.argv[1] if len(sys.argv) > 1 else "004"
    if exp == "002":
        print("The Unknown World — Experiment 002 (feature-salience prior)\n")
        print_report(run_rototest(seed=7))
        return
    if exp == "003":
        print(
            "The Unknown World — Experiment 003 "
            "(measurement hardening: paired seeded rototest v2)\n"
        )
        print_paired_report(paired_rototest(seed=7, n_worlds=5, n_seeds=16))
        return
    result = rototest_004(seed=7, n_worlds=5, n_seeds=16)
    print(
        "The Unknown World — Experiment 004 "
        "(change-aware salience: the prior can be wrong)\n"
    )
    print_004_report(result)


if __name__ == "__main__":
    main()