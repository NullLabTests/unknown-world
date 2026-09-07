"""Experiment driver: 007 (the closed loop) by default; `run.py 002/003/004/005/006` reproduce legacy runs."""
import sys

from protocol import (
    paired_rototest,
    print_004_report,
    print_005_report,
    print_006_report,
    print_007_report,
    print_paired_report,
    print_report,
    rototest_004,
    rototest_005,
    rototest_006,
    rototest_007,
    run_rototest,
)


def main():
    exp = sys.argv[1] if len(sys.argv) > 1 else "007"
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
    if exp == "004":
        result = rototest_004(seed=7, n_worlds=5, n_seeds=16)
        print(
            "The Unknown World — Experiment 004 "
            "(change-aware salience: the prior can be wrong)\n"
        )
        print_004_report(result)
        return
    if exp == "005":
        result = rototest_005(seed=7, n_worlds=5, n_seeds=16)
        print(
            "The Unknown World — Experiment 005 "
            "(the prior earns its hazard rate)\n"
        )
        print_005_report(result)
        return
    if exp == "006":
        result = rototest_006(seed=7, n_worlds=5, n_seeds=16)
        print(
            "The Unknown World — Experiment 006 "
            "(the hazard learned per observation)\n"
        )
        print_006_report(result)
        return
    result = rototest_007(seed=7, n_worlds=5, n_seeds=16)
    print(
        "The Unknown World — Experiment 007 "
        "(the closed loop: a hazard that steers what the agent asks)\n"
    )
    print_007_report(result)


if __name__ == "__main__":
    main()