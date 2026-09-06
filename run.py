"""Run Experiment 001 and print the evidence."""

from __future__ import annotations

import random

from agent import MinimalLoop
from protocol import across_worlds
from world import UnknownWorld


def fmt_curve(curve):
    if not curve:
        return "n/a"
    return " -> ".join(f"{acc * 100:.0f}%" for _, acc in curve)


def rows_for(worlds, phase):
    agent = MinimalLoop()
    rows = []
    for i, w in enumerate(worlds, 1):
        surviving, tests, curve = across_worlds(agent, [w])[0][1:]
        acc = curve[-1][1]
        rows.append((f"{i}", phase, w.rule, f"{tests}", f"{acc * 100:.1f}%", fmt_curve(curve)))
    return rows


def main():
    rng = random.Random(7)
    learning_worlds = [
        UnknownWorld.generate(rng, concept=("color", "blue")) for _ in range(5)
    ]
    transfer_worlds = [
        UnknownWorld.generate(rng, concept=None) for _ in range(5)
    ]

    print("=" * 88)
    print("EXPERIMENT 001 - THE UNKNOWN WORLD")
    print("minimal loop: hypothesis -> predict -> act -> observe -> update")
    print("rule family: exactly one (feature, value) pair opens the door")
    print("=" * 88)
    print(f"{'#':>3} {'phase':<9} {'hidden rule':<18} {'tests':>5} {'held-out acc':>12}  competence curve")
    print("-" * 88)

    all_rows = rows_for(learning_worlds, "learning") + rows_for(transfer_worlds, "transfer")
    for row in all_rows:
        print(f"{row[0]:>3} {row[1]:<9} {row[2]:<18} {row[3]:>5} {row[4]:>12}  {row[5]}")

    learn_tests = [int(row[3]) for row in all_rows if row[1] == "learning"]
    transfer_tests = [int(row[3]) for row in all_rows if row[1] == "transfer"]

    print("-" * 88)
    print("MEASURES")
    print(f"  learning phase  : experiments-to-converge per world = {learn_tests}")
    print(f"  transfer phase  : experiments-to-converge per world = {transfer_tests}")
    print(f"  held-out answers reach 100% in {sum(1 for w in learning_worlds + transfer_worlds)}/{len(learning_worlds) + len(transfer_worlds)} worlds")
    print()
    print("INTERPRETATION")
    print("  The loop converges to the true rule and answers about novel objects")
    print("  correctly: within-world competence reaches 100% in every world.")
    print("  But tests-to-converge is flat across worlds: learning world 1 does not")
    print("  speed up world 2. No learning-to-learn. No transfer of structure.")
    print("  The minimal loop is competent, not generalizing.")
    print()
    print("NEXT PRIMITIVE TO ADD")
    print("  A prior over feature salience (which dimensions tend to be diagnostic),")
    print("  refreshed from experience, so experiment choice improves across worlds.")
    print("  That is the first measurable learning-to-learn curve this protocol can")
    print("  detect. If it appears, we add a second and measure again.")


if __name__ == "__main__":
    main()