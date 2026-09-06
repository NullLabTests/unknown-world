"""Rototest protocol — measure adaptation, never a fixed benchmark.

Within-world competence: answer accuracy on held-out (novel) objects as
the hypothesis set narrows — the 40% -> 70% -> 90% curve.

Across-world competence: experiments needed to see through each new world,
world after world — the learning-to-learn signal.
"""

from __future__ import annotations


def grade(agent, objects):
    if not objects:
        return None
    correct = sum(
        1 for o in objects if agent.predicts_open(o) == o.opens_door
    )
    return correct / len(objects)


def within_world_curve(agent, world):
    """Run the loop to convergence; return (surviving, tests_used, curve)."""
    agent.reset()
    agent.boot(world.presented)
    curve = [(0, grade(agent, world.held_out))]
    tests = 0
    while not agent.converged and agent.untested:
        obj = agent.next_experiment()
        if obj is None:
            break
        agent.commit(obj, world.experiment(obj))
        tests += 1
        curve.append((tests, grade(agent, world.held_out)))
    return agent.surviving, tests, curve


def across_worlds(agent, worlds):
    """tests-to-converge per world in sequence (fresh agent each world)."""
    per_world = []
    for w in worlds:
        surviving, tests, curve = within_world_curve(agent, w)
        per_world.append((w, surviving, tests, curve))
    return per_world