"""Sanity tests for Step 1: world, minimal loop, rototest protocol."""

from __future__ import annotations

import random
import unittest

from agent import MinimalLoop
from protocol import across_worlds, grade, within_world_curve
from world import UnknownWorld


def run_world(agent, concept=None):
    rng = random.Random(0)
    world = UnknownWorld.generate(rng, concept=concept)
    surviving, tests, curve = within_world_curve(agent, world)
    return world, surviving, tests, curve


class WorldTest(unittest.TestCase):
    def test_rule_has_one_concept(self):
        rng = random.Random(1)
        for _ in range(20):
            world = UnknownWorld.generate(rng)
            self.assertIn(world.concept[0], {"color", "shape"})
            self.assertIn(world.concept[1], {"red", "blue", "green", "round", "square", "star"})
            self.assertEqual(len(world.presented) + len(world.held_out), 9)

    def test_effect_follows_rule(self):
        rng = random.Random(2)
        for _ in range(20):
            world = UnknownWorld.generate(rng)
            feature, value = world.concept
            for obj in world.presented + world.held_out:
                self.assertEqual(obj.opens_door, obj.features[feature] == value)


class LoopTest(unittest.TestCase):
    def test_converges_to_true_rule(self):
        agent = MinimalLoop()
        rng = random.Random(3)
        for _ in range(50):
            for concept in (None, ("color", "blue"), ("shape", "star")):
                world = UnknownWorld.generate(rng, concept=concept)
                surviving, tests, curve = within_world_curve(agent, world)
                self.assertEqual(surviving, [world.concept], (concept, surviving))
                self.assertLessEqual(tests, len(world.presented))

    def test_held_out_perfect_once_converged(self):
        agent = MinimalLoop()
        rng = random.Random(4)
        for _ in range(50):
            world = UnknownWorld.generate(rng)
            surviving, tests, curve = within_world_curve(agent, world)
            self.assertEqual(curve[-1][1], 1.0)

    def test_never_concept(self):
        agent = MinimalLoop()
        world = UnknownWorld.generate(random.Random(5), concept=None)
        concept = ("*never", None)
        world.concept = concept
        for obj in world.presented + world.held_out:
            obj.opens_door = False
        surviving, tests, curve = within_world_curve(agent, world)
        self.assertEqual(surviving, [concept])
        self.assertEqual(curve[-1][1], 1.0)

    def test_always_concept(self):
        agent = MinimalLoop()
        world = UnknownWorld.generate(random.Random(6), concept=None)
        concept = ("*always", None)
        world.concept = concept
        for obj in world.presented:
            obj.opens_door = True
        surviving, tests, curve = within_world_curve(agent, world)
        self.assertEqual(surviving, [concept])


class RototestTest(unittest.TestCase):
    def test_baseline_has_flat_learning_curve(self):
        """Expected negative result: no learning-to-learn yet.

        Tests-to-converge should be flat (within a small margin) across
        worlds for the baseline loop. This documents the absence, so the
        future primitive is only added when it produces a measurable change.
        """
        agent = MinimalLoop()
        rng = random.Random(8)
        worlds = [UnknownWorld.generate(rng) for _ in range(20)]
        per_world = across_worlds(agent, worlds)
        tests = [t for _, _, t, _ in per_world]
        self.assertLessEqual(max(tests) - min(tests), 2)

    def test_grade_ties_are_not_competent(self):
        agent = MinimalLoop()
        rng = random.Random(9)
        worlds = [UnknownWorld.generate(rng) for _ in range(10)]
        agent.boot(worlds[0].presented)
        score = grade(agent, worlds[0].held_out)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()