import random
import unittest

from agent import Agent, predicts_open
from protocol import run_rototest
from world import UnknownWorld, FEATURE_DOMAINS, all_objects


class WorldTests(unittest.TestCase):
    def test_force_feature_color(self):
        rng = random.Random(1)
        for _ in range(20):
            w = UnknownWorld.generate(rng, force_feature="color")
            self.assertEqual(w.hidden_rule[0], "color")
            self.assertIn(w.hidden_rule[1], FEATURE_DOMAINS["color"])

    def test_force_feature_shape(self):
        rng = random.Random(2)
        w = UnknownWorld.generate(rng, force_feature="shape")
        self.assertEqual(w.hidden_rule[0], "shape")

    def test_population_has_match_and_mismatch(self):
        rng = random.Random(3)
        w = UnknownWorld.generate(rng, force_feature="color")
        m = sum(1 for o in w.objects if w.matches(o))
        self.assertGreaterEqual(m, 1)
        self.assertLess(m, len(w.objects))


class SalienceTests(unittest.TestCase):
    def test_uniform_at_init(self):
        a = Agent()
        s = a.salience()
        self.assertAlmostEqual(s["color"], s["shape"])

    def test_update_only_on_convergence(self):
        a = Agent()
        rng = random.Random(7)
        w = UnknownWorld.generate(rng, force_feature="color")
        result = a.run_world(w)
        self.assertIsNotNone(result["rule"])
        self.assertGreater(a.alpha["color"], a.alpha["shape"])

    def test_reset_salience(self):
        a = Agent()
        a.alpha["color"] = 9.0
        a.reset_salience()
        self.assertEqual(a.alpha["color"], 1.0)
        self.assertAlmostEqual(a.salience()["color"], a.salience()["shape"])

    def test_no_update_without_surviving_rule(self):
        a = Agent()
        before = dict(a.alpha)
        a.hypotheses = []
        a.note_convergence()
        self.assertEqual(a.alpha, before)

    def test_shape_still_solvable_after_color_prior(self):
        a = Agent()
        a.alpha["color"] = 20.0
        a.alpha["shape"] = 1.0
        rng = random.Random(11)
        w = UnknownWorld.generate(rng, force_feature="shape")
        result = a.run_world(w)
        self.assertEqual(result["held_out"], 100.0)
        self.assertIsNotNone(result["rule"])
        self.assertEqual(result["rule"][0], "shape")

    def test_uniform_choice_deterministic(self):
        rng = random.Random(7)
        w = UnknownWorld.generate(rng, force_feature="color")
        a1, a2 = Agent(), Agent()
        a1.begin_world(w)
        a2.begin_world(w)
        self.assertEqual(a1.next_experiment(), a2.next_experiment())


class ProtocolTests(unittest.TestCase):
    def test_rototest_deterministic(self):
        r1 = run_rototest(seed=7)
        r2 = run_rototest(seed=7)
        self.assertEqual(r1["a_learn"], r2["a_learn"])
        self.assertEqual(r1["verdict"], r2["verdict"])

    def test_held_out_often_perfect(self):
        r = run_rototest(seed=7)
        perfect = sum(1 for row in r["rows"] if row["held_out"] == 100.0)
        self.assertGreaterEqual(perfect, len(r["rows"]) - 2)


if __name__ == "__main__":
    unittest.main()