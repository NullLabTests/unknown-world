import random
import unittest

from agent import Agent, predicts_open
from protocol import (
    bootstrap_ci,
    cohens_dz,
    paired_rototest,
    permutation_pvalue,
    run_null_sequence,
    run_prior_sequence,
    run_rototest,
)
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


class StatsTests(unittest.TestCase):
    def test_bootstrap_ci_contains_mean(self):
        rng = random.Random(21)
        diffs = [rng.uniform(-1.0, 1.0) for _ in range(12)]
        lo, hi, m = bootstrap_ci(diffs, seed=5)
        self.assertLessEqual(lo, m)
        self.assertGreaterEqual(hi, m)

    def test_bootstrap_ci_deterministic(self):
        rng = random.Random(22)
        diffs = [rng.uniform(-1.0, 1.0) for _ in range(10)]
        self.assertEqual(bootstrap_ci(diffs, seed=5), bootstrap_ci(diffs, seed=5))

    def test_permutation_detects_consistent_reduction(self):
        diffs = [-2.0, -1.0, -2.0, -1.5, -2.0, -1.0, -2.0, -1.5]
        self.assertLess(permutation_pvalue(diffs, seed=5, alternative="less"), 0.05)

    def test_permutation_neutral_diffs(self):
        diffs = [0.0] * 8
        p = permutation_pvalue(diffs, seed=5, alternative="less")
        self.assertGreaterEqual(p, 0.49)

    def test_permutation_greater_opposite_tail(self):
        diffs = [2.0, 1.0, 2.0, 1.5, 2.0, 1.0, 2.0, 1.5]
        self.assertLess(permutation_pvalue(diffs, seed=5, alternative="greater"), 0.05)

    def test_permutation_deterministic(self):
        diffs = [-1.0, 0.5, -0.5, 1.0, -2.0, 0.0, -1.0, 0.25]
        self.assertEqual(
            permutation_pvalue(diffs, seed=9),
            permutation_pvalue(diffs, seed=9),
        )

    def test_cohens_dz_sign(self):
        self.assertLess(cohens_dz([-2.0, -1.0, -1.5, -1.0]), 0.0)
        self.assertIsNone(cohens_dz([1.0, 1.0, 1.0, 1.0]))


class PairedV2Tests(unittest.TestCase):
    def test_prior_sequence_accrues(self):
        a = Agent()
        rng = random.Random(13)
        worlds = [
            UnknownWorld.generate(rng, force_feature="color"),
            UnknownWorld.generate(rng, force_feature="color"),
        ]
        run_prior_sequence(a, worlds)
        self.assertGreater(a.alpha["color"], a.alpha["shape"])

    def test_null_sequence_stays_uniform(self):
        a = Agent()
        rng = random.Random(14)
        w1 = UnknownWorld.generate(rng, force_feature="color")
        w2 = UnknownWorld.generate(rng, force_feature="color")
        run_null_sequence(a, [w1, w2])
        b = Agent()
        run_null_sequence(b, [w2])
        self.assertEqual(a.alpha, b.alpha)

    def test_null_sequence_deterministic_like_prior(self):
        rng1, rng2 = random.Random(15), random.Random(15)
        w1 = UnknownWorld.generate(rng1, force_feature="color")
        w2 = UnknownWorld.generate(rng2, force_feature="color")
        self.assertEqual(
            run_null_sequence(Agent(), [w1])[0]["steps"],
            run_null_sequence(Agent(), [w2])[0]["steps"],
        )

    def test_paired_rototest_deterministic(self):
        r1 = paired_rototest(seed=9, n_worlds=3, n_seeds=5)
        r2 = paired_rototest(seed=9, n_worlds=3, n_seeds=5)
        for block in ("A-learn", "A-transfer", "B-switch", "C-mixed"):
            self.assertEqual(r1["blocks"][block]["mean"], r2["blocks"][block]["mean"])
            self.assertEqual(r1["blocks"][block]["ci95"], r2["blocks"][block]["ci95"])
        self.assertEqual(r1["verdict"], r2["verdict"])

    def test_every_world_converges_in_paired_run(self):
        r = paired_rototest(seed=9, n_worlds=2, n_seeds=3)
        for block in ("A-learn", "A-transfer", "B-switch", "C-mixed"):
            self.assertEqual(r["blocks"][block]["n"], r["blocks"][block]["n"])


if __name__ == "__main__":
    unittest.main()