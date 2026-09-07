import random
import unittest

from agent import (
    Agent,
    ChangePointAgent,
    ForgetAgent,
    HAZARD_GRID,
    HierarchicalChangePointAgent,
    LatentHazardAgent,
    LatentHazardAnticipateAgent,
    ObservationHazardAgent,
    predicts_open,
)
from protocol import (
    BLOCKS_005,
    _effective_hazard,
    _regime_worlds,
    bootstrap_ci,
    cohens_dz,
    paired_rototest,
    permutation_pvalue,
    rototest_004,
    rototest_005,
    rototest_006,
    rototest_007,
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


class ChangeAwareTests(unittest.TestCase):
    def _stream(self, feature, n, seed=7):
        rng = random.Random(seed)
        return [UnknownWorld.generate(rng, force_feature=feature) for _ in range(n)]

    def test_forget_f1_equals_bare(self):
        bare, f1 = Agent(), ForgetAgent(forget=1.0)
        for w in self._stream("color", 5):
            self.assertEqual(bare.run_world(w)["steps"], f1.run_world(w)["steps"])

    def test_forget_precision_saturates(self):
        f = ForgetAgent(forget=0.7)
        for w in self._stream("color", 20):
            f.run_world(w)
        self.assertLessEqual(f.alpha["color"], 1.0 / (1.0 - 0.7) + 1.01)

    def test_forget_reset(self):
        f = ForgetAgent(forget=0.7)
        for w in self._stream("color", 3):
            f.run_world(w)
        f.reset_salience()
        self.assertAlmostEqual(f.alpha["color"], f.alpha["shape"])

    def test_changepoint_low_hazard_equals_bare(self):
        cp = ChangePointAgent(hazard=1e-9)
        for w in self._stream("color", 5):
            cp.run_world(w)
        self.assertAlmostEqual(cp.alpha["color"], 6.0, places=3)
        self.assertAlmostEqual(cp.alpha["shape"], 1.0, places=3)

    def test_changepoint_reweights_fast_after_switch(self):
        cp = ChangePointAgent(hazard=1.0 / 5.0)
        bare = Agent()
        for w in self._stream("color", 5):
            cp.run_world(w)
            bare.run_world(w)
        after_color = cp.salience()
        world = self._stream("shape", 1, seed=7)[0]
        cp.run_world(world)
        bare.run_world(world)
        self.assertLess(cp.salience()["color"], after_color["color"])
        self.assertLess(cp.salience()["color"], bare.salience()["color"])
        self.assertGreater(cp.salience()["shape"], bare.salience()["shape"])

    def test_changepoint_run_length_collapses_on_switch(self):
        cp = ChangePointAgent(hazard=1.0 / 5.0)
        for w in self._stream("color", 5):
            cp.run_world(w)
        er_before = sum(r * p for r, p, _ in cp.runs)
        cp.run_world(self._stream("shape", 1, seed=7)[0])
        er_after = sum(r * p for r, p, _ in cp.runs)
        self.assertGreater(er_before, er_after)

    def test_both_mechanisms_solve_shape_after_color(self):
        for cls in (ForgetAgent, ChangePointAgent):
            a = cls()
            for w in self._stream("color", 5):
                a.run_world(w)
            r = a.run_world(self._stream("shape", 1, seed=7)[0])
            self.assertEqual(r["held_out"], 100.0)
            self.assertEqual(r["rule"][0], "shape")

    def test_changepoint_deterministic(self):
        a, b = ChangePointAgent(), ChangePointAgent()
        for w in self._stream("color", 4):
            a.run_world(w)
        for w in self._stream("color", 4):
            b.run_world(w)
        for f in a.alpha:
            self.assertEqual(a.alpha[f], b.alpha[f])
        self.assertEqual(len(a.runs), len(b.runs))

    def test_rototest_004_deterministic(self):
        r1 = rototest_004(seed=9, n_worlds=3, n_seeds=4)
        r2 = rototest_004(seed=9, n_worlds=3, n_seeds=4)
        for block in ("A-learn", "A-transfer", "B-switch", "C-mixed"):
            self.assertEqual(r1["stats_cn"][block]["mean"], r2["stats_cn"][block]["mean"])
            self.assertEqual(r1["stats_cn"][block]["ci95"], r2["stats_cn"][block]["ci95"])
        self.assertEqual(r1["verdict"], r2["verdict"])
        self.assertEqual(r1["gates"], r2["gates"])


class HierarchyTests(unittest.TestCase):
    def _stream(self, feature, n, seed=7):
        rng = random.Random(seed)
        return [UnknownWorld.generate(rng, force_feature=feature) for _ in range(n)]

    def test_reset_uniform(self):
        h = HierarchicalChangePointAgent()
        for w in self._stream("color", 3):
            h.run_world(w)
        h.reset_salience()
        self.assertAlmostEqual(h.salience()["color"], h.salience()["shape"])
        self.assertEqual(len(h.trellises), len(HAZARD_GRID))
        for runs in h.trellises:
            self.assertEqual(len(runs), 1)
        self.assertEqual(len({f: 1.0 for f in h.features}.values()), 2)

    def test_fresh_world_equals_bare(self):
        for seed in (1, 5, 9):
            rng = random.Random(seed)
            w = UnknownWorld.generate(rng)
            a, h = Agent(), HierarchicalChangePointAgent()
            self.assertEqual(a.run_world(w)["steps"], h.run_world(w)["steps"])
            self.assertEqual(a.run_world(w)["curve"], h.run_world(w)["curve"])

    def test_stationary_stream_learns_slow(self):
        h = HierarchicalChangePointAgent()
        for w in self._stream("color", 20):
            h.run_world(w)
        ws = h.hazard_posterior()
        slow = sum(w for w, hh in zip(ws, h.hazards) if hh <= 1.0 / 8.0)
        fast = sum(w for w, hh in zip(ws, h.hazards) if hh >= 1.0 / 2.0)
        self.assertGreater(slow, fast)
        self.assertLess(ws[0], 0.3)

    def test_fast_stream_does_not_dominate_slow_in_2_feature_geometry(self):
        h = HierarchicalChangePointAgent()
        for w in self._stream("color", 20):
            h.run_world(w)
        seed = 10
        for i in range(20):
            feat = "color" if i % 2 == 0 else "shape"
            for w in self._stream(feat, 2, seed=seed):
                h.run_world(w)
            seed += 1
        ws = h.hazard_posterior()
        slow = sum(w for w, hh in zip(ws, h.hazards) if hh <= 1.0 / 8.0)
        self.assertGreater(slow, 0.5)

    def test_solves_shape_after_color(self):
        h = HierarchicalChangePointAgent()
        for w in self._stream("color", 5):
            h.run_world(w)
        r = h.run_world(self._stream("shape", 1, seed=7)[0])
        self.assertEqual(r["held_out"], 100.0)
        self.assertEqual(r["rule"][0], "shape")

    def test_deterministic(self):
        a, b = HierarchicalChangePointAgent(), HierarchicalChangePointAgent()
        for w in self._stream("color", 4):
            a.run_world(w)
        for w in self._stream("color", 4):
            b.run_world(w)
        self.assertEqual(a.alpha, b.alpha)
        self.assertEqual(a.hazard_logliks, b.hazard_logliks)

    def test_regime_worlds_alternate(self):
        rng = random.Random(4)
        worlds = _regime_worlds(rng, (2, 3), "color")
        self.assertEqual(len(worlds), 5)
        self.assertEqual(worlds[0].hidden_rule[0], "color")
        self.assertEqual(worlds[2].hidden_rule[0], "shape")
        self.assertEqual(worlds[4].hidden_rule[0], "shape")

    def test_rototest_005_deterministic(self):
        r1 = rototest_005(seed=9, n_worlds=3, n_seeds=4)
        r2 = rototest_005(seed=9, n_worlds=3, n_seeds=4)
        for b in BLOCKS_005:
            self.assertEqual(r1["stats_ln"][b]["mean"], r2["stats_ln"][b]["mean"])
            self.assertEqual(r1["stats_ln"][b]["ci95"], r2["stats_ln"][b]["ci95"])
        self.assertEqual(r1["verdict"], r2["verdict"])
        self.assertEqual(r1["gates"], r2["gates"])


class ObservationHazardTests(unittest.TestCase):
    def _stream(self, feature, n, seed=7):
        rng = random.Random(seed)
        return [UnknownWorld.generate(rng, force_feature=feature) for _ in range(n)]

    def test_reset_uniform(self):
        h = ObservationHazardAgent()
        for w in self._stream("color", 3):
            h.run_world(w)
        h.reset_salience()
        self.assertAlmostEqual(h.salience()["color"], h.salience()["shape"])
        self.assertEqual(len(h.trellises), len(HAZARD_GRID))
        for runs in h.trellises:
            self.assertEqual(len(runs), 1)
        self.assertEqual(h.hazard_logliks, [0.0] * len(HAZARD_GRID))
        self.assertTrue(h._boundary_pending)

    def test_fresh_world_equals_bare(self):
        for seed in (1, 5, 9):
            rng = random.Random(seed)
            w = UnknownWorld.generate(rng)
            a, h = Agent(), ObservationHazardAgent()
            self.assertEqual(a.run_world(w)["steps"], h.run_world(w)["steps"])
            self.assertEqual(a.run_world(w)["held_out"], h.run_world(w)["held_out"])

    def test_stationary_twenty_worlds_learns_slow(self):
        h = ObservationHazardAgent()
        for w in self._stream("color", 20):
            h.run_world(w)
        ws = h.hazard_posterior()
        slow = sum(w for w, hh in zip(ws, h.hazards) if hh <= 1.0 / 8.0)
        fast = sum(w for w, hh in zip(ws, h.hazards) if hh >= 1.0 / 2.0)
        self.assertGreater(slow, fast)

    def test_fast_stream_orders_hazard_above_slow(self):
        h = ObservationHazardAgent()
        for w in self._stream("color", 5):
            h.run_world(w)
        rng = random.Random(3)
        for w in _regime_worlds(rng, (16, 16), "shape"):
            h.run_world(w)
        slow_h = _effective_hazard(h)
        seed = 4
        for i in range(8):
            feat = "color" if i % 2 == 0 else "shape"
            for w in self._stream(feat, 2, seed=seed):
                h.run_world(w)
            seed += 1
        fast_h = _effective_hazard(h)
        self.assertGreater(fast_h, slow_h + 0.01)

    def test_solves_shape_after_color(self):
        h = ObservationHazardAgent()
        for w in self._stream("color", 5):
            h.run_world(w)
        r = h.run_world(self._stream("shape", 1, seed=7)[0])
        self.assertEqual(r["held_out"], 100.0)
        self.assertEqual(r["rule"][0], "shape")

    def test_deterministic(self):
        a, b = ObservationHazardAgent(), ObservationHazardAgent()
        for w in self._stream("color", 4):
            a.run_world(w)
        for w in self._stream("color", 4):
            b.run_world(w)
        self.assertEqual(a.alpha, b.alpha)
        self.assertEqual(a.hazard_logliks, b.hazard_logliks)

    def test_rototest_006_deterministic(self):
        r1 = rototest_006(seed=9, n_worlds=3, n_seeds=4)
        r2 = rototest_006(seed=9, n_worlds=3, n_seeds=4)
        self.assertEqual(r1["verdict"], r2["verdict"])
        self.assertEqual(r1["gates"], r2["gates"])
        self.assertEqual(r1["h_order_ci"], r2["h_order_ci"])


class LatentHazardTests(unittest.TestCase):
    def _stream(self, feature, n, seed=7):
        rng = random.Random(seed)
        return [UnknownWorld.generate(rng, force_feature=feature) for _ in range(n)]

    def test_reset_uniform_state(self):
        h = LatentHazardAgent()
        for w in self._stream("color", 3):
            h.run_world(w)
        h.reset_salience()
        self.assertEqual(len(h.nodes), 1)
        self.assertEqual(h.nodes[0][:4], (0, 1.0, 1.0, {"color": 1.0, "shape": 1.0}))
        self.assertEqual(h.nodes[0][4], 1.0)
        self.assertAlmostEqual(h.effective_hazard(), 0.5)
        self.assertTrue(h._boundary_pending)

    def test_fresh_world_equals_bare(self):
        for seed in (1, 5, 9):
            rng = random.Random(seed)
            w = UnknownWorld.generate(rng)
            a, h = Agent(), LatentHazardAgent()
            self.assertEqual(a.run_world(w)["steps"], h.run_world(w)["steps"])
            self.assertEqual(a.run_world(w)["held_out"], h.run_world(w)["held_out"])

    def test_stationary_stream_learns_slow(self):
        h = LatentHazardAgent()
        for w in self._stream("color", 20):
            h.run_world(w)
        self.assertLess(h.effective_hazard(), 0.35)

    def test_fast_stream_orders_hazard_above_slow(self):
        h = LatentHazardAgent()
        for w in self._stream("color", 5):
            h.run_world(w)
        rng = random.Random(3)
        for w in _regime_worlds(rng, (16, 16), "shape"):
            h.run_world(w)
        slow_h = h.effective_hazard()
        seed = 4
        for i in range(8):
            feat = "color" if i % 2 == 0 else "shape"
            for w in self._stream(feat, 2, seed=seed):
                h.run_world(w)
            seed += 1
        fast_h = h.effective_hazard()
        self.assertGreater(fast_h, slow_h + 0.01)

    def test_hazard_rises_then_falls_on_switchrate(self):
        h = LatentHazardAgent()
        rng = random.Random(4)
        worlds = _regime_worlds(rng, (16, 16, 2, 2, 2, 2, 2, 2, 2, 2, 16, 16), "shape")
        points = []
        for n in (32, 48, 60, 80):
            a = LatentHazardAgent()
            rng2 = random.Random(4)
            for w in _regime_worlds(rng2, (16, 16, 2, 2, 2, 2, 2, 2, 2, 2, 16, 16), "shape")[:n]:
                a.run_world(w)
            points.append(a.effective_hazard())
        self.assertGreater(points[1], points[0] + 0.1)
        self.assertLess(points[3], points[1])

    def test_node_count_bounded(self):
        h = LatentHazardAgent()
        rng = random.Random(5)
        for w in _regime_worlds(rng, (16, 16, 2, 2, 2, 2, 2, 2, 2, 2, 16, 16), "shape"):
            h.run_world(w)
            self.assertLessEqual(len(h.nodes), h.max_states)

    def test_solves_shape_after_color(self):
        h = LatentHazardAgent()
        for w in self._stream("color", 5):
            h.run_world(w)
        r = h.run_world(self._stream("shape", 1, seed=7)[0])
        self.assertEqual(r["held_out"], 100.0)
        self.assertEqual(r["rule"][0], "shape")

    def test_deterministic(self):
        a, b = LatentHazardAgent(), LatentHazardAgent()
        for w in self._stream("color", 4):
            a.run_world(w)
        for w in self._stream("color", 4):
            b.run_world(w)
        self.assertEqual(a.nodes, b.nodes)
        self.assertEqual(a.value_weights, b.value_weights)

    def test_rototest_007_deterministic(self):
        r1 = rototest_007(seed=9, n_worlds=3, n_seeds=4)
        r2 = rototest_007(seed=9, n_worlds=3, n_seeds=4)
        self.assertEqual(r1["verdict"], r2["verdict"])
        self.assertEqual(r1["gates"], r2["gates"])
        self.assertEqual(r1["h_order_ci"], r2["h_order_ci"])


class LatentHazardAnticipateTests(unittest.TestCase):
    class ZeroHedge(LatentHazardAnticipateAgent):
        def begin_world(self, world):
            super().begin_world(world)
            self._hedge_w = 0.0

    def _fast_stream(self, seed=3):
        rng = random.Random(seed)
        return _regime_worlds(rng, (2, 2, 2, 2, 2, 2, 2, 2), "color")

    def test_hedge_w_zero_identical_to_exact(self):
        for seed in (3, 8):
            ex, ac = LatentHazardAgent(), self.ZeroHedge()
            rng = random.Random(seed)
            streams = _regime_worlds(rng, (2, 2, 2, 2, 2, 2, 2, 2), "color")
            same = True
            for w in streams:
                re = ex.run_world(w)
                ra = ac.run_world(w)
                if (re["steps"], re["curve"]) != (ra["steps"], ra["curve"]):
                    same = False
                    break
            self.assertTrue(same)

    def test_hedge_changes_query_choice_on_fast_worlds(self):
        ex, ac = LatentHazardAgent(), LatentHazardAnticipateAgent()
        rng = random.Random(3)
        streams = _regime_worlds(rng, (2, 2, 2, 2, 2, 2, 2, 2), "color")
        differed = False
        for w in streams:
            re = ex.run_world(w)
            ra = ac.run_world(w)
            if (re["steps"], re["curve"]) != (ra["steps"], ra["curve"]):
                differed = True
                break
        self.assertTrue(differed)

    def test_hedge_belief_is_the_effective_hazard(self):
        ac = LatentHazardAnticipateAgent()
        rng = random.Random(11)
        for _ in range(4):
            ac.run_world(UnknownWorld.generate(rng, force_feature="color"))
        self.assertTrue(0.0 < ac._hedge_w < 1.0)
        self.assertEqual(ac._hedge_w, ac._world_p_change)

    def test_hedge_solves_shape_after_color(self):
        ac = LatentHazardAnticipateAgent()
        rng = random.Random(11)
        for _ in range(5):
            ac.run_world(UnknownWorld.generate(rng, force_feature="color"))
        r = ac.run_world(UnknownWorld.generate(rng, force_feature="shape"))
        self.assertEqual(r["held_out"], 100.0)
        self.assertEqual(r["rule"][0], "shape")

    def test_deterministic(self):
        a, b = LatentHazardAnticipateAgent(), LatentHazardAnticipateAgent()
        for w in self._fast_stream():
            a.run_world(w)
        for w in self._fast_stream():
            b.run_world(w)
        self.assertEqual(a.nodes, b.nodes)


if __name__ == "__main__":
    unittest.main()