"""Version-space loop + optional feature-salience prior (Experiment 002)."""
from __future__ import annotations

import math

from world import FEATURES, all_atomic_rules

ALWAYS = ("*", "always")
NEVER = ("*", "never")


def predicts_open(rule, obj):
    if rule == ALWAYS:
        return True
    if rule == NEVER:
        return False
    feature, value = rule
    return getattr(obj, feature) == value


class Agent:
    def __init__(self, features=None):
        self.features = tuple(features) if features is not None else FEATURES
        self.alpha = {f: 1.0 for f in self.features}
        self.hypotheses = []
        self.untested = []
        self.tested = []
        self.history = []  # (obj, opened)

    def reset_salience(self):
        self.alpha = {f: 1.0 for f in self.features}

    def salience(self):
        total = sum(self.alpha.values()) or 1.0
        return {f: self.alpha[f] / total for f in self.features}

    def begin_world(self, world):
        self.hypotheses = all_atomic_rules() + [ALWAYS, NEVER]
        self.untested = list(world.objects)
        self.tested = []
        self.history = []

    def _mass(self, rule):
        sal = self.salience()
        if rule[0] in sal:
            return sal[rule[0]]
        return sum(sal.values()) / len(sal) if sal else 1.0

    def _normalized_masses(self):
        raw = [self._mass(h) for h in self.hypotheses]
        s = sum(raw) or 1.0
        return [x / s for x in raw]

    def _weighted_ig(self, obj):
        masses = self._normalized_masses()
        on_mass = 0.0
        off_mass = 0.0
        on_n = 0
        off_n = 0
        for h, p in zip(self.hypotheses, masses):
            if predicts_open(h, obj):
                on_mass += p
                on_n += 1
            else:
                off_mass += p
                off_n += 1
        n = len(self.hypotheses)
        prior = math.log2(n) if n > 1 else 0.0

        def term(mass, count):
            if mass <= 0.0 or count <= 0:
                return 0.0
            if count == 1:
                return 0.0
            return mass * math.log2(count)

        remaining = term(on_mass, on_n) + term(off_mass, off_n)
        return prior - remaining

    def next_experiment(self):
        if not self.untested or len(self.hypotheses) <= 1:
            return None
        best = None
        best_ig = -1.0
        for obj in self.untested:
            ig = self._weighted_ig(obj)
            if ig > best_ig:
                best_ig = ig
                best = obj
        return best

    def observe(self, obj, opened):
        self.history.append((obj, opened))
        if obj in self.untested:
            self.untested.remove(obj)
        self.tested.append(obj)
        keep = []
        for h in self.hypotheses:
            if predicts_open(h, obj) == opened:
                keep.append(h)
        self.hypotheses = keep

    def converged(self):
        feature_rules = [h for h in self.hypotheses if h[0] != "*"]
        return len(feature_rules) == 1 and len(self.hypotheses) <= 2

    def surviving_rule(self):
        feature_rules = [h for h in self.hypotheses if h[0] != "*"]
        if len(feature_rules) == 1:
            return feature_rules[0]
        if len(self.hypotheses) == 1 and self.hypotheses[0][0] != "*":
            return self.hypotheses[0]
        return None

    def note_convergence(self):
        rule = self.surviving_rule()
        if rule is None:
            return
        feature = rule[0]
        if feature in self.alpha:
            self.alpha[feature] += 1.0

    def answer(self, obj):
        if not self.hypotheses:
            return False
        votes = sum(1 for h in self.hypotheses if predicts_open(h, obj))
        return votes * 2 >= len(self.hypotheses)

    def held_out_accuracy(self, world):
        labels = world.held_out_labels()
        if not labels:
            return 0.0
        ok = 0
        for obj, truth in labels:
            if self.answer(obj) == truth:
                ok += 1
        return 100.0 * ok / len(labels)

    def run_world(self, world, max_steps=32):
        self.begin_world(world)
        curve = []
        steps = 0
        while steps < max_steps:
            curve.append(self.held_out_accuracy(world))
            if len(self.hypotheses) == 1:
                break
            obj = self.next_experiment()
            if obj is None:
                break
            opened = world.touch(obj)
            self.observe(obj, opened)
            steps += 1
        curve.append(self.held_out_accuracy(world))
        if len(self.hypotheses) == 1:
            self.note_convergence()
        return {
            "steps": steps,
            "curve": curve,
            "held_out": self.held_out_accuracy(world),
            "rule": self.surviving_rule(),
            "salience": dict(self.salience()),
        }


# ---------------------------------------------------------------------------
# Experiment 004: salience that can be wrong, and can change its mind.
# ---------------------------------------------------------------------------


class ForgetAgent(Agent):
    """Salience with bounded precision (Itti & Baldi 2005; forget factor `f`).

    Dirichlet-style counts updated as alpha[f] = f*alpha[f] + (1 if winner
    else 0), so counts saturate around 1/(1-f) instead of growing without
    bound. At f=1.0 this reduces exactly to the Experiment 002 bare prior.
    """

    def __init__(self, features=None, forget=0.7):
        super().__init__(features)
        self.forget = forget

    def note_convergence(self):
        rule = self.surviving_rule()
        if rule is None:
            return
        winner = rule[0]
        if winner in self.alpha:
            self.alpha = {
                f: self.forget * self.alpha[f] + (1.0 if f == winner else 0.0)
                for f in self.features
            }


class ChangePointAgent(Agent):
    """Evidence-gated forgetting via a run-length monitor (Adams & MacKay 2007).

    Maintains an exact posterior over run length r (worlds since the last
    regime change) over the sequence of converging winner features, with a
    per-world change hazard `h`. States are (r, prob, Dirichlet-count-vector)
    triples; growth carries a state to r+1, a change emits a fresh state at
    r=0 whose counts restart from the uniform base (this is the key
    difference from fixed forgetting: never-winning features are *restored*,
    not decayed toward zero). Salience counts used for ACT are the
    posterior-weighted average of per-state counts.
    """

    def __init__(self, features=None, hazard=1.0 / 5.0, run_cap=20, max_states=64):
        super().__init__(features)
        self.hazard = hazard
        self.run_cap = run_cap
        self.max_states = max_states
        self.runs = None
        self.reset_salience()

    def reset_salience(self):
        super().reset_salience()
        self.runs = [(0, 1.0, {f: 1.0 for f in self.features})]

    def _expected_counts(self):
        total = sum(p for _, p, _ in self.runs) or 1.0
        out = {f: 0.0 for f in self.features}
        for _, p, counts in self.runs:
            for f in self.features:
                out[f] += (p / total) * counts[f]
        return out

    def note_convergence(self):
        rule = self.surviving_rule()
        if rule is None:
            return
        winner = rule[0]
        if winner not in self.alpha:
            return

        fs = list(self.features)

        def bump(counts):
            c = dict(counts)
            c[winner] += 1.0
            return c

        fresh = {f: 1.0 for f in fs}
        fresh[winner] += 1.0

        # Exact unmerged trellis: growth -> r+1 with bumped counts; change ->
        # r=0 with fresh uniform-based counts. Dedupe only identical states.
        new = []
        for r, p, counts in self.runs:
            total = sum(counts.values())
            if total <= 0:
                continue
            pred = counts[winner] / float(total)
            new.append((min(r + 1, self.run_cap), p * (1.0 - self.hazard) * pred, bump(counts)))
            new.append((0, p * self.hazard * pred, dict(fresh)))

        dedup = {}
        for r, p, counts in new:
            key = (r, tuple(sorted(counts.items())))
            if key not in dedup:
                dedup[key] = [0.0, counts]
            dedup[key][0] += p
        merged = [(r, p, counts) for (r, _), (p, counts) in dedup.items()]
        merged.sort(key=lambda s: -s[1])
        if len(merged) > self.max_states:
            merged = merged[: self.max_states]
        total_p = sum(p for _, p, _ in merged) or 1.0
        self.runs = [(r, p / total_p, counts) for r, p, counts in merged]
        self.alpha = self._expected_counts()


# ---------------------------------------------------------------------------
# Experiment 005: learning the hazard rate (hierarchical change-point)
# ---------------------------------------------------------------------------


HAZARD_GRID = (
    1.0 / 2.0,
    1.0 / 3.0,
    1.0 / 4.0,
    1.0 / 6.0,
    1.0 / 8.0,
    1.0 / 16.0,
    1.0 / 32.0,
)


class HierarchicalChangePointAgent(Agent):
    """Prior whose *rate* of forgetting is learned from the stream (Wilson et al. 2010).

    Runs one exact Dirichlet-categorical run-length trellis per candidate on
    a hazard grid, accumulates each candidate's conditional predictive
    marginal likelihood (the per-event run-length normalizer; the hazard
    dependence reaches it through the way each candidate's run-length
    posterior evolves), and model-averages the salience counts over the
    hazard posterior. The experimenter sets no hazard.

    At a uniform hazard posterior this behaves exactly like the bare
    Experiment 002 prior inside a single world, so the identical-machinery
    null control stays valid.
    """

    def __init__(self, features=None, hazards=HAZARD_GRID, run_cap=20, max_states=64):
        super().__init__(features)
        self.hazards = tuple(hazards)
        self.run_cap = run_cap
        self.max_states = max_states
        self.reset_salience()

    def reset_salience(self):
        super().reset_salience()
        self.hazard_logliks = [0.0] * len(self.hazards)
        self.trellises = [
            [(0, 1.0, {f: 1.0 for f in self.features})] for _ in self.hazards
        ]

    def hazard_posterior(self):
        peak = max(self.hazard_logliks)
        weights = [math.exp(x - peak) for x in self.hazard_logliks]
        total = sum(weights) or 1.0
        return [w / total for w in weights]

    def _expected_counts_trellis(self, runs):
        total = sum(p for _, p, _ in runs) or 1.0
        out = {f: 0.0 for f in self.features}
        for _, p, counts in runs:
            for f in self.features:
                out[f] += (p / total) * counts[f]
        return out

    def _trellis_step(self, runs, winner, hazard):
        """One run-length update for one candidate. Returns (runs, ll_delta).

        ll_delta is the conditional predictive mass (the normalization
        constant of the run-length posterior update) for this observation
        under this candidate hazard.
        """
        fs = list(self.features)

        def bump(counts):
            c = dict(counts)
            c[winner] += 1.0
            return c

        fresh = {f: 1.0 for f in fs}
        fresh[winner] += 1.0

        new = []
        for r, p, counts in runs:
            total = sum(counts.values())
            if total <= 0:
                continue
            pred = counts[winner] / float(total)
            new.append((min(r + 1, self.run_cap), p * (1.0 - hazard) * pred, bump(counts)))
            new.append((0, p * hazard * pred, dict(fresh)))

        ll_delta = sum(p for _, p, _ in new)

        dedup = {}
        for r, p, counts in new:
            key = (r, tuple(sorted(counts.items())))
            if key not in dedup:
                dedup[key] = [0.0, counts]
            dedup[key][0] += p
        merged = [(r, p, counts) for (r, _), (p, counts) in dedup.items()]
        merged.sort(key=lambda s: -s[1])
        if len(merged) > self.max_states:
            merged = merged[: self.max_states]
        total_p = sum(p for _, p, _ in merged) or 1.0
        return [(r, p / total_p, counts) for r, p, counts in merged], ll_delta

    def note_convergence(self):
        rule = self.surviving_rule()
        if rule is None:
            return
        winner = rule[0]
        if winner not in self.alpha:
            return

        new_trellises = []
        ll_deltas = []
        for runs, hazard in zip(self.trellises, self.hazards):
            updated, ll = self._trellis_step(runs, winner, hazard)
            new_trellises.append(updated)
            ll_deltas.append(ll)
        self.trellises = new_trellises

        for j, ll in enumerate(ll_deltas):
            self.hazard_logliks[j] += math.log(max(ll, 1e-300))

        weights = self.hazard_posterior()
        out = {f: 0.0 for f in self.features}
        for w, runs in zip(weights, self.trellises):
            exp = self._expected_counts_trellis(runs)
            for f in self.features:
                out[f] += w * exp[f]
        self.alpha = out