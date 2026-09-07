"""Version-space loop + optional feature-salience prior (Experiment 002)."""
from __future__ import annotations

import math

from world import FEATURES, FEATURE_DOMAINS, all_atomic_rules

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

    def effective_hazard(self):
        hps = self.hazard_posterior()
        return sum(w * h for w, h in zip(hps, self.hazards))

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


# ---------------------------------------------------------------------------
# Experiment 006: the hazard learned *per observation* (adaptive hazard)
# ---------------------------------------------------------------------------


class ObservationHazardAgent(Agent):
    """Per-observation run-length monitor with an earned hazard (Experiment 006).

    Experiment 005 inferred the hazard from a *compressed* stream: one winner
    feature per world, updated only at convergence. Standard BOCPD
    (Adams & MacKay 2007; Wilson, Nassar & Gold 2010) instead updates at
    every *observation*, and that is what this agent does. The regime
    variable is the *feature* of the hidden rule, which can only change at a
    world boundary; the change hazard is inferred from the stream rather than
    set by the experimenter.

    Mechanics, per observation (object touched + outcome):

    - Each candidate hazard on HAZARD_GRID carries one exact run-length
      trellis of nodes ``(r, p, c)`` where ``c`` is a Dirichlet belief over
      which feature the current run is governed by. A change resets the run
      (r=0, c back to the uniform base); growth carries r+1 and keeps c.
    - The transition prior is *boundary-aware*: a feature cannot change
      inside a world, so the hazard-weighted change branch is only active on
      the first observation of each world. This is exactly the regression
      that 005 collapsed: the world boundary (not every world convergence)
      is the hazard event.
    - The emission is the marginal predictive of the outcome under each
      feature, computed from *within-world value evidence* (the hidden value
      is fresh every world, so the value belief resets at every boundary).
      The whole observation stream therefore moves the hazard posterior,
      not just the winning feature.

    Salience for ACT is recomputed only at world convergence (note the
    identical-machinery discipline of Experiment 003): within a world the
    agent behaves exactly like the base Agent given the same alpha, so the
    paired null control stays valid.
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
        self.value_weights = {
            f: {v: 1.0 for v in FEATURE_DOMAINS[f]} for f in self.features
        }
        self._boundary_pending = True

    def begin_world(self, world):
        super().begin_world(world)
        self.value_weights = {
            f: {v: 1.0 for v in FEATURE_DOMAINS[f]} for f in self.features
        }
        self._boundary_pending = True
        self._world_p_change = self.effective_hazard()

    def hazard_posterior(self):
        peak = max(self.hazard_logliks)
        weights = [math.exp(x - peak) for x in self.hazard_logliks]
        total = sum(weights) or 1.0
        return [w / total for w in weights]

    def effective_hazard(self):
        hps = self.hazard_posterior()
        return sum(w * h for w, h in zip(hps, self.hazards))

    def _feature_predictive(self, obj, opened):
        """Per-feature likelihood of this observation given value evidence.

        For each feature f the likelihood is the posterior predictive of the
        outcome under the world's hidden value given the observations of the
        current world. A feature the current world outright contradicts
        returns 0.5 (maximal uncertainty) rather than 0, so contradicted
        nodes drain away observation-by-observation instead of dying in one
        step.
        """
        out = {}
        for f in self.features:
            w = self.value_weights[f]
            total = sum(w.values())
            if total <= 0:
                out[f] = 0.5
                continue
            p_open = sum(w[v] for v in w if getattr(obj, f) == v) / total
            out[f] = p_open if opened else (1.0 - p_open)
        return out

    def observe(self, obj, opened):
        super().observe(obj, opened)
        lik = self._feature_predictive(obj, opened)
        boundary = self._boundary_pending
        self._boundary_pending = False

        base = {f: 1.0 for f in self.features}
        new_trellises = []
        ll_deltas = []
        for runs, hazard in zip(self.trellises, self.hazards):
            new_runs = []
            for r, p, c in runs:
                tot = sum(c.values())
                if tot <= 0:
                    continue
                em = sum((c[f] / tot) * lik[f] for f in self.features)
                cfeat = {f: c[f] * lik[f] for f in self.features}
                if boundary:
                    new_runs.append(
                        (min(r + 1, self.run_cap), p * (1.0 - hazard) * em, cfeat)
                    )
                    new_runs.append((0, p * hazard * em, dict(base)))
                else:
                    new_runs.append((min(r + 1, self.run_cap), p * em, cfeat))
            ll_delta = sum(p for _, p, _ in new_runs) or 1e-300
            ll_deltas.append(ll_delta)

            new_runs.sort(key=lambda s: -s[1])
            if len(new_runs) > self.max_states:
                new_runs = new_runs[: self.max_states]
            total_p = sum(p for _, p, _ in new_runs) or 1.0
            new_runs = [(r, p / total_p, c) for r, p, c in new_runs]
            new_trellises.append(new_runs)

        self.trellises = new_trellises
        for j, ll in enumerate(ll_deltas):
            self.hazard_logliks[j] += math.log(ll)

        matched = {f: getattr(obj, f) for f in self.features}
        for f in self.features:
            w = self.value_weights[f]
            for v in w:
                if (v == matched[f]) != opened:
                    w[v] = 0.0

    def note_convergence(self):
        rule = self.surviving_rule()
        if rule is None:
            return
        weights = self.hazard_posterior()
        out = {f: 0.0 for f in self.features}
        for w, runs in zip(weights, self.trellises):
            total = sum(p for _, p, _ in runs) or 1.0
            for _, p, c in runs:
                for f in self.features:
                    out[f] += w * (p / total) * c[f]
        self.alpha = out


# ---------------------------------------------------------------------------
# Experiment 007: the exact latent-hazard hierarchy + hazard-aware ACT
# ---------------------------------------------------------------------------

# Pre-committed 007 hyperparameters (set before any measurement, not tuned):
META_HAZARD = 1.0 / 50.0  # Wilson et al. h^(0): rate the hazard itself changes
HAZARD_AP = 1.0           # Beta prior params on the hazard (uniform)
HAZARD_BP = 1.0


class LatentHazardAgent(Agent):
    """Exact three-level latent-hazard hierarchy (Wilson, Nassar & Gold 2010).

    The hazard grid of Experiments 005/006 is removed entirely. The hazard
    rate is a latent Beta-Bernoulli variable carried inside the hierarchy,
    exactly as in Wilson et al. equations (16)-(24): each node holds four
    sufficient statistics ``(r2, a, b, c)``:

    - ``r2``  the data-level run length (observations since the current
      feature regime began),
    - ``a``   the change-point count of the Beta posterior on the hazard,
    - ``b``   the non-change-point count of the Beta posterior on the hazard,
    - ``c``   a Dirichlet belief over which feature governs the current run.

    The high-level run length is ``r1 = a + b`` (eqn 42), and the node's
    hazard estimate is ``h~ = (a + a_p) / (a + b + a_p + b_p)`` (eqn 43) with
    the pre-committed Beta prior ``(a_p, b_p) = (HAZARD_AP, HAZARD_BP)``.

    Every observation a node spawns four children (eqn 24), weighted by
    ``(1-h0)(1-h~)``, ``(1-h0)h~``, ``h0(1-h~)``, ``h0 h~`` where
    ``h0 = META_HAZARD`` is the rate at which the hazard *itself* changes:

    - no hazard change, data grows:        ``r2+1, a, b+1, c . lik``
    - no hazard change, data changes:      ``0, a+1, b, fresh``
    - hazard change, data grows:           ``r2+1, a_p, b_p, c . lik``
    - hazard change, data changes:         ``0, a_p, b_p, fresh``

    The two data-change children reset the run (``r2 -> 0``, feature counts
    back to the uniform base) and are gated to the first observation of each
    world, exactly as in Experiment 006 (a feature cannot change inside a
    world). The two hazard-change children reset the Beta counts to the prior:
    that is the hierarchy's mechanism for *forgetting a dead hazard rate* and
    re-learning from scratch — the property a non-constant rate requires.

    The feature emission is Experiment 006's: the within-world value belief's
    posterior predictive of the observed outcome per feature, so the whole
    per-observation stream moves the hazard posterior. Salience for ACT still
    updates only at convergence (identical-machinery discipline).

    Pruning is Wilson et al. section 5's similarity grouping: nodes are
    merged whose low-level run-length (``log(r2 + v_p)``), high-level
    run-length (``log(r1 + a_p + b_p)``) and hazard estimate (``h~``) fall in
    the same bin, then a hard ``max_states`` cap.
    """

    def __init__(
        self,
        features=None,
        meta_hazard=META_HAZARD,
        ap=HAZARD_AP,
        bp=HAZARD_BP,
        run_cap=24,
        max_states=384,
        k1=0.2,
        k2=1.0,
        vp=1.0,
    ):
        super().__init__(features)
        self.meta_hazard = float(meta_hazard)
        self.ap = float(ap)
        self.bp = float(bp)
        self.run_cap = run_cap
        self.max_states = max_states
        self.k1 = float(k1)
        self.k2 = float(k2)
        self.vp = float(vp)
        self.nodes = None
        self.value_weights = None
        self._boundary_pending = False
        self._world_p_change = 0.5
        self.reset_salience()

    def reset_salience(self):
        super().reset_salience()
        self.nodes = [(0, self.ap, self.bp, {f: 1.0 for f in self.features}, 1.0)]
        self.value_weights = {
            f: {v: 1.0 for v in FEATURE_DOMAINS[f]} for f in self.features
        }
        self._boundary_pending = True

    def begin_world(self, world):
        super().begin_world(world)
        self.value_weights = {
            f: {v: 1.0 for v in FEATURE_DOMAINS[f]} for f in self.features
        }
        self._boundary_pending = True
        self._world_p_change = self.effective_hazard()

    def effective_hazard(self):
        """Posterior-mean hazard E[h~] over the hierarchy's nodes (eqn 43)."""
        total = sum(p for *_, p in self.nodes) or 1.0
        out = 0.0
        for _, a, b, _, p in self.nodes:
            out += (p / total) * (a + self.ap) / (a + b + self.ap + self.bp)
        return out

    def _feature_predictive(self, obj, opened):
        out = {}
        for f in self.features:
            w = self.value_weights[f]
            total = sum(w.values())
            if total <= 0:
                out[f] = 0.5
                continue
            p_open = sum(w[v] for v in w if getattr(obj, f) == v) / total
            out[f] = p_open if opened else (1.0 - p_open)
        return out

    def observe(self, obj, opened):
        super().observe(obj, opened)
        lik = self._feature_predictive(obj, opened)
        boundary = self._boundary_pending
        self._boundary_pending = False

        fresh = {f: 1.0 for f in self.features}
        new = []
        for r, a, b, c, p in self.nodes:
            tot = sum(c.values())
            if tot <= 0:
                continue
            em = sum((c[f] / tot) * lik[f] for f in self.features)
            cfeat = {f: c[f] * lik[f] for f in self.features}
            if not boundary:
                # Within a world the hidden feature cannot change, so the
                # observation is pure data-level growth: the hazard posterior
                # (a Bernoulli trial over world boundaries) is untouched, while
                # the data run extends and the feature belief updates.
                new.append((min(r + 1, self.run_cap), a, b, cfeat, p * em))
                continue
            # One hazard event per world boundary (eqn 24): the two data-change
            # children reset the run, the two hazard-change children reset the
            # Beta counts to the prior, h0 = META_HAZARD scales the branches in
            # which the hazard itself changed.
            tilde_h = (a + self.ap) / (a + b + self.ap + self.bp)
            g = min(r + 1, self.run_cap)
            # no hazard change, data grows        (eqn 24 case 1)
            new.append((g, a, b + 1.0, cfeat, p * (1.0 - tilde_h) * (1.0 - self.meta_hazard) * em))
            # no hazard change, data changes      (eqn 24 case 2)
            new.append((0, a + 1.0, b, dict(fresh), p * tilde_h * (1.0 - self.meta_hazard) * em))
            # hazard change, data grows           (eqn 24 case 3)
            new.append((g, self.ap, self.bp, cfeat, p * (1.0 - tilde_h) * self.meta_hazard * em))
            # hazard change, data changes         (eqn 24 case 4)
            new.append((0, self.ap, self.bp, dict(fresh), p * tilde_h * self.meta_hazard * em))

        merged = {}
        for r, a, b, c, p in new:
            r1 = a + b
            tilde_h = (a + self.ap) / (r1 + self.ap + self.bp)
            key = (
                int(math.log((r + self.vp) / self.vp) / math.log(1.0 + self.k2)),
                int(math.log((r1 + self.ap + self.bp) / (self.ap + self.bp)) / math.log(1.0 + self.k1)),
                int(tilde_h / self.k1),
            )
            slot = merged.get(key)
            if slot is None:
                slot = [0.0, 0.0, 0.0, 0.0, {f: 0.0 for f in self.features}]
                merged[key] = slot
            slot[0] += p
            slot[1] += p * r
            slot[2] += p * a
            slot[3] += p * b
            for f in c:
                slot[4][f] += p * c[f]

        nodes = []
        for wp, wr, wa, wb, wc in merged.values():
            tp = wp or 1.0
            nodes.append(
                (wr / tp, wa / tp, wb / tp, {f: wc[f] / tp for f in self.features}, wp)
            )
        nodes.sort(key=lambda s: -s[4])
        if len(nodes) > self.max_states:
            nodes = nodes[: self.max_states]
        total_p = sum(p for *_, p in nodes) or 1.0
        self.nodes = [(r, a, b, c, p / total_p) for r, a, b, c, p in nodes]

        matched = {f: getattr(obj, f) for f in self.features}
        for f in self.features:
            w = self.value_weights[f]
            for v in w:
                if (v == matched[f]) != opened:
                    w[v] = 0.0

    def note_convergence(self):
        rule = self.surviving_rule()
        if rule is None:
            return
        total = sum(p for *_, p in self.nodes) or 1.0
        out = {f: 0.0 for f in self.features}
        for _, _, _, c, p in self.nodes:
            for f in self.features:
                out[f] += (p / total) * c[f]
        self.alpha = out


class LatentHazardAnticipateAgent(LatentHazardAgent):
    """Hazard-aware ACT (Experiment 007): query value hedged under change.

    The information gain of every candidate query is computed twice: once
    under the current run's feature belief (standard weighted IG) and once
    under a fresh world's belief (the hierarchy's change-branch predictive:
    uniform feature counts). The value of the query is the mixture

        IG = (1 - w) * IG_salience + w * IG_fresh,

    blended by ``w`` = the agent's own hazard estimate at the world boundary:
    its belief that the rule of the world it is about to touch is *not* the
    current run's feature. Asking what matters if the world just changed is
    the proactive case: when the learned rate of change is high, the agent
    values queries that would pay off under either hypothesis of the
    generative feature. At ``w = 0`` (a belief of no change) the hedged
    query value reduces exactly to the standard ACT, so the paired
    identical-machinery control stays valid.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._hedge_w = 0.0

    def begin_world(self, world):
        super().begin_world(world)
        self._hedge_w = self._world_p_change

    def _ig_from_masses(self, on_mass, off_mass, on_n, off_n):
        n = len(self.hypotheses)
        prior = math.log2(n) if n > 1 else 0.0

        def term(mass, count):
            if mass <= 0.0 or count <= 0:
                return 0.0
            if count == 1:
                return 0.0
            return mass * math.log2(count)

        return prior - (term(on_mass, on_n) + term(off_mass, off_n))

    def _weighted_ig(self, obj):
        w = getattr(self, "_hedge_w", 0.0)
        if w <= 0.0:
            return super()._weighted_ig(obj)

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
        ig_sal = self._ig_from_masses(on_mass, off_mass, on_n, off_n)
        ig_fresh = self._ig_from_masses(on_n / n, off_n / n, on_n, off_n)
        return (1.0 - w) * ig_sal + w * ig_fresh