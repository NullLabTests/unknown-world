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