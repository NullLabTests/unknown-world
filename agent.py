"""The Minimal Loop.

Hypothesis set over the world -> predict consequences -> act (the most
informative experiment) -> observe -> update the hypothesis set.

The agent optimizes its uncertainty about the world, not a reward.
Experiments are chosen to split the hypothesis space as evenly as
possible: the action that would most efficiently reveal which rule holds.
"""

from __future__ import annotations


class MinimalLoop:
    def __init__(self):
        self.reset()

    def reset(self):
        self.alphabet = {}
        self.hypotheses = []
        self.untested = []
        self.tested = []

    def boot(self, presented):
        for obj in presented:
            for f, v in obj.features.items():
                self.alphabet.setdefault(f, set()).add(v)
        self.hypotheses = [("*always", None)]
        self.hypotheses += [
            (f, v) for f in self.alphabet for v in sorted(self.alphabet[f])
        ]
        self.hypotheses.append(("*never", None))
        self.untested = list(presented)

    def predicts(self, h, obj):
        if h[0] == "*always":
            return True
        if h[0] == "*never":
            return False
        return obj.features.get(h[0]) == h[1]

    def next_experiment(self):
        best, best_gain = None, -1
        for obj in self.untested:
            on = sum(1 for h in self.hypotheses if self.predicts(h, obj))
            off = len(self.hypotheses) - on
            gain = min(on, off)
            if gain > best_gain:
                best_gain, best = gain, obj
        return best

    def commit(self, obj, effect):
        if obj in self.untested:
            self.untested.remove(obj)
            self.tested.append(obj)
        self.hypotheses = [
            h for h in self.hypotheses if self.predicts(h, obj) == effect
        ]

    @property
    def converged(self):
        return len(self.hypotheses) == 1

    @property
    def surviving(self):
        return list(self.hypotheses)

    def decide(self, obj):
        """Votes of the surviving hypotheses: (says opens, says closed)."""
        open_votes = sum(1 for h in self.hypotheses if self.predicts(h, obj))
        return open_votes, len(self.hypotheses) - open_votes

    def predicts_open(self, obj):
        return self.decide(obj)[0] > self.decide(obj)[1]