"""World: generator of minimal unknown-worlds (Experiment 001).

An UnknownWorld is a small population of objects over a feature alphabet.
Exactly one (feature, value) pair is the hidden rule; objects matching it
produce the goal effect (open the door), all others do not. The rule is
unknown to the agent and must be discovered by experiment.

Later stages parameterize: multiple effects, noise, action cost, changing
rules, other agents. This file carries only what Stage 1 exercises.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass

FEATURE_DOMAINS = {
    "color": ("red", "blue", "green"),
    "shape": ("round", "square", "star"),
}


@dataclass
class Object:
    """A perceived thing. `features` are visible; `opens_door` is the goal effect."""

    features: dict
    opens_door: bool


def _iter_population():
    names = list(FEATURE_DOMAINS)
    for combo in itertools.product(*[FEATURE_DOMAINS[n] for n in names]):
        yield dict(zip(names, combo))


@dataclass
class UnknownWorld:
    concept: tuple
    presented: list
    held_out: list

    @classmethod
    def generate(cls, rng, concept=None, n_present=None, noise=0.0):
        feature = concept[0] if concept else rng.choice(list(FEATURE_DOMAINS))
        value = concept[1] if concept else rng.choice(FEATURE_DOMAINS[feature])
        concept = (feature, value)
        population = []
        for features in _iter_population():
            effect = features[feature] == value
            if noise and rng.random() < noise:
                effect = not effect
            population.append(Object(features, bool(effect)))
        rng.shuffle(population)
        if n_present is None:
            n_present = len(population) - 1
        presented, held_out = population[:n_present], population[n_present:]
        return cls(concept, presented, held_out)

    def experiment(self, obj):
        """TOUCH: reveal whether the object opens the door."""
        return obj.opens_door

    @property
    def rule(self):
        return f"{self.concept[0]} == {self.concept[1]!r}"