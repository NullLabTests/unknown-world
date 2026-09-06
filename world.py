"""Minimal unknown-worlds: objects with color/shape; one hidden {feature: value} rule."""
from __future__ import annotations

FEATURE_DOMAINS = {
    "color": ("red", "blue", "green"),
    "shape": ("round", "square", "star"),
}

FEATURES = tuple(FEATURE_DOMAINS.keys())


class Object:
    __slots__ = ("color", "shape")

    def __init__(self, color, shape):
        self.color = color
        self.shape = shape

    def as_dict(self):
        return {"color": self.color, "shape": self.shape}

    def __repr__(self):
        return "Object(color=%r, shape=%r)" % (self.color, self.shape)

    def __eq__(self, other):
        return isinstance(other, Object) and self.color == other.color and self.shape == other.shape

    def __hash__(self):
        return hash((self.color, self.shape))


def all_objects():
    out = []
    for color in FEATURE_DOMAINS["color"]:
        for shape in FEATURE_DOMAINS["shape"]:
            out.append(Object(color, shape))
    return out


class UnknownWorld:
    def __init__(self, objects, hidden_rule, held_out, noise=0.0, rng=None):
        self.objects = list(objects)
        self.hidden_rule = tuple(hidden_rule)  # (feature, value)
        self.held_out = list(held_out)
        self.noise = float(noise)
        self.rng = rng

    def matches(self, obj, rule=None):
        feature, value = rule if rule is not None else self.hidden_rule
        return getattr(obj, feature) == value

    def touch(self, obj):
        opens = self.matches(obj)
        if self.noise and self.rng is not None and self.rng.random() < self.noise:
            opens = not opens
        return opens

    def held_out_labels(self):
        return [(obj, self.matches(obj)) for obj in self.held_out]

    @classmethod
    def generate(cls, rng, concept=None, force_feature=None, n_present=6, n_held_out=3, noise=0.0):
        if force_feature is not None:
            if force_feature not in FEATURE_DOMAINS:
                raise ValueError("unknown feature %r" % (force_feature,))
            feature = force_feature
            if concept is not None and concept[0] != feature:
                raise ValueError("concept feature %r != force_feature %r" % (concept[0], feature))
            value = concept[1] if concept is not None else rng.choice(list(FEATURE_DOMAINS[feature]))
        elif concept is not None:
            feature, value = concept
        else:
            feature = rng.choice(list(FEATURES))
            value = rng.choice(list(FEATURE_DOMAINS[feature]))
        rule = (feature, value)

        pool = all_objects()
        matching = [o for o in pool if getattr(o, feature) == value]
        others = [o for o in pool if getattr(o, feature) != value]
        rng.shuffle(matching)
        rng.shuffle(others)

        # both sides present; held-out includes at least one of each when possible
        present, held = [], []
        need_match = max(1, n_present // 3)
        need_other = n_present - need_match
        present.extend(matching[:need_match])
        present.extend(others[:need_other])
        used = set(present)
        rest = [o for o in pool if o not in used]
        rng.shuffle(rest)
        # prefer mixed held-out
        held_m = [o for o in matching if o not in used]
        held_o = [o for o in others if o not in used]
        if held_m:
            held.append(held_m[0])
        if held_o:
            held.append(held_o[0])
        for o in rest:
            if len(held) >= n_held_out:
                break
            if o not in held:
                held.append(o)
        while len(present) < n_present and rest:
            o = rest.pop()
            if o not in present and o not in held:
                present.append(o)
        rng.shuffle(present)
        return cls(present, rule, held[:n_held_out], noise=noise, rng=rng)


def all_atomic_rules():
    rules = []
    for feature, values in FEATURE_DOMAINS.items():
        for value in values:
            rules.append((feature, value))
    return rules