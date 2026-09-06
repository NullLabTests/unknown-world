# The Unknown World — Experiment 001

> A laboratory for discovering agents: tiny unknown-worlds, the smallest
> possible learning mechanism installed into them, and a protocol that
> measures *adaptation* rather than performance on a benchmark.

<p align="center">
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-brightgreen.svg"></a>
  <a href="https://www.python.org/"><img alt="Python 3.8+" src="https://img.shields.io/badge/python-3.8%2B-blue.svg"></a>
  <a href="https://github.com/NullLabTests/unknown-world/blob/main/README.md"><img alt="Experiment" src="https://img.shields.io/badge/experiment-001-orange.svg"></a>
  <a href="https://github.com/NullLabTests/unknown-world/actions/workflows/tests.yml"><img alt="Tests" src="https://img.shields.io/github/actions/workflow/status/NullLabTests/unknown-world/tests.yml?label=tests"></a>
  <a href="https://github.com/NullLabTests/unknown-world"><img alt="Repo" src="https://img.shields.io/github/repo-size/NullLabTests/unknown-world.svg"></a>
</p>

## The question

We are not asking "how do we build AGI?"

We are asking: **what is the minimum set of mechanisms from which general
intelligence can emerge?** Not an LLM, not a transformer, not a swarm, not an
SNN — those are implementation possibilities. We want the necessary
computational structure first, then evidence.

Our working hypothesis:

> General intelligence may emerge when a system can autonomously construct,
> test, revise, and reuse predictive models across increasingly novel
> environments, while also modeling and improving its own learning process.

This repository is Experiment 001 of that program: the smallest possible
artificial world in which the primitive loop matters, a minimal agent that
installs it, and an evaluation protocol built to detect generalization —
or the lack of it.

## What the world is

An `UnknownWorld` is a small population of objects with visible features:

```
color ∈ {red, blue, green}
shape ∈ {round, square, star}
```

Exactly one `{feature: value}` pair is the **hidden rule**: objects matching
it open the door, all others do not. The agent does not know the rule. It
never sees the rule. It must discover it by touching objects and observing
the effect — and then it must answer about objects it has never touched:

```
observation:   object A  object B  object C  ...   ("hit the blue one")
interaction:   touch → does it open the door?      (the experiment)
answer:        D → ?                               (the transfer)
```

## What the agent is — the primitive loop

```
             hypothesis set
                 │
                 ▼
            PREDICT          (expect the effect for each candidate action)
                 │
                 ▼
              ACT            (touch the most informative object)
                 │
                 ▼
            OBSERVE          (the actual effect)
                 │
                 ▼
             UPDATE          (eliminate hypotheses that would have been wrong)
                 │
                 └──────────► repeat until one rule survives
```

Experiments are chosen to **maximize information gain**, not reward: touch
the object whose outcome would split the surviving hypothesis set most
evenly. The agent is optimizing its uncertainty about the world, not its
immediate payoff.

That decision is the point of the experiment.

## Protocol — rototest

We never declare intelligence from a fixed benchmark. Competence is measured
under change:

| Measure | What it detects |
| --- | --- |
| **Within-world** — held-out answer accuracy as the hypothesis set narrows | the competence curve (`0% → 100%`) |
| **Across-world** — experiments-to-converge, world after world, with rules re-bound to novel objects and novel bindings | the learning-to-learn signal |

Expected forms the results may take:

> 40% → 70% → 90% on a completely new task after learning a previous one
> is interesting. Noticing *"my current learning strategy is failing"* is
> where it gets genuinely interesting.

## Results (measured, not asserted)

```
  # phase     hidden rule        tests held-out acc  competence curve
------------------------------------------------------------------------
  1 learning  color == 'blue'        3       100.0%  0% -> 0% -> 100% -> 100%
  2 learning  color == 'blue'        3       100.0%  0% -> 100% -> 0% -> 100%
  3 learning  color == 'blue'        3       100.0%  100% -> 100% -> 100% -> 100%
  4 learning  color == 'blue'        3       100.0%  0% -> 0% -> 0% -> 100%
  5 learning  color == 'blue'        3       100.0%  100% -> 0% -> 100% -> 100%
  1 transfer  color == 'red'         3       100.0%  100% -> 100% -> 100% -> 100%
  2 transfer  shape == 'star'        3       100.0%  0% -> 0% -> 0% -> 100%
  3 transfer  color == 'red'         3       100.0%  100% -> 0% -> 100% -> 100%
  4 transfer  shape == 'star'        3       100.0%  100% -> 100% -> 100% -> 100%
  5 transfer  shape == 'round'       3       100.0%  100% -> 0% -> 100% -> 100%
------------------------------------------------------------------------
  learning phase  : experiments-to-converge per world = [3, 3, 3, 3, 3]
  transfer phase  : experiments-to-converge per world = [3, 3, 3, 3, 3]
  held-out answers reach 100% in 10/10 worlds
```

**Interpretation**

- The loop converges to the true rule and answers correctly about novel
  objects in every world: within-world competence reaches 100%. The
  primitive works.
- Experiments-to-converge is **flat** across learning and transfer phases:
  learning world 1 does not speed up world 2. The baseline is
  **competent, not generalizing**. No learning-to-learn, no transfer of
  structure.

That flat row is the finding. It isolates the first missing primitive.

## Reproduce

```bash
python3 run.py                        # the experiment and its interpretation
python3 -m unittest discover -s .     # the sanity suite
```

Stdlib only. No dependencies. Deterministic seed (`random.Random(7)`).

## Next primitive (only when the protocol can detect it)

A prior over feature salience — which dimensions tend to be diagnostic —
refreshed from experience, so experiment choice improves across worlds. The
flat row must turn into a descending one before the primitive is kept.

> Rule: never declare intelligence from performance on a fixed benchmark.
> Measure the system's ability to adapt when the rules, tasks, environment,
> and available information change.

## Layout

| File | Role |
| --- | --- |
| `world.py` | generator of minimal unknown-worlds |
| `agent.py` | the minimal loop (hypothesis → predict → act → observe → update) |
| `protocol.py` | rototest harness: within-world curve + across-world signal |
| `run.py` | runs Experiment 001 and prints the evidence table |
| `test_step1.py` | sanity suite (world, loop, protocol) |

## License

MIT © 2026 NullLabTests