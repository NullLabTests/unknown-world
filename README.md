# The Unknown World — Experiments 001 & 002

> A laboratory for discovering agents: tiny unknown-worlds, the smallest
> possible learning mechanism installed into them, and a protocol that
> measures *adaptation* rather than performance on a benchmark.

<p align="center">
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-brightgreen.svg"></a>
  <a href="https://www.python.org/"><img alt="Python 3.8+" src="https://img.shields.io/badge/python-3.8%2B-blue.svg"></a>
  <a href="https://github.com/NullLabTests/unknown-world/blob/main/README.md"><img alt="Experiment" src="https://img.shields.io/badge/latest%20experiment-002-orange.svg"></a>
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

This repository is that program, experiment by experiment: the smallest
possible artificial world, a minimal agent, and an evaluation protocol built
to detect generalization — or the lack of it.

## What the world is

An `UnknownWorld` is a small population of objects with visible features:

```
color ∈ {red, blue, green}
shape ∈ {round, square, star}
```

Exactly one `{feature: value}` pair is the **hidden rule**: objects matching
it open the door, all others do not. Per world the agent sees 6 presented
objects and is later graded on 3 held-out objects (at least one match and one
non-match, when the geometry allows). The generator offers a `force_feature`
hook so the protocol can constrain the hidden rule's feature while value and
object population stay randomized. The rule is never revealed to the agent:

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

Since Experiment 002, ACT weighs each surviving hypothesis by the salience of
its feature:

- Dirichlet-style counts `alpha[feature]`, initialized `1.0` (uniform);
  `reset_salience()` restores uniformity.
- After a world converges to a single feature rule, `alpha[rule.feature]`
  is incremented. No convergence, no update. Salience (but not the
  hypothesis set) carries across worlds within a run; nothing is persisted.
- `salience()` returns normalized feature weights; information-gain scores
  use those masses instead of `1/|H|`. Low-salience features are never
  hard-banned — a rule on a low-weight feature must remain solvable, at a
  cost the protocol is designed to see.

## Protocol — rototest

We never declare intelligence from a fixed benchmark. Competence is measured
under change:

| Measure | What it detects |
| --- | --- |
| **Within-world** — held-out answer accuracy as the hypothesis set narrows | the competence curve (`67% → 100%`) |
| **Across-world** — experiments-to-converge, world after world, under controlled generators | the learning-to-learn signal |

`run.py` prints evidence from three blocks, the verdict computed by the
harness, not by hand:

- **Block A — stationary family**: learn 5 color-rule worlds, then 5 more
  color-rule worlds (novel values/populations), one agent, salience carries.
  A working prior shows transfer n_exp ≤ learning n_exp.
- **Block B — family switch**: the same agent then faces 5 shape-rule worlds.
  A real bias pays a on-first-switch cost, then recovers as `shape` counts
  catch up. If held-out fails to reach 100% or the loop never converges,
  that is a fatal reject.
- **Block C — mixed control**: a fresh agent, 5+5 unconstrained worlds. The
  mixed mean must not degrade versus the 001 baseline; end-of-run salience
  should roughly track empirical feature frequency.

## Experiment 001 — results (historical)

Measured under the original schema (9 objects, 8 presented / 1 held-out,
`Object` with a `features` dict). The schema has since been superseded; this
table is the record of that run.

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

The loop converges to the true rule and answers correctly about novel objects
in every world, but experiments-to-converge is **flat** across phases: no
learning-to-learn. Competent, not generalizing. That flat row isolated the
first missing primitive — a prior over which *dimensions* tend to be
diagnostic — and defined Experiment 002.

## Experiment 002 — feature-salience prior

Hypothesis: a Dirichlet count over *features*, updated on convergence and
used as hypothesis mass in information-gain ACT, should bend the across-world
n_exp row when the diagnostic dimension is stationary, and show a transient
cost when it switches.

Protocol addition: Block A color-heavy learn/transfer; Block B inherited
switch to shape; Block C fresh mixed control.

Results (`python3 run.py`, seed 7, verbatim):

```
The Unknown World — Experiment 002 (feature-salience prior)

  # block            hidden rule           tests held-out acc  competence curve                 n_exp  salience
------------------------------------------------------------------------------------------------------------------------
  A-learn-color 1  color == 'blue'           4      100.0%  67% -> 67% -> 67% -> 67% -> 100% -> 100%     4  {color=0.667, shape=0.333}
  A-learn-color 2  color == 'blue'           2      100.0%  67% -> 100% -> 100% -> 100%          2  {color=0.750, shape=0.250}
  A-learn-color 3  color == 'red'            3      100.0%  67% -> 67% -> 67% -> 100% -> 100%     3  {color=0.800, shape=0.200}
  A-learn-color 4  color == 'blue'           3      100.0%  67% -> 67% -> 100% -> 100% -> 100%     3  {color=0.833, shape=0.167}
  A-learn-color 5  color == 'red'            4      100.0%  67% -> 67% -> 67% -> 67% -> 100% -> 100%     4  {color=0.857, shape=0.143}
  A-xfer-color 1   color == 'red'            2      100.0%  67% -> 67% -> 100% -> 100%           2  {color=0.875, shape=0.125}
  A-xfer-color 2   color == 'red'            2      100.0%  67% -> 67% -> 100% -> 100%           2  {color=0.889, shape=0.111}
  A-xfer-color 3   color == 'blue'           2      100.0%  67% -> 100% -> 100% -> 100%          2  {color=0.900, shape=0.100}
  A-xfer-color 4   color == 'blue'           3      100.0%  67% -> 67% -> 67% -> 100% -> 100%     3  {color=0.909, shape=0.091}
  A-xfer-color 5   color == 'red'            2      100.0%  67% -> 100% -> 100% -> 100%          2  {color=0.917, shape=0.083}
  B-switch-shape 1 shape == 'round'          3      100.0%  67% -> 100% -> 33% -> 100% -> 100%     3  {color=0.846, shape=0.154}
  B-switch-shape 2 shape == 'round'          3      100.0%  67% -> 67% -> 33% -> 100% -> 100%     3  {color=0.786, shape=0.214}
  B-switch-shape 3 shape == 'round'          3      100.0%  67% -> 67% -> 67% -> 100% -> 100%     3  {color=0.733, shape=0.267}
  B-switch-shape 4 shape == 'square'         3      100.0%  67% -> 67% -> 67% -> 100% -> 100%     3  {color=0.688, shape=0.312}
  B-switch-shape 5 shape == 'star'           3      100.0%  67% -> 67% -> 67% -> 100% -> 100%     3  {color=0.647, shape=0.353}
  C-mixed-learn 1  color == 'red'            3      100.0%  67% -> 67% -> 67% -> 100% -> 100%     3  {color=0.667, shape=0.333}
  C-mixed-learn 2  shape == 'round'          3      100.0%  67% -> 100% -> 33% -> 100% -> 100%     3  {color=0.500, shape=0.500}
  C-mixed-learn 3  shape == 'star'           4      100.0%  67% -> 67% -> 67% -> 67% -> 100% -> 100%     4  {color=0.400, shape=0.600}
  C-mixed-learn 4  color == 'red'            3      100.0%  67% -> 67% -> 33% -> 100% -> 100%     3  {color=0.500, shape=0.500}
  C-mixed-learn 5  shape == 'round'          2      100.0%  67% -> 100% -> 100% -> 100%          2  {color=0.429, shape=0.571}
  C-mixed-xfer 1   shape == 'square'         3      100.0%  67% -> 67% -> 100% -> 100% -> 100%     3  {color=0.375, shape=0.625}
  C-mixed-xfer 2   shape == 'star'           3      100.0%  67% -> 67% -> 67% -> 100% -> 100%     3  {color=0.333, shape=0.667}
  C-mixed-xfer 3   shape == 'star'           3      100.0%  67% -> 67% -> 33% -> 100% -> 100%     3  {color=0.300, shape=0.700}
  C-mixed-xfer 4   shape == 'round'          3      100.0%  67% -> 67% -> 67% -> 100% -> 100%     3  {color=0.273, shape=0.727}
  C-mixed-xfer 5   color == 'blue'           3      100.0%  67% -> 100% -> 33% -> 100% -> 100%     3  {color=0.333, shape=0.667}
------------------------------------------------------------------------------------------------------------------------
  A learning (color) n_exp = [4, 2, 3, 3, 4]  mean=3.20
  A transfer (color) n_exp = [2, 2, 2, 3, 2]  mean=2.20
  B switch   (shape) n_exp = [3, 3, 3, 3, 3]  mean=3.00
  C mixed learn      n_exp = [3, 3, 4, 3, 2]  mean=3.00
  C mixed xfer       n_exp = [3, 3, 3, 3, 3]  mean=3.00
  held-out answers reach 100% in 25/25 worlds
  verdict: inconclusive
  Signals mixed (bent=True switch_cost=True recovered=False control_ok=True). Do not keep the primitive.
```

The verdict line is produced by the harness; it is not hand-edited.

**Measurement notes**

- Correctness holds: 25/25 worlds converge to the true rule and answer 100%
  on held-out objects, including shape-rule worlds faced after a
  color-heavy history. The prior does not break solvability.
- The prior is mechanically active: across 300 sampled worlds, experiment
  choice differed from the uniform-selience agent in 268/300. It is not
  inert decoration.
- The verdict is **seed-fragile**: over seeds 0–7 the harness returns
  kept ×2, inconclusive ×4, rejected ×2. The acceptance thresholds are set
  on roughly ±1 experiment with n=5 worlds per block — too fine a knife.
  At seed 7 the A-transfer row descended (3.20 → 2.20) but Block B showed
  no within-block recovery, so the harness withholds *kept*.

**Verdict, as measured: the primitive is not kept.**

## Next

Before the next candidate primitive is judged, the protocol itself has to
get more discriminating: larger block sizes or a multi-seed verdict so the
accept/reject line stops wobbling on ±1 experiment. Then the prior question
returns — with the transfer row bending (as it did at seed 7) *and* a
stable, measurable switch cost + recovery in Block B — or it is replaced by
a different mechanism.

> Rule: never declare intelligence from performance on a fixed benchmark.
> Measure the system's ability to adapt when the rules, tasks, environment,
> and available information change.

## Reproduce

```bash
python3 run.py                        # Experiment 002 and its verdict
python3 -m unittest discover -s .     # the sanity suite (11 tests)
```

Stdlib only. No dependencies. Deterministic seed (`random.Random(7)`).

## Layout

| File | Role |
| --- | --- |
| `world.py` | generator of minimal unknown-worlds (`Object` color/shape, hidden rule, `force_feature` hook) |
| `agent.py` | the loop + feature-salience prior (Dirichlet `alpha`, weighted ACT, `reset_salience()`) |
| `protocol.py` | rototest harness: Blocks A/B/C, evidence table, harness verdict |
| `run.py` | runs Experiment 002 and prints the evidence |
| `test_step1.py` | sanity suite (11 tests: world, salience, protocol) |

## License

MIT © 2026 NullLabTests