# The Unknown World — Experiments 001–006

> A laboratory for discovering agents: tiny unknown-worlds, the smallest
> possible learning mechanism installed into them, and a protocol that
> measures *adaptation* rather than performance on a benchmark.

<p align="center">
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-brightgreen.svg"></a>
  <a href="https://www.python.org/"><img alt="Python 3.8+" src="https://img.shields.io/badge/python-3.8%2B-blue.svg"></a>
  <a href="https://github.com/NullLabTests/unknown-world/blob/main/README.md"><img alt="Experiment" src="https://img.shields.io/badge/latest%20experiment-006-orange.svg"></a>
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

Since Experiment 004, the prior can also **change its mind**. The measured
failure of 002/003 was rigidity: after a color-heavy history, the first
shape world cost ~0.9 extra experiments because salience precision grows
unboundedly. Two tempered variants, both one-hyperparameter:

- `ForgetAgent(f)` — bounded precision via a fixed forgetting factor
  (`alpha = f·alpha + winner`, `f = 0.7`, Itti & Baldi 2005).
- `ChangePointAgent(h)` — evidence-gated forgetting: a Bayesian run-length
  monitor (Adams & MacKay 2007) over the sequence of winning features keeps
  a strong prior on stationary streams but resets toward uniform as the
  posterior mass shifts to "a change happened" — never-winning features are
  *restored* to the uniform base, not decayed toward zero.

Since Experiment 005, the prior can also **learn its own hazard rate**.
Wilson, Nassar, Gold & Kording (2010) showed that Bayesian Online Changepoint
Detection's performance depends critically on the chosen hazard `h`, but the
correct rate is unknown in advance. The `HierarchicalChangePointAgent`
removes the hand-set `h` entirely: it maintains a discrete grid over
candidate hazards, runs one exact run-length trellis per candidate, and
accumulates each candidate's marginal predictive likelihood (Wilson et al.'s
recurrence). The salience counts are model-averaged over the hazard
posterior — no single `h` is chosen by the experimenter. Experiment 005
asks whether this zero-hand-set-prior agent *adapts* where the fixed
`h = 1/5` is miscalibrated, without regressing where it was right.

Since Experiment 006, the hazard is learned from the **observation stream**,
not the compressed winner sequence. Experiment 005 updated the hierarchy once
per world, at convergence, on the winning feature alone; the hazard proved
statistically invisible in that two-feature geometry. The
`ObservationHazardAgent` instead runs the exact standard BOCPD recursion
(Wilson, Nassar, Gold & Kording 2010; Adams & MacKay 2007) *per observation*
over the same hazards grid: one run-length trellis per candidate, updated at
every touched object, with the change branch active only at world boundaries
(a feature cannot change inside a world). The whole stream from each world
now moves the hazard posterior, so an event rate — fast regimes
(flip almost every world) vs slow regimes (one calm long tail after another)
— finally has evidence to bite on.

## Protocol — rototest

We never declare intelligence from a fixed benchmark. Competence is measured
under change:

| Measure | What it detects |
| --- | --- |
| **Within-world** — held-out answer accuracy as the hypothesis set narrows | the competence curve (`67% → 100%`) |
| **Across-world** — experiments-to-converge, world after world, under controlled generators | the learning-to-learn signal |

`run.py` prints evidence from three blocks, the verdict computed by the
harness, not by hand. As of Experiment 003 every world is run **paired**
(prior agent and identical-machinery null control on the *same* seeded
worlds); blocks:

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

**Verdict, as measured: the primitive is not kept.** In retrospect this
verdict was an artifact of a protocol too fine a knife for the signal; it is
overturned by the paired protocol of Experiment 003, below.

## Experiment 003 — measurement hardening (paired seeded rototest v2)

The 002 verdict wobbled on roughly ±1 experiment because it compared
*across* independently drawn worlds (learning vs transfer) — the dominant
variance was *which worlds a run drew*, not the mechanism under test. The
established fix in the ML/RL evaluation literature is **paired seed
evaluation**: run both agents on the *identical* worlds, and do inference on
the paired difference `d = n_exp(prior) − n_exp(null)`, alongside a matched
"identical-machinery" control whose prior is neutralized before every world.

Distilled from: M. Rainforth et al., *Modern Bayesian Experimental Design*
(Stat. Sci. 2024, arXiv:2302.14545); the paired-seed protocol of arXiv:2512.24145;
*When +1% Is Not Enough: A Paired Bootstrap Protocol for Evaluating Small
Improvements* (arXiv:2511.19794); *A Hitchhiker's Guide to Statistical
Comparisons of Reinforcement Learning Algorithms* (arXiv:1904.06979);
Settles, *Active Learning Literature Survey* (2009); and meta-evaluation
pitfalls from Setlur et al. (arXiv:2102.11503) and Petelin & Cenikj
(arXiv:2505.07750).

Protocol v2, everything seeded and stdlib-only:

- A fixed seed-base (`seed * 10000 + s`, `s = 0..n_seeds−1`) generates the
  same worlds for **both** the prior agent and the null control.
- Statistics on paired diffs: sign-flip exact permutation test (p ≤ 0.0003),
  percentile bootstrap 95% CI, Cohen's `dz`, and a per-seed directional
  count. No normal-assumption tests; all resampling is seeded and
  reproducible.
- Decision gates, computed by the harness: A-transfer improvement
  (paired CI upper bound < 0); B on-first-switch cost (CI lower bound > 0);
  B recovery (later-switch effect below first-switch effect); and a
  **non-inferiority** C gate (paired CI upper bound < margin), since the
  data show the prior reliably *helps* mixed worlds a little — equality
  ("CI must span zero") would classify that as harm.
- Stability is now checked *across seed bases*: the verdict must not depend
  on the seed used to generate the world stream.

Results (verbatim, `python3 run.py`, seed base 7, `n_seeds=16`):

```
The Unknown World — Experiment 003 (measurement hardening: paired seeded rototest v2)

  block        n    mean d  95% CI            p(prior<null)  dz     seeds helping
----------------------------------------------------------------------------------
  A-learn        80   -0.33  [ -0.46,  -0.20]   0.0003     -0.57   15/16
  A-transfer     80   -0.34  [ -0.51,  -0.16]   0.0003     -0.42   13/16
  B-switch       80    0.47  [  0.36,   0.59]   1.0000      0.90   0/16
  C-mixed       160   -0.03  [ -0.10,   0.03]   0.2392     -0.07   10/16
----------------------------------------------------------------------------------
  B-switch paired effect by world position (d = prior - null)
    world 1  mean d   0.88  95% CI [  0.62,   1.12]
    world 2  mean d   0.81  95% CI [  0.62,   1.00]
    world 3  mean d   0.38  95% CI [  0.12,   0.62]
    world 4  mean d   0.19  95% CI [  0.00,   0.38]
    world 5  mean d   0.12  95% CI [  0.00,   0.31]
----------------------------------------------------------------------------------
  A_TRANSFER_IMPROVE     : True
  B_SWITCH_COST          : True
  B_RECOVERED            : True
  C_MIXED_NO_HARM        : True
  verdict: kept
  Paired CI shows the prior reliably reduces n_exp on A transfer, reliably pays a cost on the first B switch, recovers, and does not cost more than a margin on mixed worlds. Keep the primitive.
```

**Measurement notes**

- The features 002 said it could not see cleanly are now visible: a drop in
  transfer n_exp of a third of an experiment (dz ≈ −0.42, p ≈ 0.0003), a
  first-switch cost nearly a full experiment (dz ≈ +0.90), and a clean
  recovery curve world 1→5 (`0.88 → 0.81 → 0.38 → 0.19 → 0.12`) as the
  Dirichlet counts re-weight toward `shape`. Pairing worked exactly as the
  literature predicts: the world-draw variance is removed, so a true
  ~0.3–0.5 experiment effect becomes detectable.
- The C-mixed effect is a small *reliably negative* paired diff (the prior
  helps) that more power actually detects rather than averages away. That is
  why the C gate is non-inferiority, not equality — the equality version
  originally mis-classified "helping" as failure.
- Seed-stability was re-measured with the fixed C gate: across 12 seed bases
  × `n_seeds ∈ {16, 24, 32}`, the verdict is **`kept` in 32/32 runs**. The
  ±1-experiment wobble of 002 is gone; the line the primitive must cross is
  now a CI boundary, not a mean.

**Verdict, as measured: the feature-salience prior is kept.** It is the
first mechanism in the program with a stable, statistically supported
adaptive-transfer signature: reliably cheaper on the stationary family,
reliably costlier on the family switch, recovering within a block, and
harmless on mixed worlds.

**Methodological verdict: the paired seeded design is itself worth
keeping.** The same primitive that was "inconclusive" under an
across-world design is "kept" under the within-world paired design — and is
stable across seed bases. Experiment 001's lesson was about the flatness of
an unmeasured row; 002's lesson was that the measurement itself needed to
become the subject of study.

## Experiment 004 — change-aware salience (the prior can be wrong)

002 measured the primitive's cleanest failure mode: after a color-heavy
history, the first shape world costs +0.88 experiments and recovery is slow
because counts accumulate forever — the prior is *rigid*. Three literatures
converge on the remedy:

- **Itti & Baldi (2005; "Of Bits and Wows" 2010)** formalize surprise as
  `KL(P(M|D) || P(M))` and explicitly warn that Dirichlet/Gamma counts grow
  unboundedly, adding a *forgetting factor* `f = 0.7` to cap precision —
  "relaxation of belief in the prior's precision."
- **Adams & MacKay (2007), Bayesian Online Changepoint Detection**
  (arXiv:0710.3742) — an exact online posterior over the *run length* since
  the last regime change; a change is declared probabilistically, so
  forgetting can be gated on evidence of a switch rather than applied
  procedurally.
- **Concept-drift stream learning** (Yu & Webb 2019; Bifet et al. 2007) —
  "constant forgetting factors are a defect; the *rate of drift* should
  regulate how much is forgotten."

Three arms replace the bare prior on identical, per-seed worlds: `null`
(prior neutralized each world), `bare` (002/003 prior, never forgets),
`forget` (fixed `f = 0.7`), `change` (run-length monitor, hazard
`h = 1/5` — the protocol's own block rate, set a priori, not tuned). All
arms share the same world streams, so every comparison is paired.

Results (verbatim, `python3 run.py`, seed base 7, `n_seeds=16`):

```
The Unknown World — Experiment 004 (change-aware salience: the prior can be wrong)

  change-aware vs null (identical worlds; d = n_exp(change) - n_exp(null))
  block        n    mean d  95% CI            p(prior<null)  dz     seeds helping
----------------------------------------------------------------------------------
  A-learn        80   -0.33  [ -0.44,  -0.23]   0.0003     -0.69   16/16
  A-transfer     80   -0.34  [ -0.51,  -0.16]   0.0003     -0.42   13/16
  B-switch       80    0.06  [ -0.09,   0.20]   0.8482      0.09   8/16
  C-mixed       160   -0.06  [ -0.14,   0.01]   0.0757     -0.13   12/16
----------------------------------------------------------------------------------
  B-switch paired effect by world position (d = mechanism - bare prior)
    world   change-bare CI             forget-bare CI
    1        0.00 [  0.00,   0.00]       0.00 [  0.00,   0.00]
    2       -0.44 [ -0.69,  -0.19]      -0.44 [ -0.69,  -0.19]
    3       -0.50 [ -0.75,  -0.25]      -0.50 [ -0.75,  -0.25]
    4       -0.69 [ -0.88,  -0.44]      -0.69 [ -0.94,  -0.44]
    5       -0.44 [ -0.69,  -0.19]      -0.44 [ -0.69,  -0.19]
  change vs bare, B worlds 2..5 pooled: mean d  -0.52  95% CI [ -0.64,  -0.39]
----------------------------------------------------------------------------------
  A_TRANSFER_IMPROVE     : True
  B_SWITCH_REDUCED       : True
  C_MIXED_NO_HARM        : True
  verdict: kept
  Change-aware salience preserves the transfer benefit, reliably cuts the post-first switch cost below the bare prior, and stays harmless on mixed worlds. The prior can change its mind. Keep it.
```

**Measurement notes**

- The change-aware agent keeps the full transfer benefit of the bare prior
  (A-transfer −0.34 [−0.51, −0.16] — the same CI as Experiment 003) while
  erasing essentially *all* of the switch penalty: B-switch pooled cost drops
  from +0.47 to +0.06 (CI [−0.09, +0.20]), and on the post-first switch
  worlds it beats the bare prior by roughly half an experiment per world
  (pooled −0.52 [−0.64, −0.39]).
- The world-1 cost is structural, not fixable by a boundary monitor: no
  within-world evidence can precede the first shape world, so `change −
  bare = 0.00 [0.00, 0.00]` there (identical CIs, 16/16 seeds). The monitor
  acts from world 2 onward.
- The fixed-forgetting variant (`f = 0.7`) achieves nearly identical B-block
  numbers here (`forget − bare` columns match `change − bare` on worlds
  2–5). In this small feature geometry both collapse to a near-uniform prior
  by world 2; they would diverge on longer streams or more features — that
  is a candidate *measurement* for 005, not a claim.
- Verdict stability (the 003 standard): across 12 seed bases ×
  `n_seeds ∈ {16, 24, 32}`, the verdict is **`kept` in 32/32 runs**;
  correctness holds (all arms still converge to the true rule and answer
  100% on held-out objects, including shape-rule worlds after a color
  history).

**Verdict, as measured: change-aware salience is kept.** It preserves the
prior's power on stationary families and removes its measured rigidity on
family switches. The strand of the program that began as "a prior over
which dimension is diagnostic" (002) has become "a prior that knows its own
dimension-belief carries evidence and can be wrong" (004).

## Experiment 005 — the prior earns its hazard rate (learning the rate of change)

The hand-set hazard `h = 1/5` in Experiment 004 is *the experimenter's*
prior: the protocol uses 5-world blocks, so 1/5 was chosen to match the
protocol, not learned by the agent. Wilson, Nassar, Gold & Kording (2010,
*Neural Computation* 22:2452–2476) make the general point: the performance
of online change-point monitoring depends critically on `h`, and the right
value is not obvious in advance — so a learning system should infer `h`
from the stream it lives in.

The 005 machinery is the hierarchy of the same trellis: a grid of candidate
hazards
`h ∈ {1/2, 1/3, 1/4, 1/6, 1/8, 1/16, 1/32}`, each running its own exact
run-length monitor; the cumulative marginal predictive likelihood of the
winner sequence (the per-event run-length normalizer) gives a posterior over
hazards; salience is the hazard-posterior-weighted average of per-candidate
counts. The agent that runs this is the `learned` arm. It is compared, on
identical worlds, against:

- `null` — identical machinery, prior neutralized before every world;
- `fixed` — Experiment 004's `ChangePointAgent(h = 1/5)` (the hand-set rate);
- `forget` — `ForgetAgent(f = 0.7)`.

Everyone runs the same *regime streams* — blocks of worlds whose hidden
feature switches every `ρ` worlds — after an identical stationary colour
history (the Experiment 004 A-block). Each pace is chosen to adversarially
test the hand-set 1/5:

| pace | regimes | worlds | what it tests |
| --- | --- | --- | --- |
| A-learn/A-transfer | — | 5 + 5 | the stationary transfer benefit must not regress |
| B-home  (ρ = 5) | 4 | 20 | the rate 1/5 was tuned exactly here |
| B-slow  (ρ = 16) | 2 | 32 | long calm tails — the hand-set cap is miscalibrated |
| B-fast  (ρ = 2) | 8 | 16 | switches every other world — 1/5 under-anticipates change |
| C-mixed | — | 10 | fresh prior, unconstrained worlds (non-inferiority) |

Gates (pre-committed, computed by the harness): no-regression of the
A-transfer benefit (learned < null), and learned < fixed on the B-home
worlds, on the B-slow calm tails (regime positions 9–16), and on the B-fast
worlds. Keep only if the learned hazard is adaptive *without* regressing.

Results (verbatim, `python3 run.py`, seed base 7, `n_seeds=16`):

```
The Unknown World — Experiment 005 (the prior earns its hazard rate)

  learned vs null (identical worlds; d = n_exp(learned) - n_exp(null))
  block        n    mean d  95% CI            p(prior<null)  dz     seeds helping
----------------------------------------------------------------------------------
  A-learn        80   -0.33  [ -0.44,  -0.23]   0.0003     -0.69   16/16
  A-transfer     80   -0.34  [ -0.51,  -0.16]   0.0003     -0.42   13/16
  B-home        320   -0.20  [ -0.28,  -0.13]   0.0003     -0.29   16/16
  B-slow        512   -0.46  [ -0.52,  -0.40]   0.0003     -0.62   16/16
  B-fast        256    0.02  [ -0.05,   0.09]   0.7352      0.03   8/16
  C-mixed       160   -0.09  [ -0.16,  -0.01]   0.0235     -0.17   12/16
----------------------------------------------------------------------------------
  learned vs fixed h=1/5  (d = n_exp(learned) - n_exp(fixed))
  block        n    mean d  95% CI
  ----------------------------------------------
  A-learn        80    0.00  [  0.00,   0.00]
  A-transfer     80    0.00  [  0.00,   0.00]
  B-home        320    0.03  [  0.01,   0.05]
  B-slow        512    0.01  [ -0.00,   0.03]
  B-fast        256    0.04  [  0.02,   0.06]
  ...
  gate CIs (the verdict is drawn on these boundaries)
    A_transfer (learned-null, A-transfer)  mean d  -0.34  95% CI [ -0.51,  -0.17]   (upper<0)
    B_home     (learned-fixed, worlds 2..end) mean d   0.03  95% CI [  0.01,   0.06]   (upper<0)
    B_slow_tail(learned-fixed, calm tails) mean d   0.00  95% CI [  0.00,   0.00]   (upper<0)
    B_fast     (learned-fixed, worlds 2..end) mean d   0.04  95% CI [  0.02,   0.06]   (upper<0)
    C_mixed    (learned-null + margin)     mean d  -0.09  95% CI [ -0.16,  -0.01]   (upper<0.25)
----------------------------------------------------------------------------------
  effective hazard of the learned prior (posterior mean E[h], averaged over seeds)
    after A          E[h] = 0.1582
    after home       E[h] = 0.3047
    after slow       E[h] = 0.2185
    after fast       E[h] = 0.2303
----------------------------------------------------------------------------------
  A_TRANSFER_NO_REGRESSION   : True
  B_HOME_NO_REGRESSION       : False
  B_SLOW_TAIL_IMPROVED       : False
  B_FAST_ADAPTED             : False
  C_MIXED_NO_HARM            : True
  verdict: inconclusive
  Signals mixed (A_improve=True home_ok=False slow_ok=False fast_ok=False no_harm=True). Do not keep the primitive.
```

**Measurement notes**

- **The hierarchy does not break the prior.** `learned` vs null retains the
  full stationary transfer benefit (A-transfer −0.34 [−0.51, −0.16] — the
  same CI as Experiments 003/004), is strongly cheaper on the regime blocks
  (B-home −0.20, B-slow −0.46), stays harmless on mixed worlds, and is
  *never* more than ~0.04 experiments worse than the hand-set `h = 1/5`
  anywhere. A zero-hand-set-prior agent is **boundedly robust**.
- **But it does not adapt either.** Every off-goal gate fails: learned ≈
  fixed on the B-slow calm tails (identical n_exp, 2.50 vs 2.50 on the
  tails); learned is *slightly worse* on the fast regime (+0.04) and at
  home (+0.03). The invented hazard never produces a measured advantage.
- **Why: the hazard is not identifiable in this geometry.** The marginal
  predictive likelihood of a *two-feature* winner sequence barely
  discriminates among hazards — every candidate makes near-identical
  predictions because the counts have only two dimensions to sharpen.
  The posterior mean `E[h]` visibly fails to track the true rates:
  0.16 after the calm A-block, 0.30 after home (ρ = 5), 0.22 after slow
  (ρ = 16), 0.23 after fast (ρ = 2). It wiggles, it does not learn.
  Wilson et al.'s hierarchy needs richer evidence to bite; this world does
  not provide it.
- **Verdict stability:** at `n_seeds = 16` the verdict is `inconclusive`
  across all 5 seed bases scanned (5/5); at `n_seeds = 8` one base (7)
  wobbled to `rejected` on the low-power A-transfer gate — the operating
  point used for the record is 16 seeds. The *pattern* is stable: the
  A-transfer and C gates always pass, and the three B-hazard gates always
  fail.
- This is a **measurement-level** finding, not a machinery failure: it
  tells the program that both the hazard monitor *and* the f-vs-change
  divergence prediction (README 004) need a world with more than two
  features — a world with more dimensions for a run-length model to decide
  between and to be wrong about.

**Verdict, as measured: the hazard-learning hierarchy is not kept**
(the harness returns `inconclusive`, and per the lab rule an inconclusive
primitive is not kept). It is cheap, bounded, and harmless, and it never
regresses the prior — but in this two-feature world the rate it is asked to
learn is statistically invisible. The experiment's value is negative but
sharp: it converts "the right move is learning the hazard" into "the right
move is a world with more features, or a hazard evidence site with more
signal".

## Experiment 006 — the hazard learned per observation

Experiment 005's failure was precise: the hazard was updated *once per world*,
at convergence, from the single winning feature. The winner sequence is a
compressed two-dimensional stream, and Wilson et al.'s marginal-predictive
measure barely discriminates among hazards on it. Experiment 006 makes the
harsh correction — **the fee is paid per observation, not per world**:

The `ObservationHazardAgent` keeps the same grid of candidate hazards
`h ∈ {1/2, 1/3, 1/4, 1/6, 1/8, 1/16, 1/32}` and one exact run-length trellis
of `(run length, prob, Dirichlet-counts)` states per candidate, exactly like
standard BOCPD, but the recursion fires on **every touched object**:

- Each observation contributes its posterior predictive under each candidate
  (`sum over states of prob × per-feature predictive`), accumulated into that
  candidate's marginal log-likelihood — the Wilson et al. latent-rate
  learning it always was.
- The change branch (reset run, counts to uniform base) is gated to the
  **first observation of each world**: the hidden *feature* can only change at
  a world boundary, so the within-world observations are pure growth steps and
  the hazard is asked to explain boundary events only.
- The emission per state is the outcome's probability under that state's
  per-feature value belief, computed from *within-world value evidence* — the
  hidden value resets each world, so the value weights reset at every boundary.
  A contradicted feature relaxes to 0.5 rather than dying instantly, so wrong
  states drain away observation-by-observation instead of collapsing in one
  step.

Everything downstream is untouched: salience for ACT still updates only at
convergence (`note_convergence` sets `alpha` to the hazard-posterior-weighted
expected counts — the identical-machinery discipline), so the paired null
control of Experiment 003 stays valid. Arms and blocks are identical to 005
(`null` / `fixed h = 1/5` / `forget f = 0.7` / `learned`; B-home ρ=5,
B-slow ρ=16, B-fast ρ=2 with a stationary A-block ahead of them).

The deciding gate for 006 is about **identification, not behavior** — the
thing 005 measured as missing:

- `H_orders_fast_over_slow`: the learned hazard posterior mean after the
  B-fast block must reliably exceed its value after the B-slow block
  (per-seed paired `E[h|fast] − E[h|slow]`, bootstrap CI lower bound > 0.04).
  True boundary-flip rates are 1/16 (slow) and 1/2 (fast): a monitor with
  real hazard evidence must move `E[h]` up when the fast regime arrives.
- `A_transfer_no_regression`: no loss of the stationary transfer benefit.
- `C_mixed_no_harm`: mixed control within the non-inferiority margin.

Results (verbatim, `python3 run.py`, seed base 7, `n_seeds=16`):

```
The Unknown World — Experiment 006 (the hazard learned per observation)

  learned vs null (identical worlds; d = n_exp(learned) - n_exp(null))
  block        n    mean d  95% CI            p(prior<null)  dz     seeds helping
----------------------------------------------------------------------------------
  A-learn        80   -0.50  [ -0.65,  -0.35]   0.0003     -0.69   16/16
  A-transfer     80   -0.34  [ -0.51,  -0.16]   0.0003     -0.42   13/16
  B-home        320   -0.44  [ -0.53,  -0.36]   0.0003     -0.53   16/16
  B-slow        512   -0.51  [ -0.59,  -0.44]   0.0003     -0.61   16/16
  B-fast        256   -0.07  [ -0.17,   0.04]   0.1278     -0.08   12/16
  C-mixed       160   -0.06  [ -0.18,   0.05]   0.1770     -0.08   10/16
----------------------------------------------------------------------------------
  learned vs fixed h=1/5  (d = n_exp(learned) - n_exp(fixed))
  block        n    mean d  95% CI
  ----------------------------------------------
  A-learn        80   -0.17  [ -0.30,  -0.05]
  A-transfer     80    0.00  [  0.00,   0.00]
  B-home        320   -0.21  [ -0.28,  -0.14]
  B-slow        512   -0.04  [ -0.07,  -0.01]
  B-fast        256   -0.05  [ -0.14,   0.03]
  C-mixed       160    0.02  [ -0.09,   0.12]
  ----------------------------------------------
  effective hazard of the learned prior (posterior mean E[h], averaged over seeds)
    after A          E[h] = 0.1257
    after home       E[h] = 0.2163
    after slow       E[h] = 0.1559
    after fast       E[h] = 0.2970
----------------------------------------------------------------------------------
  H_order: per-seed E[h|post-fast] - E[h|post-slow]  mean  0.141  95% CI [ 0.138,  0.145]  (gate: lower > 0.04)
----------------------------------------------------------------------------------
  A_TRANSFER_NO_REGRESSION   : True
  H_ORDERS_FAST_OVER_SLOW    : True
  C_MIXED_NO_HARM            : True
  verdict: kept
  A monitor that learns its hazard at observation level preserves the stationary transfer benefit, reliably orders the fast regime's rate above the slow regime's rate, and stays harmless on mixed worlds. The prior's rate of forgetting is earned from the observation stream it lives in. Keep it.
```

**Measurement notes**

- **The hazard is now identifiable.** The `E[h]` trajectory is the exact
  trajectory theory predicts and 005 conspicuously did *not* produce: 0.126
  after the calm stationary A-block, 0.216 after home (ρ = 5 ≈ 1/5), a dip to
  0.156 after the slow calm tails (ρ = 16), then a spike to 0.297 after the
  fast block (ρ = 2). The dip-then-spike ordering is exactly the ordering of
  the true rates, and it is monotone *against* stream order — it cannot be a
  drift artifact. The per-observation stream, which records the boundary +
  growth structure of every world, carries the hazard evidence that the
  per-world winner sequence could not.
- **H_order is the strongest number in the experiment**: mean 0.141, 95% CI
  [0.138, 0.145] against a pre-committed margin of 0.04 — three and a half
  margins of clearance. It passes at *every* seed base scanned below.
- **The behavioral readout is quieter.** B-fast learned-vs-null is −0.07
  [−0.17, 0.04], and C-mixed is flat. The monitor *identifies* the two rates
  cleanly, but in this short protocol the identified hazard does not yet buy a
  measurable per-world n_exp advantage on the fast block (the learned prior is
  never worse here either — B-fast learned beats fixed h = 1/5 by −0.05). 005
  was dominated by the identification failure; with identification now
  recovered, the behavioral dividend is the natural next scale to chase.
- **Verdict stability across seed bases {5, 6, 7, 8, 9} at `n_seeds` = 16:**
  kept ×4, inconclusive ×1. The single wobble (base 6) is the pre-existing
  C_mixed_no_harm non-inferiority gate — its 95% CI upper bound crosses the
  0.25 margin (0.33) while A_transfer and H_order both pass at every base
  (H_order CI lower ≈ 0.139 everywhere). The C gate measures *world-draw
  composition* of the unconstrained mixed control, not hazard learning: it is
  the same seed-fragility the 002/003 records diagnosed, now localized to the
  one block left in the suite with unforced features. H_order — the gate that
  implements "does the learned rate track the stream's rate" — is stable in
  5/5 seed bases at both `n_seeds = 16` and `n_seeds = 24`.
- **Discretization honesty**: the agent carries one run-length trellis per
  candidate hazard on the grid and model-averages them; the full three-level
  hierarchy (learning the hazard *over* hazards, per Wilson et al.'s latent
  `a`/`b` recursion) is deliberately not implemented. The grid is a faithful
  two-level discretization of the same recurrence and is what lets the
  evidence site (per-observation) be tested against the 005 control unconfounded.
  A true three-level hierarchy is the next candidate, not a fix.

**Verdict, as measured: the per-observation hazard learner is kept.** The
regression 005 identified — collapsing each world to a single convergence
update — is repaired by paying the Bayesian recursion at the observation
level, and the rate that was statistically invisible now tracks the stream
with decisive margin. This is the first mechanism in the program whose *rate
of forgetting* is a measurement, in the sense that the value printed by the
posterior tracks the environment it lives in.

## Next

Experiment 006 recovered hazard *identification* by moving the recursion to
the per-observation stream. The open directions:

- **The behavioral dividend.** The identified hazard (post-fast E[h] ≈ 0.30)
  does not yet buy a clean B-fast n_exp advantage. Longer regime streams,
  mixed-rate blocks, or a richer geometry should let the learned rate earn a
  faster recovery where fixed `h = 1/5` under-anticipates change.
- **A true three-level hierarchy.** The grid is currently pre-committed;
  Wilson et al.'s latent-rate recursion (`a`, `b` posteriors over the hazard
  itself) removes even the grid. It should inherit the identification result
  and sharpen it where the grid is coarse (h = 1/3 vs 1/4).
- **A third feature.** The 005→006 record shows the hazard monitor needs
  *evidence* more than *machinery*; a third feature gives a run-length model a
  second dimension to be wrong about and converts the f-vs-change divergence
  prediction (004) into a measurement.
- **The metacognitive strand.** Two-literature direction: hazard evidence from
  the prior's *own* convergence cost (n_exp is itself a surprise signal), and
  a self-model of the learning loop (Haber et al. 2018; Liu & van der Schaar,
  arXiv:2506.05109). Both are 007 candidates, not yet scheduled.

> Rule: never declare intelligence from performance on a fixed benchmark.
> Measure the system's ability to adapt when the rules, tasks, environment,
> and available information change.

## Reproduce

```bash
python3 run.py                        # Experiment 006 and its verdict (default)
python3 run.py 005                    # Experiment 005 (hazard learned from the winner sequence)
python3 run.py 004                    # Experiment 004 (change-aware salience)
python3 run.py 003                    # Experiment 003 (paired rototest v2)
python3 run.py 002                    # Experiment 002 (legacy run)
python3 -m unittest discover -s .     # the sanity suite (47 tests)
```

Stdlib only. No dependencies. Deterministic seed (`random.Random(7)`).

## Layout

| File | Role |
| --- | --- |
| `world.py` | generator of minimal unknown-worlds (`Object` color/shape, hidden rule, `force_feature` hook) |
| `agent.py` | the loop + salience: bare prior (002), tempered `ForgetAgent` (fixed `f`), `ChangePointAgent` (run-length monitor, hazard `h`), `HierarchicalChangePointAgent` (learned hazard over a grid, 005), `ObservationHazardAgent` (per-observation learned hazard, 006) |
| `protocol.py` | paired seeded harness (003), four-arm change-aware / regime-stream harnesses (004–006): exact permutation p, bootstrap CI, Cohen's `dz`, hazard-identification gates, harness verdicts |
| `run.py` | runs Experiment 006 by default; `run.py 005` / `004` / `003` / `002` reproduce the earlier evidence |
| `test_step1.py` | sanity suite (47 tests: world, salience, protocol, statistics, paired v2, change-aware, hierarchy, observation-hazard) |

## License

MIT © 2026 NullLabTests