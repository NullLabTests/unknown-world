"""Rototest: within-world competence curve + across-world n_exp under controlled generators.

v2 (Experiment 003) adds a *paired* design: every world is seen by both the
prior-bearing agent and an identical-machinery control whose prior is
neutralized before each world. Paired differences remove the dominant variance
source (which worlds a run happened to draw), making ~1-experiment effects
detectable. Inference is seeded and stdlib-only (sign-flip permutation test,
percentile bootstrap CI, Cohen's dz).
"""
from __future__ import annotations

import random
import statistics

from agent import (
    Agent,
    ForgetAgent,
    ChangePointAgent,
    HierarchicalChangePointAgent,
    ObservationHazardAgent,
)
from world import UnknownWorld

BLOCKS = ("A-learn", "A-transfer", "B-switch", "C-mixed")


def fmt_rule(rule):
    if rule is None:
        return "?"
    return "%s == %r" % (rule[0], rule[1])


def fmt_curve(curve):
    return " -> ".join("%d%%" % int(round(x)) for x in curve)


def fmt_salience(sal):
    parts = ["%s=%.3f" % (k, sal[k]) for k in sorted(sal)]
    return "{" + ", ".join(parts) + "}"


def run_block(name, agent, rng, n_worlds, force_feature=None, start_index=1):
    rows = []
    n_exps = []
    for i in range(n_worlds):
        world = UnknownWorld.generate(rng, force_feature=force_feature)
        result = agent.run_world(world)
        n_exps.append(result["steps"])
        rows.append({
            "block": name,
            "index": start_index + i,
            "hidden": world.hidden_rule,
            "steps": result["steps"],
            "held_out": result["held_out"],
            "curve": result["curve"],
            "found": result["rule"],
            "salience": result["salience"],
        })
    return rows, n_exps


def mean(xs):
    return sum(xs) / float(len(xs)) if xs else 0.0


def interpret(block_a_learn, block_a_xfer, block_b_shape, block_c_all, baseline=3.0):
    a_l, a_t = mean(block_a_learn), mean(block_a_xfer)
    b_first = mean(block_b_shape[:1]) if block_b_shape else 0.0
    b_last = mean(block_b_shape[-2:]) if len(block_b_shape) >= 2 else mean(block_b_shape)
    c_mean = mean(block_c_all)

    bent = a_t + 1e-9 < a_l and a_t < baseline - 0.05
    switch_cost = b_first > a_t + 0.05
    recovered = b_last + 1e-9 <= b_first
    control_ok = c_mean <= baseline + 0.51

    if bent and switch_cost and recovered and control_ok:
        verdict = "kept"
        why = (
            "Block A transfer mean n_exp fell below learning and below the 001 baseline; "
            "Block B paid a switch cost then recovered; mixed control did not degrade."
        )
    elif not bent and abs(a_t - a_l) < 0.51 and abs(a_t - baseline) < 0.51:
        verdict = "rejected"
        why = (
            "Block A transfer row stayed flat at the 001 baseline. "
            "Salience did not produce a learning-to-learn signal."
        )
    else:
        verdict = "inconclusive"
        why = (
            "Signals mixed (bent=%s switch_cost=%s recovered=%s control_ok=%s). "
            "Do not keep the primitive."
            % (bent, switch_cost, recovered, control_ok)
        )
    return verdict, why, {
        "A_learn": a_l,
        "A_xfer": a_t,
        "B_first": b_first,
        "B_last": b_last,
        "C_mean": c_mean,
        "bent": bent,
        "switch_cost": switch_cost,
        "recovered": recovered,
        "control_ok": control_ok,
    }


def run_rototest(seed=7, n=5):
    import random

    rng = random.Random(seed)
    rows = []

    agent_a = Agent()
    r, a_learn = run_block("A-learn-color", agent_a, rng, n, force_feature="color", start_index=1)
    rows.extend(r)
    r, a_xfer = run_block("A-xfer-color", agent_a, rng, n, force_feature="color", start_index=1)
    rows.extend(r)

    # Block B inherits salience from A (same agent)
    r, b_shape = run_block("B-switch-shape", agent_a, rng, n, force_feature="shape", start_index=1)
    rows.extend(r)

    agent_c = Agent()  # fresh salience: 001-style mixed control
    r, c_learn = run_block("C-mixed-learn", agent_c, rng, n, force_feature=None, start_index=1)
    rows.extend(r)
    r, c_xfer = run_block("C-mixed-xfer", agent_c, rng, n, force_feature=None, start_index=1)
    rows.extend(r)

    verdict, why, stats = interpret(a_learn, a_xfer, b_shape, c_learn + c_xfer)
    return {
        "rows": rows,
        "a_learn": a_learn,
        "a_xfer": a_xfer,
        "b_shape": b_shape,
        "c_learn": c_learn,
        "c_xfer": c_xfer,
        "verdict": verdict,
        "why": why,
        "stats": stats,
    }


def print_report(result):
    print("  # block            hidden rule           tests held-out acc  competence curve                 n_exp  salience")
    print("-" * 120)
    for row in result["rows"]:
        print(
            "  %-16s %-21s %5d %10.1f%%  %-32s %5d  %s"
            % (
                "%s %d" % (row["block"], row["index"]),
                fmt_rule(row["hidden"]),
                row["steps"],
                row["held_out"],
                fmt_curve(row["curve"]),
                row["steps"],
                fmt_salience(row["salience"]),
            )
        )
    print("-" * 120)
    print("  A learning (color) n_exp = %s  mean=%.2f" % (result["a_learn"], mean(result["a_learn"])))
    print("  A transfer (color) n_exp = %s  mean=%.2f" % (result["a_xfer"], mean(result["a_xfer"])))
    print("  B switch   (shape) n_exp = %s  mean=%.2f" % (result["b_shape"], mean(result["b_shape"])))
    print("  C mixed learn      n_exp = %s  mean=%.2f" % (result["c_learn"], mean(result["c_learn"])))
    print("  C mixed xfer       n_exp = %s  mean=%.2f" % (result["c_xfer"], mean(result["c_xfer"])))
    perfect = sum(1 for row in result["rows"] if row["held_out"] >= 100.0 - 1e-9)
    print("  held-out answers reach 100%% in %d/%d worlds" % (perfect, len(result["rows"])))
    print("  verdict: %s" % result["verdict"])
    print("  %s" % result["why"])


# ---------------------------------------------------------------------------
# Experiment 003: paired seeded rototest (measurement hardening)
# ---------------------------------------------------------------------------


def run_prior_sequence(agent, worlds):
    """Run `agent` through a sequence of worlds, salience carrying over."""
    return [agent.run_world(w) for w in worlds]


def run_null_sequence(agent, worlds):
    """Run the same machinery with the prior neutralized before every world."""
    outcomes = []
    for w in worlds:
        agent.reset_salience()
        outcomes.append(agent.run_world(w))
    return outcomes


def _resample_seed(base_seed, salt):
    return base_seed * 7919 + salt


def bootstrap_ci(diffs, seed, n_boot=1999, level=0.95):
    """Percentile bootstrap CI for the mean of `diffs`. Returns (lo, hi, mean)."""
    n = len(diffs)
    if n == 0:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    means = [0.0] * n_boot
    for i in range(n_boot):
        total = 0.0
        for _ in range(n):
            total += diffs[rng.randrange(n)]
        means[i] = total / n
    means.sort()
    k = int(n_boot * (1.0 - level) / 2.0)
    return means[k], means[n_boot - 1 - k], sum(diffs) / n


def permutation_pvalue(diffs, seed, n_perm=3999, alternative="less"):
    """Sign-flip permutation test on paired differences (distribution-free).

    alternative='less' tests that the pairwise mean is below zero, i.e. that
    the prior reduces the number of experiments to convergence.
    """
    n = len(diffs)
    if n == 0:
        return float("nan")
    obs = sum(diffs) / n
    rng = random.Random(seed)
    count = 0
    for _ in range(n_perm):
        total = 0.0
        for d in diffs:
            total += d if rng.random() < 0.5 else -d
        m = total / n
        if alternative == "less":
            count += m <= obs
        elif alternative == "greater":
            count += m >= obs
        else:
            count += abs(m) >= abs(obs)
    return (count + 1) / float(n_perm + 1)


def cohens_dz(diffs):
    """Standardized paired effect size; None when degenerate (all diffs equal)."""
    n = len(diffs)
    if n < 2:
        return None
    sd = statistics.stdev(diffs)
    if sd == 0.0:
        return None
    return statistics.mean(diffs) / sd


def _block_stats(name, index, diffs, per_seed_means, seed):
    lo, hi, m = bootstrap_ci(diffs, _resample_seed(seed, 3 + index))
    return {
        "name": name,
        "n": len(diffs),
        "mean": m,
        "ci95": (lo, hi),
        "p_less": permutation_pvalue(diffs, _resample_seed(seed, 5 + index)),
        "dz": cohens_dz(diffs),
        "seeds_help": sum(1 for x in per_seed_means if x <= 0.0),
        "per_seed": list(per_seed_means),
    }


def _judge(a_learn, a_xfer, b_curve, c, first_ds, last_ds, seed, c_margin=0.25):
    a_learn = a_learn["ci95"]
    a_xfer = a_xfer["ci95"]
    c_ci = c["ci95"]
    b_first_ci = bootstrap_ci(first_ds, _resample_seed(seed, 11))[:2]
    _, _, last_m = bootstrap_ci(last_ds, _resample_seed(seed, 12))
    _, _, first_m = bootstrap_ci(first_ds, _resample_seed(seed, 13))

    a_improve = a_xfer[1] < 0.0  # upper CI below control: reliably fewer experiments
    cost = b_first_ci[0] > 0.0  # lower CI above control: switch is reliably costly
    recovered = last_m + 0.05 < first_m  # later-switch cost below first-switch cost
    # Non-inferiority, not equality: the prior must not cost more than a margin
    # of experiments on mixed worlds, even if it reliably helps by a little.
    no_harm = c_ci[1] < c_margin

    gates = {
        "A_transfer_improve": a_improve,
        "B_switch_cost": cost,
        "B_recovered": recovered,
        "C_mixed_no_harm": no_harm,
    }

    if all(gates.values()):
        verdict = "kept"
        why = (
            "Paired CI shows the prior reliably reduces n_exp on A transfer, "
            "reliably pays a cost on the first B switch, recovers, and does not "
            "cost more than a margin on mixed worlds. Keep the primitive."
        )
    elif not a_improve:
        verdict = "rejected"
        why = (
            "Paired CI on the A-transfer effect covers zero: no measurable "
            "learning-to-learn benefit over the identical-machinery control. "
            "Do not keep the primitive."
        )
    else:
        verdict = "inconclusive"
        why = (
            "Signals mixed across gates (A_improve=%s B_cost=%s B_recovered=%s "
            "C_no_harm=%s). Do not keep the primitive."
            % (
                gates["A_transfer_improve"],
                gates["B_switch_cost"],
                gates["B_recovered"],
                gates["C_mixed_no_harm"],
            )
        )
    return verdict, why, gates


def paired_rototest(seed=7, n_worlds=5, n_seeds=16):
    """Rototest v2: prior vs null on identical, per-seed generated worlds.

    Every world drives both agents, so within-seed world identity is identical;
    the paired difference d = n_exp(prior) - n_exp(null) is the signal.
    """
    per_world = {block: [] for block in BLOCKS}
    seeds_order = {block: [] for block in BLOCKS}
    b_curve = {}

    for s in range(n_seeds):
        srng = random.Random(seed * 10000 + s)
        a_learn = [UnknownWorld.generate(srng, force_feature="color") for _ in range(n_worlds)]
        a_transfer = [UnknownWorld.generate(srng, force_feature="color") for _ in range(n_worlds)]
        b_switch = [UnknownWorld.generate(srng, force_feature="shape") for _ in range(n_worlds)]
        c_worlds = [UnknownWorld.generate(srng) for _ in range(2 * n_worlds)]

        pri, nul = Agent(), Agent()
        la_p = run_prior_sequence(pri, a_learn)
        la_n = run_null_sequence(nul, a_learn)
        ta_p = run_prior_sequence(pri, a_transfer)
        ta_n = run_null_sequence(nul, a_transfer)
        sw_p = run_prior_sequence(pri, b_switch)
        sw_n = run_null_sequence(nul, b_switch)
        prc, nuc = Agent(), Agent()
        cm_p = run_prior_sequence(prc, c_worlds)
        cm_n = run_null_sequence(nuc, c_worlds)

        def feed(name, ps, ns):
            ds = [p["steps"] - n["steps"] for p, n in zip(ps, ns)]
            per_world[name].append(ds)
            seeds_order[name].append(sum(ds) / len(ds))

        feed("A-learn", la_p, la_n)
        feed("A-transfer", ta_p, ta_n)
        feed("B-switch", sw_p, sw_n)
        feed("C-mixed", cm_p, cm_n)

        for t, (p, n) in enumerate(zip(sw_p, sw_n), start=1):
            b_curve.setdefault(t, []).append(p["steps"] - n["steps"])

    blocks = {}
    for i, block in enumerate(BLOCKS):
        diffs = []
        for ds in per_world[block]:
            diffs.extend(ds)
        blocks[block] = _block_stats(block, i, diffs, seeds_order[block], seed)

    curve = {}
    for t in sorted(b_curve):
        ds = b_curve[t]
        lo, hi, m = bootstrap_ci(ds, _resample_seed(seed, 100 + t))
        curve[t] = {"mean": m, "ci95": (lo, hi), "n": len(ds)}

    first_ds = b_curve[1]
    last_ds = []
    for t in range(max(1, n_worlds - 1), n_worlds + 1):
        last_ds.extend(b_curve[t])

    verdict, why, gates = _judge(
        blocks["A-learn"], blocks["A-transfer"], curve, blocks["C-mixed"],
        first_ds, last_ds, seed,
    )
    return {
        "blocks": blocks,
        "curve": curve,
        "first": first_ds,
        "last": last_ds,
        "n_seeds": n_seeds,
        "n_worlds": n_worlds,
        "seed_base": seed,
        "verdict": verdict,
        "why": why,
        "gates": gates,
    }


def print_paired_report(result):
    b = result["blocks"]
    print(
        "  block        n    mean d  95% CI            p(prior<null)  dz     seeds helping"
    )
    print("-" * 82)
    for block in BLOCKS:
        s = b[block]
        dz = "  n/a" if s["dz"] is None else "%5.2f" % s["dz"]
        print(
            "  %-12s %4d  %6.2f  [%6.2f, %6.2f]   %s     %s   %d/%d"
            % (
                block,
                s["n"],
                s["mean"],
                s["ci95"][0],
                s["ci95"][1],
                ("%.4f" % s["p_less"]),
                dz,
                s["seeds_help"],
                result["n_seeds"],
            )
        )
    print("-" * 82)
    print("  B-switch paired effect by world position (d = prior - null)")
    for t in sorted(result["curve"]):
        c = result["curve"][t]
        print(
            "    world %d  mean d %6.2f  95%% CI [%6.2f, %6.2f]"
            % (t, c["mean"], c["ci95"][0], c["ci95"][1])
        )
    print("-" * 82)
    for gate, ok in result["gates"].items():
        print("  %-22s : %s" % (gate.upper(), ok))
    print("  verdict: %s" % result["verdict"])
    print("  %s" % result["why"])


# ---------------------------------------------------------------------------
# Experiment 004: change-aware salience (four arms on identical worlds)
# ---------------------------------------------------------------------------

ARMS = ("null", "bare", "forget", "change")
C_MARGIN = 0.25  # non-inferiority margin for the mixed control


def _arm_factory(arm):
    if arm == "forget":
        return lambda: ForgetAgent(forget=0.7)
    if arm == "change":
        return lambda: ChangePointAgent(hazard=1.0 / 5.0)
    return Agent


def _rototest_004_arms(seed, n_worlds, n_seeds):
    """Return {arm: {block: [steps per world]}} over identical per-seed worlds."""
    per_arm = {a: {b: [] for b in BLOCKS} for a in ARMS}
    b_pos = {a: {t: [] for t in range(1, n_worlds + 1)} for a in ARMS}

    for s in range(n_seeds):
        srng = random.Random(seed * 10000 + s)
        segs = [
            [UnknownWorld.generate(srng, force_feature="color") for _ in range(n_worlds)],
            [UnknownWorld.generate(srng, force_feature="color") for _ in range(n_worlds)],
            [UnknownWorld.generate(srng, force_feature="shape") for _ in range(n_worlds)],
            [UnknownWorld.generate(srng) for _ in range(2 * n_worlds)],
        ]
        for arm in ARMS:
            factory = _arm_factory(arm)
            if arm == "null":
                agent = factory()
                steps_by_seg = [run_null_sequence(agent, seg) for seg in segs]
            else:
                agent = factory()
                steps_by_seg = [run_prior_sequence(agent, seg) for seg in segs[:3]]
                fresh = factory()
                steps_by_seg.append(run_prior_sequence(fresh, segs[3]))
            for blk, steps in zip(BLOCKS, steps_by_seg):
                per_arm[arm][blk].extend(s["steps"] for s in steps)
            for t, res in enumerate(steps_by_seg[2], start=1):
                b_pos[arm][t].append(res["steps"])
    return per_arm, b_pos


def _pair(a, b):
    return [x - y for x, y in zip(a, b)]


def rototest_004(seed=7, n_worlds=5, n_seeds=16):
    """Change-aware salience vs bare prior vs fixed-forgetting vs null.

    Arms run the exact same world streams. The kept/reject question is
    whether evidence-gated forgetting (C) reduces the measured B-switch cost
    relative to the bare prior (B worlds after the first) without giving up
    the transfer benefit or harming the mixed control.
    """
    per_arm, b_pos = _rototest_004_arms(seed, n_worlds, n_seeds)

    def blk(arm, b):
        return per_arm[arm][b]

    def pos_diffs(arm_a, arm_b, t):
        return _pair(b_pos[arm_a][t], b_pos[arm_b][t])

    cn = {b: _pair(blk("change", b), blk("null", b)) for b in BLOCKS}
    ctb = {b: _pair(blk("change", b), blk("bare", b)) for b in BLOCKS}
    ftb = {b: _pair(blk("forget", b), blk("bare", b)) for b in BLOCKS}

    stats_cn = {}
    for i, b in enumerate(BLOCKS):
        per_seed = []
        for s in range(n_seeds):
            chunk = cn[b][s * n_worlds : (s + 1) * n_worlds]
            per_seed.append(sum(chunk) / len(chunk))
        stats_cn[b] = _block_stats(b, i, cn[b], per_seed, seed)

    b_reduced_ds = []
    for t in range(2, n_worlds + 1):
        b_reduced_ds.extend(pos_diffs("change", "bare", t))
    b_reduced_ci = bootstrap_ci(b_reduced_ds, _resample_seed(seed, 31))

    a_ci = stats_cn["A-transfer"]["ci95"]
    c_ci = stats_cn["C-mixed"]["ci95"]
    first_ds = pos_diffs("change", "bare", 1)
    first_ci = bootstrap_ci(first_ds, _resample_seed(seed, 32))

    a_improve = a_ci[1] < 0.0
    b_reduced = b_reduced_ci[1] < 0.0
    no_harm = c_ci[1] < C_MARGIN
    gates = {"A_transfer_improve": a_improve, "B_switch_reduced": b_reduced, "C_mixed_no_harm": no_harm}

    if all(gates.values()):
        verdict = "kept"
        why = (
            "Change-aware salience preserves the transfer benefit, reliably "
            "cuts the post-first switch cost below the bare prior, and stays "
            "harmless on mixed worlds. The prior can change its mind. Keep it."
        )
    elif not a_improve:
        verdict = "rejected"
        why = (
            "Change-aware salience lost the transfer benefit over the "
            "identical-machinery control. The added machinery broke the prior."
        )
    else:
        verdict = "inconclusive"
        why = (
            "Signals mixed (A_improve=%s B_reduced=%s C_no_harm=%s). "
            "Do not keep the primitive."
            % (a_improve, b_reduced, no_harm)
        )

    curve = {}
    for t in range(1, n_worlds + 1):
        cb = pos_diffs("change", "bare", t)
        fb = pos_diffs("forget", "bare", t)
        cb_ci = bootstrap_ci(cb, _resample_seed(seed, 100 + t))
        fb_ci = bootstrap_ci(fb, _resample_seed(seed, 200 + t))
        curve[t] = {
            "change_minus_bare": (cb_ci[0], cb_ci[1], cb_ci[2]),
            "forget_minus_bare": (fb_ci[0], fb_ci[1], fb_ci[2]),
        }

    return {
        "stats_cn": stats_cn,
        "curve": curve,
        "b_reduced": b_reduced_ds,
        "b_reduced_ci": b_reduced_ci,
        "first": first_ds,
        "first_ci": first_ci,
        "gates": gates,
        "verdict": verdict,
        "why": why,
        "n_seeds": n_seeds,
        "n_worlds": n_worlds,
        "seed_base": seed,
    }


def print_004_report(result):
    stats_cn = result["stats_cn"]
    print(
        "  change-aware vs null (identical worlds; d = n_exp(change) - n_exp(null))"
    )
    print(
        "  block        n    mean d  95% CI            p(prior<null)  dz     seeds helping"
    )
    print("-" * 82)
    for block in BLOCKS:
        s = stats_cn[block]
        dz = "  n/a" if s["dz"] is None else "%5.2f" % s["dz"]
        print(
            "  %-12s %4d  %6.2f  [%6.2f, %6.2f]   %s     %s   %d/%d"
            % (
                block,
                s["n"],
                s["mean"],
                s["ci95"][0],
                s["ci95"][1],
                "%.4f" % s["p_less"],
                dz,
                s["seeds_help"],
                result["n_seeds"],
            )
        )
    print("-" * 82)
    print("  B-switch paired effect by world position (d = mechanism - bare prior)")
    print("    world   change-bare CI             forget-bare CI")
    for t in sorted(result["curve"]):
        c, f = result["curve"][t]["change_minus_bare"], result["curve"][t]["forget_minus_bare"]
        print(
            "    %d      %6.2f [%6.2f, %6.2f]     %6.2f [%6.2f, %6.2f]"
            % (t, c[2], c[0], c[1], f[2], f[0], f[1])
        )
    pooled = result["b_reduced_ci"]
    print(
        "  change vs bare, B worlds 2..%d pooled: mean d %6.2f  95%% CI [%6.2f, %6.2f]"
        % (result["n_worlds"], sum(result["b_reduced"]) / len(result["b_reduced"]), pooled[0], pooled[1])
    )
    print("-" * 82)
    for gate, ok in result["gates"].items():
        print("  %-22s : %s" % (gate.upper(), ok))
    print("  verdict: %s" % result["verdict"])
    print("  %s" % result["why"])


# ---------------------------------------------------------------------------
# Experiment 005: learning the hazard rate (regime streams on identical worlds)
# ---------------------------------------------------------------------------

ARMS_005 = ("null", "fixed", "forget", "learned")
PACES = ("home", "slow", "fast")
REGIME_WIDTHS = {"home": (5, 5, 5, 5), "slow": (16, 16), "fast": (2, 2, 2, 2, 2, 2, 2, 2)}
PACES_START = {"home": "color", "slow": "shape", "fast": "color"}
N_MIXED = 10


def _regime_worlds(srng, widths, start_feature):
    """A block of worlds whose hidden feature alternates every `width` worlds.

    Forces are feature-only, so value and object population stay randomized
    per world. The caller chooses the starting feature so a block can
    *continue* the previous block's regime (calm entry: no forced switch at
    the block boundary).
    """
    worlds = []
    feature = start_feature
    for width in widths:
        for _ in range(width):
            worlds.append(UnknownWorld.generate(srng, force_feature=feature))
        feature = "shape" if feature == "color" else "color"
    return worlds


BLOCKS_005 = ("A-learn", "A-transfer", "B-home", "B-slow", "B-fast", "C-mixed")


def _arm_factory_005(arm):
    if arm == "fixed":
        return lambda: ChangePointAgent(hazard=1.0 / 5.0)
    if arm == "forget":
        return lambda: ForgetAgent(forget=0.7)
    if arm == "learned":
        return lambda: HierarchicalChangePointAgent()
    return Agent


def _pace_length(pace):
    return sum(REGIME_WIDTHS[pace])


def _rototest_005_arms(seed, n_worlds, n_seeds):
    """Return per-arm step lists, per-position B steps, and hazard means."""
    block_len = {"A-learn": n_worlds, "A-transfer": n_worlds, "C-mixed": N_MIXED}
    for p in PACES:
        block_len["B-" + p] = _pace_length(p)

    per_arm = {a: {b: [] for b in BLOCKS_005} for a in ARMS_005}
    b_pos = {a: {p: {t: [] for t in range(1, block_len["B-" + p] + 1)} for p in PACES} for a in ARMS_005}
    hazard_eff = {"learned": {p: [] for p in ("post-A", "post-home", "post-slow", "post-fast")}}
    slow_tail = {a: [] for a in ARMS_005}

    order = ("A-learn", "A-transfer", "B-home", "B-slow", "B-fast", "C-mixed")

    for s in range(n_seeds):
        srng = random.Random(seed * 10000 + s)
        segs = {
            "A-learn": [UnknownWorld.generate(srng, force_feature="color") for _ in range(n_worlds)],
            "A-transfer": [UnknownWorld.generate(srng, force_feature="color") for _ in range(n_worlds)],
            "B-home": _regime_worlds(srng, REGIME_WIDTHS["home"], PACES_START["home"]),
            "B-slow": _regime_worlds(srng, REGIME_WIDTHS["slow"], PACES_START["slow"]),
            "B-fast": _regime_worlds(srng, REGIME_WIDTHS["fast"], PACES_START["fast"]),
            "C-mixed": [UnknownWorld.generate(srng) for _ in range(N_MIXED)],
        }

        for arm in ARMS_005:
            factory = _arm_factory_005(arm)
            if arm == "null":
                agent = factory()
                runs = {b: run_null_sequence(agent, segs[b]) for b in order}
            else:
                agent = factory()
                prior_blocks = ("A-learn", "A-transfer", "B-home", "B-slow", "B-fast")
                runs = {}
                for b in prior_blocks:
                    runs[b] = run_prior_sequence(agent, segs[b])
                    if arm == "learned" and b in ("A-transfer", "B-home", "B-slow", "B-fast"):
                        key = "post-A" if b == "A-transfer" else "post-" + b[2:]
                        hps = agent.hazard_posterior()
                        e_h = sum(w * h for w, h in zip(hps, agent.hazards))
                        hazard_eff["learned"][key].append(e_h)
                fresh = factory()
                runs["C-mixed"] = run_prior_sequence(fresh, segs["C-mixed"])

            for b in order:
                res_list = runs[b]
                per_arm[arm][b].extend(r["steps"] for r in res_list)
                if b.startswith("B-"):
                    pace = b[2:]
                    for t, r in enumerate(res_list, start=1):
                        b_pos[arm][pace][t].append(r["steps"])
                if b == "B-slow":
                    for t, r in enumerate(res_list, start=1):
                        if (t - 1) % REGIME_WIDTHS["slow"][0] + 1 >= 9:
                            slow_tail[arm].append(r["steps"])

    return per_arm, b_pos, hazard_eff, slow_tail


def _pair_blocks(a_steps, b_steps):
    return [x - y for x, y in zip(a_steps, b_steps)]


def rototest_005(seed=7, n_worlds=5, n_seeds=16):
    """Can the prior *earn* its hazard rate instead of being handed one?

    Arms: `null` (identical machinery, prior neutralized each world), `fixed`
    (Experiment 004's change-aware monitor, hand-set h=1/5), `forget`
    (Itti-Baldi f=0.7), `learned` (hierarchical change-point: model-averages
    a hazard grid, no hand-set hazard). All arms run identical regime streams
    after an identical stationary colour history:

    - Block A: 5 learn + 5 transfer colour worlds (the 004 stationary strand).
    - B-home:  4 regimes of width 5  (the rate h=1/5 was tuned to this).
    - B-slow:  2 regimes of width 16 (long calm tails; the fixed cap is the
               *wrong* prior here).
    - B-fast:  8 regimes of width 2  (switches every other world; h=1/5
               under-anticipates change).
    - C-mixed: fresh agent on unconstrained worlds (non-inferiority control).

    Gates (pre-committed, computed by the harness):
    - A_transfer_no_regression: learned < null on A-transfer (CI upper < 0).
    - B_home_no_regression:     learned < fixed on B-home worlds 2..20.
    - B_slow_tail_improved:     learned < fixed on the calm tails
                                (regime positions 9..16).
    - B_fast_adapted:           learned < fixed on B-fast worlds 2..16.
    - C_mixed_no_harm:          learned < null + margin on mixed worlds.
    """
    per_arm, b_pos, hazard_eff, slow_tail = _rototest_005_arms(seed, n_worlds, n_seeds)

    saw = {b: len(per_arm["learned"][b]) // n_seeds for b in BLOCKS_005}

    ln = {b: _pair_blocks(per_arm["learned"][b], per_arm["null"][b]) for b in BLOCKS_005}
    lf = {b: _pair_blocks(per_arm["learned"][b], per_arm["fixed"][b]) for b in BLOCKS_005}

    stats_ln = {}
    for i, b in enumerate(BLOCKS_005):
        per_seed = []
        wpt = saw[b]
        for s in range(n_seeds):
            chunk = ln[b][s * wpt : (s + 1) * wpt]
            per_seed.append(sum(chunk) / len(chunk))
        stats_ln[b] = _block_stats(b, i, ln[b], per_seed, seed)

    def pool_lf(pace, positions):
        diffs = []
        for t in positions:
            for j in range(n_seeds):
                diffs.append(b_pos["learned"][pace][t][j] - b_pos["fixed"][pace][t][j])
        return diffs

    pos_lf = {
        p: {t: [x - y for x, y in zip(b_pos["learned"][p][t], b_pos["fixed"][p][t])]
            for t in b_pos["learned"][p]}
        for p in PACES
    }

    home_ds = pool_lf("home", range(2, saw["B-home"] + 1))
    slow_tail_ds = []
    for t in range(1, saw["B-slow"] + 1):
        if (t - 1) % 16 + 1 >= 9:
            slow_tail_ds.extend(pos_lf["slow"][t])
    fast_ds = pool_lf("fast", range(2, saw["B-fast"] + 1))

    a_ci = bootstrap_ci(ln["A-transfer"], _resample_seed(seed, 201))
    home_ci = bootstrap_ci(home_ds, _resample_seed(seed, 202))
    slow_ci = bootstrap_ci(slow_tail_ds, _resample_seed(seed, 203))
    fast_ci = bootstrap_ci(fast_ds, _resample_seed(seed, 204))
    c_ci = bootstrap_ci(ln["C-mixed"], _resample_seed(seed, 205))

    a_improve = a_ci[1] < 0.0
    home_ok = home_ci[1] < 0.0
    slow_ok = slow_ci[1] < 0.0
    fast_ok = fast_ci[1] < 0.0
    no_harm = c_ci[1] < C_MARGIN

    gates = {
        "A_transfer_no_regression": a_improve,
        "B_home_no_regression": home_ok,
        "B_slow_tail_improved": slow_ok,
        "B_fast_adapted": fast_ok,
        "C_mixed_no_harm": no_harm,
    }

    if all(gates.values()):
        verdict = "kept"
        why = (
            "A hazard learned from the stream preserves the stationary "
            "transfer benefit, is not worse than the hand-set h=1/5 exactly "
            "where 1/5 was tuned, beats it on the slow calm tails (the fixed "
            "cap) and on the fast regime (switches every other world), and "
            "stays harmless on mixed worlds. The experimenter no longer "
            "chooses the rate of forgetting. Keep it."
        )
    elif not a_improve:
        verdict = "rejected"
        why = (
            "Learned-hazard salience lost the transfer benefit over the "
            "identical-machinery control: the hierarchy broke the prior."
        )
    else:
        verdict = "inconclusive"
        why = (
            "Signals mixed (A_improve=%s home_ok=%s slow_ok=%s fast_ok=%s "
            "no_harm=%s). Do not keep the primitive."
            % (a_improve, home_ok, slow_ok, fast_ok, no_harm)
        )

    slow_lf_ci = bootstrap_ci(lf["B-slow"], _resample_seed(seed, 206))

    return {
        "stats_ln": stats_ln,
        "lf": lf,
        "pos_lf": pos_lf,
        "home_ds": home_ds,
        "home_ci": home_ci,
        "slow_tail_ds": slow_tail_ds,
        "slow_ci": slow_ci,
        "fast_ds": fast_ds,
        "fast_ci": fast_ci,
        "slow_tail_steps": slow_tail,
        "saw": saw,
        "a_ci": a_ci,
        "c_ci": c_ci,
        "slow_lf_ci": slow_lf_ci,
        "hazard_eff": hazard_eff,
        "gates": gates,
        "verdict": verdict,
        "why": why,
        "n_seeds": n_seeds,
        "n_worlds": n_worlds,
        "seed_base": seed,
    }


def print_005_report(result):
    rho = REGIME_WIDTHS
    saw = result["saw"]
    stats = result["stats_ln"]
    print("  learned vs null (identical worlds; d = n_exp(learned) - n_exp(null))")
    print(
        "  block        n    mean d  95% CI            p(prior<null)  dz     seeds helping"
    )
    print("-" * 82)
    for b in BLOCKS_005:
        s = stats[b]
        dz = "  n/a" if s["dz"] is None else "%5.2f" % s["dz"]
        print(
            "  %-12s %4d  %6.2f  [%6.2f, %6.2f]   %s     %s   %d/%d"
            % (
                b,
                s["n"],
                s["mean"],
                s["ci95"][0],
                s["ci95"][1],
                "%.4f" % s["p_less"],
                dz,
                s["seeds_help"],
                result["n_seeds"],
            )
        )
    print("-" * 82)
    print("  learned vs fixed h=1/5  (d = n_exp(learned) - n_exp(fixed))")
    print("  block        n    mean d  95% CI")
    print("  " + "-" * 46)
    for b in BLOCKS_005:
        ds = result["lf"][b]
        lo, hi, m = bootstrap_ci(ds, _resample_seed(result["seed_base"], 300 + BLOCKS_005.index(b)))
        print("  %-12s %4d  %6.2f  [%6.2f, %6.2f]" % (b, len(ds), m, lo, hi))
    print("  " + "-" * 46)

    def ef_ci(ds, salt):
        lo, hi, m = bootstrap_ci(ds, _resample_seed(result["seed_base"], salt))
        return m, lo, hi

    print("  regime-relative position, learned - fixed (mean d, 95% CI)")
    for pace in PACES:
        width = rho[pace][0]
        print("  -- %-5s (regime width %d) --" % (pace, width))
        for pos in range(1, width + 1):
            ds = []
            for t in range(1, saw["B-" + pace] + 1):
                if (t - 1) % width + 1 == pos:
                    ds.extend(result["pos_lf"][pace][t])
            m, lo, hi = ef_ci(ds, 400 + 3 * {"home": 0, "slow": 1, "fast": 2}[pace] + pos)
            print(
                "    pos %2d  mean d %6.2f  95%% CI [%6.2f, %6.2f]"
                % (pos, m, lo, hi)
            )
    print("-" * 82)

    m, lo, hi = ef_ci(result["slow_tail_ds"], 500)
    print(
        "  learned vs fixed, B-slow calm tails (positions 9..16): mean d %6.2f  95%% CI [%6.2f, %6.2f]"
        % (m, lo, hi)
    )
    st = result["slow_tail_steps"]
    print(
        "  mean n_exp per arm on the B-slow calm tails: learned %.2f  fixed %.2f  forget %.2f  null %.2f"
        % (
            sum(st["learned"]) / len(st["learned"]),
            sum(st["fixed"]) / len(st["fixed"]),
            sum(st["forget"]) / len(st["forget"]),
            sum(st["null"]) / len(st["null"]),
        )
    )
    print("-" * 82)

    print("  gate CIs (the verdict is drawn on these boundaries)")
    rows = [
        ("A_transfer (learned-null, A-transfer)", "a_ci", "upper<0"),
        ("B_home     (learned-fixed, worlds 2..end)", "home_ci", "upper<0"),
        ("B_slow_tail(learned-fixed, calm tails)", "slow_ci", "upper<0"),
        ("B_fast     (learned-fixed, worlds 2..end)", "fast_ci", "upper<0"),
        ("C_mixed    (learned-null + margin)", "c_ci", "upper<0.25"),
    ]
    for label, key, rule in rows:
        lo, hi, m = result[key]
        print(
            "    %-38s mean d %6.2f  95%% CI [%6.2f, %6.2f]   (%s)"
            % (label, m, lo, hi, rule)
        )
    print("-" * 82)

    print("  effective hazard of the learned prior (posterior mean E[h], averaged over seeds)")
    for name in ("post-A", "post-home", "post-slow", "post-fast"):
        vals = result["hazard_eff"]["learned"][name]
        mean_h = sum(vals) / len(vals) if vals else 0.0
        print("    after %-8s   E[h] = %.4f" % (name.replace("post-", ""), mean_h))
    print("-" * 82)
    for gate, ok in result["gates"].items():
        print("  %-26s : %s" % (gate.upper(), ok))
    print("  verdict: %s" % result["verdict"])
    print("  %s" % result["why"])


# ---------------------------------------------------------------------------
# Experiment 006: the hazard learned *per observation* (adaptive hazard)
# ---------------------------------------------------------------------------

H_MARGIN = 0.04  # pre-committed ordering margin for the fast>slow hazard gate

ARMS_006 = ("null", "fixed", "forget", "learned")


def _arm_factory_006(arm):
    if arm == "fixed":
        return lambda: ChangePointAgent(hazard=1.0 / 5.0)
    if arm == "forget":
        return lambda: ForgetAgent(forget=0.7)
    if arm == "learned":
        return lambda: ObservationHazardAgent()
    return Agent


def _rototest_006_arms(seed, n_worlds, n_seeds):
    """Return per-arm step lists, B-position lists, and per-seed hazard means."""
    block_len = {"A-learn": n_worlds, "A-transfer": n_worlds, "C-mixed": N_MIXED}
    for p in PACES:
        block_len["B-" + p] = _pace_length(p)

    per_arm = {a: {b: [] for b in BLOCKS_005} for a in ARMS_006}
    b_pos = {
        a: {p: {t: [] for t in range(1, block_len["B-" + p] + 1)} for p in PACES}
        for a in ARMS_006
    }
    hazard_seeds = []
    slow_tail = {a: [] for a in ARMS_006}

    order = ("A-learn", "A-transfer", "B-home", "B-slow", "B-fast", "C-mixed")

    for s in range(n_seeds):
        srng = random.Random(seed * 10000 + s)
        segs = {
            "A-learn": [UnknownWorld.generate(srng, force_feature="color") for _ in range(n_worlds)],
            "A-transfer": [UnknownWorld.generate(srng, force_feature="color") for _ in range(n_worlds)],
            "B-home": _regime_worlds(srng, REGIME_WIDTHS["home"], PACES_START["home"]),
            "B-slow": _regime_worlds(srng, REGIME_WIDTHS["slow"], PACES_START["slow"]),
            "B-fast": _regime_worlds(srng, REGIME_WIDTHS["fast"], PACES_START["fast"]),
            "C-mixed": [UnknownWorld.generate(srng) for _ in range(N_MIXED)],
        }

        for arm in ARMS_006:
            factory = _arm_factory_006(arm)
            if arm == "null":
                agent = factory()
                runs = {b: run_null_sequence(agent, segs[b]) for b in order}
            else:
                agent = factory()
                prior_blocks = ("A-learn", "A-transfer", "B-home", "B-slow", "B-fast")
                runs = {}
                for b in prior_blocks:
                    runs[b] = run_prior_sequence(agent, segs[b])
                    if arm == "learned":
                        if b == "A-learn":
                            hazard_seeds.append(dict.fromkeys(
                                ("post-A", "post-home", "post-slow", "post-fast"), 0.0
                            ))
                        if b == "A-transfer":
                            hazard_seeds[-1]["post-A"] = _effective_hazard(agent)
                        if b == "B-home":
                            hazard_seeds[-1]["post-home"] = _effective_hazard(agent)
                        if b == "B-slow":
                            hazard_seeds[-1]["post-slow"] = _effective_hazard(agent)
                        if b == "B-fast":
                            hazard_seeds[-1]["post-fast"] = _effective_hazard(agent)
                fresh = factory()
                runs["C-mixed"] = run_prior_sequence(fresh, segs["C-mixed"])

            for b in order:
                res_list = runs[b]
                per_arm[arm][b].extend(r["steps"] for r in res_list)
                if b.startswith("B-"):
                    pace = b[2:]
                    for t, r in enumerate(res_list, start=1):
                        b_pos[arm][pace][t].append(r["steps"])
                if b == "B-slow":
                    for t, r in enumerate(res_list, start=1):
                        if (t - 1) % REGIME_WIDTHS["slow"][0] + 1 >= 9:
                            slow_tail[arm].append(r["steps"])

    return per_arm, b_pos, hazard_seeds, slow_tail


def _effective_hazard(agent):
    """Posterior-mean hazard E[h] of a hazard-learning agent."""
    hps = agent.hazard_posterior()
    return sum(w * h for w, h in zip(hps, agent.hazards))


def rototest_006(seed=7, n_worlds=5, n_seeds=16, h_margin=H_MARGIN):
    """Does a *per-observation* hazard learner retrieve the regime rate?

    Experiment 005 compressed each world into a single winner feature and
    updated the hazard posterior only at convergence; the hazard proved
    unidentifiable from that 2-feature winner sequence. Experiment 006 feeds
    the same regime streams to `ObservationHazardAgent`, which runs an exact
    run-length trellis per candidate hazard updated at every *observation*,
    with the change branch active only at world boundaries (a feature cannot
    change inside a world). The protocol is otherwise identical to 005
    (identical worlds, arms null/fixed/forget/learned).

    The deciding gate is about *identification*, not behaviour:
    - H_orders_fast_over_slow: the learned hazard's posterior mean after the
      B-fast block reliably exceeds its value after the B-slow block by more
      than `h_margin`. The true boundary-flip rates are 1/16 (slow) and 1/2
      (fast) per world; a monitor with real hazard evidence must move E[h]
      up when the fast regime arrives.
    - A_transfer_no_regression: learned is not worse than null on the
      stationary A-transfer benefit (identical-machinery discipline).
    - C_mixed_no_harm: learned is not worse than null by more than the
      non-inferiority margin on mixed worlds.

    Keep the primitive only if the per-observation monitor orders the rates
    it is asked to live with, without breaking the prior's measured benefits.
    """
    per_arm, b_pos, hazard_seeds, slow_tail = _rototest_006_arms(seed, n_worlds, n_seeds)

    saw = {b: len(per_arm["learned"][b]) // n_seeds for b in BLOCKS_005}

    ln = {b: _pair_blocks(per_arm["learned"][b], per_arm["null"][b]) for b in BLOCKS_005}
    lf = {b: _pair_blocks(per_arm["learned"][b], per_arm["fixed"][b]) for b in BLOCKS_005}

    stats_ln = {}
    for i, b in enumerate(BLOCKS_005):
        per_seed = []
        wpt = saw[b]
        for s in range(n_seeds):
            chunk = ln[b][s * wpt : (s + 1) * wpt]
            per_seed.append(sum(chunk) / len(chunk))
        stats_ln[b] = _block_stats(b, i, ln[b], per_seed, seed)

    h_eff = {k: [d[k] for d in hazard_seeds] for k in ("post-A", "post-home", "post-slow", "post-fast")}
    h_order_ds = [f - s for f, s in zip(h_eff["post-fast"], h_eff["post-slow"])]
    h_order_ci = bootstrap_ci(h_order_ds, _resample_seed(seed, 211))

    a_ci = bootstrap_ci(ln["A-transfer"], _resample_seed(seed, 212))
    c_ci = bootstrap_ci(ln["C-mixed"], _resample_seed(seed, 213))

    a_improve = a_ci[1] < 0.0
    h_ok = h_order_ci[0] > h_margin
    no_harm = c_ci[1] < C_MARGIN

    gates = {
        "A_transfer_no_regression": a_improve,
        "H_orders_fast_over_slow": h_ok,
        "C_mixed_no_harm": no_harm,
    }

    if all(gates.values()):
        verdict = "kept"
        why = (
            "A monitor that learns its hazard at observation level preserves the "
            "stationary transfer benefit, reliably orders the fast regime's rate "
            "above the slow regime's rate, and stays harmless on mixed worlds. "
            "The prior's rate of forgetting is earned from the observation "
            "stream it lives in. Keep it."
        )
    elif not a_improve:
        verdict = "rejected"
        why = (
            "Per-observation hazard learning lost the transfer benefit over the "
            "identical-machinery control: the monitor broke the prior."
        )
    else:
        verdict = "inconclusive"
        why = (
            "Signals mixed (A_improve=%s H_order=%s no_harm=%s). Do not keep "
            "the primitive." % (a_improve, h_ok, no_harm)
        )

    return {
        "stats_ln": stats_ln,
        "lf": lf,
        "hazard_eff": h_eff,
        "h_order_ds": h_order_ds,
        "h_order_ci": h_order_ci,
        "h_margin": h_margin,
        "slow_tail_steps": slow_tail,
        "saw": saw,
        "a_ci": a_ci,
        "c_ci": c_ci,
        "gates": gates,
        "verdict": verdict,
        "why": why,
        "n_seeds": n_seeds,
        "n_worlds": n_worlds,
        "seed_base": seed,
    }


def print_006_report(result):
    saw = result["saw"]
    stats = result["stats_ln"]
    print("  learned vs null (identical worlds; d = n_exp(learned) - n_exp(null))")
    print(
        "  block        n    mean d  95% CI            p(prior<null)  dz     seeds helping"
    )
    print("-" * 82)
    for b in BLOCKS_005:
        s = stats[b]
        dz = "  n/a" if s["dz"] is None else "%5.2f" % s["dz"]
        print(
            "  %-12s %4d  %6.2f  [%6.2f, %6.2f]   %s     %s   %d/%d"
            % (
                b,
                s["n"],
                s["mean"],
                s["ci95"][0],
                s["ci95"][1],
                "%.4f" % s["p_less"],
                dz,
                s["seeds_help"],
                result["n_seeds"],
            )
        )
    print("-" * 82)
    print("  learned vs fixed h=1/5  (d = n_exp(learned) - n_exp(fixed))")
    print("  block        n    mean d  95% CI")
    print("  " + "-" * 46)
    for b in BLOCKS_005:
        ds = result["lf"][b]
        lo, hi, m = bootstrap_ci(ds, _resample_seed(result["seed_base"], 300 + BLOCKS_005.index(b)))
        print("  %-12s %4d  %6.2f  [%6.2f, %6.2f]" % (b, len(ds), m, lo, hi))
    print("  " + "-" * 46)

    print("  effective hazard of the learned prior (posterior mean E[h], averaged over seeds)")
    for name in ("post-A", "post-home", "post-slow", "post-fast"):
        vals = result["hazard_eff"][name]
        mean_h = sum(vals) / len(vals) if vals else 0.0
        print("    after %-8s   E[h] = %.4f" % (name.replace("post-", ""), mean_h))
    print("-" * 82)

    lo, hi, m = result["h_order_ci"]
    print(
        "  H_order: per-seed E[h|post-fast] - E[h|post-slow]  mean %6.3f  95%% CI [%6.3f, %6.3f]  (gate: lower > %4.2f)"
        % (m, lo, hi, result["h_margin"])
    )
    print("-" * 82)
    for gate, ok in result["gates"].items():
        print("  %-26s : %s" % (gate.upper(), ok))
    print("  verdict: %s" % result["verdict"])
    print("  %s" % result["why"])