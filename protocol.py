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

from agent import Agent
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