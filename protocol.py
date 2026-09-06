"""Rototest: within-world competence curve + across-world n_exp under controlled generators."""
from __future__ import annotations

from agent import Agent
from world import UnknownWorld


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