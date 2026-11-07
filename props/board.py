"""Leaderboard numbers from props_scores.json.

A subtype's score is the percentage of its pairs that pass, a group score is the mean of its subtypes, and Overall
is the mean of Recall, Resist and Reuse. A pair without a result counts as failed. Follow and Exec are rates over
sides (Exec over the sides that have execution questions) and are diagnostic only. With ci=True the 95% intervals
come from a bootstrap over source-video clusters (pairs.jsonl "cluster", 2,000 draws, seed 0).

    props board --bench data/PROPS-Benchmark --outputs runs/mymodel [--subset phase_a] [--ci]
"""
import json
from pathlib import Path

from .data import SUBTYPES, Bench

FAIL = {"follow": False, "other": False, "ok": False, "missing": True}


def pct(v):
    return round(100 * sum(v) / len(v), 1) if v else None


def board(bench, outputs, subset="all", ci=False, draws=2000, seed=0):
    b = Bench(bench)
    pairs = b.pairs(subset)
    got = json.loads((Path(outputs) / "props_scores.json").read_text())
    res = [got.get(p["pair"]) if got.get(p["pair"]) and len(got[p["pair"]]) == 2 else {"A": FAIL, "B": FAIL}
           for p in pairs]
    ok = [x["A"]["ok"] and x["B"]["ok"] for x in res]
    row = {"pairs": len(pairs), "scored": sum(p["pair"] in got for p in pairs)}
    for g, subs in SUBTYPES.items():
        for s in subs:
            row[s] = pct([o for p, o in zip(pairs, ok) if p["subtype"] == s])
        vals = [row[s] for s in subs if row[s] is not None]
        row[g] = round(sum(vals) / len(vals), 1) if vals else None
    gs = [row[g] for g in SUBTYPES if row[g] is not None]
    row["Overall"] = round(sum(gs) / 3, 1) if len(gs) == 3 else None
    sides = [y for x in res for y in x.values()]
    row["Follow"] = pct([y["follow"] for y in sides])
    row["Exec"] = pct([y["exec"] for y in sides if "exec" in y])
    row["missing_sides"] = sum(bool(y.get("missing")) for y in sides)
    if ci:
        row["ci"] = bootstrap(pairs, ok, draws, seed)
    return row


def bootstrap(pairs, ok, draws=2000, seed=0):
    """95% intervals of Overall and the group scores; clusters are resampled with replacement."""
    import numpy as np
    cl = [p["cluster"] for p in pairs]
    ids = {c: i for i, c in enumerate(sorted(set(cl)))}
    ci, n_c = np.array([ids[c] for c in cl]), len(ids)
    rng = np.random.default_rng(seed)
    w = np.stack([np.bincount(rng.integers(0, n_c, n_c), minlength=n_c) for _ in range(draws)])[:, ci].astype(float)
    ok = np.array(ok, float)
    out, gs = {}, {}
    for g, subs in SUBTYPES.items():
        vals = []
        for s in subs:
            m = np.array([p["subtype"] == s for p in pairs])
            if m.any():
                with np.errstate(invalid="ignore", divide="ignore"):
                    vals.append((w[:, m] * ok[m]).sum(1) / w[:, m].sum(1))
        if vals:
            gs[g] = np.nanmean(np.stack(vals), 0)
    if len(gs) == 3:
        gs["Overall"] = (gs["Recall"] + gs["Resist"] + gs["Reuse"]) / 3
    for k, v in gs.items():
        out[k] = [round(float(100 * np.nanpercentile(v, 2.5)), 1), round(float(100 * np.nanpercentile(v, 97.5)), 1)]
    return out


def show(row):
    keys = ["Overall"] + list(SUBTYPES) + [s for ss in SUBTYPES.values() for s in ss] + ["Follow", "Exec"]
    w = max(len(k) for k in keys)
    for k in keys:
        v = row.get(k)
        extra = f"  [{row['ci'][k][0]}, {row['ci'][k][1]}]" if row.get("ci", {}).get(k) else ""
        print(f"{k:<{w}}  {'-' if v is None else f'{v:5.1f}'}{extra}")
    print(f"pairs {row['pairs']}, scored {row['scored']}, missing sides {row['missing_sides']}")
