"""Leaderboard numbers from props_scores.json.

A subtype's score is the percentage of its pairs that pass, a group score is the mean of its subtypes, and Overall
is the mean of Recall, Resist and Reuse. A pair without a result counts as failed, and the board says how many.
Follow and Exec are rates over sides (Exec over the sides that have execution questions) and are diagnostic only. With ci=True the 95% intervals
come from a bootstrap over source-video clusters (pairs.jsonl "cluster", 2,000 draws, seed 0).

    props board --outputs runs/mymodel [--split phase_a] [--ci]
"""
import hashlib, json, time
from pathlib import Path

from . import __version__
from .data import SUBTYPES, Bench, read_json, write_json

FAIL = {"follow": False, "other": False, "ok": False, "missing": True}


def pct(v):
    return round(100 * sum(v) / len(v), 1) if v else None


def board(bench, outputs, split="full", ci=False, draws=2000, seed=0):
    b = Bench(bench)
    pairs = b.pairs(split)
    f = Path(outputs) / "props_scores.json"
    got = read_json(f, {})
    res = [got.get(p["pair"]) if got.get(p["pair"]) and len(got[p["pair"]]) == 2 else {"A": FAIL, "B": FAIL}
           for p in pairs]
    ok = [x["A"]["ok"] and x["B"]["ok"] for x in res]
    row = {"split": split, "pairs": len(pairs), "scored": sum(p["pair"] in got for p in pairs)}
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
    row["unverified_sides"] = sum(bool(y.get("unverified")) for y in sides)  # scored with --allow-unverified
    row["withheld_pairs"] = sum(bool(x["A"].get("withheld")) for x in res)  # need a relit Ego4D frame, count as failed
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
    print(f"split {row.get('split')}: {row['pairs']} pairs, {row['scored']} scored, {row['missing_sides']} sides without an output")
    if row["scored"] < row["pairs"]:
        print(f"note: {row['pairs'] - row['scored']} pairs are not scored yet and count as failed; run `props score` first")
    if row.get("withheld_pairs"):
        print(f"note: {row['withheld_pairs']} pairs need a relit Ego4D frame that is not shared and count as failed")


def code_commit():
    """The git commit of this checkout, with +dirty when files differ from it, or None outside git."""
    import subprocess
    repo = Path(__file__).resolve().parents[1]
    try:
        c = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        d = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"],
                           capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return c.stdout.strip() + ("+dirty" if d.stdout.strip() else "")


def environment():
    """Python and the versions of the packages that scoring uses."""
    import platform
    from importlib.metadata import PackageNotFoundError, version
    out = {"python": platform.python_version()}
    for k in ("pillow", "numpy", "openai", "huggingface_hub"):
        try:
            out[k] = version(k)
        except PackageNotFoundError:
            out[k] = None
    return out


def record(bench, outputs, row, judge_url=None, grounder_url=None, judge_model="judge", grounder_model="grounder"):
    """Write <outputs>/props_board.json: the board row and what it was computed with. Attach this file to a
    submission. `complete` is true only when every field is known and the run can enter the leaderboard."""
    from .judge import endpoints
    from .score import RULES
    root = Path(bench)
    dl = root / "props_download.json"  # written by `props download`
    meta = {"code": __version__, "code_commit": code_commit(), "protocol": RULES, "environment": environment(),
            "benchmark": json.loads(dl.read_text()) if dl.exists() else None,
            "pairs_sha1": hashlib.sha1((root / "pairs.jsonl").read_bytes()).hexdigest(),
            "judge": endpoints(judge_url, judge_model) if judge_url else None,
            "grounder": endpoints(grounder_url, grounder_model) if grounder_url else None,
            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    unknown = [k for k, v in meta.items() if v is None]
    unknown += [f"{r} {k} of {u}" for r in ("judge", "grounder") for u, e in (meta[r] or {}).items()
                for k, v in e.items() if v is None]
    run = read_json(Path(outputs) / "props_run.json", {})  # written by `props generate`
    why = [w for w, bad in (
        ("some version information is unknown", unknown),
        ("the code differs from its git commit", str(meta["code_commit"]).endswith("+dirty")),
        ("the split is not full or phase_a", row["split"] not in ("full", "phase_a")),
        ("not every pair was scored", row["scored"] < row["pairs"]),
        (f"{row['missing_sides']} sides have no frame (scored with --allow-missing)", row["missing_sides"]),
        ("Reuse queries were scored with --allow-unverified", row.get("unverified_sides")),
        ("the frames were drawn with --ablate", run.get("ablate", "none") != "none"),
        ("the weights of a server have no revision", any(not (e["weights"] or {}).get("revision")
                                                          for r in ("judge", "grounder") for e in (meta[r] or {}).values())),
        ("the frames were drawn from another version of the data", run.get("data_commit") not in
         (None, (meta["benchmark"] or {}).get("commit"))),
    ) if bad]
    meta["complete"], meta["incomplete_because"] = not why, why
    if unknown:
        print(f"[props board] warning: unknown {', '.join(unknown)}; a leaderboard row needs them "
              "(download with `props download`, run from a git checkout, serve with vLLM)")
    if why:
        print("[props board] complete: false, so this board cannot be submitted to the leaderboard: " + "; ".join(why))
    f = Path(outputs) / "props_board.json"
    write_json(f, {"meta": meta, "board": row}, indent=1)
    return f
