"""Recompute the leaderboard from the score files in leaderboard/runs and write
    leaderboard/board.json    every row on every split, used by the project page
    leaderboard/README.md     the full tables
and the short table between the leaderboard markers of README.md.

    python scripts/make_leaderboard.py --bench data/PROPS-Benchmark
    python scripts/make_leaderboard.py --bench data/PROPS-Benchmark --check   # only compare with board.json

Only pairs.jsonl and judge/ of the benchmark are read, so the images need not be downloaded.
"""
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from props.board import board  # noqa: E402
from props.data import SUBTYPES  # noqa: E402

LB = ROOT / "leaderboard"
SPLITS = {"full": "full", "phase_a": "phase_a"}  # split -> systems.json tag
COLS = ["Overall", "Recall", "Resist", "Reuse"] + [s for ss in SUBTYPES.values() for s in ss] + ["Follow", "Exec"]
SHORT = {s: s.split("-")[1] for ss in SUBTYPES.values() for s in ss}


def rows(bench):
    meta = json.loads((LB / "systems.json").read_text())
    out = {}
    for split, tag in SPLITS.items():
        out[split] = []
        for s in meta["systems"]:
            if tag not in s["splits"]:
                continue
            r = board(bench, LB / "runs" / s["key"], split, ci=True)
            out[split].append({"key": s["key"], "name": s["name"], "family": s["family"],
                               **{c: r[c] for c in COLS}, "ci": r["ci"]["Overall"], "unscored": r["pairs"] - r["scored"]})
    order = {f: i for i, (f, _) in enumerate(meta["families"])}
    for v in out.values():  # families in their order, systems by Overall within a family
        v.sort(key=lambda r: (order[r["family"]], -r["Overall"]))
    return {"families": meta["families"], "pairs": {k: len(v) and board(bench, LB / "runs" / v[0]["key"], k)["pairs"]
                                                    for k, v in out.items()}, "splits": out}


def table(data, split, cols, families=None):
    fam = dict(data["families"])
    head = ["System"] + [SHORT.get(c, c) for c in cols]
    L = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] + ["---:"] * len(cols)) + "|"]
    last = None
    for r in data["splits"][split]:
        if families and r["family"] not in families:
            continue
        if r["family"] != last:
            L.append(f"| *{fam[r['family']]}* |" + " |" * len(cols))
            last = r["family"]
        cells = [f"{r[c]:.1f}" if r[c] is not None else "-" for c in cols]
        if cols[0] == "Overall":
            cells[0] += f" <sub>[{r['ci'][0]:.1f}, {r['ci'][1]:.1f}]</sub>"
        L.append(f"| {r['name']} | " + " | ".join(cells) + " |")
    return "\n".join(L)


def short(data):
    """One row per system: Overall on the full benchmark and on phase A."""
    fa = {r["key"]: r for r in data["splits"]["phase_a"]}
    ff = {r["key"]: r for r in data["splits"]["full"]}
    fam = dict(data["families"])
    L = ["| System | Full | Phase A |", "|---|---:|---:|"]
    last = None
    order = {f: i for i, (f, _) in enumerate(data["families"])}
    for r in sorted(data["splits"]["phase_a"], key=lambda r: (order[r["family"]],
                                                               -(ff[r["key"]]["Overall"] if r["key"] in ff else r["Overall"]))):
        if r["family"] == "reference":
            continue
        if r["family"] != last:
            L.append(f"| *{fam[r['family']]}* | | |")
            last = r["family"]
        f = ff.get(r["key"])
        L.append(f"| {r['name']} | {f['Overall']:.1f} | {fa[r['key']]['Overall']:.1f} |" if f else
                 f"| {r['name']} | | {fa[r['key']]['Overall']:.1f} |")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default=str(ROOT / "data" / "PROPS-Benchmark"))
    ap.add_argument("--check", action="store_true", help="compare with leaderboard/board.json and write nothing")
    a = ap.parse_args()
    data = rows(a.bench)
    if a.check:
        old = json.loads((LB / "board.json").read_text())
        bad = [(s, x["key"]) for s in data["splits"] for x, y in zip(data["splits"][s], old["splits"][s]) if x != y]
        print("board.json is up to date" if not bad else f"{len(bad)} rows differ, first {bad[0]}")
        return 1 if bad else 0
    (LB / "board.json").write_text(json.dumps(data, indent=1))
    n = data["pairs"]
    doc = ["# Leaderboard", "",
           "Pass rates of pairs in %, per subtype and as the means of Recall, Resist and Reuse; Overall is the mean of",
           "the three groups, with a 95% interval from a bootstrap over source-video clusters. Follow and Exec are",
           "diagnostic rates over sides. Recompute every number with `python scripts/make_leaderboard.py`.", "",
           f"## Full benchmark ({n['full']:,} pairs, open systems)", "", table(data, "full", COLS), "",
           f"## Phase A ({n['phase_a']:,} pairs, every system)", "", table(data, "phase_a", COLS), ""]
    (LB / "README.md").write_text("\n".join(doc))
    readme = ROOT / "README.md"
    s = readme.read_text()
    a_, b_ = "<!-- leaderboard:start -->", "<!-- leaderboard:end -->"
    if a_ in s and b_ in s:
        s = s[:s.index(a_) + len(a_)] + "\n" + short(data) + "\n" + s[s.index(b_):]
        readme.write_text(s)
    print(f"wrote {LB / 'board.json'}, {LB / 'README.md'} and the table in README.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
