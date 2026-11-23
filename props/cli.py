"""The `props` command.

    props download     --out data [--train] [--no-ego] [--revision REV]
    props rebuild-ego  --data data/PROPS-Benchmark --ego4d DIR --egoexo4d DIR
    props generate     --adapter SPEC --out runs/NAME [--split full] -- ADAPTER ARGS
    props export       --out runs/NAME [--split full]                # frames to draw with your own pipeline
    props collect      --out runs/NAME --table frames.csv            # frames saved under your own names
    props check        --outputs runs/NAME [--split full]
    props score        --outputs runs/NAME --judge URL --grounder URL [--split full]
    props board        --outputs runs/NAME [--split full] [--ci] [--json]
    props evaluate     --outputs runs/NAME --judge URL --grounder URL [--split full]   # check + score + board

Every command takes --bench (default data/PROPS-Benchmark). Exit code 0 means done, 1 means some work failed
(the message says what to run again), 2 means bad input.
"""
import argparse, json, os, sys
from pathlib import Path

SPLITS = ["full", "phase_a", "open"]  # `ready` is for smoke tests of `props generate` only


def _download(a):
    from .data import REPO_BENCH, REPO_TRAIN, download
    token = a.token or os.environ.get("HF_TOKEN")
    for repo in ([] if a.train_only else [REPO_BENCH]) + ([REPO_TRAIN] if a.train or a.train_only else []):
        dst = Path(a.out) / repo.split("/")[-1]
        print(f"{repo} -> {dst}", flush=True)
        download(dst, repo, token, ego=not a.no_ego, keep_tars=a.keep_tars, revision=a.revision)
    print("done. Ego4D / Ego-Exo4D images are not included; see `props rebuild-ego -h`.")
    return 0


def _rebuild(a):
    from .rebuild_ego import rebuild
    r = rebuild(a.data, a.ego4d, a.egoexo4d, a.workers, a.max_dist, a.force, a.ffmpeg)
    bad = [k for k, s in r["status"].items() if not s.startswith("ok") and s != "withheld"]
    return 1 if bad else 0


def _generate(a, rest):
    from .generate import run
    out = a.out or str(Path("runs") / Path(a.adapter.split(":")[0]).stem.split(".")[-1])
    return run(a.bench, a.adapter, out, a.split, a.shard, a.limit, a.ablate, rest)


def _export(a):
    from .manifest import export
    ready, waiting, blocked, done = export(a.bench, a.out, a.split)
    return 1 if blocked else 0


def _collect(a):
    from .manifest import collect
    n, bad = collect(a.bench, a.out, a.table, a.split)
    return 1 if bad else 0


def _check(a):
    from .manifest import check
    miss, stale, unknown = check(a.bench, a.outputs, a.split, adopt=a.adopt)
    return 1 if miss or stale or unknown else 0


def _verified(a, who):
    """The provenance gate of `score` and `evaluate`: no Reuse query drawn from an older or unrecorded evidence frame."""
    from .manifest import check
    miss, stale, unknown = check(a.bench, a.outputs, a.split, verbose=not a.allow_unverified)
    if (stale or unknown) and not a.allow_unverified:
        print(f"[props {who}] {len(stale) + len(unknown)} Reuse queries were drawn from an older or an unknown "
              "evidence frame, so nothing was scored (see above). --allow-unverified scores them anyway; such a "
              "board cannot be submitted.")
        return None
    return miss


def _inputs_ok(a, who):
    """The frames must have been drawn from the data now on disk: same dataset commit, every input image present
    (an Ego image that failed its check after `props rebuild-ego` is set aside and counts as absent)."""
    from .data import Bench, data_commit, read_json
    run = read_json(Path(a.outputs) / "props_run.json", {})
    now = data_commit(a.bench)
    if run.get("data_commit") and now and run["data_commit"] != now:
        print(f"[props {who}] the frames were drawn from dataset commit {run['data_commit'][:10]}, but {a.bench} is "
              f"at {now[:10]}; draw them again from this version, or download the old one with --revision")
        return False
    b = Bench(a.bench)
    absent = [p["pair"] for p in b.pairs(a.split) if not b.ready(p) and not b.withheld(p)]
    if absent:
        print(f"[props {who}] {len(absent)} pairs have input images that are not on disk (first {absent[0]}), so "
              "nothing was scored. Run `props rebuild-ego`, or check its report for rejected images.")
        return False
    return True


def _complete_frames(a, who):
    """The gates of `score` and `evaluate`: the data the frames were drawn from, the provenance of the Reuse
    queries, then missing frames. An earlier board is moved aside first, so a failed run never leaves it behind."""
    old = Path(a.outputs) / "props_board.json"
    if old.exists():
        os.replace(old, old.with_name("props_board.previous.json"))
    if not _inputs_ok(a, who):
        return False
    miss = _verified(a, who)
    if miss is None:
        return False
    if miss and not a.allow_missing:
        print(f"[props {who}] {len(miss)} frames are missing, so nothing was scored. Draw them, or pass "
              "--allow-missing if your model did not return them (a missing frame fails its side).")
        return False
    return True


def _score(a):
    from .score import score
    if not _complete_frames(a, "score"):
        return 2
    done, errors = score(a.bench, a.outputs, a.judge, a.grounder, a.split, a.judge_model, a.grounder_model, a.conc)
    print(f"{len(done)} pairs in {Path(a.outputs) / 'props_scores.json'}")
    return 1 if errors else 0


def _board(a):
    from .board import board, show
    row = board(a.bench, a.outputs, a.split, a.ci)
    if a.json:
        print(json.dumps(row, indent=1))
    else:
        show(row)
    return 0


def _evaluate(a):
    from .board import board, record, show
    from .score import score
    if not _complete_frames(a, "evaluate"):
        return 2
    _, errors = score(a.bench, a.outputs, a.judge, a.grounder, a.split, a.judge_model, a.grounder_model, a.conc)
    if errors:
        print(f"[props evaluate] {len(errors)} pairs could not be scored; no board was written. "
              "Check the servers and run the same command again; scored pairs are kept.")
        return 1
    row = board(a.bench, a.outputs, a.split, ci=True)
    show(row)
    f = record(a.bench, a.outputs, row, a.judge, a.grounder, a.judge_model, a.grounder_model)
    print(f"[props evaluate] board -> {f}")
    return 0


def parser():
    ap = argparse.ArgumentParser(prog="props", description="PROPS benchmark: download, generate, score, board")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def bench(p):
        p.add_argument("--bench", default="data/PROPS-Benchmark", help="PROPS-Benchmark folder")

    def split(p, extra=()):
        p.add_argument("--split", default="full", choices=SPLITS + list(extra),
                       help="full (leaderboard), phase_a, open (no Ego4D needed); see props/data.py")

    def servers(p):
        p.add_argument("--judge", required=True, help="OpenAI-compatible base URL(s) of Gemma-4-31B-it, comma separated")
        p.add_argument("--grounder", required=True, help="base URL(s) of Qwen3.8-27B, comma separated")
        p.add_argument("--judge-model", default="judge", help="served model name of the judge")
        p.add_argument("--grounder-model", default="grounder", help="served model name of the grounder")
        p.add_argument("--conc", type=int, default=16, help="requests in flight per model")
        p.add_argument("--allow-unverified", action="store_true",
                       help="score Reuse queries whose evidence frame has no record or has changed")
        p.add_argument("--allow-missing", action="store_true", help="score even if frames are missing (they fail)")

    p = sub.add_parser("download", help="download the datasets from Hugging Face")
    p.add_argument("--out", default="data")
    p.add_argument("--train", action="store_true", help="also download PROPS-Train")
    p.add_argument("--train-only", action="store_true", help="download PROPS-Train only")
    p.add_argument("--no-ego", action="store_true", help="skip the Ego4D / Ego-Exo4D pointers and patches")
    p.add_argument("--keep-tars", action="store_true", help="keep the PROPS-Train image tars after unpacking them")
    p.add_argument("--revision", help="dataset revision (tag or commit) to download, default the latest")
    p.add_argument("--token", help="Hugging Face token with read access (default: $HF_TOKEN or the cached login)")

    p = sub.add_parser("rebuild-ego", help="rebuild the Ego4D / Ego-Exo4D images from the official videos")
    p.add_argument("--data", required=True, help="PROPS-Benchmark or PROPS-Train folder")
    p.add_argument("--ego4d", help="Ego4D root that holds v2/clips/")
    p.add_argument("--egoexo4d", help="Ego-Exo4D root that holds takes/")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-dist", type=int, default=8, help="largest accepted pHash Hamming distance (of 64 bits)")
    p.add_argument("--ffmpeg", help="ffmpeg binary, if not in PATH (PROPS-Train keyframes_fps1 rows)")
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("generate", help="draw the frames of a run with an adapter (arguments after -- go to it)")
    bench(p), split(p, ["ready"])
    p.add_argument("--adapter", required=True, help="module (props.adapters.qwen_edit) or file (my_model.py[:Class])")
    p.add_argument("--out", help="run folder (default runs/<adapter name>)")
    p.add_argument("--shard", default="0/1", help="i/n: this process takes every n-th item from the i-th")
    p.add_argument("--limit", type=int, default=0, help="first N items only (smoke test)")
    p.add_argument("--ablate", default="none", choices=["none", "nof0", "prevonly", "textonly", "shuffle"])

    p = sub.add_parser("export", help="write requests.jsonl: the frames to draw now, with their inputs")
    bench(p), split(p)
    p.add_argument("--out", required=True, help="run folder; frames go to <out>/<item>/<step>.png")

    p = sub.add_parser("collect", help="copy frames listed in a CSV (item,step,path) into a run folder")
    bench(p), split(p)
    p.add_argument("--out", required=True, help="run folder")
    p.add_argument("--table", required=True, help="CSV with the columns item, step, path")

    p = sub.add_parser("check", help="list the frames of a run that are not on disk yet")
    bench(p), split(p)
    p.add_argument("--outputs", required=True, help="run folder")
    p.add_argument("--adopt", action="store_true",
                   help="record the evidence frames on disk for Reuse queries that have no record (frames placed by hand)")

    p = sub.add_parser("score", help="score a run with the judge and the grounder")
    bench(p), split(p), servers(p)
    p.add_argument("--outputs", required=True, help="run folder")

    p = sub.add_parser("board", help="leaderboard numbers of a scored run")
    bench(p), split(p)
    p.add_argument("--outputs", required=True, help="run folder")
    p.add_argument("--ci", action="store_true", help="95%% intervals from a bootstrap over source-video clusters")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("evaluate", help="check, score and board a run; writes <outputs>/props_board.json")
    bench(p), split(p), servers(p)
    p.add_argument("--outputs", required=True, help="run folder")
    return ap


def main(argv=None):
    try:
        return _main(argv)
    except BrokenPipeError:  # `props board | head` ends quietly (SIGPIPE stays ignored: sockets to the servers)
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 1


def _main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    rest = []
    if "--" in argv:
        k = argv.index("--")
        argv, rest = argv[:k], argv[k + 1:]
    a = parser().parse_args(argv)
    if a.cmd == "generate":
        try:
            return _generate(a, rest)
        except FileNotFoundError as e:
            print(f"props generate: {e}", file=sys.stderr)
            return 2
    if rest:
        parser().error("arguments after -- are only used by `props generate`")
    cmds = {"download": _download, "rebuild-ego": _rebuild, "export": _export, "collect": _collect,
            "check": _check, "score": _score, "board": _board, "evaluate": _evaluate}
    try:
        return cmds[a.cmd](a)
    except FileNotFoundError as e:
        print(f"props {a.cmd}: {e}", file=sys.stderr)
        return 2
    except BrokenPipeError:
        raise
    except (ConnectionError, ValueError) as e:  # servers down, or serving different weights
        print(f"props {a.cmd}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
