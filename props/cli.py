"""The `props` command.

    props download     --out data [--train] [--no-ego] [--token T]
    props rebuild-ego  --data data/PROPS-Benchmark --ego4d DIR --egoexo4d DIR
    props generate     --bench data/PROPS-Benchmark --adapter SPEC --out runs/NAME [--subset ready] -- ADAPTER ARGS
    props score        --bench data/PROPS-Benchmark --outputs runs/NAME --judge URL --grounder URL
    props board        --bench data/PROPS-Benchmark --outputs runs/NAME [--subset phase_a] [--ci] [--json]
"""
import argparse, json, os, sys
from pathlib import Path


def _download(a):
    from .data import REPO_BENCH, REPO_TRAIN, download
    token = a.token or os.environ.get("HF_TOKEN")
    for repo in [REPO_BENCH] + ([REPO_TRAIN] if a.train else []):
        dst = Path(a.out) / repo.split("/")[-1]
        print(f"{repo} -> {dst}", flush=True)
        download(dst, repo, token, ego=not a.no_ego, keep_tars=a.keep_tars)
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
    return run(a.bench, a.adapter, out, a.subset, a.shard, a.limit, a.ablate, rest)


def _score(a):
    from .score import score
    done = score(a.bench, a.outputs, a.judge, a.grounder, a.subset, a.judge_model, a.grounder_model, a.conc)
    print(f"{len(done)} pairs in {Path(a.outputs) / 'props_scores.json'}")
    return 0


def _board(a):
    from .board import board, show
    row = board(a.bench, a.outputs, a.subset, a.ci)
    if a.json:
        print(json.dumps(row, indent=1))
    else:
        show(row)
    return 0


def parser():
    ap = argparse.ArgumentParser(prog="props", description="PROPS benchmark: download, rebuild, generate, score, board")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("download", help="download the datasets from Hugging Face")
    p.add_argument("--out", default="data")
    p.add_argument("--train", action="store_true", help="also download PROPS-Train")
    p.add_argument("--no-ego", action="store_true", help="skip the Ego4D / Ego-Exo4D pointers and patches")
    p.add_argument("--keep-tars", action="store_true", help="keep the PROPS-Train image tars after unpacking them")
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
    p.add_argument("--bench", default="data/PROPS-Benchmark")
    p.add_argument("--adapter", required=True, help="module (props.adapters.qwen_edit) or file (my_model.py[:Class])")
    p.add_argument("--out", help="run folder (default runs/<adapter name>)")
    p.add_argument("--subset", default="ready", choices=["all", "open", "phase_a", "ready"])
    p.add_argument("--shard", default="0/1", help="i/n: this process takes every n-th item from the i-th")
    p.add_argument("--limit", type=int, default=0, help="first N items only (smoke test)")
    p.add_argument("--ablate", default="none", choices=["none", "nof0", "prevonly", "textonly", "shuffle"])

    p = sub.add_parser("score", help="score a run with the judge and the grounder")
    p.add_argument("--bench", default="data/PROPS-Benchmark")
    p.add_argument("--outputs", required=True, help="run folder written by `props generate`")
    p.add_argument("--judge", required=True, help="OpenAI-compatible base URL(s) of Gemma-4-31B-it, comma separated")
    p.add_argument("--grounder", required=True, help="base URL(s) of Qwen3.8-27B, comma separated")
    p.add_argument("--judge-model", default="judge")
    p.add_argument("--grounder-model", default="grounder")
    p.add_argument("--subset", default="all", choices=["all", "open", "phase_a", "ready"])
    p.add_argument("--conc", type=int, default=16, help="requests in flight per model")

    p = sub.add_parser("board", help="leaderboard numbers of a scored run")
    p.add_argument("--bench", default="data/PROPS-Benchmark")
    p.add_argument("--outputs", required=True)
    p.add_argument("--subset", default="all", choices=["all", "open", "phase_a"])
    p.add_argument("--ci", action="store_true", help="95%% intervals from a bootstrap over source-video clusters")
    p.add_argument("--json", action="store_true")
    return ap


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    rest = []
    if "--" in argv:
        k = argv.index("--")
        argv, rest = argv[:k], argv[k + 1:]
    a = parser().parse_args(argv)
    if a.cmd == "generate":
        return _generate(a, rest)
    if rest:
        parser().error("arguments after -- are only used by `props generate`")
    return {"download": _download, "rebuild-ego": _rebuild, "score": _score, "board": _board}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
