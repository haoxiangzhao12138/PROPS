"""Evaluate frames drawn outside the `props generate` runner.

Many models run in their own environment, on another cluster or behind a batch API. For them PROPS works in three
commands instead of an adapter:

    props export  --out runs/mymodel             # requests.jsonl: every frame to draw, with its inputs
    # draw each request with your own pipeline and save it at request["output"]
    # (or keep your own file names and list them in a CSV, see `props collect`)
    props check   --outputs runs/mymodel         # what is still missing
    props evaluate --outputs runs/mymodel ...    # score and leaderboard row

Most items need one frame, the query. Reuse items need two: the model first draws the evidence step, and that frame
stays in the history of the query, so the model is asked to keep a look it drew itself. `props export` therefore
writes a request only once all of its inputs exist. The first export lists every query of Recall and Resist and the
evidence steps of Reuse. After those evidence frames are saved, export again: the second round lists the Reuse
queries, whose history now holds your own evidence frames.

A Reuse query must be drawn from the evidence frame that is on disk. Every way of filling a run folder (`props
export`, `props collect`, `props generate`) writes the sha1 of the evidence frames a query was drawn from to
<out>/<item>/props_inputs.json. `props check` compares them with the frames on disk, so a query drawn from an older
evidence frame (the evidence was drawn again later) is found, and so is a query with no such record.
"""
import csv, json, os, shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from .data import Bench, load_hashes, note_run, output_path, read_json, save_hashes, sha1, write_json
from .generate import context, save


def _history(b, out, item, j):
    """Paths of the images shown before step j (given frames and the model's own kept frames), or None if one of
    the model's own frames is not drawn yet."""
    paths = []
    for k in range(j):
        f = item["frames"][k]["image"]
        if f is not None:
            p = b.path(f)
        elif k in item["keep_own"]:
            p = output_path(out, item["id"], k)
            if not p.exists():
                return None
        else:
            raise ValueError(f"{item['id']}: frame {k} is neither given nor kept")
        paths.append(p.resolve())
    return paths




def note_inputs(out, iid, step, own):
    """Record the sha1 of the model's own frames that step `step` of item `iid` was drawn from. One file per item,
    so parallel `props generate` shards never write the same file."""
    if not own:
        return
    f = Path(out) / iid / "props_inputs.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    rec = read_json(f, {})
    rec[str(step)] = own
    write_json(f, rec)


def provenance(out, item, step):
    """Whether step `step` of `item` was drawn from the model's own frames now on disk: None if its history holds
    none of them, else "ok", "stale" (they changed since) or "unknown" (no record)."""
    now = own_inputs(out, item, step)
    if not now:
        return None
    rec = read_json(Path(out) / item["id"] / "props_inputs.json", {})
    return "unknown" if str(step) not in rec else "ok" if rec[str(step)] == now else "stale"


def own_inputs(out, item, step):
    """sha1 of the model's own frames on disk that step `step` keeps in its history, or None if one is missing."""
    hs = {}
    for k in item["keep_own"]:
        if k < step:
            e = output_path(out, item["id"], k)
            if not e.exists():
                return None
            hs[str(k)] = sha1(e)
    return hs


def requests(bench, out, split="full"):
    """(ready, waiting, blocked, done): the steps that can be drawn now, the steps that wait for one of the model's
    own frames, the steps whose given input images are not on disk (Ego pairs before `props rebuild-ego`), and the
    number of steps already drawn."""
    b, out = Bench(bench), Path(out)
    ready, waiting, blocked, done = [], [], [], 0
    for p in b.pairs(split):
        for iid in p["items"]:
            it = b.item(iid)
            for j in it["draw"]:
                if output_path(out, iid, j).exists():
                    done += 1
                    continue
                hist = _history(b, out, it, j)
                if hist is None:
                    waiting.append((iid, j))
                    continue
                if not all(h.exists() for h in hist):
                    blocked.append((iid, j))
                    continue
                ctx = context(it, j, hist, it["frames"][0]["size"])
                ready.append({"item": iid, "step": j, "query": j == it["query"],
                              "own": {str(k): sha1(hist[k]) for k in it["keep_own"] if k < j},
                              "output": str((out / iid / f"{j}.png").resolve()),
                              "history": [{"image": str(h), "time": "start" if k == 0 else it["steps"][k - 1]["time"],
                                           "script": None if k == 0 else it["steps"][k - 1]["script"]}
                                          for k, h in enumerate(hist)],
                              "time": ctx["dt_text"][-1], "script": ctx["script"], "prompt": ctx["prompt"],
                              "size": ctx["size"], "seed": ctx["seed"]})
    return ready, waiting, blocked, done


def export(bench, out, split="full"):
    """Write <out>/requests.jsonl with the frames that can be drawn now. Returns (ready, waiting, blocked, done)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ready, waiting, blocked, done = requests(bench, out, split)
    f = out / "requests.jsonl"
    with open(f, "w") as fh:
        for r in ready:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    for r in ready:
        note_inputs(out, r["item"], r["step"], r["own"])
    note_run(out, bench)
    print(f"[props export] split {split}: {len(ready)} frames to draw now -> {f}")
    if waiting:
        print(f"[props export] {len(waiting)} Reuse queries wait for the evidence frames in this list; "
              "draw them, then run `props export` again")
    if blocked:
        print(f"[props export] {len(blocked)} frames need input images that are not on disk; check that "
              "`props download` finished, run `props rebuild-ego` for the Ego4D / Ego-Exo4D pairs, or use --split open")
    print(f"[props export] {done} frames already drawn")
    return len(ready), len(waiting), len(blocked), done


def place(src, dst, size):
    """Put one frame into the run folder: a PNG of the right size is copied as it is, anything else is converted."""
    with Image.open(src) as im:
        if im.format == "PNG" and im.size == tuple(size) and im.mode == "RGB":
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_suffix(".tmp.png")
            shutil.copyfile(src, tmp)
            os.replace(tmp, dst)
        else:
            save(im, dst, size)


def collect(bench, out, table, split="full", workers=16):
    """Copy frames that you saved under your own names into the run folder. `table` is a CSV with the columns
    item, step, path (one row per drawn frame); `props export` lists the item and step of every frame."""
    b, out = Bench(bench), Path(out)
    known = {(i, j): b.item(i)["frames"][0]["size"] for p in b.pairs(split) for i in p["items"] for j in b.item(i)["draw"]}
    bad, rows = [], []
    with open(table, newline="") as fh:
        for row in csv.DictReader(fh):
            key = (row["item"].strip(), int(row["step"]))
            if key not in known:
                bad.append(f"{key[0]} step {key[1]} is not a frame to draw in split {split}")
                continue
            rows.append((key, Path(row["path"].strip()).expanduser()))

    def one(r):
        (key, src) = r
        try:
            place(src, output_path(out, *key), known[key])
            return key, None
        except (OSError, ValueError) as e:  # missing or unreadable file
            return key, f"{src}: {e}"
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(one, rows))
    bad += [e for _, e in res if e]
    copied = {k for k, e in res if not e}
    n = len(copied)
    if copied:
        note_run(out, bench)
    for iid, j in copied:  # a query listed with its evidence frame in the same table was drawn from that frame
        it = b.item(iid)
        if all((iid, k) in copied for k in it["keep_own"] if k < j):
            note_inputs(out, iid, j, own_inputs(out, it, j))
    print(f"[props collect] {n} frames copied into {out}")
    for x in bad[:20]:
        print("  skipped:", x)
    if len(bad) > 20:
        print(f"  ... and {len(bad) - 20} more")
    return n, bad


def check(bench, out, split="full", verbose=True, adopt=False):
    """Which drawn frames are missing, and which Reuse queries were drawn from an evidence frame that has changed
    since or has no record. Returns (missing, stale, unknown), lists of (item, step). adopt=True records the
    evidence frames now on disk for the queries without a record (for frames placed in the folder by hand)."""
    b, out = Bench(bench), Path(out)
    load_hashes(out)
    miss, stale, unknown, total = [], [], [], 0
    for p in b.pairs(split):
        for iid in p["items"]:
            it = b.item(iid)
            for j in it["draw"]:
                total += 1
                if not output_path(out, iid, j).exists():
                    miss.append((iid, j))
                    continue
                st = provenance(out, it, j)  # None: nothing of its own in the history, or the evidence is missing
                if st == "unknown" and adopt:
                    note_inputs(out, iid, j, own_inputs(out, it, j))
                elif st == "unknown":
                    unknown.append((iid, j))
                elif st == "stale":
                    stale.append((iid, j))
    save_hashes(out)
    if verbose:
        print(f"[props check] split {split}: {total - len(miss)} of {total} frames on disk")
        for iid, j in miss[:10]:
            print(f"  missing {output_path(out, iid, j)}")
        if len(miss) > 10:
            print(f"  ... and {len(miss) - 10} more (`props export` lists them with their inputs)")
        for iid, j in stale[:10]:
            print(f"  {output_path(out, iid, j)} was drawn from an evidence frame that has changed since; "
                  "delete it and draw it again")
        if len(stale) > 10:
            print(f"  ... and {len(stale) - 10} more such queries")
        if unknown:
            print(f"  {len(unknown)} Reuse queries have no record of the evidence frame they were drawn from "
                  f"(first {output_path(out, *unknown[0])}). Draw them through `props export` or `props generate`, "
                  "copy them with `props collect`, or, if you are sure they were drawn from the evidence frames now "
                  "on disk, run `props check --adopt`")
    return miss, stale, unknown
