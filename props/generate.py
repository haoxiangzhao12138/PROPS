"""Run a generator on PROPS items, teacher-forced.

For each item the runner walks the steps in order. A step not in item["draw"] is a given history frame. A step in
item["draw"] is drawn by the model from the history so far. Reuse items also draw their evidence step, and the
steps in item["keep_own"] then stay in the history as the model drew them (resized to the frame size), so the
model is later asked to keep a look it drew itself. The last drawn step is the query.

An adapter is any class with
    def __init__(self, args):      # args: the command-line arguments the runner did not use
    def generate(self, ctx):       # -> PIL.Image (or (PIL.Image, dict) to save a json note next to the frame)
ctx holds:
    item_id, j (step to draw, 1-based), K (number of steps), size (w, h) of the frames,
    history  [PIL] frame 0 .. j-1 (frame0 = history[0], prev = history[-1]),
    scripts  [str] scripts 1 .. j (the last one is `script`, the step to draw), dt_text [str] their times,
    prompt   the shared instruction (props.prompts.generation_prompt), seed (int, fixed per item and step).
Give the model the full history in its native input format, up to the number of images it accepts.
Outputs: <out>/<item id>/<j>.png at the frame size.
"""
import importlib, importlib.util, json, os, random, sys, time, traceback, zlib
from pathlib import Path

from PIL import Image

from .data import Bench, output_path
from .prompts import generation_prompt


def load_adapter(spec, args):
    """spec: a module (props.adapters.baselines) or a file (my_model.py), optionally with :ClassName (default Model)."""
    mod, _, cls = spec.partition(":")
    if mod.endswith(".py"):
        s = importlib.util.spec_from_file_location(Path(mod).stem, mod)
        m = importlib.util.module_from_spec(s)
        s.loader.exec_module(m)
    else:
        m = importlib.import_module(mod)
    return getattr(m, cls or "Model")(args)


def context(item, j, history, size):
    st = item["steps"]
    cur = st[j - 1].get("draw_script", st[j - 1]["script"])
    return {"item_id": item["id"], "j": j, "K": len(st), "size": tuple(size),
            "history": history, "frame0": history[0], "prev": history[-1],
            "scripts": [st[k]["script"] for k in range(j - 1)] + [cur],
            "dt_text": [st[k]["time"] for k in range(j)], "script": cur,
            "prompt": generation_prompt(item, j),
            "seed": zlib.crc32(f"{item['id']}|{j}".encode()) & 0x7FFFFFFF}


def ablate(hist, how, key):
    """Test-time changes of the shown images (history kept, weights fixed): nof0 = frame 0 grey,
    prevonly = all but the previous frame grey, textonly = all grey, shuffle = order permuted (seeded)."""
    grey = lambda im: Image.new("RGB", im.size, (128, 128, 128))  # noqa: E731
    if how == "nof0":
        return [grey(hist[0])] + hist[1:]
    if how == "prevonly":
        return [grey(x) for x in hist[:-1]] + hist[-1:]
    if how == "textonly":
        return [grey(x) for x in hist]
    if how == "shuffle":
        out = list(hist)
        random.Random(key).shuffle(out)
        return out
    return hist


def save(img, path, size):
    img = img.convert("RGB")
    if img.size != tuple(size):
        img = img.resize(tuple(size), Image.BICUBIC)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.png")
    img.save(tmp)
    os.replace(tmp, path)


def run(bench_dir, adapter, out, split="full", shard="0/1", limit=0, how="none", adapter_args=()):
    from .manifest import note_inputs, own_inputs
    b = Bench(bench_dir)
    pairs = b.pairs(split)
    iids = [i for p in pairs for i in p["items"]]
    si, sn = map(int, shard.split("/"))
    mine = [i for n, i in enumerate(iids) if n % sn == si]
    mine = mine[:limit] if limit else mine
    absent = [p["pair"] for p in pairs if set(p["items"]) & set(mine) and not b.ready(p)]
    if absent:
        print(f"[props generate] {len(absent)} pairs to draw have input images that are not on disk "
              f"(first {absent[0]}). Check that `props download` finished and run `props rebuild-ego` for the "
              "Ego4D / Ego-Exo4D frames.")
        return 2
    from .data import read_json
    rf = Path(out) / "props_run.json"
    run_note = read_json(rf, {})
    if run_note.get("ablate", how) != how:
        print(f"[props generate] {out} holds frames drawn with --ablate {run_note['ablate']}; use another --out")
        return 2
    from .data import note_run
    note_run(out, bench_dir, ablate=how, adapter=adapter)
    model = load_adapter(adapter, list(adapter_args))
    print(f"[props generate] {adapter}: {len(mine)} items (shard {shard}, split {split}) -> {out}", flush=True)
    t0, n_done, n_fail = time.time(), 0, 0
    for iid in mine:
        it = b.item(iid)
        size = it["frames"][0]["size"]
        given = b.frames(it)
        hist = [given[0]]
        for j in range(1, len(it["steps"]) + 1):
            if j not in it["draw"]:
                hist.append(given[j])
                continue
            p = output_path(out, iid, j)
            if p.exists():
                img = Image.open(p).convert("RGB")
            else:
                try:
                    r = model.generate(context(it, j, ablate(hist, how, f"{iid}:{j}"), size))
                except Exception:  # noqa: BLE001  a failed step leaves no file; the side then fails when scored
                    traceback.print_exc()
                    n_fail += 1
                    break
                img, note = r if isinstance(r, tuple) else (r, None)
                save(img, p, size)
                note_inputs(out, iid, j, own_inputs(out, it, j))
                if note is not None:
                    p.with_suffix(".json").write_text(json.dumps(note))
                n_done += 1
            if j < len(it["steps"]):
                assert j in it["keep_own"], f"{iid}: step {j} is drawn but not kept"
                hist.append(img.convert("RGB").resize(tuple(size)))
        print(f"{iid} | {n_done} frames, {(time.time() - t0) / max(n_done, 1):.1f} s/frame", flush=True)
    return 1 if n_fail else 0


if __name__ == "__main__":
    from .cli import main
    sys.exit(main(["generate"] + sys.argv[1:]))
