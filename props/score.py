"""Score a run on PROPS: Follow and Exec per side, pass per pair.

    props score --bench data/PROPS-Benchmark --outputs runs/mymodel \
        --judge http://HOST:9000/v1 --grounder http://HOST:8000/v1

Follow. The judge sees the two reference crops of X and the output frame and answers 1, 2 or neither. It is asked
twice with the crops swapped; the side follows only if both answers name the side's own look.
Reuse. The reference crops are cut from the model's own drawn evidence frames: the grounder boxes X there, the
judge and the grounder check that the crop shows an object of the right kind, and a box far from the box found on
the true frame is dropped when both say it is not one. If X is not found on either side, both sides fail.
Exec. The judge answers the execution questions of the side on the output frame; all must be right.
A side passes when it follows and executes; a pair passes when both sides pass. A missing output fails its side.
Results: <outputs>/props_scores.json {pair: {"A": side result, "B": side result}}.
"""
import asyncio, json
from pathlib import Path

from PIL import Image

from . import prompts as T
from .data import REUSE, Bench, output_path
from .judge import Client, answer

FAIL = {"fwd": "", "rev": "", "follow": False, "other": False, "ok": False}


def px_box(b, size):
    w, h = size
    x1, y1, x2, y2 = [float(v) for v in b]
    return [max(0, min(x1, x2) * w / 1000), max(0, min(y1, y2) * h / 1000),
            min(w, max(x1, x2) * w / 1000), min(h, max(y1, y2) * h / 1000)]


def obj_crop(img, b, pad=0.15):
    w, h = b[2] - b[0], b[3] - b[1]
    return img.crop((int(max(0, b[0] - pad * w)), int(max(0, b[1] - pad * h)),
                     int(min(img.width, b[2] + pad * w)), int(min(img.height, b[3] + pad * h))))


def box_iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x2 - x1) * max(0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


async def own_crop(judge, grounder, out, kind, sd):
    """Reuse: the crop of X in the frame the model drew at the evidence step, or None."""
    d = Path(out) / sd["item"]
    img, f = d / f"{sd['evidence_step']}.png", d / "own_e.json"
    if not img.exists():
        return None
    if f.exists():
        o = json.loads(f.read_text())
        return d / "own_e.jpg" if o["found"] else None

    async def is_kind(c):
        rs = await asyncio.gather(*[q.ask(T.CHECK_SYS, [T.OWN_CROP.format(kind=kind), c], max_side=1280)
                                    for q in (judge, grounder)])
        return any(answer(r).lower() != "no" for r in rs)
    look, where = sd.get("look") or "", sd.get("where") or ""
    im, ok, r = Image.open(img).convert("RGB"), False, {}
    for p in (T.GROUND.format(more="", name=kind, look=look, where=where),
              T.OWN_GROUND.format(name=kind, look=look, where=where),
              T.OWN_GROUND.format(name=kind, look="any colour", where=where)):
        r = await grounder.ask(T.GROUND_SYS, [p, img], max_side=1280)
        r = r if isinstance(r, dict) else {}
        ok = bool(r.get("found") and len(r.get("bbox") or []) == 4)
        if ok:
            obj_crop(im, px_box(r["bbox"], im.size)).save(d / "own_e.jpg", quality=95)
            ok = await is_kind(d / "own_e.jpg")
        if ok:
            break
    o = {"found": ok, "bbox": r.get("bbox")}
    if ok and sd.get("gt_own_bbox") and box_iou(r["bbox"], sd["gt_own_bbox"]) < 0.3:
        rs = await asyncio.gather(*[q.ask(T.CHECK_SYS, [T.OWN_STRICT.format(kind=kind), d / "own_e.jpg"], max_side=1280)
                                    for q in (judge, grounder)])
        o["veto"] = all(answer(x).lower() == "no" for x in rs)
        o["found"] = not o["veto"]
    f.write_text(json.dumps(o))
    return d / "own_e.jpg" if o["found"] else None


async def side(judge, kind, v, crops, img, exec_qs):
    cA, cB = crops
    if cA is None or cB is None:
        return {**FAIL, "own_found": False, "n_change": 0}

    async def ask(c1, c2):
        return answer(await judge.ask(T.CHOICE_SYS, [T.CHOICE.format(kind=kind), c1, c2, img], max_side=1024)).strip('"')
    fwd, rev = await ask(cA, cB), await ask(cB, cA)
    picked = {"1": "A", "2": "B"}.get(fwd), {"1": "B", "2": "A"}.get(rev)
    res = {"fwd": fwd, "rev": rev, "follow": picked[0] == picked[1] == v,
           "other": picked[0] == picked[1] and picked[0] not in (None, v)}
    if exec_qs:
        n = len(exec_qs)
        r = await judge.ask(T.JSYS, [T.SEE.format(qs=T.numbered(exec_qs)), img], max_side=896, max_tokens=60 + 8 * n)
        ans = [str(x).strip().lower() for x in ((r.get("answers") if isinstance(r, dict) else None) or [])]
        ans = ans + ["unsure"] * (n - len(ans))
        res["exec"] = all(y == x["a"] for x, y in zip(exec_qs, ans))
        res["exec_ans"] = ans
    res["n_change"] = len(exec_qs)
    res["ok"] = res["follow"] and res.get("exec", True)
    return res


async def score_pair(b, judge, grounder, out, pair):
    jd = b.judge(pair["pair"])
    sides, kind = jd["sides"], jd["kind"]
    if pair["subtype"] in REUSE:
        cs = await asyncio.gather(*[own_crop(judge, grounder, out, kind, sides[v]) for v in "AB"])
        crops = tuple(cs)
    else:
        crops = (b.path(sides["A"]["crop"]), b.path(sides["B"]["crop"]))
    res = {}
    for v in "AB":
        it = b.item(sides[v]["item"])
        img = output_path(out, it["id"], it["query"])
        if not img.exists():
            res[v] = {**FAIL, "missing": True, "n_change": 0}
            continue
        res[v] = await side(judge, kind, v, crops, img, sides[v]["exec"])
    return res


async def score_async(bench, out, judge_url, grounder_url, subset="all", judge_model="judge", grounder_model="grounder",
                      conc=16, cache=None):
    b = Bench(bench)
    cache = cache or str(Path(out) / ".judge_cache")
    judge = Client(judge_url, judge_model, concurrency=conc, cache=cache)
    grounder = Client(grounder_url, grounder_model, concurrency=conc, cache=cache)
    f = Path(out) / "props_scores.json"
    done = json.loads(f.read_text()) if f.exists() else {}
    todo = [p for p in b.pairs(subset)  # pairs scored while an output was missing are scored again
            if p["pair"] not in done or any(s.get("missing") for s in done[p["pair"]].values())]
    print(f"[props score] {len(todo)} pairs to score ({len(done) - len(todo) + sum(p['pair'] not in done for p in todo)} already done)", flush=True)
    for n in range(0, len(todo), 200):  # write the scores every 200 pairs
        chunk = todo[n:n + 200]
        rs = await asyncio.gather(*[score_pair(b, judge, grounder, out, p) for p in chunk], return_exceptions=True)
        for p, r in zip(chunk, rs):
            if isinstance(r, Exception):
                print(f"{p['pair']}: {str(r)[:160]} (not saved; rerun to retry)", flush=True)
            else:
                done[p["pair"]] = r
        f.write_text(json.dumps(done, indent=1))
        print(f"[props score] {min(n + 200, len(todo))}/{len(todo)}", flush=True)
    return done


def score(*a, **k):
    return asyncio.run(score_async(*a, **k))
