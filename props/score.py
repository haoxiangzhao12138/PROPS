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
import asyncio, hashlib, json, os, time
from pathlib import Path

from PIL import Image

from . import prompts as T
from .data import REUSE, Bench, load_hashes, output_path, read_json, save_hashes, sha1, write_json
from .judge import Client, answer, identity
from .manifest import provenance

FAIL = {"fwd": "", "rev": "", "follow": False, "other": False, "ok": False}
PROTOCOL = "props-score-v1"  # bump when the decision rules change
PROMPTS = hashlib.sha1(json.dumps([T.CHOICE_SYS, T.CHOICE, T.JSYS, T.SEE, T.GROUND_SYS, T.GROUND, T.OWN_GROUND,
                                   T.CHECK_SYS, T.OWN_CROP, T.OWN_STRICT]).encode()).hexdigest()[:12]
CODE = hashlib.sha1(b"".join((Path(__file__).parent / f).read_bytes()
                             for f in ("score.py", "judge.py", "prompts.py"))).hexdigest()[:12]
RULES = f"{PROTOCOL}/{PROMPTS}/{CODE}"  # the decision rules, the prompts and the scoring code; part of every fingerprint


def fingerprint(b, out, pair, judge_id, grounder_id):
    """What a pair's score depends on: the rules and prompts, the two models (name, weights, vLLM version), the item
    and judge files of the pair, its reference crops, the content of the drawn frames it reads and, for Reuse,
    whether each query is recorded as drawn from its evidence frame."""
    jd, files = b.judge(pair["pair"]), [sha1(b.root / "judge" / f"{pair['pair']}.json")]
    for v in "AB":
        sd = jd["sides"][v]
        it = b.item(sd["item"])
        files += [sha1(b.root / "items" / f"{it['id']}.json"), sha1(output_path(out, it["id"], it["query"]))]
        if pair["subtype"] in REUSE:  # the evidence frame, the crop cut from it, and the provenance of the query
            d = Path(out) / it["id"]
            files += [sha1(output_path(out, it["id"], sd["evidence_step"])), sha1(d / "own_e.json"),
                      sha1(d / "own_e.jpg"), provenance(out, it, it["query"])]
        else:
            files.append(sha1(b.path(sd["crop"])))
    return hashlib.sha1(json.dumps([RULES, judge_id, grounder_id, files]).encode()).hexdigest()[:16]


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


async def own_crop(judge, grounder, out, kind, sd, size):
    """Reuse: the crop of X in the frame the model drew at the evidence step (at the frame size), or None."""
    d = Path(out) / sd["item"]
    img, f = d / f"{sd['evidence_step']}.png", d / "own_e.json"
    if not img.exists():
        return None
    task = [kind, sd.get("look"), sd.get("where"), sd.get("gt_own_bbox")]
    key = {"src": sha1(img), "protocol": RULES,  # the evidence frame, the description and the models of this crop
           "task": hashlib.sha1(json.dumps(task).encode()).hexdigest()[:16],
           "models": [getattr(judge, "ident", None), getattr(grounder, "ident", None)]}
    o = read_json(f)
    if o and all(o.get(k) == v for k, v in key.items()) and (not o.get("found") or (d / "own_e.jpg").exists()):
        return d / "own_e.jpg" if o["found"] else None

    async def is_kind(c):
        rs = await asyncio.gather(*[q.ask(T.CHECK_SYS, [T.OWN_CROP.format(kind=kind), c], max_side=1280)
                                    for q in (judge, grounder)])
        return any(answer(r).lower() != "no" for r in rs)
    look, where = sd.get("look") or "", sd.get("where") or ""
    im, ok, r = Image.open(img).convert("RGB"), False, {}
    if im.size != tuple(size):
        im = im.resize(tuple(size), Image.BICUBIC)
    for n, p in enumerate((T.GROUND.format(more="", name=kind, look=look, where=where),
                           T.OWN_GROUND.format(name=kind, look=look, where=where),
                           T.OWN_GROUND.format(name=kind, look="any colour", where=where))):
        r = await grounder.ask(T.GROUND_SYS, [p, (img, size)], max_side=1280)
        r = r if isinstance(r, dict) else {}
        ok = bool(r.get("found") and len(r.get("bbox") or []) == 4)
        if ok:
            cand = d / f"own_e_try{n}.jpg"  # one file per attempt, so no attempt is judged on an earlier crop
            obj_crop(im, px_box(r["bbox"], im.size)).save(cand, quality=95)
            ok = await is_kind(cand)
            if ok:
                os.replace(cand, d / "own_e.jpg")
        if ok:
            break
    o = {"found": ok, "bbox": r.get("bbox"), **key}
    if ok and sd.get("gt_own_bbox") and box_iou(r["bbox"], sd["gt_own_bbox"]) < 0.3:
        rs = await asyncio.gather(*[q.ask(T.CHECK_SYS, [T.OWN_STRICT.format(kind=kind), d / "own_e.jpg"], max_side=1280)
                                    for q in (judge, grounder)])
        o["veto"] = all(answer(x).lower() == "no" for x in rs)
        o["found"] = not o["veto"]
    write_json(f, o)
    return d / "own_e.jpg" if o["found"] else None


async def side(judge, kind, v, crops, img, exec_qs):
    """img: (path of the drawn frame, frame size)."""
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


def img_ok(out, item):
    return output_path(out, item["id"], item["query"]).exists()


async def score_pair(b, judge, grounder, out, pair):
    jd = b.judge(pair["pair"])
    sides, kind = jd["sides"], jd["kind"]
    if pair["subtype"] in REUSE:
        cs = await asyncio.gather(*[own_crop(judge, grounder, out, kind, sides[v],
                                             b.item(sides[v]["item"])["frames"][0]["size"]) for v in "AB"])
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
        res[v] = await side(judge, kind, v, crops, (img, it["frames"][0]["size"]), sides[v]["exec"])
    if pair["subtype"] in REUSE:  # a side whose own evidence frame was not drawn is a missing output, not drift
        for v in "AB":
            it = b.item(sides[v]["item"])
            if not output_path(out, it["id"], sides[v]["evidence_step"]).exists():
                res[v]["missing"] = True
            elif img_ok(out, it) and provenance(out, it, it["query"]) != "ok":  # scored with --allow-unverified
                res[v]["unverified"] = True
    return res


async def score_async(bench, out, judge_url, grounder_url, split="full", judge_model="judge", grounder_model="grounder",
                      conc=16, cache=None):
    """Score every pair of the split whose result is missing or out of date. Returns (scores, errors).
    A pair is scored again when one of the frames it reads changes, or the protocol or the weights behind a server
    change (fingerprints in props_scores.fp.json; the weights are those the servers report in /v1/models).
    Requests that fail are reported as errors and the pair is left unscored, so a server problem never turns into
    a low score."""
    b = Bench(bench)
    cache = cache or str(Path(out) / ".judge_cache")
    cache = cache if cache != "none" else None
    (wj, kj), (wg, kg) = identity(judge_url, judge_model), identity(grounder_url, grounder_model)
    if not (kj and kg):  # the weights could change without a trace, so nothing earlier is reused
        print("[props score] warning: a server reports no weight revision or no vLLM version (serve with "
              "scripts/serve_judge.sh); every pair is scored again and no reply is cached", flush=True)
        cache, wj, wg = None, f"{wj}#{time.time_ns()}", f"{wg}#{time.time_ns()}"
    load_hashes(out)
    judge = Client(judge_url, judge_model, concurrency=conc, cache=cache, ident=wj)  # cache None: no disk cache
    grounder = Client(grounder_url, grounder_model, concurrency=conc, cache=cache, ident=wg)
    f, ff = Path(out) / "props_scores.json", Path(out) / "props_scores.fp.json"
    done, fps = read_json(f, {}), read_json(ff, {})
    want = {p["pair"]: fingerprint(b, out, p, wj, wg) for p in b.pairs(split)}
    todo = [p for p in b.pairs(split) if p["pair"] not in done or fps.get(p["pair"]) != want[p["pair"]]]
    print(f"[props score] {len(todo)} of {len(want)} pairs to score", flush=True)
    errors = {}
    for n in range(0, len(todo), 200):  # write the scores every 200 pairs
        chunk = todo[n:n + 200]
        rs = await asyncio.gather(*[score_pair(b, judge, grounder, out, p) for p in chunk], return_exceptions=True)
        for p, r in zip(chunk, rs):
            if isinstance(r, Exception):
                errors[p["pair"]] = str(r)[:300]
                done.pop(p["pair"], None)
                fps.pop(p["pair"], None)
            else:  # the fingerprint after scoring, which covers the Reuse crops just cut
                done[p["pair"]], fps[p["pair"]] = r, fingerprint(b, out, p, wj, wg)
        write_json(f, done, indent=1)
        write_json(ff, fps, indent=0)
        print(f"[props score] {min(n + 200, len(todo))}/{len(todo)}, {len(errors)} errors", flush=True)
    save_hashes(out)
    ef = Path(out) / "props_score_errors.json"
    if not errors and ef.exists():
        ef.unlink()  # the errors of an earlier run were retried
    if errors:
        write_json(ef, errors, indent=1)
        print(f"[props score] {len(errors)} pairs failed to score (server errors, see props_score_errors.json); "
              "they are left unscored, run the same command again to retry", flush=True)
    return done, errors


def score(*a, **k):
    return asyncio.run(score_async(*a, **k))
