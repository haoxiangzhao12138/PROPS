"""Rebuild the Ego4D and Ego-Exo4D images of a PROPS dataset from the official videos.

PROPS-Benchmark and PROPS-Train do not contain Ego4D or Ego-Exo4D pixels. Each such image has a row in ego/pointers.jsonl that says how
to make it from the official video, and this script makes it and checks it against the released perceptual hash.
Get the videos from https://ego4d-data.org first:
    Ego4D      ego4d --datasets clips --version v2          -> <ego4d>/v2/clips/<uid>.mp4
    Ego-Exo4D  egoexo --parts takes                         -> <egoexo4d>/takes/<take>/frame_aligned_videos/<cam>.mp4
then
    props rebuild-ego --data data/PROPS-Benchmark --ego4d <ego4d> --egoexo4d <egoexo4d>

Pointer rows of PROPS-Benchmark ("how"):
    frame     decode the frame with presentation timestamp pts; if "undistort", map the Aria fisheye to a 90 degree
              1024 px pinhole view, else shrink the long side to at most 1280 px; save as JPEG quality 93
    edit      paste the RGBA patch (edited pixels of one look) into its base frame at "box"; save as PNG
    crop      cut "rect" from the image "of"; save as JPEG quality 95
    withheld  built on a relit frame that no pointer can rebuild; the pair says so in pairs.jsonl "rebuild"
Pointer rows of PROPS-Train ("recipe"; frames of at most 640 px):
    seek            seek to second t and take the first decoded frame; save as JPEG quality 85
    keyframes_fps1  ffmpeg keyframes at one frame per second, frame t (needs ffmpeg in PATH or --ffmpeg)
Decoders differ slightly between versions, so a rebuilt frame matches the released one closely but not bit for bit.
The check compares 64-bit pHashes and accepts a Hamming distance of at most --max-dist. An image that fails the
check, and every edit or crop made from it, is renamed to <name>.rejected, so its pairs do not count as ready.
"""
import argparse, collections, json, os, shutil, subprocess, tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

MAX_SIDE, UNDIST, F_640, FOV = 1280, 1024, 277.0, 90.0


def phash(p):
    """64-bit DCT perceptual hash (32x32 grey, top-left 8x8 without DC vs median), hex."""
    g = np.asarray(Image.open(p).convert("L").resize((32, 32), Image.LANCZOS), np.float64)
    n = np.arange(32)
    D = np.cos(np.pi * (2 * n[None, :] + 1) * n[:, None] / 64)
    c = (D @ g @ D.T)[:8, :8].flatten()
    bits = c > np.median(c[1:])
    return "%016x" % int("".join("1" if b else "0" for b in bits), 2)


def dist(a, b):
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def hq(rgb, und):
    import cv2
    im = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = im.shape[:2]
    if und:  # equidistant fisheye with the nominal Aria focal length -> 90 degree pinhole
        f = F_640 * max(w, h) / 640
        K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
        fo = (UNDIST / 2) / np.tan(np.radians(FOV / 2))
        Ko = np.array([[fo, 0, UNDIST / 2], [0, fo, UNDIST / 2], [0, 0, 1.0]])
        m1, m2 = cv2.fisheye.initUndistortRectifyMap(K, np.zeros(4), np.eye(3), Ko, (UNDIST, UNDIST), cv2.CV_32FC1)
        return cv2.remap(im, m1, m2, cv2.INTER_CUBIC)
    k = MAX_SIDE / max(h, w)
    return cv2.resize(im, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA) if k < 1 else im


def frames_bench(video, rows, root):
    """Benchmark frames of one video, by pts."""
    import av
    import cv2
    out = {}
    with av.open(str(video)) as c:
        s = c.streams.video[0]
        s.codec_context.thread_count = 4
        if rows[0].get("keyframes_only"):
            s.codec_context.skip_frame = "NONKEY"
        for r in sorted(rows, key=lambda r: r["pts"] if r.get("pts") is not None else r["t"] / s.time_base):
            pts = r["pts"] if r.get("pts") is not None else int(r["t"] / s.time_base)
            c.seek(int(pts), stream=s, backward=True)
            fr = next((f for f in c.decode(s) if f.pts is not None and f.pts >= pts), None)
            if fr is None:
                out[r["image"]] = "no frame decoded"
                continue
            dst = root / r["image"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dst), hq(fr.to_ndarray(format="rgb24"), r["undistort"]), [cv2.IMWRITE_JPEG_QUALITY, 93])
            out[r["image"]] = "ok" if fr.pts == pts else f"pts {fr.pts} instead of {pts}"
    return out


def frames_train(video, rows, root):
    """Training frames of one video (640 px), with the recipe they were cut with."""
    out = {}
    if rows[0]["recipe"] == "seek":
        import av
        with av.open(str(video)) as c:
            s = c.streams.video[0]
            for r in rows:
                c.seek(int(r["t"] / s.time_base), stream=s)
                fr = next(c.decode(s), None)
                if fr is None:
                    out[r["image"]] = "no frame decoded"
                    continue
                im = fr.to_image()
                im.thumbnail((640, 640))
                dst = root / r["image"]
                dst.parent.mkdir(parents=True, exist_ok=True)
                im.save(dst, quality=85)
                out[r["image"]] = "ok"
        return out
    with tempfile.TemporaryDirectory() as tmp:
        cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-skip_frame", "nokey", "-i", str(video),
               "-vf", "fps=1,scale='min(640,iw)':-2", "-q:v", "3", "-frame_pts", "1", f"{tmp}/%06d.jpg"]
        p = subprocess.run(cmd, capture_output=True, text=True)
        for r in rows:
            src = Path(tmp) / f"{r['t']:06d}.jpg"
            if p.returncode or not src.exists():
                out[r["image"]] = "ffmpeg failed" if p.returncode else "frame not produced"
                continue
            dst = root / r["image"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst)
            out[r["image"]] = "ok"
    return out


def one_video(job):
    video, rows, root = job
    try:
        return (frames_train if "recipe" in rows[0] else frames_bench)(video, rows, Path(root))
    except Exception as e:  # noqa: BLE001
        return {r["image"]: f"error: {str(e)[-160:]}" for r in rows}


def paste(root, r):
    base = np.asarray(Image.open(root / r["base"]).convert("RGB")).copy()
    if "patch" in r:
        pa = np.asarray(Image.open(root / r["patch"]).convert("RGBA"))
        x0, y0, x1, y1 = r["box"]
        m = pa[..., 3] > 0
        base[y0:y1, x0:x1][m] = pa[..., :3][m]
    (root / r["image"]).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(base).save(root / r["image"])


def crop(root, r):
    (root / r["image"]).parent.mkdir(parents=True, exist_ok=True)
    with Image.open(root / r["of"]) as im:
        im.convert("RGB").crop(tuple(r["rect"])).save(root / r["image"], quality=95)


def check(root, r, max_dist):
    p = root / r["image"]
    if not p.exists():
        return None
    d = dist(phash(p), r["phash"])
    return d if d <= max_dist else -d


def rebuild(data, ego4d=None, egoexo4d=None, workers=8, max_dist=8, force=False, ffmpeg=None):
    root = Path(data)
    rows = [json.loads(l) for l in (root / "ego" / "pointers.jsonl").read_text().splitlines() if l.strip()]
    roots = {"Ego4D": ego4d, "Ego-Exo4D": egoexo4d}
    if ffmpeg:
        os.environ["PATH"] = str(Path(ffmpeg).resolve().parent) + os.pathsep + os.environ["PATH"]
    status, dists = {}, {}
    todo = [r for r in rows if r.get("how", "frame") == "frame"]
    if not force:  # keep frames already rebuilt and within the hash distance
        done = {r["image"] for r in todo if (check(root, r, max_dist) if (root / r["image"]).exists() else -1) >= 0}
        status.update({i: "ok (kept)" for i in done})
        todo = [r for r in todo if r["image"] not in done]
    jobs = collections.defaultdict(list)
    for r in todo:
        if not roots.get(r["dataset"]):
            status[r["image"]] = f"no --{r['dataset'].lower().replace('-', '')} root given"
            continue
        v = Path(roots[r["dataset"]]) / r["video"]
        if not v.exists():
            status[r["image"]] = f"video missing: {v}"
            continue
        jobs[(str(v), r.get("recipe"), r.get("keyframes_only"))].append(r)
    print(f"{len(rows)} pointers, {len(todo)} frames to decode from {len(jobs)} videos")
    with ProcessPoolExecutor(max(1, workers)) as ex:
        for i, res in enumerate(ex.map(one_video, [(Path(k[0]), v, str(root)) for k, v in jobs.items()]), 1):
            status.update(res)
            if i % 50 == 0 or i == len(jobs):
                print(f"  videos {i}/{len(jobs)}", flush=True)
    for how, fn, need in (("edit", paste, "base"), ("crop", crop, "of")):
        for r in rows:
            if r.get("how") != how:
                continue
            if not (root / r[need]).exists():
                status[r["image"]] = f"{need} missing"
                continue
            try:
                fn(root, r)
                status[r["image"]] = "ok"
            except Exception as e:  # noqa: BLE001
                status[r["image"]] = f"error: {str(e)[-160:]}"
    for r in rows:
        if r.get("how") == "withheld":
            status[r["image"]] = "withheld"
            continue
        s = status.get(r["image"], "missing")
        if not s.startswith(("ok", "pts ")):
            continue
        d = check(root, r, max_dist)
        if d is None:
            status[r["image"]] = "missing"
            continue
        dists[r["image"]] = abs(d)
        if d < 0:
            status[r["image"]] = f"hash distance {-d} > {max_dist}"
        elif s.startswith("pts "):
            status[r["image"]] = "ok, nearby pts"
    for how, need in (("edit", "base"), ("crop", "of")):  # an edit or crop of a rejected image is rejected too
        for r in rows:
            if r.get("how") == how and status.get(r["image"], "").startswith("ok") \
                    and r[need] in status and not status[r[need]].startswith("ok"):
                status[r["image"]] = f"{need} rejected"
    for r in rows:  # move every image that failed aside, so that no pair counts as ready with it
        q = root / r["image"]
        if not status.get(r["image"], "").startswith(("ok", "withheld")) and q.exists():
            os.replace(q, q.with_name(q.name + ".rejected"))
    report = {"max_dist": max_dist, "status": status, "dist": dists}
    (root / "ego" / "rebuild_report.json").write_text(json.dumps(report, indent=0))
    summary(status, dists)
    return report


def summary(status, dists):
    c = collections.Counter(s.split(":")[0] for s in status.values())
    for k, n in c.most_common():
        print(f"  {n:6d}  {k}")
    if dists:
        d = np.array(list(dists.values()))
        print(f"  pHash distance: median {np.median(d):.0f}, 99th percentile {np.percentile(d, 99):.0f}, max {d.max()}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="props rebuild-ego", description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True, help="PROPS-Benchmark or PROPS-Train folder")
    ap.add_argument("--ego4d", help="Ego4D root that holds v2/clips/")
    ap.add_argument("--egoexo4d", help="Ego-Exo4D root that holds takes/")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-dist", type=int, default=8, help="largest accepted pHash Hamming distance (of 64 bits)")
    ap.add_argument("--ffmpeg", help="ffmpeg binary, if not in PATH (PROPS-Train keyframes_fps1 rows)")
    ap.add_argument("--force", action="store_true", help="decode again even if a matching image exists")
    a = ap.parse_args(argv)
    rebuild(a.data, a.ego4d, a.egoexo4d, a.workers, a.max_dist, a.force, a.ffmpeg)


if __name__ == "__main__":
    main()
