"""Load PROPS-Benchmark from a local copy of the Hugging Face dataset.

    from props.data import Bench
    b = Bench("data/PROPS-Benchmark")          # after `props download` (and `props rebuild-ego` for the Ego pairs)
    for pair in b.pairs("phase_a"):              # "full" or "phase_a"
        for iid in pair["items"]:
            item = b.item(iid)                   # frames, scripts, steps to draw
            imgs = b.frames(item)                # PIL images, None where the model draws

Layout of the dataset folder:
    pairs.jsonl          one pair per line (subtype, group, gap, phase_a, cluster, items, sources, needs_ego)
    items/<pair>_<A|B>.json   model input of one side
    judge/<pair>.json    scoring material of both sides (never shown to the model)
    images/              every image except Ego4D / Ego-Exo4D frames, named by sha1
    ego/pointers.jsonl   how to rebuild each Ego4D / Ego-Exo4D image from the official videos
    ego/patches/         RGBA patches with the edited pixels of the Ego looks
"""
import hashlib, json, os
from functools import lru_cache
from pathlib import Path

from PIL import Image

REPO_BENCH = "CUDAOUTOFMEMORY/PROPS-Benchmark"
REPO_TRAIN = "CUDAOUTOFMEMORY/PROPS-Train"
SUBTYPES = {"Recall": ["Recall-Start", "Recall-Mid", "Recall-Far"],
            "Resist": ["Resist-Stale", "Resist-Intercut", "Resist-Lookalike"],
            "Reuse": ["Reuse-Own", "Reuse-Replaced"]}
REUSE = set(SUBTYPES["Reuse"])
SPLITS = {"full": lambda p: True,
          "phase_a": lambda p: p["phase_a"]}


def download(root, repo=REPO_BENCH, token=None, ego=True, keep_tars=False, revision=None):
    """Download a PROPS dataset repo into `root`. ego=False skips the Ego pointers and patches.
    props_download.json records the repo, the revision asked for and the commit it resolved to.
    PROPS-Train ships its images as images/part-NNN.tar; they are unpacked into `root` and, unless keep_tars,
    deleted one by one. A part that was unpacked leaves images/part-NNN.tar.done and is not downloaded again."""
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")  # slow links time out at the default 10 s
    from huggingface_hub import snapshot_download
    root = Path(root)
    done = sorted(p.name[:-5] for p in (root / "images").glob("part-*.tar.done")) if root.exists() else []
    ignore = ([] if ego else ["ego/*"]) + [f"images/{n}" for n in done]
    from huggingface_hub import HfApi
    sha = HfApi().repo_info(repo, repo_type="dataset", revision=revision, token=token).sha  # pin the commit first
    fetch(lambda: snapshot_download(repo_id=repo, repo_type="dataset", local_dir=str(root), token=token, revision=sha,
                                    ignore_patterns=ignore or None, max_workers=16),
          progress=lambda: sum(1 for _ in root.rglob("*")) if root.exists() else 0)
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "props_download.json", {"repo": repo, "revision": revision or "main", "commit": sha})
    unpack(root, keep_tars)
    return root


def fetch(call, tries=6, progress=None):
    """Run a Hugging Face download, again after a dropped connection (files already on disk are kept, so each try
    resumes). Access errors are not retried. Hugging Face limits the number of files per five minutes, so when it
    says so (HTTP 429) the download waits five minutes and goes on, as often as it takes. It gives up after `tries`
    dropped connections in a row with no new file on disk (`progress` counts them)."""
    import time
    k, last = 0, progress() if progress else 0
    while True:
        try:
            return call()
        except Exception as e:  # noqa: BLE001
            code = getattr(getattr(e, "response", None), "status_code", None)
            if code == 429:
                print("[props download] Hugging Face asks to wait (too many files in five minutes); going on in 5 min",
                      flush=True)
                time.sleep(300)
                continue
            now = progress() if progress else 0
            k, last = (0 if now > last else k) + 1, max(now, last)
            if code in (401, 403, 404) or k == tries:
                raise
            print(f"[props download] interrupted ({type(e).__name__}: {str(e)[:120]}); retrying in {30 * k} s",
                  flush=True)
            time.sleep(30 * k)


def unpack(root, keep_tars=False):
    """Unpack images/part-NNN.tar of PROPS-Train into `root` (the paths inside start with images/)."""
    import tarfile
    root = Path(root)
    parts = [p for p in sorted((root / "images").glob("part-*.tar"))
             if not (root / "images" / f"{p.name}.done").exists()]
    for i, tar in enumerate(parts):
        with tarfile.open(tar) as tf:
            members = tf.getmembers()
            for m in members:
                if not m.isfile() or m.name.startswith("/") or ".." in Path(m.name).parts:
                    raise ValueError(f"unexpected entry {m.name!r} in {tar.name}")
            kw = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
            tf.extractall(root, members=members, **kw)
        (root / "images" / f"{tar.name}.done").write_text(f"{len(members)}\n")
        if not keep_tars:
            tar.unlink()
        print(f"unpacked {tar.name} ({i + 1}/{len(parts)}, {len(members)} images)", flush=True)


def _jsonl(p):
    with open(p) as f:
        return [json.loads(line) for line in f if line.strip()]


class Bench:
    def __init__(self, root):
        self.root = Path(root)
        if not (self.root / "pairs.jsonl").exists():
            raise FileNotFoundError(f"no PROPS-Benchmark at {self.root} (pairs.jsonl not found); run `props download` "
                                    "or pass --bench")
        self._pairs = _jsonl(self.root / "pairs.jsonl")
        self.by_id = {p["pair"]: p for p in self._pairs}

    def __len__(self):
        return len(self._pairs)

    def path(self, rel):
        return self.root / rel

    def has(self, rel):
        return rel is None or (self.root / rel).exists()

    @lru_cache(maxsize=None)
    def item(self, iid):
        return json.loads((self.root / "items" / f"{iid}.json").read_text())

    @lru_cache(maxsize=None)
    def judge(self, pid):
        return json.loads((self.root / "judge" / f"{pid}.json").read_text())

    def ready(self, pair):
        """True when every model input and evidence crop of the pair is on disk (Ego pairs: after rebuild-ego)."""
        for iid in pair["items"]:
            if not all(self.has(f["image"]) for f in self.item(iid)["frames"]):
                return False
        return all(self.has(s.get("crop")) for s in self.judge(pair["pair"])["sides"].values())

    def pairs(self, split="full"):
        """The two splits of the leaderboard (see SPLITS):
        full      all 2,257 pairs
        phase_a   600 pairs, a fixed draw of 75 per subtype, for systems too costly to run on all pairs"""
        if split not in SPLITS:
            raise ValueError(f"unknown split {split!r}; one of {', '.join(SPLITS)}")
        return [p for p in self._pairs if SPLITS[split](p)]

    def frames(self, item):
        """PIL images of the item frames, None at the steps the model draws."""
        return [Image.open(self.root / f["image"]).convert("RGB") if f["image"] else None for f in item["frames"]]


def write_json(path, obj, **kw):
    """Write JSON through a temporary file in the same folder, so an interrupted run never leaves half a file."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, **kw))
    os.replace(tmp, path)


def read_json(path, default=None):
    """JSON of a file, or `default` if it does not exist or cannot be parsed (then it is reported)."""
    path = Path(path)
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except ValueError:
        print(f"[props] warning: {path} is damaged and is ignored", flush=True)
        return default


_SHA = {}  # absolute path -> [size, mtime_ns, ctime_ns, inode, sha1, time of hashing]


def sha1(path):
    """sha1 of a file's content, or None if it does not exist. A file whose size, modification time, status change
    time and inode are all unchanged is not read again (writing a file, or resetting its times with os.utime,
    changes the status change time). As in git, a hash taken less than 2 s after the file last changed is not
    trusted, because file systems with coarse timestamps can give a quick rewrite the same times. The cache can be
    kept between runs with load_hashes / save_hashes."""
    import time
    p = Path(path)
    try:
        st = p.stat()
    except FileNotFoundError:
        return None
    k, stamp = str(p.resolve()), [st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_ino]
    e = _SHA.get(k)
    if e and len(e) == 6 and e[:4] == stamp and e[5] - max(stamp[1], stamp[2]) / 1e9 > 2:
        return e[4]
    h = hashlib.sha1(p.read_bytes()).hexdigest()
    _SHA[k] = stamp + [h, time.time()]
    return h


def load_hashes(out_dir):
    _SHA.update(read_json(Path(out_dir) / ".props_sha1.json", {}))


def save_hashes(out_dir):
    if Path(out_dir).exists():
        root = str(Path(out_dir).resolve())
        write_json(Path(out_dir) / ".props_sha1.json", {k: v for k, v in _SHA.items() if k.startswith(root)})


def data_commit(bench):
    """The commit of the dataset that `props download` fetched, or None."""
    return (read_json(Path(bench) / "props_download.json", {}) or {}).get("commit")


def note_run(out_dir, bench, **kw):
    """Record in <out>/props_run.json the dataset commit the frames are drawn from (and what else is given)."""
    f = Path(out_dir) / "props_run.json"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    write_json(f, {**read_json(f, {}), "data_commit": data_commit(bench), **kw})


def output_path(out_dir, iid, j):
    """Where a run stores the frame it drew for step j of item iid: <out_dir>/<iid>/<j>.png."""
    return Path(out_dir) / iid / f"{j}.png"
