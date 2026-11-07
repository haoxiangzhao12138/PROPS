"""Load PROPS-Benchmark from a local copy of the Hugging Face dataset.

    from props.data import Bench
    b = Bench("data/PROPS-Benchmark")          # after `props download` (and `props rebuild-ego` for the Ego pairs)
    for pair in b.pairs("open"):                 # "all", "open", "phase_a" or "ready"
        for iid in pair["items"]:
            item = b.item(iid)                   # frames, scripts, steps to draw
            imgs = b.frames(item)                # PIL images, None where the model draws

Layout of the dataset folder:
    pairs.jsonl          one pair per line (subtype, group, gap, phase_a, cluster, items, sources, needs_ego, rebuild)
    items/<pair>_<A|B>.json   model input of one side
    judge/<pair>.json    scoring material of both sides (never shown to the model)
    images/              every image except Ego4D / Ego-Exo4D frames, named by sha1
    ego/pointers.jsonl   how to rebuild each Ego4D / Ego-Exo4D image from the official videos
    ego/patches/         RGBA patches with the edited pixels of the Ego looks
"""
import json
from functools import lru_cache
from pathlib import Path

from PIL import Image

REPO_BENCH = "CUDAOUTOFMEMORY/PROPS-Benchmark"
REPO_TRAIN = "CUDAOUTOFMEMORY/PROPS-Train"
SUBTYPES = {"Recall": ["Recall-Start", "Recall-Mid", "Recall-Far"],
            "Resist": ["Resist-Stale", "Resist-Intercut", "Resist-Lookalike"],
            "Reuse": ["Reuse-Own", "Reuse-Replaced"]}
REUSE = set(SUBTYPES["Reuse"])


def download(root, repo=REPO_BENCH, token=None, ego=True, keep_tars=False):
    """Download a PROPS dataset repo into `root`. The repos are private for now: pass a token with read access
    (or log in with `huggingface-cli login`). ego=False skips the Ego pointers and patches.
    PROPS-Train ships its images as images/part-NNN.tar; they are unpacked into `root` and, unless keep_tars,
    deleted one by one. A part that was unpacked leaves images/part-NNN.tar.done and is not downloaded again."""
    from huggingface_hub import snapshot_download
    root = Path(root)
    done = sorted(p.name[:-5] for p in (root / "images").glob("part-*.tar.done")) if root.exists() else []
    ignore = ([] if ego else ["ego/*"]) + [f"images/{n}" for n in done]
    snapshot_download(repo_id=repo, repo_type="dataset", local_dir=str(root), token=token,
                      ignore_patterns=ignore or None, max_workers=16)
    unpack(root, keep_tars)
    return root


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

    def pairs(self, subset="all"):
        """all: the 2,257 pairs; open: the 1,123 pairs without Ego4D / Ego-Exo4D content (no rebuild needed);
        phase_a: a fixed draw of 600 pairs (75 per subtype); ready: pairs on disk now."""
        if subset == "all":
            return list(self._pairs)
        if subset == "open":
            return [p for p in self._pairs if p["open"]]
        if subset == "phase_a":
            return [p for p in self._pairs if p["phase_a"]]
        if subset == "ready":
            return [p for p in self._pairs if self.ready(p)]
        raise ValueError(subset)

    def frames(self, item):
        """PIL images of the item frames, None at the steps the model draws."""
        return [Image.open(self.root / f["image"]).convert("RGB") if f["image"] else None for f in item["frames"]]


def output_path(out_dir, iid, j):
    """Where a run stores the frame it drew for step j of item iid: <out_dir>/<iid>/<j>.png."""
    return Path(out_dir) / iid / f"{j}.png"
