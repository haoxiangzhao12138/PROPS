"""Reference rows that need no model, for checking a setup end to end.

    props generate --adapter props.adapters.baselines -- --kind copyprev
kinds:
    copy0     frame 0 copied: keeps a look seen at frame 0 but does not carry out the script (fails Exec)
    copyprev  the previous frame copied (fails Exec)
    gt        the true frame (needs --bench)
On the full benchmark copy0 scores 0.1% and copyprev 0.0%.
"""
import argparse
import json
from pathlib import Path

from PIL import Image


class Model:
    def __init__(self, args):
        ap = argparse.ArgumentParser(prog="baselines")
        ap.add_argument("--kind", required=True, choices=["copy0", "copyprev", "gt"])
        ap.add_argument("--bench", help="dataset folder (kind gt)")
        self.a = ap.parse_args(args)

    def generate(self, ctx):
        if self.a.kind == "copy0":
            return ctx["frame0"]
        if self.a.kind == "copyprev":
            return ctx["prev"]
        root = Path(self.a.bench)
        pid, v = ctx["item_id"].rsplit("_", 1)
        gt = json.loads((root / "judge" / f"{pid}.json").read_text())["sides"][v]["gt"][str(ctx["j"])]
        return Image.open(root / gt)
