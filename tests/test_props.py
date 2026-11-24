"""Offline tests of the scoring and export logic with fake servers. Run with `python -m pytest tests` or
`python tests/test_props.py`. No GPU, network or dataset download is needed."""
import asyncio, base64, csv, io, json, sys, tempfile
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import props.score as scorer  # noqa: E402
from props import prompts as T  # noqa: E402
from props.board import board  # noqa: E402
from props.cli import main  # noqa: E402
from props.judge import content  # noqa: E402
from props.manifest import check, export  # noqa: E402


def make_bench(root, subtype="Recall-Start"):
    """One pair. Recall: draw step 1 (the query). Reuse: draw the evidence step 1, keep it, draw the query 2."""
    reuse = subtype.startswith("Reuse")
    bench, out = root / "bench", root / "out"
    for p in (bench / "items", bench / "judge", bench / "images", out):
        p.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), "white").save(bench / "images" / "f0.png")
    pair = {"pair": "p1", "subtype": subtype, "items": ["p1_A", "p1_B"], "cluster": "c1", "phase_a": True,
            "needs_ego": False}
    (bench / "pairs.jsonl").write_text(json.dumps(pair) + "\n")
    sides = {}
    for v in "AB":
        n = 2 if reuse else 1
        item = {"id": f"p1_{v}", "frames": [{"image": "images/f0.png", "size": [32, 32]}]
                + [{"image": None, "size": [32, 32]}] * n,
                "steps": [{"time": "moments later", "script": f"step {k}"} for k in range(1, n + 1)],
                "draw": list(range(1, n + 1)), "keep_own": [1] if reuse else [], "query": n}
        (bench / "items" / f"p1_{v}.json").write_text(json.dumps(item))
        Image.new("RGB", (16, 16), "red" if v == "A" else "blue").save(bench / "images" / f"{v}.jpg")
        sides[v] = {"item": f"p1_{v}", "exec": [{"q": "Is the door open?", "a": "yes"}],
                    **({"evidence_step": 1, "look": "", "where": ""} if reuse else {"crop": f"images/{v}.jpg"})}
    (bench / "judge" / "p1.json").write_text(json.dumps({"kind": "cup", "sides": sides}))
    return bench, out


def colour(path_or_url):
    if isinstance(path_or_url, tuple):  # (drawn frame, frame size)
        path_or_url = path_or_url[0]
    im = Image.open(path_or_url) if isinstance(path_or_url, Path) else \
        Image.open(io.BytesIO(base64.b64decode(path_or_url.split(",", 1)[1])))
    r, g, b = im.convert("RGB").getpixel((im.width // 2, im.height // 2))
    return "red" if r > 150 and b < 100 else "blue" if b > 150 and r < 100 else "other"


class Fake:
    """Answers like a perfect judge: the output shows the look whose crop has its colour."""
    calls, fail = 0, False

    def __init__(self, *a, **k):
        self.ident = k.get("ident")

    async def ask(self, system, parts, max_side, **k):
        type(self).calls += 1
        if type(self).fail:
            raise RuntimeError("server down")
        if system == T.CHOICE_SYS:
            c1, c2, img = colour(parts[1]), colour(parts[2]), colour(parts[3])
            return {"answer": "1" if img == c1 else "2" if img == c2 else "neither"}
        if system == T.JSYS:
            return {"answers": ["yes"]}
        if system == T.GROUND_SYS:
            return {"found": True, "bbox": [0, 0, 1000, 1000]}
        return {"answer": "yes"}


def draw(out, iid, j, col):
    (out / iid).mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), col).save(out / iid / f"{j}.png")


def setup(fake=Fake):
    scorer.Client = fake
    scorer.identity = lambda urls, model: (f"{model}@test", True)  # no real server to ask for its weights
    fake.calls, fake.fail = 0, False


def test_kind_check_reads_each_new_crop():
    """The first box is wrong (red half), the second right (blue half): the kind check must see the second crop."""
    class Vision:
        def __init__(self):
            self.boxes, self.seen = 0, []

        async def ask(self, system, parts, max_side, **k):
            if system == T.GROUND_SYS:
                self.boxes += 1
                return {"found": True, "bbox": [0, 0, 490, 1000] if self.boxes == 1 else [510, 0, 1000, 1000]}
            c = colour(content(parts, max_side)[1]["image_url"]["url"])
            self.seen.append(c)
            return {"answer": "yes" if c == "blue" else "no"}
    with tempfile.TemporaryDirectory() as t:
        d = Path(t) / "p1_A"
        d.mkdir()
        im = Image.new("RGB", (64, 32), "red")
        im.paste("blue", (32, 0, 64, 32))
        im.save(d / "1.png")
        j, g = Vision(), Vision()
        r = asyncio.run(scorer.own_crop(j, g, t, "cup", {"item": "p1_A", "evidence_step": 1}, (64, 32)))
        assert r is not None and colour(r) == "blue" and j.seen == ["red", "blue"]


def test_rescore_when_output_or_judge_changes():
    setup()
    with tempfile.TemporaryDirectory() as t:
        bench, out = make_bench(Path(t))
        draw(out, "p1_A", 1, "red"), draw(out, "p1_B", 1, "blue")
        done, err = scorer.score(bench, out, "u", "u")
        assert not err and done["p1"]["A"]["ok"] and done["p1"]["B"]["ok"]
        Fake.calls = 0
        scorer.score(bench, out, "u", "u")
        assert Fake.calls == 0  # nothing changed, nothing asked
        draw(out, "p1_A", 1, "blue")
        done, _ = scorer.score(bench, out, "u", "u")
        assert Fake.calls > 0 and not done["p1"]["A"]["ok"]
        Fake.calls = 0
        scorer.score(bench, out, "u", "u", judge_model="other")
        assert Fake.calls > 0


def test_reuse_missing_evidence_then_supplied():
    setup()
    with tempfile.TemporaryDirectory() as t:
        bench, out = make_bench(Path(t), "Reuse-Own")
        draw(out, "p1_A", 2, "red"), draw(out, "p1_B", 2, "blue")
        done, _ = scorer.score(bench, out, "u", "u")
        assert done["p1"]["A"]["missing"] and not done["p1"]["A"]["ok"]
        draw(out, "p1_A", 1, "red"), draw(out, "p1_B", 1, "blue")
        done, _ = scorer.score(bench, out, "u", "u")
        assert done["p1"]["A"]["ok"] and done["p1"]["B"]["ok"]


def test_server_errors_are_not_scores():
    setup()
    with tempfile.TemporaryDirectory() as t:
        bench, out = make_bench(Path(t))
        draw(out, "p1_A", 1, "red"), draw(out, "p1_B", 1, "blue")
        Fake.fail = True
        done, err = scorer.score(bench, out, "u", "u")
        assert "p1" in err and "p1" not in done
        assert board(bench, out)["scored"] == 0
        args = ["--bench", str(bench), "--outputs", str(out), "--judge", "u", "--grounder", "u", "--split", "full"]
        assert main(["score"] + args) == 1
        assert main(["evaluate"] + args) == 1 and not (out / "props_board.json").exists()
        Fake.fail = False
        assert main(["evaluate"] + args) == 0 and (out / "props_board.json").exists()


def test_export_two_rounds_and_stale_query():
    with tempfile.TemporaryDirectory() as t:
        bench, out = make_bench(Path(t), "Reuse-Own")
        ready, waiting, blocked, done = export(bench, out, "full")
        assert (ready, waiting, blocked, done) == (2, 2, 0, 0)
        reqs = [json.loads(x) for x in (out / "requests.jsonl").read_text().splitlines()]
        assert {r["step"] for r in reqs} == {1} and len(reqs[0]["history"]) == 1
        draw(out, "p1_A", 1, "red"), draw(out, "p1_B", 1, "blue")
        assert export(bench, out, "full")[:2] == (2, 0)
        reqs = [json.loads(x) for x in (out / "requests.jsonl").read_text().splitlines()]
        assert {r["step"] for r in reqs} == {2}
        assert reqs[0]["history"][1]["image"] == str((out / "p1_A" / "1.png").resolve())
        draw(out, "p1_A", 2, "red"), draw(out, "p1_B", 2, "blue")
        assert check(bench, out, "full", verbose=False) == ([], [], [])
        draw(out, "p1_A", 1, "green")  # evidence drawn again after its query
        assert check(bench, out, "full", verbose=False) == ([], [("p1_A", 2)], [])


def test_collect_from_csv():
    with tempfile.TemporaryDirectory() as t:
        bench, out = make_bench(Path(t))
        mine = Path(t) / "mine"
        mine.mkdir()
        Image.new("RGB", (64, 64), "red").save(mine / "a.jpg")
        with open(Path(t) / "frames.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerows([["item", "step", "path"], ["p1_A", 1, mine / "a.jpg"], ["p1_B", 1, mine / "nope.jpg"],
                         ["p9_A", 1, mine / "a.jpg"]])
        rc = main(["collect", "--bench", str(bench), "--out", str(out), "--table", str(Path(t) / "frames.csv"),
                   "--split", "full"])
        assert rc == 1 and Image.open(out / "p1_A" / "1.png").size == (32, 32)
        assert check(bench, out, "full", verbose=False)[0] == [("p1_B", 1)]


def test_collect_and_hand_placed_reuse_frames():
    """Frames copied by `props collect` keep the link from query to evidence; frames placed by hand have none."""
    with tempfile.TemporaryDirectory() as t:
        bench, out = make_bench(Path(t), "Reuse-Own")
        mine = Path(t) / "mine"
        mine.mkdir()
        rows = [["item", "step", "path"]]
        for v, col in (("A", "red"), ("B", "blue")):
            for j in (1, 2):
                Image.new("RGB", (32, 32), col).save(mine / f"{v}{j}.png")
                rows.append([f"p1_{v}", j, mine / f"{v}{j}.png"])
        with open(Path(t) / "frames.csv", "w", newline="") as fh:
            csv.writer(fh).writerows(rows)
        args = ["--bench", str(bench), "--split", "full"]
        assert main(["collect", "--out", str(out), "--table", str(Path(t) / "frames.csv")] + args) == 0
        assert check(bench, out, "full", verbose=False) == ([], [], [])
        Image.new("RGB", (32, 32), "green").save(mine / "A1.png")  # evidence collected again, query not
        with open(Path(t) / "again.csv", "w", newline="") as fh:
            csv.writer(fh).writerows([["item", "step", "path"], ["p1_A", 1, mine / "A1.png"]])
        main(["collect", "--out", str(out), "--table", str(Path(t) / "again.csv")] + args)
        assert check(bench, out, "full", verbose=False)[1] == [("p1_A", 2)]
        (out / "p1_B" / "props_inputs.json").unlink()  # as if placed by hand
        assert check(bench, out, "full", verbose=False)[2] == [("p1_B", 2)]
        assert main(["check", "--outputs", str(out), "--adopt"] + args) == 1  # p1_A is still stale
        assert check(bench, out, "full", verbose=False)[1:] == ([("p1_A", 2)], [])


def test_generate_records_evidence():
    with tempfile.TemporaryDirectory() as t:
        bench, out = make_bench(Path(t), "Reuse-Own")
        args = ["generate", "--bench", str(bench), "--out", str(out), "--split", "full",
                "--adapter", "props.adapters.baselines", "--", "--kind", "copyprev"]
        assert main(args) == 0
        assert check(bench, out, "full", verbose=False) == ([], [], [])
        draw(out, "p1_B", 1, "green")
        assert check(bench, out, "full", verbose=False)[1] == [("p1_B", 2)]


def test_own_crop_recut_when_models_change():
    setup()
    with tempfile.TemporaryDirectory() as t:
        bench, out = make_bench(Path(t), "Reuse-Own")
        for v, col in (("A", "red"), ("B", "blue")):
            draw(out, f"p1_{v}", 1, col), draw(out, f"p1_{v}", 2, col)
        scorer.score(bench, out, "u", "u")
        Fake.calls = 0
        scorer.score(bench, out, "u", "u", grounder_model="other")
        grounds = json.loads((out / "p1_A" / "own_e.json").read_text())
        assert Fake.calls > 0 and grounds["models"][1] == "other@test"


def test_server_identity_fails_closed():
    import props.judge as J
    old = J.served, J._get
    weights = {"a": {"judge": "w1", "judge@r1": "w1"}, "b": {"judge": "w2"}, "d": {"other": "w1"}}
    J.served, J._get = (lambda u: weights.get(u)), (lambda u, timeout=10: {"version": "0.30.0"})
    try:
        assert J.identity("a,a", "judge") == ("judge@w1@r1@vllm-0.30.0", True)
        for urls, err in (("a,b", ValueError), ("a,c", ConnectionError), ("d", ValueError)):
            try:  # different weights, a server that does not answer, a server with another model name
                J.identity(urls, "judge")
                assert False, urls
            except err:
                pass
    finally:
        J.served, J._get = old


def test_hash_cache_sees_a_rewrite_with_the_old_times():
    import os
    from props.data import sha1
    with tempfile.TemporaryDirectory() as t:
        f = Path(t) / "x.bin"
        f.write_bytes(b"aaaa")
        st = f.stat()
        h = sha1(f)
        f.write_bytes(b"bbbb")
        os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns))
        assert sha1(f) != h


def test_deleted_own_crop_is_cut_again():
    setup()
    with tempfile.TemporaryDirectory() as t:
        bench, out = make_bench(Path(t), "Reuse-Own")
        for v, col in (("A", "red"), ("B", "blue")):
            draw(out, f"p1_{v}", 1, col), draw(out, f"p1_{v}", 2, col)
        scorer.score(bench, out, "u", "u")
        Fake.calls = 0
        scorer.score(bench, out, "u", "u")
        assert Fake.calls == 0  # nothing changed
        (out / "p1_A" / "own_e.jpg").unlink()
        done, _ = scorer.score(bench, out, "u", "u")
        assert Fake.calls > 0 and (out / "p1_A" / "own_e.jpg").exists() and done["p1"]["A"]["ok"]


def test_score_command_checks_provenance():
    setup()
    with tempfile.TemporaryDirectory() as t:
        bench, out = make_bench(Path(t), "Reuse-Own")
        for v, col in (("A", "red"), ("B", "blue")):
            draw(out, f"p1_{v}", 1, col), draw(out, f"p1_{v}", 2, col)  # placed by hand, no record
        args = ["--bench", str(bench), "--outputs", str(out), "--judge", "u", "--grounder", "u", "--split", "full"]
        assert main(["score"] + args) == 2 and Fake.calls == 0
        assert main(["score"] + args + ["--allow-unverified"]) == 0 and Fake.calls > 0
        assert main(["evaluate"] + args + ["--allow-unverified"]) == 0
        assert json.loads((out / "props_board.json").read_text())["meta"]["complete"] is False


def test_rescore_when_reference_crop_changes():
    setup()
    with tempfile.TemporaryDirectory() as t:
        bench, out = make_bench(Path(t))
        draw(out, "p1_A", 1, "red"), draw(out, "p1_B", 1, "blue")
        scorer.score(bench, out, "u", "u")
        Fake.calls = 0
        Image.new("RGB", (16, 16), "blue").save(bench / "images" / "A.jpg")
        done, _ = scorer.score(bench, out, "u", "u")
        assert Fake.calls > 0 and not done["p1"]["A"]["follow"]


def test_collect_query_without_its_evidence_is_unknown():
    with tempfile.TemporaryDirectory() as t:
        bench, out = make_bench(Path(t), "Reuse-Own")
        mine = Path(t) / "mine"
        mine.mkdir()
        draw(out, "p1_A", 1, "red"), draw(out, "p1_B", 1, "blue")  # evidence placed by hand
        rows = [["item", "step", "path"]]
        for v in "AB":
            Image.new("RGB", (32, 32), "red").save(mine / f"{v}2.png")
            rows.append([f"p1_{v}", 2, mine / f"{v}2.png"])
        with open(Path(t) / "q.csv", "w", newline="") as fh:
            csv.writer(fh).writerows(rows)
        main(["collect", "--bench", str(bench), "--split", "full", "--out", str(out), "--table", str(Path(t) / "q.csv")])
        assert check(bench, out, "full", verbose=False)[2] == [("p1_A", 2), ("p1_B", 2)]


def test_adopted_queries_are_scored_again():
    """A pair scored with --allow-unverified keeps that mark until it is scored again from recorded frames."""
    setup()
    with tempfile.TemporaryDirectory() as t:
        bench, out = make_bench(Path(t), "Reuse-Own")
        for v, col in (("A", "red"), ("B", "blue")):
            draw(out, f"p1_{v}", 1, col), draw(out, f"p1_{v}", 2, col)
        args = ["--bench", str(bench), "--outputs", str(out), "--judge", "u", "--grounder", "u", "--split", "full"]
        assert main(["score"] + args + ["--allow-unverified"]) == 0
        assert json.loads((out / "props_scores.json").read_text())["p1"]["A"]["unverified"]
        assert main(["check", "--bench", str(bench), "--outputs", str(out), "--split", "full", "--adopt"]) == 0
        Fake.calls = 0
        assert main(["score"] + args) == 0 and Fake.calls > 0
        assert "unverified" not in json.loads((out / "props_scores.json").read_text())["p1"]["A"]


def test_missing_frames_block_score_and_old_board_is_removed():
    setup()
    with tempfile.TemporaryDirectory() as t:
        bench, out = make_bench(Path(t))
        draw(out, "p1_A", 1, "red"), draw(out, "p1_B", 1, "blue")
        args = ["--bench", str(bench), "--outputs", str(out), "--judge", "u", "--grounder", "u", "--split", "full"]
        assert main(["evaluate"] + args) == 0 and (out / "props_board.json").exists()
        (out / "p1_B" / "1.png").unlink()
        assert main(["score"] + args) == 2
        assert main(["evaluate"] + args) == 2 and not (out / "props_board.json").exists()
        assert main(["evaluate"] + args + ["--allow-missing"]) == 0
        assert json.loads((out / "props_board.json").read_text())["board"]["missing_sides"] == 1


def test_frames_of_another_size_are_scored_at_the_frame_size():
    from props.judge import content
    with tempfile.TemporaryDirectory() as t:
        f = Path(t) / "x.png"
        Image.new("RGB", (64, 20), "red").save(f)
        url = content([(f, (32, 32))], 1024)[0]["image_url"]["url"]
        im = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
        assert im.size == (32, 32)


def test_ablation_runs_are_kept_apart():
    with tempfile.TemporaryDirectory() as t:
        bench, out = make_bench(Path(t))
        gen = ["generate", "--bench", str(bench), "--out", str(out), "--split", "full",
               "--adapter", "props.adapters.baselines"]
        assert main(gen + ["--ablate", "nof0", "--", "--kind", "copyprev"]) == 0
        assert main(gen + ["--", "--kind", "copyprev"]) == 2
        row = board(bench, out, "full")
        from props.board import record
        meta = json.loads(record(bench, out, row).read_text())["meta"]
        assert not meta["complete"] and any("ablate" in w for w in meta["incomplete_because"])


def test_unknown_revision_reuses_nothing():
    setup()
    scorer.identity = lambda urls, model: (f"{model}@root@None@vllm-None", False)
    with tempfile.TemporaryDirectory() as t:
        bench, out = make_bench(Path(t))
        draw(out, "p1_A", 1, "red"), draw(out, "p1_B", 1, "blue")
        scorer.score(bench, out, "u", "u")
        Fake.calls = 0
        scorer.score(bench, out, "u", "u")
        assert Fake.calls > 0 and not (out / ".judge_cache").exists()


def test_frames_from_another_data_version_or_rejected_inputs_are_not_scored():
    setup()
    with tempfile.TemporaryDirectory() as t:
        bench, out = make_bench(Path(t))
        (bench / "props_download.json").write_text(json.dumps({"commit": "c1"}))
        export(bench, out, "full")
        draw(out, "p1_A", 1, "red"), draw(out, "p1_B", 1, "blue")
        args = ["--bench", str(bench), "--outputs", str(out), "--judge", "u", "--grounder", "u", "--split", "full"]
        assert main(["evaluate"] + args) == 0
        (bench / "props_download.json").write_text(json.dumps({"commit": "c2"}))
        assert main(["score"] + args) == 2 and not (out / "props_board.json").exists()
        (bench / "props_download.json").write_text(json.dumps({"commit": "c1"}))
        (bench / "images" / "f0.png").rename(bench / "images" / "f0.png.rejected")
        assert main(["evaluate"] + args) == 2


def test_download_waits_out_the_rate_limit():
    import time
    from props.data import fetch

    class Limited(Exception):
        response = type("R", (), {"status_code": 429})()
    calls, slept, sleep = [], [], time.sleep
    time.sleep = slept.append
    try:
        assert fetch(lambda: calls.append(1) or (len(calls) > 8 and "done") or (_ for _ in ()).throw(Limited())) == "done"
    finally:
        time.sleep = sleep
    assert len(calls) == 9 and slept == [300] * 8


def test_download_goes_on_while_files_arrive():
    import time
    from props.data import fetch
    calls, sleep = [], time.sleep
    time.sleep = lambda s: None
    def call():
        calls.append(1)
        if len(calls) < 10:
            raise TimeoutError("read timed out")
        return "done"
    try:
        assert fetch(call, tries=3, progress=lambda: len(calls)) == "done"  # a new file before every failure
        calls.clear()
        try:
            fetch(call, tries=3, progress=lambda: 0)  # nothing arrives: give up after three failures
            assert False
        except TimeoutError:
            assert len(calls) == 3
    finally:
        time.sleep = sleep


def test_download_unpacks_the_tars_of_every_folder():
    import tarfile
    from props.data import Bench, unpack
    with tempfile.TemporaryDirectory() as t:
        bench, _ = make_bench(Path(t))
        hub = Path(t) / "hub"
        (hub / "ego").mkdir(parents=True)
        (bench / "ego" / "patches").mkdir(parents=True)
        Image.new("RGBA", (4, 4)).save(bench / "ego" / "patches" / "x.png")
        for folder, content in (("items", "items"), ("judge", "judge"), ("images", "images"), ("ego", "ego/patches")):
            (hub / folder).mkdir(exist_ok=True)
            with tarfile.open(hub / folder / "part-000.tar", "w") as tf:
                for p in sorted((bench / content).rglob("*")):
                    tf.add(p, arcname=p.relative_to(bench).as_posix())
        (hub / "pairs.jsonl").write_text((bench / "pairs.jsonl").read_text())
        try:
            Bench(hub)
            assert False
        except FileNotFoundError:
            pass
        unpack(hub)
        assert not list(hub.glob("*/part-*.tar")) and len(list(hub.glob("*/part-*.tar.done"))) == 4
        assert (hub / "ego" / "patches" / "x.png").exists() and (hub / "images" / "f0.png").exists()
        b = Bench(hub)
        assert [p["pair"] for p in b.pairs("full")] == ["p1"] and b.ready(b.pairs("full")[0])


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
    print("all tests passed")
