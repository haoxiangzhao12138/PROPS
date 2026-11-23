# Builds the PROPS project page, which is published from the gh-pages branch. The page is static: index.html holds
# the leaderboard data, and static/app.js draws the table and the charts in the browser, so GitHub Pages serves it as is.
#   leaderboard  leaderboard/board.json of this branch (scripts/make_leaderboard.py)
#   page code    page/style.css and page/app.js, copied to static/ on every build
#   abstract     <paper>/sections/abstract.tex, kept in static/abstract.html
#   figures      <paper>/figs/*.pdf, rendered with pdftoppm into static/
#   paper        <paper>/main.pdf, copied to static/PROPS_arxiv.pdf until the arXiv link exists
#   Chinese      page/zh.py and page/abstract_zh.html, shown by the language switch of the page
#
# Check out gh-pages next to this repository once, then rebuild after every change to the leaderboard:
#   git worktree add ../PROPS-site gh-pages
#   python3 page/make_page.py --out ../PROPS-site                    leaderboard and page code, keeps the figures
#   python3 page/make_page.py --out ../PROPS-site --paper DIR        also the abstract, figures and PDF from DIR
#   python3 -m http.server 8000 --bind 127.0.0.1 -d ../PROPS-site     preview
import argparse, html, json, re, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import zh as Z  # noqa: E402
LINKS = {  # placeholders until the release is public
    "paper": "static/PROPS_arxiv.pdf",
    "arxiv": "",  # the arXiv abstract page, once the id exists; until then the button says coming soon
    "code": "https://github.com/haoxiangzhao12138/PROPS",
    "bench": "https://huggingface.co/datasets/CUDAOUTOFMEMORY/PROPS-Benchmark",
    "train": "https://huggingface.co/datasets/CUDAOUTOFMEMORY/PROPS-Train",
}
AUTHORS = ("Xiangzhao Hao<sup>1,2*</sup>, Dianyi Wang<sup>2,3*</sup>, Keyan Zhou<sup>2,3</sup>, Yiqian Zhang<sup>4</sup>, "
           "Zefeng Zhang<sup>5</sup>, Bailin Li<sup>6</sup>, Kun Zhan<sup>6</sup>, Junchi Yan<sup>1,2&dagger;</sup>")
AFFIL = ("<sup>1</sup>Shanghai Jiao Tong University &nbsp; <sup>2</sup>Shanghai Innovation Institute &nbsp; "
         "<sup>3</sup>Fudan University &nbsp; <sup>4</sup>Hangzhou Dianzi University<br><sup>5</sup>Institute of Information "
         "Engineering, University of Chinese Academy of Sciences &nbsp; <sup>6</sup>Li Auto Inc.")
EMAILS = ["haoxiangzhao@sii.edu.cn", "dywang24@m.fudan.edu.cn", "jonaszhou01@gmail.com", "yiqian.zyq@gmail.com",
          "zhangzef1999@163.com", "libailin@lixiang.com", "zhankun@lixiang.com", "yanjunchi@sjtu.edu.cn"]  # author order
FIGS = {"overview": "ppt_rings", "method": "ppt_method", "pipeline": "ppt_pipeline", "train": "ppt_train",
        "agents": "ppt_agents", "qual": "ppt_qual"}
CAP = {
    "overview": "PROPS and PROPS-Train. Left: pairs by requirement and subtype, and the two looks of the prop in two "
                "pairs per requirement, with Reuse crops cut from frames drawn by SenseNova-U1.5. Right: chains by "
                "track, with intercut chains split by the number of shots cut away and synthetic chains by kind of "
                "change, and the keyframes of one chain per track with its supervised targets marked.",
    "method": "Twin histories. The twins share every script and differ only in how the prop looks, so one answer drawn "
              "from the scripts cannot match both. Bottom: the history of twin A in each of the eight subtypes.",
    "pipeline": "Building PROPS on one Recall-Start pair. One edit call changes the prop in the evidence and query "
                "frames, once per look; the judge must pick the right look on the true query frame of both twins; "
                "each pair is laid out as one subtype.",
    "train": "PROPS-Train: chains and supervised targets per track, synthetic chains per change class, and shots "
             "between leaving and returning to a storyline in intercut chains.",
    "agents": "One Resist-Stale history through two agent pipelines. The closed planner passes the evidence frame to "
              "the editor and names the look to keep.",
    "qual": "Examples from phase A.",
}
# The three requirements, their subtypes (short name, pairs in the benchmark, where the valid look is) and the
# leaderboard columns in order: Overall, each group mean followed by its subtypes, then the two diagnostics.
GROUPS = [
    ("Recall", "the valid look is several shots back", [
        ("Start", 623, "in frame 0"),
        ("Mid", 225, "in a middle keyframe"),
        ("Far", 221, "in frame 0, with frames cut in from other chains so that it lies up to eight shots back")]),
    ("Resist", "an outdated look, one from another place or one on a similar object competes with the valid one", [
        ("Stale", 187, "in a later script that says the prop was replaced; frame 0 shows the other twin's look"),
        ("Intercut", 187, "in the chain, which alternates with another video that often shows an object of the same kind"),
        ("Lookalike", 205, "in the chain; an inserted frame shows an object of the same kind in the other twin's look")]),
    ("Reuse", "the valid look was drawn by the model itself earlier in the history", [
        ("Own", 187, "in the evidence frame the model drew itself, from a script that names the look once"),
        ("Replaced", 422, "in the replacement the model drew itself, from a script that names the new look")]),
]
TRACKS = [("synth", 56.1, "synthetic chains with two to five changes in time, space or cause"),
          ("intercut", 42.1, "a return to chain A after one to six shots of chain B"),
          ("edit", 14.0, "one scripted change per step, with everything else kept"),
          ("real", 8.1, "the camera returns to an earlier place in a real video")]
CLOSED = {"closed", "closed_agent"}
SHAPE = {"umm": "o", "image": "s", "closed": "s", "agent": "d", "closed_agent": "d", "small_agent": "d", "video": "t"}
LABELS = ["GPT-5.6 + GPT Image 2.5", "Nano Banana 2", "SenseNova-U1.5 + PROPS-Train", "SenseNova-U1.5",
          "FLUX.3-klein"]  # named in the scatter plot, as in the paper; hover shows every other point
def metrics():
    m = [{"k": "Overall", "short": "Overall", "group": "", "axis": "Overall pass rate, % of pairs",
          "desc": "Mean of Recall, Resist and Reuse. A pair passes only if both twins show their own look of the prop "
                  "and carry out the request."}]
    for g, what, subs in GROUPS:
        names = [s for s, _, _ in subs]
        m.append({"k": g, "short": g, "group": g, "axis": f"{g}, % of pairs that pass", "axis_zh": Z.AXIS.format(k=g),
                  "desc": f"Mean of {', '.join(names[:-1])} and {names[-1]}: {what}.",
                  "desc_zh": Z.GROUP_DESC.format(subs="、".join(names[:-1]) + " 和 " + names[-1], what=Z.GROUP[g])})
        m += [{"k": f"{g}-{s}", "short": s, "group": g, "axis": f"{g}-{s}, % of pairs that pass",
               "axis_zh": Z.AXIS.format(k=f"{g}-{s}"), "desc": f"{n} pairs. The valid look is {where}.",
               "desc_zh": Z.SUB_DESC.format(n=n, where=Z.WHERE[f"{g}-{s}"])} for s, n, where in subs]
    m += [
        {"k": "Follow", "short": "Follow", "group": "Diagnostic", "axis": "Follow: own look shown, % of sides",
         "desc": "Share of sides whose output shows its own look of the prop. A diagnostic, not part of Overall."},
        {"k": "Exec", "short": "Exec", "group": "Diagnostic", "axis": "Exec: request carried out, % of sides",
         "desc": "Share of sides whose output carries out the scripted request. A diagnostic, not part of Overall."}]
    for x in m:
        if x["k"] in Z.METRIC:
            x["desc_zh"], x["axis_zh"] = Z.METRIC[x["k"]]
    return m


def abstract(paper):
    s = (paper / "sections/abstract.tex").read_text()
    s = s.split("\\begin{abstract}")[1].split("\\end{abstract}")[0]
    s = s.replace("\\bench", "PROPS").replace("\\train", "PROPS-Train").replace("~", " ")
    s = html.escape(" ".join(s.split()))
    return re.sub(r"\\emph\{([^}]*)\}", r"<em>\1</em>", s)


def assets(paper, out):
    st = out / "static"
    st.mkdir(parents=True, exist_ok=True)
    exe = shutil.which("pdftoppm") or str(Path.home() / ".local/bin/pdftoppm")
    for dst, src in FIGS.items():  # 320 dpi: twice the page width in pixels, sharp on high-density screens
        subprocess.run([exe, "-png", "-r", "320", "-singlefile", str(paper / f"figs/{src}.pdf"), str(st / dst)],
                       check=True)
    if (paper / "main.pdf").exists():
        shutil.copy(paper / "main.pdf", st / "PROPS_arxiv.pdf")
    (st / "abstract.html").write_text(abstract(paper) + "\n")


def data():
    """Rows of both splits for app.js. The reference rows (ground truth, copying a history frame) are kept apart:
    the scatter plot marks them, and the table leaves them out."""
    d = json.loads((ROOT / "leaderboard/board.json").read_text())
    fam, ms = dict(d["families"]), metrics()
    out = {"metrics": ms, "groups": [[g, [f"{g}-{s}" for s, _, _ in subs]] for g, _, subs in GROUPS],
           "families": {k: v for k, v in fam.items() if k != "reference"}, "pairs": d["pairs"],
           "splits": {}, "refs": {}, "labels": LABELS, "ui": Z.UI}
    for split in ("full", "phase_a"):
        rows, refs = [], []
        for r in d["splits"][split]:
            v = {m["k"]: r[m["k"]] for m in ms}
            if r["family"] == "reference":
                refs.append({"name": r["name"], "name_zh": Z.REF[r["name"]], "v": v})
                continue
            base = r["name"].split(" + PROPS-Train")[0] if r["family"] == "train" else None
            bfam = next((o["family"] for o in d["splits"][split] if o["name"] == base), "umm")
            rows.append({"name": r["name"], "family": r["family"], "kind": fam[r["family"]],
                         "kind_zh": Z.FAMILY[r["family"]], "v": v, "ci": r["ci"],
                         "unscored": r.get("unscored", 0), "closed": r["family"] in CLOSED,
                         "ours": r["family"] == "train", "base": base, "shape": SHAPE[bfam if base else r["family"]],
                         "color": "ours" if base else "closed" if r["family"] in CLOSED else "open"})
        out["splits"][split] = rows
        gt = next(r for r in refs if r["name"].startswith("Ground-truth"))
        out["refs"][split] = {"gt": gt, "ctrl": [r for r in refs if r is not gt]}
    return out


def get(D, split, name):
    return next(r for r in D["splits"][split] if r["name"] == name)["v"]["Overall"]


def findings(D):
    F, A = (lambda n: f"{get(D, 'full', n):.1f}"), (lambda n: f"{get(D, 'phase_a', n):.1f}")
    v = {"f3": F("FLUX.3-klein"), "sn": F("SenseNova-U1.5"), "top": A("GPT-5.6 + GPT Image 2.5"),
         "q38": A("Qwen3.8 + Qwen-Image-Edit"), "q38max": A("Qwen3.8-Max + Qwen-Image-Edit"),
         "g56": A("GPT-5.6 + Qwen-Image-Edit"), "g56gi": A("GPT-5.6 + GPT Image 2.5"),
         "snft": F("SenseNova-U1.5 + PROPS-Train"), "bg": F("BAGEL"), "bgft": F("BAGEL + PROPS-Train")}
    en = [
        ("Prop drift is widespread.",
         "The best open systems without fine-tuning, FLUX.3-klein and SenseNova-U1.5, pass {f3}% and {sn}% of the "
         "pairs on the full benchmark. The best system on phase A, GPT-5.6 with GPT Image 2.5, passes {top}%."),
        ("Carrying out the request is not keeping the look.",
         "Systems that carry out the request on most sides often draw the twin's look or neither look, so they pass far "
         "fewer pairs. In the plot of Overall against Exec, most systems carry out the request on far more sides than "
         "they pass pairs."),
        ("Agents are limited by both understanding and generation.",
         "With Qwen-Image-Edit as the editor, a closed planner raises the agent from {q38}% to {q38max}% on phase A, "
         "and GPT Image 2.5 raises the GPT-5.6 agent from {g56}% to {g56gi}%."),
        ("PROPS-Train reduces prop drift.",
         "Fine-tuning raises SenseNova-U1.5 from {sn}% to {snft}% and BAGEL from {bg}% to {bgft}% on the full "
         "benchmark, and LoRA fine-tuning helps Qwen-Image-2.1 and Qwen-Image-Edit as well."),
    ]
    return [(a, b.format(**v), za, zb.format(**v)) for (a, b), (za, zb) in zip(en, Z.FIND)]


def T(en, zh, tag="span", attrs=""):
    """The same text in both languages; the page shows one of them (body class zh or not)."""
    return f'<{tag} class="en"{attrs}>{en}</{tag}><{tag} class="zh"{attrs}>{zh}</{tag}>'


def fig(k):
    return (f'<figure><img src="static/{k}.png" alt="{k}" loading="lazy"><figcaption>'
            f'{T(html.escape(CAP[k]), html.escape(Z.CAP[k]))}</figcaption></figure>')


BIB = """@misc{props2026,
  title  = {{PROPS}: Can Visual Generators Keep the Current Look of a Prop?},
  author = {Hao, Xiangzhao and Wang, Dianyi and Zhou, Keyan and Zhang, Yiqian and Zhang, Zefeng and Li, Bailin
            and Zhan, Kun and Yan, Junchi},
  year   = {2026}
}"""

USAGE = """git clone https://github.com/haoxiangzhao12138/PROPS && cd PROPS
bash scripts/setup.sh --judge                       # props, PROPS-Benchmark, vLLM and the two scoring models
props export --out runs/mymodel                     # the frames to draw, with their inputs (or write an adapter)
bash scripts/evaluate.sh runs/mymodel --gpus 0,1    # serve the judge and the grounder, score, print the board"""
GCOL = {"Recall": "#4A6C94", "Resist": "#A8693F", "Reuse": "#557F62"}
NAV = [("overview", "Overview"), ("twins", "Twin histories"), ("train", "PROPS-Train"), ("leaderboard", "Leaderboard"),
       ("findings", "Findings"), ("evaluate", "Evaluate"), ("citation", "Citation")]
a = lambda s: html.escape(s, quote=True)


def cards(D):
    out = []
    for g, what, subs in GROUPS:
        b = "".join(f'<div class="sub" data-k="{g}-{s}"><b>{s}</b><span class="n">'
                    f'{T(f"{n} pairs", Z.PAIRS.format(n=n))}</span><br>'
                    f'{T("The valid look is " + html.escape(where) + ".", Z.VALID.format(where=Z.WHERE[g + "-" + s]))}'
                    f'<span class="best"></span></div>' for s, n, where in subs)
        out.append(f'<div class="gcard" style="--g:{GCOL[g]}"><h3>{g}</h3>'
                   f'{T(html.escape(what[0].upper() + what[1:]) + ".", Z.GROUP[g] + "。", "p")}{b}</div>')
    return '<div class="groups">' + "".join(out) + "</div>"


def tracks():
    tot, top = sum(v for _, v, _ in TRACKS), max(v for _, v, _ in TRACKS)
    return '<div class="tracks">' + "".join(
        f'<div class="trk" data-tip="{a(f"<b>{t}</b>: {what}<br>{v:.1f}K targets, {100 * v / tot:.0f}% of the training set")}" '
        f'data-tip-zh="{a(Z.TRACK_TIP.format(t=t, what=Z.TRACK[t], v=f"{v:.1f}", pct=f"{100 * v / tot:.0f}"))}">'
        f'<code>{t}</code><div><div class="bar" style="width:{100 * v / top:.1f}%"></div></div>'
        f'<span class="v">{v:.1f}K</span></div>' for t, v, what in TRACKS) + "</div>"


def code(s):
    return f'<div class="code"><button class="act">{T("Copy", "复制")}</button><pre>{html.escape(s)}</pre></div>'


EN = {
    "twins": """Each pair has two histories that end on the same request but show one prop in two different looks, so each
look is right on one side and wrong on the other. A pair passes only if both outputs show their own look and carry
out the request. The {pairs} pairs fall into three requirements with eight subtypes.""",
    "train": """80,250 training chains give 120,339 supervised targets in four tracks. Every target brings back an earlier
state or carries scripted changes forward, so it cannot be drawn by copying the previous frame. The training data
and the benchmark share no source video group.""",
    "full": "Full benchmark: {n} pairs, open systems",
    "phase_a": "Phase A: {n} pairs, every system",
    "lb_note": """Pass rate of pairs in %. A subtype score is the percentage of its pairs that pass, Recall, Resist and
Reuse are the means of their subtypes, and Overall is the mean of the three. The bar after Overall shows its 95%
bootstrap interval over source-video clusters. Follow is the rate of sides that show their own look and Exec the rate
of sides that carry out the request. † marks a closed model or planner; closed systems were run on phase A only.
Names in pink are fine-tuned on PROPS-Train, and the small number after their Overall is the gain over the base
model. Hover a cell for the details of a system; a click on a column sorts by it. To add a row, open an issue in
the code repository.""",
    "scatter": "Overall against Exec on phase A",
    "sc_note": """Every system on the {n} pairs of phase A, where open and closed systems share the same pairs. Most
systems carry out the request on far more sides than they pass pairs: they draw the twin's look or neither look. The
shaded regions enclose the open and the closed systems, the star marks the ground-truth frames and the crosses
copying frame 0 or the previous frame, and arrows lead from a base model to its fine-tune on PROPS-Train. Hover a
point for its scores.""",
    "eval_note": "The README of the code repository describes every step, the scoring models and the splits.",
}
NOTE_STYLE = ' style="color:var(--mid);font-size:.88em"'
LANG_INIT = ("<script>try{if((localStorage.getItem('props-lang')||(navigator.language||'').slice(0,2))==='zh')"
             "document.documentElement.classList.add('zh')}catch(e){}</script>")


def page(out):
    D = data()
    n = D["pairs"]
    e = html.escape
    tx = lambda k, tag="span", attrs="", **f: T(EN[k].format(**f), Z.TEXT[k].format(**f), tag, attrs)
    find = "".join(f"<p>{T(f'<b>{e(x)}</b> {e(y)}', f'<b>{e(zx)}</b> {e(zy)}')}</p>" for x, y, zx, zy in findings(D))
    btn = "".join(f'<a href="{LINKS[k]}">{T(t, Z.BTN[k])}</a>' if LINKS[k] else
                  f'<a class="off">{T(t + " (coming soon)", Z.BTN[k] + "（即将上线）")}</a>' for k, t in
                  (("paper", "Paper (PDF)"), ("arxiv", "arXiv"), ("code", "Code"), ("bench", "PROPS-Benchmark"),
                   ("train", "PROPS-Train")))
    nav = "".join(f'<a href="#{k}">{T(t, Z.NAV[k])}</a>' for k, t in NAV)
    mails = "".join(f'<a href="mailto:{m}">{m}</a>' for m in EMAILS)
    h2 = lambda k, en: f"<h2>{T(en, Z.HEAD[k])}</h2>"
    absx = T((out / "static/abstract.html").read_text().strip(), (HERE / "abstract_zh.html").read_text().strip(), "p")
    blob = json.dumps(D, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PROPS: Can Visual Generators Keep the Current Look of a Prop?</title>
<link rel="stylesheet" href="static/style.css">{LANG_INIT}</head><body>
<nav class="top"><div class="in"><a class="brand" href="#">PROPS</a>{nav}<button class="lang" id="lang">{T("中文", "EN")}</button></div></nav>
<header class="hero"><h1><span>PROPS</span>: Can Visual Generators Keep the Current Look of a Prop?</h1>
<div class="zh subtitle">{Z.TITLE}</div>
<div class="authors">{AUTHORS}</div><div class="affil">{AFFIL}</div>
<div class="emails">{mails}</div><div class="btns">{btn}</div></header><main>
<section id="overview">{fig("overview")}
{h2("abstract", "Abstract")}{absx}</section>
<section id="twins">{h2("twins", "Twin histories")}
{tx("twins", "p", pairs=f"{n['full']:,}")}
{cards(D)}{fig("method")}{fig("pipeline")}</section>
<section id="train">{h2("train", "PROPS-Train")}
{tx("train", "p")}
{tracks()}{fig("train")}</section>
<section id="leaderboard">{h2("leaderboard", "Leaderboard")}
<h3>{tx("full", n=f"{n['full']:,}")}</h3>
<div class="lb"><table class="board" id="lb_full"></table></div>
<h3>{tx("phase_a", n=n['phase_a'])}</h3>
<div class="lb"><table class="board" id="lb_phase_a"></table></div>
{tx("lb_note", "p", NOTE_STYLE)}
<h3>{tx("scatter")}</h3>
<div class="chart"><svg id="sc" viewBox="0 0 860 540"></svg></div>
{tx("sc_note", "p", NOTE_STYLE, n=n['phase_a'])}</section>
<section id="findings">{h2("findings", "Findings")}
<div class="row"><div class="find">{find}</div>{fig("agents")}</div>{fig("qual")}</section>
<section id="evaluate">{h2("evaluate", "Evaluate your model")}{code(USAGE)}
{tx("eval_note", "p", NOTE_STYLE)}</section>
<section id="citation">{h2("citation", "Citation")}{code(BIB)}</section></main>
<footer>PROPS: Can Visual Generators Keep the Current Look of a Prop?</footer>
<div id="tip"></div><div class="lightbox" id="lightbox"><img alt=""></div>
<script type="application/json" id="props-data">{blob}</script>
<script src="static/app.js"></script></body></html>"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build the project page of the gh-pages branch.")
    ap.add_argument("--out", type=Path, required=True, help="the gh-pages checkout")
    ap.add_argument("--paper", type=Path, help="paper sources with sections/, figs/ and main.pdf")
    a_ = ap.parse_args()
    if a_.paper:
        assets(a_.paper, a_.out)
    need = [f"static/{k}.png" for k in FIGS] + ["static/abstract.html"]
    if [p for p in need if not (a_.out / p).exists()]:
        sys.exit(f"{a_.out} has no figures or abstract yet; run once with --paper")
    for f in ("style.css", "app.js"):
        shutil.copy(HERE / f, a_.out / "static" / f)
    (a_.out / "index.html").write_text(page(a_.out))
    (a_.out / ".nojekyll").touch()
    print("wrote", a_.out / "index.html")
