# PROPS

Code for the PROPS benchmark and the PROPS-Train training set.

Many generation tasks draw the next image from a visual history: storyboards, illustrated stories, multi-turn
editing. A request often says what happens next but not how an earlier object looks, so the generator has to take
the current look of that object from the earlier images and the changes made since. PROPS measures whether it does.
Each of its 2,257 pairs has two twin histories that end on the same request but show one prop in two different
looks, so each look is right on one side and wrong on the other. A pair passes only if both outputs show their own
look and carry out the request.

This repository has everything needed to run and score a generator on PROPS:

| step | command | what it does |
|---|---|---|
| 1 | `props download` | fetch PROPS-Benchmark (and PROPS-Train) from Hugging Face |
| 2 | `props rebuild-ego` | rebuild the Ego4D and Ego-Exo4D frames from your own copy of those datasets |
| 3 | `props generate` | run a generator on every pair through a small adapter class |
| 4 | `props score` | ask the judge and the grounder about every output |
| 5 | `props board` | print the leaderboard row, optionally with bootstrap intervals |

## Install

```bash
git clone <this repo> props && cd props
pip install -e .            # loading, generation runner, scoring client, board
pip install -e ".[ego]"     # adds PyAV and OpenCV for props rebuild-ego
```

Python 3.9 or newer. Scoring talks to two OpenAI-compatible servers (see [Scoring](#scoring)); it needs no GPU on
the machine that runs `props score`.

## Download

The two datasets are private Hugging Face repos for now. With an account that has access, log in once with
`huggingface-cli login` or pass a read token:

```bash
props download --out data                    # data/PROPS-Benchmark  (about 3.0 GB)
props download --out data --train            # also data/PROPS-Train (about 24 GB)
HF_TOKEN=hf_xxx props download --out data    # token from the environment
```

| repo | content |
|---|---|
| `CUDAOUTOFMEMORY/PROPS-Benchmark` | 2,257 pairs (4,514 model inputs), judge files, images, Ego pointers |
| `CUDAOUTOFMEMORY/PROPS-Train` | 80,250 chains, 120,339 training targets and 6,277 validation targets in LLaVA-style JSONL, images, Ego pointers |

## Ego4D and Ego-Exo4D frames

Ego4D and Ego-Exo4D may only be shared with people who signed their licence agreements, so their frames are not in
the download. Every other source is shipped as images. For each Ego frame the repo has a pointer: the video, the
presentation timestamp of the keyframe, how it was resized or undistorted, the edited pixels of each look as an
RGBA patch with its box, and a perceptual hash. `props rebuild-ego` decodes the frames from your copy, pastes the
patches, cuts the crops and checks every result against its hash.

```bash
# official downloads (after signing the agreements)
ego4d  --output_directory ~/ego4d    --datasets clips --version v2   # -> ~/ego4d/v2/clips/<uid>.mp4
egoexo -o ~/egoexo4d --parts takes --views ego                     # -> ~/egoexo4d/takes/<take>/frame_aligned_videos/

props rebuild-ego --data data/PROPS-Benchmark --ego4d ~/ego4d --egoexo4d ~/egoexo4d --workers 16
props rebuild-ego --data data/PROPS-Train     --ego4d ~/ego4d --egoexo4d ~/egoexo4d --ffmpeg $(which ffmpeg)
```

Only the videos that PROPS uses are opened (532 for the benchmark, 767 for the training set). The report lands in
`ego/rebuild_report.json`.
Decoders differ a little between versions, so a rebuilt JPEG is close to the released one but not identical; the
check accepts a pHash Hamming distance of up to 8 of 64 bits (`--max-dist`). In our tests the distance was 0.

What each pair needs is in the `rebuild` field of `pairs.jsonl`:

| `rebuild` | pairs | meaning |
|---|---|---|
| `images` | 1,123 | no Ego frame; usable right after the download (the open subset, `--subset open`) |
| `full` | 905 | every image rebuilds from the pointers |
| `no_gt` | 203 | every model input and evidence crop rebuilds; the true target frames are relit Ego frames and are withheld. Scoring does not use them, only the `gt` baseline does |
| `partial` | 26 | Resist-Intercut pairs whose history contains a relit Ego frame, which no pointer can rebuild |

`props generate --subset ready` runs exactly the pairs whose inputs are on disk.

## Data format

```
PROPS-Benchmark/
  pairs.jsonl            one pair per line
  items/<pair>_<A|B>.json  the model input of one side
  judge/<pair>.json      scoring material of both sides (never shown to the model)
  images/                every released image, named by sha1
  ego/pointers.jsonl     one row per Ego4D / Ego-Exo4D image
  ego/patches/           RGBA patches with the edited pixels of the Ego looks
```

A line of `pairs.jsonl`:

```json
{"pair": "p0001", "subtype": "Recall-Start", "group": "Recall", "gap": 3, "phase_a": true, "dev": false,
 "cluster": "c0067", "items": ["p0001_A", "p0001_B"], "sources": ["Ego4D"], "needs_ego": true, "open": false,
 "rebuild": "full"}
```

`subtype` is one of Recall-Start, Recall-Mid, Recall-Far, Resist-Stale, Resist-Intercut, Resist-Lookalike, Reuse-Own
and Reuse-Replaced. `gap` is the number of steps between the evidence of the look and the query. `phase_a` marks a
fixed random draw of 600 pairs (75 per subtype) for systems that are too expensive to run on all pairs, `dev` the
160 pairs used to pick fine-tuning checkpoints, and `cluster` the group of source videos that the bootstrap
resamples.

An item (`items/p0001_A.json`) is what the generator sees:

```json
{"id": "p0001_A", "pair": "p0001", "side": "A", "subtype": "Recall-Start",
 "frames": [{"image": "images/48/48...png", "size": [1280, 720]}, ..., {"image": null, "size": [1280, 720]}],
 "steps":  [{"time": "moments later", "script": "Cut to the kitchen counter by the window. ..."}, ...],
 "draw": [3], "keep_own": [], "query": 3}
```

`frames[0]` is frame 0 and `frames[k]` the keyframe of step k. `steps[k-1]` holds the time and the shot script of
step k. The model draws the steps in `draw`, the last of which is the `query`. Reuse items also draw their evidence
step; it is listed in `keep_own` and stays in the history as the model drew it, so the model is later asked to keep a
look it drew itself. A drawn step may carry a `draw_script`, the script the model receives for that step.

The judge file of a pair holds, per side, the execution questions with their answers, the true target frames (`gt`)
and the reference crop of the prop (`crop`), or for Reuse pairs the evidence step and the description the grounder
uses to find the prop in the model's own frame.

## Generation

A generator is plugged in through an adapter: a Python class with `__init__(self, args)` and `generate(self, ctx)`,
which returns a PIL image. `ctx` is one step of one item:

| key | content |
|---|---|
| `item_id`, `j`, `K` | the item, the step to draw (1-based) and the number of steps |
| `history` | PIL images of frame 0 to keyframe j-1, as the runner holds them (given frames and the model's kept frames) |
| `frame0`, `prev` | `history[0]` and `history[-1]` |
| `scripts`, `dt_text` | the scripts of steps 1 to j and their times; `script` is the one to draw |
| `prompt` | the shared instruction with all scripts, for models that take one prompt and a list of images |
| `size`, `seed` | the frame size the output is resized to, and a seed fixed per item and step |

Give the model the whole history in its native input format, up to the number of images it accepts. Copy
`props/adapters/template.py` to start; `props/adapters/qwen_edit.py` is the adapter for Qwen-Image-Edit-2511.

```bash
# smoke test without a model: copy the previous frame
props generate --adapter props.adapters.baselines --out runs/copyprev --limit 20 -- --kind copyprev
# a real model, split over 8 GPUs
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i props generate --adapter props.adapters.qwen_edit --out runs/qwen_edit --shard $i/8 \
      -- --path Qwen/Qwen-Image-Edit-2511 &
done; wait
# your own model
props generate --adapter my_model.py --out runs/my_model -- --path /models/my-model
```

Everything after `--` goes to the adapter. The runner is teacher-forced: every step outside `draw` gets the true
frame, and a drawn step that is not the query enters the history as the model drew it. Outputs are written to
`runs/<name>/<item id>/<step>.png`; a rerun skips frames that exist, so an interrupted run resumes.
`--ablate nof0|prevonly|textonly|shuffle` changes the shown images and keeps the scripts: frame 0 grey, every image
but the previous frame grey, every image grey, or the history in shuffled order.

## Scoring

Scoring uses two open models served with vLLM: Gemma-4-31B-it as the judge and Qwen3.8-27B as the grounder, each
on one GPU with 80 GB or more. Several servers of each can be listed and the client spreads the requests over them.

```bash
bash scripts/serve_judge.sh judge 0-3       # http://HOST:9000/v1 ... :9003/v1
bash scripts/serve_judge.sh grounder 4-7    # http://HOST:8004/v1 ... :8007/v1
props score --outputs runs/qwen_edit \
    --judge http://HOST:9000/v1,http://HOST:9001/v1,http://HOST:9002/v1,http://HOST:9003/v1 \
    --grounder http://HOST:8004/v1,http://HOST:8005/v1,http://HOST:8006/v1,http://HOST:8007/v1
```

For every side the judge sees the reference crops of both looks and the output, and says which look the output
shows. It is asked twice with the crops in both orders, and the side follows only if both answers name its own look.
The judge then answers the yes or no execution questions of the side on the output, and all must be right. For
Reuse pairs the reference crops are cut from the frames the model drew at the evidence step, located by the
grounder. A side passes when it follows and executes, and a pair passes when both sides pass; a missing output fails
its side. The prompts are in `props/prompts.py`. Replies are cached in `runs/<name>/.judge_cache`, so rescoring is
free and an interrupted run resumes. Pairs scored while an output was missing are scored again on the next call.

## Leaderboard

```bash
props board --outputs runs/qwen_edit                    # full benchmark
props board --outputs runs/qwen_edit --subset phase_a   # the 600 pairs of phase A
props board --outputs runs/qwen_edit --ci               # with 95% intervals
```

A subtype score is the percentage of its pairs that pass. Recall, Resist and Reuse are the means of their subtypes,
and Overall is the mean of the three. Follow (sides that show their own look) and Exec (sides that carry out the
request) are diagnostic rates. Intervals come from a bootstrap over source-video clusters with 2,000 draws.

Report the full benchmark, or phase A for systems that are too expensive to run on all pairs. To add a row to the
leaderboard, send the run folder (the PNG outputs and `props_scores.json`) and a short description of the system to
the maintainers. Scores on the open subset are fine for development, but leaderboard rows use the full benchmark or
phase A.

## Example

`examples/sensenova-u1.5/props_scores.json` holds the scores of SenseNova-U1.5. The board gives Overall 26.7 on the
full benchmark and 29.8 on phase A:

```bash
props board --outputs examples/sensenova-u1.5 --ci
props board --outputs examples/sensenova-u1.5 --subset phase_a
```

The bootstrap resamples the 329 source-video clusters of the `cluster` field, which gives an interval of 24.1 to 29.4
for this row. The judge samples at temperature 0.2, so scoring the same outputs again from an empty cache can give
slightly different numbers.

## PROPS-Train

```
PROPS-Train/
  chains.jsonl          one chain per line: frames, steps, scripts and track
  train.jsonl           120,339 supervised targets in LLaVA-style format
  val.jsonl             6,277 validation targets
  images/part-NNN.tar   124,486 released images in tars of about 1 GB
  ego/pointers.jsonl    7,344 Ego4D / Ego-Exo4D keyframes (790 chains)
```

`props download --train` unpacks each tar into the dataset folder and deletes it, so the paths in `train.jsonl`
(`images/synth/...`, `images/frames/...`) point to files on disk. Pass `--keep-tars` to keep them; an interrupted
download resumes and skips the parts already unpacked.

A line of `train.jsonl` holds the images of the history and the target as `image` (the last one is the target),
the instruction as a `human` turn with one `<image>` per history frame and the target as the `gpt` turn. Its `track`
is `synth` (synthetic chains), `intercut` (intercut returns), `edit` (verified edits) or `real` (real revisits).
Only the real revisits use video frames; the 1,724 targets whose chain needs an Ego4D or Ego-Exo4D keyframe have
`"needs_ego": true` and can be dropped if those datasets are not at hand. Training without the real revisits scores
as well as with them.

## Licences

The code is released under the Apache License 2.0 (`LICENSE`). Every image keeps the licence of the video it comes
from; the per-source terms are in the dataset cards on Hugging Face. Ego4D and
Ego-Exo4D frames are not distributed and are used under their own licence agreements.
