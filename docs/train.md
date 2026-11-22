# PROPS-Train

PROPS-Train keeps only steps whose target brings back an earlier state or carries scripted changes forward, so a
model cannot learn it by copying the previous frame. It has 80,250 chains with 293K steps over 132K distinct
images. They give 120,339 supervised targets in four tracks:

| track | targets | what the target needs |
|---|---:|---|
| `synth` | 56.1K | synthetic chains with two to five changes in time, space or cause; every later target keeps all earlier changes |
| `intercut` | 42.1K | a return to chain A after one to six shots of chain B; the model has to recover the state of A |
| `edit` | 14.0K | one scripted change per step, with everything else kept, including earlier additions |
| `real` | 8.1K | the camera returns to an earlier place in a real video; the target looks more like that earlier view than like the previous frame |

The training data and the benchmark share no source video group.

## Download

```bash
props download --out data --train-only   # data/PROPS-Train, about 24 GB (--train also fetches the benchmark)
```

The images come as `images/part-NNN.tar` (about 1 GB each). `props download --train` unpacks each part into the
dataset folder and deletes it, so the paths in `train.jsonl` point to files on disk. `--keep-tars` keeps the parts.
An interrupted download resumes and skips the parts already unpacked (they leave `images/part-NNN.tar.done`).

```
data/PROPS-Train/
  chains.jsonl          one chain per line: frames, steps with time and script, track, sources
  train.jsonl           120,339 supervised targets in LLaVA-style format
  val.jsonl             6,277 validation targets
  images/               124,486 released images (after unpacking)
  ego/pointers.jsonl    7,344 Ego4D / Ego-Exo4D keyframes of 790 real-revisit chains
```

## Format

A line of `train.jsonl`:

```json
{"id": "ed_sb_00000_1", "j": 1, "track": "edit", "role": "edit",
 "image": ["images/synth/sb_00000/0.jpg", "images/synth/sb_00000/1.jpg"],
 "conversations": [{"from": "human", "value": "... Frame 0: <image>\nNow draw ONLY Keyframe 1 (moments later): ..."},
                   {"from": "gpt", "value": "<image>"}],
 "needs_ego": false}
```

`image` lists the history frames in order and then the target. The `human` turn has one `<image>` per history
frame and ends with the script to draw; the `gpt` turn is the target. Most multi-image SFT code for LLaVA-style data
reads this directly; map the last image to the generation target of your model.

## Ego4D and Ego-Exo4D

Only real revisits use video frames. The 1,724 training targets and 92 validation targets that need an Ego4D or
Ego-Exo4D keyframe have `"needs_ego": true`. Rebuild them as in [docs/ego.md](ego.md)
(`props rebuild-ego --data data/PROPS-Train ...`), or drop them:

```python
import json
rows = [r for r in map(json.loads, open("data/PROPS-Train/train.jsonl")) if not r["needs_ego"]]
```

## Fine-tuned rows of the leaderboard

The weights of the four fine-tuned systems on the leaderboard are not released. They were trained on the shuffled
four-track mixture of `train.jsonl` with AdamW and a 100-step warmup, on eight GPUs per node:

| system | training | steps | learning rate | weights scored |
|---|---|---:|---|---|
| SenseNova-U1.5 + PROPS-Train | full, one node, at most 16,384 tokens per GPU, no weight decay | 2,000 | 2e-6, cosine to 10% after the warmup | EMA, decay 0.998 |
| BAGEL + PROPS-Train | full, two nodes, about 32K tokens per GPU | 2,000 | 1e-5, constant | EMA, decay 0.999 |
| Qwen-Image-2.1 + PROPS-Train (LoRA) | LoRA of rank 64 on the attention and MLP layers, one sample per GPU | 4,000 | 1e-4, constant | the adapter |
| Qwen-Image-Edit + PROPS-Train (LoRA) | as above, on a 32K subset of the targets with the same share of each track | 2,000 | 1e-4, constant | the adapter |

No run accumulates gradients.
