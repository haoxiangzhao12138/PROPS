# PROPS-Benchmark: data format

## Layout

```
data/PROPS-Benchmark/
  pairs.jsonl                one pair per line
  items/<pair>_<A|B>.json    the model input of one side
  judge/<pair>.json          scoring material of both sides (never shown to the model)
  images/                    every released image, named by sha1
  ego/pointers.jsonl         how to rebuild each Ego4D / Ego-Exo4D image (docs/ego.md)
  ego/patches/               RGBA patches with the edited pixels of the Ego looks
```

## Loading

```python
from props.data import Bench

b = Bench("data/PROPS-Benchmark")
for pair in b.pairs("open"):              # full, phase_a or open
    for iid in pair["items"]:             # the two twins, <pair>_A and <pair>_B
        item = b.item(iid)
        frames = b.frames(item)           # PIL images; None at the steps the model draws
        scripts = [s["script"] for s in item["steps"]]
        print(iid, pair["subtype"], item["draw"], scripts[item["query"] - 1][:60])
    break
```

Only `pairs.jsonl`, `items/` and `judge/` are needed to read the metadata or to recompute a board from stored
scores; the images are needed to generate.

## Pairs

```json
{"pair": "p0001", "subtype": "Recall-Start", "group": "Recall", "gap": 3, "phase_a": true,
 "cluster": "c0067", "items": ["p0001_A", "p0001_B"], "sources": ["Ego4D"], "needs_ego": true,
 "open": false, "rebuild": "full"}
```

| field | meaning |
|---|---|
| `subtype`, `group` | one of the eight subtypes below and its group |
| `gap` | steps between the last frame that shows the look and the query |
| `phase_a` | one of the 600 pairs of phase A, a fixed random draw of 75 per subtype |
| `cluster` | the group of source videos that the bootstrap resamples (329 clusters) |
| `sources` | the video datasets the frames come from |
| `needs_ego`, `open` | whether the pair has an Ego4D or Ego-Exo4D frame; `open` pairs have none |
| `rebuild` | what the release gives for this pair (docs/ego.md) |

| group | subtype | pairs | where the valid look is |
|---|---|---:|---|
| Recall | Start | 623 | in frame 0 |
| | Mid | 225 | in a middle keyframe |
| | Far | 221 | in frame 0, with frames cut in from other chains so that it lies up to eight shots back |
| Resist | Stale | 187 | in a later script that says the prop was replaced; frame 0 shows the other twin's look |
| | Intercut | 187 | in the chain, which alternates with another video that often shows an object of the same kind |
| | Lookalike | 205 | in the chain; an inserted frame shows an object of the same kind in the other twin's look |
| Reuse | Own | 187 | in the evidence frame the model drew itself, from a script that names the look once |
| | Replaced | 422 | in the replacement the model drew itself, from a script that names the new look |

## Items

```json
{"id": "p0001_A", "pair": "p0001", "side": "A", "subtype": "Recall-Start",
 "frames": [{"image": "images/48/48...png", "size": [1280, 720]}, ..., {"image": null, "size": [1280, 720]}],
 "steps":  [{"time": "moments later", "script": "Cut to the kitchen counter by the window. ..."}, ...],
 "draw": [3], "keep_own": [], "query": 3}
```

`frames[0]` is frame 0 and `frames[k]` the keyframe of step k; `image` is null where the model draws. `steps[k-1]`
holds the time and the shot script of step k. The model draws the steps in `draw`, and the last of them is the
`query`. In Reuse items the first drawn step is the evidence step: it is listed in `keep_own` and stays in the
history as the model drew it. A drawn step may carry a `draw_script`, the script the model receives for that step.

## Judge files

Per side: `exec`, the yes or no execution questions with their answers; `gt`, the true target frames by step;
and either `crop`, the reference crop of the prop, or for Reuse pairs `evidence_step` with `look` and `where`, the
description the grounder uses to find the prop in the model's own evidence frame. `kind` names the prop. The model
never sees these files.
