# Drawing the frames

A run folder holds one PNG per drawn step: `runs/<name>/<item>/<step>.png`. Each item has the steps listed in its
`draw` field (docs/dataset.md). Recall and Resist items draw one step, the query. Reuse items draw two: the evidence
step first, then the query, whose history holds the evidence frame the model drew. The split decides which items
are drawn: `full` or `phase_a`. `props export` prints how many frames a split needs.

There are three ways to fill a run folder. All of them end with `props check --outputs runs/<name>`, which lists the
frames that are still missing.

## With your own pipeline: `props export`

```bash
props export --out runs/mymodel [--split full]
```

writes `runs/mymodel/requests.jsonl`, one line per frame that can be drawn now. Draw each request and save the frame
as a PNG at its `output` path. A frame of another size is resized to `size` when it is scored, as `props collect`
and `props generate` do when they save it. Then run `props export` again.
The first round lists every Recall and Resist query and the evidence steps of Reuse; the second lists the Reuse
queries, whose history now holds your evidence frames. When a round lists 0 frames, the run is complete.

### requests.jsonl

```json
{"item": "p0002_A", "step": 4, "query": true,
 "output": "/abs/path/runs/mymodel/p0002_A/4.png",
 "history": [{"image": "/abs/path/data/PROPS-Benchmark/images/ee/eeee...jpg", "time": "start", "script": null},
             {"image": "/abs/path/data/PROPS-Benchmark/images/02/02aa...jpg", "time": "moments later", "script": "Cut to the wall ..."},
             {"image": "/abs/path/runs/mymodel/p0002_A/2.png", "time": "moments later", "script": "Back at the living room ..."},
             {"image": "/abs/path/data/PROPS-Benchmark/images/99/9936...jpg", "time": "a few minutes later", "script": "Cut to the wall ..."}],
 "time": "a few seconds later", "script": "Back at the living room by the wall-mounted TV ...",
 "prompt": "You are drawing the next keyframe of a storyboard made from real footage. ...",
 "size": [1280, 720], "seed": 1307857510, "own": {"2": "8ad74f77..."}}
```

| field | meaning |
|---|---|
| `item`, `step` | the item and the step to draw (1-based); `query` is true for the step that is scored |
| `output` | where to save the frame |
| `history` | frame 0 and the keyframes before `step`, in order, each with the time and the script of its step; frame 0 has none |
| `time`, `script` | the time and the script of the step to draw |
| `prompt` | the instruction with every earlier script and the script to draw, for models that take one prompt and a list of images |
| `size`, `seed` | the frame size, and a seed fixed per item and step |
| `own` | sha1 of the model's own frames in `history` (Reuse queries); it is also written to `<item>/props_inputs.json` |

Give the model the whole history in its native input format, up to the number of images it accepts. An interleaved
model should see frame 0, then each keyframe after its script, and end with the script to draw.

### Reuse queries and their evidence frames

A Reuse query must be drawn from the evidence frame that is on disk. All three ways below write the sha1 of the
evidence frames each query was drawn from to `<item>/props_inputs.json`, and `props check` and `props evaluate`
stop when a query was drawn from an evidence frame that has changed since, or has no such record. If you draw an
evidence frame again, delete its query and draw it again. If you placed frames in the run folder by hand and are
sure each query was drawn from the evidence frame now on disk, `props check --adopt` writes the missing records.

## Frames saved under other names: `props collect`

Write a CSV with the columns `item,step,path`, one row per drawn frame:

```
item,step,path
p0001_A,3,/data/mymodel/out/p0001_A_3.jpg
p0002_A,2,/data/mymodel/out/p0002_A_2.jpg
p0002_A,4,/data/mymodel/out/p0002_A_4.jpg
```

```bash
props collect --out runs/mymodel --table frames.csv
```

copies every listed frame into the run folder as a PNG at the frame size, and lists the rows it skipped (an unknown
item or step, a file that does not exist or cannot be read). `scripts/evaluate.sh RUN --table frames.csv` runs this
step first. A Reuse query is recorded as drawn from its evidence frame only when both are listed in the same table;
a query listed without its evidence frame stays without a record, and `props check` reports it.

## With an adapter: `props generate`

An adapter is a Python class:

```python
class Model:
    def __init__(self, args):     # args: the command-line arguments after --
        ...
    def generate(self, ctx):      # one step of one item -> PIL.Image
        ...
```

`ctx` holds the same inputs as a request, as Python objects:

| key | content |
|---|---|
| `item_id`, `j`, `K` | the item, the step to draw and the number of steps |
| `history` | PIL images of frame 0 to keyframe j-1; `frame0` and `prev` are its first and last entries |
| `scripts`, `dt_text` | the scripts of steps 1 to j and their times; `script` is the one to draw |
| `prompt`, `size`, `seed` | as in `requests.jsonl` |

Copy [props/adapters/template.py](../props/adapters/template.py) to start.
[props/adapters/qwen_edit.py](../props/adapters/qwen_edit.py) is the adapter of Qwen-Image-Edit-2511 and
[props/adapters/baselines.py](../props/adapters/baselines.py) the copy baselines.

```bash
# smoke test without a model: copy the previous frame on 20 items
props generate --adapter props.adapters.baselines --out runs/copyprev --limit 20 -- --kind copyprev
# one process per GPU
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i props generate --adapter my_model.py --out runs/mymodel --shard $i/8 -- --path /models/x &
done; wait
```

Everything after `--` goes to the adapter. The runner walks the steps of each item in order, gives the true frame
at every step outside `draw`, and keeps the evidence frame of Reuse items in the history as the model drew it.
Frames that exist are not drawn again, so an interrupted run resumes. A step that raises an error leaves no file and
the runner moves on to the next item; its side fails when scored, and the exit code is 1.

`--ablate nof0|prevonly|textonly|shuffle` changes the shown images and keeps the scripts: frame 0 grey, every image
but the previous frame grey, every image grey, or the history in a shuffled order. These are the test-time
interventions of the paper; they are analyses, not leaderboard runs. The run folder records the ablation in
`props_run.json`, `props generate` refuses to mix ablations in one folder, and the board of such a run is marked
`complete: false`.
