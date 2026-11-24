# PROPS: Can Visual Generators Keep the Current Look of a Prop?

Paper (arXiv, coming soon) ·
[Project page](https://haoxiangzhao12138.github.io/PROPS/) ·
[PROPS-Benchmark](https://huggingface.co/datasets/CUDAOUTOFMEMORY/PROPS-Benchmark) ·
[PROPS-Train](https://huggingface.co/datasets/CUDAOUTOFMEMORY/PROPS-Train)

Many generation tasks draw the next image from a visual history: storyboards, illustrated stories, multi-turn
editing. A request often says what happens next but not how an earlier object looks, so the generator has to take
the current look of that object from the history. PROPS measures whether it does. Each of its 2,257 pairs has two
twin histories that end on the same request but show one prop in two different looks. A pair passes only if both
outputs show their own look and carry out the request.

This repository downloads the data, gets the frames of your generator, scores them with two open models and prints
the leaderboard row. The release holds code and data only: PROPS-Benchmark, PROPS-Train and the score files of the
leaderboard. It holds no model weights. The evaluated models come from their own repositories, and the two scoring
models are downloaded from Hugging Face at pinned revisions. The `main` branch holds the code and the leaderboard,
and the `gh-pages` branch holds the built project page.

1. [Quick start](#quick-start)
2. [Evaluate your model](#evaluate-your-model)
3. [Scoring models](#scoring-models)
4. [Splits](#splits)
5. [Leaderboard](#leaderboard)
6. [Documentation](#documentation)
7. [Citation](#citation)

## Quick start

```bash
git clone https://github.com/haoxiangzhao12138/PROPS && cd PROPS
bash scripts/setup.sh                 # props in .venv, PROPS-Benchmark in data/ (3.2 GB)
source .venv/bin/activate
props board --outputs leaderboard/runs/sensenova_full --ci
```

The last command recomputes the SenseNova-U1.5 row of the paper (Overall 26.7) from its stored scores. It needs no
GPU and checks that the install and the download work. PROPS-Benchmark comes as a few tars, which `props download`
unpacks; behind an HTTP proxy, `export HF_HUB_DISABLE_XET=1` speeds up the download.

## Evaluate your model

### 1. Draw the frames

Every frame goes to `runs/<name>/<item>/<step>.png`. Most sides need one frame, the query. A Reuse side needs two:
the model first draws an evidence step, and that frame stays in the history of the query. Choose one of two ways.

**Your own pipeline.** Use this when the model runs in its own environment, on another machine or behind an API.

```bash
props export --out runs/mymodel      # writes runs/mymodel/requests.jsonl
```

Each line of `requests.jsonl` is one frame to draw. It holds the history images in order with their times and
scripts, the script of the new step, a ready-made prompt, the frame size, a seed, and `output`, the path to save the
frame at. Draw the requests and save each frame at its `output`. A Reuse query can be drawn only after its
evidence frame exists, so the first export lists the evidence steps and not their queries. Run `props export` again
after drawing them; it then lists the queries with your evidence frames in their history.
[docs/generation.md](docs/generation.md#requestsjsonl) describes every field.

If your frames are already saved under other names, list them in a CSV with the columns `item,step,path` and copy
them into the run folder:

```bash
props collect --out runs/mymodel --table frames.csv
```

**An adapter.** A Python class whose `generate(ctx)` returns a PIL image. `props generate` calls it in the right
order and handles Reuse itself. Start from [props/adapters/template.py](props/adapters/template.py).

```bash
props generate --adapter my_model.py --out runs/mymodel -- --path /models/my-model
```

Either way, `props check --outputs runs/mymodel` lists the frames that are still missing. It also stops on a
Reuse query that was drawn from an evidence frame that has changed since
([docs/generation.md](docs/generation.md#reuse-queries-and-their-evidence-frames)).

### 2. Score

```bash
bash scripts/setup.sh --judge                       # once: vLLM and the two scoring models (118 GB)
bash scripts/evaluate.sh runs/mymodel --gpus 0,1    # judge on GPU 0, grounder on GPU 1
```

`evaluate.sh` starts the two vLLM servers, waits until they answer, scores every pair, prints the board, writes
`runs/mymodel/props_board.json` and stops the servers. If the servers already run somewhere else:

```bash
props evaluate --outputs runs/mymodel --judge http://HOST:9000/v1 --grounder http://HOST:8000/v1
```

Nothing is scored while frames are missing, and no board is written if a request to a server fails. Run the same
command again after fixing the problem; pairs that were scored are kept. A frame that is drawn again is scored
again on the next run. `props_board.json` says `complete: true` when the run can enter the leaderboard, and lists
the reasons otherwise ([docs/scoring.md](docs/scoring.md#boards-for-the-leaderboard)).

### 3. Report

Report the board of the `full` split, or of `phase_a` for systems that are too costly to run on every pair. To add
a row to the leaderboard, open an issue with `props_board.json`, a short description of the system and a link to
the run folder (the PNG frames and `props_scores.json`). `props_board.json` records the code commit, the dataset
commit, the scoring protocol and prompts, and the weights and vLLM version of every scoring server.

## Scoring models

| role | model | revision | size | what it does |
|---|---|---|---:|---|
| judge | [google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it) | `842da37` | 62.6 GB | says which of the two looks a frame shows (Follow) and answers the execution questions (Exec) |
| grounder | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | `1d4bf0f` | 55.6 GB | finds the prop in the evidence frames your model drew (Reuse only) |

Neither is gated. `setup.sh --judge` downloads them at these revisions into
`models/` and installs vLLM 0.30.0 in `.venv-vllm`, apart from `props` and from your generator. vLLM needs Python
3.10 to 3.14 and an NVIDIA driver of release 580 or newer; [docs/scoring.md](docs/scoring.md#servers) explains what
to do with an older driver. The reference scores were computed with these weights and this vLLM version, one 80 GB
GPU per server. `--gpus 0-7` starts four servers of each model, and `--tp 2` spreads each server over two GPUs for
smaller cards. On H200 GPUs, phase A takes about 12 minutes with the default two GPUs and 5 with eight, server start
included. [docs/scoring.md](docs/scoring.md) gives the scoring rules.

## Splits

| split | pairs | needs Ego4D | use |
|---|---:|---|---|
| `full` | 2,257 | yes | the leaderboard (default) |
| `phase_a` | 600 | yes | the leaderboard, for systems too costly to run on every pair; 75 pairs per subtype |

Ego4D and Ego-Exo4D frames are not in the download. It has pointers to them, and `props rebuild-ego`
rebuilds them from your own copy of those datasets ([docs/ego.md](docs/ego.md)).

## Leaderboard

Overall in %, the mean of the Recall, Resist and Reuse pass rates. Closed systems were run on phase A only.
[leaderboard/README.md](leaderboard/README.md) has every column and the 95% intervals, and
`python scripts/make_leaderboard.py` recomputes all numbers from the score files in `leaderboard/runs/`. The
weights of the four models fine-tuned on PROPS-Train are not released.
[docs/train.md](docs/train.md#fine-tuned-rows-of-the-leaderboard) gives how they were trained.

<!-- leaderboard:start -->
| System | Full | Phase A |
|---|---:|---:|
| *Unified multimodal models* | | |
| SenseNova-U1.5 | 26.7 | 29.8 |
| Emu3.5 | 15.9 | 15.6 |
| BAGEL | 4.0 | 5.0 |
| HiDream-O1 | 1.6 | 1.1 |
| LLaDA2-Uni | 1.4 | 0.9 |
| OmniGen2 | 0.7 | 0.6 |
| Lumina-DiMOO | 0.1 | 0.1 |
| Janus-Pro | 0.0 | 0.0 |
| Show-o2 | 0.0 | 0.0 |
| *Image generators and editors* | | |
| FLUX.3-klein | 30.4 | 29.0 |
| FLUX.2-klein | 3.1 | 3.0 |
| VINCIE | 2.9 | 4.1 |
| Qwen-Image-2.1 | 2.2 | 2.5 |
| FLUX.2-dev | 2.0 | 2.2 |
| FireRed-Edit | 1.5 | 1.9 |
| Qwen-Image-Edit | 0.8 | 1.0 |
| FLUX.1-Kontext | 0.0 | 0.0 |
| *Agents (open planner + editor)* | | |
| Qwen3.8 + Qwen-Image-Edit | 19.6 | 19.1 |
| Qwen3.8 + Qwen-Image-2.1 | 16.9 | 17.6 |
| *Keyframe-conditioned video models* | | |
| LTX-2.5 | 8.3 | 7.5 |
| Wan-VACE | 0.8 | 0.9 |
| StoryMem | 0.4 | 0.8 |
| HunyuanVideo-I2V | 0.1 | 0.2 |
| Wan-TI2V | 0.0 | 0.1 |
| Wan-I2V | 0.0 | 0.0 |
| *Closed image generators* | | |
| Nano Banana 2 | | 50.4 |
| Seedream 5 | | 45.7 |
| Nano Banana Pro | | 39.3 |
| *Agents with a closed planner or editor* | | |
| GPT-5.6 + GPT Image 2.5 | | 71.0 |
| Qwen3.8-Max + Qwen-Image-Edit | | 60.9 |
| Gemini-3.1-Pro + Qwen-Image-Edit | | 60.2 |
| Kimi-K3 + Qwen-Image-Edit | | 56.7 |
| GPT-5.6 + Qwen-Image-Edit | | 55.7 |
| Gemini-3.1-Pro + Qwen-Image-2.1 | | 55.2 |
| Kimi-K3 (no images) + Qwen-Image-Edit | | 42.2 |
| Qwen3.8 + GPT Image 2.5 | | 29.3 |
| *Agents with a smaller open planner or another open editor* | | |
| Qwen3.8 + FLUX.3-klein | | 20.2 |
| Qwen3.5-9B + FLUX.3-klein | | 19.7 |
| Qwen3.5-9B + Qwen-Image-Edit | | 19.0 |
| Qwen3.5-9B + FLUX.3-klein, no upsampler | | 18.9 |
| *Fine-tuned on PROPS-Train* | | |
| SenseNova-U1.5 + PROPS-Train | 47.0 | 49.4 |
| BAGEL + PROPS-Train | 14.9 | 14.9 |
| Qwen-Image-2.1 + PROPS-Train (LoRA) | 11.2 | 11.3 |
| Qwen-Image-Edit + PROPS-Train (LoRA) | 10.8 | 8.7 |
<!-- leaderboard:end -->

## Documentation

| file | content |
|---|---|
| [docs/dataset.md](docs/dataset.md) | folder layout, fields of pairs, items and judge files, a loading example |
| [docs/generation.md](docs/generation.md) | adapters, `requests.jsonl`, `props collect`, ablations |
| [docs/scoring.md](docs/scoring.md) | how a pair is scored, servers, caching and rescoring, exit codes |
| [docs/ego.md](docs/ego.md) | rebuilding the Ego4D and Ego-Exo4D frames |
| [docs/train.md](docs/train.md) | PROPS-Train: format and use for fine-tuning |
| [CHANGELOG.md](CHANGELOG.md) | versions of the code and the data |
| [page/make_page.py](page/make_page.py) | builds the project page of the `gh-pages` branch from `leaderboard/board.json` |

## Citation

```bibtex
@misc{props2026,
  title  = {{PROPS}: Can Visual Generators Keep the Current Look of a Prop?},
  author = {Hao, Xiangzhao and Wang, Dianyi and Zhou, Keyan and Zhang, Yiqian and Zhang, Zefeng and Li, Bailin
            and Zhan, Kun and Yan, Junchi},
  year   = {2026}
}
```
