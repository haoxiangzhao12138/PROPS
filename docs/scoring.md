# Scoring

## How a pair is scored

For every side the judge (Gemma-4-31B-it) sees the reference crops of both looks of the prop and the output frame,
and says which look the frame shows. It is asked twice, with the crops in both orders, and the side follows only if
both answers name its own look. The judge then answers the yes or no execution questions of the side on the output;
the side executes if every answer is right. A side passes when it follows and executes, and a pair passes when both
sides pass. A missing frame fails its side.

In Reuse pairs the look to keep is the one the model drew at the evidence step, so the reference crops are cut from
the model's own evidence frames. The grounder (Qwen3.8-27B) boxes the prop there; the judge and the grounder check
that the crop shows an object of the right kind, and a box far from the box on the true frame is dropped when both
say it is not one. If the prop is found on neither side, both sides fail.

A subtype's score is the percentage of its pairs that pass. Recall, Resist and Reuse are the means of their
subtypes, and Overall is the mean of the three, so the 623 Recall-Start pairs do not dominate. Follow and Exec are
rates over sides and serve as diagnostics. The 95% intervals come from a bootstrap over the 329 source-video
clusters (2,000 draws, seed 0). Every prompt is in [props/prompts.py](../props/prompts.py); changing one changes the
scores, so leaderboard runs use them as they are.

## Servers

`scripts/evaluate.sh` starts and stops the servers itself. To run them separately:

```bash
bash scripts/serve_judge.sh judge 0,1,2,3          # four judge servers on ports 9000-9003
bash scripts/serve_judge.sh grounder 4,5,6,7       # four grounder servers on ports 8004-8007
props evaluate --outputs runs/mymodel \
    --judge $(paste -sd, logs/vllm/judge.urls) --grounder $(paste -sd, logs/vllm/grounder.urls)
```

vLLM 0.30.0 from PyPI is built for CUDA 13 and needs an NVIDIA driver of release 580 or newer (`nvidia-smi`
shows it). With an older driver on a data-center GPU, install NVIDIA's CUDA 13 forward-compatibility package
(`cuda-compat-13-0`) and set `CUDA_COMPAT=/usr/local/cuda-13.0/compat` before `serve_judge.sh` or `evaluate.sh`;
`serve_judge.sh` checks the GPU first and says so when this is needed.

Each server needs one GPU with 80 GB or more. With smaller GPUs, `TP=2` (or `--tp 2` in `evaluate.sh`) spreads a
server over two. The servers listen on 127.0.0.1; set `HOST=0.0.0.0` to reach them from another machine, and
`ADVERTISE` to the name or IP that machine uses if it differs from `hostname -f`. Requests
are spread over every URL given, 16 in flight per model by default (`--conc`). The weights are those of
[scripts/versions.sh](../scripts/versions.sh); `JUDGE_MODEL` and `GROUNDER_MODEL` point to other copies.
`serve_judge.sh` serves each model under two names, `judge` and `judge@<revision>` (likewise for the grounder), so
that clients on other machines learn which revision of the weights a server runs.

Requests use temperature 0.2 with thinking off, and every reply is cached, so scoring the same frames again gives
the same numbers.

Time on one node with 8 H200 GPUs for `phase_a` (600 pairs):

| setting | servers ready | scoring |
|---|---:|---:|
| `--gpus 0-7` (four servers of each model) | about 2 min | about 3 min |
| `--gpus 0,1` (the default) | about 2 min | about 10 min |

The full benchmark has 3.8 times as many pairs and takes about that much longer.

## What is written

| file in the run folder | content |
|---|---|
| `props_scores.json` | the result of every scored pair: per side `follow`, `exec`, `ok`, and `missing` or `unverified` when they apply |
| `props_scores.fp.json` | what each result was computed from: the sha1 of the frames, crops, item and judge files it read, the provenance of its Reuse queries, the scoring protocol and prompts, and the name, weights and vLLM version of both models |
| `props_score_errors.json` | pairs that could not be scored because a request failed |
| `props_board.json` | the board row (written by `props evaluate`) with the code version and commit, protocol, benchmark commit, and the weights and vLLM version of every server |
| `.judge_cache/` | every reply, by request and weights |
| `<item>/own_e.jpg`, `own_e.json` | the crop of the prop cut from the model's own evidence frame (Reuse), with the sha1 of that frame, the description of the prop, the protocol and the models it was cut with |
| `.props_sha1.json` | sha1 of the files read, by path, size and modification time, so a rerun does not read them again |
| `<item>/props_inputs.json` | the sha1 of the evidence frames each Reuse query was drawn from (docs/generation.md) |

The scoring code, the prompts and the decision rules are part of every fingerprint, so a new version of `props`
scores the pairs again; the cached replies make this cheap when the requests are unchanged. A server that reports no
weight revision or no vLLM version (one not started with `scripts/serve_judge.sh`) is scored without any cache or
earlier result. A pair is scored again when a frame it reads changes (a query or a Reuse evidence frame drawn again), or when the
protocol or the weights behind a server change (as vLLM reports them in `/v1/models`). Its Reuse crop is cut again
when the evidence frame, the description of the prop, the protocol or the models change. Everything else comes
from the files above, so a rerun after an interruption asks only for what is left. When several URLs are given for
one model, every server must answer, serve that model name (`--judge-model`, `--grounder-model`) and report the same
weights and vLLM version, or scoring stops.

## Exit codes

| code | meaning |
|---|---|
| 0 | done |
| 1 | some work failed; the message says what, and running the same command again retries it |
| 2 | bad input: no benchmark at `--bench`, frames missing, or a Reuse query drawn from an older or unrecorded evidence frame |

`props score` and `props evaluate` stop before scoring when the frames were drawn from another version of the data
than the one at `--bench`, when an input image is not on disk (an Ego image that failed its check is set aside),
when a frame is missing, or when a Reuse query was drawn from an older or unrecorded evidence frame. Both first move
an earlier `props_board.json` to `props_board.previous.json`. `--allow-missing` scores anyway and fails each side without a frame;
`--allow-unverified` scores the unverified queries and marks their sides `unverified` until they are scored again
from recorded frames. A request that fails after its retries leaves its pair unscored and listed in
`props_score_errors.json`, so a server problem never turns into a low score.

## Boards for the leaderboard

`props evaluate` writes a new board only when every pair of the split was scored. The board lists in `incomplete_because` every reason that keeps it
out of the leaderboard, and `complete` is true only when there is none:

- the split is `full` or `phase_a`, and every pair was scored;
- no side lacks a frame and no side is unverified;
- the frames were not drawn with `--ablate`;
- the dataset was fetched with `props download`, so its commit is known;
- the code runs from a git checkout without changes;
- every server reports its weights with their revision and its vLLM version.

A system that refuses some requests, as closed APIs sometimes do, can still be reported: score it with
`--allow-missing`, so each refused frame fails its side, and say so in the issue.
