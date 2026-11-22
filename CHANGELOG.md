# Changelog

The code version is in `pyproject.toml`; the data version is the release tag of the two Hugging Face repos.
`props evaluate` writes both into `props_board.json`, together with the scoring protocol and the weights of the
two scoring models. A row of the leaderboard always names the versions it was scored with.

## 1.0.0

- PROPS-Benchmark: 2,257 pairs. Official splits `full` (2,231 pairs) and `phase_a` (590) leave out the 26 pairs
  whose history needs relit Ego4D frames; `paper` and `paper_a` keep every pair, for the numbers of the paper.
- PROPS-Train: 80,250 chains, 120,339 training and 6,277 validation targets.
- Both datasets are tagged `v1.0.0` on Hugging Face, and `scripts/setup.sh` downloads this tag.
- Scoring protocol `props-score-v1`: judge google/gemma-4-31B-it at 842da37, grounder Qwen/Qwen3.8-27B at 1d4bf0f,
  served with vLLM 0.30.0.
- Commands: download, rebuild-ego, generate, export, collect, check, score, board, evaluate.
- Leaderboard of the paper (44 systems and 3 reference rows), recomputed from the score files in `leaderboard/runs/`.
- No model weights are released, including those of the four models fine-tuned on PROPS-Train.
- Project page: `page/make_page.py` builds it from `leaderboard/board.json` into the `gh-pages` branch.
