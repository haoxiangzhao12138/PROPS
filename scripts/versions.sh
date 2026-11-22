# Versions that produced the reference scores. Sourced by setup.sh, serve_judge.sh and evaluate.sh.
# The scores in leaderboard/ were computed with these weights and this vLLM; other versions may shift them a little.
JUDGE_REPO=google/gemma-4-31B-it
JUDGE_REV=842da3794eaa0b77d5f08bae87a17459d91ff475      # 62.6 GB
GROUNDER_REPO=Qwen/Qwen3.8-27B
GROUNDER_REV=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0   # 55.6 GB
VLLM_VERSION=0.30.0
BENCH_REV=${BENCH_REV:-v1.0.0}   # PROPS-Benchmark release tag
TRAIN_REV=${TRAIN_REV:-v1.0.0}   # PROPS-Train release tag
# `props download` resolves a revision to its commit before downloading and writes it to props_download.json.
