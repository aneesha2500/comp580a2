# COMP 480/580 Assignment 2: MinHash for Efficient Agent Trajectories

The three required files:

| File | What it does |
|---|---|
| `minhash.py` | MinHash (3-word shingles, 128 hash functions, fixed seed 580) and banded LSH (32 bands x 4 rows), written from scratch. Only `hashlib` is used, to turn a shingle into a 64-bit integer. |
| `policy.py` | Task-level reuse. MinHash/LSH finds similar verified records, a per-dataset threshold filters them, and a validation gate (the fields each dataset's `INSTRUCTIONS.md` says must match) blocks hard negatives. New answers are indexed only after `verify.py` passes. |
| `content.py` | Content-level reuse. A tool output that is a near-duplicate (numbers masked) of one already shown in full is replaced by a lossless exact line diff. Every fold is rebuilt and checked against the original before it is used, otherwise the original is returned. |

Helper scripts (not part of the method):

- `run_experiments.py` runs the Spartacus agent in the four configurations and logs accuracy, tool calls, model requests, input tokens, completion time and MinHash overhead.
- `offline_eval.py` runs the evaluation that needs no model: MinHash accuracy, the task-reuse threshold sweep (with and without validation), and the content-reuse replay.
- `make_plots.py` draws every plot into `results/plots/` and writes `results/tables.md`.
- `tests/test_reuse.py` has small self-checks.

Nothing reads `hidden/` or `MANIFEST.json` to make a decision. Only `verify.py` (unchanged, run as a subprocess) and the scoring part of `offline_eval.py` read `hidden/`.

## Setup

```sh
git clone https://github.com/sahiljoshi515/Spartacus
git clone https://github.com/sahiljoshi515/Assignment2_COMP_580
git clone <this repo> minhash-agent
cd minhash-agent
python3 -m pip install matplotlib          # only make_plots.py needs it
mkdir -p ~/.config/spartacus && echo "GEMINI_API_KEY=..." > ~/.config/spartacus/env
```

## Reproduce

```sh
# 1. self-checks
python3 tests/test_reuse.py

# 2. offline evaluation (no API key needed)
python3 offline_eval.py --assignment ../Assignment2_COMP_580 --spartacus ../Spartacus

# 3. agent runs: 8 datasets x 4 configurations, gemini-3.5-flash-lite
python3 run_experiments.py --assignment ../Assignment2_COMP_580 --spartacus ../Spartacus \
    --datasets tickets bugs docs hosts configs code revisions logs \
    --configs baseline task content both --model gemini-3.5-flash-lite

# 4. plots and tables
python3 make_plots.py
```

Add `--limit 20` to step 3 for a cheaper run on the first 20 tasks of each dataset. `--threshold 0.5` overrides the per-dataset task-reuse threshold for a threshold experiment.

All randomness is fixed: the MinHash hash functions use seed 580, task order is by `seq`, and the datasets are fixed by their generators' seeds.
