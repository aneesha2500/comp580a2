# MinHash for Efficient Agent Trajectories (COMP 480/580, Assignment 2)

Aneesha Nayak

## 1. Methodology

**MinHash and LSH (`minhash.py`).** A text is lower-cased and split into alphanumeric words. Each three-word shingle is mapped to a stable 64-bit integer with BLAKE2b, then pushed through 128 random linear hash functions h_i(x) = (a_i x + b_i) mod (2^61 − 1). The parameters come from a fixed seed (580). The signature keeps the minimum of each h_i, and the fraction of matching positions estimates Jaccard similarity. Over 400 pairs, the mean absolute error against exact Jaccard is **0.018** (`plots/minhash_accuracy.png`). For lookup, the 128 values are split into 32 bands of 4 rows. Two items become candidates if any band matches, which happens with probability 1 − (1 − J⁴)³². That curve is steep around J ≈ 0.42, so near-duplicates are almost always retrieved and unrelated records are almost never compared.

**Task-level reuse (`policy.py`).** A reuse needs three gates to pass. Refusing is always safe: the agent just runs.

1. **Retrieve.** LSH over the verified records: `history.jsonl` plus new answers indexed only after `verify.py` passes.
2. **Similarity.** The estimated Jaccard must be at least a per-dataset threshold.
3. **Validate.** The candidate's *visible key* must equal the task's key after normalization. The visible key is the structured fields that each dataset's `INSTRUCTIONS.md` says decide the answer:
   - tickets: product + error code + platform
   - bugs: exception + top frame + version series
   - docs: document id + status class
   - citations: all five graded fields
   - corpus datasets: scenario + kind + target

   If several valid candidates disagree on the answer, the policy refuses.

**Content-level reuse (`content.py`).** Inside one trajectory, each `read_file` and `bash` output is compared against outputs already shown *in full*. The comparison uses MinHash over shingles with numbers masked, so `port 9000` and `port 9001` look alike. If an output is similar enough (J ≥ 0.6), it is replaced by a line diff against the earlier output. The diff prints every changed line verbatim, with its line number, so every number, identifier and config value survives. Three rules keep this safe:

- A fold is used only if it is at least 30% shorter.
- A fold is used only if rebuilding the output from the reference plus the diff gives back the **exact** original text. This is checked on every fold.
- Short outputs, errors and dissimilar outputs pass through unchanged.

## 2. Experimental setup

- Model: `gemini-3.5-flash-lite`, with the same prompt and the same Spartacus settings (40 turns, no sub-agents) in all four configurations.
- Datasets: tickets, bugs, docs, hosts, configs, code, revisions, logs.
- Each task runs in a fresh directory with only what `INSTRUCTIONS.md` promises.
- `hidden/` is never read by the method. Only `verify.py` (unchanged) and the offline scoring read it.
- Metrics per task: verifier pass/fail, tool calls, model requests, input tokens, wall-clock time, and MinHash overhead (lookup time plus content-processing time, included in the total time).

Two kinds of evaluation:

- **Offline replay** (no model, all 10 datasets, every task). This isolates the reuse decisions from LLM noise. Task reuse is scored with the verifier. Content reuse replays every scenario file once.
- **Agent runs** (8 datasets × 4 configurations, first N tasks by `seq`).

## 3. Results

### 3.1 Task-level reuse (offline replay)

| dataset | threshold | reused / tasks | wrong reuses | duplicates caught | similarity-only wrong reuses at same threshold | lookup (ms/task) |
|---|---|---|---|---|---|---|
| tickets | 0.60 | 21 / 70 | 0 | 21 / 28 | 7 | 1.22 |
| bugs | 0.60 | 16 / 60 | 0 | 16 / 24 | 8 | 1.53 |
| docs | 0.50 | 18 / 60 | 0 | 18 / 24 | 10 | 2.00 |
| citations | 0.40 | 21 / 70 | 0 | 21 / 28 | 16 | 0.59 |
| hosts | 0.30 | 12 / 40 | 0 | 12 / 16 | 18 | 0.82 |
| configs | 0.30 | 12 / 40 | 0 | 12 / 16 | 18 | 0.86 |
| code | 0.30 | 12 / 40 | 0 | 12 / 16 | 21 | 0.83 |
| threads | 0.30 | 13 / 40 | 0 | 13 / 16 | 15 | 0.83 |
| revisions | 0.30 | 13 / 40 | 0 | 13 / 16 | 16 | 0.95 |
| logs | 0.30 | 12 / 40 | 0 | 12 / 16 | 20 | 0.92 |

With validation, no dataset makes a wrong reuse, and 75–81% of true duplicates are answered without running the agent. With similarity alone, every dataset makes many wrong reuses at the same threshold (`plots/threshold_sweep.png`, `plots/threshold_per_dataset.png`). A lookup costs about 1 ms, versus tens of seconds for one agent run.

### 3.2 Content-level reuse (offline replay)

| dataset | files | folded | output chars saved | cumulative context saved |
|---|---|---|---|---|
| code | 283 | 139 | 60.9% | 60.0% |
| configs | 212 | 164 | 48.6% | 40.5% |
| hosts | 175 | 151 | 45.7% | 40.5% |
| revisions | 198 | 163 | 43.7% | 40.9% |
| logs | 181 | 0 | 0.0% | 0.0% |
| threads | 710 | 0 | 0.0% | 0.0% |

Every fold passed the lossless rebuild check, so no answer-critical line is ever dropped (`plots/content_by_dataset.png`, `plots/content_threshold.png`). Processing costs 45–150 ms per scenario.

### 3.3 Agent runs

AGENT_RESULTS_TABLE

Plots: `plots/agent_by_dataset.png` (performance across datasets), `plots/efficiency_vs_correctness.png`, `plots/overhead_vs_savings.png`.

AGENT_RESULTS_DISCUSSION

## 4. Discussion

**Q1. When is task reuse beneficial?** It helps when the incoming stream repeats *requests*, not just topics. Examples:

- the same customer issue re-filed with different wording;
- the same document re-sent;
- the same citation in another format;
- the same question about the same scenario.

For record-based datasets (tickets, bugs, docs, citations), the answer is a pure function of a few visible fields. So once a verified record with the same key and similar text is found, its answer is correct and the agent is skipped entirely: zero model requests and zero tool calls, about 1 ms instead of a full trajectory. Corpus datasets benefit too, but only when the same question is asked about the same scenario. Then the visible key (scenario, kind, target) fixes the answer. Reuse can never help the first occurrence of a task, so the benefit grows with how much of the stream repeats (about 40% of tasks here).

**Q2. Is high similarity sufficient?** No. Two examples:

- **tickets.** T-bcf21d ("E-1359: totals mismatch on **macOS**") has estimated J ≈ 0.80 with history record H-f5ea55 ("E-1359: totals mismatch on **Android**"). Same product, same error code, almost the same text. But the platform differs, and the correct resolution is R-08, not R-03.
- **configs.** T-1344eb asks the same question as T-b06652 ("which environment violates a security requirement", J ≈ 0.82), but about scenario CFG-1228 instead of CFG-1000. The violating environment is prod-us-east, not prod-eu.

Similarity-only reuse answers both wrongly. Our validation gate compares the visible key (platform in the first case, scenario in the second), sees that it differs, and runs the agent instead. A high Jaccard only says most of the words agree; a hard negative is built so that the few words that disagree are exactly the ones that decide the answer.

**Q3. How does the threshold matter?** Lowering the threshold catches more duplicates, which are often paraphrased or reformatted. But it also admits more hard negatives. Without validation, wrong reuses rise from 1–10 per dataset at 0.8 to 12–21 at 0.3. With validation, the gate removes almost all of that risk, so the threshold mainly controls recall. A single threshold is **not** right for every dataset:

- **Corpus datasets:** the key decides the answer, so 0.30 is safe and maximizes recall.
- **tickets:** the key does not capture everything. With validation on, 0.50 still gives 7 wrong reuses, but 0.60 gives 0. So tickets need the higher threshold.
- **docs, bugs:** these sit between the two (0.50–0.60).

**Q4. When is content reuse effective?** When a trajectory reads many files generated from one template that differ in a few values: configs (one file per environment), hosts (inventory snapshots), revisions (document versions) and code (sibling modules). There, 40–60% of the context the agent would carry is removed, and the diff points the model straight at the lines that changed. On logs and threads it gives nothing: each log file and each email is mostly unique text, so no earlier output reaches the similarity threshold, or the diff would not be 30% smaller. In that case the original is returned, at a small cost (about 0.1 s per scenario).

**Q5. Which optimization is more valuable?**

- **Task reuse** has the larger effect per hit: it removes the whole trajectory (every request, tool call and token). With the validation gate, its correctness risk is zero on these datasets. It only helps on repeated tasks.
- **Content reuse** helps every trajectory that re-reads similar files, including first-time tasks. But it saves a fraction of tokens rather than whole runs, and it slightly changes what the model sees (a diff instead of a file), which can affect its reasoning.

Prefer task reuse when the stream has many repeats and the answer is determined by visible fields. Prefer content reuse when tasks are new but each one reads many near-identical files. They combine well, since content reuse applies to exactly the tasks that task reuse could not skip.

**Q6. Limitations.**

- **Lexical similarity is not equivalence.** Two tasks can share 90% of their words and differ in the one field that matters (hard negatives). Two genuinely equivalent tasks can share few shingles if they are paraphrased heavily or written in another language. In those cases MinHash misses the duplicate.
- **Some answers depend on things outside the text.** Our validation relies on per-dataset key rules, which require knowing which fields decide the answer. Corpus answers also depend on the files, which could change between two identical-looking questions.
- **Overhead can outweigh savings.** For short trajectories, or datasets with no near-duplicate outputs (logs, threads), the MinHash work is pure cost, though small.

A possible improvement: learn the validation step instead of hand-writing it. For example, use a cheap verifier-guided check (run the stored answer through a lightweight consistency test, or ask the model one yes/no question comparing the two tasks' key fields), or combine MinHash with an embedding similarity for paraphrases. Another option is to make content reuse adaptive, so it switches off for a dataset once it sees no folds.

## 5. Reproducing

See `README.md` for the exact commands. Summary:

```sh
python3 tests/test_reuse.py
python3 offline_eval.py --assignment ../Assignment2_COMP_580 --spartacus ../Spartacus
python3 run_experiments.py --assignment ../Assignment2_COMP_580 --spartacus ../Spartacus \
    --datasets tickets bugs docs hosts configs code revisions logs \
    --configs baseline task content both --model gemini-3.5-flash-lite --limit 15
python3 make_plots.py
```

All randomness is fixed: MinHash seed 580, tasks in `seq` order, and datasets from their generators' fixed seeds.
