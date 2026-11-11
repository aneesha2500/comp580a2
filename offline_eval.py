"""Offline evaluation that needs no model calls (and so no API key).

  1. MinHash accuracy: estimated vs exact Jaccard on real task pairs.
  2. Task-level reuse sweep: replay every dataset's incoming stream through policy.py at a
     range of thresholds, with and without the validation gate. Tasks the policy does not
     answer are treated as solved-and-verified by an *oracle agent* (its label is indexed,
     standing in for an agent answer that passed verify.py). Every reused answer is graded
     by the dataset's own, unmodified verify.py grade().
  3. Content-level replay: an agent that reads every file of each corpus scenario once,
     in sorted order, through content.py. Measures characters saved, folds, losslessness,
     and whether the answer-carrying lines stay visible.

hidden/ is read here ONLY to score (dup/hard-negative labels, answer lines) -- never by
policy.py or content.py.

    python offline_eval.py --assignment ../Assignment2_COMP_580 --spartacus ../Spartacus
"""

import argparse
import importlib.util
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from content import ContentReuser                              # noqa: E402
from minhash import MinHasher, estimate_jaccard, exact_jaccard, shingles  # noqa: E402
from policy import SPECS, TaskReusePolicy                     # noqa: E402

DATASETS = ["tickets", "bugs", "docs", "citations", "hosts", "configs", "code", "threads", "revisions", "logs"]
CORPUS = ["hosts", "configs", "code", "threads", "revisions", "logs"]
THRESHOLDS = [round(0.2 + 0.05 * i, 2) for i in range(16)]    # 0.20 .. 0.95


def load(assignment, dataset):
    base = os.path.join(assignment, "assignment2", "datasets", dataset)
    env = os.path.join(base, "env")
    folder = os.path.join(env, "incoming")
    tasks = sorted((json.load(open(os.path.join(folder, f), encoding="utf-8")) for f in os.listdir(folder)),
                   key=lambda t: (t.get("seq", 0), t["tid"]))
    labels = json.load(open(os.path.join(base, "hidden", "labels.json"), encoding="utf-8"))
    spec = importlib.util.spec_from_file_location("verify_" + dataset, os.path.join(base, "verify.py"))
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    return base, env, tasks, labels, verifier


def kind_of(label):
    return "duplicate" if label.get("dup_of") else ("hard_negative" if label.get("hard_negative_of") else "original")


def replay(env, tasks, labels, verifier, dataset, threshold, validate):
    policy = TaskReusePolicy(dataset, env, threshold=threshold, validate=validate)
    out = {"tasks": len(tasks), "reused": 0, "correct_reuse": 0, "wrong_reuse": 0, "wrong_on_hard_neg": 0,
           "dups": 0, "dups_reused": 0, "lookup_ms": 0.0}
    for task in tasks:
        label = labels[task["tid"]]
        out["dups"] += kind_of(label) == "duplicate"
        start = time.perf_counter()
        decision = policy.lookup(task)
        out["lookup_ms"] += (time.perf_counter() - start) * 1000
        if decision["reuse"]:
            ok = all(p for p, _ in verifier.grade(task["tid"], decision["answer"]).values())
            out["reused"] += 1
            out["correct_reuse" if ok else "wrong_reuse"] += 1
            out["wrong_on_hard_neg"] += (not ok) and kind_of(label) == "hard_negative"
            out["dups_reused"] += ok and kind_of(label) == "duplicate"
        else:
            policy.record(task, label["answer"])   # oracle agent: a verified answer gets indexed
    out["lookup_ms"] /= max(1, len(tasks))
    return out


def task_sweep(assignment):
    rows = []
    for dataset in DATASETS:
        _, env, tasks, labels, verifier = load(assignment, dataset)
        for validate in (True, False):
            for threshold in THRESHOLDS + [None]:
                r = replay(env, tasks, labels, verifier, dataset, threshold, validate)
                r.update(dataset=dataset, validate=validate,
                         threshold=SPECS[dataset]["threshold"] if threshold is None else threshold,
                         default=threshold is None)
                rows.append(r)
        d = [r for r in rows if r["dataset"] == dataset and r["default"] and r["validate"]][0]
        print("%-10s default t=%.2f reused %2d/%2d  wrong %d  dup recall %d/%d  lookup %.2f ms" % (
            dataset, d["threshold"], d["reused"], d["tasks"], d["wrong_reuse"], d["dups_reused"], d["dups"],
            d["lookup_ms"]))
    return rows


def content_replay(assignment, spartacus, threshold=None):
    sys.path.insert(0, os.path.abspath(spartacus))
    from spartacus import tools
    rows = []
    for dataset in CORPUS:
        base = os.path.join(assignment, "assignment2", "datasets", dataset)
        truth = json.load(open(os.path.join(base, "hidden", "content.json"), encoding="utf-8"))
        scen_dir = os.path.join(base, "env", "scenarios")
        for scenario in sorted(os.listdir(scen_dir)):
            root = os.path.join(scen_dir, scenario)
            read = {t.name: t for t in tools.core_tools(root)}["read_file"]
            files = sorted(os.path.relpath(os.path.join(b, f), root) for b, _, fs in os.walk(root) for f in fs)
            reuser = ContentReuser() if threshold is None else ContentReuser(threshold=threshold)
            answer_lines = truth.get(scenario, {}).get("answer_lines", {})
            shown_ans = carried_ans = 0
            base_ctx = fold_ctx = 0            # cumulative input if each read is one more turn
            base_run = fold_run = 0
            start = time.perf_counter()
            for path in files:
                raw = read.run(path=path)
                shown = reuser.process("read_file(%s)" % path, raw)
                base_run += len(raw)
                fold_run += len(shown)
                base_ctx += base_run
                fold_ctx += fold_run
                for n in answer_lines.get(path, []):
                    line = raw.split("\n")[n - 1] if n <= len(raw.split("\n")) else ""
                    if shown is raw or line in shown:
                        shown_ans += 1
                    else:
                        carried_ans += 1        # identical to the reference line, recoverable
            rows.append({"dataset": dataset, "scenario": scenario, "files": len(files),
                         "folds": reuser.stats["folded"], "chars_in": reuser.stats["chars_in"],
                         "chars_out": reuser.stats["chars_out"], "ctx_base": base_ctx, "ctx_fold": fold_ctx,
                         "answer_lines_explicit": shown_ans, "answer_lines_by_reference": carried_ans,
                         "ms": (time.perf_counter() - start) * 1000})
    return rows


def minhash_accuracy(assignment, pairs=400, seed=580):
    rng = random.Random(seed)
    hasher = MinHasher()
    errors = []
    for dataset in DATASETS:
        _, env, tasks, _, _ = load(assignment, dataset)
        text = SPECS[dataset]["text"]
        history = [json.loads(l) for l in open(os.path.join(env, "history.jsonl"), encoding="utf-8") if l.strip()]
        pool = [text(r) for r in tasks + history]
        for _ in range(pairs // len(DATASETS)):
            a, b = rng.sample(pool, 2) if rng.random() < 0.5 else (pool[0], pool[0])
            if rng.random() < 0.5:   # make near-duplicates too: drop some words from a copy
                words = a.split()
                b = " ".join(w for w in words if rng.random() > 0.1)
            sa, sb = shingles(a), shingles(b)
            errors.append({"exact": exact_jaccard(sa, sb),
                           "estimate": estimate_jaccard(hasher.signature(sa), hasher.signature(sb))})
    return errors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--assignment", required=True)
    parser.add_argument("--spartacus", required=True)
    parser.add_argument("--out", default=os.path.join(HERE, "results"))
    args = parser.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    acc = minhash_accuracy(args.assignment)
    mae = sum(abs(e["exact"] - e["estimate"]) for e in acc) / len(acc)
    print("MinHash: mean |estimate - exact| = %.4f over %d pairs" % (mae, len(acc)))
    json.dump(acc, open(os.path.join(args.out, "minhash_accuracy.json"), "w"))

    json.dump(task_sweep(args.assignment), open(os.path.join(args.out, "task_sweep.json"), "w"), indent=1)

    content = content_replay(args.assignment, args.spartacus)
    json.dump(content, open(os.path.join(args.out, "content_replay.json"), "w"), indent=1)
    sweep = []
    for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        for r in content_replay(args.assignment, args.spartacus, threshold=t):
            r["threshold"] = t
            sweep.append(r)
    json.dump(sweep, open(os.path.join(args.out, "content_sweep.json"), "w"))
    for dataset in CORPUS:
        rs = [r for r in content if r["dataset"] == dataset]
        cin, cout = sum(r["chars_in"] for r in rs), sum(r["chars_out"] for r in rs)
        print("%-10s content: %3d folds / %3d files, chars -%.1f%%, cumulative ctx -%.1f%%, "
              "answer lines explicit %d / by reference %d, %.1f ms per scenario" % (
                  dataset, sum(r["folds"] for r in rs), sum(r["files"] for r in rs), 100 * (1 - cout / cin),
                  100 * (1 - sum(r["ctx_fold"] for r in rs) / sum(r["ctx_base"] for r in rs)),
                  sum(r["answer_lines_explicit"] for r in rs), sum(r["answer_lines_by_reference"] for r in rs),
                  sum(r["ms"] for r in rs) / len(rs)))


if __name__ == "__main__":
    main()
