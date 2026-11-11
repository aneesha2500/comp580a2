"""Turn results/*.json(l) into the plots and tables used in the report.

    python make_plots.py            # reads results/, writes results/plots/ and results/tables.md

Agent plots are drawn only when results/agent_runs.jsonl exists (it is produced by
run_experiments.py, which needs a Gemini key).
"""

import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
PLOTS = os.path.join(RES, "plots")
CHOSEN = ["tickets", "bugs", "docs", "hosts", "configs", "code", "revisions", "logs"]
CONFIGS = ["baseline", "task", "content", "both"]


def _load(name):
    path = os.path.join(RES, name)
    if not os.path.exists(path):
        return None
    if name.endswith(".jsonl"):
        return [json.loads(line) for line in open(path) if line.strip()]
    return json.load(open(path))


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, name), dpi=150)
    plt.close(fig)


def plot_threshold(sweep, md):
    rows = [r for r in sweep if not r["default"] and r["dataset"] in CHOSEN]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for validate, style in ((True, "-o"), (False, "--s")):
        by_t = defaultdict(lambda: [0, 0, 0])
        for r in rows:
            if r["validate"] == validate:
                by_t[r["threshold"]][0] += r["reused"]
                by_t[r["threshold"]][1] += r["wrong_reuse"]
                by_t[r["threshold"]][2] += r["tasks"]
        ts = sorted(by_t)
        name = "MinHash + validation (ours)" if validate else "MinHash similarity only"
        axes[0].plot(ts, [100 * by_t[t][0] / by_t[t][2] for t in ts], style, label=name)
        axes[1].plot(ts, [by_t[t][1] for t in ts], style, label=name)
    axes[0].set(xlabel="similarity threshold", ylabel="tasks answered by reuse (%)", title="Reuse rate")
    axes[1].set(xlabel="similarity threshold", ylabel="wrong reused answers", title="Incorrect reuse")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    _save(fig, "threshold_sweep.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    for ds in ("tickets", "bugs", "docs", "configs"):
        rs = sorted([r for r in rows if r["dataset"] == ds and r["validate"]], key=lambda r: r["threshold"])
        ax.plot([r["threshold"] for r in rs], [r["wrong_reuse"] for r in rs], "-o", label=ds, markersize=3)
    ax.set(xlabel="similarity threshold", ylabel="wrong reused answers (with validation)",
           title="Where the threshold still matters")
    ax.grid(alpha=0.3)
    ax.legend()
    _save(fig, "threshold_per_dataset.png")

    md.append("## Task-level reuse at the chosen thresholds (offline replay, all 10 datasets)\n")
    md.append("| dataset | threshold | reused / tasks | wrong reuses | duplicates caught | lookup (ms/task) |")
    md.append("|---|---|---|---|---|---|")
    for r in sweep:
        if r["default"] and r["validate"]:
            md.append("| %s | %.2f | %d / %d | %d | %d / %d | %.2f |" % (
                r["dataset"], r["threshold"], r["reused"], r["tasks"], r["wrong_reuse"],
                r["dups_reused"], r["dups"], r["lookup_ms"]))
    md.append("")


def plot_content(content, sweep, md):
    datasets = sorted({r["dataset"] for r in content})
    saved = []
    md.append("## Content-level reuse (offline replay: every file of every scenario read once)\n")
    md.append("| dataset | files | folded | output chars saved | cumulative context saved | ms / scenario |")
    md.append("|---|---|---|---|---|---|")
    for ds in datasets:
        rs = [r for r in content if r["dataset"] == ds]
        cin, cout = sum(r["chars_in"] for r in rs), sum(r["chars_out"] for r in rs)
        ctx = 1 - sum(r["ctx_fold"] for r in rs) / float(sum(r["ctx_base"] for r in rs))
        saved.append(100 * (1 - cout / float(cin)))
        md.append("| %s | %d | %d | %.1f%% | %.1f%% | %.1f |" % (
            ds, sum(r["files"] for r in rs), sum(r["folds"] for r in rs), saved[-1], 100 * ctx,
            sum(r["ms"] for r in rs) / len(rs)))
    md.append("")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(datasets, saved, color="#1D9E75")
    ax.set(ylabel="tool-output characters saved (%)", title="Content-level reuse by dataset")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "content_by_dataset.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    for ds in datasets:
        pts = defaultdict(lambda: [0, 0])
        for r in sweep:
            if r["dataset"] == ds:
                pts[r["threshold"]][0] += r["chars_in"]
                pts[r["threshold"]][1] += r["chars_out"]
        ts = sorted(pts)
        ax.plot(ts, [100 * (1 - pts[t][1] / float(pts[t][0])) for t in ts], "-o", label=ds, markersize=3)
    ax.set(xlabel="content similarity threshold", ylabel="characters saved (%)",
           title="Content reuse vs threshold")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    _save(fig, "content_threshold.png")


def plot_minhash(acc, md):
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.scatter([a["exact"] for a in acc], [a["estimate"] for a in acc], s=8, alpha=0.6)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set(xlabel="exact Jaccard", ylabel="MinHash estimate (128 hashes)", title="MinHash accuracy")
    _save(fig, "minhash_accuracy.png")
    mae = sum(abs(a["exact"] - a["estimate"]) for a in acc) / len(acc)
    md.append("MinHash estimate vs exact Jaccard: mean absolute error %.4f over %d pairs.\n" % (mae, len(acc)))


def plot_agent(runs, md):
    agg = defaultdict(lambda: defaultdict(float))
    for r in runs:
        a = agg[(r["dataset"], r["config"])]
        a["n"] += 1
        a["passed"] += r["passed"]
        a["reused"] += r["reused"]
        for k in ("tool_calls", "model_requests", "input_tokens", "seconds", "lookup_s", "content_s"):
            a[k] += r.get(k, 0)
    datasets = [d for d in CHOSEN if any((d, c) in agg for c in CONFIGS)]
    md.append("## Agent runs (gemini-3.5-flash-lite)\n")
    md.append("| dataset | config | accuracy | reused | tool calls | model requests | input tokens | time (s) | MinHash overhead (s) |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for ds in datasets:
        for c in CONFIGS:
            a = agg.get((ds, c))
            if a:
                md.append("| %s | %s | %.1f%% | %d | %d | %d | %d | %.0f | %.2f |" % (
                    ds, c, 100 * a["passed"] / a["n"], a["reused"], a["tool_calls"], a["model_requests"],
                    a["input_tokens"], a["seconds"], a["lookup_s"] + a["content_s"]))
    md.append("")

    metrics = [("passed", "accuracy (%)"), ("input_tokens", "input tokens"), ("tool_calls", "tool calls"),
               ("seconds", "completion time (s)")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    width = 0.2
    for ax, (key, title) in zip(axes.flat, metrics):
        for i, c in enumerate(CONFIGS):
            vals = []
            for ds in datasets:
                a = agg.get((ds, c), {})
                v = a.get(key, 0)
                vals.append(100 * v / a["n"] if key == "passed" and a else v)
            ax.bar([x + i * width for x in range(len(datasets))], vals, width, label=c)
        ax.set_xticks([x + 1.5 * width for x in range(len(datasets))])
        ax.set_xticklabels(datasets, rotation=30, fontsize=8)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    axes[0][0].legend(fontsize=8)
    _save(fig, "agent_by_dataset.png")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for c in CONFIGS:
        xs, ys = [], []
        for ds in datasets:
            base, a = agg.get((ds, "baseline")), agg.get((ds, c))
            if base and a and base["input_tokens"]:
                xs.append(100 * (1 - a["input_tokens"] / base["input_tokens"]))
                ys.append(100 * a["passed"] / a["n"] - 100 * base["passed"] / base["n"])
        ax.scatter(xs, ys, label=c)
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set(xlabel="input tokens saved vs baseline (%)", ylabel="accuracy change vs baseline (points)",
           title="Efficiency vs correctness (one point per dataset)")
    ax.grid(alpha=0.3)
    ax.legend()
    _save(fig, "efficiency_vs_correctness.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    over, save = [], []
    for c in ("task", "content", "both"):
        o = sum(agg[(d, c)]["lookup_s"] + agg[(d, c)]["content_s"] for d in datasets if (d, c) in agg)
        s = sum(agg[(d, "baseline")]["seconds"] - agg[(d, c)]["seconds"] for d in datasets
                if (d, c) in agg and (d, "baseline") in agg)
        over.append(o)
        save.append(s)
    ax.bar([0, 1, 2], save, 0.4, label="time saved vs baseline (s)")
    ax.bar([0.4, 1.4, 2.4], over, 0.4, label="MinHash overhead (s)")
    ax.set_xticks([0.2, 1.2, 2.2])
    ax.set_xticklabels(["task", "content", "both"])
    ax.set(title="Computational overhead vs savings")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    _save(fig, "overhead_vs_savings.png")


def main():
    os.makedirs(PLOTS, exist_ok=True)
    md = ["# Results tables\n"]
    acc = _load("minhash_accuracy.json")
    if acc:
        plot_minhash(acc, md)
    sweep = _load("task_sweep.json")
    if sweep:
        plot_threshold(sweep, md)
    content, csweep = _load("content_replay.json"), _load("content_sweep.json")
    if content and csweep:
        plot_content(content, csweep, md)
    runs = _load("agent_runs.jsonl")
    if runs:
        plot_agent(runs, md)
    with open(os.path.join(RES, "tables.md"), "w") as handle:
        handle.write("\n".join(md) + "\n")
    print("wrote %s and %s" % (PLOTS, os.path.join(RES, "tables.md")))


if __name__ == "__main__":
    main()
