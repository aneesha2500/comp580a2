"""Run the agent on a dataset under one of four configurations and log every metric.

    baseline  plain agent
    task      task-level reuse (policy.py) before the agent
    content   content-level reuse (content.py) on read_file / bash outputs
    both      both of the above

Every task gets a fresh working directory holding exactly what INSTRUCTIONS.md promises:
INSTRUCTIONS.md, history.jsonl, task.json, any reference file (citations' db.json) and,
for corpus datasets, the task's scenario files. hidden/ is never copied or read; only
verify.py reads it, as a subprocess, after the answer is written.

Example:
    python run_experiments.py --assignment ../Assignment2_COMP_580 --spartacus ../Spartacus \
        --datasets tickets configs --configs baseline task content both
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from content import ContentReuser   # noqa: E402
from policy import TaskReusePolicy  # noqa: E402

MODEL = "gemini-3.5-flash-lite"
PROMPT = ("Solve the task in task.json by following INSTRUCTIONS.md in this directory. "
          "Write the final answer to answer.json exactly as INSTRUCTIONS.md specifies.")
CONFIGS = ("baseline", "task", "content", "both")


def load_tasks(env):
    folder = os.path.join(env, "incoming")
    tasks = [json.load(open(os.path.join(folder, f), encoding="utf-8")) for f in sorted(os.listdir(folder))]
    return sorted(tasks, key=lambda t: (t.get("seq", 0), t["tid"]))


def _code(value):
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())


def make_workdir(env, task):
    """A fresh directory with the files the agent is allowed to see for this task."""
    work = tempfile.mkdtemp(prefix="a2_%s_" % task["tid"])
    for name in sorted(os.listdir(env)):
        path = os.path.join(env, name)
        if name in ("incoming", "scenarios"):
            continue
        (shutil.copytree if os.path.isdir(path) else shutil.copy)(path, os.path.join(work, name))
    scenarios = os.path.join(env, "scenarios")
    if task.get("scenario") and os.path.isdir(scenarios):
        match = [d for d in os.listdir(scenarios) if _code(d) == _code(task["scenario"])]
        if match:
            shutil.copytree(os.path.join(scenarios, match[0]), work, dirs_exist_ok=True)
    with open(os.path.join(work, "task.json"), "w", encoding="utf-8") as handle:
        json.dump(task, handle, indent=2)
    return work


def verify(dataset_dir, tid, answer_path):
    """Run the provided verifier unchanged; True on PASS."""
    done = subprocess.run([sys.executable, os.path.join(dataset_dir, "verify.py"), "--tid", tid,
                           "--answer", answer_path], capture_output=True, text=True)
    return done.returncode == 0


def wrap_tools(workdir, reuser, timer):
    """read_file and bash outputs go through the content layer before the model sees them."""
    from spartacus import tools
    core = {t.name: t for t in tools.core_tools(workdir)}

    def wrapped(name, label_of):
        original = core[name]

        def run(**kwargs):
            output = original.run(**kwargs)
            start = time.perf_counter()
            shown = reuser.process(label_of(kwargs), output)
            timer["content_s"] += time.perf_counter() - start
            return shown
        return tools.Tool(name, original.spec, run)

    return [wrapped("read_file", lambda kw: "read_file(%s)" % kw.get("path")),
            wrapped("bash", lambda kw: "bash(%s)" % str(kw.get("command"))[:80])]


def run_agent(workdir, model, max_turns, use_content):
    from spartacus import Harness
    metrics = {"model_requests": 0, "tool_calls": 0, "input_tokens": 0, "output_tokens": 0, "content_s": 0.0}

    def on_event(kind, payload):
        if kind == "assistant":
            metrics["model_requests"] += 1
            usage = payload.get("usage") or {}
            metrics["input_tokens"] += usage.get("input", 0)
            metrics["output_tokens"] += usage.get("output", 0)
        elif kind == "tool_start":
            metrics["tool_calls"] += 1

    reuser = ContentReuser() if use_content else None
    extra = wrap_tools(workdir, reuser, metrics) if use_content else None
    agent = Harness(workdir, model=model, on_event=on_event, extra_tools=extra, max_turns=max_turns,
                    persist=False, enable_subagents=False)
    try:
        agent.run(PROMPT)
    except Exception as error:   # a provider failure is a failed task, not a crashed experiment
        metrics["error"] = "%s: %s" % (type(error).__name__, error)
    if reuser:
        metrics["content_folds"] = reuser.stats["folded"]
        metrics["content_chars_saved"] = reuser.stats["chars_in"] - reuser.stats["chars_out"]
    return metrics


def run(dataset, config, args):
    dataset_dir = os.path.join(args.assignment, "assignment2", "datasets", dataset)
    env = os.path.join(dataset_dir, "env")
    use_task, use_content = config in ("task", "both"), config in ("content", "both")
    policy = TaskReusePolicy(dataset, env, threshold=args.threshold) if use_task else None
    tasks = load_tasks(env)[:args.limit] if args.limit else load_tasks(env)
    for task in tasks:
        row = {"dataset": dataset, "config": config, "tid": task["tid"], "reused": False,
               "model_requests": 0, "tool_calls": 0, "input_tokens": 0, "output_tokens": 0,
               "lookup_s": 0.0, "content_s": 0.0}
        start = time.perf_counter()
        workdir = make_workdir(env, task)
        answer_path = os.path.join(workdir, "answer.json")
        decision = {"reuse": False}
        if policy:
            t0 = time.perf_counter()
            decision = policy.lookup(task)
            row["lookup_s"] = time.perf_counter() - t0
            row["reuse_reason"] = decision["reason"]
        if decision["reuse"]:
            row.update(reused=True, reuse_source=decision["source"], similarity=decision["similarity"])
            with open(answer_path, "w", encoding="utf-8") as handle:
                json.dump(decision["answer"], handle)
        else:
            row.update(run_agent(workdir, args.model, args.max_turns, use_content))
        row["passed"] = verify(dataset_dir, task["tid"], answer_path)
        row["seconds"] = time.perf_counter() - start
        if policy and row["passed"] and not row["reused"]:
            with open(answer_path, encoding="utf-8") as handle:
                policy.record(task, json.load(handle))     # index only verified answers
        shutil.rmtree(workdir, ignore_errors=True)
        print("%-9s %-8s %s %s reused=%s tools=%d req=%d in_tok=%d %.1fs" % (
            dataset, config, task["tid"], "PASS" if row["passed"] else "FAIL", row["reused"],
            row["tool_calls"], row["model_requests"], row["input_tokens"], row["seconds"]), flush=True)
        yield row

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--assignment", required=True, help="path to the Assignment2_COMP_580 checkout")
    parser.add_argument("--spartacus", required=True, help="path to the Spartacus checkout")
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=CONFIGS)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument("--threshold", type=float, default=None, help="override the per-dataset threshold")
    parser.add_argument("--limit", type=int, default=0, help="only the first N tasks (0 = all)")
    parser.add_argument("--out", default=os.path.join(HERE, "results", "agent_runs.jsonl"))
    args = parser.parse_args(argv)
    sys.path.insert(0, os.path.abspath(args.spartacus))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as out:
        for dataset in args.datasets:
            for config in args.configs:
                for row in run(dataset, config, args):
                    out.write(json.dumps(row) + "\n")
                    out.flush()


if __name__ == "__main__":
    main()
