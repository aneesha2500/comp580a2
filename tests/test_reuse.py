"""Small self-checks: python -m pytest tests  (or: python tests/test_reuse.py)."""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from content import ContentReuser                                      # noqa: E402
from minhash import LSHIndex, MinHasher, estimate_jaccard, exact_jaccard, shingles  # noqa: E402


def _numbered(lines):
    return "\n".join("%d\t%s" % (i + 1, line) for i, line in enumerate(lines))


def test_estimate_close_to_exact():
    a = "the quick brown fox jumps over the lazy dog near the river bank today"
    b = a.replace("lazy", "sleepy")
    h = MinHasher()
    sa, sb = shingles(a), shingles(b)
    assert abs(estimate_jaccard(h.signature(sa), h.signature(sb)) - exact_jaccard(sa, sb)) < 0.15


def test_signatures_are_deterministic():
    assert MinHasher().signature_of_text("a b c d e") == MinHasher().signature_of_text("a b c d e")


def test_lsh_finds_near_duplicate_and_not_unrelated():
    h, index = MinHasher(), LSHIndex()
    base = " ".join("word%d" % i for i in range(200))
    index.add("same", h.signature_of_text(base))
    index.add("other", h.signature_of_text(" ".join("x%d" % i for i in range(200))))
    hits = [k for k, _ in index.query(h.signature_of_text(base.replace("word7 ", "")), 0.5)]
    assert hits == ["same"]


def test_folds_are_lossless_under_random_edits():
    rng = random.Random(0)
    template = ["key_%d: value %d   # shared comment text for line %d" % (i, i * 7, i) for i in range(60)]
    for trial in range(200):
        reuser = ContentReuser()
        reuser.process("ref", _numbered(template))
        lines = list(template)
        for _ in range(rng.randint(1, 6)):
            pos = rng.randrange(len(lines))
            op = rng.choice(["change", "insert", "delete"])
            if op == "change":
                lines[pos] = lines[pos].replace("value", "value %d" % rng.randint(0, 99))
            elif op == "insert":
                lines.insert(pos, "new_key_%d: %d" % (trial, rng.randint(0, 99)))
            elif len(lines) > 1:
                del lines[pos]
        text = _numbered(lines)
        shown = reuser.process("new", text)
        if shown != text:                       # folded: every changed line must be printed verbatim
            assert shown.startswith("[reuse]")
            changed = [l for l in lines if l not in template]
            assert all(l in shown for l in changed)


def test_short_or_unrelated_output_is_untouched():
    reuser = ContentReuser()
    assert reuser.process("a", "short") == "short"
    other = _numbered(["completely different %d" % i for i in range(80)])
    reuser.process("b", _numbered(["line %d of text" % i for i in range(80)]))
    assert reuser.process("c", other) == other


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
