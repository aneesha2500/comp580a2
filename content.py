"""Content-level reuse: inside one trajectory, send a near-duplicate tool output as a diff.

When the agent reads nine config files built from one template, it pays for ~75 lines
nine times, and every later turn re-sends all nine. Here each tool output is checked
against the outputs already shown *in full* earlier in the same trajectory:

  1. Retrieval: MinHash + LSH over 3-word shingles with numbers masked ("port 9000" and
     "port 9001" shingle the same), so files that differ only in values look alike.
  2. Exact diff: difflib on the real lines (numbers NOT masked) against the best match.
     The model gets "[reuse] same as <earlier output> except:" plus every changed line,
     verbatim, with its line number. Every number, identifier and changed key survives
     because every line that differs in any character is printed.
  3. Safety checks, any of which returns the original output unchanged:
       - the output is short, or no earlier output is similar enough;
       - the fold does not save at least MIN_SAVING of the characters;
       - rebuilding the output from the reference + the diff does not give back the
         exact original text (the fold must be lossless, checked every time).

Only outputs shown in full become references, so the model never has to chain two
folds together to recover a file.
"""

import difflib
import re

from minhash import LSHIndex, MinHasher, shingles

SIM_THRESHOLD = 0.60   # masked-number Jaccard needed to try a fold
MIN_CHARS = 400        # shorter outputs are cheap anyway
MIN_SAVING = 0.30      # the fold must be at least 30% smaller than the original
_NUMBERED = re.compile(r"^(\d+)\t(.*)$")


def _split_numbering(text):
    """read_file returns 'N<TAB>line'; diff on the lines themselves so shifts don't matter."""
    lines = text.split("\n")
    parsed = [_NUMBERED.match(line) for line in lines]
    if lines and all(parsed):
        return [m.group(2) for m in parsed], True
    return lines, False


class ContentReuser:
    """One per trajectory. Call process() on every tool output before the model sees it."""

    def __init__(self, threshold=SIM_THRESHOLD, min_saving=MIN_SAVING):
        self.threshold = threshold
        self.min_saving = min_saving
        self.hasher = MinHasher()
        self.index = LSHIndex()
        self.shown = {}            # label -> lines of an output the model saw in full
        self.stats = {"outputs": 0, "folded": 0, "chars_in": 0, "chars_out": 0}

    def process(self, label, text):
        """Return the text to show the model: a lossless fold, or the original."""
        text = str(text)
        self.stats["outputs"] += 1
        self.stats["chars_in"] += len(text)
        out = self._try_fold(label, text)
        if out is None:
            out = text
            self._remember(label, text)
        else:
            self.stats["folded"] += 1
        self.stats["chars_out"] += len(out)
        return out

    def _remember(self, label, text):
        if len(text) < MIN_CHARS or text.startswith(("ERROR", "BLOCKED")):
            return
        key = label if label not in self.shown else "%s (#%d)" % (label, len(self.shown) + 1)
        self.shown[key] = _split_numbering(text)[0]
        self.index.add(key, self.hasher.signature(shingles(text, mask_numbers=True)))

    def _try_fold(self, label, text):
        if len(text) < MIN_CHARS or not self.shown:
            return None
        sig = self.hasher.signature(shingles(text, mask_numbers=True))
        lines = _split_numbering(text)[0]
        for ref, sim in self.index.query(sig, self.threshold)[:3]:
            fold = self._fold(label, lines, ref, sim)
            if fold and len(fold) <= (1 - self.min_saving) * len(text) and self._lossless(fold, ref, lines):
                return fold
        return None

    def _fold(self, label, lines, ref, sim):
        old = self.shown[ref]
        ops = difflib.SequenceMatcher(None, old, lines, autojunk=False).get_opcodes()
        if all(tag == "equal" for tag, *_ in ops):
            return "[reuse] %s is identical to the earlier output of %s shown above (%d lines)." % (
                label, ref, len(lines))
        out = ["[reuse] %s = earlier output of %s (above), except: each '@ earlier a-b -> now c-d' "
               "replaces earlier lines a-b with the '+ N' lines below it (N = line number now). "
               "All other lines are identical." % (label, ref)]
        for tag, i1, i2, j1, j2 in ops:
            if tag == "equal":
                continue
            out.append("@ earlier %d-%d -> now %d-%d" % (i1 + 1, i2, j1 + 1, j2))
            out.extend("+ %d\t%s" % (j1 + 1 + n, line) for n, line in enumerate(lines[j1:j2]))
        out.append("[/reuse] %d lines total (masked-number similarity %.2f)" % (len(lines), sim))
        return "\n".join(out)

    def _lossless(self, fold, ref, lines):
        """Rebuild the output from the reference and the fold; it must match exactly."""
        old = self.shown[ref]
        if "is identical to the earlier output" in fold.split("\n")[0]:
            return old == lines
        rebuilt, cursor = [], 0
        for row in fold.split("\n")[1:-1]:
            head = re.match(r"@ earlier (\d+)-(\d+) -> now \d+-\d+$", row)
            added = re.match(r"\+ \d+\t(.*)$", row, re.S)
            if head:
                rebuilt.extend(old[cursor:int(head.group(1)) - 1])
                cursor = int(head.group(2))
            elif added:
                rebuilt.append(added.group(1))
            else:
                return False
        rebuilt.extend(old[cursor:])
        return rebuilt == lines

    def saved_fraction(self):
        if not self.stats["chars_in"]:
            return 0.0
        return 1 - self.stats["chars_out"] / float(self.stats["chars_in"])
