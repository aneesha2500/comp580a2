"""Task-level reuse: answer an incoming task from a verified earlier one, or say "run the agent".

A reuse decision has three gates, and all three must pass:

  1. Retrieval (MinHash + LSH). The task's text is shingled and its signature is looked up
     in an LSH index of every verified record. This is cheap and finds near-duplicates even
     when they are reformatted, reordered, lightly edited or paraphrased.
  2. Similarity. The best candidates must have estimated Jaccard >= the dataset's threshold.
  3. Validation. The candidate's *visible key* -- the structured fields each dataset's
     INSTRUCTIONS.md says must match (e.g. product + error code + platform for tickets,
     scenario + kind + target for the corpus datasets) -- must equal the task's key after
     normalisation. This is what stops hard negatives: two tasks can share 90% of their
     words and still differ in the one field that changes the answer.

If several candidates pass but carry different answers, the policy refuses (ambiguous).
If anything is missing or cannot be parsed, the policy refuses. Refusing is always safe:
the agent simply runs as normal.

Only verified answers are ever indexed: history.jsonl records (verified by construction)
and new answers passed to `record()`, which the runner calls only after verify.py passes.
Nothing here reads hidden/ or MANIFEST.json.
"""

import json
import os
import re

from minhash import LSHIndex, MinHasher


# ---------------------------------------------------------------------------
# small normalisers (the same rules the datasets' INSTRUCTIONS.md describe)
# ---------------------------------------------------------------------------
def _code(value):
    """Case and punctuation do not matter: E-1042 == e1042, CFG-1234 == CFG1234."""
    return re.sub(r"[^a-z0-9]", "", str(value if value is not None else "").lower())


def _text(value):
    return " ".join(re.findall(r"[a-z0-9]+", str(value or "").lower()))


def _frame(frame):
    """lattice/net/http.py::fetch == lattice.net.http.fetch (line numbers dropped)."""
    frame = str(frame).strip().lower()
    frame = re.sub(r"\.py::", ".", frame).replace("/", ".").replace("::", ".")
    return re.sub(r":\d+$", "", frame)


def _series(version):
    """3.1.2, v3.1.4 and 3.1.0 are all series 3.1."""
    match = re.search(r"(\d+)\.(\d+)", str(version))
    return "%s.%s" % match.groups() if match else _text(version)


def _author(name):
    """'Priya Raman', 'Raman, P.' and 'P Raman' -> ('raman', 'p')."""
    name = str(name).strip()
    if "," in name:
        surname, given = [p.strip() for p in name.split(",", 1)]
    else:
        parts = name.split()
        surname, given = (parts[-1], " ".join(parts[:-1])) if len(parts) > 1 else (name, "")
    return (_text(surname), _text(given)[:1])


def _doi(doi):
    doi = str(doi or "").strip().lower()
    doi = re.sub(r"^(https?://)?(dx\.)?doi\.org/", "", doi)
    return re.sub(r"^doi:\s*", "", doi).rstrip(". ")


# docs: the identifier gives the family, a status word gives the class (INSTRUCTIONS.md)
_DOC_ID = re.compile(r"\b(CHG|INC|POL)-\d{3,4}\b|\bREL-\d+\.\d+\b", re.I)
_DOC_CLASSES = {
    "request": r"\brequest|\bpending", "approved": r"\bapprov|\bauthori[sz]",
    "rejected": r"\breject|\bdeclin", "open": r"\bongoing|under investigation",
    "resolved": r"\bresolv|\bclosed", "draft": r"\bdraft|for comment",
    "final": r"\beffective|in force",
}
_DOC_FAMILY = {"CHG": ("request", "approved", "rejected"), "INC": ("open", "resolved"),
               "POL": ("draft", "final"), "REL": ()}


def _doc_key(record):
    match = _DOC_ID.search(record.get("title", ""))
    if not match:
        return None
    ident = match.group(0).upper()
    family = ident.split("-")[0]
    body = str(record.get("body", "")).lower()
    found = [c for c in _DOC_FAMILY[family] if re.search(_DOC_CLASSES[c], body)]
    if family != "REL" and len(found) != 1:
        if family == "CHG" and set(found) - {"request"}:      # 'request' is also an ordinary word
            found = [c for c in found if c != "request"]
        if len(found) != 1:
            return None                                       # unclear class -> never reuse
    return (ident, found[0] if found else "notes")


def _citation_key(record):
    c = record.get("citation") or {}
    try:
        year = int(c.get("year"))
    except (TypeError, ValueError):
        return None
    return (_doi(c.get("doi")), _text(c.get("title")), tuple(_author(a) for a in c.get("authors") or []),
            _code(c.get("venue")), year)


# ---------------------------------------------------------------------------
# per-dataset rules: which fields must match, what text to compare, where the answer is
# ---------------------------------------------------------------------------
def _corpus_spec():
    return {
        "key": lambda r: (_code(r.get("scenario")), _code(r.get("kind")), _code(r.get("target"))),
        "text": lambda r: "%s\n%s" % (r.get("subject", ""), r.get("body", "")),
        "answer": lambda r: r.get("answer"),
        "threshold": 0.30,   # the key decides the answer; similarity only has to retrieve it
    }


SPECS = {
    "tickets": {
        "key": lambda r: (_code(r.get("product")), _code(r.get("error_code")), _code(r.get("platform"))),
        "text": lambda r: "%s\n%s" % (r.get("subject", ""), r.get("body", "")),
        "answer": lambda r: {"resolution_id": r["resolution_id"]},
        "threshold": 0.60,   # the cue lives only in the text, so the text has to agree
    },
    "bugs": {
        "key": lambda r: (_code(r.get("exception")), _frame(r.get("top_frame")), _series(r.get("version"))),
        "text": lambda r: "%s\n%s" % (r.get("title", ""), r.get("body", "")),
        "answer": lambda r: {"component": r["component"], "root_cause": r["root_cause"]},
        "threshold": 0.60,
    },
    "docs": {
        "key": _doc_key,
        "text": lambda r: "%s\n%s" % (r.get("title", ""), r.get("body", "")),
        # a history record that is the same document is, by definition, what it duplicates
        "answer": lambda r: {"category": r["category"], "duplicate_of": r["hid"]},
        "threshold": 0.50,
    },
    "citations": {
        "key": _citation_key,
        "text": lambda r: r.get("raw", ""),
        "answer": lambda r: r.get("verdict"),
        "threshold": 0.40,   # all five graded values are already in the key
    },
}
for _name in ("hosts", "configs", "code", "threads", "revisions", "logs"):
    SPECS[_name] = _corpus_spec()


class TaskReusePolicy:
    """MinHash index of verified records plus the validation gate."""

    def __init__(self, dataset, env_dir, threshold=None, validate=True):
        if dataset not in SPECS:
            raise ValueError("no reuse rules for dataset %r" % dataset)
        self.dataset = dataset
        self.spec = SPECS[dataset]
        self.threshold = self.spec["threshold"] if threshold is None else threshold
        self.validate = validate   # False only for the "similarity alone" ablation
        self.hasher = MinHasher()
        self.index = LSHIndex()
        self.entries = {}          # key -> {"vkey": ..., "answer": ...}
        self._load_history(os.path.join(env_dir, "history.jsonl"))

    def _load_history(self, path):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    self._add("H:" + record["hid"], record, self.spec["answer"](record))

    def _add(self, key, record, answer):
        vkey = self._safe_key(record)
        if vkey is None or not isinstance(answer, dict):
            return False
        self.entries[key] = {"vkey": vkey, "answer": answer}
        self.index.add(key, self.hasher.signature_of_text(self.spec["text"](record)))
        return True

    def _safe_key(self, record):
        try:
            return self.spec["key"](record)
        except (KeyError, TypeError, AttributeError):
            return None

    def lookup(self, task):
        """Return a decision dict. decision["reuse"] is True only when it is safe to skip the agent."""
        vkey = self._safe_key(task)
        if vkey is None:
            return {"reuse": False, "reason": "task fields could not be validated"}
        sig = self.hasher.signature_of_text(self.spec["text"](task))
        similar = self.index.query(sig, self.threshold)
        if not similar:
            return {"reuse": False, "reason": "no record above threshold %.2f" % self.threshold}
        valid = [(k, s) for k, s in similar if not self.validate or self.entries[k]["vkey"] == vkey]
        if not valid:
            best = similar[0]
            return {"reuse": False, "reason": "similar (%s, J~%.2f) but validation fields differ" % best,
                    "rejected": best[0], "similarity": best[1]}
        answers = {json.dumps(self.entries[k]["answer"], sort_keys=True) for k, _ in valid}
        if self.validate and len(answers) > 1:
            return {"reuse": False, "reason": "matching records disagree on the answer"}
        source, sim = valid[0]
        return {"reuse": True, "answer": dict(self.entries[source]["answer"]),
                "source": source, "similarity": sim, "reason": "validated near-duplicate"}

    def record(self, task, answer):
        """Index a newly solved task. Call ONLY after verify.py has passed it."""
        return self._add("T:" + task["tid"], task, answer)
