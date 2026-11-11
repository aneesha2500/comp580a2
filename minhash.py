"""MinHash and LSH, written from scratch.

A text becomes a set of three-word shingles. Each shingle is hashed once to a 64-bit
integer (hashlib, so it is stable across runs and machines), then pushed through 128
random linear hash functions h_i(x) = (a_i * x + b_i) mod P. The signature keeps the
minimum of each h_i over the set. Two signatures agree in position i with probability
equal to the Jaccard similarity of the two sets, so the fraction of matching positions
estimates it.

LSH splits the 128 values into BANDS bands of ROWS rows. Two items become candidates
when at least one band is identical, which happens with probability 1 - (1 - J^ROWS)^BANDS.
With 32 bands of 4 rows the curve is steep around J = (1/32)^(1/4) ~= 0.42, so pairs
above ~0.6 are almost always found and pairs below ~0.2 are almost never compared.
"""

import hashlib
import random
import re

NUM_HASHES = 128
SHINGLE_SIZE = 3
BANDS, ROWS = 32, 4          # BANDS * ROWS must equal NUM_HASHES
PRIME = (1 << 61) - 1        # a Mersenne prime larger than any 64-bit shingle id mod it
MAX_HASH = PRIME             # value used for an empty set
SEED = 580                   # fixed, so every run builds the same hash functions

_WORD = re.compile(r"[a-z0-9]+")
_HEXISH = re.compile(r"^[0-9a-f]+$")
_DIGITS = re.compile(r"\d+")


def tokenize(text, mask_numbers=False):
    """Lower-case alphanumeric words. With mask_numbers, digits (and hex ids) become '0'."""
    words = _WORD.findall(str(text).lower())
    if mask_numbers:
        words = ["0" if _HEXISH.match(w) and any(c.isdigit() for c in w) else _DIGITS.sub("0", w)
                 for w in words]
    return words


def shingles(text, k=SHINGLE_SIZE, mask_numbers=False):
    """The set of k-word shingles of a text (a short text gives one shingle of all its words)."""
    words = tokenize(text, mask_numbers)
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


def _shingle_id(shingle):
    """A stable 64-bit integer for one shingle."""
    return int.from_bytes(hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest(), "big")


class MinHasher:
    """Holds the 128 hash functions; turns a shingle set into a signature."""

    def __init__(self, num_hashes=NUM_HASHES, seed=SEED):
        rng = random.Random(seed)
        self.params = [(rng.randrange(1, PRIME), rng.randrange(0, PRIME)) for _ in range(num_hashes)]

    def signature(self, shingle_set):
        ids = [_shingle_id(s) for s in shingle_set]
        if not ids:
            return tuple([MAX_HASH] * len(self.params))
        return tuple(min((a * x + b) % PRIME for x in ids) for a, b in self.params)

    def signature_of_text(self, text, mask_numbers=False):
        return self.signature(shingles(text, mask_numbers=mask_numbers))


def estimate_jaccard(sig_a, sig_b):
    """Fraction of positions where the two signatures agree."""
    return sum(1 for x, y in zip(sig_a, sig_b) if x == y) / float(len(sig_a))


def exact_jaccard(set_a, set_b):
    """Exact Jaccard, used only in analysis to check the estimate."""
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / float(len(set_a | set_b))


class LSHIndex:
    """Banded LSH over MinHash signatures: add items, then query for similar ones."""

    def __init__(self, bands=BANDS, rows=ROWS):
        assert bands * rows == NUM_HASHES
        self.bands, self.rows = bands, rows
        self.buckets = [dict() for _ in range(bands)]   # band -> {band values: [keys]}
        self.signatures = {}                            # key -> full signature

    def _bands(self, sig):
        for b in range(self.bands):
            yield b, sig[b * self.rows:(b + 1) * self.rows]

    def add(self, key, sig):
        self.signatures[key] = sig
        for b, chunk in self._bands(sig):
            self.buckets[b].setdefault(chunk, []).append(key)

    def candidates(self, sig):
        """Keys that share at least one band with `sig` (the LSH candidates)."""
        found = []
        seen = set()
        for b, chunk in self._bands(sig):
            for key in self.buckets[b].get(chunk, ()):
                if key not in seen:
                    seen.add(key)
                    found.append(key)
        return found

    def query(self, sig, threshold=0.0):
        """[(key, estimated Jaccard)] for candidates at or above `threshold`, best first."""
        scored = [(key, estimate_jaccard(sig, self.signatures[key])) for key in self.candidates(sig)]
        return sorted([s for s in scored if s[1] >= threshold], key=lambda s: (-s[1], str(s[0])))

    def __len__(self):
        return len(self.signatures)
