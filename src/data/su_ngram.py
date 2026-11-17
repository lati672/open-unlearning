"""Deterministic token-ID N-gram assistants and selectors for the SU baseline."""

from collections import Counter, defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class NGramAssistant:
    """Count within examples, using shorter history at the start of each example."""

    def __init__(self, *, training_split, tokenizer_id, tokenizer_revision, vocabulary_size,
                 n=3, smoothing="mle", alpha=0.1):
        if n < 2 or vocabulary_size < 1:
            raise ValueError("Expected n >= 2 and a positive vocabulary size")
        if smoothing not in {"mle", "add_alpha"} or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("Expected mle/add_alpha and a positive finite alpha")
        self.metadata = {
            "training_split": training_split, "tokenizer_id": tokenizer_id,
            "tokenizer_revision": tokenizer_revision, "n": n,
            "smoothing": smoothing, "alpha": alpha, "vocabulary_size": vocabulary_size,
            "sample_count": 0, "token_count": 0, "predicted_token_count": 0,
        }
        self.counts = defaultdict(Counter)
        self.totals = Counter()

    def fit(self, sequences):
        width = self.metadata["n"] - 1
        vocabulary_size = self.metadata["vocabulary_size"]
        for sequence in sequences:
            ids = tuple(sequence)
            if any(type(token) is not int or not 0 <= token < vocabulary_size for token in ids):
                raise ValueError("Sequence token is outside the configured vocabulary")
            self.metadata["sample_count"] += 1
            self.metadata["token_count"] += len(ids)
            self.metadata["predicted_token_count"] += max(0, len(ids) - 1)
            for pos in range(1, len(ids)):
                context = ids[max(0, pos - width):pos]
                self.counts[context][ids[pos]] += 1
                self.totals[context] += 1
        return self

    def probability(self, context, token):
        vocabulary_size = self.metadata["vocabulary_size"]
        context = tuple(context)[-(self.metadata["n"] - 1):]
        if any(type(value) is not int or not 0 <= value < vocabulary_size for value in (*context, token)):
            raise ValueError("Query token is outside the configured vocabulary")
        counts = self.counts.get(context, {})
        total = self.totals[context]
        count = counts.get(token, 0)
        if self.metadata["smoothing"] == "mle":
            return count / total if total else 0.0
        alpha = self.metadata["alpha"]
        return (count + alpha) / (total + alpha * vocabulary_size)

    def scores(self, ids):
        width = self.metadata["n"] - 1
        return [None] + [self.probability(ids[max(0, pos - width):pos], ids[pos])
                         for pos in range(1, len(ids))] if ids else []

    def save(self, path):
        payload = {
            "metadata": self.metadata,
            "contexts": [[list(context), self.totals[context], sorted(self.counts[context].items())]
                         for context in sorted(self.counts)],
        }
        digest = sha256(canonical_json(payload).encode()).hexdigest()
        Path(path).write_text(canonical_json({**payload, "sha256": digest}) + "\n")
        return digest

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text())
        expected = payload.pop("sha256")
        if sha256(canonical_json(payload).encode()).hexdigest() != expected:
            raise ValueError("Assistant artifact SHA-256 mismatch")
        metadata = dict(payload["metadata"])
        sizes = {key: metadata.pop(key) for key in ("sample_count", "token_count", "predicted_token_count")}
        result = cls(**metadata)
        result.metadata.update(sizes)
        for context, total, pairs in payload["contexts"]:
            key = tuple(context)
            counts = Counter(dict(pairs))
            if key in result.counts or len(pairs) != len(counts) or sum(counts.values()) != total:
                raise ValueError("Invalid assistant context counts")
            result.counts[key] = counts
            result.totals[key] = total
        if sum(result.totals.values()) != sizes["predicted_token_count"]:
            raise ValueError("Assistant count total differs from metadata")
        return result


def window_union(centers, radius):
    """Expand each center and clip to the sequence, preserving boolean union."""
    if radius < 0:
        raise ValueError("Window radius must be nonnegative")
    expanded = [False] * len(centers)
    for pos, selected in enumerate(centers):
        if selected:
            start = max(0, pos - radius)
            end = min(len(centers), pos + radius + 1)
            expanded[start:end] = [True] * (end - start)
    return expanded


def paper_mask(scores, center_candidates, eligible, threshold, radius):
    if not 0 <= threshold <= 1 or len(scores) != len(center_candidates) or len(scores) != len(eligible):
        raise ValueError("Invalid paper mask input")
    centers = [bool(candidate and score is not None and score > threshold)
               for score, candidate in zip(scores, center_candidates)]
    expanded = window_union(centers, radius)
    return centers, expanded, [window and valid for window, valid in zip(expanded, eligible)]


def score_matched_mask(scores, eligible, count):
    if len(scores) != len(eligible):
        raise ValueError("Scores and eligibility must align")
    candidates = [pos for pos, valid in enumerate(eligible) if valid]
    if not 0 <= count <= len(candidates) or any(scores[pos] is None for pos in candidates):
        raise ValueError("Matched count or eligible scores are invalid")
    # Stable, ascending target position breaks ties in a deterministic way.
    candidates.sort(key=lambda pos: (-scores[pos], pos))
    chosen = set(candidates[:count])
    return [pos in chosen for pos in range(len(scores))]
