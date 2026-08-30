"""Deterministic Spanish report-generation metrics for Objective 6."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence
from itertools import pairwise

import numpy as np

from .text import normalise_report, tokenise_report

PAD_CHEST_6 = (
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Pneumothorax",
)

_LABEL_TERMS = {
    "Atelectasis": ("atelectasis", "atelectasia"),
    "Cardiomegaly": ("cardiomegaly", "cardiomegalia"),
    "Consolidation": ("consolidation", "consolidacion", "condensacion"),
    "Edema": ("edema", "pulmonary edema", "edema pulmonar"),
    "Effusion": ("effusion", "pleural effusion", "derrame pleural"),
    "Pneumothorax": ("pneumothorax", "neumotorax"),
}
_NEGATION_CUES = (
    "no", "sin", "ausencia de", "ausente", "descarta", "descartado",
    "descartada", "ni", "libre de", "sin signos de", "no se observa",
    "no se observan", "no evidencia de",
)


def _strip_accents(value: object) -> str:
    import unicodedata

    text = normalise_report(value)
    return "".join(
        character for character in unicodedata.normalize("NFD", text)
        if unicodedata.category(character) != "Mn"
    )


def parse_padchest6_labels(value: object) -> np.ndarray:
    """Map a private PadChest label string to the six locked binary targets."""

    text = _strip_accents(value)
    return np.asarray(
        [int(any(term in text for term in _LABEL_TERMS[label])) for label in PAD_CHEST_6],
        dtype=np.int8,
    )


def ngrams(tokens: Sequence[str], order: int) -> Counter[tuple[str, ...]]:
    if order <= 0:
        raise ValueError("N-gram order must be positive")
    return Counter(tuple(tokens[index : index + order]) for index in range(len(tokens) - order + 1))


def bleu_statistics(reference: Sequence[str], hypothesis: Sequence[str]) -> np.ndarray:
    """Return clipped/total counts for BLEU orders 1-4 and both lengths."""

    values: list[float] = []
    for order in range(1, 5):
        reference_counts = ngrams(reference, order)
        hypothesis_counts = ngrams(hypothesis, order)
        clipped = sum(min(count, reference_counts[gram]) for gram, count in hypothesis_counts.items())
        values.extend((float(clipped), float(sum(hypothesis_counts.values()))))
    values.extend((float(len(reference)), float(len(hypothesis))))
    return np.asarray(values, dtype=np.float64)


def corpus_bleu(statistics: np.ndarray, maximum_order: int) -> float:
    """Corpus BLEU with deterministic add-one smoothing for zero n-gram counts."""

    if not 1 <= maximum_order <= 4:
        raise ValueError("maximum_order must be in [1, 4]")
    totals = np.asarray(statistics, dtype=np.float64).sum(axis=0)
    precisions = []
    for order in range(maximum_order):
        clipped, total = totals[2 * order : 2 * order + 2]
        precisions.append((clipped + 1.0) / (total + 1.0))
    reference_length, hypothesis_length = totals[-2:]
    if hypothesis_length <= 0:
        return 0.0
    brevity = 1.0 if hypothesis_length > reference_length else math.exp(
        1.0 - reference_length / hypothesis_length
    )
    return float(brevity * math.exp(sum(math.log(value) for value in precisions) / maximum_order))


def rouge_l_f1(reference: Sequence[str], hypothesis: Sequence[str]) -> float:
    if not reference or not hypothesis:
        return 0.0
    previous = [0] * (len(hypothesis) + 1)
    for reference_token in reference:
        current = [0]
        for column, hypothesis_token in enumerate(hypothesis, start=1):
            if reference_token == hypothesis_token:
                current.append(previous[column - 1] + 1)
            else:
                current.append(max(previous[column], current[-1]))
        previous = current
    lcs = previous[-1]
    precision = lcs / len(hypothesis)
    recall = lcs / len(reference)
    return float(2.0 * precision * recall / (precision + recall)) if lcs else 0.0


def exact_token_meteor(reference: Sequence[str], hypothesis: Sequence[str]) -> float:
    """METEOR using exact Unicode-token matches only (no English stem/synonyms)."""

    if not reference or not hypothesis:
        return 0.0
    positions: dict[str, list[int]] = {}
    for index, token in enumerate(reference):
        positions.setdefault(token, []).append(index)
    used: set[int] = set()
    alignment: list[int] = []
    for token in hypothesis:
        candidates = [index for index in positions.get(token, ()) if index not in used]
        if candidates:
            chosen = candidates[0]
            used.add(chosen)
            alignment.append(chosen)
    matches = len(alignment)
    if not matches:
        return 0.0
    precision = matches / len(hypothesis)
    recall = matches / len(reference)
    harmonic = 10.0 * precision * recall / (recall + 9.0 * precision)
    chunks = 1 + sum(
        current != previous + 1 for previous, current in pairwise(alignment)
    )
    penalty = 0.5 * (chunks / matches) ** 3
    return float(harmonic * (1.0 - penalty))


def cider_document_frequency(references: Iterable[Sequence[str]]) -> list[Counter[tuple[str, ...]]]:
    document_frequency = [Counter() for _ in range(4)]
    for tokens in references:
        for order in range(1, 5):
            document_frequency[order - 1].update(ngrams(tokens, order).keys())
    return document_frequency


def _tfidf(
    tokens: Sequence[str], order: int, document_frequency: Counter[tuple[str, ...]], documents: int
) -> dict[tuple[str, ...], float]:
    counts = ngrams(tokens, order)
    total = sum(counts.values())
    if not total:
        return {}
    return {
        gram: (count / total) * math.log(
            max(1.0, documents / max(1, document_frequency[gram]))
        )
        for gram, count in counts.items()
    }


def cider_d_score(
    reference: Sequence[str],
    hypothesis: Sequence[str],
    document_frequency: Sequence[Counter[tuple[str, ...]]],
    documents: int,
    *,
    sigma: float = 6.0,
) -> float:
    if not reference or not hypothesis:
        return 0.0
    similarities = []
    for order in range(1, 5):
        reference_vector = _tfidf(reference, order, document_frequency[order - 1], documents)
        hypothesis_vector = _tfidf(hypothesis, order, document_frequency[order - 1], documents)
        dot = sum(value * reference_vector.get(gram, 0.0) for gram, value in hypothesis_vector.items())
        reference_norm = math.sqrt(sum(value * value for value in reference_vector.values()))
        hypothesis_norm = math.sqrt(sum(value * value for value in hypothesis_vector.values()))
        cosine = dot / (reference_norm * hypothesis_norm) if reference_norm and hypothesis_norm else 0.0
        penalty = math.exp(-((len(hypothesis) - len(reference)) ** 2) / (2.0 * sigma * sigma))
        similarities.append(cosine * penalty)
    return float(10.0 * np.mean(similarities))


def repeated_ngram(tokens: Sequence[str], order: int = 4) -> bool:
    return any(count > 1 for count in ngrams(tokens, order).values())


def clinical_scores(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    reference = np.asarray(reference, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    if reference.shape != prediction.shape or reference.ndim != 2:
        raise ValueError("Clinical arrays must be equal two-dimensional shapes")
    true_positive = np.logical_and(reference, prediction).sum(axis=0).astype(float)
    false_positive = np.logical_and(~reference, prediction).sum(axis=0).astype(float)
    false_negative = np.logical_and(reference, ~prediction).sum(axis=0).astype(float)
    precision = np.divide(true_positive, true_positive + false_positive, out=np.zeros_like(true_positive), where=(true_positive + false_positive) > 0)
    recall = np.divide(true_positive, true_positive + false_negative, out=np.zeros_like(true_positive), where=(true_positive + false_negative) > 0)
    per_label_f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) > 0)
    micro_tp = float(true_positive.sum())
    micro_fp = float(false_positive.sum())
    micro_fn = float(false_negative.sum())
    micro_precision = micro_tp / (micro_tp + micro_fp) if micro_tp + micro_fp else 0.0
    micro_recall = micro_tp / (micro_tp + micro_fn) if micro_tp + micro_fn else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if micro_precision + micro_recall else 0.0
    output = {
        "micro_concept_precision": micro_precision,
        "micro_concept_recall": micro_recall,
        "micro_concept_f1": micro_f1,
        "macro_concept_f1": float(per_label_f1.mean()),
    }
    output.update({f"{label}_f1": float(value) for label, value in zip(PAD_CHEST_6, per_label_f1)})
    return output


def explicit_contradictions(report: object, reference_labels: Sequence[int]) -> tuple[int, int]:
    """Count explicit positive/negative statements that contradict structured labels."""

    tokens = [_strip_accents(token) for token in tokenise_report(report)]
    contradictions = 0
    mentions = 0
    for column, label in enumerate(PAD_CHEST_6):
        terms = _LABEL_TERMS[label]
        statement: int | None = None
        for term in terms:
            term_tokens = [_strip_accents(token) for token in tokenise_report(term)]
            width = len(term_tokens)
            for index in range(len(tokens) - width + 1):
                if tokens[index : index + width] != term_tokens:
                    continue
                prefix = " ".join(tokens[max(0, index - 3) : index])
                negated = any(prefix.endswith(cue) for cue in _NEGATION_CUES)
                statement = 0 if negated else 1
                break
            if statement is not None:
                break
        if statement is not None:
            mentions += 1
            contradictions += int(statement != int(reference_labels[column]))
    return contradictions, mentions
