"""Deterministic private translation utilities for Objective 6 English v2."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Mapping


PAD_CHEST_6 = (
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Pneumothorax",
)

_SPANISH_TERMS: Mapping[str, tuple[str, ...]] = {
    "Atelectasis": ("atelectasia", "atelectasias"),
    "Cardiomegaly": ("cardiomegalia",),
    "Consolidation": ("consolidacion", "condensacion", "consolidaciones"),
    "Edema": ("edema pulmonar", "edema"),
    "Effusion": ("derrame pleural", "derrames pleurales"),
    "Pneumothorax": ("neumotorax",),
}

_ENGLISH_TERMS: Mapping[str, tuple[str, ...]] = {
    "Atelectasis": ("atelectasis", "atelectatic", "subsegmental collapse"),
    "Cardiomegaly": (
        "cardiomegaly",
        "enlarged cardiac silhouette",
        "cardiac enlargement",
        "enlarged heart",
    ),
    "Consolidation": ("consolidation", "airspace opacity", "air-space opacity"),
    "Edema": ("pulmonary edema", "oedema", "edema"),
    "Effusion": ("pleural effusion", "pleural fluid"),
    "Pneumothorax": ("pneumothorax", "pneumothoraces"),
}

_SPANISH_NEGATIONS = (
    "no", "sin", "ni", "ausencia de", "ausente", "descarta",
    "descartado", "descartada", "no se observa", "no se observan",
    "sin signos de", "sin evidencia de",
)

_ENGLISH_NEGATIONS = (
    "no", "not", "without", "neither", "absence of", "absent",
    "negative for", "free of", "no evidence of", "no sign of",
)

_SPANISH_POST_NEGATIONS = (
    "ausente", "ausentes", "no observado", "no observada", "no observados",
    "no observadas", "descartado", "descartada", "descartados", "descartadas",
)

_ENGLISH_POST_NEGATIONS = (
    "is absent", "are absent", "is not seen", "are not seen", "not seen",
    "is not identified", "are not identified", "not identified", "is not present",
    "are not present", "not present", "is excluded", "are excluded",
)

_NUMBER = re.compile(r"(?<!\w)[+-]?\d+(?:[.,]\d+)?(?!\w)")

_ENGLISH_MARKERS = re.compile(
    r"\b(the|without|with|and|of|is|are|was|were|lung|heart|pleural)\b",
    flags=re.IGNORECASE,
)

_SPANISH_MARKERS = re.compile(
    r"\b(el|la|los|las|sin|con|y|de|se|pulmon|pulmonar|corazon)\b",
    flags=re.IGNORECASE,
)

_CANONICAL_ENGLISH = {
    "Atelectasis": "atelectasis",
    "Cardiomegaly": "cardiomegaly",
    "Consolidation": "consolidation",
    "Edema": "pulmonary edema",
    "Effusion": "pleural effusion",
    "Pneumothorax": "pneumothorax",
}


def normalise_translation_text(value: object) -> str:
    """Normalize Unicode and whitespace without changing clinical content."""

    text = unicodedata.normalize("NFC", str(value))
    return " ".join(text.split()).strip()


def private_report_sha256(value: object) -> str:
    """Return the private normalized-report key used for de-duplication."""

    text = normalise_translation_text(value)
    if not text:
        raise ValueError("Objective 6 v2 cannot translate an empty report")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _fold(value: object) -> str:
    text = normalise_translation_text(value).casefold()
    return "".join(
        character
        for character in unicodedata.normalize("NFD", text)
        if unicodedata.category(character) != "Mn"
    )


def _polarity(
    value: object,
    terms: Mapping[str, tuple[str, ...]],
    negations: tuple[str, ...],
    post_negations: tuple[str, ...],
) -> dict[str, int]:
    text = _fold(value)
    output: dict[str, int] = {}
    for label in PAD_CHEST_6:
        best: tuple[int, int] | None = None
        for term in terms[label]:
            folded_term = _fold(term)
            pattern = re.compile(rf"(?<!\w){re.escape(folded_term)}(?!\w)")
            match = pattern.search(text)
            if match is not None and (best is None or match.start() < best[0]):
                best = (match.start(), match.end())
        if best is None:
            continue
        index, end = best
        # A generous character window handles multiword radiology negations
        # while remaining inside the current sentence/clause.
        boundary = max(text.rfind(".", 0, index), text.rfind(";", 0, index))
        prefix = text[max(boundary + 1, index - 64) : index].strip()
        next_periods = [
            boundary
            for boundary in (text.find(".", end), text.find(";", end))
            if boundary >= 0
        ]
        clause_end = min(next_periods) if next_periods else len(text)
        suffix = text[end : min(clause_end, end + 48)].strip()

        def contains_phrase(window: str, phrase: str) -> bool:
            return re.search(
                rf"(?<!\w){re.escape(_fold(phrase))}(?!\w)", window
            ) is not None

        negated_before = any(contains_phrase(prefix, cue) for cue in negations)
        negated_after = any(contains_phrase(suffix, cue) for cue in post_negations)
        output[label] = 0 if negated_before or negated_after else 1
    return output


def spanish_concept_polarity(value: object) -> dict[str, int]:
    """Extract explicitly mentioned PadChest-6 concept polarity in Spanish."""

    return _polarity(
        value, _SPANISH_TERMS, _SPANISH_NEGATIONS, _SPANISH_POST_NEGATIONS
    )


def english_concept_polarity(value: object) -> dict[str, int]:
    """Extract explicitly mentioned PadChest-6 concept polarity in English."""

    return _polarity(
        value, _ENGLISH_TERMS, _ENGLISH_NEGATIONS, _ENGLISH_POST_NEGATIONS
    )


def normalized_numbers(value: object) -> tuple[str, ...]:
    """Extract normalized numeric values while tolerating decimal punctuation."""

    numbers: list[str] = []
    for match in _NUMBER.findall(normalise_translation_text(value)):
        number = match.replace(",", ".")
        try:
            number = f"{float(number):g}"
        except ValueError:
            pass
        numbers.append(number)
    return tuple(numbers)


def numbers_preserved(source: object, translation: object) -> bool:
    """Return whether every source numeric value survives the translation."""

    source_numbers = normalized_numbers(source)
    target_numbers = list(normalized_numbers(translation))
    for number in source_numbers:
        if number not in target_numbers:
            return False
        target_numbers.remove(number)
    return True


def concept_polarity_counts(source: object, translation: object) -> tuple[int, int]:
    """Return matching and eligible explicitly mentioned concept counts."""

    source_values = spanish_concept_polarity(source)
    target_values = english_concept_polarity(translation)
    eligible = len(source_values)
    matches = sum(target_values.get(label) == value for label, value in source_values.items())
    return matches, eligible


def language_marker_scores(value: object) -> tuple[int, int]:
    """Return English and Spanish marker counts without inferring clinical content."""

    text = _fold(value)
    return len(_ENGLISH_MARKERS.findall(text)), len(_SPANISH_MARKERS.findall(text))


def shield_numeric_tokens(value: object) -> tuple[str, dict[str, str]]:
    """Replace source numeric tokens with deterministic translation sentinels."""

    text = normalise_translation_text(value)
    mapping: dict[str, str] = {}
    parts: list[str] = []
    cursor = 0
    for index, match in enumerate(_NUMBER.finditer(text)):
        sentinel = f"ZXQNUMTOKEN{index:03d}QXZ"
        parts.extend((text[cursor : match.start()], sentinel))
        mapping[sentinel] = match.group(0)
        cursor = match.end()
    parts.append(text[cursor:])
    return "".join(parts), mapping


def restore_numeric_tokens(
    value: object, mapping: Mapping[str, str]
) -> tuple[str, tuple[str, ...]]:
    """Restore numeric sentinels, tolerating tokenizer-inserted spaces and case."""

    text = normalise_translation_text(value)
    missing: list[str] = []
    for sentinel, number in mapping.items():
        index = int(re.search(r"(\d{3})", sentinel).group(1))  # type: ignore[union-attr]
        pattern = re.compile(
            rf"z\s*x\s*q\s*num\s*token\s*0*{index}\s*q\s*x\s*z",
            flags=re.IGNORECASE,
        )
        text, replacements = pattern.subn(number, text, count=1)
        if replacements == 0:
            missing.append(sentinel)
    return normalise_translation_text(text), tuple(missing)


def enforce_source_concept_polarity(source: object, translation: object) -> str:
    """Ensure source-explicit concepts have one consistent English polarity."""

    source_values = spanish_concept_polarity(source)
    if not source_values:
        return normalise_translation_text(translation)
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?;])\s+", normalise_translation_text(translation))
        if sentence.strip()
    ]
    retained: list[str] = []
    represented: set[str] = set()
    for sentence in sentences:
        target_values = english_concept_polarity(sentence)
        conflicts = {
            label
            for label, polarity in target_values.items()
            if label in source_values and source_values[label] != polarity
        }
        if conflicts:
            continue
        represented.update(
            label
            for label, polarity in target_values.items()
            if source_values.get(label) == polarity
        )
        retained.append(sentence)
    for label, polarity in source_values.items():
        if label in represented:
            continue
        term = _CANONICAL_ENGLISH[label]
        retained.append(f"There is {term}." if polarity else f"There is no {term}.")
    return normalise_translation_text(" ".join(retained))
