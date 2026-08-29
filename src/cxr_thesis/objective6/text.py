"""Deterministic, training-only text processing for Spanish CXR reports."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
import unicodedata
from collections.abc import Iterable, Sequence


PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
SPECIAL_TOKENS = (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN)

_TOKEN_PATTERN = re.compile(
    r"[^\W\d_]+(?:['’][^\W\d_]+)?|\d+(?:[.,]\d+)?|[.,;:!?()/%+\-]",
    flags=re.UNICODE,
)


def normalise_report(value: object) -> str:
    """Normalise spacing/case without translating or inventing report content."""

    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).lower().strip()
    if text in {"", "nan", "none", "null"}:
        return ""
    return " ".join(text.split())


def tokenise_report(value: object) -> list[str]:
    """Tokenise Unicode Spanish text and retain clinically useful punctuation."""

    return _TOKEN_PATTERN.findall(normalise_report(value))


@dataclass(frozen=True)
class ReportVocabulary:
    """A deterministic vocabulary fitted on training reports only."""

    tokens: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.tokens[: len(SPECIAL_TOKENS)] != SPECIAL_TOKENS:
            raise ValueError("Vocabulary must begin with the four special tokens")
        if len(set(self.tokens)) != len(self.tokens):
            raise ValueError("Vocabulary tokens must be unique")

    @classmethod
    def build(
        cls,
        reports: Iterable[object],
        *,
        minimum_frequency: int = 2,
        maximum_size: int = 12000,
    ) -> ReportVocabulary:
        if minimum_frequency <= 0:
            raise ValueError("minimum_frequency must be positive")
        if maximum_size <= len(SPECIAL_TOKENS):
            raise ValueError("maximum_size is too small")
        counts: Counter[str] = Counter()
        for report in reports:
            counts.update(tokenise_report(report))
        candidates = [
            token
            for token, frequency in counts.items()
            if frequency >= minimum_frequency and token not in SPECIAL_TOKENS
        ]
        candidates.sort(key=lambda token: (-counts[token], token))
        room = maximum_size - len(SPECIAL_TOKENS)
        return cls(tuple(SPECIAL_TOKENS) + tuple(candidates[:room]))

    @property
    def token_to_id(self) -> dict[str, int]:
        return {token: index for index, token in enumerate(self.tokens)}

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def bos_id(self) -> int:
        return 1

    @property
    def eos_id(self) -> int:
        return 2

    @property
    def unk_id(self) -> int:
        return 3

    def encode(self, report: object, *, maximum_length: int) -> list[int]:
        if maximum_length < 2:
            raise ValueError("maximum_length must leave room for BOS and EOS")
        lookup = self.token_to_id
        content = [lookup.get(token, self.unk_id) for token in tokenise_report(report)]
        return [self.bos_id, *content[: maximum_length - 2], self.eos_id]

    def decode(self, identifiers: Sequence[int]) -> str:
        output: list[str] = []
        for identifier in identifiers:
            index = int(identifier)
            if index < 0 or index >= len(self.tokens):
                token = UNK_TOKEN
            else:
                token = self.tokens[index]
            if token == EOS_TOKEN:
                break
            if token not in {PAD_TOKEN, BOS_TOKEN}:
                output.append(token)
        text = " ".join(output)
        return re.sub(r"\s+([.,;:!?%)])", r"\1", text).strip()

    def to_dict(self) -> dict[str, object]:
        return {"format_version": 1, "tokens": list(self.tokens)}

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> ReportVocabulary:
        if payload.get("format_version") != 1:
            raise ValueError("Unsupported vocabulary format")
        tokens = payload.get("tokens")
        if not isinstance(tokens, list) or not all(
            isinstance(token, str) for token in tokens
        ):
            raise ValueError("Vocabulary tokens are invalid")
        return cls(tuple(tokens))
