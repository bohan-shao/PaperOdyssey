"""Which bracketed numbers in an answer are citations; shared by paper QA, the UI and the answer judge.

Answers cite the supplied sources as [n], numbered from 1 (SCHEMA.md §10). Grouped forms such as
[1, 2], [1-3] or [01] are rewritten as [1][2], [1][2][3], [1]. A bracket whose numbers are not all
supplied sources (the interval [0, 1], a year range) stays plain text, and so do a lone [0] and
index notation: brackets attached to a name, as in X[0], W_2[1], x^{[0]} or A[i][1] (D50).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator

# Brackets right after an ASCII letter, digit, "_", "^" or "{", with the brackets chained to them
# (A[i][1]). ASCII only: Chinese answers attach citations to the text without a space ("方法[1]").
_NOTATION = re.compile(r"(?<=[A-Za-z0-9_^{])(?:\[[^\[\]\n]*\])+")
_NUMERIC_BRACKET = re.compile(r"\[\s*\d[\d\s,;\-–]*\]")
_SINGLE = re.compile(r"\[\s*(\d+)\s*\]")
_RANGE = re.compile(r"\s*(\d+)\s*[-–]\s*(\d+)\s*")
_NUMBERS = re.compile(r"\s*\d+(?:\s+\d+)*\s*")
_CITATION = re.compile(r"(\s*)\[([1-9]\d*)\]")  # a normalized citation and the whitespace before it


def _outside_notation(pattern: re.Pattern[str], text: str) -> Iterator[re.Match[str]]:
    start = 0
    for notation in _NOTATION.finditer(text):
        yield from pattern.finditer(text, start, notation.start())
        start = notation.end()
    yield from pattern.finditer(text, start)


def _sub(pattern: re.Pattern[str], text: str, repl: Callable[[re.Match[str]], str]) -> str:
    parts, end = [], 0
    for match in _outside_notation(pattern, text):
        parts += [text[end:match.start()], repl(match)]
        end = match.end()
    return "".join(parts) + text[end:]


def _group_numbers(inner: str, source_count: int) -> list[int] | None:
    """Sources named by a bracket body such as "1, 2" or "1-3"; None if it is not a citation.

    Every number must be a supplied source, so an interval such as [0, 1] stays plain text.
    """
    numbers: list[int] = []
    for part in re.split(r"[,;]", inner):
        if bounds := _RANGE.fullmatch(part):
            low, high = int(bounds[1]), int(bounds[2])
            if not 1 <= low <= high <= source_count:
                return None
            numbers.extend(range(low, high + 1))
        elif _NUMBERS.fullmatch(part):
            values = [int(n) for n in part.split()]
            if not all(1 <= n <= source_count for n in values):
                return None
            numbers.extend(values)
        else:
            return None
    return numbers


def normalize_citations(answer: str, source_count: int) -> str:
    """Rewrite [1, 2] / [1-3] / [01] as [1][2] / [1][2][3] / [1]; keep notation and other brackets."""
    def rewrite(match: re.Match[str]) -> str:
        if single := _SINGLE.fullmatch(match[0]):
            number = int(single[1])  # a lone number is a citation unless 0; bounds checked later
            return f"[{number}]" if number else match[0]
        numbers = _group_numbers(match[0][1:-1], source_count)
        return match[0] if numbers is None else "".join(f"[{n}]" for n in dict.fromkeys(numbers))

    return _sub(_NUMERIC_BRACKET, answer, rewrite)


def cited_numbers(answer: str) -> list[int]:
    """Distinct source numbers cited by a normalized answer, in first-appearance order."""
    return list(dict.fromkeys(int(match[2]) for match in _outside_notation(_CITATION, answer)))


def sub_citations(answer: str, repl: Callable[[re.Match[str]], str]) -> str:
    """`re.sub` over the citations of a normalized answer (`DownstreamResult.answer`).

    Each match is one [n]: group 1 is the whitespace before it, group 2 the source number.
    Index notation and other brackets are left as they are.
    """
    return _sub(_CITATION, answer, repl)
