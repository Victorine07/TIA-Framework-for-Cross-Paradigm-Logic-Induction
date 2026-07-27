#!/usr/bin/env python3
"""
text_normalization.py

Shared text cleanup helpers for Python -> Isabelle/HOL evaluation.

Purpose:
- centralize cleanup logic used by zero-shot / few-shot / fine-tuned evaluators
- extract the most likely Isabelle/HOL code span from raw model generations
- keep normalization conservative so metrics still measure real differences


Implements the output normalization step shared across all evaluation modes in Section 4: Evaluation Framework.

"""

from __future__ import annotations

import re
from typing import List


FENCE_PATTERN = re.compile(r"```(?:isabelle|hol|Isabelle)?", re.IGNORECASE)
COMMENT_PATTERN = re.compile(r"\(\*.*?\*\)", re.DOTALL)


def clean_code(code_str: str) -> str:
    if not code_str:
        return ""

    code_str = COMMENT_PATTERN.sub("", code_str)
    code_str = FENCE_PATTERN.sub("", code_str)
    code_str = re.sub(r"```", "", code_str)

    lines = [line.rstrip() for line in code_str.splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines).strip()


def _compact_nonempty_lines(text: str) -> str:
    lines: List[str] = []
    for line in text.splitlines():
        if line.strip():
            lines.append(line)
    return clean_code("\n".join(lines))


def extract_isabelle_code(text: str) -> str:
    if not text:
        return ""

    text = text.strip()
    text = FENCE_PATTERN.sub("", text)
    text = re.sub(r"```", "", text)

    patterns = [
        r"(definition.*?end)",
        r"(fun.*?end)",
        r"(lemma.*?end)",
        r"(primrec.*?end)",
        r"(abbreviation.*?end)",
        r"(definition.*?(?=\ndefinition|\nfun|\nlemma|\nprimrec|\nabbreviation|\Z))",
        r"(fun.*?(?=\ndefinition|\nfun|\nlemma|\nprimrec|\nabbreviation|\Z))",
        r"(lemma.*?(?=\ndefinition|\nfun|\nlemma|\nprimrec|\nabbreviation|\Z))",
        r"(primrec.*?(?=\ndefinition|\nfun|\nlemma|\nprimrec|\nabbreviation|\Z))",
        r"(abbreviation.*?(?=\ndefinition|\nfun|\nlemma|\nprimrec|\nabbreviation|\Z))",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            if isinstance(matches[0], tuple):
                pieces = ["".join(m).strip() for m in matches]
                return _compact_nonempty_lines("\n\n".join(p for p in pieces if p))
            return _compact_nonempty_lines(matches[0].strip())

    isabelle_like = []
    started = False
    starters = ("definition", "fun", "lemma", "abbreviation", "primrec", "function")
    continuations = (
        "where",
        "end",
        "by",
        "let",
        "if",
        "then",
        "else",
        "case",
        "of",
        "|",
    )

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            if started:
                isabelle_like.append(line)
            continue

        if stripped.startswith(starters):
            started = True
            isabelle_like.append(line)
            continue

        if started:
            if (
                "::" in stripped
                or stripped.startswith(continuations)
                or stripped.startswith('"')
                or "⇒" in stripped
                or "=>" in stripped
                or "=" in stripped
            ):
                isabelle_like.append(line)
            else:
                break

    if isabelle_like:
        return clean_code("\n".join(isabelle_like))

    return clean_code(text)


def clean_generation(text: str) -> str:
    if not text:
        return ""

    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    stop_markers = [
        "### Task:",
        "### Input:",
        "### Output:",
        "### Instruction:",
        "Task:",
        "Input:",
        "Output:",
        "Instruction:",
    ]
    cut_positions = [text.find(marker) for marker in stop_markers if marker in text]
    cut_positions = [p for p in cut_positions if p > 0]
    if cut_positions:
        text = text[: min(cut_positions)].strip()

    return extract_isabelle_code(text)
