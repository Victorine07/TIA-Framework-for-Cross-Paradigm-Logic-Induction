# base_extractor.py

from __future__ import annotations

import ast
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


SCHEMA_VERSION = "tia-jsonl-v2"

REQUIRED_TOP_LEVEL_FIELDS: Tuple[str, ...] = (
    "schema_version",
    "record_id",
    "dataset_split",
    "task",
    "cipher",
    "family",
    "subfamily",
    "variant",
    "tier",
    "tier_role",
    "component_type",
    "component_name",
    "alignment_kind",
    "semantic_group",
    "difficulty",
    "identity",
    "source_pair",
    "local_semantics",
    "structural_context",
    "translation_contract",
    "prompt_metadata",
    "provenance",
    "quality",
    "labels",
    "instruction",
)

VALID_TIERS = {"T1", "T2", "T3", "T4"}
VALID_SPLITS = {"train", "valid", "test"}
VALID_DIFFICULTIES = {"easy", "medium", "hard"}
VALID_SOURCE_KINDS = {
    "constant", "function", "method", "class", "definition", "fun", "primrec",
}
VALID_ALIGNMENT_KINDS = {
    "direct_constant_alignment", "direct_symbol_alignment", "functional_alignment",
    "structural_alignment", "orchestration_alignment",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_cipher_name(cipher: str) -> str:
    return cipher.strip().lower().replace("-", "_").replace(" ", "_")


def title_cipher_name(cipher: str) -> str:
    parts = normalize_cipher_name(cipher).split("_")
    return "_".join(part.capitalize() for part in parts)


def sanitize_code_block(code: str) -> str:
    return code.strip().replace("\r\n", "\n").replace("\r", "\n")


def first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


@dataclass
# Corresponds to Algorithm 1 (Section 3: Tiered Isomorphic Alignment - Dataset Construction and Registry): the aligned instruction-input-output record
# produced for each (p_i, i_j) pair. Fields encode tier label, parametric metadata,
# and source/target code for one Python ↔ Isabelle/HOL component pair.
class JsonlRecord:
    schema_version: str
    record_id: str
    dataset_split: str
    task: str
    cipher: str
    family: str
    subfamily: str
    variant: Dict[str, Any]
    tier: str
    tier_role: str
    component_type: str
    component_name: str
    alignment_kind: str
    semantic_group: str
    difficulty: str
    identity: Dict[str, Any]
    source_pair: Dict[str, Any]
    local_semantics: Dict[str, Any]
    structural_context: Dict[str, Any]
    translation_contract: Dict[str, Any]
    prompt_metadata: Dict[str, Any]
    provenance: Dict[str, Any]
    quality: Dict[str, Any]
    labels: Dict[str, Any]
    instruction: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        validate_record_dict(data)
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


def validate_record_dict(data: Dict[str, Any]) -> None:
    missing = [field for field in REQUIRED_TOP_LEVEL_FIELDS if field not in data]
    if missing:
        raise ValueError(f"Missing required top-level fields: {missing}")

    for field in REQUIRED_TOP_LEVEL_FIELDS:
        if data[field] is None:
            raise ValueError(f"Required top-level field cannot be None: {field}")

    if data["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}, got {data['schema_version']}")

    if data["dataset_split"] not in VALID_SPLITS:
        raise ValueError(f"Invalid dataset_split: {data['dataset_split']}")

    if data["tier"] not in VALID_TIERS:
        raise ValueError(f"Invalid tier: {data['tier']}")

    if data["difficulty"] not in VALID_DIFFICULTIES:
        raise ValueError(f"Invalid difficulty: {data['difficulty']}")

    if data["alignment_kind"] not in VALID_ALIGNMENT_KINDS:
        raise ValueError(f"Invalid alignment_kind: {data['alignment_kind']}")


class BaseCipherExtractor(ABC):
    # Implements the registry-driven extractor described in Section 3: Tiered Isomorphic Alignment
    # - Dataset Construction and Registry; instantiated per cipher family via Algorithm 1 (same section).
    # Subclasses override extract_components() to produce T1–T4 aligned pairs.
    """
    Flexible base extractor for all cipher families.
    No longer forces SPARX-specific keys.
    """

    def __init__(
        self,
        root_dir: str | Path,
        cipher: str,
        family: str,
        subfamily: str,
        block_size: int,
        key_size: int,
        variant_config: Dict[str, Any],
        dataset_split: str = "train",
    ) -> None:
        self.root_dir = Path(root_dir).resolve()
        self.cipher = normalize_cipher_name(cipher)
        self.family = family
        self.subfamily = subfamily
        self.block_size = int(block_size)
        self.key_size = int(key_size)
        self.dataset_split = dataset_split

        if self.dataset_split not in VALID_SPLITS:
            raise ValueError(f"Invalid dataset_split: {self.dataset_split}")

        self.variant_config = self._normalize_variant_config(variant_config)

        # Directory structure
        self.python_ciphers_dir = self.root_dir / "python ciphers"
        self.thy_ciphers_dir = self.root_dir / "thy ciphers"

        # File paths (to be resolved by subclasses if needed)
        self.python_file = None
        self.thy_file = None
        self.python_source = ""
        self.thy_source = ""
        self.python_ast = None

    def set_source_files(self, py_path: Path, thy_path: Path) -> None:
        """Set source files and load content."""
        self.python_file = py_path
        self.thy_file = thy_path
        self.python_source = self._read_text_file(py_path)
        self.thy_source = self._read_text_file(thy_path)
        self.python_ast = self._safe_parse_python(self.python_source)

    def _normalize_variant_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Flexible normalization - preserves all keys, adds defaults only for missing required ones.
        """
        normalized = dict(config)
        
        # Required metadata keys (with intelligent defaults)
        # These are for metadata only - extraction logic uses the actual values
        defaults = {
            "word_size": config.get("word_size", 16),
            "branches": config.get("branches", config.get("n_branches", 2)),
            "words_per_block": config.get("words_per_block", config.get("n_words", 2)),
            "steps": config.get("steps", config.get("n_steps", 1)),
            "rounds_per_step": config.get("rounds_per_step", config.get("rounds", 1)),
            "total_rounds": config.get("total_rounds", config.get("rounds", 1)),
        }
        
        # Only add missing keys
        for key, default_value in defaults.items():
            if key not in normalized:
                normalized[key] = default_value
        
        normalized["name"] = f"{self.block_size}_{self.key_size}"
        normalized["block_size"] = self.block_size
        normalized["key_size"] = self.key_size
        
        return normalized

    def _read_text_file(self, path: Path) -> str:
        try:
            return sanitize_code_block(path.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            return sanitize_code_block(path.read_text(encoding="utf-8", errors="ignore"))

    def _safe_parse_python(self, source: str) -> Optional[ast.AST]:
        try:
            return ast.parse(source)
        except SyntaxError:
            return None

    def variant_dict(self) -> Dict[str, Any]:
        return {
            "name": self.variant_config.get("name", f"{self.block_size}_{self.key_size}"),
            "block_size": self.block_size,
            "key_size": self.key_size,
            "word_size": self.variant_config.get("word_size"),
            "branches": self.variant_config.get("branches"),
            "words_per_block": self.variant_config.get("words_per_block"),
            "steps": self.variant_config.get("steps"),
            "rounds_per_step": self.variant_config.get("rounds_per_step"),
            "total_rounds": self.variant_config.get("total_rounds"),
        }

    def make_record_id(self, tier: str, component_name: str) -> str:
        return f"{self.cipher}::{self.variant_dict()['name']}::{tier}::{component_name}"

    # ========== Extraction Helpers ==========

    def extract_python_function(self, function_name: str) -> Optional[str]:
        # Implements DecomposePython(c, τ) from Algorithm 1 (Section 3: Tiered Isomorphic Alignment - Dataset Construction and Registry): isolates
        # a tier-specific Python component from the cipher source file.
        """Extract a top-level function from Python source."""
        if self.python_ast is not None:
            result = self._extract_python_function_ast(function_name)
            if result:
                return sanitize_code_block(result)
        return self._extract_python_function_regex(function_name)

    def _extract_python_function_ast(self, function_name: str) -> Optional[str]:
        assert self.python_ast is not None
        for node in ast.walk(self.python_ast):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                return ast.get_source_segment(self.python_source, node)
            if isinstance(node, ast.AsyncFunctionDef) and node.name == function_name:
                return ast.get_source_segment(self.python_source, node)
        return None

    def _extract_python_function_regex(self, function_name: str) -> Optional[str]:
        lines = self.python_source.splitlines()
        start_idx = None
        base_indent = None

        for i, line in enumerate(lines):
            stripped = line.lstrip()
            if stripped.startswith(f"def {function_name}(") or stripped.startswith(f"async def {function_name}("):
                start_idx = i
                base_indent = len(line) - len(stripped)
                break

        if start_idx is None:
            return None

        collected: List[str] = []
        for j in range(start_idx, len(lines)):
            line = lines[j]
            stripped = line.strip()
            current_indent = len(line) - len(line.lstrip())

            if j > start_idx and stripped:
                if current_indent <= (base_indent or 0) and not line.lstrip().startswith("@"):
                    if line.lstrip().startswith("def ") or line.lstrip().startswith("async def ") or line.lstrip().startswith("class "):
                        break

            collected.append(line)

        code = "\n".join(collected).rstrip()
        return sanitize_code_block(code) if code.strip() else None

    def extract_python_assignment(self, symbol_name: str) -> Optional[str]:
        pattern = rf"(?m)^\s*{re.escape(symbol_name)}\s*=\s*.+$"
        match = re.search(pattern, self.python_source)
        return sanitize_code_block(match.group(0)) if match else None

    def extract_isabelle_definition(self, symbol_name: str) -> Optional[str]:
        # Implements MapIsabelle(t, τ) from Algorithm 1 (Section 3: Tiered Isomorphic Alignment - Dataset Construction and Registry): extracts the
        # corresponding Isabelle/HOL definition for 1-to-1 alignment with the Python
        # component at the same tier.
        """Extract Isabelle definition by name."""
        lines = self.thy_source.splitlines()
        start_idx = None

        start_pattern = re.compile(rf"^\s*(definition|fun|function|primrec)\s+{re.escape(symbol_name)}\b")

        for i, line in enumerate(lines):
            if start_pattern.search(line):
                start_idx = i
                break

        if start_idx is None:
            return None

        collected: List[str] = []
        paren_balance = 0
        bracket_balance = 0
        quote_count = 0

        for i in range(start_idx, len(lines)):
            line = lines[i]
            stripped = line.strip()

            if i > start_idx:
                if re.match(r"^\s*(definition|fun|function|primrec|lemma|theorem|section|subsection)\b", stripped):
                    if paren_balance == 0 and bracket_balance == 0 and quote_count % 2 == 0:
                        break

            collected.append(line)
            paren_balance += line.count("(") - line.count(")")
            bracket_balance += line.count("[") - line.count("]")
            quote_count += line.count('"')

        code = sanitize_code_block("\n".join(collected))
        return code if code else None

    def find_isabelle_definition_flexible(self, *possible_names: str) -> Optional[Tuple[str, str]]:
        """Try multiple possible Isabelle definition names."""
        for name in possible_names:
            code = self.extract_isabelle_definition(name)
            if code:
                return name, code
        return None

    # ========== Semantic Analysis ==========

    def analyze_python_semantics(self, code: str) -> Dict[str, Any]:
        # Implements ExtractParametricMetadata(c) from Algorithm 1 (Section 3: Tiered Isomorphic Alignment - Dataset Construction and Registry):
        # computes semantic operator labels, structural flags (loops, recursion,
        # conditionals), and transformation patterns used to enrich S_meta.
        lowered = code.lower()
        try:
            contains_loop = any(isinstance(node, (ast.For, ast.While)) for node in ast.walk(ast.parse(code)))
        except SyntaxError:
            # Fall back to a line-anchored heuristic (avoids matching "for"/"while"
            # appearing as prose inside a docstring or comment).
            contains_loop = bool(re.search(r"^\s*(for|while)\b", code, re.MULTILINE))
        return {
            "operators": [op for op in ("xor", "rotate", "shift", "mask", "add", "sub", "mod", "list", "slice") if op in lowered],
            "contains_loop": contains_loop,
            "contains_recursion": self._contains_self_call(code),
            "contains_conditionals": any(token in lowered for token in ("if ", "elif ", "else:")),
            "contains_bitwise": any(token in code for token in ("^", "&", "|", "<<", ">>")),
            "contains_arithmetic": any(token in code for token in ("+", "-", "*", "%")),
            "line_count": len([line for line in code.splitlines() if line.strip()]),
            "first_line": first_nonempty_line(code),
        }

    def _contains_self_call(self, code: str) -> bool:
        first = first_nonempty_line(code)
        m = re.match(r"^(?:async\s+def|def)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", first)
        if not m:
            return False
        name = m.group(1)
        body = "\n".join(code.splitlines()[1:])
        return re.search(rf"\b{re.escape(name)}\s*\(", body) is not None

    # ========== Abstract Methods ==========

    @abstractmethod
    def extract_components(self) -> List[JsonlRecord]:
        # Abstract entry point for Algorithm 1 (Section 3: Tiered Isomorphic Alignment - Dataset Construction and Registry): subclasses implement
        # the full loop over tiers T1–T4, producing aligned (instr, p_i, i_j, S_meta)
        # records for every component of the cipher.
        """Extract all components - must be implemented by subclass."""
        raise NotImplementedError
        
