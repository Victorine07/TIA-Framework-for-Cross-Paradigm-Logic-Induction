from .base_extractor import BaseCipherExtractor
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import re


class ChamExtractor(BaseCipherExtractor):
    # Cipher-specific implementation of Algorithm 1 (Section 3: Tiered Isomorphic Alignment - Dataset Construction and Registry) for the CHAM
    # family (ARX). Produces T1–T4 aligned Python ↔ Isabelle/HOL component pairs.
    """
    Source-aligned CHAM extractor with explicit tier metadata.

    Extracts from Python files like cham_64_128.py, cham_128_128.py,
    cham_128_256.py and matches Isabelle theories like Cham_64_128.thy.

    Test vectors are intentionally ignored: they are not part of the
    aligned extraction target.
    """

    def extract_components(self) -> List:
        """Extract all components organized by tier."""
        examples: List = []

        examples.extend(self._extract_t1_constants())
        examples.extend(self._extract_t2_primitives())
        examples.extend(self._extract_t3_structural_components())
        examples.extend(self._extract_t4_orchestration_components())

        return [ex for ex in examples if ex is not None]

    def _get_variant_prefix(self) -> str:
        """Get the Isabelle/HOL prefix for this variant."""
        return f"cham_{self.block_size}_{self.key_size}"

    def _get_cham_params(self) -> Dict[str, int]:
        """Get CHAM variant parameters."""
        word_size = self.block_size // 4
        key_words = self.key_size // word_size
        block_words = self.block_size // word_size

        rounds_map = {
            (64, 128): 80,
            (128, 128): 112,
            (128, 256): 120,
        }
        rounds = rounds_map.get((self.block_size, self.key_size), 80)

        return {
            "word_size": word_size,
            "key_words": key_words,
            "block_words": block_words,
            "rounds": rounds,
        }

    def _extract_python_function(self, function_name: str) -> Optional[str]:
        """
        Extract a top-level function from Python source.
        Uses flexible regex pattern to handle type hints and spaces.
        """
        lines = self.python_source.split("\n")
        result: List[str] = []
        in_function = False
        indent_level = -1

        pattern = re.compile(rf"^\s*def\s+{re.escape(function_name)}\s*\(")

        for i, line in enumerate(lines):
            stripped = line.strip()
            current_indent = len(line) - len(line.lstrip())

            if not in_function:
                if pattern.match(line):
                    in_function = True
                    indent_level = current_indent
                    result.append(line)
                    continue

            if in_function:
                if i > 0 and stripped:
                    if current_indent <= indent_level and not stripped.startswith("#"):
                        if re.match(r"^\s*def\s+", line) or re.match(r"^\s*class\s+", line):
                            break
                        if stripped and current_indent <= indent_level:
                            break

                result.append(line)

                if stripped.startswith("@") and current_indent == 0:
                    break

        code = "\n".join(result).strip()
        return code if code else None

    def _extract_python_assignment(self, name: str) -> Optional[str]:
        """Extract a top-level assignment from Python source."""
        pattern = re.compile(rf"^\s*{re.escape(name)}\s*=\s*.+$", re.MULTILINE)
        match = pattern.search(self.python_source)
        return match.group(0).strip() if match else None

    def _find_isabelle(self, *possible_names: str) -> Optional[Tuple[str, str]]:
        """Find Isabelle definition by trying multiple names."""
        for name in possible_names:
            code = self.extract_isabelle_definition(name)
            if code:
                return name, code
        return None

    def create_example(
        self,
        template: Dict,
        py_code: str,
        thy_code: str,
        name: str = "",
    ) -> Optional[Dict]:
        """Create a training example from template and code."""
        if not py_code or not thy_code:
            return None

        py_code = py_code.strip()
        thy_code = thy_code.strip()

        semantics = self.analyze_python_semantics(py_code)

        if template.get("force_no_loops"):
            semantics["contains_loop"] = False

        transformations = template.get("transformations", [])
        instruction = self._create_instruction(template, semantics, transformations)
        metadata = self._create_metadata(template, semantics, transformations)

        params = self._get_cham_params()

        metadata.update({
            "tier": template.get("tier"),
            "tier_role": template.get("tier_role"),
            "semantic_name": template.get("semantic_name"),
            "python_symbol": template.get("python_symbol"),
            "python_resolution": template.get("python_resolution"),
            "isabelle_symbol": template.get("isabelle_symbol"),
            "isabelle_resolution": template.get("isabelle_resolution"),
            "component_name": name or template.get("semantic_name") or template.get("type"),
            "cipher": self.cipher,
            "family": self.family,
            "variant": {
                "block_size": self.block_size,
                "key_size": self.key_size,
                **params,
            },
            "source_aligned": True,
            "synthetic_python": template.get("synthetic_python", False),
            "synthetic_isabelle": template.get("synthetic_isabelle", False),
            "extraction_time": datetime.now().isoformat(),
        })

        return {
            "instruction": instruction,
            "input": py_code,
            "output": thy_code,
            "metadata": metadata,
        }

    def _create_instruction(self, template: Dict, semantics: Dict, transformations: List[str]) -> str:
        """Create instruction text for the example."""
        base = (f"Translate the CHAM-{self.block_size}/{self.key_size} "
                f"{template['type']} from Python to Isabelle/HOL.")

        tier = template.get("tier")
        if tier:
            base += f" This is a {tier} component."

        if transformations:
            hints = [self._transformation_to_hint(t) for t in transformations[:3]]
            base += f" Apply these transformations: {', '.join(hints)}."

        if semantics.get("contains_loop"):
            base += " Convert loops to recursion where appropriate."

        semantic_name = template.get("semantic_name")
        if semantic_name:
            base += f" Preserve the semantics of {semantic_name}."

        return base

    def _transformation_to_hint(self, transformation: str) -> str:
        """Convert transformation rule to hint."""
        hints = {
            "bitwise_to_word": "use Isabelle word library operations",
            "rotate_conversion": "convert Python rotations to word_rotl/word_rotr",
            "loop_to_recursion": "convert iterative loops to recursive functions",
        }
        return hints.get(transformation, transformation)

    def _create_metadata(self, template: Dict, semantics: Dict, transformations: List[str]) -> Dict:
        """Create metadata for the example."""
        params = self._get_cham_params()
        return {
            "cipher": self.cipher,
            "family": self.family,
            "component": template["type"],
            "difficulty": template.get("difficulty", "medium"),
            "semantic_group": template.get("semantic_group", "general"),
            "semantics": semantics,
            "transformations": transformations,
            "variant_params": params,
        }

    # ========================================================================
    # T1: Constants Extraction
    # ========================================================================

    def _extract_t1_constants(self) -> List:
        # Corresponds to Section 3: Tiered Isomorphic Alignment - The TIA Hierarchical Taxonomy, T1 (Lexical Foundation): atomic constants
        # (word sizes, round counts, rotation amounts) as definition-aligned pairs.
        """Extract T1 constants."""
        prefix = self._get_variant_prefix()
        examples: List = []

        constants = [
            {
                "semantic_name": "block_size",
                "python_symbol": f"{prefix}_block_size",
                "python_value": self._extract_python_assignment(f"{prefix}_block_size"),
                "isabelle_names": [f"{prefix}_block_size"],
                "type": "Block Size Constant",
            },
            {
                "semantic_name": "key_size",
                "python_symbol": f"{prefix}_key_size",
                "python_value": self._extract_python_assignment(f"{prefix}_key_size"),
                "isabelle_names": [f"{prefix}_key_size"],
                "type": "Key Size Constant",
            },
            {
                "semantic_name": "word_size",
                "python_symbol": f"{prefix}_word_size",
                "python_value": self._extract_python_assignment(f"{prefix}_word_size"),
                "isabelle_names": [f"{prefix}_word_size"],
                "type": "Word Size Constant",
            },
            {
                "semantic_name": "block_words",
                "python_symbol": f"{prefix}_block_words",
                "python_value": self._extract_python_assignment(f"{prefix}_block_words"),
                "isabelle_names": [f"{prefix}_block_words", f"{prefix}_nwords"],
                "type": "Block Words Constant",
            },
            {
                "semantic_name": "key_words",
                "python_symbol": f"{prefix}_key_words",
                "python_value": self._extract_python_assignment(f"{prefix}_key_words"),
                "isabelle_names": [f"{prefix}_key_words", f"{prefix}_keywords"],
                "type": "Key Words Constant",
            },
            {
                "semantic_name": "rounds",
                "python_symbol": f"{prefix}_rounds",
                "python_value": self._extract_python_assignment(f"{prefix}_rounds"),
                "isabelle_names": [f"{prefix}_rounds"],
                "type": "Rounds Constant",
            },
        ]

        for spec in constants:
            py_value = spec["python_value"]
            if not py_value:
                continue

            found = self._find_isabelle(*spec["isabelle_names"])
            if not found:
                continue

            thy_symbol, thy_code = found

            template = {
                "type": spec["type"],
                "tier": "T1",
                "tier_role": "variant_constant",
                "semantic_name": spec["semantic_name"],
                "python_symbol": spec["python_symbol"],
                "python_resolution": "constant",
                "isabelle_symbol": thy_symbol,
                "isabelle_resolution": "definition",
                "difficulty": "easy",
                "semantic_group": "constants",
                "transformations": [],
                "force_no_loops": True,
            }

            example = self.create_example(template, py_value, thy_code, spec["semantic_name"])
            if example:
                examples.append(example)

        return examples

    # ========================================================================
    # T2: Primitives Extraction
    # ========================================================================

    def _extract_t2_primitives(self) -> List:
        # Corresponds to Section 3: Tiered Isomorphic Alignment - The TIA Hierarchical Taxonomy, T2 (Functional Units): bitwise rotations,
        # modular addition, and S-box lookups as typed fun/definition pairs.
        """Extract T2 primitives."""
        prefix = self._get_variant_prefix()
        ws = self._get_cham_params()["word_size"]
        examples: List = []

        specs = [
            {
                "type": "Rotation Left Primitive",
                "semantic_name": "rotate_left",
                "python_names": [
                    f"{prefix}_rol",
                    "rol",
                    f"rotl{ws}",
                    "rotl",
                ],
                "isabelle_names": [f"{prefix}_rol"],
                "group": "bitwise_operations",
                "difficulty": "easy",
                "transformations": ["bitwise_to_word", "rotate_conversion"],
            },
            {
                "type": "Rotation Right Primitive",
                "semantic_name": "rotate_right",
                "python_names": [
                    f"{prefix}_ror",
                    "ror",
                    f"rotr{ws}",
                    "rotr",
                ],
                "isabelle_names": [f"{prefix}_ror"],
                "group": "bitwise_operations",
                "difficulty": "easy",
                "transformations": ["bitwise_to_word", "rotate_conversion"],
            },
            {
                "type": "Key Mix Low",
                "semantic_name": "keymix_low",
                "python_names": [
                    f"{prefix}_keymix_low",
                    "keymix_low",
                ],
                "isabelle_names": [f"{prefix}_keymix_low"],
                "group": "key_mixing",
                "difficulty": "easy",
                "transformations": ["bitwise_to_word"],
            },
            {
                "type": "Key Mix High",
                "semantic_name": "keymix_high",
                "python_names": [
                    f"{prefix}_keymix_high",
                    "keymix_high",
                ],
                "isabelle_names": [f"{prefix}_keymix_high"],
                "group": "key_mixing",
                "difficulty": "easy",
                "transformations": ["bitwise_to_word"],
            },
            {
                "type": "Even Round Function",
                "semantic_name": "encrypt_round_even",
                "python_names": [
                    f"{prefix}_encrypt_round_even",
                    "encrypt_round_even",
                    f"{prefix}_round_even",
                    "round_even",
                ],
                "isabelle_names": [
                    f"{prefix}_encrypt_round_even",
                    f"{prefix}_round_even",
                ],
                "group": "round_operations",
                "difficulty": "medium",
                "transformations": ["bitwise_to_word"],
            },
            {
                "type": "Odd Round Function",
                "semantic_name": "encrypt_round_odd",
                "python_names": [
                    f"{prefix}_encrypt_round_odd",
                    "encrypt_round_odd",
                    f"{prefix}_round_odd",
                    "round_odd",
                ],
                "isabelle_names": [
                    f"{prefix}_encrypt_round_odd",
                    f"{prefix}_round_odd",
                ],
                "group": "round_operations",
                "difficulty": "medium",
                "transformations": ["bitwise_to_word"],
            },
            {
                "type": "Even Round Inverse",
                "semantic_name": "decrypt_round_even",
                "python_names": [
                    f"{prefix}_decrypt_round_even",
                    "decrypt_round_even",
                    f"{prefix}_inv_round_even",
                    "inv_round_even",
                ],
                "isabelle_names": [
                    f"{prefix}_decrypt_round_even",
                    f"{prefix}_inv_round_even",
                ],
                "group": "round_operations",
                "difficulty": "medium",
                "transformations": ["bitwise_to_word"],
            },
            {
                "type": "Odd Round Inverse",
                "semantic_name": "decrypt_round_odd",
                "python_names": [
                    f"{prefix}_decrypt_round_odd",
                    "decrypt_round_odd",
                    f"{prefix}_inv_round_odd",
                    "inv_round_odd",
                ],
                "isabelle_names": [
                    f"{prefix}_decrypt_round_odd",
                    f"{prefix}_inv_round_odd",
                ],
                "group": "round_operations",
                "difficulty": "medium",
                "transformations": ["bitwise_to_word"],
            },
        ]

        for spec in specs:
            py_name = None
            py_code = None
            for candidate in spec["python_names"]:
                py_code = self._extract_python_function(candidate)
                if py_code:
                    py_name = candidate
                    break

            if not py_code:
                continue

            found = self._find_isabelle(*spec["isabelle_names"])
            if not found:
                continue

            thy_symbol, thy_code = found

            template = {
                "type": spec["type"],
                "tier": "T2",
                "tier_role": "primitive_transform",
                "semantic_name": spec["semantic_name"],
                "python_symbol": py_name,
                "python_resolution": "function",
                "isabelle_symbol": thy_symbol,
                "isabelle_resolution": "definition",
                "semantic_group": spec["group"],
                "difficulty": spec["difficulty"],
                "transformations": spec["transformations"],
            }

            example = self.create_example(template, py_code, thy_code, spec["semantic_name"])
            if example:
                examples.append(example)

        return examples

    # ========================================================================
    # T3: Structural Components Extraction
    # ========================================================================

    def _extract_t3_structural_components(self) -> List:
        # Corresponds to Section 3: Tiered Isomorphic Alignment - The TIA Hierarchical Taxonomy, T3 (Structural Composition): round functions
        # where Python for-loops map to Isabelle primrec/fold recursive definitions.
        """Extract T3 structural components."""
        prefix = self._get_variant_prefix()
        examples: List = []

        specs = [
            {
                "type": "Key Schedule Round-Key Fill (Recursive)",
                "semantic_name": "fill_round_keys",
                "python_names": [
                    f"{prefix}_fill_round_keys",
                    "fill_round_keys",
                ],
                "isabelle_names": [f"{prefix}_fill_round_keys"],
                "group": "key_expansion",
                "difficulty": "hard",
                "transformations": ["loop_to_recursion"],
            },
            {
                "type": "Key Schedule",
                "semantic_name": "generate_round_keys",
                "python_names": [
                    f"{prefix}_generate_round_keys",
                    "generate_round_keys",
                    f"{prefix}_key_schedule",
                    "key_schedule",
                ],
                "isabelle_names": [
                    f"{prefix}_generate_round_keys",
                    f"{prefix}_key_schedule",
                ],
                "group": "key_expansion",
                "difficulty": "medium",
                "transformations": [],
            },
        ]

        for spec in specs:
            py_name = None
            py_code = None
            for candidate in spec["python_names"]:
                py_code = self._extract_python_function(candidate)
                if py_code:
                    py_name = candidate
                    break

            if not py_code:
                continue

            found = self._find_isabelle(*spec["isabelle_names"])
            if not found:
                continue

            thy_symbol, thy_code = found

            template = {
                "type": spec["type"],
                "tier": "T3",
                "tier_role": "structural_transform",
                "semantic_name": spec["semantic_name"],
                "python_symbol": py_name,
                "python_resolution": "function",
                "isabelle_symbol": thy_symbol,
                "isabelle_resolution": "definition",
                "semantic_group": spec["group"],
                "difficulty": spec["difficulty"],
                "transformations": spec["transformations"],
            }

            example = self.create_example(template, py_code, thy_code, spec["semantic_name"])
            if example:
                examples.append(example)

        return examples

    # ========================================================================
    # T4: Orchestration Components Extraction
    # ========================================================================

    def _extract_t4_orchestration_components(self) -> List:
        # Corresponds to Section 3: Tiered Isomorphic Alignment - The TIA Hierarchical Taxonomy, T4 (Top-Level Orchestration): complete cipher
        # interface and key schedule as fully recursive formal specifications.
        """Extract T4 orchestration components."""
        prefix = self._get_variant_prefix()
        examples: List = []

        specs = [
            {
                "type": "Block to Words Conversion",
                "semantic_name": "block_to_words",
                "python_names": [
                    f"{prefix}_block_to_words",
                    "block_to_words",
                ],
                "isabelle_names": [f"{prefix}_block_to_words"],
                "group": "data_conversion",
                "difficulty": "easy",
                "transformations": ["bitwise_to_word"],
            },
            {
                "type": "Words to Block Conversion",
                "semantic_name": "words_to_block",
                "python_names": [
                    f"{prefix}_words_to_block",
                    "words_to_block",
                ],
                "isabelle_names": [f"{prefix}_words_to_block"],
                "group": "data_conversion",
                "difficulty": "easy",
                "transformations": ["bitwise_to_word"],
            },
            {
                "type": "Key to Words Conversion",
                "semantic_name": "key_to_words",
                "python_names": [
                    f"{prefix}_key_to_words",
                    "key_to_words",
                ],
                "isabelle_names": [f"{prefix}_key_to_words"],
                "group": "data_conversion",
                "difficulty": "easy",
                "transformations": ["bitwise_to_word"],
            },
            {
                "type": "Encrypt Round Iterator (Recursive)",
                "semantic_name": "encrypt_block_iter",
                "python_names": [
                    f"{prefix}_encrypt_block_iter",
                    "encrypt_block_iter",
                ],
                "isabelle_names": [f"{prefix}_encrypt_block_iter"],
                "group": "round_iteration",
                "difficulty": "hard",
                "transformations": ["loop_to_recursion"],
            },
            {
                "type": "Decrypt Round Iterator (Recursive)",
                "semantic_name": "decrypt_block_iter",
                "python_names": [
                    f"{prefix}_decrypt_block_iter",
                    "decrypt_block_iter",
                ],
                "isabelle_names": [f"{prefix}_decrypt_block_iter"],
                "group": "round_iteration",
                "difficulty": "hard",
                "transformations": ["loop_to_recursion"],
            },
            {
                "type": "Encrypt Block",
                "semantic_name": "encrypt_block",
                "python_names": [
                    f"{prefix}_encrypt_block",
                    "encrypt_block",
                ],
                "isabelle_names": [f"{prefix}_encrypt_block"],
                "group": "block_operations",
                "difficulty": "medium",
                "transformations": [],
            },
            {
                "type": "Decrypt Block",
                "semantic_name": "decrypt_block",
                "python_names": [
                    f"{prefix}_decrypt_block",
                    "decrypt_block",
                ],
                "isabelle_names": [f"{prefix}_decrypt_block"],
                "group": "block_operations",
                "difficulty": "medium",
                "transformations": [],
            },
            {
                "type": "Top-Level Encrypt",
                "semantic_name": "encrypt",
                "python_names": [
                    f"{prefix}_encrypt",
                    "encrypt",
                ],
                "isabelle_names": [f"{prefix}_encrypt"],
                "group": "top_level",
                "difficulty": "medium",
                "transformations": [],
            },
            {
                "type": "Top-Level Decrypt",
                "semantic_name": "decrypt",
                "python_names": [
                    f"{prefix}_decrypt",
                    "decrypt",
                ],
                "isabelle_names": [f"{prefix}_decrypt"],
                "group": "top_level",
                "difficulty": "medium",
                "transformations": [],
            },
        ]

        for spec in specs:
            py_name = None
            py_code = None
            for candidate in spec["python_names"]:
                py_code = self._extract_python_function(candidate)
                if py_code:
                    py_name = candidate
                    break

            if not py_code:
                continue

            found = self._find_isabelle(*spec["isabelle_names"])
            if not found:
                continue

            thy_symbol, thy_code = found

            template = {
                "type": spec["type"],
                "tier": "T4",
                "tier_role": "orchestration",
                "semantic_name": spec["semantic_name"],
                "python_symbol": py_name,
                "python_resolution": "function",
                "isabelle_symbol": thy_symbol,
                "isabelle_resolution": "definition",
                "semantic_group": spec["group"],
                "difficulty": spec["difficulty"],
                "transformations": spec["transformations"],
            }

            example = self.create_example(template, py_code, thy_code, spec["semantic_name"])
            if example:
                examples.append(example)

        return examples
