from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
import re

from .base_extractor import BaseCipherExtractor


class SimonExtractor(BaseCipherExtractor):
    # Cipher-specific implementation of Algorithm 1 (Section 3: Tiered Isomorphic Alignment - Dataset Construction and Registry) for the SIMON
    # family (Feistel). Produces T1–T4 aligned Python ↔ Isabelle/HOL component pairs.
    """
    Debug-heavy, source-aligned SIMON extractor.

    This version is intentionally verbose so we can pinpoint where all-zero
    extraction is happening:
      - files not loaded
      - python symbol mismatch
      - isabelle symbol mismatch
      - example creation failure
    """

    DEBUG = True

    def __init__(
        self,
        root_dir: str,
        cipher: str,
        family: str,
        subfamily: str,
        block_size: int,
        key_size: int,
        variant_config: Dict[str, Any],
        dataset_split: str = "train",
        cipher_name: Optional[str] = None,
        cipher_info: Optional[Dict[str, Any]] = None,
        py_content: Optional[str] = None,
        thy_content: Optional[str] = None,
    ) -> None:
        if cipher_name is not None:
            cipher = cipher_name
        if cipher_info is not None and isinstance(cipher_info, dict):
            family = cipher_info.get("cipher_family", family)

        super().__init__(
            root_dir=root_dir,
            cipher=cipher,
            family=family,
            subfamily=subfamily,
            block_size=block_size,
            key_size=key_size,
            variant_config=variant_config,
            dataset_split=dataset_split,
        )

        if py_content is not None:
            self.python_source = py_content
        if thy_content is not None:
            self.thy_source = thy_content

    # ------------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------------

    def dbg(self, msg: str) -> None:
        if self.DEBUG:
            print(f"[SIMON {self.block_size}/{self.key_size}] {msg}")

    def preview(self, text: str, n: int = 200) -> str:
        if not text:
            return "<EMPTY>"
        return text[:n].replace("\n", "\\n")

    def debug_file_state(self) -> None:
        self.dbg("========== FILE STATE ==========")
        self.dbg(f"root_dir={self.root_dir}")
        self.dbg(f"python_file={self.python_file}")
        self.dbg(f"thy_file={self.thy_file}")

        if self.python_file is not None:
            self.dbg(f"python_file exists={Path(self.python_file).exists()}")
        if self.thy_file is not None:
            self.dbg(f"thy_file exists={Path(self.thy_file).exists()}")

        self.dbg(f"python_source_len={len(self.python_source)}")
        self.dbg(f"thy_source_len={len(self.thy_source)}")
        self.dbg(f"python_source_preview={self.preview(self.python_source, 300)}")
        self.dbg(f"thy_source_preview={self.preview(self.thy_source, 300)}")

        self.dbg(f"python_ast_is_none={self.python_ast is None}")
        self.dbg("================================")

    def debug_definition_lines(self) -> None:
        self.dbg("---- Isabelle definition lines preview ----")
        count = 0
        for line in self.thy_source.splitlines():
            s = line.strip()
            if s.startswith(("definition ", "fun ", "function ", "primrec ")):
                self.dbg(f"DEF-LINE: {s}")
                count += 1
                if count >= 25:
                    break
        if count == 0:
            self.dbg("No definition/fun/function/primrec lines found in thy_source.")
        self.dbg("-------------------------------------------")

    def debug_python_symbols(self) -> None:
        self.dbg("---- Python symbol preview ----")
        fn_lines = []
        for line in self.python_source.splitlines():
            s = line.strip()
            if s.startswith("def ") or s.startswith("async def "):
                fn_lines.append(s)
            elif re.match(r"^[A-Z][A-Z0-9_]*\s*=", s):
                fn_lines.append(s)
        for item in fn_lines[:30]:
            self.dbg(f"PY-LINE: {item}")
        if not fn_lines:
            self.dbg("No obvious Python defs/constant assignments found.")
        self.dbg("-------------------------------")

    # ------------------------------------------------------------------
    # Helpers specific to SIMON
    # ------------------------------------------------------------------

    def find_isabelle_definition_flexible(
        self, possible_names: List[str]
    ) -> Optional[Tuple[str, str]]:
        self.dbg(f"Try Isabelle candidates: {possible_names}")
        for name in possible_names:
            code = self.extract_isabelle_definition_debug(name)
            if code:
                self.dbg(f"FOUND Isabelle symbol: {name}")
                return name, code
            self.dbg(f"MISS Isabelle symbol: {name}")
        return None

    def get_variant_prefix(self) -> str:
        return f"simon_{self.block_size}_{self.key_size}"

    def get_simon_params(self) -> Dict[str, Any]:
        variant_table: Dict[Tuple[int, int], Dict[str, int]] = {
            (32, 64):   {"rounds": 32, "key_words": 4},
            (48, 72):   {"rounds": 36, "key_words": 3},
            (48, 96):   {"rounds": 36, "key_words": 4},
            (64, 96):   {"rounds": 42, "key_words": 3},
            (64, 128):  {"rounds": 44, "key_words": 4},
            (96, 96):   {"rounds": 52, "key_words": 2},
            (96, 144):  {"rounds": 54, "key_words": 3},
            (128, 128): {"rounds": 68, "key_words": 2},
            (128, 192): {"rounds": 69, "key_words": 3},
            (128, 256): {"rounds": 72, "key_words": 4},
        }
        key = (self.block_size, self.key_size)
        if key not in variant_table:
            raise ValueError(f"Unsupported SIMON variant {self.block_size}/{self.key_size}")

        return {
            "block_size": self.block_size,
            "key_size": self.key_size,
            "word_size": self.block_size // 2,
            "rounds": variant_table[key]["rounds"],
            "key_words": variant_table[key]["key_words"],
            "branches": 2,
            "words_per_block": 2,
            "steps": 1,
            "rounds_per_step": variant_table[key]["rounds"],
            "total_rounds": variant_table[key]["rounds"],
        }

    def find_python_function_flexible(
        self, possible_names: List[str]
    ) -> Optional[Tuple[str, str]]:
        self.dbg(f"Try Python function candidates: {possible_names}")
        for name in possible_names:
            code = self.extract_python_function(name)
            if code:
                self.dbg(f"FOUND Python function: {name}")
                return name, code
            self.dbg(f"MISS Python function: {name}")
        return None

    def find_python_assignment_flexible(
        self, possible_names: List[str]
    ) -> Optional[Tuple[str, str]]:
        self.dbg(f"Try Python assignment candidates: {possible_names}")
        for name in possible_names:
            code = self.extract_python_assignment(name)
            if code:
                self.dbg(f"FOUND Python assignment: {name} -> {code}")
                return name, code
            self.dbg(f"MISS Python assignment: {name}")
        return None

    def extract_isabelle_definition_debug(self, symbol_name: str) -> Optional[str]:
        lines = self.thy_source.splitlines()
        start_idx = None

        start_pattern = re.compile(
            rf"^\s*(definition|fun|function|primrec)\s+{re.escape(symbol_name)}(?:\b|\s|::)"
        )

        for i, line in enumerate(lines):
            if start_pattern.search(line):
                start_idx = i
                self.dbg(f"Isabelle start match for {symbol_name!r} at line {i+1}: {line.strip()}")
                break

        if start_idx is None:
            self.dbg(f"No direct Isabelle start match for {symbol_name!r}")
            near = []
            for line in lines:
                s = line.strip()
                if symbol_name in s or s.startswith(("definition ", "fun ", "function ", "primrec ")):
                    near.append(s)
                if len(near) >= 12:
                    break
            for cand in near:
                self.dbg(f"  near: {cand}")
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

        code = "\n".join(collected).strip()
        if code:
            self.dbg(f"Extracted Isabelle code for {symbol_name!r}, len={len(code)}")
            return code
        self.dbg(f"Matched Isabelle start but empty body for {symbol_name!r}")
        return None

    # ------------------------------------------------------------------
    # Templates
    # ------------------------------------------------------------------

    def constant_template(
        self,
        *,
        type_name: str,
        semantic_name: str,
        python_symbol: str,
        python_resolution: str,
        isabelle_symbol: str,
        isabelle_resolution: str,
    ) -> Dict[str, Any]:
        return {
            "type": type_name,
            "tier": "T1",
            "tier_role": "variant_constant",
            "difficulty": "easy",
            "semantic_group": "constants",
            "semantic_name": semantic_name,
            "python_symbol": python_symbol,
            "python_resolution": python_resolution,
            "isabelle_symbol": isabelle_symbol,
            "isabelle_resolution": isabelle_resolution,
            "force_no_loops": True,
        }

    def function_template(
        self,
        *,
        type_name: str,
        tier: str,
        tier_role: str,
        semantic_group: str,
        semantic_name: str,
        python_symbol: str,
        python_resolution: str,
        isabelle_symbol: str,
        isabelle_resolution: str,
        difficulty: str = "medium",
    ) -> Dict[str, Any]:
        return {
            "type": type_name,
            "tier": tier,
            "tier_role": tier_role,
            "difficulty": difficulty,
            "semantic_group": semantic_group,
            "semantic_name": semantic_name,
            "python_symbol": python_symbol,
            "python_resolution": python_resolution,
            "isabelle_symbol": isabelle_symbol,
            "isabelle_resolution": isabelle_resolution,
        }

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------

    def extract_components(self) -> List[Dict[str, Any]]:
        # Implements Algorithm 1 (Section 3: Tiered Isomorphic Alignment - Dataset Construction and Registry): produces all T1–T4 aligned
        # (instr, source, target, metadata) pairs for this cipher variant.
        self.dbg("START extract_components")
        self.debug_file_state()
        self.debug_python_symbols()
        self.debug_definition_lines()

        prefix = self.get_variant_prefix()
        self.dbg(f"variant_prefix={prefix}")
        self.dbg(f"variant_params={self.get_simon_params()}")

        smoke_tests = [
            f"{prefix}_z0",
            f"{prefix}_F_function",
            f"{prefix}_encrypt",
            f"{prefix}_decrypt",
            f"{prefix}_encrypt_block",
            f"{prefix}_decrypt_block",
        ]
        for name in smoke_tests:
            ok = self.extract_isabelle_definition_debug(name) is not None
            self.dbg(f"SMOKE Isabelle {name}: {ok}")

        examples: List[Dict[str, Any]] = []
        examples.extend(self.extract_t1_constants())
        examples.extend(self.extract_t2_primitives())
        examples.extend(self.extract_t3_structural_components())
        examples.extend(self.extract_t4_orchestration_components())

        filtered = [ex for ex in examples if ex is not None]
        self.dbg(f"END extract_components total_examples={len(filtered)}")
        return filtered

    # ------------------------------------------------------------------
    # T1
    # ------------------------------------------------------------------
    
    def extract_t1_constants(self) -> List[Dict[str, Any]]:
        prefix = self.get_variant_prefix()
        self.dbg("ENTER T1")
        examples: List[Dict[str, Any]] = []
    
        specs = [
            dict(
                semantic_name="word_size",
                python_names=["WORD_SIZE"],
                isabelle_names=[f"{prefix}_word_size"],
                type="Word Size Constant",
            ),
            dict(
                semantic_name="block_size",
                python_names=["BLOCK_SIZE"],
                isabelle_names=[f"{prefix}_block_size"],
                type="Block Size Constant",
            ),
            dict(
                semantic_name="key_size",
                python_names=["KEY_SIZE"],
                isabelle_names=[f"{prefix}_key_size"],
                type="Key Size Constant",
            ),
            dict(
                semantic_name="rounds",
                python_names=["ROUNDS"],
                isabelle_names=[f"{prefix}_rounds"],
                type="Rounds Constant",
            ),
            dict(
                semantic_name="key_words",
                python_names=["KEY_WORDS"],
                isabelle_names=[f"{prefix}_key_words"],
                type="Key Words Constant",
            ),
            dict(
                semantic_name="word_mask",
                python_names=["WORD_MASK"],
                isabelle_names=[f"{prefix}_word_mask"],
                type="Word Mask Constant",
            ),
            dict(
                semantic_name="round_constant",
                python_names=["ROUND_CONSTANT"],
                isabelle_names=[f"{prefix}_round_constant", f"{prefix}_rho_const"],
                type="Round Constant",
            ),
            dict(
                semantic_name="z_sequence",
                python_names=["Z_SEQUENCE"],
                isabelle_names=[f"{prefix}_z_sequence", f"{prefix}_z0"],
                type="Z Sequence Constant",
            ),
        ]
    
        for spec in specs:
            self.dbg(f"T1 spec={spec['semantic_name']}")
            py_found = self.find_python_assignment_flexible(spec["python_names"])
            if not py_found:
                self.dbg(f"T1 skip {spec['semantic_name']}: no Python assignment found")
                continue
            py_symbol, py_assign = py_found
    
            found = self.find_isabelle_definition_flexible(spec["isabelle_names"])
            if not found:
                self.dbg(f"T1 skip {spec['semantic_name']}: no Isabelle definition found")
                continue
            thy_symbol, thy_code = found
    
            template = self.constant_template(
                type_name=spec["type"],
                semantic_name=spec["semantic_name"],
                python_symbol=py_symbol,
                python_resolution="source_assignment",
                isabelle_symbol=thy_symbol,
                isabelle_resolution="definition",
            )
            example = self.create_example(template, py_assign, thy_code, spec["semantic_name"])
            if example:
                examples.append(example)
                self.dbg(f"T1 added example {spec['semantic_name']}")
    
        self.dbg(f"EXIT T1 count={len(examples)}")
        return examples
    
    
    def extract_t2_primitives(self) -> List[Dict[str, Any]]:
        prefix = self.get_variant_prefix()
        self.dbg("ENTER T2")
        examples: List[Dict[str, Any]] = []
    
        specs = [
            dict(
                type="Rotation Left Primitive",
                semantic_name="rol",
                python_names=[f"{prefix}_rol"],
                isabelle_names=[f"{prefix}_rol"],
                group="bitwise_operations",
                difficulty="easy",
            ),
            dict(
                type="Rotation Right Primitive",
                semantic_name="ror",
                python_names=[f"{prefix}_ror"],
                isabelle_names=[f"{prefix}_ror"],
                group="bitwise_operations",
                difficulty="easy",
            ),
            dict(
                type="Round Function",
                semantic_name="round_function",
                python_names=[f"{prefix}_f"],
                isabelle_names=[f"{prefix}_f"],
                group="round_operations",
                difficulty="medium",
            ),
            dict(
                type="Encrypt Round",
                semantic_name="encrypt_round",
                python_names=[f"{prefix}_encrypt_round"],
                isabelle_names=[f"{prefix}_encrypt_round"],
                group="round_operations",
                difficulty="medium",
            ),
            dict(
                type="Decrypt Round",
                semantic_name="decrypt_round",
                python_names=[f"{prefix}_decrypt_round"],
                isabelle_names=[f"{prefix}_decrypt_round"],
                group="round_operations",
                difficulty="medium",
            ),
        ]
    
        for spec in specs:
            self.dbg(f"T2 spec={spec['semantic_name']}")
            py_found = self.find_python_function_flexible(spec["python_names"])
            if not py_found:
                self.dbg(f"T2 skip {spec['semantic_name']}: no Python function found")
                continue
            py_symbol, py_code = py_found
    
            found = self.find_isabelle_definition_flexible(spec["isabelle_names"])
            if not found:
                self.dbg(f"T2 skip {spec['semantic_name']}: no Isabelle definition found")
                continue
            thy_symbol, thy_code = found
    
            template = self.function_template(
                type_name=spec["type"],
                tier="T2",
                tier_role="primitive_transform",
                semantic_group=spec["group"],
                semantic_name=spec["semantic_name"],
                python_symbol=py_symbol,
                python_resolution="top_level_function",
                isabelle_symbol=thy_symbol,
                isabelle_resolution="definition",
                difficulty=spec["difficulty"],
            )
            example = self.create_example(template, py_code, thy_code, spec["semantic_name"])
            if example:
                examples.append(example)
                self.dbg(f"T2 added example {spec['semantic_name']}")
    
        self.dbg(f"EXIT T2 count={len(examples)}")
        return examples
    
    
    def extract_t3_structural_components(self) -> List[Dict[str, Any]]:
        prefix = self.get_variant_prefix()
        self.dbg("ENTER T3")
        examples: List[Dict[str, Any]] = []
    
        specs = [
            dict(
                type="Key to Words",
                semantic_name="key_to_words",
                python_names=[f"{prefix}_key_to_words"],
                isabelle_names=[f"{prefix}_key_to_words"],
                group="key_schedule",
                difficulty="medium",
            ),
            dict(
                type="Key Schedule Recursive",
                semantic_name="generate_round_keys_rec",
                python_names=[f"{prefix}_generate_round_keys_rec"],
                isabelle_names=[f"{prefix}_generate_round_keys_rec"],
                group="key_schedule",
                difficulty="hard",
            ),
            dict(
                type="Key Schedule",
                semantic_name="generate_round_keys",
                python_names=[f"{prefix}_generate_round_keys"],
                isabelle_names=[f"{prefix}_generate_round_keys"],
                group="key_schedule",
                difficulty="hard",
            ),
        ]
    
        for spec in specs:
            self.dbg(f"T3 spec={spec['semantic_name']}")
            py_found = self.find_python_function_flexible(spec["python_names"])
            if not py_found:
                self.dbg(f"T3 skip {spec['semantic_name']}: no Python function found")
                continue
            py_symbol, py_code = py_found
    
            found = self.find_isabelle_definition_flexible(spec["isabelle_names"])
            if not found:
                self.dbg(f"T3 skip {spec['semantic_name']}: no Isabelle definition found")
                continue
            thy_symbol, thy_code = found
    
            template = self.function_template(
                type_name=spec["type"],
                tier="T3",
                tier_role="structural_transform",
                semantic_group=spec["group"],
                semantic_name=spec["semantic_name"],
                python_symbol=py_symbol,
                python_resolution="top_level_function",
                isabelle_symbol=thy_symbol,
                isabelle_resolution="definition",
                difficulty=spec["difficulty"],
            )
            example = self.create_example(template, py_code, thy_code, spec["semantic_name"])
            if example:
                examples.append(example)
                self.dbg(f"T3 added example {spec['semantic_name']}")
    
        self.dbg(f"EXIT T3 count={len(examples)}")
        return examples
    
    
    def extract_t4_orchestration_components(self) -> List[Dict[str, Any]]:
        prefix = self.get_variant_prefix()
        self.dbg("ENTER T4")
        examples: List[Dict[str, Any]] = []
    
        specs = [
            dict(
                type="Block to Words Conversion",
                semantic_name="block_to_words",
                python_names=[f"{prefix}_block_to_words"],
                isabelle_names=[f"{prefix}_block_to_words"],
                group="conversion",
                difficulty="easy",
            ),
            dict(
                type="Words to Block Conversion",
                semantic_name="words_to_block",
                python_names=[f"{prefix}_words_to_block"],
                isabelle_names=[f"{prefix}_words_to_block"],
                group="conversion",
                difficulty="easy",
            ),
            dict(
                type="Encrypt Block Iter",
                semantic_name="encrypt_block_iter",
                python_names=[f"{prefix}_encrypt_block_iter"],
                isabelle_names=[f"{prefix}_encrypt_block_iter"],
                group="block_operations",
                difficulty="medium",
            ),
            dict(
                type="Encrypt Block",
                semantic_name="encrypt_block",
                python_names=[f"{prefix}_encrypt_block"],
                isabelle_names=[f"{prefix}_encrypt_block"],
                group="block_operations",
                difficulty="medium",
            ),
            dict(
                type="Decrypt Block",
                semantic_name="decrypt_block",
                python_names=[f"{prefix}_decrypt_block"],
                isabelle_names=[f"{prefix}_decrypt_block"],
                group="block_operations",
                difficulty="medium",
            ),
            dict(
                type="Top-Level Encrypt",
                semantic_name="encrypt",
                python_names=[f"{prefix}_encrypt"],
                isabelle_names=[f"{prefix}_encrypt"],
                group="top_level",
                difficulty="medium",
            ),
            dict(
                type="Top-Level Decrypt",
                semantic_name="decrypt",
                python_names=[f"{prefix}_decrypt"],
                isabelle_names=[f"{prefix}_decrypt"],
                group="top_level",
                difficulty="medium",
            ),
        ]

        for spec in specs:
            self.dbg(f"T4 spec={spec['semantic_name']}")
            py_found = self.find_python_function_flexible(spec["python_names"])
            if not py_found:
                self.dbg(f"T4 skip {spec['semantic_name']}: no Python function found")
                continue
            py_symbol, py_code = py_found
    
            found = self.find_isabelle_definition_flexible(spec["isabelle_names"])
            if not found:
                self.dbg(f"T4 skip {spec['semantic_name']}: no Isabelle definition found")
                continue
            thy_symbol, thy_code = found
    
            template = self.function_template(
                type_name=spec["type"],
                tier="T4",
                tier_role="orchestration",
                semantic_group=spec["group"],
                semantic_name=spec["semantic_name"],
                python_symbol=py_symbol,
                python_resolution="top_level_function",
                isabelle_symbol=thy_symbol,
                isabelle_resolution="definition",
                difficulty=spec["difficulty"],
            )
            example = self.create_example(template, py_code, thy_code, spec["semantic_name"])
            if example:
                examples.append(example)
                self.dbg(f"T4 added example {spec['semantic_name']}")
    
        self.dbg(f"EXIT T4 count={len(examples)}")
        return examples
        

    # ------------------------------------------------------------------
    # Metadata / example creation
    # ------------------------------------------------------------------

    def get_transformation_rules(self, component_type: str) -> List[str]:
        rules: List[str] = []
        lowered = component_type.lower()
        if "round" in lowered or "function" in lowered:
            rules.append("bitwise_to_word")
        if "key" in lowered or "schedule" in lowered:
            rules.append("loop_to_recursion")
        if "block" in lowered:
            rules.append("bitwise_to_word")
        return rules

    def transformation_to_hint(self, transformation: str) -> str:
        hint_map = {
            "bitwise_to_word": "Use Isabelle word library operations.",
            "loop_to_recursion": "Convert iterative loops to recursive functions.",
        }
        return hint_map.get(transformation, transformation.replace("_", " "))

    def tier_name(self, tier: str) -> str:
        names = {
            "T1": "T1 Lexical Foundation",
            "T2": "T2 Functional Unit",
            "T3": "T3 Structural Composition",
            "T4": "T4 Top-Level Orchestration",
        }
        return names.get(tier, tier)

    def create_instruction(
        self,
        template: Dict[str, Any],
        semantics: Dict[str, Any],
        transformations: List[str],
    ) -> str:
        base = (
            f"Translate the SIMON-{self.block_size}/{self.key_size} {template.get('type')} "
            f"({template.get('semantic_name')}) from Python to Isabelle/HOL.\n"
        )
        tier = template.get("tier")
        if tier:
            base += f"This is a {self.tier_name(tier)} component.\n"
        if transformations:
            hints = [self.transformation_to_hint(t) for t in transformations]
            base += "Apply these transformations: " + "; ".join(hints) + ".\n"
        if semantics.get("contains_loop"):
            base += "Convert loops to recursion where appropriate.\n"
        semantic_name = template.get("semantic_name")
        if semantic_name:
            base += f"Preserve the semantics of `{semantic_name}`.\n"
        return base

    def create_metadata(
        self,
        template: Dict[str, Any],
        semantics: Dict[str, Any],
        transformations: List[str],
    ) -> Dict[str, Any]:
        return {
            "cipher": self.cipher,
            "family": self.family,
            "component": template.get("type"),
            "difficulty": template.get("difficulty", "medium"),
            "semantic_group": template.get("semantic_group", "general"),
            "semantics": semantics,
            "transformations": transformations,
            "variant_params": self.get_simon_params(),
        }

    def create_example(
        self,
        template: Dict[str, Any],
        py_code: str,
        thy_code: str,
        name: str,
    ) -> Optional[Dict[str, Any]]:
        self.dbg(f"create_example start name={name}")
        if not py_code:
            self.dbg(f"create_example abort {name}: empty py_code")
            return None
        if not thy_code:
            self.dbg(f"create_example abort {name}: empty thy_code")
            return None

        py_code = py_code.strip()
        thy_code = thy_code.strip()

        semantics = self.analyze_python_semantics(py_code)
        transformations = self.get_transformation_rules(template.get("type", ""))
        instruction = self.create_instruction(template, semantics, transformations)
        metadata = self.create_metadata(template, semantics, transformations)

        metadata.update(
            {
                "tier": template.get("tier"),
                "tier_role": template.get("tier_role"),
                "semantic_name": template.get("semantic_name"),
                "python_symbol": template.get("python_symbol"),
                "python_resolution": template.get("python_resolution"),
                "isabelle_symbol": template.get("isabelle_symbol"),
                "isabelle_resolution": template.get("isabelle_resolution"),
                "cipher": "SIMON",
                "family": self.family,
                "variant": {
                    "block_size": self.block_size,
                    "key_size": self.key_size,
                },
                "source_aligned": True,
                "synthetic_python": False,
                "synthetic_isabelle": False,
                "extraction_time": datetime.now().isoformat(),
            }
        )

        ex = {
            "instruction": instruction,
            "input": py_code,
            "output": thy_code,
            "metadata": metadata,
        }
        self.dbg(f"create_example success name={name} input_len={len(py_code)} output_len={len(thy_code)}")
        return ex