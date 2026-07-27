from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from datetime import datetime
import re

from .base_extractor import BaseCipherExtractor


class HightExtractor(BaseCipherExtractor):
    # Cipher-specific implementation of Algorithm 1 (Section 3: Tiered Isomorphic Alignment - Dataset Construction and Registry) for the HIGHT
    # family (Feistel). Produces T1–T4 aligned Python ↔ Isabelle/HOL component pairs.
    """
    Source-aligned HIGHT-64/128 extractor with explicit tier metadata.

    Tiers:

    - T1: Constants                 (sizes, mask, delta)
    - T2: Primitives                (rol/ror, f0/f1, per-round transforms)
    - T3: Structural Components     (key reversal, whitening keys, key schedule)
    - T4: Orchestration             (round iteration, bytes-level & word-level wrappers)
    """

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
        # legacy args for compatibility
        cipher_name: Optional[str] = None,
        cipher_info: Optional[Dict[str, Any]] = None,
        py_content: Optional[str] = None,
        thy_content: Optional[str] = None,
    ) -> None:
        # Allow legacy calls
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

        # Optional overrides for direct-content usage
        if py_content is not None:
            self.python_source = py_content
        if thy_content is not None:
            self.thy_source = thy_content

    # ------------------------------------------------------------------
    # Helpers specific to HIGHT
    # ------------------------------------------------------------------

    def find_isabelle_definition_flexible(self, possible_names: List[str]) -> Optional[Tuple[str, str]]:
        for name in possible_names:
            code = self.extract_isabelle_definition(name)
            if code:
                return name, code
        return None
    
    def get_variant_prefix(self) -> str:
        """
        Isabelle prefix for this variant.

        Matches `Hight_64_128` theory naming and definitions like
        `hight_64_128_block_size`, `hight_64_128_encrypt_bytes`, etc.
        """
        return f"hight_{self.block_size}_{self.key_size}"

    # ------------------------------------------------------------------
    # Template constructors (same pattern as SparxExtractor)
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
    # Core extraction entry point
    # ------------------------------------------------------------------

    def extract_components(self) -> List[Dict[str, Any]]:
        # Implements Algorithm 1 (Section 3: Tiered Isomorphic Alignment - Dataset Construction and Registry): produces all T1–T4 aligned
        # (instr, source, target, metadata) pairs for this cipher variant.
        examples: List[Dict[str, Any]] = []

        examples.extend(self.extract_t1_constants())
        examples.extend(self.extract_t2_primitives())
        examples.extend(self.extract_t3_structural_components())
        examples.extend(self.extract_t4_orchestration_components())

        # Filter out Nones
        return [ex for ex in examples if ex is not None]

    # ------------------------------------------------------------------
    # T1: Constants
    # ------------------------------------------------------------------

    def extract_t1_constants(self) -> List[Dict[str, Any]]:
        prefix = self.get_variant_prefix()
        examples: List[Dict[str, Any]] = []

        specs = [
            dict(
                semantic_name="block_size",
                python_symbol="HIGHT_BLOCK_SIZE",
                isabelle_names=[f"{prefix}_block_size"],
                type="Block Size Constant",
            ),
            dict(
                semantic_name="key_size",
                python_symbol="HIGHT_KEY_SIZE",
                isabelle_names=[f"{prefix}_key_size"],
                type="Key Size Constant",
            ),
            dict(
                semantic_name="block_bytes",
                python_symbol="HIGHT_BLOCK_BYTES",
                isabelle_names=[f"{prefix}_block_bytes"],
                type="Block Bytes Constant",
            ),
            dict(
                semantic_name="key_bytes",
                python_symbol="HIGHT_KEY_BYTES",
                isabelle_names=[f"{prefix}_key_bytes"],
                type="Key Bytes Constant",
            ),
            dict(
                semantic_name="internal_rounds",
                python_symbol="HIGHT_INTERNAL_ROUNDS",
                isabelle_names=[f"{prefix}_internal_rounds"],
                type="Internal Rounds Constant",
            ),
            dict(
                semantic_name="total_stages",
                python_symbol="HIGHT_TOTAL_STAGES",
                isabelle_names=[f"{prefix}_total_stages"],
                type="Total Stages Constant",
            ),
            dict(
                semantic_name="byte_mask",
                python_symbol="HIGHT_BYTE_MASK",
                isabelle_names=[f"{prefix}_byte_mask"],
                type="Byte Mask Constant",
            ),
            dict(
                semantic_name="delta",
                python_symbol="HIGHT_DELTA",
                isabelle_names=[f"{prefix}_delta"],
                type="Delta Constant",
            ),
        ]

        for spec in specs:
            py_assign = self.extract_python_assignment(spec["python_symbol"])
            if not py_assign:
                continue

            found = self.find_isabelle_definition_flexible(spec["isabelle_names"])
            if not found:
                continue
            thy_symbol, thy_code = found

            template = self.constant_template(
                type_name=spec["type"],
                semantic_name=spec["semantic_name"],
                python_symbol=spec["python_symbol"],
                python_resolution="variant_metadata" if spec["semantic_name"] not in {"delta"} else "source_assignment",
                isabelle_symbol=thy_symbol,
                isabelle_resolution="definition",
            )
            example = self.create_example(template, py_assign, thy_code, spec["semantic_name"])
            if example:
                examples.append(example)

        return examples

    # ------------------------------------------------------------------
    # T2: Primitives
    # ------------------------------------------------------------------

    def extract_t2_primitives(self) -> List[Dict[str, Any]]:
        prefix = self.get_variant_prefix()
        examples: List[Dict[str, Any]] = []

        specs = [
            dict(
                type="Rotation Primitive",
                semantic_name="rol8",
                python_names=["hight_rol8"],
                isabelle_names=[f"{prefix}_rol8"],
                group="bitwise_operations",
                difficulty="easy",
            ),
            dict(
                type="Rotation Primitive",
                semantic_name="ror8",
                python_names=["hight_ror8"],
                isabelle_names=[f"{prefix}_ror8"],
                group="bitwise_operations",
                difficulty="easy",
            ),
            dict(
                type="Boolean Function",
                semantic_name="f0",
                python_names=["hight_f0"],
                isabelle_names=[f"{prefix}_f0"],
                group="round_operations",
                difficulty="easy",
            ),
            dict(
                type="Boolean Function",
                semantic_name="f1",
                python_names=["hight_f1"],
                isabelle_names=[f"{prefix}_f1"],
                group="round_operations",
                difficulty="easy",
            ),
            dict(
                type="Initial Whitening",
                semantic_name="initial_transformation",
                python_names=["hight_initial_transformation"],
                isabelle_names=[f"{prefix}_initial_transformation"],
                group="whitening",
                difficulty="medium",
            ),
            dict(
                type="Initial Whitening Inverse",
                semantic_name="initial_transformation_inv",
                python_names=["hight_initial_transformation_inv"],
                isabelle_names=[f"{prefix}_initial_transformation_inv"],
                group="whitening",
                difficulty="medium",
            ),
            dict(
                type="Round Function",
                semantic_name="encrypt_round",
                python_names=["hight_encrypt_round"],
                isabelle_names=[f"{prefix}_encrypt_round"],
                group="round_operations",
                difficulty="medium",
            ),
            dict(
                type="Round Function Inverse",
                semantic_name="decrypt_round",
                python_names=["hight_decrypt_round"],
                isabelle_names=[f"{prefix}_decrypt_round"],
                group="round_operations",
                difficulty="medium",
            ),
            dict(
                type="Final Whitening",
                semantic_name="final_transformation",
                python_names=["hight_final_transformation"],
                isabelle_names=[f"{prefix}_final_transformation"],
                group="whitening",
                difficulty="medium",
            ),
            dict(
                type="Final Whitening Inverse",
                semantic_name="final_transformation_inv",
                python_names=["hight_final_transformation_inv"],
                isabelle_names=[f"{prefix}_final_transformation_inv"],
                group="whitening",
                difficulty="medium",
            ),
        ]

        for spec in specs:
            pyname: Optional[str] = None
            pycode: Optional[str] = None
            for candidate in spec["python_names"]:
                pycode = self.extract_python_function(candidate)
                if pycode:
                    pyname = candidate
                    break
            if not pycode:
                continue

            found = self.find_isabelle_definition_flexible(spec["isabelle_names"])
            if not found:
                continue
            thy_symbol, thy_code = found

            template = self.function_template(
                type_name=spec["type"],
                tier="T2",
                tier_role="primitive_transform",
                semantic_group=spec["group"],
                semantic_name=spec["semantic_name"],
                python_symbol=pyname,
                python_resolution="top_level_function",
                isabelle_symbol=thy_symbol,
                isabelle_resolution="definition",
                difficulty=spec["difficulty"],
            )
            example = self.create_example(template, pycode, thy_code, spec["semantic_name"])
            if example:
                examples.append(example)

        return examples

    # ------------------------------------------------------------------
    # T3: Structural Components
    # ------------------------------------------------------------------

    def extract_t3_structural_components(self) -> List[Dict[str, Any]]:
        prefix = self.get_variant_prefix()
        examples: List[Dict[str, Any]] = []

        specs = [
            dict(
                type="Key Reversal",
                semantic_name="reverse_master_key",
                python_names=["hight_reverse_master_key"],
                isabelle_names=[f"{prefix}_reverse_master_key"],
                group="key_schedule",
                difficulty="easy",
            ),
            dict(
                type="Initial Whitening Keys",
                semantic_name="initial_whitening_keys",
                python_names=["hight_initial_whitening_keys"],
                isabelle_names=[f"{prefix}_initial_whitening_keys"],
                group="key_schedule",
                difficulty="easy",
            ),
            dict(
                type="Final Whitening Keys",
                semantic_name="final_whitening_keys",
                python_names=["hight_final_whitening_keys"],
                isabelle_names=[f"{prefix}_final_whitening_keys"],
                group="key_schedule",
                difficulty="easy",
            ),
            dict(
                type="Whitening Keys (Combined)",
                semantic_name="whitening_keys",
                python_names=["hight_whitening_keys"],
                isabelle_names=[],  # optional: there is no direct combined function in Isabelle
                group="key_schedule",
                difficulty="medium",
            ),
            dict(
                type="Round Subkeys Scan",
                semantic_name="subkeys_for_round_scan",
                python_names=["hight_subkeys_for_round_scan"],
                isabelle_names=[f"{prefix}_subkeys_for_round_scan"],
                group="key_schedule",
                difficulty="hard",
            ),
            dict(
                type="Round Subkeys",
                semantic_name="subkeys_for_round",
                python_names=["hight_subkeys_for_round"],
                isabelle_names=[f"{prefix}_subkeys_for_round"],
                group="key_schedule",
                difficulty="medium",
            ),
            dict(
                type="Round Keys Recursion",
                semantic_name="generate_round_keys_rec",
                python_names=["hight_generate_round_keys_rec"],
                isabelle_names=[f"{prefix}_generate_round_keys_rec"],
                group="key_schedule",
                difficulty="hard",
            ),
            dict(
                type="Round Keys",
                semantic_name="generate_round_keys",
                python_names=["hight_generate_round_keys"],
                isabelle_names=[f"{prefix}_generate_round_keys"],
                group="key_schedule",
                difficulty="medium",
            ),
        ]

        for spec in specs:
            # whitening_keys has no direct Isabelle counterpart; skip matching if names list empty
            if not spec["isabelle_names"] and spec["semantic_name"] == "whitening_keys":
                pycode = self.extract_python_function(spec["python_names"][0])
                if not pycode:
                    continue
                # synthetic Isabelle: we skip because dataset focuses on direct alignments
                continue

            pyname: Optional[str] = None
            pycode: Optional[str] = None
            for candidate in spec["python_names"]:
                pycode = self.extract_python_function(candidate)
                if pycode:
                    pyname = candidate
                    break
            if not pycode:
                continue

            found = self.find_isabelle_definition_flexible(spec["isabelle_names"])
            if not found:
                continue
            thy_symbol, thy_code = found

            template = self.function_template(
                type_name=spec["type"],
                tier="T3",
                tier_role="structural_transform",
                semantic_group=spec["group"],
                semantic_name=spec["semantic_name"],
                python_symbol=pyname,
                python_resolution="top_level_function",
                isabelle_symbol=thy_symbol,
                isabelle_resolution="definition",
                difficulty=spec["difficulty"],
            )
            example = self.create_example(template, pycode, thy_code, spec["semantic_name"])
            if example:
                examples.append(example)

        return examples

    # ------------------------------------------------------------------
    # T4: Orchestration
    # ------------------------------------------------------------------

    def extract_t4_orchestration_components(self) -> List[Dict[str, Any]]:
        prefix = self.get_variant_prefix()
        examples: List[Dict[str, Any]] = []

        specs = [
            dict(
                type="Encrypt Round Step",
                semantic_name="encrypt_rounds_step",
                python_names=["hight_encrypt_rounds_step"],
                isabelle_names=[f"{prefix}_encrypt_rounds_step"],
                group="orchestration",
                difficulty="medium",
            ),
            dict(
                type="Encrypt Round Iterator",
                semantic_name="encrypt_rounds_iterate",
                python_names=["hight_encrypt_rounds_iterate"],
                isabelle_names=[f"{prefix}_encrypt_rounds_iterate"],
                group="orchestration",
                difficulty="hard",
            ),
            dict(
                type="Decrypt Round Step",
                semantic_name="decrypt_rounds_step",
                python_names=["hight_decrypt_rounds_step"],
                isabelle_names=[f"{prefix}_decrypt_rounds_step"],
                group="orchestration",
                difficulty="medium",
            ),
            dict(
                type="Decrypt Round Iterator",
                semantic_name="decrypt_rounds_iterate",
                python_names=["hight_decrypt_rounds_iterate"],
                isabelle_names=[f"{prefix}_decrypt_rounds_iterate"],
                group="orchestration",
                difficulty="hard",
            ),
            dict(
                type="Bytes-level Encrypt",
                semantic_name="encrypt_bytes",
                python_names=["hight_encrypt_bytes"],
                isabelle_names=[f"{prefix}_encrypt_bytes"],
                group="toplevel",
                difficulty="medium",
            ),
            dict(
                type="Bytes-level Decrypt",
                semantic_name="decrypt_bytes",
                python_names=["hight_decrypt_bytes"],
                isabelle_names=[f"{prefix}_decrypt_bytes"],
                group="toplevel",
                difficulty="medium",
            ),
            dict(
                type="Block to Bytes",
                semantic_name="block_to_bytes",
                python_names=["hight_block_to_bytes"],
                isabelle_names=[f"{prefix}_block_to_bytes"],
                group="data_conversion",
                difficulty="easy",
            ),
            dict(
                type="Bytes to Block",
                semantic_name="bytes_to_block",
                python_names=["hight_bytes_to_block"],
                isabelle_names=[f"{prefix}_bytes_to_block"],
                group="data_conversion",
                difficulty="easy",
            ),
            dict(
                type="Int to Key Bytes",
                semantic_name="int_to_key_bytes",
                python_names=[f"{prefix}_int_to_key_bytes"],
                isabelle_names=[f"{prefix}_int_to_key_bytes"],
                group="data_conversion",
                difficulty="easy",
            ),
            dict(
                type="Top-Level Encrypt",
                semantic_name="encrypt",
                python_names=["hight_encrypt"],
                isabelle_names=[f"{prefix}_encrypt"],
                group="toplevel",
                difficulty="medium",
            ),
            dict(
                type="Top-Level Decrypt",
                semantic_name="decrypt",
                python_names=["hight_decrypt"],
                isabelle_names=[f"{prefix}_decrypt"],
                group="toplevel",
                difficulty="medium",
            ),
        ]

        for spec in specs:
            pyname: Optional[str] = None
            pycode: Optional[str] = None
            for candidate in spec["python_names"]:
                pycode = self.extract_python_function(candidate)
                if pycode:
                    pyname = candidate
                    break
            if not pycode:
                continue

            found = self.find_isabelle_definition_flexible(spec["isabelle_names"])
            if not found:
                continue
            thy_symbol, thy_code = found

            template = self.function_template(
                type_name=spec["type"],
                tier="T4",
                tier_role="orchestration",
                semantic_group=spec["group"],
                semantic_name=spec["semantic_name"],
                python_symbol=pyname,
                python_resolution="top_level_function",
                isabelle_symbol=thy_symbol,
                isabelle_resolution="definition",
                difficulty=spec["difficulty"],
            )
            example = self.create_example(template, pycode, thy_code, spec["semantic_name"])
            if example:
                examples.append(example)

        return examples

    # ------------------------------------------------------------------
    # Transformation hints, metadata, etc. (reuse Sparx style)
    # ------------------------------------------------------------------

    def get_transformation_rules(self, component_type: str) -> List[str]:
        """Map component type to transformation rule tags."""
        rules: List[str] = []
        lowered = component_type.lower()
        if "rotation" in lowered:
            rules.extend(["bitwise_to_word", "rotate_conversion"])
        if "round" in lowered or "whitening" in lowered:
            rules.append("bitwise_to_word")
        if "schedule" in lowered or "keys" in lowered:
            rules.append("loop_to_recursion")
        if "block" in lowered or "bytes" in lowered or "int" in lowered:
            rules.append("bitwise_to_word")
        return rules

    def transformation_to_hint(self, transformation: str) -> str:
        hint_map = {
            "bitwise_to_word": "Use Isabelle word library operations.",
            "rotate_conversion": "Convert Python rotations to word_rotl/word_rotr.",
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
            f"Translate the HIGHT-64/128 {template.get('type')} "
            f"({template.get('semantic_name')}) from Python to Isabelle/HOL.\n"
        )
        tier = template.get("tier")
        if tier:
            base += f"This is a {self.tier_name(tier)} component.\n"

        if transformations:
            hints = [self.transformation_to_hint(t) for t in transformations]
            base += "Apply these transformations: " + "; ".join(hints) + ".\n"

        if semantics.get("has_loops"):
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
            "variant_params": {
                "block_size": self.block_size,
                "key_size": self.key_size,
            },
        }

    def create_example(
        self,
        template: Dict[str, Any],
        py_code: str,
        thy_code: str,
        name: str,
    ) -> Optional[Dict[str, Any]]:
        if not py_code or not thy_code:
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
                "cipher": "HIGHT",
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

        return {
            "instruction": instruction,
            "input": py_code,
            "output": thy_code,
            "metadata": metadata,
        }