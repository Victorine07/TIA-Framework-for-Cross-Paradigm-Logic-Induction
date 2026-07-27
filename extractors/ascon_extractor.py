from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from .base_extractor import BaseCipherExtractor


class AsconExtractor(BaseCipherExtractor):
    # Cipher-specific implementation of Algorithm 1 (Section 3: Tiered Isomorphic Alignment - Dataset Construction and Registry) for the ASCON
    # family (Permutation/sponge). Produces T1–T4 aligned Python ↔ Isabelle/HOL component pairs.
    """
    Source-aligned ASCON extractor with explicit tier metadata.

    Tiers:

    - T1: Constants                 (key/nonce/rate sizes, round counts)
    - T2: Primitives                (permutation round: constant, substitution, diffusion)
    - T3: (none -- ASCON has no key schedule)
    - T4: Orchestration             (permutation iteration, AEAD building blocks, encrypt/decrypt)
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
    # Helpers specific to ASCON
    # ------------------------------------------------------------------

    def find_isabelle_definition_flexible(self, possible_names: List[str]) -> Optional[Tuple[str, str]]:
        for name in possible_names:
            code = self.extract_isabelle_definition(name)
            if code:
                return name, code
        return None

    def get_variant_prefix(self) -> str:
        """Isabelle/Python prefix for this variant, e.g. `ascon_64_128`."""
        return f"ascon_{self.block_size}_{self.key_size}"

    # ------------------------------------------------------------------
    # Template constructors
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

        return [ex for ex in examples if ex is not None]

    # ------------------------------------------------------------------
    # T1: Constants
    # ------------------------------------------------------------------

    def extract_t1_constants(self) -> List[Dict[str, Any]]:
        prefix = self.get_variant_prefix()
        examples: List[Dict[str, Any]] = []

        specs = [
            dict(
                semantic_name="key_size",
                python_symbol="KEY_SIZE",
                isabelle_names=[f"{prefix}_key_size"],
                type="Key Size Constant",
            ),
            dict(
                semantic_name="nonce_size",
                python_symbol="NONCE_SIZE",
                isabelle_names=[f"{prefix}_nonce_size"],
                type="Nonce Size Constant",
            ),
            dict(
                semantic_name="rate",
                python_symbol="RATE",
                isabelle_names=[f"{prefix}_rate"],
                type="Rate Constant",
            ),
            dict(
                semantic_name="rounds_a",
                python_symbol="ROUNDS_A",
                isabelle_names=[f"{prefix}_rounds_a"],
                type="Initialization/Finalization Rounds Constant",
            ),
            dict(
                semantic_name="rounds_b",
                python_symbol="ROUNDS_B",
                isabelle_names=[f"{prefix}_rounds_b"],
                type="Intermediate Rounds Constant",
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
                python_resolution="source_assignment",
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
                type="Round Constant Addition",
                semantic_name="add_round_constant",
                python_names=[f"{prefix}_add_round_constant"],
                isabelle_names=[f"{prefix}_add_round_constant"],
                group="round_operations",
                difficulty="easy",
            ),
            dict(
                type="Substitution Layer",
                semantic_name="substitution_layer",
                python_names=[f"{prefix}_substitution_layer"],
                isabelle_names=[f"{prefix}_substitution_layer"],
                group="substitution",
                difficulty="hard",
            ),
            dict(
                type="Linear Diffusion Layer",
                semantic_name="linear_diffusion_layer",
                python_names=[f"{prefix}_linear_diffusion_layer"],
                isabelle_names=[f"{prefix}_linear_diffusion_layer"],
                group="diffusion",
                difficulty="medium",
            ),
            dict(
                type="Permutation Round",
                semantic_name="permutation_round",
                python_names=[f"{prefix}_permutation_round"],
                isabelle_names=[f"{prefix}_permutation_round"],
                group="round_operations",
                difficulty="medium",
            ),
        ]

        for spec in specs:
            pycode = self.extract_python_function(spec["python_names"][0])
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
                python_symbol=spec["python_names"][0],
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
    # T3: Structural Components (none for ASCON -- no key schedule)
    # ------------------------------------------------------------------

    def extract_t3_structural_components(self) -> List[Dict[str, Any]]:
        return []

    # ------------------------------------------------------------------
    # T4: Orchestration
    # ------------------------------------------------------------------

    def extract_t4_orchestration_components(self) -> List[Dict[str, Any]]:
        prefix = self.get_variant_prefix()
        examples: List[Dict[str, Any]] = []

        specs = [
            dict(
                type="Permutation Iterator",
                semantic_name="permutation_iterate",
                python_names=[f"{prefix}_permutation_iterate"],
                isabelle_names=[f"{prefix}_permutation_iterate"],
                group="orchestration",
                difficulty="medium",
            ),
            dict(
                type="Bytes to Word Conversion",
                semantic_name="bytes_to_word",
                python_names=[f"{prefix}_bytes_to_word"],
                isabelle_names=[f"{prefix}_bytes_to_word"],
                group="data_conversion",
                difficulty="easy",
            ),
            dict(
                type="Word to Bytes Conversion",
                semantic_name="word_to_bytes",
                python_names=[f"{prefix}_word_to_bytes"],
                isabelle_names=[f"{prefix}_word_to_bytes"],
                group="data_conversion",
                difficulty="easy",
            ),
            dict(
                type="Bytes to State Conversion",
                semantic_name="bytes_to_state",
                python_names=[f"{prefix}_bytes_to_state"],
                isabelle_names=[f"{prefix}_bytes_to_state"],
                group="data_conversion",
                difficulty="easy",
            ),
            dict(
                type="Padding",
                semantic_name="pad",
                python_names=[f"{prefix}_pad"],
                isabelle_names=[f"{prefix}_pad"],
                group="mode_helpers",
                difficulty="easy",
            ),
            dict(
                type="AEAD Initialization",
                semantic_name="initialize",
                python_names=[f"{prefix}_initialize"],
                isabelle_names=[f"{prefix}_initialize"],
                group="aead",
                difficulty="medium",
            ),
            dict(
                type="Absorb AD Blocks Accumulator (Recursive)",
                semantic_name="absorb_ad_blocks",
                python_names=[f"{prefix}_absorb_ad_blocks"],
                isabelle_names=[f"{prefix}_absorb_ad_blocks"],
                group="aead",
                difficulty="hard",
            ),
            dict(
                type="Associated Data Processing",
                semantic_name="process_associated_data",
                python_names=[f"{prefix}_process_associated_data"],
                isabelle_names=[f"{prefix}_process_associated_data"],
                group="aead",
                difficulty="medium",
            ),
            dict(
                type="Squeeze Plaintext Blocks Accumulator (Recursive)",
                semantic_name="squeeze_pt_blocks",
                python_names=[f"{prefix}_squeeze_pt_blocks"],
                isabelle_names=[f"{prefix}_squeeze_pt_blocks"],
                group="aead",
                difficulty="hard",
            ),
            dict(
                type="Plaintext Processing",
                semantic_name="process_plaintext",
                python_names=[f"{prefix}_process_plaintext"],
                isabelle_names=[f"{prefix}_process_plaintext"],
                group="aead",
                difficulty="easy",
            ),
            dict(
                type="Squeeze Ciphertext Blocks Accumulator (Recursive)",
                semantic_name="squeeze_ct_blocks",
                python_names=[f"{prefix}_squeeze_ct_blocks"],
                isabelle_names=[f"{prefix}_squeeze_ct_blocks"],
                group="aead",
                difficulty="hard",
            ),
            dict(
                type="Ciphertext Processing",
                semantic_name="process_ciphertext",
                python_names=[f"{prefix}_process_ciphertext"],
                isabelle_names=[f"{prefix}_process_ciphertext"],
                group="aead",
                difficulty="easy",
            ),
            dict(
                type="AEAD Finalization",
                semantic_name="finalize",
                python_names=[f"{prefix}_finalize"],
                isabelle_names=[f"{prefix}_finalize"],
                group="aead",
                difficulty="medium",
            ),
            dict(
                type="Top-Level Encrypt",
                semantic_name="encrypt",
                python_names=[f"{prefix}_encrypt"],
                isabelle_names=[f"{prefix}_encrypt"],
                group="toplevel",
                difficulty="medium",
            ),
            dict(
                type="Top-Level Decrypt",
                semantic_name="decrypt",
                python_names=[f"{prefix}_decrypt"],
                isabelle_names=[f"{prefix}_decrypt"],
                group="toplevel",
                difficulty="medium",
            ),
        ]

        for spec in specs:
            pycode = self.extract_python_function(spec["python_names"][0])
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
                python_symbol=spec["python_names"][0],
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
    # Transformation hints, metadata, etc.
    # ------------------------------------------------------------------

    def get_transformation_rules(self, component_type: str) -> List[str]:
        rules: List[str] = []
        lowered = component_type.lower()
        if "layer" in lowered or "round" in lowered or "permutation" in lowered:
            rules.append("bitwise_to_word")
        if "processing" in lowered or "aead" in lowered or "accumulator" in lowered or "recursive" in lowered:
            rules.append("loop_to_recursion")
        return rules

    def transformation_to_hint(self, transformation: str) -> str:
        hint_map = {
            "bitwise_to_word": "Use Isabelle word library operations.",
            "loop_to_recursion": "Convert variable-length iteration over byte lists to recursive functions.",
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
            f"Translate the ASCON-{self.block_size}/{self.key_size} {template.get('type')} "
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
                "cipher": "ASCON",
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
