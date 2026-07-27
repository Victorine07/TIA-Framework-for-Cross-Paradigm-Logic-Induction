from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from .base_extractor import BaseCipherExtractor


class GiftCofbExtractor(BaseCipherExtractor):
    # Cipher-specific implementation of Algorithm 1 (Section 3: Tiered Isomorphic Alignment - Dataset Construction and Registry) for the GIFT-COFB
    # family (Permutation/AEAD). Produces T1–T4 aligned Python ↔ Isabelle/HOL component pairs.
    """
    Source-aligned GIFT-COFB extractor with explicit tier metadata.

    Tiers:

    - T1: Constants                 (sizes, rounds, round constants)
    - T2: Primitives                (bitsliced S-box, GF doubling/tripling, G permutation)
    - T3: Structural Components     (key schedule, bundled round function, full permutation)
    - T4: Orchestration             (AD/message processing, encrypt/decrypt)
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
    # Helpers
    # ------------------------------------------------------------------

    def find_isabelle_definition_flexible(self, possible_names: List[str]) -> Optional[Tuple[str, str]]:
        for name in possible_names:
            code = self.extract_isabelle_definition(name)
            if code:
                return name, code
        return None

    def get_variant_prefix(self) -> str:
        return f"gift_cofb_{self.block_size}_{self.key_size}"

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
            dict(semantic_name="block_size", python_symbol="BLOCK_SIZE",
                 isabelle_names=[f"{prefix}_block_size"], type="Block Size Constant"),
            dict(semantic_name="key_size", python_symbol="KEY_SIZE",
                 isabelle_names=[f"{prefix}_key_size"], type="Key Size Constant"),
            dict(semantic_name="nonce_size", python_symbol="NONCE_SIZE",
                 isabelle_names=[f"{prefix}_nonce_size"], type="Nonce Size Constant"),
            dict(semantic_name="tag_size", python_symbol="TAG_SIZE",
                 isabelle_names=[f"{prefix}_tag_size"], type="Tag Size Constant"),
            dict(semantic_name="rounds", python_symbol="ROUNDS",
                 isabelle_names=[f"{prefix}_rounds"], type="Rounds Constant"),
            dict(semantic_name="rconst", python_symbol="RCONST",
                 isabelle_names=[f"{prefix}_rconst"], type="Round Constants"),
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
            dict(type="Rotate Right (32-bit)", semantic_name="ror32",
                 python_names=[f"{prefix}_ror32"], isabelle_names=[f"{prefix}_ror32"],
                 group="bit_ops", difficulty="easy"),
            dict(type="Swap-Move Self", semantic_name="swapmove_self",
                 python_names=[f"{prefix}_swapmove_self"], isabelle_names=[f"{prefix}_swapmove_self"],
                 group="bit_ops", difficulty="easy"),
            dict(type="Bitsliced S-box", semantic_name="sbox",
                 python_names=[f"{prefix}_sbox"], isabelle_names=[f"{prefix}_sbox"],
                 group="substitution", difficulty="medium"),
            dict(type="GF Doubling (Half Block)", semantic_name="double_half_block",
                 python_names=[f"{prefix}_double_half_block"], isabelle_names=[f"{prefix}_double_half_block"],
                 group="mask_update", difficulty="medium"),
            dict(type="GF Tripling (Half Block)", semantic_name="triple_half_block",
                 python_names=[f"{prefix}_triple_half_block"], isabelle_names=[f"{prefix}_triple_half_block"],
                 group="mask_update", difficulty="medium"),
            dict(type="G Permutation", semantic_name="g_function",
                 python_names=[f"{prefix}_g_function"], isabelle_names=[f"{prefix}_g_function"],
                 group="mask_update", difficulty="medium"),
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
    # T3: Structural Components
    # ------------------------------------------------------------------

    def extract_t3_structural_components(self) -> List[Dict[str, Any]]:
        prefix = self.get_variant_prefix()
        examples: List[Dict[str, Any]] = []

        specs = [
            dict(type="Key Update", semantic_name="key_update",
                 python_names=[f"{prefix}_key_update"], isabelle_names=[f"{prefix}_key_update"],
                 group="key_schedule", difficulty="medium"),
            dict(type="Rearrange RKey 0", semantic_name="rearrange_rkey_0",
                 python_names=[f"{prefix}_rearrange_rkey_0"], isabelle_names=[f"{prefix}_rearrange_rkey_0"],
                 group="key_schedule", difficulty="medium"),
            dict(type="Rearrange RKey 1", semantic_name="rearrange_rkey_1",
                 python_names=[f"{prefix}_rearrange_rkey_1"], isabelle_names=[f"{prefix}_rearrange_rkey_1"],
                 group="key_schedule", difficulty="medium"),
            dict(type="Rearrange RKey 2", semantic_name="rearrange_rkey_2",
                 python_names=[f"{prefix}_rearrange_rkey_2"], isabelle_names=[f"{prefix}_rearrange_rkey_2"],
                 group="key_schedule", difficulty="medium"),
            dict(type="Rearrange RKey 3", semantic_name="rearrange_rkey_3",
                 python_names=[f"{prefix}_rearrange_rkey_3"], isabelle_names=[f"{prefix}_rearrange_rkey_3"],
                 group="key_schedule", difficulty="medium"),
            dict(type="Key Triple Update 0", semantic_name="key_triple_update_0",
                 python_names=[f"{prefix}_key_triple_update_0"], isabelle_names=[f"{prefix}_key_triple_update_0"],
                 group="key_schedule", difficulty="medium"),
            dict(type="Key Double Update 1", semantic_name="key_double_update_1",
                 python_names=[f"{prefix}_key_double_update_1"], isabelle_names=[f"{prefix}_key_double_update_1"],
                 group="key_schedule", difficulty="medium"),
            dict(type="Key Triple Update 1", semantic_name="key_triple_update_1",
                 python_names=[f"{prefix}_key_triple_update_1"], isabelle_names=[f"{prefix}_key_triple_update_1"],
                 group="key_schedule", difficulty="medium"),
            dict(type="Key Double Update 2", semantic_name="key_double_update_2",
                 python_names=[f"{prefix}_key_double_update_2"], isabelle_names=[f"{prefix}_key_double_update_2"],
                 group="key_schedule", difficulty="medium"),
            dict(type="Key Triple Update 2", semantic_name="key_triple_update_2",
                 python_names=[f"{prefix}_key_triple_update_2"], isabelle_names=[f"{prefix}_key_triple_update_2"],
                 group="key_schedule", difficulty="medium"),
            dict(type="Key Double Update 3", semantic_name="key_double_update_3",
                 python_names=[f"{prefix}_key_double_update_3"], isabelle_names=[f"{prefix}_key_double_update_3"],
                 group="key_schedule", difficulty="medium"),
            dict(type="Key Triple Update 3", semantic_name="key_triple_update_3",
                 python_names=[f"{prefix}_key_triple_update_3"], isabelle_names=[f"{prefix}_key_triple_update_3"],
                 group="key_schedule", difficulty="medium"),
            dict(type="Key Double Update 4", semantic_name="key_double_update_4",
                 python_names=[f"{prefix}_key_double_update_4"], isabelle_names=[f"{prefix}_key_double_update_4"],
                 group="key_schedule", difficulty="medium"),
            dict(type="Key Triple Update 4", semantic_name="key_triple_update_4",
                 python_names=[f"{prefix}_key_triple_update_4"], isabelle_names=[f"{prefix}_key_triple_update_4"],
                 group="key_schedule", difficulty="medium"),
            dict(type="Phase 4 Key Schedule Step", semantic_name="phase4_step",
                 python_names=[f"{prefix}_phase4_step"], isabelle_names=[f"{prefix}_phase4_step"],
                 group="key_schedule", difficulty="hard"),
            dict(type="Round Key Schedule", semantic_name="precompute_round_keys",
                 python_names=[f"{prefix}_precompute_round_keys"], isabelle_names=[f"{prefix}_precompute_round_keys"],
                 group="key_schedule", difficulty="hard"),
            dict(type="Quintuple Round", semantic_name="quintuple_round",
                 python_names=[f"{prefix}_quintuple_round"], isabelle_names=[f"{prefix}_quintuple_round"],
                 group="round_function", difficulty="hard"),
            dict(type="Permutation", semantic_name="permutation",
                 python_names=[f"{prefix}_permutation"], isabelle_names=[f"{prefix}_permutation"],
                 group="round_function", difficulty="medium"),
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
                tier="T3",
                tier_role="structural_transform",
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
    # T4: Orchestration
    # ------------------------------------------------------------------

    def extract_t4_orchestration_components(self) -> List[Dict[str, Any]]:
        prefix = self.get_variant_prefix()
        examples: List[Dict[str, Any]] = []

        specs = [
            dict(type="Padding", semantic_name="padding",
                 python_names=[f"{prefix}_padding"], isabelle_names=[f"{prefix}_padding"],
                 group="mode_helpers", difficulty="easy"),
            dict(type="Initialize", semantic_name="initialize",
                 python_names=[f"{prefix}_initialize"], isabelle_names=[f"{prefix}_initialize"],
                 group="orchestration", difficulty="medium"),
            dict(type="Process AD Blocks Loop", semantic_name="process_ad_blocks_loop",
                 python_names=[f"{prefix}_process_ad_blocks_loop"], isabelle_names=[f"{prefix}_process_ad_blocks_loop"],
                 group="aead", difficulty="hard"),
            dict(type="Process Associated Data", semantic_name="process_associated_data",
                 python_names=[f"{prefix}_process_associated_data"], isabelle_names=[f"{prefix}_process_associated_data"],
                 group="orchestration", difficulty="medium"),
            dict(type="Encrypt Blocks Loop", semantic_name="encrypt_blocks_loop",
                 python_names=[f"{prefix}_encrypt_blocks_loop"], isabelle_names=[f"{prefix}_encrypt_blocks_loop"],
                 group="aead", difficulty="hard"),
            dict(type="Encrypt Message Blocks", semantic_name="encrypt_message_blocks",
                 python_names=[f"{prefix}_encrypt_message_blocks"], isabelle_names=[f"{prefix}_encrypt_message_blocks"],
                 group="orchestration", difficulty="medium"),
            dict(type="Decrypt Blocks Loop", semantic_name="decrypt_blocks_loop",
                 python_names=[f"{prefix}_decrypt_blocks_loop"], isabelle_names=[f"{prefix}_decrypt_blocks_loop"],
                 group="aead", difficulty="hard"),
            dict(type="Decrypt Message Blocks", semantic_name="decrypt_message_blocks",
                 python_names=[f"{prefix}_decrypt_message_blocks"], isabelle_names=[f"{prefix}_decrypt_message_blocks"],
                 group="orchestration", difficulty="medium"),
            dict(type="Top-Level Encrypt", semantic_name="encrypt",
                 python_names=[f"{prefix}_encrypt"], isabelle_names=[f"{prefix}_encrypt"],
                 group="toplevel", difficulty="medium"),
            dict(type="Top-Level Decrypt", semantic_name="decrypt",
                 python_names=[f"{prefix}_decrypt"], isabelle_names=[f"{prefix}_decrypt"],
                 group="toplevel", difficulty="medium"),
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
        if "box" in lowered or "block" in lowered or "permutation" in lowered or "round" in lowered:
            rules.append("bitwise_to_word")
        if "schedule" in lowered or "process" in lowered or "encrypt" in lowered or "decrypt" in lowered:
            rules.append("loop_to_recursion")
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
            f"Translate the GIFT-COFB-{self.block_size}/{self.key_size} {template.get('type')} "
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
                "cipher": "GIFT-COFB",
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
