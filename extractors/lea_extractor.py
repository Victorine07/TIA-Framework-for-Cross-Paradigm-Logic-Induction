# LEA_EXTRACTOR.PY

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from .base_extractor import BaseCipherExtractor, JsonlRecord, SCHEMA_VERSION, utc_now_iso


class LeaExtractor(BaseCipherExtractor):
    # Cipher-specific implementation of Algorithm 1 (Section 3: Tiered Isomorphic Alignment - Dataset Construction and Registry) for the LEA
    # family (ARX, held-out unseen cipher). Produces T1–T4 aligned pairs.
    """LEA extractor compatible with the stricter BaseCipherExtractor JSONL schema."""

    def extract_components(self) -> List[JsonlRecord]:
        records: List[JsonlRecord] = []

        records.extend(self._extract_t1_constants())
        records.extend(self._extract_t2_primitives())
        records.extend(self._extract_t3_key_schedule())
        records.extend(self._extract_t4_orchestration())

        return records

    def _variant_prefix(self) -> str:
        return f"lea_{self.block_size}_{self.key_size}"

    def _variant_upper_prefix(self) -> str:
        return f"LEA_{self.block_size}_{self.key_size}"

    def _lea_params(self) -> Dict[str, Any]:
        rounds_map = {128: 24, 192: 28, 256: 32}
        key_words_map = {128: 4, 192: 6, 256: 8}
        return {
            "word_size": 32,
            "block_words": 4,
            "key_words": key_words_map[self.key_size],
            "rounds": rounds_map[self.key_size],
            "round_key_words": 6,
        }

    def _make_record(
        self,
        *,
        tier: str,
        tier_role: str,
        component_type: str,
        component_name: str,
        alignment_kind: str,
        semantic_group: str,
        difficulty: str,
        py_name: str,
        py_code: str,
        isa_name: str,
        isa_code: str,
        source_kind_python: str,
        source_kind_isabelle: str,
        tags: Optional[Sequence[str]] = None,
        notes: Optional[str] = None,
    ) -> JsonlRecord:
        params = self._lea_params()
        semantics = self.analyze_python_semantics(py_code)
        record_id = self.make_record_id(tier, component_name)

        instruction = {
            "prompt": (
                f"Translate the LEA-{self.block_size}/{self.key_size} {component_type} "
                f"from Python to Isabelle/HOL. Preserve LEA ARX semantics, 32-bit word behavior, "
                f"and the variant-specific structure."
            ),
            "source_language": "python",
            "target_language": "isabelle",
            "style_constraints": [
                "preserve naming alignment where reasonable",
                "preserve word-level semantics",
                "respect recursive/iterative orchestration structure",
            ],
        }

        return JsonlRecord(
            schema_version=SCHEMA_VERSION,
            record_id=record_id,
            dataset_split=self.dataset_split,
            task="code_translation",
            cipher=self.cipher,
            family=self.family,
            subfamily=self.subfamily,
            variant=self.variant_dict(),
            tier=tier,
            tier_role=tier_role,
            component_type=component_type,
            component_name=component_name,
            alignment_kind=alignment_kind,
            semantic_group=semantic_group,
            difficulty=difficulty,
            identity={
                "cipher": self.cipher,
                "variant_name": self.variant_dict()["name"],
                "component_name": component_name,
                "python_symbol": py_name,
                "isabelle_symbol": isa_name,
            },
            source_pair={
                "python": {
                    "symbol_name": py_name,
                    "source_kind": source_kind_python,
                    "code": py_code,
                    "file": str(self.python_file) if self.python_file else "",
                },
                "isabelle": {
                    "symbol_name": isa_name,
                    "source_kind": source_kind_isabelle,
                    "code": isa_code,
                    "file": str(self.thy_file) if self.thy_file else "",
                },
            },
            local_semantics={
                "python_analysis": semantics,
                "word_size": params["word_size"],
                "round_key_words": params["round_key_words"],
                "notes": notes or "",
            },
            structural_context={
                "variant_prefix": self._variant_prefix(),
                "block_size": self.block_size,
                "key_size": self.key_size,
                "word_size": params["word_size"],
                "block_words": params["block_words"],
                "key_words": params["key_words"],
                "rounds": params["rounds"],
            },
            translation_contract={
                "preserve_word_arithmetic": True,
                "preserve_bitwise_structure": True,
                "preserve_control_flow_intent": True,
                "expected_alignment": alignment_kind,
            },
            prompt_metadata={
                "tier_hint": tier,
                "tier_role": tier_role,
                "semantic_group": semantic_group,
                "difficulty": difficulty,
            },
            provenance={
                "python_file": str(self.python_file) if self.python_file else "",
                "thy_file": str(self.thy_file) if self.thy_file else "",
                "extracted_at": utc_now_iso(),
                "extractor": self.__class__.__name__,
            },
            quality={
                "is_direct_match": py_name == isa_name,
                "has_python_source": bool(py_code.strip()),
                "has_isabelle_source": bool(isa_code.strip()),
            },
            labels={
                "tags": list(tags or []),
                "family": self.family,
                "subfamily": self.subfamily,
                "tier": tier,
            },
            instruction=instruction,
        )

    def _extract_t1_constants(self) -> List[JsonlRecord]:
        # Corresponds to Section 3: Tiered Isomorphic Alignment - The TIA Hierarchical Taxonomy, T1 (Lexical Foundation): atomic constants
        # (word sizes, round counts, rotation amounts) as definition-aligned pairs.
        records: List[JsonlRecord] = []
        prefix = self._variant_prefix()
        upper = self._variant_upper_prefix()
        params = self._lea_params()

        specs = [
            (f"{upper}_WORD_SIZE", f"{prefix}_word_size", "word_size"),
            (f"{upper}_ROUNDS", f"{prefix}_rounds", "rounds"),
            (f"{upper}_KEY_WORDS", f"{prefix}_m", "key_words"),
            (f"{upper}_BLOCK_WORDS", f"{prefix}_block_words", "block_words"),
            (f"{upper}_MOD_MASK", f"{prefix}_mod_mask", "mod_mask"),
        ]

        for py_symbol, isa_symbol, logical_name in specs:
            py_code = self.extract_python_assignment(py_symbol)
            isa_code = self.extract_isabelle_definition(isa_symbol)
            if py_code and isa_code:
                records.append(self._make_record(
                    tier="T1",
                    tier_role="lexical_foundation",
                    component_type="constant",
                    component_name=logical_name,
                    alignment_kind="direct_constant_alignment",
                    semantic_group="constants",
                    difficulty="easy",
                    py_name=py_symbol,
                    py_code=py_code,
                    isa_name=isa_symbol,
                    isa_code=isa_code,
                    source_kind_python="constant",
                    source_kind_isabelle="definition",
                    tags=["constant", logical_name],
                ))

        delta_py = self.extract_python_assignment(f"{upper}_DELTA")
        delta_isa = self.extract_isabelle_definition(f"{prefix}_delta")
        if delta_py and delta_isa:
            records.append(self._make_record(
                tier="T1",
                tier_role="lexical_foundation",
                component_type="constant_table",
                component_name="delta_constants",
                alignment_kind="direct_constant_alignment",
                semantic_group="constants",
                difficulty="easy",
                py_name=f"{upper}_DELTA",
                py_code=delta_py,
                isa_name=f"{prefix}_delta",
                isa_code=delta_isa,
                source_kind_python="constant",
                source_kind_isabelle="definition",
                tags=["constant", "delta", "table"],
            ))

        return records

    def _extract_t2_primitives(self) -> List[JsonlRecord]:
        # Corresponds to Section 3: Tiered Isomorphic Alignment - The TIA Hierarchical Taxonomy, T2 (Functional Units): bitwise rotations,
        # modular addition, and S-box lookups as typed fun/definition pairs.
        records: List[JsonlRecord] = []
        prefix = self._variant_prefix()

        specs = [
            (f"{prefix}_rol", "rotation_left", "easy", "bitwise_primitives"),
            (f"{prefix}_ror", "rotation_right", "easy", "bitwise_primitives"),
            (f"{prefix}_encrypt_round", "encrypt_round", "medium", "round_operations"),
            (f"{prefix}_decrypt_round", "decrypt_round", "medium", "round_operations"),
        ]

        for symbol, component_name, difficulty, semantic_group in specs:
            py_code = self.extract_python_function(symbol)
            isa_code = self.extract_isabelle_definition(symbol)
            if py_code and isa_code:
                records.append(self._make_record(
                    tier="T2",
                    tier_role="functional_unit",
                    component_type="function",
                    component_name=component_name,
                    alignment_kind="functional_alignment",
                    semantic_group=semantic_group,
                    difficulty=difficulty,
                    py_name=symbol,
                    py_code=py_code,
                    isa_name=symbol,
                    isa_code=isa_code,
                    source_kind_python="function",
                    source_kind_isabelle="definition",
                    tags=["t2", component_name],
                ))

        return records

    def _extract_t3_key_schedule(self) -> List[JsonlRecord]:
        # Corresponds to Section 3: Tiered Isomorphic Alignment - The TIA Hierarchical Taxonomy, T3 (Structural Composition): key schedule steps
        # where Python for-loops map to Isabelle primrec/fold recursive definitions.
        records: List[JsonlRecord] = []
        prefix = self._variant_prefix()

        direct_specs = [
            (f"{prefix}_extract_key_words", "extract_key_words", "medium"),
            (f"{prefix}_step_key_expansion", "step_key_expansion", "hard"),
            (f"{prefix}_generate_round_keys", "generate_round_keys", "hard"),
        ]

        for symbol, component_name, difficulty in direct_specs:
            py_code = self.extract_python_function(symbol)
            isa_code = self.extract_isabelle_definition(symbol)
            if py_code and isa_code:
                records.append(self._make_record(
                    tier="T3",
                    tier_role="structural_component",
                    component_type="function",
                    component_name=component_name,
                    alignment_kind="structural_alignment",
                    semantic_group="key_schedule",
                    difficulty=difficulty,
                    py_name=symbol,
                    py_code=py_code,
                    isa_name=symbol,
                    isa_code=isa_code,
                    source_kind_python="function",
                    source_kind_isabelle="definition",
                    tags=["t3", "key_schedule", component_name],
                ))

        iter_symbol = f"{prefix}_gen_round_keys_iter"
        py_iter = self.extract_python_function(iter_symbol)
        isa_iter = self.extract_isabelle_definition(iter_symbol)
        if py_iter and isa_iter:
            records.append(self._make_record(
                tier="T3",
                tier_role="structural_component",
                component_type="helper_function",
                component_name="round_key_iteration",
                alignment_kind="structural_alignment",
                semantic_group="key_schedule",
                difficulty="hard",
                py_name=iter_symbol,
                py_code=py_iter,
                isa_name=iter_symbol,
                isa_code=isa_iter,
                source_kind_python="function",
                source_kind_isabelle="function",
                tags=["t3", "key_schedule", "iteration"],
            ))

        return records

    def _extract_t4_orchestration(self) -> List[JsonlRecord]:
        # Corresponds to Section 3: Tiered Isomorphic Alignment - The TIA Hierarchical Taxonomy, T4 (Top-Level Orchestration): complete cipher
        # interface and key schedule as fully recursive formal specifications.
        records: List[JsonlRecord] = []
        prefix = self._variant_prefix()

        specs = [
            (f"{prefix}_block_to_words", "block_to_words", "function", "easy", "block_conversion"),
            (f"{prefix}_words_to_block", "words_to_block", "function", "easy", "block_conversion"),
            (f"{prefix}_encrypt_iter", "encrypt_iteration", "function", "medium", "round_iteration"),
            (f"{prefix}_decrypt_iter", "decrypt_iteration", "function", "medium", "round_iteration"),
            (f"{prefix}_encrypt_block", "encrypt_block", "function", "medium", "block_cipher"),
            (f"{prefix}_decrypt_block", "decrypt_block", "function", "medium", "block_cipher"),
            (f"{prefix}_encrypt", "encrypt", "function", "medium", "top_level"),
            (f"{prefix}_decrypt", "decrypt", "function", "medium", "top_level"),
        ]

        for symbol, component_name, component_type, difficulty, semantic_group in specs:
            py_code = self.extract_python_function(symbol)
            isa_code = self.extract_isabelle_definition(symbol)
            if py_code and isa_code:
                records.append(self._make_record(
                    tier="T4",
                    tier_role="orchestration",
                    component_type=component_type,
                    component_name=component_name,
                    alignment_kind="orchestration_alignment",
                    semantic_group=semantic_group,
                    difficulty=difficulty,
                    py_name=symbol,
                    py_code=py_code,
                    isa_name=symbol,
                    isa_code=isa_code,
                    source_kind_python="function",
                    source_kind_isabelle="definition",
                    tags=["t4", semantic_group, component_name],
                ))

        return records