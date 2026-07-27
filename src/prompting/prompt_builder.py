#!/usr/bin/env python3
"""
src/prompting/prompt_builder.py

Shared prompt-assembly logic for Python -> Isabelle/HOL translation tasks.

Why this exists:
    Prompt formatting was previously embedded inside
    04_run_zero_shot_baseline.py. The fine-tuning data builder
    (03_build_finetune_data.py) needs byte-for-byte identical prompt
    construction so zero-shot and fine-tuned evaluation stay comparable.
    Moved here so both stages import one implementation instead of
    maintaining two copies that could silently drift apart.

Field mapping (actual dataset schema):
    metadata.variant        -- {'block_size': 48, 'key_size': 72}
    metadata.variant_params -- {'block_size':48,'word_size':24,'rounds':36,...}
    metadata.semantics      -- {'operators':['rotate','xor'],'contains_bitwise':True,...}
    metadata.transformations-- ['bitwise_to_word','rotate_conversion',...]
    metadata.tier           -- 'T1'|'T2'|'T3'|'T4'
    metadata.tier_role      -- 'variant_constant'|'primitive_transform'|...
    metadata.family         -- 'ARX'|'Feistel'|'SPN'
    metadata.cipher         -- 'SIMON'|'SPECK'|'SKINNY'|...

    NOTE: 'algorithm_params' does NOT exist in the dataset records.
    All metadata strategies must use 'variant_params' for cipher parameters.
"""

from __future__ import annotations

from typing import Any, Dict, List

from evaluation.text_normalization import clean_code


class EnhancedMetadataHandler:
    def __init__(self, strategy: str = "none") -> None:
        self.strategy = strategy

    def enrich(self, instruction: str, metadata: Dict[str, Any]) -> str:
        # Corresponds to Section 3: Tiered Isomorphic Alignment - Metadata Strategy Ablation: dispatches to one
        # of the five prompt information-density strategies evaluated in Tables 3 and 4.
        constraint = "Provide ONLY the Isabelle/HOL code. No explanations."
        if self.strategy == "none" or not metadata:
            return f"Task: {instruction}\n{constraint}"

        if self.strategy == "full":
            return self.enrich_full(instruction, metadata, constraint)
        if self.strategy == "structured":
            return self.enrich_structured(instruction, metadata, constraint)
        if self.strategy == "algorithmic":
            return self.enrich_algorithmic(instruction, metadata, constraint)
        if self.strategy == "alljson":
            return self.enrich_alljson(instruction, metadata, constraint)

        return f"Task: {instruction}\n{constraint}"

    def enrich_full(self, instruction: str, metadata: Dict[str, Any], constraint: str) -> str:
        # Implements the "Full" strategy (Section 3: Tiered Isomorphic Alignment - Metadata Strategy Ablation): all variant parameters, operators,
        # and transformation patterns — maximum information density.
        # Merge variant and variant_params for complete cipher spec
        tech_context: Dict[str, Any] = {}
        tech_context.update(metadata.get("variant", {}) or {})
        tech_context.update(metadata.get("variant_params", {}) or {})

        cipher_name = metadata.get("cipher", "Unknown")
        family = metadata.get("family", "Unknown")

        tech_bits = []
        for k, v in tech_context.items():
            clean_key = k.replace("_", " ").title()
            val_str = ", ".join(map(str, v)) if isinstance(v, list) else str(v)
            tech_bits.append(f"{clean_key}: {val_str}")

        # Append semantic operators and transformations for richer context
        semantics = metadata.get("semantics", {}) or {}
        operators = semantics.get("operators", [])
        if operators:
            tech_bits.append(f"Operators: {', '.join(operators)}")

        transformations = metadata.get("transformations", []) or []
        if transformations:
            tech_bits.append(f"Transformations: {', '.join(transformations)}")

        spec_sheet = f"Cipher {cipher_name} ({family}). " + "; ".join(tech_bits)
        return f"Technical Context: {spec_sheet}\nTask: {instruction}\n{constraint}"

    def enrich_structured(self, instruction: str, metadata: Dict[str, Any], constraint: str) -> str:
        # Implements the "Structured" strategy (Section 3: Tiered Isomorphic Alignment - Metadata Strategy Ablation): family-specific curated
        # parameter block (rotation constants for ARX/Feistel, S-box sizes for SPN).
        # Use variant_params (the actual field) for cipher parameters
        params = metadata.get("variant_params", {}) or {}
        variant = metadata.get("variant", {}) or {}
        cipher_name = metadata.get("cipher", "Unknown")
        family = metadata.get("family", "Unknown")

        if not params and not variant:
            return f"Task: {instruction}\n{constraint}"

        # word_size: prefer variant_params (richer) over variant
        word_size = params.get("word_size", variant.get("word_size", "NA"))
        block_size = variant.get("block_size", params.get("block_size", "NA"))

        priority_params = [
            ("Cipher", f"{cipher_name} ({family})"),
            ("Block Size", block_size),
            ("Word Size", word_size),
        ]

        if family in {"ARX", "Feistel"}:
            for key in ["alpha_rotation", "beta_rotation", "rotation_constants"]:
                if key in params:
                    priority_params.append((key.replace("_", " ").title(), params[key]))
            if "arx_order" in params:
                priority_params.append(("Operation Sequence", params["arx_order"]))
            # f_function absent in current dataset; semantics.operators carries this info
            semantics = metadata.get("semantics", {}) or {}
            operators = semantics.get("operators", [])
            if operators:
                priority_params.append(("Operators", ", ".join(operators)))

        elif family == "SPN":
            if "sbox_size" in params:
                priority_params.append(("S-Box Size", f'{params["sbox_size"]}-bit'))
            if "sbox_count" in params:
                priority_params.append(("S-Box Count", params["sbox_count"]))
            if "permutation_type" in params:
                priority_params.append(("Permutation", params["permutation_type"]))
            nibble_val = variant.get("nibbles_per_block") or variant.get("nibblesperblock")
            if nibble_val:
                priority_params.append(("Nibble Count", nibble_val))

        if "rounds" in params:
            priority_params.append(("Total Rounds", params["rounds"]))
        elif "rounds" in variant:
            priority_params.append(("Total Rounds", variant["rounds"]))
        elif "steps" in variant:
            priority_params.append(("Step Structure", variant["steps"]))

        component_type = metadata.get("tier_role") or metadata.get("component_type")
        if component_type:
            priority_params.append(("Component Role", component_type))

        tier = metadata.get("tier")
        if tier:
            priority_params.append(("Tier", tier))

        param_lines = []
        for key, value in priority_params:
            val_str = ", ".join(map(str, value)) if isinstance(value, list) else str(value)
            param_lines.append(f"- {key}: {val_str}")

        param_block = "\n".join(param_lines)
        return f"Cryptographic Context:\n{param_block}\nTask: {instruction}\n{constraint}"

    def enrich_algorithmic(self, instruction: str, metadata: Dict[str, Any], constraint: str) -> str:
        # Implements the "Algorithmic" strategy (Section 3: Tiered Isomorphic Alignment - Metadata Strategy Ablation): compact word-level
        # precision and operation-set annotation.
        # Use variant_params (not algorithm_params which does not exist in the dataset)
        params = metadata.get("variant_params", {}) or {}
        semantics = metadata.get("semantics", {}) or {}
        transformations = metadata.get("transformations", []) or []

        core_info: List[str] = []

        # Word-level arithmetic precision
        if "word_size" in params:
            core_info.append(f'{params["word_size"]}b')

        # Operational primitives from semantics
        operators = semantics.get("operators", [])
        if operators:
            core_info.append(", ".join(operators))

        # Code-structure transformations applied to this component
        if transformations:
            core_info.append(", ".join(transformations))

        # Round count for structural context
        if "rounds" in params:
            core_info.append(f'{params["rounds"]}r')

        if core_info:
            spec_line = "; ".join(core_info)
            return f"Algorithmic Hint: {spec_line}\nTask: {instruction}\n{constraint}"

        # Minimal fallback: at least provide family + tier role so this never
        # collapses silently to the 'none' prompt when variant_params is sparse
        family = metadata.get("family", "")
        tier_role = metadata.get("tier_role", metadata.get("tier", ""))
        if family or tier_role:
            hint = " | ".join(x for x in [family, tier_role] if x)
            return f"Algorithmic Hint: {hint}\nTask: {instruction}\n{constraint}"

        return f"Task: {instruction}\n{constraint}"

    def enrich_alljson(self, instruction: str, metadata: Dict[str, Any], constraint: str) -> str:
        # Implements the "Alljson" strategy (Section 3: Tiered Isomorphic Alignment - Metadata Strategy Ablation): flattened JSON of the full
        # metadata record — maximum information density at the cost of noise and verbosity.
        def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                elif isinstance(v, list):
                    items.append((new_key, ", ".join(map(str, v))))
                else:
                    items.append((new_key, v))
            return dict(items)

        flat_metadata = flatten_dict(metadata)
        spec_lines = []
        for key, value in flat_metadata.items():
            clean_key = key.replace("_", " ").replace(".", " / ").title()
            spec_lines.append(f"- {clean_key}: {value}")
        spec_block = "\n".join(spec_lines)
        return f"Complete Metadata Specification:\n{spec_block}\nTask: {instruction}\n{constraint}"


def example_identifier(example: Dict[str, Any]) -> Any:
    """Best available stable identifier for a TIA dataset record.

    The dataset schema does not currently populate "id"/"example_id" (they
    are always null in practice). Falls back to the extractor-assigned
    isabelle_symbol (unique per component), so per-example provenance --
    e.g. which few-shot support examples were used -- stays traceable
    without needing a real ID field. Shared because this exact fallback
    logic was about to be copy-pasted into a third script.
    """
    identifier = example.get("id", example.get("example_id"))
    if identifier is not None:
        return identifier
    metadata = example.get("metadata", {}) or {}
    return metadata.get("isabelle_symbol") or metadata.get("python_symbol")


def build_translation_prompt(example: Dict[str, Any], metadata_strategy: str) -> str:
    """Assemble the Python -> Isabelle/HOL prompt prefix, ending at '### Output:\\n'.

    Shared by zero-shot inference (04_run_zero_shot_baseline.py) and
    fine-tuning data construction (03_build_finetune_data.py) so the two
    settings never see different prompt text for the same example.
    """
    instruction = example.get("instruction", "").strip()
    input_text = example.get("input", "").strip()
    metadata = example.get("metadata", {}) or {}

    handler = EnhancedMetadataHandler(metadata_strategy)
    task_prefix = handler.enrich(instruction, metadata)

    return (
        "You are translating Python cryptographic code into Isabelle/HOL.\n"
        "Return ONLY the Isabelle/HOL code.\n"
        "Do not include explanations, markdown fences, or extra commentary.\n\n"
        f"{task_prefix}\n\n"
        f"### Input:\n{input_text}\n\n"
        "### Output:\n"
    )


def build_few_shot_prompt(
    query_example: Dict[str, Any],
    support_examples: List[Dict[str, Any]],
    metadata_strategy: str,
) -> str:
    """Few-shot variant of `build_translation_prompt`.

    Prepends K worked (Python input -> Isabelle/HOL output) demonstrations
    before the final query. Demonstrations use the same metadata_strategy
    as the query so the prompt style stays internally consistent. Callers
    should use `build_translation_prompt` directly when there are no
    support examples (k=0) rather than calling this with an empty list, so
    the pure zero-shot prompt text never changes.
    """
    handler = EnhancedMetadataHandler(metadata_strategy)

    blocks: List[str] = []
    for support in support_examples:
        support_instruction = support.get("instruction", "").strip()
        support_input = support.get("input", "").strip()
        support_output = clean_code(support.get("output", ""))
        support_prefix = handler.enrich(support_instruction, support.get("metadata", {}) or {})
        blocks.append(
            f"{support_prefix}\n\n"
            f"### Input:\n{support_input}\n\n"
            f"### Output:\n{support_output}"
        )

    demonstrations = "\n\n".join(blocks)

    query_instruction = query_example.get("instruction", "").strip()
    query_input = query_example.get("input", "").strip()
    query_prefix = handler.enrich(query_instruction, query_example.get("metadata", {}) or {})

    header = (
        "You are translating Python cryptographic code into Isabelle/HOL.\n"
        "Below are worked examples, followed by a new task. Return ONLY the "
        "Isabelle/HOL code for the new task.\n"
        "Do not include explanations, markdown fences, or extra commentary.\n\n"
    )
    if demonstrations:
        header += demonstrations + "\n\n---\n\n"

    return (
        header
        + f"{query_prefix}\n\n"
        + f"### Input:\n{query_input}\n\n"
        + "### Output:\n"
    )
