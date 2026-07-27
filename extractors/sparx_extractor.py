# sparx_extractor.py

from .base_extractor import BaseCipherExtractor
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import re


class SparxExtractor(BaseCipherExtractor):
    # Cipher-specific implementation of Algorithm 1 (Section 3: Tiered Isomorphic Alignment - Dataset Construction and Registry) for the SPARX
    # family (ARX). Produces T1–T4 aligned Python ↔ Isabelle/HOL component pairs.
    """Source-aligned SPARX extractor with explicit tier metadata."""

    def __init__(
        self,
        root_dir: str,
        cipher: str,
        family: str,
        subfamily: str,
        block_size: int,
        key_size: int,
        variant_config: dict,
        dataset_split: str = "train",
        # Legacy parameters for backward compatibility with run_sparx_dataset.py
        cipher_name: str = None,
        cipher_info: dict = None,
        py_content: str = None,
        thy_content: str = None,
    ):
        # Handle legacy calls from run_sparx_dataset.py
        if cipher_name is not None:
            cipher = cipher_name
        if cipher_info is not None and isinstance(cipher_info, dict):
            family = cipher_info.get("cipher_family", family)
        
        # Call parent with correct signature
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
        
        # Override content if provided directly (for legacy compatibility)
        if py_content is not None:
            self.python_source = py_content
            self.python_ast = self._safe_parse_python(py_content)
        if thy_content is not None:
            self.thy_source = thy_content
    
    def extract_components(self) -> List[Dict]:
        # Implements the outer loop of Algorithm 1 (Section 3: Tiered Isomorphic Alignment - Dataset Construction and Registry): iterates over all
        # four TIA tiers and collects aligned component records for this cipher variant.
        examples: List[Dict] = []

        examples.extend(self._extract_t1_constants())
        examples.extend(self._extract_t2_primitives())
        examples.extend(self._extract_t3_structural_components())
        examples.extend(self._extract_t4_orchestration_components())

        return [ex for ex in examples if ex is not None]

    def create_example(
        self,
        template: Dict,
        py_code: str,
        thy_code: str,
        name: str = "",
    ) -> Optional[Dict]:
        if not py_code or not thy_code:
            return None
    
        py_code = py_code.strip()
        thy_code = thy_code.strip()
        
        # Use base class semantic analysis
        semantics = self.analyze_python_semantics(py_code)
    
        if template.get("force_no_loops"):
            semantics["has_loops"] = False
    
        transformations = self.get_transformation_rules(template["type"])
        instruction = self._create_instruction(template, semantics, transformations)
        metadata = self._create_metadata(template, semantics, transformations)
    
        # Get variant parameters
        params = self._get_sparx_params()
        
        metadata.update(
            {
                "tier": template.get("tier"),
                "tier_role": template.get("tier_role"),
                "semantic_name": template.get("semantic_name"),
                "python_symbol": template.get("python_symbol"),
                "python_resolution": template.get("python_resolution"),
                "isabelle_symbol": template.get("isabelle_symbol"),
                "isabelle_resolution": template.get("isabelle_resolution"),
                "component_name": name or template.get("semantic_name") or template.get("type"),
                "cipher": "SPARX",
                "family": "ARX",
                "variant": {
                    "block_size": self.block_size,
                    "key_size": self.key_size,
                    **params,  # Use the params dict
                },
                "source_aligned": True,
                "synthetic_python": template.get("synthetic_python", False),
                "synthetic_isabelle": template.get("synthetic_isabelle", False),
                "extraction_time": datetime.now().isoformat(),
            }
        )
    
        return {
            "instruction": instruction,
            "input": py_code,
            "output": thy_code,
            "metadata": metadata,
        }

    def get_transformation_rules(self, component_type: str) -> List[str]:
        rules: List[str] = []
        lowered = component_type.lower()

        if "constant" in lowered:
            return []
        if "rotation" in lowered:
            rules.extend(["bitwise_to_word", "rotate_conversion"])
        if "permutation" in lowered or "round" in lowered:
            rules.append("bitwise_to_word")
        if (
            "schedule" in lowered
            or "encrypt" in lowered
            or "decrypt" in lowered
            or "iterate" in lowered
        ):
            rules.append("loop_to_recursion")
        if "conversion" in lowered or "block" in lowered:
            rules.append("bitwise_to_word")

        return rules

    def _transformation_to_hint(self, transformation: str) -> str:
        """Convert a transformation rule to a human-readable hint."""
        hint_map = {
            "bitwise_to_word": "use bitwise word operations",
            "rotate_conversion": "convert Python rotations to word_rotl/word_rotr",
            "loop_to_recursion": "convert iterative loops to recursive functions",
            "bitwise_to_word": "use Isabelle word library operations",
            "constant_propagation": "inline constants where appropriate",
            "type_annotation": "add explicit word type annotations",
        }
        return hint_map.get(transformation, transformation.replace("_", " "))
    
    def _tier_name(self, tier: str) -> str:
        """Get human-readable tier name."""
        tier_names = {
            "T1": "Lexical Foundation",
            "T2": "Functional Unit",
            "T3": "Structural Composition",
            "T4": "Top-Level Orchestration",
        }
        return tier_names.get(tier, tier)
    
    def _transformation_rules_for_type(self, component_type: str) -> List[str]:
        """Get transformation rules specific to component type."""
        rules = []
        lowered = component_type.lower()
        
        if "constant" in lowered:
            return []
        if "rotation" in lowered:
            rules.extend(["bitwise_to_word", "rotate_conversion"])
        if "permutation" in lowered or "round" in lowered:
            rules.append("bitwise_to_word")
        if "schedule" in lowered or "encrypt" in lowered or "decrypt" in lowered:
            rules.append("loop_to_recursion")
        if "conversion" in lowered or "block" in lowered:
            rules.append("bitwise_to_word")
        
        return rules
        

    def _create_instruction(
        self,
        template: Dict,
        semantics: Dict,
        transformations: List[str],
    ) -> str:
        base = (
            f"Translate the SPARX-{self.block_size}/{self.key_size} "
            f"{template['type']} from Python to Isabelle/HOL."
        )

        tier = template.get("tier")
        if tier:
            base += f" This is a {tier} component."

        if transformations:
            hints = [self._transformation_to_hint(t) for t in transformations[:3]]
            base += f" Apply these transformations: {', '.join(hints)}."

        if semantics.get("has_loops"):
            base += " Convert loops to recursion where appropriate."

        semantic_name = template.get("semantic_name")
        if semantic_name:
            base += f" Preserve the semantics of {semantic_name}."

        return base

    def _create_metadata(
        self,
        template: Dict,
        semantics: Dict,
        transformations: List[str],
    ) -> Dict:
        params = self._get_sparx_params()
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

    def _get_variant_prefix(self) -> str:
        """Get the Isabelle/HOL prefix for this variant."""
        return f"sparx_{self.block_size}_{self.key_size}"

    def _get_variant_prefix_upper(self) -> str:
        """Get the Python constant prefix for this variant (uppercase)."""
        return f"SPARX_{self.block_size}_{self.key_size}"
    

    def _get_sparx_params(self) -> Dict[str, int]:
        if self.block_size == 64 and self.key_size == 128:
            return {
                "word_size": 16,
                "branches": 2,        # was n_branches
                "words_per_block": 4, # was n_words
                "steps": 8,           # was n_steps
                "rounds_per_step": 3,
                "total_rounds": 24,
                "key_words": 8,
                "round_key_words": 2,
                # Keep old names for backward compatibility
                "n_branches": 2,
                "n_words": 4,
                "n_steps": 8,
            }
        if self.block_size == 128 and self.key_size == 128:
            return {
                "word_size": 16,
                "branches": 4,
                "words_per_block": 8,
                "steps": 8,
                "rounds_per_step": 4,
                "total_rounds": 32,
                "key_words": 8,
                "round_key_words": 4,
                "n_branches": 4,
                "n_words": 8,
                "n_steps": 8,
            }
        if self.block_size == 128 and self.key_size == 256:
            return {
                "word_size": 16,
                "branches": 4,
                "words_per_block": 8,
                "steps": 10,
                "rounds_per_step": 4,
                "total_rounds": 40,
                "key_words": 16,
                "round_key_words": 8,
                "n_branches": 4,
                "n_words": 8,
                "n_steps": 10,
            }
        raise ValueError(f"Unsupported SPARX variant: {self.block_size}/{self.key_size}")

    def _extract_python_function(self, function_name: str) -> Optional[str]:
        """Extract a top-level function from Python source."""
        lines = self.python_source.split("\n")  # Changed from self.py_content
        result: List[str] = []
        in_function = False
        indent_level = -1
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            current_indent = len(line) - len(line.lstrip())
            
            if not in_function:
                if stripped.startswith(f"def {function_name}(") or stripped == f"def {function_name}():":
                    in_function = True
                    indent_level = current_indent
                    result.append(line)
                    continue
            
            if in_function:
                if i > 0:
                    if stripped and current_indent <= indent_level:
                        if not stripped.startswith("#"):
                            if stripped.startswith("def "):
                                break
                            if stripped.startswith("class "):
                                break
                
                result.append(line)
                
                if stripped.startswith("@") and current_indent == 0:
                    break
        
        code = "\n".join(result).strip()
        return code if code else None

        
    def _extract_python_assignment(self, name: str) -> Optional[str]:
        """Extract a top-level assignment from Python source."""
        # Use python_source from base class, not py_content
        pattern = re.compile(rf"^\s*{re.escape(name)}\s*=\s*.+$", re.MULTILINE)
        match = pattern.search(self.python_source)  # Changed from self.py_content
        return match.group(0).strip() if match else None

        
    def _find_isabelle(self, *possible_names: str) -> Optional[Tuple[str, str]]:
        """Find Isabelle definition by trying multiple names."""
        for name in possible_names:
            # Use base class method that works with thy_source
            code = self.extract_isabelle_definition(name)
            if code:
                return name, code
        return None

        
    def _constant_template(
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

    def _function_template(
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

    def _extract_t1_constants(self) -> List[Dict]:
        # Corresponds to Section 3: Tiered Isomorphic Alignment - The TIA Hierarchical Taxonomy, T1 (Lexical Foundation): extracts atomic values
        # (word sizes, round counts, rotation amounts) as definition-aligned pairs.
        prefix = self._get_variant_prefix()
        params = self._get_sparx_params()
        examples: List[Dict] = []
    
        upper = self._get_variant_prefix_upper()
        specs = [
            ("block_size", f"{upper}_BLOCK_SIZE", f"{prefix}_block_size", "Block Size Constant"),
            ("key_size", f"{upper}_KEY_SIZE", f"{prefix}_key_size", "Key Size Constant"),
            ("n_steps", f"{upper}_N_STEPS", f"{prefix}_n_steps", "Number of Steps Constant"),
            ("rounds_per_step", f"{upper}_ROUNDS_PER_STEP", f"{prefix}_rounds_per_step", "Rounds Per Step Constant"),
            ("word_size", f"{upper}_WORD_SIZE", f"{prefix}_word_size", "Word Size Constant"),
            ("n_branches", f"{upper}_N_BRANCHES", f"{prefix}_n_branches", "Number of Branches Constant"),
            ("n_words", f"{upper}_N_WORDS", f"{prefix}_n_words", "Number of Words Constant"),
            ("round_key_words", f"{upper}_ROUND_KEY_WORDS", f"{prefix}_round_key_words", "Round Key Words Constant"),
        ]

        for semantic_name, py_symbol, isa_symbol, type_name in specs:
            py_value = self._extract_python_assignment(py_symbol)
            if not py_value:
                continue
            found = self._find_isabelle(isa_symbol)
            if not found:
                continue
            thy_symbol, thy_code = found
            template = self._constant_template(
                type_name=type_name,
                semantic_name=semantic_name,
                python_symbol=py_symbol,
                python_resolution="source_assignment",
                isabelle_symbol=thy_symbol,
                isabelle_resolution="definition",
            )
            example = self.create_example(template, py_value, thy_code, semantic_name)
            if example:
                examples.append(example)

        return examples

    def _extract_t2_primitives(self) -> List[Dict]:
        # Corresponds to Section 3: Tiered Isomorphic Alignment - The TIA Hierarchical Taxonomy, T2 (Functional Units): extracts bitwise rotations,
        # modular addition, and S-box lookups as typed fun/definition pairs.
        prefix = self._get_variant_prefix()
        params = self._get_sparx_params()
        examples: List[Dict] = []

        specs = [
            {
                "type": "Rotation Left Primitive",
                "semantic_name": "rol",
                "python_names": [f"{prefix}_rol"],
                "isabelle_names": [f"{prefix}_rol"],
                "group": "bitwise_operations",
                "difficulty": "easy",
            },
            {
                "type": "Rotation Right Primitive",
                "semantic_name": "ror",
                "python_names": [f"{prefix}_ror"],
                "isabelle_names": [f"{prefix}_ror"],
                "group": "bitwise_operations",
                "difficulty": "easy",
            },
            {
                "type": "A Permutation",
                "semantic_name": "a_perm",
                "python_names": [f"{prefix}_a_perm"],
                "isabelle_names": [f"{prefix}_a_perm"],
                "group": "round_operations",
                "difficulty": "medium",
            },
            {
                "type": "A Permutation Inverse",
                "semantic_name": "a_perm_inv",
                "python_names": [f"{prefix}_a_perm_inv"],
                "isabelle_names": [f"{prefix}_a_perm_inv"],
                "group": "round_operations",
                "difficulty": "medium",
            },
            {
                "type": "Linear Word Function",
                "semantic_name": "l_w",
                "python_names": [f"{prefix}_l_w"],
                "isabelle_names": [f"{prefix}_l_w"],
                "group": "diffusion",
                "difficulty": "easy",
            },
            {
                "type": "Linear Layer",
                "semantic_name": "linear_layer",
                "python_names": [f"{prefix}_linear_layer"],
                "isabelle_names": [f"{prefix}_linear_layer"],
                "group": "diffusion",
                "difficulty": "medium",
            },
            {
                "type": "Linear Layer Inverse",
                "semantic_name": "linear_layer_inv",
                "python_names": [f"{prefix}_linear_layer_inv"],
                "isabelle_names": [f"{prefix}_linear_layer_inv"],
                "group": "diffusion",
                "difficulty": "medium",
            },
            {
                "type": "Apply Encrypt Round",
                "semantic_name": "apply_encrypt_round",
                "python_names": [f"{prefix}_apply_encrypt_round"],
                "isabelle_names": [f"{prefix}_apply_encrypt_round"],
                "group": "round_operations",
                "difficulty": "medium",
            },
            {
                "type": "Apply Decrypt Round",
                "semantic_name": "apply_decrypt_round",
                "python_names": [f"{prefix}_apply_decrypt_round"],
                "isabelle_names": [f"{prefix}_apply_decrypt_round"],
                "group": "round_operations",
                "difficulty": "medium",
            },
        ]

        for spec in specs:
            py_name: Optional[str] = None
            py_code: Optional[str] = None
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

            template = self._function_template(
                type_name=spec["type"],
                tier="T2",
                tier_role="primitive_transform",
                semantic_group=spec["group"],
                semantic_name=spec["semantic_name"],
                python_symbol=py_name,
                python_resolution="top_level_function",
                isabelle_symbol=thy_symbol,
                isabelle_resolution="definition",
                difficulty=spec["difficulty"],
            )
            example = self.create_example(
                template, py_code, thy_code, spec["semantic_name"]
            )
            if example:
                examples.append(example)

        return examples

    def _extract_t3_structural_components(self) -> List[Dict]:
        # Corresponds to Section 3: Tiered Isomorphic Alignment - The TIA Hierarchical Taxonomy, T3 (Structural Composition): extracts round
        # functions and key schedule steps where Python for-loops map to Isabelle
        # primrec/fold recursive definitions.
        prefix = self._get_variant_prefix()
        examples: List[Dict] = []

        specs = [
            {
                "type": "Key Extraction",
                "semantic_name": "extract_key_words",
                "python_names": [f"{prefix}_extract_key_words"],
                "isabelle_names": [f"{prefix}_extract_key_words"],
                "group": "key_expansion",
                "difficulty": "medium",
            },
            {
                "type": "Key State Permutation",
                "semantic_name": "k_perm",
                "python_names": [f"{prefix}_k_perm"],
                "isabelle_names": [f"{prefix}_k_perm"],
                "group": "key_expansion",
                "difficulty": "hard",
            },
            {
                "type": "Key Schedule Recursive Step",
                "semantic_name": "gen_key_schedule_iterate",
                "python_names": [f"{prefix}_gen_key_schedule_iterate"],
                "isabelle_names": [f"{prefix}_gen_key_schedule_iterate"],
                "group": "key_expansion",
                "difficulty": "hard",
            },
            {
                "type": "Key Schedule",
                "semantic_name": "generate_key_schedule",
                "python_names": [f"{prefix}_generate_key_schedule"],
                "isabelle_names": [f"{prefix}_generate_key_schedule"],
                "group": "key_expansion",
                "difficulty": "medium",
            },
        ]

        for spec in specs:
            py_name: Optional[str] = None
            py_code: Optional[str] = None
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

            template = self._function_template(
                type_name=spec["type"],
                tier="T3",
                tier_role="structural_transform",
                semantic_group=spec["group"],
                semantic_name=spec["semantic_name"],
                python_symbol=py_name,
                python_resolution="top_level_function",
                isabelle_symbol=thy_symbol,
                isabelle_resolution="definition",
                difficulty=spec["difficulty"],
            )
            example = self.create_example(
                template, py_code, thy_code, spec["semantic_name"]
            )
            if example:
                examples.append(example)

        return examples

    def _extract_t4_orchestration_components(self) -> List[Dict]:
        # Corresponds to Section 3: Tiered Isomorphic Alignment - The TIA Hierarchical Taxonomy, T4 (Top-Level Orchestration): extracts the complete
        # cipher interface and key schedule as fully recursive formal specifications.
        prefix = self._get_variant_prefix()
        examples: List[Dict] = []

        specs = [
            {
                "type": "Block to Words Conversion",
                "semantic_name": "block_to_words",
                "python_names": [f"{prefix}_block_to_words"],
                "isabelle_names": [f"{prefix}_block_to_words"],
                "group": "data_conversion",
                "difficulty": "easy",
            },
            {
                "type": "Words to Block Conversion",
                "semantic_name": "words_to_block",
                "python_names": [f"{prefix}_words_to_block"],
                "isabelle_names": [f"{prefix}_words_to_block"],
                "group": "data_conversion",
                "difficulty": "easy",
            },
            {
                "type": "Encrypt Round Iteration",
                "semantic_name": "encrypt_round_iterate",
                "python_names": [f"{prefix}_encrypt_round_iterate"],
                "isabelle_names": [f"{prefix}_encrypt_round_iterate"],
                "group": "round_iteration",
                "difficulty": "hard",
            },
            {
                "type": "Decrypt Round Iteration",
                "semantic_name": "decrypt_round_iterate",
                "python_names": [f"{prefix}_decrypt_round_iterate"],
                "isabelle_names": [f"{prefix}_decrypt_round_iterate"],
                "group": "round_iteration",
                "difficulty": "hard",
            },
            {
                "type": "Encrypt Step Iteration",
                "semantic_name": "encrypt_step_iterate",
                "python_names": [f"{prefix}_encrypt_step_iterate"],
                "isabelle_names": [f"{prefix}_encrypt_step_iterate"],
                "group": "step_iteration",
                "difficulty": "medium",
            },
            {
                "type": "Decrypt Step Iteration",
                "semantic_name": "decrypt_step_iterate",
                "python_names": [f"{prefix}_decrypt_step_iterate"],
                "isabelle_names": [f"{prefix}_decrypt_step_iterate"],
                "group": "step_iteration",
                "difficulty": "medium",
            },
            {
                "type": "Encrypt Steps Iteration",
                "semantic_name": "encrypt_steps_iterate",
                "python_names": [f"{prefix}_encrypt_steps_iterate"],
                "isabelle_names": [f"{prefix}_encrypt_steps_iterate"],
                "group": "step_iteration",
                "difficulty": "hard",
            },
            {
                "type": "Decrypt Steps Iteration",
                "semantic_name": "decrypt_steps_iterate",
                "python_names": [f"{prefix}_decrypt_steps_iterate"],
                "isabelle_names": [f"{prefix}_decrypt_steps_iterate"],
                "group": "step_iteration",
                "difficulty": "hard",
            },
            {
                "type": "Encrypt Block",
                "semantic_name": "encrypt_block",
                "python_names": [f"{prefix}_encrypt_block"],
                "isabelle_names": [f"{prefix}_encrypt_block"],
                "group": "block_cipher",
                "difficulty": "medium",
            },
            {
                "type": "Decrypt Block",
                "semantic_name": "decrypt_block",
                "python_names": [f"{prefix}_decrypt_block"],
                "isabelle_names": [f"{prefix}_decrypt_block"],
                "group": "block_cipher",
                "difficulty": "medium",
            },
            {
                "type": "Top-Level Encrypt",
                "semantic_name": "encrypt",
                "python_names": [f"{prefix}_encrypt"],
                "isabelle_names": [f"{prefix}_encrypt"],
                "group": "top_level",
                "difficulty": "medium",
            },
            {
                "type": "Top-Level Decrypt",
                "semantic_name": "decrypt",
                "python_names": [f"{prefix}_decrypt"],
                "isabelle_names": [f"{prefix}_decrypt"],
                "group": "top_level",
                "difficulty": "medium",
            },
        ]

        for spec in specs:
            py_name: Optional[str] = None
            py_code: Optional[str] = None
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

            template = self._function_template(
                type_name=spec["type"],
                tier="T4",
                tier_role="orchestration",
                semantic_group=spec["group"],
                semantic_name=spec["semantic_name"],
                python_symbol=py_name,
                python_resolution="top_level_function",
                isabelle_symbol=thy_symbol,
                isabelle_resolution="definition",
                difficulty=spec["difficulty"],
            )
            example = self.create_example(
                template, py_code, thy_code, spec["semantic_name"]
            )
            if example:
                examples.append(example)

        return examples