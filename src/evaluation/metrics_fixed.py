
"""
metrics_fixed.py
Cryptographic Translation Metrics

"""

import re
import difflib
from typing import List, Dict, Set, Tuple, Any
import numpy as np
from collections import Counter
import difflib


class FixedCryptographicMetrics:
    """
    Fixed metrics that actually measure cryptographic translation quality.
    """
    
    def __init__(self):
        # COMPLETE operator sets based on dataset
        self.crypto_operators = {
            # Bitwise operations
            'xor', 'and', 'or', 'not',
            
            # Rotations
            'word_rotl', 'word_rotr',
            
            # Arithmetic operations (CRITICAL - was missing!)
            '+', '-', '*', 'div', 'mod', '=',
            
            # Type conversions
            'word_of_int', 'int_of_word', 'word_of_nat', 'of_nat',
            'ucast',
            
            # Bit operations
            'push_bit', 'drop_bit', 'take_bit', 'bit',
            'shiftl', 'shiftr',  # Alternative names
            
            # List operations (Feistel)
            'hd', 'tl', '#', '@', '!',
            'take', 'drop', 'rev',
            
            # SPN specific
            'sbox', 'permute', 'sbox_lookup',
            
            # Higher-order functions
            'foldl', 'foldr', 'map', 'fold',  
            
            # Power/Exponentiation (in PRESENT)
            '^',  # Power operator in Isabelle
            
            # Set operations (in masks)
            'mask',
            
            # Other common operators 
            'sum',  # Σ in Isabelle
            'uint',  # Unsigned int conversion
            'unat',  # Unsigned nat conversion
        }
        
        # Special symbols that are operators in Isabelle
        self.symbolic_operators = {
            '+', '-', '*', '/', '=', '≠', '<', '>', '≤', '≥',
            '⇒', '→', '↔', '∧', '∨', '¬', '∀', '∃',
            ':', '::', '#', '@', '!', '^', '~'
        }
        
        # Common Isabelle functions that should NOT be treated as operators
        self.non_operators = {
            # Pattern matching (SPN)
            'case', 'of',
            # Control flow
            'if', 'then', 'else', 'let', 'in',
            'length',
            'definition', 'fun', 'function', 'primrec', 'lemma', 'theorem',
            'where', 'imports', 'begin', 'end', 'theory',
            'type_synonym', 'datatype', 'record', 'locale',
            'text', 'value', 'section', 'subsection',
            'apply', 'by', 'using', 'from', 'with',
            'proof', 'qed', 'show', 'have', 'assume',
            'unfolding', 'simp', 'auto', 'arith', 'blast',
            'induction', 'cases', 'rule', 'intro', 'elim'
        }


    
    def syntax_validity(self, generated: str, reference: str) -> float:
        # Corresponds to Section 4: Evaluation Framework, Eq. 1: SV = F_basic × F_align
        #print(f"\n=== DEBUG SV ===")
        #print(f"Generated: {generated}")
        #print(f"Reference: {reference}")
        
        # 1. BASIC VALIDITY
        basic_score = self._basic_syntax_validity(generated)
        #print(f"basic_score: {basic_score}")
        
        if basic_score == 0.0:
            return 0.0
        
        # 2. STYLE ALIGNMENT
        style_score = self._style_alignment(generated, reference)
        #print(f"style_score: {style_score}")
        
        # 3. COMBINE
        final = basic_score * style_score
        #print(f"final SV: {final}")
        return final
    
    def _extract_command(self, code: str):
        #print(f"\n_extract_command called with: '{code[:50]}...'")
        commands = ["definition", "fun", "function", "primrec"]
        for line in code.split('\n'):
            line = line.strip()
            #print(f"  Checking line: '{line}'")
            if line.startswith('(*') and line.endswith('*)'):
                #print(f"    Skipping: comment-only line")
                continue
            if line.startswith('(*'):
                #print(f"    Skipping: starts with comment")
                continue
                
            for cmd in commands:
                if line.startswith(cmd + ' ') or line == cmd:
                    #print(f"    Found command: {cmd}")
                    return cmd
        print(f"  No command found")
        return None

    
    def _basic_syntax_validity(self, code: str) -> float:
        # Implements F_basic in Eq. 1 (Section 4: Evaluation Framework): stack-based balanced-delimiter
        # verification of a mandatory Isabelle entry-point (definition/fun/primrec/lemma).
        """Check if code has valid Isabelle syntax structure."""
        clean_code = self._remove_comments(code.strip())
        valid_commands = ["definition", "fun", "function", "primrec", "lemma"]
        
        lines = [l.strip() for l in clean_code.split('\n') if l.strip()]
        first_command = None
        start_idx = -1
        
        for i, line in enumerate(lines):
            for cmd in valid_commands:
                if line.startswith(cmd + ' ') or line == cmd:
                    first_command = cmd
                    start_idx = i
                    break
            if first_command: 
                break
        
        if not first_command: 
            return 0.0
        
        code_block = "\n".join(lines[start_idx:])
        
        if not self._check_balanced_parentheses(code_block):
            return 0.0
        
        # Simplified: Just check for basic structure
        if first_command == "definition" and 'where' not in code_block:
            return 0.5
        
        if first_command in ["fun", "function", "primrec"]:
            if 'where' not in code_block and '=' not in code_block:
                return 0.5
        
        return 1.0  # All basic checks pass

    def _style_alignment(self, generated: str, reference: str) -> float:
        # Implements F_align in Eq. 1 (Section 4: Evaluation Framework): command similarity (0.4 weight) +
        # Isabelle-specific marker coverage (0.6 weight) − 0.3 per detected foreign
        # anti-pattern (Python def, C-style braces, SML-style val).
        """Isolate Isabelle from SML/Pseudocode hallucinations."""
        gen_clean = self._remove_comments(generated)
        ref_clean = self._remove_comments(reference)

        #print(f"gen_clean AFTER _remove_comments: {gen_clean} " )
        #print(f"ref_clean AFTER _remove_comments: {ref_clean} " )
        
        gen_cmd = self._extract_command(gen_clean)
        ref_cmd = self._extract_command(ref_clean)
        
        if not ref_cmd: return 0.7
        
        # Command alignment
        if gen_cmd == ref_cmd:
            cmd_score = 1.0
        elif gen_cmd and ref_cmd:
            related = {
                'definition': ['fun', 'function'],
                'fun': ['definition', 'function'],
                'function': ['fun', 'definition'],
                'primrec': ['fun', 'definition']
            }
            cmd_score = 0.5 if gen_cmd in related.get(ref_cmd, []) else 0.2
        else:
            cmd_score = 0.0
        
        # Isabelle-specific markers
        isabelle_markers = ['::', 'where', '⇒', '=>', 'word', 'nat', 'list', '×']
        gen_markers = {m for m in isabelle_markers if m in gen_clean}
        ref_markers = {m for m in isabelle_markers if m in ref_clean}
        
        if ref_markers:
            marker_score = len(gen_markers & ref_markers) / len(ref_markers)
        else:
            marker_score = 0.5 if gen_markers else 1.0
        
        # ANTI-PATTERN DETECTION (Crucial for isolating Isabelle from SML)
        anti_patterns = [
            r'val\s+\w+\s*=',
            r'let\s+val\b',
            r'match\s+\w+\s+with',
            r'\[\|',
            r'type\s+\w+\s*=',
            r'def\s+\w+\s*\(',           # Python
            r'function\s+\w+\s*\(.*\)\s*{',  # JavaScript/C
            r'=>\s*{',                    # Arrow functions
            r'lambda\s+',                 # Python lambda
            r'\(\s*function\b',           # JS function
            r'\[\s*\]',                   # Empty list wrong syntax
            # FIXED: Only match single colon for type annotations
            r'\b:\s*[a-zA-Z_][a-zA-Z0-9_]*\b',  # Python/ML type annotations like ": int"
        ]
        
        penalty = 0.0
        for pattern in anti_patterns:
            if re.search(pattern, gen_clean):
                penalty += 0.3

            
        #print(f"gen_markers: {gen_markers}")
        #print(f"ref_markers: {ref_markers}")
        #print(f"marker_score: {marker_score}")
        #print(f"cmd_score: {cmd_score}")
        #print(f"penalty: {penalty}")
        
        style_score = (cmd_score * 0.4) + (marker_score * 0.6)
        return max(0.0, style_score - penalty)



    def semantic_match(self, generated: str, reference: str) -> float:
        # Corresponds to Section 4: Evaluation Framework, Eq. 2: SM = (0.3·R_seq + 0.7·R_occ)·(1 − ΣP_crit)
        gen_logic = self._remove_comments_and_strings(generated)
        ref_logic = self._remove_comments_and_strings(reference)

        gen_ops = self._extract_operators_list(gen_logic)
        ref_ops = self._extract_operators_list(ref_logic)

        if not ref_ops:
            return 1.0 if not gen_ops else 0.5
        if not gen_ops:
            # Check if there's ANY meaningful content
            #if self._has_semantic_content(generated):
            #    return 0.3  # Partial credit for trying
            return 0.0

        # 1. Frequency Match (How many of each?)
        gen_counts = Counter(gen_ops)
        ref_counts = Counter(ref_ops)
        intersection = sum((gen_counts & ref_counts).values())
        occurrence_ratio = intersection / max(sum(ref_counts.values()), sum(gen_counts.values()))

        # 2. STRICT Sequence Match
        # difflib can be too lenient. For code, we want to know if the 
        # sequence is a perfect match or has the same flow.
        seq_matcher = difflib.SequenceMatcher(None, gen_ops, ref_ops)
        sequence_ratio = seq_matcher.ratio()

        # Implements the 70/30 R_occ/R_seq weighting from Eq. 2 (Section 4: Evaluation Framework).
        # The 0.7 weight on occurrence reflects commutativity of XOR and modular
        # addition in ARX ciphers: operator presence matters more than order.
        base_similarity = (sequence_ratio * 0.3) + (occurrence_ratio * 0.7)
        
        # 4. Critical Logic Penalty (The Deal Breaker)
        # We must detect if the sequence order itself contains a swap 
        # of non-commutative operations.
        #print('GENERATED OPERATIONS: ', gen_ops)
        #print('REFERENCE OPERATIONS: ' ,ref_ops)
        penalty = self._calculate_penalty(set(gen_ops), set(ref_ops))
        
        # If the order is flipped for simple operators, SequenceMatcher.ratio() 
        # should be lower than 1.0. If it's 1.0, it means it's a perfect sequence.
        
        final_score = base_similarity * (1.0 - min(1.0, penalty))
        
        return max(0.0, min(1.0, final_score))
           
    
    def value_consistency(self, generated: str, reference: str) -> float:
        # Corresponds to Section 4: Evaluation Framework, Eq. 3: VC = (ω·R_occ + (1−ω)·R_seq) − ΣP_{size,rot}
        """VC with commutative context awareness."""
        # 1. Extract ordered constants
        gen_vals = self._extract_all_constants_ordered(generated)
        ref_vals = self._extract_all_constants_ordered(reference)
        
        if not ref_vals:
            return 1.0 if not gen_vals else 0.7
        if not gen_vals:
            return 0.0
        
        # 2. Frequency/Set Score
        gen_counts = Counter(gen_vals)
        ref_counts = Counter(ref_vals)
        intersection = sum((gen_counts & ref_counts).values())
        occurrence_ratio = intersection / max(sum(ref_counts.values()), sum(gen_counts.values()))
        
        # 3. Sequence Score
        seq_score = difflib.SequenceMatcher(None, gen_vals, ref_vals).ratio()
        
        # Context-adaptive weight ω from Eq. 3 (Section 4: Evaluation Framework):
        # ω=0.8 for commutative contexts (XOR, addition); ω=0.4 for non-commutative.
        if self._is_commutative_context(generated, reference):
            base_score = (occurrence_ratio * 0.8) + (seq_score * 0.2)
        else:
            base_score = (occurrence_ratio * 0.4) + (seq_score * 0.6)
        
        # 5. Penalties
        penalty = 0.0
        if self._has_wrong_rotation(generated, reference): 
            penalty += 0.5
        if self._has_wrong_sizes(generated, reference): 
            penalty += 0.5
        
        return max(0.0, min(1.0, base_score - penalty))
    
    def _is_commutative_context(self, gen: str, ref: str) -> bool:
        # Determines the context-adaptive weight ω defined in Section 4: Evaluation Framework, Eq. 3.
        # Returns True (ω=0.8) when commutative operators dominate; False (ω=0.4) otherwise.
        """
        Check for commutative operators in Isabelle, handling both
        prefix (xor, and, or) and infix (+, *, &&, ||) operators.
        """
        # Clean the code
        gen_clean = self._remove_comments_and_strings(gen)
        ref_clean = self._remove_comments_and_strings(ref)
        
        # Convert to lowercase for case-insensitive matching
        gen_lower = gen_clean.lower()
        ref_lower = ref_clean.lower()
        
        # COMMUTATIVE OPERATORS in Isabelle:
        # 1. Prefix operators (need space after): xor, and, or, not (but not is unary!)
        # 2. Infix operators: +, *, &&, ||, and also and, or can be infix
        
        # Pattern for prefix commutative operators
        prefix_patterns = [
            r'\bxor\s+',     # xor followed by space
            r'\band\s+',     # and followed by space  
            r'\bor\s+',      # or followed by space
        ]
        
        # Pattern for infix commutative operators  
        # These appear between operands: a + b, a * b, a && b, a || b
        # Also "and" and "or" can be infix: a and b, a or b
        infix_patterns = [
            r'[a-zA-Z0-9_\)]\s*\+\s*[a-zA-Z0-9_\(]',  # a + b
            r'[a-zA-Z0-9_\)]\s*\*\s*[a-zA-Z0-9_\(]',  # a * b
            r'[a-zA-Z0-9_\)]\s*&&\s*[a-zA-Z0-9_\(]',  # a && b  
            r'[a-zA-Z0-9_\)]\s*\|\|\s*[a-zA-Z0-9_\(]', # a || b
            r'[a-zA-Z0-9_\)]\s+and\s+[a-zA-Z0-9_\(]',  # a and b (infix)
            r'[a-zA-Z0-9_\)]\s+or\s+[a-zA-Z0-9_\(]',   # a or b (infix)
        ]
        
        # Check prefix operators
        def has_prefix_ops(code):
            for pattern in prefix_patterns:
                if re.search(pattern, code):
                    return True
            return False
        
        # Check infix operators  
        def has_infix_ops(code):
            for pattern in infix_patterns:
                if re.search(pattern, code):
                    return True
            return False
        
        # Determine if commutative context
        gen_has_commutative = has_prefix_ops(gen_lower) or has_infix_ops(gen_lower)
        ref_has_commutative = has_prefix_ops(ref_lower) or has_infix_ops(ref_lower)
        
        # For commutative weighting, BOTH should have commutative ops
        # OR if XOR is in either (always commutative for cryptography)
        if not (gen_has_commutative and ref_has_commutative):
            # Special case: XOR is always commutative if present
            xor_pattern = r'\bxor\b'
            if re.search(xor_pattern, gen_lower) or re.search(xor_pattern, ref_lower):
                return True
            return False
        
        return True

        
    def _extract_all_constants_ordered(self, code: str) -> List[str]:
        """Two-pass approach for maximum reliability."""
        if not code:
            return []
        
        clean_code = self._remove_comments_and_strings(code)
        
        # PASS 1: Remove variable names with numbers
        # Replace patterns like simon_64_128, uint64, word32, etc.
        var_pattern = r'\b[a-zA-Z_]+[a-zA-Z0-9_]*\d+[a-zA-Z0-9_]*\b'
        cleaned = re.sub(var_pattern, ' VAR ', clean_code)
        
        # PASS 2: Extract all remaining numbers
        # Now we can safely extract all numbers
        pattern = r'\b(\d+|0x[0-9a-fA-F]+|0b[01]+)\b'
        
        matches = []
        for match in re.finditer(pattern, cleaned):
            matches.append((match.group(1), match.start()))
        
        # Sort by position
        matches.sort(key=lambda x: x[1])
        
        return [m[0] for m in matches]
    

    
    def _has_wrong_rotation(self, gen: str, ref: str) -> bool:
        # Implements the rotation-amount component of P_{size,rot} in Eq. 3 (Section 4: Evaluation Framework).
        # A mismatch in word_rotl/word_rotr constants triggers a 0.5 VC penalty.
        """Check if rotation amounts are different"""
        # Look for pattern: word_rotl/word_rotr followed by a number
        gen_rots = re.findall(r'word_rot[LlRr]\s+(\d+)', gen)
        ref_rots = re.findall(r'word_rot[LlRr]\s+(\d+)', ref)
        
        if not gen_rots or not ref_rots:
            return False
        
        # Compare all rotation amounts found
        return set(gen_rots) != set(ref_rots)
    
    def _has_wrong_sizes(self, gen: str, ref: str) -> bool:
        # Implements the bit-width component of P_{size,rot} in Eq. 3 (Section 4: Evaluation Framework).
        # Mismatched word/block/key sizes trigger a 0.5 VC penalty.
        """Check if word/block/key sizes are different"""
        # Look for size patterns
        size_patterns = [
            r'(\d+)\s+word',  # "32 word"  
            r'word_size\s*=\s*(\d+)',
            r'block_size\s*=\s*(\d+)',
            r'key_size\s*=\s*(\d+)',
            r'_size\s*::[^=]*=\s*(\d+)'  # "x_size :: nat = 32"
        ]
        
        gen_sizes = set()
        ref_sizes = set()
        
        for pattern in size_patterns:
            gen_sizes.update(re.findall(pattern, gen, re.IGNORECASE))
            ref_sizes.update(re.findall(pattern, ref, re.IGNORECASE))
        
        # Compare sizes
        return bool(gen_sizes and ref_sizes and gen_sizes != ref_sizes)
   

    def evaluate(self, generated: str, reference: str) -> Dict[str, float]:
        """Run all three metrics"""
        return {
            "syntax_validity": self.syntax_validity(generated, reference),
            "semantic_match": self.semantic_match(generated, reference),
            "value_consistency": self.value_consistency(generated, reference)
        }
    
    def _remove_comments(self, code: str) -> str:
        """Remove Isabelle comments"""
        # Remove block comments
        code = re.sub(r'\(\*.*?\*\)', '', code, flags=re.DOTALL)
        
        # Remove trailing comments on lines
        lines = []
        for line in code.split('\n'):
            if '(*' in line:
                line = line.split('(*')[0]
            lines.append(line.strip())
        
        return '\n'.join([l for l in lines if l])
    
    def _check_balanced_parentheses(self, code: str) -> bool:
        """Check balanced parentheses"""
        stack = []
        for char in code:
            if char == '(':
                stack.append('(')
            elif char == ')':
                if not stack:
                    return False
                stack.pop()
        return len(stack) == 0
        
    def _extract_operators_list(self, code: str) -> List[str]:
        # Extracts the ordered operator sequence used to compute R_occ and R_seq
        # in Eq. 2 (Section 4: Evaluation Framework); preserves appearance order for sequence matching.
        """Extract ALL operators in their order of appearance."""
        if not code:
            return []
        
        # 1. Aggressive Clean: Remove comments, strings, and types
        code_clean = self._remove_comments_and_strings(code)
        
        # 2. Extract using a single pass to maintain order
        # We look for alphanumeric words OR specific symbolic operators
        tokens = re.findall(r'[a-zA-Z_]\w*|[\+\-\*\/\=\≠\<\>\≤\≥\⇒\→\↔\∧\∨\¬\∀\∃\:\#\@\!\^\~]+', code_clean)
        
        ordered_ops = []
        for t in tokens:
            # Check if token is a known word-operator or a symbolic operator
            if t.lower() in self.crypto_operators or t in self.symbolic_operators:
                # Ensure it's not a "non_operator" like 'definition'
                if t.lower() not in self.non_operators:
                    ordered_ops.append(t.lower())
        
        return ordered_ops

    def _remove_comments_and_strings(self, code: str) -> str:
        """Remove comments but preserve string content (extract numbers from strings)."""
        if not code: 
            return ""
        
        # 1. Remove block comments
        code = re.sub(r'\(\*.*?\*\)', ' ', code, flags=re.DOTALL)
        
        # 2. Handle string literals - EXTRACT NUMBERS FROM THEM!
        def process_string(match):
            content = match.group(1)  # Get content without quotes
            # Return the content so numbers can be extracted from it
            return f' {content} '
        
        # Replace strings but keep their content
        code = re.sub(r'"([^"]*)"', process_string, code)
        
        # 3. Char literals
        code = re.sub(r"'(.)'", ' CHAR ', code)
        
        return code
    

    def _calculate_penalty(self, gen_ops: Set[str], ref_ops: Set[str]) -> float:
        # Implements P_crit in Eq. 2 (Section 4: Evaluation Framework): 0.3 penalty per mismatched critical
        # operator group (bitwise, arithmetic, rotations, list ops), capped at 1.0.
        """
        Refined Penalty: 0.3 per group mismatch.
        Allows for a 'three-strikes' degradation rather than a cliff.
        """
        penalty = 0.0
        
        # Critical operator groups
        critical_groups = [
            {'xor', 'and', 'or', 'not'},  # Bitwise
            {'+', '-', '*', 'div', 'mod'},  # Arithmetic
            {'word_rotl', 'word_rotr'},  # Rotations
            {'hd', 'tl'},  # List head/tail
        ]
        
        for group in critical_groups:
            gen_in_group = gen_ops.intersection(group)
            ref_in_group = ref_ops.intersection(group)
            
            if ref_in_group and gen_in_group:
                # If the sets of operators used within a functional group 
                # do not match exactly, apply a linear penalty.
                if gen_in_group != ref_in_group:
                    penalty += 0.3 
        
        return min(1, penalty)
   
    def _extract_constants(self, code: str) -> List[str]:
        """
        Extract all constants from code.
        
        Returns: List of constant strings (normalized)
        """
        constants = []
        
        # Remove comments and strings (but keep string content for S-boxes)
        clean_code = self._remove_comments(code)
        
        # Extract numeric constants
        # Decimal
        decimals = re.findall(r'\b(\d+)\b', clean_code)
        constants.extend(decimals)
        
        # Hexadecimal (0x...)
        hexs = re.findall(r'\b(0x[0-9a-fA-F]+)\b', clean_code)
        constants.extend(hexs)
        
        # Binary (0b...)
        binaries = re.findall(r'\b(0b[01]+)\b', clean_code)
        constants.extend(binaries)

        return constants

# ========== TEST WITH DEBUGGING ==========

def test_with_debugging():
    """Test with detailed debugging output"""
    print("=" * 80)
    print("FIXED METRICS WITH DEBUGGING")
    print("=" * 80)
    
    metrics = FixedCryptographicMetrics()
    
    test_cases = [
        {
            "name": "Test 1: Simple constant (perfect match)",
            "generated": 'definition test :: nat where "test = 5"',
            "reference": 'definition test :: nat where "test = 5"',
            "debug": True
        },
        {
            "name": "Test 2: Constant with different function name",
            "generated": 'definition rounds :: nat where "rounds = 42"',
            "reference": 'definition simon_rounds :: nat where "simon_rounds = 42"',
            "debug": True
        },
        {
            "name": "Test 3: ARX with xor (correct)",
            "generated": 'definition f :: "32 word ⇒ 32 word" where "f x = xor x x"',
            "reference": 'definition g :: "32 word ⇒ 32 word" where "g x = xor x x"',
            "debug": True
        },
        {
            "name": "Test 4: ARX with xor vs and (WRONG!)",
            "generated": 'definition f :: "32 word ⇒ 32 word" where "f x = and x x"',
            "reference": 'definition g :: "32 word ⇒ 32 word" where "g x = xor x x"',
            "debug": True
        },
        {
            "name": "Test 5: ARX with rotation (correct)",
            "generated": 'definition f :: "32 word ⇒ 32 word" where "f x = word_rotl 3 x"',
            "reference": 'definition g :: "32 word ⇒ 32 word" where "g x = word_rotl 3 x"',
            "debug": True
        },
        {
            "name": "Test 6: ARX with different rotation amount (WRONG!)",
            "generated": 'definition f :: "32 word ⇒ 32 word" where "f x = word_rotl 7 x"',
            "reference": 'definition g :: "32 word ⇒ 32 word" where "g x = word_rotl 3 x"',
            "debug": True
        },
        {
            "name": "Test 7: SPN S-box values",
            "generated": 'definition sbox :: "4 word ⇒ 4 word" where "sbox x = case x of 0x0 ⇒ 0xC | 0x1 ⇒ 0x5"',
            "reference": 'definition sbox :: "4 word ⇒ 4 word" where "sbox x = case x of 0x0 ⇒ 0xC | 0x1 ⇒ 0x5"',
            "debug": True
        },
        {
            "name": "Test 8: SPN with wrong S-box value (WRONG!)",
            "generated": 'definition sbox :: "4 word ⇒ 4 word" where "sbox x = case x of 0x0 ⇒ 0xD | 0x1 ⇒ 0x5"',
            "reference": 'definition sbox :: "4 word ⇒ 4 word" where "sbox x = case x of 0x0 ⇒ 0xC | 0x1 ⇒ 0x5"',
            "debug": True
        }
    ]
    
    for test in test_cases:
        print(f"\n{test['name']}")
        print("-" * 40)
        
        # Debug: Show what operators are extracted
        if test.get("debug"):
            gen_ops = metrics._extract_operators(test["generated"])
            ref_ops = metrics._extract_operators(test["reference"])
            print(f"Generated operators: {gen_ops}")
            print(f"Reference operators: {ref_ops}")
            
            gen_const = metrics._extract_constants(test["generated"])
            ref_const = metrics._extract_constants(test["reference"])
            print(f"Generated constants: {gen_const}")
            print(f"Reference constants: {ref_const}")
        
        # Run evaluation
        results = metrics.evaluate(test["generated"], test["reference"])
        
        print(f"SV (Syntax): {results['syntax_validity']:.3f}")
        print(f"SM (Semantic): {results['semantic_match']:.3f}")
        print(f"VC (Values): {results['value_consistency']:.3f}")
        
        # Show expected pattern
        if "WRONG" in test["name"]:
            print("Expected: Low SM/VC scores (should be < 0.5)")
        else:
            print("Expected: High scores (should be > 0.8)")

# ========== TEST WITH EXAMPLES ==========

def test_simple_metrics():
    """
    Test the simple metrics with realistic examples.
    """
    print("=" * 80)
    print("SIMPLE METRICS TEST")
    print("=" * 80)
    
    metrics = SimpleCryptographicMetrics()
    
    # Test cases from JSON examples
    test_cases = [
        
        {
            "name": "Perfect constant match (Index 20)",
            "generated": 'definition speck_96_96_rounds :: nat where "speck_96_96_rounds = 28"',
            "reference": 'definition speck_96_96_rounds :: nat where "speck_96_96_rounds = 28"',
            "expected": {"SV": 1.0, "OM": 1.0, "SM": 1.0}
        },
        {
            "name": "Same constant, different var name",
            "generated": 'definition rounds :: nat where "rounds = 42"',
            "reference": 'definition simon_rounds :: nat where "simon_rounds = 42"',
            "expected": {"SV": 1.0, "OM": 1.0, "SM": 1.0}  # Should match despite different names
        },
        {
            "name": "ARX operation match",
            "generated": 'definition speck_round :: "32 word ⇒ 32 word ⇒ 32 word" where "speck_round x k = xor (word_rotl 3 x) k"',
            "reference": 'definition speck_round :: "32 word ⇒ 32 word ⇒ 32 word" where "speck_round x k = xor (word_rotl 3 x) k"',
            "expected": {"SV": 1.0, "OM": 1.0, "SM": 1.0}
        },
        {
            "name": "ARX with different variable names",
            "generated": 'definition encrypt :: "32 word ⇒ 32 word ⇒ 32 word" where "encrypt plaintext key = xor (word_rotl 3 plaintext) key"',
            "reference": 'definition encrypt :: "32 word ⇒ 32 word ⇒ 32 word" where "encrypt x k = xor (word_rotl 3 x) k"',
            "expected": {"SV": 1.0, "OM": 1.0, "SM": 1.0}  # Structure matches despite different var names
        },
        {
            "name": "Wrong operator",
            "generated": 'definition wrong :: "32 word ⇒ 32 word ⇒ 32 word" where "wrong x k = and (word_rotl 3 x) k"',
            "reference": 'definition correct :: "32 word ⇒ 32 word ⇒ 32 word" where "correct x k = xor (word_rotl 3 x) k"',
            "expected": {"SV": 1.0, "OM": 0.0, "SM": 0.8}  # Syntax OK, operators wrong, structure similar
        },
        {
            "name": "Missing where keyword",
            "generated": 'definition test :: nat "test = 5"',
            "reference": 'definition test :: nat where "test = 5"',
            "expected": {"SV": 0.5, "OM": 1.0, "SM": 0.8}  # Partial syntax, operators match
        },
        {
            "name": "Completely wrong",
            "generated": 'this is not isabelle code',
            "reference": 'definition test :: nat where "test = 5"',
            "expected": {"SV": 0.0, "OM": 0.0, "SM": 0.0}
        },
        {
            "name": "Feistel with list operations",
            "generated": 'definition key_schedule :: "32 word list ⇒ 32 word list" where "key_schedule keys = hd keys # tl keys"',
            "reference": 'definition key_schedule :: "32 word list ⇒ 32 word list" where "key_schedule keys = hd keys # tl keys"',
            "expected": {"SV": 1.0, "OM": 1.0, "SM": 1.0}
        }
    ]
    
    print("\nRunning tests...\n")
    
    for i, test in enumerate(test_cases, 1):
        print(f"Test {i}: {test['name']}")
        print("-" * 40)
        
        results = metrics.evaluate(test["generated"], test["reference"])
        
        # Print results
        print(f"SV (Syntax Validity): {results['syntax_validity']:.3f}")
        print(f"OM (Operator Match): {results['operator_match']:.3f}")
        print(f"SM (Structural Match): {results['structural_match']:.3f}")
        
        # Check against expected
        expected = test["expected"]
        passed = True
        
        for metric, exp_val in expected.items():
            actual = results[{
                "SV": "syntax_validity",
                "OM": "operator_match",
                "SM": "structural_match"
            }[metric]]
            
            if abs(actual - exp_val) > 0.1:
                status = "✗"
                passed = False
            else:
                status = "✓"
            
            print(f"  {metric}: Expected {exp_val:.1f}, Got {actual:.3f} {status}")
        
        if not passed:
            print("  Details:")
            print(f"    Generated: {test['generated'][:50]}...")
            print(f"    Reference: {test['reference'][:50]}...")
        
        print()
    
    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

# ========== COMPREHENSIVE TEST ==========

def comprehensive_test():
    """Comprehensive test with all edge cases"""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE TEST")
    print("=" * 80)
    
    metrics = FixedCryptographicMetrics()
    
    tests = [
        # Group 1: Simple constants (should all be high scores)
        ("Constant exact match", 
         'definition a :: nat where "a = 28"',
         'definition b :: nat where "b = 28"',
         {"SV": 1.0, "SM": 1.0, "VC": 1.0}),
        
        ("Constant different value",
         'definition a :: nat where "a = 42"',
         'definition b :: nat where "b = 28"',
         {"SV": 1.0, "SM": 1.0, "VC": 0.0}),
        
        # Group 2: ARX operations
        ("ARX correct operators",
         'definition f :: "word ⇒ word" where "f x = xor (word_rotl 3 x) x"',
         'definition g :: "word ⇒ word" where "g x = xor (word_rotl 3 x) x"',
         {"SV": 1.0, "SM": 1.0, "VC": 1.0}),
        
        ("ARX wrong operator (and instead of xor)",
         'definition f :: "word ⇒ word" where "f x = and (word_rotl 3 x) x"',
         'definition g :: "word ⇒ word" where "g x = xor (word_rotl 3 x) x"',
         {"SV": 1.0, "SM": 0.5, "VC": 1.0}),  # SM penalized for wrong operator
        
        ("ARX wrong rotation amount",
         'definition f :: "word ⇒ word" where "f x = xor (word_rotl 7 x) x"',
         'definition g :: "word ⇒ word" where "g x = xor (word_rotl 3 x) x"',
         {"SV": 1.0, "SM": 1.0, "VC": 0.0}),  # VC 0 because constant wrong
        
        # Group 3: Mixed operators
        ("ARX with addition",
         'definition f :: "word ⇒ word ⇒ word" where "f x y = xor (x + y) x"',
         'definition g :: "word ⇒ word ⇒ word" where "g x y = xor (x + y) x"',
         {"SV": 1.0, "SM": 1.0, "VC": 1.0}),
        
        ("ARX with wrong arithmetic",
         'definition f :: "word ⇒ word ⇒ word" where "f x y = xor (x - y) x"',
         'definition g :: "word ⇒ word ⇒ word" where "g x y = xor (x + y) x"',
         {"SV": 1.0, "SM": 0.5, "VC": 1.0}),  # SM penalized for - vs +
        
        # Group 4: Feistel operations
        ("Feistel list operations",
         'definition f :: "word list ⇒ word" where "f xs = hd xs"',
         'definition g :: "word list ⇒ word" where "g xs = hd xs"',
         {"SV": 1.0, "SM": 1.0, "VC": 1.0}),
        
        ("Feistel wrong list op",
         'definition f :: "word list ⇒ word" where "f xs = tl xs"',
         'definition g :: "word list ⇒ word" where "g xs = hd xs"',
         {"SV": 1.0, "SM": 0.5, "VC": 1.0}),
        
        # Group 5: Syntax errors
        ("Missing where keyword",
         'definition f :: nat "f = 5"',
         'definition g :: nat where "g = 5"',
         {"SV": 0.5, "SM": 1.0, "VC": 1.0}),
        
        ("Not Isabelle code",
         'this is not code',
         'definition g :: nat where "g = 5"',
         {"SV": 0.0, "SM": 0.0, "VC": 0.0}),
        
        # Index 21: Perfect constant match
        ("Constant perfect match",
         'definition speck_96_96_key_size :: nat where "speck_96_96_key_size = 96"',
         'definition speck_96_96_key_size :: nat where "speck_96_96_key_size = 96"',
         {"SM": 1.0, "VC": 1.0}),
        
        # Index 22: Similar but different variable names
        ("Same structure, different var names",
         'definition simeck_64_128_encrypt_block :: "32 word × 32 word ⇒ 32 word list ⇒ 32 word × 32 word" where "simeck_64_128_encrypt_block state keys = simeck_64_128_encrypt_iterate state keys"',
         'definition simeck_64_128_encrypt_block :: "32 word × 32 word ⇒ 32 word list ⇒ 32 word × 32 word" where "simeck_64_128_encrypt_block plaintext keys = simeck_64_128_encrypt_iterate plaintext keys"',
         {"SM": 1.0, "VC": 1.0}),  # Should be high
        
        # Index 24: S-box with different values
        ("S-box with different values",
         'definition gift_128_128_sbox_table :: "nat list" where "gift_128_128_sbox_table = [1, 10, 4, 12, 6, 13, 3, 9, 2, 11, 4, 12, 6, 0, 8, 13]"',
         'definition gift_128_128_sbox_table :: "nat list" where "gift_128_128_sbox_table = [1, 10, 4, 12, 6, 15, 3, 9, 2, 13, 11, 7, 5, 0, 8, 14]"',
         {"SM": 1.0, "VC": 0.5}),  # SM high (same structure), VC medium (some values match)
        
        # Index 26: Complex with operations
        ("Complex with ucast, drop_bit, etc.",
         'definition simon_64_96_decrypt :: "64 word ⇒ 32 word list ⇒ 64 word" where "simon_64_96_decrypt ciphertext keys = (let left = ucast (drop_bit 32 ciphertext); right = ucast ciphertext; (p_l, p_r) = simon_64_96_decrypt_block (left, right) keys in or (push_bit 32 (ucast p_l)) (ucast p_r))"',
         'definition simon_64_96_decrypt :: "64 word ⇒ 32 word list ⇒ 64 word" where "simon_64_96_decrypt ciphertext keys = (let left = ucast (drop_bit 32 ciphertext); right = ucast ciphertext; (p_l, p_r) = simon_64_96_decrypt_block (left, right) keys in or (push_bit 32 (ucast p_l)) (ucast p_r))"',
         {"SM": 1.0, "VC": 1.0}),
        
        # Index 30: Different word sizes (WRONG!)
        ("Different word sizes (should be wrong)",
         'definition speck_96_144_generate_key_schedule :: "24 word list ⇒ 24 word list" where "speck_96_144_generate_key_schedule initial_key_words = (let k0 = [initial_key_words ! 0]; l0 = [initial_key_words ! 1, initial_key_words ! 2] in speck_96_144_gen_key_schedule_rec l0 k0 0)"',
         'definition speck_96_144_generate_key_schedule :: "48 word list ⇒ 48 word list" where "speck_96_144_generate_key_schedule initial_key_words = (let k0 = [initial_key_words ! 0]; l0 = [initial_key_words ! 1, initial_key_words ! 2] in speck_96_144_gen_key_schedule_rec l0 k0 0)"',
         {"SM": 1.0, "VC": 0.0}),  # SM high (same structure), VC 0 (different constants)
        
        # Index 41: Complex with let and operations
        ("Complex split block",
         'definition speck_64_96_split_block :: "64 word ⇒ (32 word × 32 word)" where "speck_64_96_split_block block = (let left = ucast (drop_bit 32 block); right = ucast block in (left, right))"',
         'definition speck_64_96_split_block :: "64 word ⇒ (32 word × 32 word)" where "speck_64_96_split_block block = (let left = ucast (drop_bit 32 block); right = ucast block in (left, right))"',
         {"SM": 1.0, "VC": 1.0}),
        
    ]
    
    passed = 0
    total = len(tests)
    
    for name, gen, ref, expected in tests:
        print(f"\n{name}")
        print(f"Generated: {gen}")
        print(f"Reference: {ref}")
        
        results = metrics.evaluate(gen, ref)
        print(results)
        
        # Check each metric
        test_passed = True
        for metric, exp_val in expected.items():
            actual = results[{
                "SV": "syntax_validity",
                "SM": "semantic_match",
                "VC": "value_consistency"
            }[metric]]
            
            # Allow some tolerance
            if abs(actual - exp_val) > 0.2:
                print(f"  {metric}: Expected {exp_val:.1f}, Got {actual:.3f} ✗")
                test_passed = False
            else:
                print(f"  {metric}: Expected {exp_val:.1f}, Got {actual:.3f} ✓")
        
        if test_passed:
            passed += 1
    
    print(f"\nPassed: {passed}/{total} tests")
    return passed == total


def main():# Group 4: Feistel operations
    """Main function"""
    
    # Run debugging test
    test_with_debugging()
    
    # Run comprehensive test
    success = comprehensive_test()
    
    if success:
        print("\n" + "=" * 80)
        print("ALL TESTS PASSED! ✓")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("SOME TESTS FAILED - check implementation")
        print("=" * 80)

if __name__ == "__main__":
    main()



    
    
