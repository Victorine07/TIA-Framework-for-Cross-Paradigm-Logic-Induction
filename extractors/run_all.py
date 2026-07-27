from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .sparx_extractor import SparxExtractor
from .speck_extractor import SpeckExtractor
from .lea_extractor import LeaExtractor
from .hight_extractor import HightExtractor
from .cham_extractor import ChamExtractor
from .simon_extractor import SimonExtractor
from .present_extractor import PresentExtractor
from .gift_extractor import GiftExtractor
from .ascon_extractor import AsconExtractor
from .simeck_extractor import SimeckExtractor
from .rectangle_extractor import RectangleExtractor
from .skinny_extractor import SkinnyExtractor
from .gift_cofb_extractor import GiftCofbExtractor
from .xtea_extractor import XteaExtractor



# ======================================================================
# Normalization helpers
# ======================================================================

def normalize_lea_record(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    LeaExtractor returns JsonlRecord-like dicts (or JsonlRecord objects).
    Convert to our unified training format.
    """
    if hasattr(raw, "to_dict"):
        raw = raw.to_dict()

    instr_obj = raw.get("instruction", {}) or {}
    instruction = instr_obj.get("prompt", "").strip()

    src_pair = raw.get("source_pair", {}) or {}
    py_src = src_pair.get("python", {}) or {}
    isa_src = src_pair.get("isabelle", {}) or {}

    input_code = (py_src.get("code") or "").strip()
    output_code = (isa_src.get("code") or "").strip()

    # Fold all non-code metadata plus any existing metadata dict
    base_md: Dict[str, Any] = {}
    for k, v in raw.items():
        if k in {"instruction", "source_pair"}:
            continue
        if k == "metadata":
            continue
        base_md[k] = v

    existing_md = raw.get("metadata") or {}
    if isinstance(existing_md, dict):
        base_md.update(existing_md)

    return {
        "instruction": instruction,
        "input": input_code,
        "output": output_code,
        "metadata": base_md,
    }


def normalize_unified_record(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    SPARX/SPECK already return unified-style dicts, but normalize to ensure
    keys are present and types are consistent.
    """
    instr = raw.get("instruction", "")
    if isinstance(instr, dict):
        instr = instr.get("prompt", "") or ""
    inp = raw.get("input", "") or ""
    out = raw.get("output", "") or ""
    md = raw.get("metadata") or {}
    if not isinstance(md, dict):
        md = {}

    return {
        "instruction": str(instr).strip(),
        "input": str(inp).strip(),
        "output": str(out).strip(),
        "metadata": md,
    }


def add_common_metadata(ex: Dict[str, Any], cipher_variant: str, split: str) -> Dict[str, Any]:
    md = ex.get("metadata")
    if not isinstance(md, dict):
        md = {}
        ex["metadata"] = md
    md["cipher_variant"] = cipher_variant
    md["split"] = split
    return ex


def tier_counts(examples: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts = {"T1": 0, "T2": 0, "T3": 0, "T4": 0, "UNKNOWN": 0}
    for ex in examples:
        md = ex.get("metadata", {}) or {}
        t = md.get("tier") or md.get("tier_hint")
        if t in counts:
            counts[t] += 1
        else:
            counts["UNKNOWN"] += 1
    return counts


def write_jsonl(examples: Sequence[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


# ======================================================================
# Variant extraction functions
# ======================================================================

def extract_sparx_variant(root_dir: Path, block_size: int, key_size: int, split: str) -> List[Dict[str, Any]]:
    """
    SparxExtractor.__init__ signature from your code:

    SparxExtractor(
        root_dir: str,
        cipher: str,
        family: str,
        subfamily: str,
        block_size: int,
        key_size: int,
        variant_config: dict,
        dataset_split: str = "train",
        ...
    )
    """
    variant_cfg = SparxExtractor(
        root_dir=root_dir,
        cipher="sparx",
        family="ARX",
        subfamily="SPARX",
        block_size=block_size,
        key_size=key_size,
        variant_config={},  # will be normalized by _get_sparx_params anyway
        dataset_split=split,
    )

    # Important: set source files so python_source / thy_source get loaded.
    py_path = root_dir / "python ciphers" / f"sparx_{block_size}_{key_size}.py"
    thy_path = root_dir / "thy ciphers" / f"Sparx_{block_size}_{key_size}.thy"
    variant_cfg.set_source_files(py_path, thy_path)

    raw_examples = variant_cfg.extract_components()
    out: List[Dict[str, Any]] = []
    cipher_variant = f"sparx_{block_size}_{key_size}"
    for r in raw_examples:
        ex = normalize_unified_record(r)
        ex = add_common_metadata(ex, cipher_variant=cipher_variant, split=split)
        out.append(ex)
    return out


def extract_speck_variant(root_dir: Path, block_size: int, key_size: int, split: str) -> List[Dict[str, Any]]:
    """
    SpeckExtractor in your code extends BaseCipherExtractor too;
    its __init__ signature is similar: (root_dir, cipher, family, subfamily, block_size, key_size, variant_config, dataset_split=...).
    """
    extractor = SpeckExtractor(
        root_dir=root_dir,
        cipher="speck",
        family="ARX",
        subfamily="SPECK",
        block_size=block_size,
        key_size=key_size,
        variant_config={},  # params inferred internally
        dataset_split=split,
    )

    py_path = root_dir / "python ciphers" / f"speck_{block_size}_{key_size}.py"
    thy_path = root_dir / "thy ciphers" / f"Speck_{block_size}_{key_size}.thy"
    extractor.set_source_files(py_path, thy_path)

    raw_examples = extractor.extract_components()
    out: List[Dict[str, Any]] = []
    cipher_variant = f"speck_{block_size}_{key_size}"
    for r in raw_examples:
        ex = normalize_unified_record(r)
        ex = add_common_metadata(ex, cipher_variant=cipher_variant, split=split)
        out.append(ex)
    return out


def extract_lea_variant(root_dir: Path, block_size: int, key_size: int, split: str) -> List[Dict[str, Any]]:
    extractor = LeaExtractor(
        root_dir=root_dir,
        cipher="lea",
        family="ARX",
        subfamily="LEA",
        block_size=block_size,
        key_size=key_size,
        variant_config={},
        dataset_split=split,
    )

    py_path = root_dir / "python ciphers" / f"lea_{block_size}_{key_size}.py"
    thy_path = root_dir / "thy ciphers" / f"Lea_{block_size}_{key_size}.thy"
    extractor.set_source_files(py_path, thy_path)

    raw_records = extractor.extract_components()
    out: List[Dict[str, Any]] = []
    cipher_variant = f"lea_{block_size}_{key_size}"
    for r in raw_records:
        # This line is missing in your current behavior, based on output
        ex = normalize_lea_record(r)
        ex = add_common_metadata(ex, cipher_variant=cipher_variant, split=split)
        out.append(ex)
    return out

def extract_hight_variant(root_dir: Path, block_size: int, key_size: int, split: str) -> List[Dict[str, Any]]:
    extractor = HightExtractor(
        root_dir=root_dir,
        cipher="hight",
        family="Feistel",
        subfamily="HIGHT",
        block_size=block_size,
        key_size=key_size,
        variant_config={
            "block_size": block_size,
            "key_size": key_size,
            "word_size": 8,
            "key_words": 16,
            "rounds": 32,
            "branches": 8,
            "words_per_block": 8,
            "steps": 1,
            "rounds_per_step": 32,
            "total_rounds": 32,
            "total_stages": 34,
        },
        dataset_split=split,
    )

    py_path = root_dir / "python ciphers" / f"hight_{block_size}_{key_size}.py"
    thy_path = root_dir / "thy ciphers" / f"Hight_{block_size}_{key_size}.thy"
    extractor.set_source_files(py_path, thy_path)

    raw_records = extractor.extract_components()

    out: List[Dict[str, Any]] = []
    cipher_variant = f"hight_{block_size}_{key_size}"
    for r in raw_records:
        ex = normalize_unified_record(r)
        ex = add_common_metadata(ex, cipher_variant=cipher_variant, split=split)
        out.append(ex)
    return out


def extract_cham_variant(root_dir: Path, block_size: int, key_size: int, split: str) -> List[Dict[str, Any]]:
    extractor = ChamExtractor(
        root_dir=root_dir,
        cipher="cham",
        family="ARX",
        subfamily="CHAM",
        block_size=block_size,
        key_size=key_size,
        variant_config={},  # params inferred internally
        dataset_split=split,
    )

    py_path = root_dir / "python ciphers" / f"cham_{block_size}_{key_size}.py"
    thy_path = root_dir / "thy ciphers" / f"Cham_{block_size}_{key_size}.thy"
    extractor.set_source_files(py_path, thy_path)

    raw_records = extractor.extract_components()

    out: List[Dict[str, Any]] = []
    cipher_variant = f"cham_{block_size}_{key_size}"
    for r in raw_records:
        ex = normalize_unified_record(r)
        ex = add_common_metadata(ex, cipher_variant=cipher_variant, split=split)
        out.append(ex)
    return out


def extract_simon_variant(root_dir: Path, block_size: int, key_size: int, split: str) -> List[Dict[str, Any]]:
    extractor = SimonExtractor(
        root_dir=root_dir,
        cipher="simon",
        family="Feistel",
        subfamily="SIMON",
        block_size=block_size,
        key_size=key_size,
        variant_config={},  # params inferred internally
        dataset_split=split,
    )

    py_path = root_dir / "python ciphers" / f"simon_{block_size}_{key_size}.py"
    thy_path = root_dir / "thy ciphers" / f"Simon_{block_size}_{key_size}.thy"
    extractor.set_source_files(py_path, thy_path)

    raw_records = extractor.extract_components()

    out: List[Dict[str, Any]] = []
    cipher_variant = f"simon_{block_size}_{key_size}"
    for r in raw_records:
        ex = normalize_unified_record(r)
        ex = add_common_metadata(ex, cipher_variant=cipher_variant, split=split)
        out.append(ex)
    return out


def extract_present_variant(root_dir: Path, block_size: int, key_size: int, split: str) -> List[Dict[str, Any]]:
    extractor = PresentExtractor(
        root_dir=root_dir,
        cipher="present",
        family="SPN",
        subfamily="PRESENT",
        block_size=block_size,
        key_size=key_size,
        variant_config={},
        dataset_split=split,
    )

    py_path = root_dir / "python ciphers" / f"present_{block_size}_{key_size}.py"
    thy_path = root_dir / "thy ciphers" / f"Present_{block_size}_{key_size}.thy"
    extractor.set_source_files(py_path, thy_path)

    raw_records = extractor.extract_components()

    out: List[Dict[str, Any]] = []
    cipher_variant = f"present_{block_size}_{key_size}"
    for r in raw_records:
        ex = normalize_unified_record(r)
        ex = add_common_metadata(ex, cipher_variant=cipher_variant, split=split)
        out.append(ex)
    return out


def extract_gift_variant(root_dir: Path, block_size: int, key_size: int, split: str) -> List[Dict[str, Any]]:
    extractor = GiftExtractor(
        root_dir=root_dir,
        cipher="gift",
        family="SPN",
        subfamily="GIFT",
        block_size=block_size,
        key_size=key_size,
        variant_config={},
        dataset_split=split,
    )

    py_path = root_dir / "python ciphers" / f"gift_{block_size}_{key_size}.py"
    thy_path = root_dir / "thy ciphers" / f"Gift_{block_size}_{key_size}.thy"
    extractor.set_source_files(py_path, thy_path)

    raw_records = extractor.extract_components()

    out: List[Dict[str, Any]] = []
    cipher_variant = f"gift_{block_size}_{key_size}"
    for r in raw_records:
        ex = normalize_unified_record(r)
        ex = add_common_metadata(ex, cipher_variant=cipher_variant, split=split)
        out.append(ex)
    return out


def extract_ascon_variant(root_dir: Path, block_size: int, key_size: int, split: str) -> List[Dict[str, Any]]:
    extractor = AsconExtractor(
        root_dir=root_dir,
        cipher="ascon",
        family="Permutation",
        subfamily="ASCON",
        block_size=block_size,
        key_size=key_size,
        variant_config={},
        dataset_split=split,
    )

    py_path = root_dir / "python ciphers" / f"ascon_{block_size}_{key_size}.py"
    thy_path = root_dir / "thy ciphers" / f"Ascon_{block_size}_{key_size}.thy"
    extractor.set_source_files(py_path, thy_path)

    raw_records = extractor.extract_components()

    out: List[Dict[str, Any]] = []
    cipher_variant = f"ascon_{block_size}_{key_size}"
    for r in raw_records:
        ex = normalize_unified_record(r)
        ex = add_common_metadata(ex, cipher_variant=cipher_variant, split=split)
        out.append(ex)
    return out


def extract_simeck_variant(root_dir: Path, block_size: int, key_size: int, split: str) -> List[Dict[str, Any]]:
    extractor = SimeckExtractor(
        root_dir=root_dir,
        cipher="simeck",
        family="Feistel",
        subfamily="SIMECK",
        block_size=block_size,
        key_size=key_size,
        variant_config={},
        dataset_split=split,
    )

    py_path = root_dir / "python ciphers" / f"simeck_{block_size}_{key_size}.py"
    thy_path = root_dir / "thy ciphers" / f"Simeck_{block_size}_{key_size}.thy"
    extractor.set_source_files(py_path, thy_path)

    raw_records = extractor.extract_components()

    out: List[Dict[str, Any]] = []
    cipher_variant = f"simeck_{block_size}_{key_size}"
    for r in raw_records:
        ex = normalize_unified_record(r)
        ex = add_common_metadata(ex, cipher_variant=cipher_variant, split=split)
        out.append(ex)
    return out


def extract_rectangle_variant(root_dir: Path, block_size: int, key_size: int, split: str) -> List[Dict[str, Any]]:
    extractor = RectangleExtractor(
        root_dir=root_dir,
        cipher="rectangle",
        family="SPN",
        subfamily="RECTANGLE",
        block_size=block_size,
        key_size=key_size,
        variant_config={},
        dataset_split=split,
    )

    py_path = root_dir / "python ciphers" / f"rectangle_{block_size}_{key_size}.py"
    thy_path = root_dir / "thy ciphers" / f"Rectangle_{block_size}_{key_size}.thy"
    extractor.set_source_files(py_path, thy_path)

    raw_records = extractor.extract_components()

    out: List[Dict[str, Any]] = []
    cipher_variant = f"rectangle_{block_size}_{key_size}"
    for r in raw_records:
        ex = normalize_unified_record(r)
        ex = add_common_metadata(ex, cipher_variant=cipher_variant, split=split)
        out.append(ex)
    return out


def extract_skinny_variant(root_dir: Path, block_size: int, key_size: int, split: str) -> List[Dict[str, Any]]:
    extractor = SkinnyExtractor(
        root_dir=root_dir,
        cipher="skinny",
        family="SPN",
        subfamily="SKINNY",
        block_size=block_size,
        key_size=key_size,
        variant_config={},
        dataset_split=split,
    )

    py_path = root_dir / "python ciphers" / f"skinny_{block_size}_{key_size}.py"
    thy_path = root_dir / "thy ciphers" / f"Skinny_{block_size}_{key_size}.thy"
    extractor.set_source_files(py_path, thy_path)

    raw_records = extractor.extract_components()

    out: List[Dict[str, Any]] = []
    cipher_variant = f"skinny_{block_size}_{key_size}"
    for r in raw_records:
        ex = normalize_unified_record(r)
        ex = add_common_metadata(ex, cipher_variant=cipher_variant, split=split)
        out.append(ex)
    return out


def extract_gift_cofb_variant(root_dir: Path, block_size: int, key_size: int, split: str) -> List[Dict[str, Any]]:
    extractor = GiftCofbExtractor(
        root_dir=root_dir,
        cipher="gift_cofb",
        family="Permutation/AEAD",
        subfamily="GIFT-COFB",
        block_size=block_size,
        key_size=key_size,
        variant_config={},
        dataset_split=split,
    )

    py_path = root_dir / "python ciphers" / f"gift_cofb_{block_size}_{key_size}.py"
    thy_path = root_dir / "thy ciphers" / f"Gift_Cofb_{block_size}_{key_size}.thy"
    extractor.set_source_files(py_path, thy_path)

    raw_records = extractor.extract_components()

    out: List[Dict[str, Any]] = []
    cipher_variant = f"gift_cofb_{block_size}_{key_size}"
    for r in raw_records:
        ex = normalize_unified_record(r)
        ex = add_common_metadata(ex, cipher_variant=cipher_variant, split=split)
        out.append(ex)
    return out


def extract_xtea_variant(root_dir: Path, block_size: int, key_size: int, split: str) -> List[Dict[str, Any]]:
    extractor = XteaExtractor(
        root_dir=root_dir,
        cipher="xtea",
        family="Feistel",
        subfamily="XTEA",
        block_size=block_size,
        key_size=key_size,
        variant_config={},
        dataset_split=split,
    )

    py_path = root_dir / "python ciphers" / f"xtea_{block_size}_{key_size}.py"
    thy_path = root_dir / "thy ciphers" / f"Xtea_{block_size}_{key_size}.thy"
    extractor.set_source_files(py_path, thy_path)

    raw_records = extractor.extract_components()

    out: List[Dict[str, Any]] = []
    cipher_variant = f"xtea_{block_size}_{key_size}"
    for r in raw_records:
        ex = normalize_unified_record(r)
        ex = add_common_metadata(ex, cipher_variant=cipher_variant, split=split)
        out.append(ex)
    return out


# ======================================================================
# Run all ciphers
# ======================================================================

def extract_all_ciphers(
    ciphers: Sequence[str],
    root_dir: Path,
    output_dir: Path,
    split: str = "train",
    skip_errors: bool = False,
) -> List[Dict[str, Any]]:
    root_dir = Path(root_dir).resolve()
    output_dir = Path(output_dir).resolve()
    all_examples: List[Dict[str, Any]] = []

    print("============================================================")
    print("Unified Cipher Extractor")
    print("============================================================")
    print(f"Root directory: {root_dir}")
    print(f"Ciphers: {', '.join(ciphers)}")
    print(f"Split: {split}")
    print(f"Output directory: {output_dir}")

    want_all = "all" in ciphers
    do_sparx = want_all or "sparx" in ciphers
    do_speck = want_all or "speck" in ciphers
    do_lea = want_all or "lea" in ciphers
    do_hight = want_all or "hight" in ciphers
    do_cham = want_all or "cham" in ciphers
    do_simon = want_all or "simon" in ciphers
    do_present = want_all or "present" in ciphers
    do_gift = want_all or "gift" in ciphers
    do_ascon = want_all or "ascon" in ciphers
    do_simeck = want_all or "simeck" in ciphers
    do_rectangle = want_all or "rectangle" in ciphers
    do_skinny = want_all or "skinny" in ciphers
    do_gift_cofb = want_all or "gift_cofb" in ciphers
    do_xtea = want_all or "xtea" in ciphers


    # SPARX
    if do_sparx:
        print("\n============================================================")
        print("Extracting SPARX")
        print("============================================================")
        sparx_variants = [(64, 128), (128, 128), (128, 256)]
        for b, k in sparx_variants:
            try:
                print(f"  Processing {b}/{k}...")
                examples = extract_sparx_variant(root_dir, b, k, split)
                path = output_dir / f"sparx_{b}_{k}_{split}.jsonl"
                write_jsonl(examples, path)
                counts = tier_counts(examples)
                print(
                    f"    {len(examples)} examples -> {path} | "
                    f"T1={counts['T1']} T2={counts['T2']} "
                    f"T3={counts['T3']} T4={counts['T4']} UNKNOWN={counts['UNKNOWN']}"
                )
                all_examples.extend(examples)
            except Exception as e:
                print(f"    ERROR extracting SPARX-{b}/{k}: {e}")
                if not skip_errors:
                    raise
    
    # SPECK
    if do_speck:
        print("\n============================================================")
        print("Extracting SPECK")
        print("============================================================")
        speck_variants = [
            (32, 64), (48, 72), (48, 96),
            (64, 96), (64, 128),
            (96, 96), (96, 144),
            (128, 128), (128, 192), (128, 256)
        ]
        for b, k in speck_variants:
            try:
                print(f"  Processing {b}/{k}...")
                examples = extract_speck_variant(root_dir, b, k, split)
                path = output_dir / f"speck_{b}_{k}_{split}.jsonl"
                write_jsonl(examples, path)
                counts = tier_counts(examples)
                print(
                    f"    {len(examples)} examples -> {path} | "
                    f"T1={counts['T1']} T2={counts['T2']} "
                    f"T3={counts['T3']} T4={counts['T4']} UNKNOWN={counts['UNKNOWN']}"
                )
                all_examples.extend(examples)
            except Exception as e:
                print(f"    ERROR extracting SPECK-{b}/{k}: {e}")
                if not skip_errors:
                    raise
                    
    # # LEA
    if do_lea:
        print("\n============================================================")
        print("Extracting LEA")
        print("============================================================")
        lea_variants = [(128, 128), (128, 192), (128, 256)]
        for b, k in lea_variants:
            try:
                print(f"  Processing {b}/{k}...")
                examples = extract_lea_variant(root_dir, b, k, split)
                path = output_dir / f"lea_{b}_{k}_{split}.jsonl"
                write_jsonl(examples, path)
                counts = tier_counts(examples)
                print(
                    f"    {len(examples)} examples -> {path} | "
                    f"T1={counts['T1']} T2={counts['T2']} "
                    f"T3={counts['T3']} T4={counts['T4']} UNKNOWN={counts['UNKNOWN']}"
                )
                all_examples.extend(examples)
            except Exception as e:
                print(f"    ERROR extracting LEA-{b}/{k}: {e}")
                if not skip_errors:
                    raise

        # HIGHT
    if do_hight:
        print("\n============================================================")
        print("Extracting HIGHT")
        print("============================================================")
        hight_variants = [(64, 128)]
        for b, k in hight_variants:
            try:
                print(f"  Processing {b}/{k}...")
                examples = extract_hight_variant(root_dir, b, k, split)
                path = output_dir / f"hight_{b}_{k}_{split}.jsonl"
                write_jsonl(examples, path)
                counts = tier_counts(examples)
                print(
                    f"    {len(examples)} examples -> {path} | "
                    f"T1={counts['T1']} T2={counts['T2']} "
                    f"T3={counts['T3']} T4={counts['T4']} UNKNOWN={counts['UNKNOWN']}"
                )
                all_examples.extend(examples)
            except Exception as e:
                print(f"    ERROR extracting HIGHT-{b}/{k}: {e}")
                if not skip_errors:
                    raise


        # CHAM
    if do_cham:
        print("\n============================================================")
        print("Extracting CHAM")
        print("============================================================")
        cham_variants = [(64, 128), (128, 128), (128, 256)]
        for b, k in cham_variants:
            try:
                print(f"  Processing {b}/{k}...")
                examples = extract_cham_variant(root_dir, b, k, split)
                path = output_dir / f"cham_{b}_{k}_{split}.jsonl"
                write_jsonl(examples, path)
                counts = tier_counts(examples)
                print(
                    f"    {len(examples)} examples -> {path} | "
                    f"T1={counts['T1']} T2={counts['T2']} "
                    f"T3={counts['T3']} T4={counts['T4']} UNKNOWN={counts['UNKNOWN']}"
                )
                all_examples.extend(examples)
            except Exception as e:
                print(f"    ERROR extracting CHAM-{b}/{k}: {e}")
                if not skip_errors:
                    raise
                    

    # SIMON
    if do_simon:
        print("\n============================================================")
        print("Extracting SIMON")
        print("============================================================")
        simon_variants = [
            (32, 64), (48, 72), (48, 96),
            (64, 96), (64, 128),
            (96, 96), (96, 144),
            (128, 128), (128, 192), (128, 256)
        ]
        for b, k in simon_variants:
            try:
                print(f"  Processing {b}/{k}...")
                examples = extract_simon_variant(root_dir, b, k, split)
                path = output_dir / f"simon_{b}_{k}_{split}.jsonl"
                write_jsonl(examples, path)
                counts = tier_counts(examples)
                print(
                    f"    {len(examples)} examples -> {path} | "
                    f"T1={counts['T1']} T2={counts['T2']} "
                    f"T3={counts['T3']} T4={counts['T4']} UNKNOWN={counts['UNKNOWN']}"
                )
                all_examples.extend(examples)
            except Exception as e:
                print(f"    ERROR extracting SIMON-{b}/{k}: {e}")
                if not skip_errors:
                    raise

    # PRESENT
    if do_present:
        print("\n============================================================")
        print("Extracting PRESENT")
        print("============================================================")
        present_variants = [(64, 80), (64, 128)]
        for b, k in present_variants:
            try:
                print(f"  Processing {b}/{k}...")
                examples = extract_present_variant(root_dir, b, k, split)
                path = output_dir / f"present_{b}_{k}_{split}.jsonl"
                write_jsonl(examples, path)
                counts = tier_counts(examples)
                print(
                    f"    {len(examples)} examples -> {path} | "
                    f"T1={counts['T1']} T2={counts['T2']} "
                    f"T3={counts['T3']} T4={counts['T4']} UNKNOWN={counts['UNKNOWN']}"
                )
                all_examples.extend(examples)
            except Exception as e:
                print(f"    ERROR extracting PRESENT-{b}/{k}: {e}")
                if not skip_errors:
                    raise

    # GIFT
    if do_gift:
        print("\n============================================================")
        print("Extracting GIFT")
        print("============================================================")
        gift_variants = [(64, 128), (128, 128)]
        for b, k in gift_variants:
            try:
                print(f"  Processing {b}/{k}...")
                examples = extract_gift_variant(root_dir, b, k, split)
                path = output_dir / f"gift_{b}_{k}_{split}.jsonl"
                write_jsonl(examples, path)
                counts = tier_counts(examples)
                print(
                    f"    {len(examples)} examples -> {path} | "
                    f"T1={counts['T1']} T2={counts['T2']} "
                    f"T3={counts['T3']} T4={counts['T4']} UNKNOWN={counts['UNKNOWN']}"
                )
                all_examples.extend(examples)
            except Exception as e:
                print(f"    ERROR extracting GIFT-{b}/{k}: {e}")
                if not skip_errors:
                    raise

    # ASCON
    if do_ascon:
        print("\n============================================================")
        print("Extracting ASCON")
        print("============================================================")
        ascon_variants = [(64, 128), (128, 128)]
        for b, k in ascon_variants:
            try:
                print(f"  Processing {b}/{k}...")
                examples = extract_ascon_variant(root_dir, b, k, split)
                path = output_dir / f"ascon_{b}_{k}_{split}.jsonl"
                write_jsonl(examples, path)
                counts = tier_counts(examples)
                print(
                    f"    {len(examples)} examples -> {path} | "
                    f"T1={counts['T1']} T2={counts['T2']} "
                    f"T3={counts['T3']} T4={counts['T4']} UNKNOWN={counts['UNKNOWN']}"
                )
                all_examples.extend(examples)
            except Exception as e:
                print(f"    ERROR extracting ASCON-{b}/{k}: {e}")
                if not skip_errors:
                    raise

    # SIMECK
    if do_simeck:
        print("\n============================================================")
        print("Extracting SIMECK")
        print("============================================================")
        simeck_variants = [(32, 64), (48, 96), (64, 128)]
        for b, k in simeck_variants:
            try:
                print(f"  Processing {b}/{k}...")
                examples = extract_simeck_variant(root_dir, b, k, split)
                path = output_dir / f"simeck_{b}_{k}_{split}.jsonl"
                write_jsonl(examples, path)
                counts = tier_counts(examples)
                print(
                    f"    {len(examples)} examples -> {path} | "
                    f"T1={counts['T1']} T2={counts['T2']} "
                    f"T3={counts['T3']} T4={counts['T4']} UNKNOWN={counts['UNKNOWN']}"
                )
                all_examples.extend(examples)
            except Exception as e:
                print(f"    ERROR extracting SIMECK-{b}/{k}: {e}")
                if not skip_errors:
                    raise

    # RECTANGLE
    if do_rectangle:
        print("\n============================================================")
        print("Extracting RECTANGLE")
        print("============================================================")
        rectangle_variants = [(64, 80), (64, 128)]
        for b, k in rectangle_variants:
            try:
                print(f"  Processing {b}/{k}...")
                examples = extract_rectangle_variant(root_dir, b, k, split)
                path = output_dir / f"rectangle_{b}_{k}_{split}.jsonl"
                write_jsonl(examples, path)
                counts = tier_counts(examples)
                print(
                    f"    {len(examples)} examples -> {path} | "
                    f"T1={counts['T1']} T2={counts['T2']} "
                    f"T3={counts['T3']} T4={counts['T4']} UNKNOWN={counts['UNKNOWN']}"
                )
                all_examples.extend(examples)
            except Exception as e:
                print(f"    ERROR extracting RECTANGLE-{b}/{k}: {e}")
                if not skip_errors:
                    raise

    # SKINNY
    if do_skinny:
        print("\n============================================================")
        print("Extracting SKINNY")
        print("============================================================")
        skinny_variants = [(64, 128), (64, 192), (128, 128), (128, 256)]
        for b, k in skinny_variants:
            try:
                print(f"  Processing {b}/{k}...")
                examples = extract_skinny_variant(root_dir, b, k, split)
                path = output_dir / f"skinny_{b}_{k}_{split}.jsonl"
                write_jsonl(examples, path)
                counts = tier_counts(examples)
                print(
                    f"    {len(examples)} examples -> {path} | "
                    f"T1={counts['T1']} T2={counts['T2']} "
                    f"T3={counts['T3']} T4={counts['T4']} UNKNOWN={counts['UNKNOWN']}"
                )
                all_examples.extend(examples)
            except Exception as e:
                print(f"    ERROR extracting SKINNY-{b}/{k}: {e}")
                if not skip_errors:
                    raise

    # GIFT-COFB
    if do_gift_cofb:
        print("\n============================================================")
        print("Extracting GIFT-COFB")
        print("============================================================")
        gift_cofb_variants = [(128, 128)]
        for b, k in gift_cofb_variants:
            try:
                print(f"  Processing {b}/{k}...")
                examples = extract_gift_cofb_variant(root_dir, b, k, split)
                path = output_dir / f"gift_cofb_{b}_{k}_{split}.jsonl"
                write_jsonl(examples, path)
                counts = tier_counts(examples)
                print(
                    f"    {len(examples)} examples -> {path} | "
                    f"T1={counts['T1']} T2={counts['T2']} "
                    f"T3={counts['T3']} T4={counts['T4']} UNKNOWN={counts['UNKNOWN']}"
                )
                all_examples.extend(examples)
            except Exception as e:
                print(f"    ERROR extracting GIFT-COFB-{b}/{k}: {e}")
                if not skip_errors:
                    raise

    # XTEA
    if do_xtea:
        print("\n============================================================")
        print("Extracting XTEA")
        print("============================================================")
        xtea_variants = [(64, 128)]
        for b, k in xtea_variants:
            try:
                print(f"  Processing {b}/{k}...")
                examples = extract_xtea_variant(root_dir, b, k, split)
                path = output_dir / f"xtea_{b}_{k}_{split}.jsonl"
                write_jsonl(examples, path)
                counts = tier_counts(examples)
                print(
                    f"    {len(examples)} examples -> {path} | "
                    f"T1={counts['T1']} T2={counts['T2']} "
                    f"T3={counts['T3']} T4={counts['T4']} UNKNOWN={counts['UNKNOWN']}"
                )
                all_examples.extend(examples)
            except Exception as e:
                print(f"    ERROR extracting XTEA-{b}/{k}: {e}")
                if not skip_errors:
                    raise

    # next cipher

    totals = tier_counts(all_examples)
    print("\n============================================================")
    print("Global Summary")
    print("============================================================")
    print(
        f"total={len(all_examples)} "
        f"T1={totals['T1']} T2={totals['T2']} "
        f"T3={totals['T3']} T4={totals['T4']} "
        f"UNKNOWN={totals['UNKNOWN']}"
    )

    # Write combined file for all ciphers/variants
    combined_path = output_dir / f"all_{split}.jsonl"
    write_jsonl(all_examples, combined_path)
    print(f"\nCombined dataset written to {combined_path}")

    return all_examples


# ======================================================================
# CLI
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Unified cipher extractor")
    parser.add_argument("--ciphers", nargs="+", default=["all"])
    parser.add_argument("--root-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--skip-errors", action="store_true")
    args = parser.parse_args()

    extract_all_ciphers(
        ciphers=args.ciphers,
        root_dir=Path(args.root_dir),
        output_dir=Path(args.output_dir),
        split=args.split,
        skip_errors=args.skip_errors,
    )


if __name__ == "__main__":
    main()