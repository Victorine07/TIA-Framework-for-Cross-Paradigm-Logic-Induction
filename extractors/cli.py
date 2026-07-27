from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Any

from .run_sparx_dataset import (
    SPARX_VARIANTS,
    extract_variant as extract_sparx_variant,
    build_variant_config as build_sparx_variant_config,
    build_cipher_info as build_sparx_cipher_info,
)
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



# ============================================================================
# SPECK FUNCTIONS
# ============================================================================

# Speck variants from the specification
SPECK_VARIANTS = [
    {"block_size": 32, "key_size": 64},
    {"block_size": 48, "key_size": 72},
    {"block_size": 48, "key_size": 96},
    {"block_size": 64, "key_size": 96},
    {"block_size": 64, "key_size": 128},
    {"block_size": 96, "key_size": 96},
    {"block_size": 96, "key_size": 144},
    {"block_size": 128, "key_size": 128},
    {"block_size": 128, "key_size": 192},
    {"block_size": 128, "key_size": 256},
]


def speck_supported_variants() -> List[Tuple[int, int]]:
    """Return list of supported Speck variants."""
    return [(v["block_size"], v["key_size"]) for v in SPECK_VARIANTS]


def extract_single_speck_variant(
    root_dir: Path,
    block_size: int,
    key_size: int,
    split: str,
) -> List[Dict[str, Any]]:
    """Extract a single Speck variant."""
    from .base_extractor import BaseCipherExtractor
    
    project_root = root_dir
    
    # Construct file paths
    py_file = project_root / "python ciphers" / f"speck_{block_size}_{key_size}.py"
    thy_file = project_root / "thy ciphers" / f"Speck_{block_size}_{key_size}.thy"
    
    if not py_file.exists():
        raise FileNotFoundError(f"Python file not found: {py_file}")
    if not thy_file.exists():
        raise FileNotFoundError(f"Isabelle file not found: {thy_file}")
    
    # Read source files
    py_content = py_file.read_text(encoding="utf-8")
    thy_content = thy_file.read_text(encoding="utf-8")

     
    word_size = block_size // 2
    key_words = key_size // word_size
    rounds_map = {
        (32, 64): 22, (48, 72): 22, (48, 96): 23,
        (64, 96): 26, (64, 128): 27, (96, 96): 28,
        (96, 144): 29, (128, 128): 32, (128, 192): 33, (128, 256): 34,
    }
    rounds = rounds_map.get((block_size, key_size), 22)

    
    variant_config = {
        "block_size": block_size,
        "key_size": key_size,
        "word_size": word_size,
        "key_words": key_words,
        "rounds": rounds,
        "branches": 2,
        "words_per_block": 2,
        "steps": 1,
        "rounds_per_step": rounds,
        "total_rounds": rounds,
    }
    
    # Create extractor
    extractor = SpeckExtractor(
        root_dir=str(project_root),
        cipher=f"speck_{block_size}_{key_size}",
        family="ARX",
        subfamily="SPECK",
        block_size=block_size,
        key_size=key_size,
        variant_config=variant_config,
        dataset_split=split,
    )
    
    # # Override content since base extractor expects files in specific locations
    # extractor.python_source = py_content
    # extractor.thy_source = thy_content
    
    extractor.set_source_files(py_file, thy_file)
    
    # Extract components
    records = extractor.extract_components()
    
    # Convert to dicts
    examples = []
    for record in records:
        if hasattr(record, 'to_dict'):
            examples.append(record.to_dict())
        else:
            examples.append(record)
    
    # Attach split metadata
    for ex in examples:
        ex.setdefault("metadata", {})["split"] = split
    
    return examples


def write_single_speck_variant(
    root_dir: Path,
    block_size: int,
    key_size: int,
    split: str,
    output_dir: Path | None,
) -> Path:
    """Extract and write a single Speck variant to JSON."""
    import json
    
    project_root = root_dir
    out_dir = output_dir or (project_root / "output")
    out_dir.mkdir(exist_ok=True)
    
    examples = extract_single_speck_variant(project_root, block_size, key_size, split)
    
    filename = f"speck_{block_size}_{key_size}_{split}.json"
    output_path = out_dir / filename
    output_path.write_text(json.dumps(examples, indent=2), encoding="utf-8")
    
    return output_path


def extract_all_speck_variants(
    root_dir: Path,
    split: str,
) -> Dict[Tuple[int, int], List[Dict[str, Any]]]:
    """Extract all Speck variants."""
    results: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for variant in SPECK_VARIANTS:
        block_size = variant["block_size"]
        key_size = variant["key_size"]
        print(f"  Extracting speck_{block_size}_{key_size}...")
        try:
            examples = extract_single_speck_variant(root_dir, block_size, key_size, split)
            results[(block_size, key_size)] = examples
        except Exception as e:
            print(f"    ERROR: {e}")
            results[(block_size, key_size)] = []
    return results


def write_all_speck_variants(
    root_dir: Path,
    split: str,
    output_dir: Path | None,
) -> List[Path]:
    """Extract and write all Speck variants."""
    written: List[Path] = []
    for variant in SPECK_VARIANTS:
        block_size = variant["block_size"]
        key_size = variant["key_size"]
        try:
            path = write_single_speck_variant(root_dir, block_size, key_size, split, output_dir)
            written.append(path)
        except Exception as e:
            print(f"  ERROR writing speck_{block_size}_{key_size}: {e}")
    return written

    


# ============================================================================
# LEA FUNCTIONS
# ============================================================================

LEA_VARIANTS = [
    {"block_size": 128, "key_size": 128},
    {"block_size": 128, "key_size": 192},
    {"block_size": 128, "key_size": 256},
]


def lea_supported_variants() -> List[Tuple[int, int]]:
    return [(v["block_size"], v["key_size"]) for v in LEA_VARIANTS]


def extract_single_lea_variant(
    root_dir: Path,
    block_size: int,
    key_size: int,
    split: str,
) -> List[Dict[str, Any]]:
    """Extract a single LEA variant."""
    project_root = root_dir
    
    py_file = project_root / "python ciphers" / f"lea_{block_size}_{key_size}.py"
    thy_file = project_root / "thy ciphers" / f"Lea_{block_size}_{key_size}.thy"
    
    if not py_file.exists():
        raise FileNotFoundError(f"Python file not found: {py_file}")
    if not thy_file.exists():
        raise FileNotFoundError(f"Isabelle file not found: {thy_file}")
    
    py_content = py_file.read_text(encoding="utf-8")
    thy_content = thy_file.read_text(encoding="utf-8")
    
    # LEA parameters
    rounds_map = {128: 24, 192: 28, 256: 32}
    key_words_map = {128: 4, 192: 6, 256: 8}
    
    variant_config = {
        "block_size": block_size,
        "key_size": key_size,
        "word_size": 32,
        "key_words": key_words_map[key_size],
        "rounds": rounds_map[key_size],
        "branches": 4,  # LEA has 4 parallel branches
        "words_per_block": 4,
        "steps": 1,
        "rounds_per_step": rounds_map[key_size],
        "total_rounds": rounds_map[key_size],
        "round_key_words": 6,
    }
    
    extractor = LeaExtractor(
        root_dir=str(project_root),
        cipher=f"lea_{block_size}_{key_size}",
        family="ARX",
        subfamily="LEA",
        block_size=block_size,
        key_size=key_size,
        variant_config=variant_config,
        dataset_split=split,
    )
    
    extractor.set_source_files(py_file, thy_file)
    
    records = extractor.extract_components()
    
    examples = []
    for record in records:
        if hasattr(record, 'to_dict'):
            examples.append(record.to_dict())
        else:
            examples.append(record)
    
    for ex in examples:
        ex.setdefault("metadata", {})["split"] = split
    
    return examples


def write_single_lea_variant(
    root_dir: Path,
    block_size: int,
    key_size: int,
    split: str,
    output_dir: Path | None,
) -> Path:
    import json
    
    project_root = root_dir
    out_dir = output_dir or (project_root / "output")
    out_dir.mkdir(exist_ok=True)
    
    examples = extract_single_lea_variant(project_root, block_size, key_size, split)
    
    filename = f"lea_{block_size}_{key_size}_{split}.json"
    output_path = out_dir / filename
    output_path.write_text(json.dumps(examples, indent=2), encoding="utf-8")
    
    return output_path


def extract_all_lea_variants(
    root_dir: Path,
    split: str,
) -> Dict[Tuple[int, int], List[Dict[str, Any]]]:
    results: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for variant in LEA_VARIANTS:
        block_size = variant["block_size"]
        key_size = variant["key_size"]
        print(f"  Extracting lea_{block_size}_{key_size}...")
        try:
            examples = extract_single_lea_variant(root_dir, block_size, key_size, split)
            results[(block_size, key_size)] = examples
        except Exception as e:
            print(f"    ERROR: {e}")
            results[(block_size, key_size)] = []
    return results


def write_all_lea_variants(
    root_dir: Path,
    split: str,
    output_dir: Path | None,
) -> List[Path]:
    written: List[Path] = []
    for variant in LEA_VARIANTS:
        block_size = variant["block_size"]
        key_size = variant["key_size"]
        try:
            path = write_single_lea_variant(root_dir, block_size, key_size, split, output_dir)
            written.append(path)
        except Exception as e:
            print(f"  ERROR writing lea_{block_size}_{key_size}: {e}")
    return written


# ============================================================================
# SPARX FUNCTIONS (existing)
# ============================================================================

### OTHERS WITH SPARX

CipherVariant = Tuple[int, int]
JsonExample = Dict[str, Any]

SingleExtractFn = Callable[[Path, int, int, str], List[JsonExample]]
SingleWriteFn = Callable[[Path, int, int, str, Path | None], Path]
AllExtractFn = Callable[[Path, str], Dict[CipherVariant, List[JsonExample]]]
AllWriteFn = Callable[[Path, str, Path | None], List[Path]]


def sparx_supported_variants() -> List[CipherVariant]:
    return [(v["block_size"], v["key_size"]) for v in SPARX_VARIANTS]


def extract_single_sparx_variant(
    root_dir: Path,
    block_size: int,
    key_size: int,
    split: str,
) -> List[JsonExample]:
    project_root = root_dir
    examples = extract_sparx_variant(project_root, block_size, key_size)
    # attach split to metadata for downstream
    for ex in examples:
        ex.setdefault("metadata", {})["split"] = split
    return examples


def write_single_sparx_variant(
    root_dir: Path,
    block_size: int,
    key_size: int,
    split: str,
    output_dir: Path | None,
) -> Path:
    from .run_sparx_dataset import extract_variant
    import json

    project_root = root_dir
    out_dir = output_dir or (project_root / "output")
    out_dir.mkdir(exist_ok=True)

    examples = extract_variant(project_root, block_size, key_size)
    for ex in examples:
        ex.setdefault("metadata", {})["split"] = split

    filename = f"sparx_{block_size}_{key_size}_{split}.json"
    output_path = out_dir / filename
    output_path.write_text(json.dumps(examples, indent=2), encoding="utf-8")
    return output_path


def extract_all_sparx_variants(
    root_dir: Path,
    split: str,
) -> Dict[CipherVariant, List[JsonExample]]:
    results: Dict[CipherVariant, List[JsonExample]] = {}
    for variant in SPARX_VARIANTS:
        block_size = variant["block_size"]
        key_size = variant["key_size"]
        examples = extract_single_sparx_variant(root_dir, block_size, key_size, split)
        results[(block_size, key_size)] = examples
    return results


def write_all_sparx_variants(
    root_dir: Path,
    split: str,
    output_dir: Path | None,
) -> List[Path]:
    written: List[Path] = []
    for variant in SPARX_VARIANTS:
        block_size = variant["block_size"]
        key_size = variant["key_size"]
        path = write_single_sparx_variant(root_dir, block_size, key_size, split, output_dir)
        written.append(path)
    return written

# ============================================================================
# HIGHT FUNCTIONS
# ============================================================================

HIGHT_VARIANTS = [
    {"block_size": 64, "key_size": 128},
]

def hight_supported_variants() -> List[Tuple[int, int]]:
    return [(v["block_size"], v["key_size"]) for v in HIGHT_VARIANTS]

def extract_single_hight_variant(
    root_dir: Path,
    block_size: int,
    key_size: int,
    split: str,
) -> List[Dict[str, Any]]:
    """Extract a single HIGHT variant."""
    project_root = root_dir

    py_file = project_root / "python ciphers" / f"hight_{block_size}_{key_size}.py"
    thy_file = project_root / "thy ciphers" / f"Hight_{block_size}_{key_size}.thy"

    if not py_file.exists():
        raise FileNotFoundError(f"Python file not found: {py_file}")
    if not thy_file.exists():
        raise FileNotFoundError(f"Isabelle file not found: {thy_file}")

    variant_config = {
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
    }

    extractor = HightExtractor(
        root_dir=str(project_root),
        cipher=f"hight_{block_size}_{key_size}",
        family="Feistel",
        subfamily="HIGHT",
        block_size=block_size,
        key_size=key_size,
        variant_config=variant_config,
        dataset_split=split,
    )

    extractor.set_source_files(py_file, thy_file)
    records = extractor.extract_components()

    examples = []
    for record in records:
        if hasattr(record, "to_dict"):
            examples.append(record.to_dict())
        else:
            examples.append(record)

    for ex in examples:
        ex.setdefault("metadata", {})["split"] = split

    return examples

def write_single_hight_variant(
    root_dir: Path,
    block_size: int,
    key_size: int,
    split: str,
    output_dir: Path | None,
) -> Path:
    import json

    project_root = root_dir
    out_dir = output_dir or (project_root / "output")
    out_dir.mkdir(exist_ok=True)

    examples = extract_single_hight_variant(project_root, block_size, key_size, split)

    filename = f"hight_{block_size}_{key_size}_{split}.json"
    output_path = out_dir / filename
    output_path.write_text(json.dumps(examples, indent=2), encoding="utf-8")
    return output_path

def extract_all_hight_variants(
    root_dir: Path,
    split: str,
) -> Dict[Tuple[int, int], List[Dict[str, Any]]]:
    results: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for variant in HIGHT_VARIANTS:
        block_size = variant["block_size"]
        key_size = variant["key_size"]
        print(f"  Extracting hight_{block_size}_{key_size}...")
        try:
            examples = extract_single_hight_variant(root_dir, block_size, key_size, split)
            results[(block_size, key_size)] = examples
        except Exception as e:
            print(f"    ERROR: {e}")
            results[(block_size, key_size)] = []
    return results


def write_all_hight_variants(
    root_dir: Path,
    split: str,
    output_dir: Path | None,
) -> List[Path]:
    written: List[Path] = []
    for variant in HIGHT_VARIANTS:
        block_size = variant["block_size"]
        key_size = variant["key_size"]
        try:
            path = write_single_hight_variant(root_dir, block_size, key_size, split, output_dir)
            written.append(path)
        except Exception as e:
            print(f"  ERROR writing hight_{block_size}_{key_size}: {e}")
    return written


# ============================================================================
# CHAM FUNCTIONS
# ============================================================================

CHAM_VARIANTS = [
    {"block_size": 64, "key_size": 128},
    {"block_size": 128, "key_size": 128},
    {"block_size": 128, "key_size": 256},
]


def cham_supported_variants() -> List[Tuple[int, int]]:
    """Return list of supported CHAM variants."""
    return [(v["block_size"], v["key_size"]) for v in CHAM_VARIANTS]


def extract_single_cham_variant(
    root_dir: Path,
    block_size: int,
    key_size: int,
    split: str,
) -> List[Dict[str, Any]]:
    """Extract a single CHAM variant."""
    project_root = root_dir

    py_file = project_root / "python ciphers" / f"cham_{block_size}_{key_size}.py"
    thy_file = project_root / "thy ciphers" / f"Cham_{block_size}_{key_size}.thy"

    if not py_file.exists():
        raise FileNotFoundError(f"Python file not found: {py_file}")
    if not thy_file.exists():
        raise FileNotFoundError(f"Isabelle file not found: {thy_file}")

    word_size = block_size // 4
    key_words = key_size // word_size
    block_words = block_size // word_size

    rounds_map = {
        (64, 128): 80,
        (128, 128): 112,
        (128, 256): 120,
    }
    rounds = rounds_map[(block_size, key_size)]

    variant_config = {
        "block_size": block_size,
        "key_size": key_size,
        "word_size": word_size,
        "key_words": key_words,
        "block_words": block_words,
        "rounds": rounds,
        "branches": 4,
        "words_per_block": block_words,
        "steps": 1,
        "rounds_per_step": rounds,
        "total_rounds": rounds,
    }

    extractor = ChamExtractor(
        root_dir=str(project_root),
        cipher=f"cham_{block_size}_{key_size}",
        family="ARX",
        subfamily="CHAM",
        block_size=block_size,
        key_size=key_size,
        variant_config=variant_config,
        dataset_split=split,
    )

    extractor.set_source_files(py_file, thy_file)
    records = extractor.extract_components()

    examples = []
    for record in records:
        if hasattr(record, "to_dict"):
            examples.append(record.to_dict())
        else:
            examples.append(record)

    for ex in examples:
        ex.setdefault("metadata", {})["split"] = split

    return examples


def write_single_cham_variant(
    root_dir: Path,
    block_size: int,
    key_size: int,
    split: str,
    output_dir: Path | None,
) -> Path:
    import json

    project_root = root_dir
    out_dir = output_dir or (project_root / "output")
    out_dir.mkdir(exist_ok=True)

    examples = extract_single_cham_variant(project_root, block_size, key_size, split)

    filename = f"cham_{block_size}_{key_size}_{split}.json"
    output_path = out_dir / filename
    output_path.write_text(json.dumps(examples, indent=2), encoding="utf-8")
    return output_path


def extract_all_cham_variants(
    root_dir: Path,
    split: str,
) -> Dict[Tuple[int, int], List[Dict[str, Any]]]:
    results: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for variant in CHAM_VARIANTS:
        block_size = variant["block_size"]
        key_size = variant["key_size"]
        print(f"  Extracting cham_{block_size}_{key_size}...")
        try:
            examples = extract_single_cham_variant(root_dir, block_size, key_size, split)
            results[(block_size, key_size)] = examples
        except Exception as e:
            print(f"    ERROR: {e}")
            results[(block_size, key_size)] = []
    return results


def write_all_cham_variants(
    root_dir: Path,
    split: str,
    output_dir: Path | None,
) -> List[Path]:
    written: List[Path] = []
    for variant in CHAM_VARIANTS:
        block_size = variant["block_size"]
        key_size = variant["key_size"]
        try:
            path = write_single_cham_variant(root_dir, block_size, key_size, split, output_dir)
            written.append(path)
        except Exception as e:
            print(f"  ERROR writing cham_{block_size}_{key_size}: {e}")
    return written
    

# ============================================================================
# SIMON FUNCTIONS
# ============================================================================

SIMON_VARIANTS = [
    {"block_size": 32, "key_size": 64},
    {"block_size": 48, "key_size": 72},
    {"block_size": 48, "key_size": 96},
    {"block_size": 64, "key_size": 96},
    {"block_size": 64, "key_size": 128},
    {"block_size": 96, "key_size": 96},
    {"block_size": 96, "key_size": 144},
    {"block_size": 128, "key_size": 128},
    {"block_size": 128, "key_size": 192},
    {"block_size": 128, "key_size": 256},
]


def simon_supported_variants() -> List[Tuple[int, int]]:
    """Return list of supported SIMON variants."""
    return [(v["block_size"], v["key_size"]) for v in SIMON_VARIANTS]


def extract_single_simon_variant(
    root_dir: Path,
    block_size: int,
    key_size: int,
    split: str,
) -> List[Dict[str, Any]]:
    """Extract a single SIMON variant."""
    project_root = root_dir

    py_file = project_root / "python ciphers" / f"simon_{block_size}_{key_size}.py"
    thy_file = project_root / "thy ciphers" / f"Simon_{block_size}_{key_size}.thy"

    if not py_file.exists():
        raise FileNotFoundError(f"Python file not found: {py_file}")
    if not thy_file.exists():
        raise FileNotFoundError(f"Isabelle file not found: {thy_file}")

    word_size = block_size // 2
    key_words = key_size // word_size

    rounds_map = {
        (32, 64): 32,
        (48, 72): 36,
        (48, 96): 36,
        (64, 96): 42,
        (64, 128): 44,
        (96, 96): 52,
        (96, 144): 54,
        (128, 128): 68,
        (128, 192): 69,
        (128, 256): 72,
    }
    rounds = rounds_map[(block_size, key_size)]

    variant_config = {
        "block_size": block_size,
        "key_size": key_size,
        "word_size": word_size,
        "key_words": key_words,
        "rounds": rounds,
        "branches": 2,
        "words_per_block": 2,
        "steps": 1,
        "rounds_per_step": rounds,
        "total_rounds": rounds,
    }

    extractor = SimonExtractor(
        root_dir=str(project_root),
        cipher=f"simon_{block_size}_{key_size}",
        family="Feistel",
        subfamily="SIMON",
        block_size=block_size,
        key_size=key_size,
        variant_config=variant_config,
        dataset_split=split,
    )

    extractor.set_source_files(py_file, thy_file)
    records = extractor.extract_components()

    examples = []
    for record in records:
        if hasattr(record, "to_dict"):
            examples.append(record.to_dict())
        else:
            examples.append(record)

    for ex in examples:
        ex.setdefault("metadata", {})["split"] = split

    return examples


def write_single_simon_variant(
    root_dir: Path,
    block_size: int,
    key_size: int,
    split: str,
    output_dir: Path | None,
) -> Path:
    import json

    project_root = root_dir
    out_dir = output_dir or (project_root / "output")
    out_dir.mkdir(exist_ok=True)

    examples = extract_single_simon_variant(project_root, block_size, key_size, split)

    filename = f"simon_{block_size}_{key_size}_{split}.json"
    output_path = out_dir / filename
    output_path.write_text(json.dumps(examples, indent=2), encoding="utf-8")
    return output_path


def extract_all_simon_variants(
    root_dir: Path,
    split: str,
) -> Dict[Tuple[int, int], List[Dict[str, Any]]]:
    results: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for variant in SIMON_VARIANTS:
        block_size = variant["block_size"]
        key_size = variant["key_size"]
        print(f"  Extracting simon_{block_size}_{key_size}...")
        try:
            examples = extract_single_simon_variant(root_dir, block_size, key_size, split)
            results[(block_size, key_size)] = examples
        except Exception as e:
            print(f"    ERROR: {e}")
            results[(block_size, key_size)] = []
    return results


def write_all_simon_variants(
    root_dir: Path,
    split: str,
    output_dir: Path | None,
) -> List[Path]:
    written: List[Path] = []
    for variant in SIMON_VARIANTS:
        block_size = variant["block_size"]
        key_size = variant["key_size"]
        try:
            path = write_single_simon_variant(root_dir, block_size, key_size, split, output_dir)
            written.append(path)
        except Exception as e:
            print(f"  ERROR writing simon_{block_size}_{key_size}: {e}")
    return written


# ============================================================================
# PRESENT FUNCTIONS
# ============================================================================

PRESENT_VARIANTS = [
    {"block_size": 64, "key_size": 80},
    {"block_size": 64, "key_size": 128},
]


def present_supported_variants() -> List[Tuple[int, int]]:
    """Return list of supported PRESENT variants."""
    return [(v["block_size"], v["key_size"]) for v in PRESENT_VARIANTS]


def extract_single_present_variant(
    root_dir: Path,
    block_size: int,
    key_size: int,
    split: str,
) -> List[Dict[str, Any]]:
    """Extract a single PRESENT variant."""
    project_root = root_dir

    py_file = project_root / "python ciphers" / f"present_{block_size}_{key_size}.py"
    thy_file = project_root / "thy ciphers" / f"Present_{block_size}_{key_size}.thy"

    if not py_file.exists():
        raise FileNotFoundError(f"Python file not found: {py_file}")
    if not thy_file.exists():
        raise FileNotFoundError(f"Isabelle file not found: {thy_file}")

    variant_config = {
        "block_size": block_size,
        "key_size": key_size,
        "word_size": 4,
        "rounds": 31,
        "branches": 1,
        "words_per_block": 16,
        "steps": 1,
        "rounds_per_step": 31,
        "total_rounds": 31,
    }

    extractor = PresentExtractor(
        root_dir=str(project_root),
        cipher=f"present_{block_size}_{key_size}",
        family="SPN",
        subfamily="PRESENT",
        block_size=block_size,
        key_size=key_size,
        variant_config=variant_config,
        dataset_split=split,
    )

    extractor.set_source_files(py_file, thy_file)
    records = extractor.extract_components()

    examples = []
    for record in records:
        if hasattr(record, "to_dict"):
            examples.append(record.to_dict())
        else:
            examples.append(record)

    for ex in examples:
        ex.setdefault("metadata", {})["split"] = split

    return examples


def write_single_present_variant(
    root_dir: Path,
    block_size: int,
    key_size: int,
    split: str,
    output_dir: Path | None,
) -> Path:
    """Extract and write a single PRESENT variant to JSON."""
    import json

    project_root = root_dir
    out_dir = output_dir or (project_root / "output")
    out_dir.mkdir(exist_ok=True)

    examples = extract_single_present_variant(project_root, block_size, key_size, split)

    filename = f"present_{block_size}_{key_size}_{split}.json"
    output_path = out_dir / filename
    output_path.write_text(json.dumps(examples, indent=2), encoding="utf-8")

    return output_path


def extract_all_present_variants(
    root_dir: Path,
    split: str,
) -> Dict[Tuple[int, int], List[Dict[str, Any]]]:
    """Extract all PRESENT variants."""
    results: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for variant in PRESENT_VARIANTS:
        block_size = variant["block_size"]
        key_size = variant["key_size"]
        print(f"  Extracting present_{block_size}_{key_size}...")
        try:
            examples = extract_single_present_variant(root_dir, block_size, key_size, split)
            results[(block_size, key_size)] = examples
        except Exception as e:
            print(f"    ERROR: {e}")
            results[(block_size, key_size)] = []
    return results


def write_all_present_variants(
    root_dir: Path,
    split: str,
    output_dir: Path | None,
) -> List[Path]:
    """Extract and write all PRESENT variants."""
    written: List[Path] = []
    for variant in PRESENT_VARIANTS:
        block_size = variant["block_size"]
        key_size = variant["key_size"]
        try:
            path = write_single_present_variant(root_dir, block_size, key_size, split, output_dir)
            written.append(path)
        except Exception as e:
            print(f"  ERROR writing present_{block_size}_{key_size}: {e}")
    return written


# ============================================================================
# GIFT FUNCTIONS
# ============================================================================

GIFT_VARIANTS = [
    {"block_size": 64, "key_size": 128},
    {"block_size": 128, "key_size": 128},
]


def gift_supported_variants() -> List[Tuple[int, int]]:
    """Return list of supported GIFT variants."""
    return [(v["block_size"], v["key_size"]) for v in GIFT_VARIANTS]


def extract_single_gift_variant(
    root_dir: Path,
    block_size: int,
    key_size: int,
    split: str,
) -> List[Dict[str, Any]]:
    """Extract a single GIFT variant."""
    project_root = root_dir

    py_file = project_root / "python ciphers" / f"gift_{block_size}_{key_size}.py"
    thy_file = project_root / "thy ciphers" / f"Gift_{block_size}_{key_size}.thy"

    if not py_file.exists():
        raise FileNotFoundError(f"Python file not found: {py_file}")
    if not thy_file.exists():
        raise FileNotFoundError(f"Isabelle file not found: {thy_file}")

    rounds_map = {(64, 128): 28, (128, 128): 40}
    rounds = rounds_map[(block_size, key_size)]

    variant_config = {
        "block_size": block_size,
        "key_size": key_size,
        "word_size": 4,
        "rounds": rounds,
        "branches": 1,
        "words_per_block": block_size // 4,
        "steps": 1,
        "rounds_per_step": rounds,
        "total_rounds": rounds,
    }

    extractor = GiftExtractor(
        root_dir=str(project_root),
        cipher=f"gift_{block_size}_{key_size}",
        family="SPN",
        subfamily="GIFT",
        block_size=block_size,
        key_size=key_size,
        variant_config=variant_config,
        dataset_split=split,
    )

    extractor.set_source_files(py_file, thy_file)
    records = extractor.extract_components()

    examples = []
    for record in records:
        if hasattr(record, "to_dict"):
            examples.append(record.to_dict())
        else:
            examples.append(record)

    for ex in examples:
        ex.setdefault("metadata", {})["split"] = split

    return examples


def write_single_gift_variant(
    root_dir: Path,
    block_size: int,
    key_size: int,
    split: str,
    output_dir: Path | None,
) -> Path:
    """Extract and write a single GIFT variant to JSON."""
    import json

    project_root = root_dir
    out_dir = output_dir or (project_root / "output")
    out_dir.mkdir(exist_ok=True)

    examples = extract_single_gift_variant(project_root, block_size, key_size, split)

    filename = f"gift_{block_size}_{key_size}_{split}.json"
    output_path = out_dir / filename
    output_path.write_text(json.dumps(examples, indent=2), encoding="utf-8")

    return output_path


def extract_all_gift_variants(
    root_dir: Path,
    split: str,
) -> Dict[Tuple[int, int], List[Dict[str, Any]]]:
    """Extract all GIFT variants."""
    results: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for variant in GIFT_VARIANTS:
        block_size = variant["block_size"]
        key_size = variant["key_size"]
        print(f"  Extracting gift_{block_size}_{key_size}...")
        try:
            examples = extract_single_gift_variant(root_dir, block_size, key_size, split)
            results[(block_size, key_size)] = examples
        except Exception as e:
            print(f"    ERROR: {e}")
            results[(block_size, key_size)] = []
    return results


def write_all_gift_variants(
    root_dir: Path,
    split: str,
    output_dir: Path | None,
) -> List[Path]:
    """Extract and write all GIFT variants."""
    written: List[Path] = []
    for variant in GIFT_VARIANTS:
        block_size = variant["block_size"]
        key_size = variant["key_size"]
        try:
            path = write_single_gift_variant(root_dir, block_size, key_size, split, output_dir)
            written.append(path)
        except Exception as e:
            print(f"  ERROR writing gift_{block_size}_{key_size}: {e}")
    return written


# ============================================================================
# ASCON FUNCTIONS
# ============================================================================

ASCON_VARIANTS = [
    {"block_size": 64, "key_size": 128},
    {"block_size": 128, "key_size": 128},
]


def ascon_supported_variants() -> List[Tuple[int, int]]:
    """Return list of supported ASCON variants (block_size here is the rate, in bits)."""
    return [(v["block_size"], v["key_size"]) for v in ASCON_VARIANTS]


def extract_single_ascon_variant(
    root_dir: Path,
    block_size: int,
    key_size: int,
    split: str,
) -> List[Dict[str, Any]]:
    """Extract a single ASCON variant."""
    project_root = root_dir

    py_file = project_root / "python ciphers" / f"ascon_{block_size}_{key_size}.py"
    thy_file = project_root / "thy ciphers" / f"Ascon_{block_size}_{key_size}.thy"

    if not py_file.exists():
        raise FileNotFoundError(f"Python file not found: {py_file}")
    if not thy_file.exists():
        raise FileNotFoundError(f"Isabelle file not found: {thy_file}")

    rounds_b_map = {(64, 128): 6, (128, 128): 8}
    rounds_b = rounds_b_map[(block_size, key_size)]

    variant_config = {
        "block_size": block_size,
        "key_size": key_size,
        "rate": block_size,
        "rounds_a": 12,
        "rounds_b": rounds_b,
    }

    extractor = AsconExtractor(
        root_dir=str(project_root),
        cipher=f"ascon_{block_size}_{key_size}",
        family="Permutation",
        subfamily="ASCON",
        block_size=block_size,
        key_size=key_size,
        variant_config=variant_config,
        dataset_split=split,
    )

    extractor.set_source_files(py_file, thy_file)
    records = extractor.extract_components()

    examples = []
    for record in records:
        if hasattr(record, "to_dict"):
            examples.append(record.to_dict())
        else:
            examples.append(record)

    for ex in examples:
        ex.setdefault("metadata", {})["split"] = split

    return examples


def write_single_ascon_variant(
    root_dir: Path,
    block_size: int,
    key_size: int,
    split: str,
    output_dir: Path | None,
) -> Path:
    """Extract and write a single ASCON variant to JSON."""
    import json

    project_root = root_dir
    out_dir = output_dir or (project_root / "output")
    out_dir.mkdir(exist_ok=True)

    examples = extract_single_ascon_variant(project_root, block_size, key_size, split)

    filename = f"ascon_{block_size}_{key_size}_{split}.json"
    output_path = out_dir / filename
    output_path.write_text(json.dumps(examples, indent=2), encoding="utf-8")

    return output_path


def extract_all_ascon_variants(
    root_dir: Path,
    split: str,
) -> Dict[Tuple[int, int], List[Dict[str, Any]]]:
    """Extract all ASCON variants."""
    results: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for variant in ASCON_VARIANTS:
        block_size = variant["block_size"]
        key_size = variant["key_size"]
        print(f"  Extracting ascon_{block_size}_{key_size}...")
        try:
            examples = extract_single_ascon_variant(root_dir, block_size, key_size, split)
            results[(block_size, key_size)] = examples
        except Exception as e:
            print(f"    ERROR: {e}")
            results[(block_size, key_size)] = []
    return results


def write_all_ascon_variants(
    root_dir: Path,
    split: str,
    output_dir: Path | None,
) -> List[Path]:
    """Extract and write all ASCON variants."""
    written: List[Path] = []
    for variant in ASCON_VARIANTS:
        block_size = variant["block_size"]
        key_size = variant["key_size"]
        try:
            path = write_single_ascon_variant(root_dir, block_size, key_size, split, output_dir)
            written.append(path)
        except Exception as e:
            print(f"  ERROR writing ascon_{block_size}_{key_size}: {e}")
    return written


# ============================================================================
# SIMECK FUNCTIONS
# ============================================================================

SIMECK_VARIANTS = [
    {"block_size": 32, "key_size": 64},
    {"block_size": 48, "key_size": 96},
    {"block_size": 64, "key_size": 128},
]


def simeck_supported_variants() -> List[Tuple[int, int]]:
    """Return list of supported SIMECK variants."""
    return [(v["block_size"], v["key_size"]) for v in SIMECK_VARIANTS]


def extract_single_simeck_variant(
    root_dir: Path,
    block_size: int,
    key_size: int,
    split: str,
) -> List[Dict[str, Any]]:
    """Extract a single SIMECK variant."""
    project_root = root_dir

    py_file = project_root / "python ciphers" / f"simeck_{block_size}_{key_size}.py"
    thy_file = project_root / "thy ciphers" / f"Simeck_{block_size}_{key_size}.thy"

    if not py_file.exists():
        raise FileNotFoundError(f"Python file not found: {py_file}")
    if not thy_file.exists():
        raise FileNotFoundError(f"Isabelle file not found: {thy_file}")

    word_size = block_size // 2
    rounds_map = {(32, 64): 32, (48, 96): 36, (64, 128): 44}
    rounds = rounds_map[(block_size, key_size)]

    variant_config = {
        "block_size": block_size,
        "key_size": key_size,
        "word_size": word_size,
        "key_words": key_size // word_size,
        "rounds": rounds,
        "branches": 2,
        "words_per_block": 2,
        "steps": 1,
        "rounds_per_step": rounds,
        "total_rounds": rounds,
    }

    extractor = SimeckExtractor(
        root_dir=str(project_root),
        cipher=f"simeck_{block_size}_{key_size}",
        family="Feistel",
        subfamily="SIMECK",
        block_size=block_size,
        key_size=key_size,
        variant_config=variant_config,
        dataset_split=split,
    )

    extractor.set_source_files(py_file, thy_file)
    records = extractor.extract_components()

    examples = []
    for record in records:
        if hasattr(record, "to_dict"):
            examples.append(record.to_dict())
        else:
            examples.append(record)

    for ex in examples:
        ex.setdefault("metadata", {})["split"] = split

    return examples


def write_single_simeck_variant(
    root_dir: Path,
    block_size: int,
    key_size: int,
    split: str,
    output_dir: Path | None,
) -> Path:
    """Extract and write a single SIMECK variant to JSON."""
    import json

    project_root = root_dir
    out_dir = output_dir or (project_root / "output")
    out_dir.mkdir(exist_ok=True)

    examples = extract_single_simeck_variant(project_root, block_size, key_size, split)

    filename = f"simeck_{block_size}_{key_size}_{split}.json"
    output_path = out_dir / filename
    output_path.write_text(json.dumps(examples, indent=2), encoding="utf-8")

    return output_path


def extract_all_simeck_variants(
    root_dir: Path,
    split: str,
) -> Dict[Tuple[int, int], List[Dict[str, Any]]]:
    """Extract all SIMECK variants."""
    results: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for variant in SIMECK_VARIANTS:
        block_size = variant["block_size"]
        key_size = variant["key_size"]
        print(f"  Extracting simeck_{block_size}_{key_size}...")
        try:
            examples = extract_single_simeck_variant(root_dir, block_size, key_size, split)
            results[(block_size, key_size)] = examples
        except Exception as e:
            print(f"    ERROR: {e}")
            results[(block_size, key_size)] = []
    return results


def write_all_simeck_variants(
    root_dir: Path,
    split: str,
    output_dir: Path | None,
) -> List[Path]:
    """Extract and write all SIMECK variants."""
    written: List[Path] = []
    for variant in SIMECK_VARIANTS:
        block_size = variant["block_size"]
        key_size = variant["key_size"]
        try:
            path = write_single_simeck_variant(root_dir, block_size, key_size, split, output_dir)
            written.append(path)
        except Exception as e:
            print(f"  ERROR writing simeck_{block_size}_{key_size}: {e}")
    return written


# ============================================================================
# RECTANGLE FUNCTIONS
# ============================================================================

RECTANGLE_VARIANTS = [
    {"block_size": 64, "key_size": 80},
    {"block_size": 64, "key_size": 128},
]


def rectangle_supported_variants() -> List[Tuple[int, int]]:
    """Return list of supported RECTANGLE variants."""
    return [(v["block_size"], v["key_size"]) for v in RECTANGLE_VARIANTS]


def extract_single_rectangle_variant(
    root_dir: Path,
    block_size: int,
    key_size: int,
    split: str,
) -> List[Dict[str, Any]]:
    """Extract a single RECTANGLE variant."""
    project_root = root_dir

    py_file = project_root / "python ciphers" / f"rectangle_{block_size}_{key_size}.py"
    thy_file = project_root / "thy ciphers" / f"Rectangle_{block_size}_{key_size}.thy"

    if not py_file.exists():
        raise FileNotFoundError(f"Python file not found: {py_file}")
    if not thy_file.exists():
        raise FileNotFoundError(f"Isabelle file not found: {thy_file}")

    variant_config = {
        "block_size": block_size,
        "key_size": key_size,
        "word_size": 4,
        "rounds": 25,
        "branches": 1,
        "words_per_block": 16,
        "steps": 1,
        "rounds_per_step": 25,
        "total_rounds": 25,
    }

    extractor = RectangleExtractor(
        root_dir=str(project_root),
        cipher=f"rectangle_{block_size}_{key_size}",
        family="SPN",
        subfamily="RECTANGLE",
        block_size=block_size,
        key_size=key_size,
        variant_config=variant_config,
        dataset_split=split,
    )

    extractor.set_source_files(py_file, thy_file)
    records = extractor.extract_components()

    examples = []
    for record in records:
        if hasattr(record, "to_dict"):
            examples.append(record.to_dict())
        else:
            examples.append(record)

    for ex in examples:
        ex.setdefault("metadata", {})["split"] = split

    return examples


def write_single_rectangle_variant(
    root_dir: Path,
    block_size: int,
    key_size: int,
    split: str,
    output_dir: Path | None,
) -> Path:
    """Extract and write a single RECTANGLE variant to JSON."""
    import json

    project_root = root_dir
    out_dir = output_dir or (project_root / "output")
    out_dir.mkdir(exist_ok=True)

    examples = extract_single_rectangle_variant(project_root, block_size, key_size, split)

    filename = f"rectangle_{block_size}_{key_size}_{split}.json"
    output_path = out_dir / filename
    output_path.write_text(json.dumps(examples, indent=2), encoding="utf-8")

    return output_path


def extract_all_rectangle_variants(
    root_dir: Path,
    split: str,
) -> Dict[Tuple[int, int], List[Dict[str, Any]]]:
    """Extract all RECTANGLE variants."""
    results: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for variant in RECTANGLE_VARIANTS:
        block_size = variant["block_size"]
        key_size = variant["key_size"]
        print(f"  Extracting rectangle_{block_size}_{key_size}...")
        try:
            examples = extract_single_rectangle_variant(root_dir, block_size, key_size, split)
            results[(block_size, key_size)] = examples
        except Exception as e:
            print(f"    ERROR: {e}")
            results[(block_size, key_size)] = []
    return results


def write_all_rectangle_variants(
    root_dir: Path,
    split: str,
    output_dir: Path | None,
) -> List[Path]:
    """Extract and write all RECTANGLE variants."""
    written: List[Path] = []
    for variant in RECTANGLE_VARIANTS:
        block_size = variant["block_size"]
        key_size = variant["key_size"]
        try:
            path = write_single_rectangle_variant(root_dir, block_size, key_size, split, output_dir)
            written.append(path)
        except Exception as e:
            print(f"  ERROR writing rectangle_{block_size}_{key_size}: {e}")
    return written


# ============================================================================
# SKINNY FUNCTIONS
# ============================================================================

SKINNY_VARIANTS = [
    {"block_size": 64, "key_size": 128},
    {"block_size": 64, "key_size": 192},
    {"block_size": 128, "key_size": 128},
    {"block_size": 128, "key_size": 256},
]


def skinny_supported_variants() -> List[Tuple[int, int]]:
    """Return list of supported SKINNY variants."""
    return [(v["block_size"], v["key_size"]) for v in SKINNY_VARIANTS]


def extract_single_skinny_variant(
    root_dir: Path,
    block_size: int,
    key_size: int,
    split: str,
) -> List[Dict[str, Any]]:
    """Extract a single SKINNY variant."""
    project_root = root_dir

    py_file = project_root / "python ciphers" / f"skinny_{block_size}_{key_size}.py"
    thy_file = project_root / "thy ciphers" / f"Skinny_{block_size}_{key_size}.thy"

    if not py_file.exists():
        raise FileNotFoundError(f"Python file not found: {py_file}")
    if not thy_file.exists():
        raise FileNotFoundError(f"Isabelle file not found: {thy_file}")

    rounds_map = {(64, 128): 36, (64, 192): 40, (128, 128): 40, (128, 256): 48}
    rounds = rounds_map[(block_size, key_size)]

    variant_config = {
        "block_size": block_size,
        "key_size": key_size,
        "word_size": block_size // 16,
        "rounds": rounds,
        "branches": 1,
        "words_per_block": 16,
        "steps": 1,
        "rounds_per_step": rounds,
        "total_rounds": rounds,
    }

    extractor = SkinnyExtractor(
        root_dir=str(project_root),
        cipher=f"skinny_{block_size}_{key_size}",
        family="SPN",
        subfamily="SKINNY",
        block_size=block_size,
        key_size=key_size,
        variant_config=variant_config,
        dataset_split=split,
    )

    extractor.set_source_files(py_file, thy_file)
    records = extractor.extract_components()

    examples = []
    for record in records:
        if hasattr(record, "to_dict"):
            examples.append(record.to_dict())
        else:
            examples.append(record)

    for ex in examples:
        ex.setdefault("metadata", {})["split"] = split

    return examples


def write_single_skinny_variant(
    root_dir: Path,
    block_size: int,
    key_size: int,
    split: str,
    output_dir: Path | None,
) -> Path:
    """Extract and write a single SKINNY variant to JSON."""
    import json

    project_root = root_dir
    out_dir = output_dir or (project_root / "output")
    out_dir.mkdir(exist_ok=True)

    examples = extract_single_skinny_variant(project_root, block_size, key_size, split)

    filename = f"skinny_{block_size}_{key_size}_{split}.json"
    output_path = out_dir / filename
    output_path.write_text(json.dumps(examples, indent=2), encoding="utf-8")

    return output_path


def extract_all_skinny_variants(
    root_dir: Path,
    split: str,
) -> Dict[Tuple[int, int], List[Dict[str, Any]]]:
    """Extract all SKINNY variants."""
    results: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for variant in SKINNY_VARIANTS:
        block_size = variant["block_size"]
        key_size = variant["key_size"]
        print(f"  Extracting skinny_{block_size}_{key_size}...")
        try:
            examples = extract_single_skinny_variant(root_dir, block_size, key_size, split)
            results[(block_size, key_size)] = examples
        except Exception as e:
            print(f"    ERROR: {e}")
            results[(block_size, key_size)] = []
    return results


def write_all_skinny_variants(
    root_dir: Path,
    split: str,
    output_dir: Path | None,
) -> List[Path]:
    """Extract and write all SKINNY variants."""
    written: List[Path] = []
    for variant in SKINNY_VARIANTS:
        block_size = variant["block_size"]
        key_size = variant["key_size"]
        try:
            path = write_single_skinny_variant(root_dir, block_size, key_size, split, output_dir)
            written.append(path)
        except Exception as e:
            print(f"  ERROR writing skinny_{block_size}_{key_size}: {e}")
    return written


# ============================================================================
# GIFT-COFB FUNCTIONS
# ============================================================================

GIFT_COFB_VARIANTS = [
    {"block_size": 128, "key_size": 128},
]


def gift_cofb_supported_variants() -> List[Tuple[int, int]]:
    """Return list of supported GIFT-COFB variants."""
    return [(v["block_size"], v["key_size"]) for v in GIFT_COFB_VARIANTS]


def extract_single_gift_cofb_variant(
    root_dir: Path,
    block_size: int,
    key_size: int,
    split: str,
) -> List[Dict[str, Any]]:
    """Extract a single GIFT-COFB variant."""
    project_root = root_dir

    py_file = project_root / "python ciphers" / f"gift_cofb_{block_size}_{key_size}.py"
    thy_file = project_root / "thy ciphers" / f"Gift_Cofb_{block_size}_{key_size}.thy"

    if not py_file.exists():
        raise FileNotFoundError(f"Python file not found: {py_file}")
    if not thy_file.exists():
        raise FileNotFoundError(f"Isabelle file not found: {thy_file}")

    variant_config = {
        "block_size": block_size,
        "key_size": key_size,
        "word_size": block_size // 4,
        "rounds": 40,
        "branches": 1,
        "words_per_block": 4,
        "steps": 1,
        "rounds_per_step": 40,
        "total_rounds": 40,
    }

    extractor = GiftCofbExtractor(
        root_dir=str(project_root),
        cipher=f"gift_cofb_{block_size}_{key_size}",
        family="Permutation/AEAD",
        subfamily="GIFT-COFB",
        block_size=block_size,
        key_size=key_size,
        variant_config=variant_config,
        dataset_split=split,
    )

    extractor.set_source_files(py_file, thy_file)
    records = extractor.extract_components()

    examples = []
    for record in records:
        if hasattr(record, "to_dict"):
            examples.append(record.to_dict())
        else:
            examples.append(record)

    for ex in examples:
        ex.setdefault("metadata", {})["split"] = split

    return examples


def write_single_gift_cofb_variant(
    root_dir: Path,
    block_size: int,
    key_size: int,
    split: str,
    output_dir: Path | None,
) -> Path:
    """Extract and write a single GIFT-COFB variant to JSON."""
    import json

    project_root = root_dir
    out_dir = output_dir or (project_root / "output")
    out_dir.mkdir(exist_ok=True)

    examples = extract_single_gift_cofb_variant(project_root, block_size, key_size, split)

    filename = f"gift_cofb_{block_size}_{key_size}_{split}.json"
    output_path = out_dir / filename
    output_path.write_text(json.dumps(examples, indent=2), encoding="utf-8")

    return output_path


def extract_all_gift_cofb_variants(
    root_dir: Path,
    split: str,
) -> Dict[Tuple[int, int], List[Dict[str, Any]]]:
    """Extract all GIFT-COFB variants."""
    results: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for variant in GIFT_COFB_VARIANTS:
        block_size = variant["block_size"]
        key_size = variant["key_size"]
        print(f"  Extracting gift_cofb_{block_size}_{key_size}...")
        try:
            examples = extract_single_gift_cofb_variant(root_dir, block_size, key_size, split)
            results[(block_size, key_size)] = examples
        except Exception as e:
            print(f"    ERROR: {e}")
            results[(block_size, key_size)] = []
    return results


def write_all_gift_cofb_variants(
    root_dir: Path,
    split: str,
    output_dir: Path | None,
) -> List[Path]:
    """Extract and write all GIFT-COFB variants."""
    written: List[Path] = []
    for variant in GIFT_COFB_VARIANTS:
        block_size = variant["block_size"]
        key_size = variant["key_size"]
        try:
            path = write_single_gift_cofb_variant(root_dir, block_size, key_size, split, output_dir)
            written.append(path)
        except Exception as e:
            print(f"  ERROR writing gift_cofb_{block_size}_{key_size}: {e}")
    return written


# ============================================================================
# XTEA FUNCTIONS
# ============================================================================

XTEA_VARIANTS = [
    {"block_size": 64, "key_size": 128},
]


def xtea_supported_variants() -> List[Tuple[int, int]]:
    """Return list of supported XTEA variants."""
    return [(v["block_size"], v["key_size"]) for v in XTEA_VARIANTS]


def extract_single_xtea_variant(
    root_dir: Path,
    block_size: int,
    key_size: int,
    split: str,
) -> List[Dict[str, Any]]:
    """Extract a single XTEA variant."""
    project_root = root_dir

    py_file = project_root / "python ciphers" / f"xtea_{block_size}_{key_size}.py"
    thy_file = project_root / "thy ciphers" / f"Xtea_{block_size}_{key_size}.thy"

    if not py_file.exists():
        raise FileNotFoundError(f"Python file not found: {py_file}")
    if not thy_file.exists():
        raise FileNotFoundError(f"Isabelle file not found: {thy_file}")

    variant_config = {
        "block_size": block_size,
        "key_size": key_size,
        "word_size": 32,
        "rounds": 32,
        "branches": 2,
        "words_per_block": 2,
        "steps": 1,
        "rounds_per_step": 32,
        "total_rounds": 32,
    }

    extractor = XteaExtractor(
        root_dir=str(project_root),
        cipher=f"xtea_{block_size}_{key_size}",
        family="Feistel",
        subfamily="XTEA",
        block_size=block_size,
        key_size=key_size,
        variant_config=variant_config,
        dataset_split=split,
    )

    extractor.set_source_files(py_file, thy_file)
    records = extractor.extract_components()

    examples = []
    for record in records:
        if hasattr(record, "to_dict"):
            examples.append(record.to_dict())
        else:
            examples.append(record)

    for ex in examples:
        ex.setdefault("metadata", {})["split"] = split

    return examples


def write_single_xtea_variant(
    root_dir: Path,
    block_size: int,
    key_size: int,
    split: str,
    output_dir: Path | None,
) -> Path:
    """Extract and write a single XTEA variant to JSON."""
    import json

    project_root = root_dir
    out_dir = output_dir or (project_root / "output")
    out_dir.mkdir(exist_ok=True)

    examples = extract_single_xtea_variant(project_root, block_size, key_size, split)

    filename = f"xtea_{block_size}_{key_size}_{split}.json"
    output_path = out_dir / filename
    output_path.write_text(json.dumps(examples, indent=2), encoding="utf-8")

    return output_path


def extract_all_xtea_variants(
    root_dir: Path,
    split: str,
) -> Dict[Tuple[int, int], List[Dict[str, Any]]]:
    """Extract all XTEA variants."""
    results: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for variant in XTEA_VARIANTS:
        block_size = variant["block_size"]
        key_size = variant["key_size"]
        print(f"  Extracting xtea_{block_size}_{key_size}...")
        try:
            examples = extract_single_xtea_variant(root_dir, block_size, key_size, split)
            results[(block_size, key_size)] = examples
        except Exception as e:
            print(f"    ERROR: {e}")
            results[(block_size, key_size)] = []
    return results


def write_all_xtea_variants(
    root_dir: Path,
    split: str,
    output_dir: Path | None,
) -> List[Path]:
    """Extract and write all XTEA variants."""
    written: List[Path] = []
    for variant in XTEA_VARIANTS:
        block_size = variant["block_size"]
        key_size = variant["key_size"]
        try:
            path = write_single_xtea_variant(root_dir, block_size, key_size, split, output_dir)
            written.append(path)
        except Exception as e:
            print(f"  ERROR writing xtea_{block_size}_{key_size}: {e}")
    return written


# ============================================================================
# SUPPORTED CIPHERS - ADD SPECK HERE
# ============================================================================

SUPPORTED_CIPHERS: Dict[str, Dict[str, Any]] = {
    "sparx": {
        "extract_single": extract_single_sparx_variant,
        "write_single": write_single_sparx_variant,
        "extract_all": extract_all_sparx_variants,
        "write_all": write_all_sparx_variants,
        "supported_variants": sparx_supported_variants,
    },
    # ADD SPECK HERE
    "speck": {
        "extract_single": extract_single_speck_variant,
        "write_single": write_single_speck_variant,
        "extract_all": extract_all_speck_variants,
        "write_all": write_all_speck_variants,
        "supported_variants": speck_supported_variants,
    },
    
    "lea": {
        "extract_single": extract_single_lea_variant,
        "write_single": write_single_lea_variant,
        "extract_all": extract_all_lea_variants,
        "write_all": write_all_lea_variants,
        "supported_variants": lea_supported_variants,
    },
    
    "hight": {
        "extract_single": extract_single_hight_variant,
        "write_single": write_single_hight_variant,
        "extract_all": extract_all_hight_variants,
        "write_all": write_all_hight_variants,
        "supported_variants": hight_supported_variants,
    },
    
    "cham": {
        "extract_single": extract_single_cham_variant,
        "write_single": write_single_cham_variant,
        "extract_all": extract_all_cham_variants,
        "write_all": write_all_cham_variants,
        "supported_variants": cham_supported_variants,
    },
    "simon": {
        "extract_single": extract_single_simon_variant,
        "write_single": write_single_simon_variant,
        "extract_all": extract_all_simon_variants,
        "write_all": write_all_simon_variants,
        "supported_variants": simon_supported_variants,
    },

    "present": {
        "extract_single": extract_single_present_variant,
        "write_single": write_single_present_variant,
        "extract_all": extract_all_present_variants,
        "write_all": write_all_present_variants,
        "supported_variants": present_supported_variants,
    },

    "gift": {
        "extract_single": extract_single_gift_variant,
        "write_single": write_single_gift_variant,
        "extract_all": extract_all_gift_variants,
        "write_all": write_all_gift_variants,
        "supported_variants": gift_supported_variants,
    },

    "ascon": {
        "extract_single": extract_single_ascon_variant,
        "write_single": write_single_ascon_variant,
        "extract_all": extract_all_ascon_variants,
        "write_all": write_all_ascon_variants,
        "supported_variants": ascon_supported_variants,
    },

    "simeck": {
        "extract_single": extract_single_simeck_variant,
        "write_single": write_single_simeck_variant,
        "extract_all": extract_all_simeck_variants,
        "write_all": write_all_simeck_variants,
        "supported_variants": simeck_supported_variants,
    },

    "rectangle": {
        "extract_single": extract_single_rectangle_variant,
        "write_single": write_single_rectangle_variant,
        "extract_all": extract_all_rectangle_variants,
        "write_all": write_all_rectangle_variants,
        "supported_variants": rectangle_supported_variants,
    },

    "skinny": {
        "extract_single": extract_single_skinny_variant,
        "write_single": write_single_skinny_variant,
        "extract_all": extract_all_skinny_variants,
        "write_all": write_all_skinny_variants,
        "supported_variants": skinny_supported_variants,
    },

    "gift_cofb": {
        "extract_single": extract_single_gift_cofb_variant,
        "write_single": write_single_gift_cofb_variant,
        "extract_all": extract_all_gift_cofb_variants,
        "write_all": write_all_gift_cofb_variants,
        "supported_variants": gift_cofb_supported_variants,
    },

    "xtea": {
        "extract_single": extract_single_xtea_variant,
        "write_single": write_single_xtea_variant,
        "extract_all": extract_all_xtea_variants,
        "write_all": write_all_xtea_variants,
        "supported_variants": xtea_supported_variants,
    },

}


def summarize_examples(examples: List[JsonExample]) -> Dict[str, int]:
    summary = {
        "total": len(examples),
        "T1": 0,
        "T2": 0,
        "T3": 0,
        "T4": 0,
    }
    for ex in examples:
        tier = ex.get("metadata", {}).get("tier")
        if tier in summary:
            summary[tier] += 1
    return summary


def print_variant_summary(cipher_name: str, block_size: int, key_size: int, examples: List[JsonExample]) -> None:
    summary = summarize_examples(examples)
    print(
        f"{cipher_name.upper()}-{block_size}/{key_size}: "
        f"total={summary['total']} "
        f"T1={summary['T1']} "
        f"T2={summary['T2']} "
        f"T3={summary['T3']} "
        f"T4={summary['T4']}"
    )


def print_all_summaries(
    cipher_name: str,
    results: Dict[CipherVariant, List[JsonExample]],
) -> None:
    for (block_size, key_size) in sorted(results.keys()):
        examples = results[(block_size, key_size)]
        print_variant_summary(cipher_name, block_size, key_size, examples)


def validate_cipher_name(cipher_name: str) -> str:
    normalized = cipher_name.strip().lower()
    if normalized not in SUPPORTED_CIPHERS:
        supported = ", ".join(sorted(SUPPORTED_CIPHERS.keys()))
        raise SystemExit(f"Unsupported cipher '{cipher_name}'. Supported ciphers: {supported}")
    return normalized


def validate_variant(cipher_name: str, block_size: int, key_size: int) -> None:
    supported_variants = SUPPORTED_CIPHERS[cipher_name]["supported_variants"]()
    if (block_size, key_size) not in supported_variants:
        supported = ", ".join(f"{b}/{k}" for (b, k) in supported_variants)
        raise SystemExit(
            f"Unsupported variant {cipher_name}-{block_size}/{key_size}. "
            f"Supported variants: {supported}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified extractor CLI for Python-to-Isabelle dataset generation."
    )
    parser.add_argument(
        "--cipher",
        type=str,
        required=True,
        help="Cipher name, e.g. sparx.",
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        default=".",
        help="Repository root containing extractor inputs and outputs.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "valid", "test"],
        help="Dataset split label.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional explicit output directory.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=None,
        help="Block size for single-variant mode.",
    )
    parser.add_argument(
        "--key-size",
        type=int,
        default=None,
        help="Key size for single-variant mode.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all supported variants for the selected cipher.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print extraction summaries without writing files.",
    )
    parser.add_argument(
        "--list-variants",
        action="store_true",
        help="List supported variants for the selected cipher and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cipher_name = validate_cipher_name(args.cipher)
    root_dir = Path(args.root_dir).resolve()

    cipher_entry = SUPPORTED_CIPHERS[cipher_name]
    supported_variants = cipher_entry["supported_variants"]()

    if args.list_variants:
        print(f"Supported variants for {cipher_name}:")
        for block_size, key_size in supported_variants:
            print(f"  - {block_size}/{key_size}")
        return

    if args.all:
        results = cipher_entry["extract_all"](
            root_dir=root_dir,
            split=args.split,
        )
        print_all_summaries(cipher_name, results)

        if not args.summary_only:
            written_paths = cipher_entry["write_all"](
                root_dir=root_dir,
                split=args.split,
                output_dir=Path(args.output_dir) if args.output_dir else None,
            )
            for path in written_paths:
                print(f"WROTE {path}")
        return

    if args.block_size is None or args.key_size is None:
        raise SystemExit(
            "For single-variant mode you must provide --block-size and --key-size, "
            "or use --all."
        )

    validate_variant(cipher_name, args.block_size, args.key_size)

    examples = cipher_entry["extract_single"](
        root_dir=root_dir,
        block_size=args.block_size,
        key_size=args.key_size,
        split=args.split,
    )
    print_variant_summary(cipher_name, args.block_size, args.key_size, examples)

    if not args.summary_only:
        output_path = cipher_entry["write_single"](
            root_dir=root_dir,
            block_size=args.block_size,
            key_size=args.key_size,
            split=args.split,
            output_dir=Path(args.output_dir) if args.output_dir else None,
        )
        print(f"WROTE {output_path}")


if __name__ == "__main__":
    main()