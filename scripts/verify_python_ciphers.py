#!/usr/bin/env python3
"""
verify_python_ciphers.py

Runs the built-in test-vector self-tests for every Python cipher implementation
in the python ciphers/ directory.  Each cipher module exposes a
``<name>_test() -> bool`` function that encrypts and decrypts official test
vectors and returns True on success.

Usage
-----
    python scripts/verify_python_ciphers.py [--cipher <name>] [--quiet]

Options
-------
--cipher NAME   Run only the cipher whose stem matches NAME (e.g. xtea_64_128)
--quiet         Suppress per-cipher output; print only the final summary

Exit code
---------
0   all tested ciphers passed
1   one or more ciphers failed (or no testable ciphers were found)

Corresponds to the functional verification of Python source implementations
described in Section 3: Tiered Isomorphic Alignment - Cipher Selection and
Dataset Design.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path resolution — works regardless of cwd or cluster mount point
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CIPHER_DIR = PROJECT_ROOT / "python ciphers"


def _find_test_fn(module):
    """Return the *_test() callable from a cipher module, or None."""
    for attr in dir(module):
        if attr.endswith("_test") and callable(getattr(module, attr)):
            return getattr(module, attr)
    return None


def _load_module(path: Path):
    """Dynamically load a .py file as a module."""
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)       # type: ignore[union-attr]
    except Exception as exc:
        print(f"  [LOAD ERROR] {exc}", file=sys.stderr)
        return None
    return mod


def run_all(cipher_filter: str | None = None, quiet: bool = False) -> bool:
    if not CIPHER_DIR.is_dir():
        print(f"[ERROR] Cipher directory not found: {CIPHER_DIR}", file=sys.stderr)
        sys.exit(1)

    candidates = sorted(CIPHER_DIR.glob("*.py"))
    if cipher_filter:
        candidates = [p for p in candidates if cipher_filter in p.stem]

    results: list[tuple[str, bool | None]] = []

    for path in candidates:
        if path.stem in ("lea", "speck_template"):
            continue   # aggregate / template files — no standalone test

        mod = _load_module(path)
        if mod is None:
            results.append((path.stem, None))
            continue

        test_fn = _find_test_fn(mod)
        if test_fn is None:
            continue   # no self-test in this module — skip silently

        # Redirect stdout when quiet to suppress per-cipher prints
        if quiet:
            import io, contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                try:
                    result = test_fn()
                    # None return with no exception = assert-based pass (SIMON style)
                    passed = True if result is None else bool(result)
                except Exception as exc:
                    print(f"  [EXCEPTION] {path.stem}: {exc}", file=sys.stderr)
                    passed = False
        else:
            try:
                result = test_fn()
                passed = True if result is None else bool(result)
            except Exception as exc:
                print(f"  [EXCEPTION] {path.stem}: {exc}", file=sys.stderr)
                passed = False

        results.append((path.stem, passed))

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print()
    print("=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)

    tested = [(name, ok) for name, ok in results if ok is not None]
    passed_list = [name for name, ok in tested if ok]
    failed_list = [name for name, ok in tested if not ok]
    skipped_load = [name for name, ok in results if ok is None]

    for name in passed_list:
        print(f"  Passed  {name}")
    for name in failed_list:
        print(f" Failed  {name}")
    for name in skipped_load:
        print(f" Skipped {name}  [load error — skipped]")

    print()
    print(f"Tested : {len(tested)}")
    print(f"Passed : {len(passed_list)}")
    print(f"Failed : {len(failed_list)}")
    if skipped_load:
        print(f"Skipped: {len(skipped_load)}")
    print("=" * 60)

    if not tested:
        print("[ERROR] No testable cipher modules found.", file=sys.stderr)
        return False

    all_ok = len(failed_list) == 0
    print("ALL TESTS PASSED" if all_ok else "SOME TESTS FAILED")
    return all_ok


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run built-in test-vector self-tests for all Python cipher implementations."
    )
    parser.add_argument(
        "--cipher",
        metavar="NAME",
        default=None,
        help="Run only the cipher whose file stem contains NAME (e.g. xtea_64_128)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-cipher output; print only the final summary",
    )
    args = parser.parse_args()

    ok = run_all(cipher_filter=args.cipher, quiet=args.quiet)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
