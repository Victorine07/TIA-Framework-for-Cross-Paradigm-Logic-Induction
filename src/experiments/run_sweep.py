#!/usr/bin/env python3
"""
src/experiments/run_sweep.py

This driver does not interpret results -- it runs each entry sequentially,
logs success/failure/timing, and writes a sweep manifest pointing at each
run's own output files for later aggregation. It does not parallelize
runs (cluster GPU jobs should typically run one sweep per allocation).

Typical usage:
    python src/experiments/run_sweep.py --config configs/sweep_example.json
    python src/experiments/run_sweep.py --config configs/sweep_example.json --dry-run
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]

ZERO_SHOT_SCRIPT = PROJECT_ROOT / "src" / "experiments" / "04_run_zero_shot_baseline.py"
FINETUNED_SCRIPT = PROJECT_ROOT / "src" / "experiments" / "04_evaluate_finetuned.py"

VALID_MODES = {"zero_shot", "few_shot", "finetuned"}


def _flag_name(key: str) -> str:
    return "--" + key.replace("_", "-")


def _extra_args_to_cli(extra_args: Dict[str, Any]) -> List[str]:
    cli: List[str] = []
    for key, value in (extra_args or {}).items():
        flag = _flag_name(key)
        if isinstance(value, bool):
            if value:
                cli.append(flag)
            # False is omitted entirely -- these are all store_true flags.
        else:
            cli.extend([flag, str(value)])
    return cli


def _few_shot_cli(run_spec: Dict[str, Any]) -> List[str]:
    cli: List[str] = []
    if "few_shot_k" in run_spec and int(run_spec["few_shot_k"]) > 0:
        cli += ["--few-shot-k", str(run_spec["few_shot_k"])]
        if "few_shot_policy" in run_spec:
            cli += ["--few-shot-policy", run_spec["few_shot_policy"]]
        if "few_shot_source" in run_spec:
            cli += ["--few-shot-source", run_spec["few_shot_source"]]
        if "few_shot_seed" in run_spec:
            cli += ["--few-shot-seed", str(run_spec["few_shot_seed"])]
    return cli


def build_command(run_spec: Dict[str, Any]) -> List[str]:
    name = run_spec.get("name", "?")
    mode = run_spec.get("mode")
    if mode not in VALID_MODES:
        raise ValueError(f"Run '{name}': unknown mode {mode!r}. Choose from {sorted(VALID_MODES)}.")

    dataset = run_spec.get("dataset")
    if not dataset:
        raise ValueError(f"Run '{name}': 'dataset' is required.")

    metadata_strategy = run_spec.get("metadata_strategy", "none")
    extra_args = run_spec.get("extra_args", {})

    if mode in ("zero_shot", "few_shot"):
        model = run_spec.get("model")
        if not model:
            raise ValueError(f"Run '{name}': 'model' is required for mode={mode}.")
        cmd = [
            sys.executable, str(ZERO_SHOT_SCRIPT),
            "--dataset", dataset,
            "--model", model,
            "--metadata-strategy", metadata_strategy,
        ]
        if mode == "few_shot":
            if not run_spec.get("few_shot_k") or int(run_spec["few_shot_k"]) <= 0:
                raise ValueError(f"Run '{name}': mode=few_shot requires few_shot_k > 0.")
            cmd += _few_shot_cli(run_spec)
        cmd += _extra_args_to_cli(extra_args)
        return cmd

    # mode == "finetuned"
    base_model = run_spec.get("base_model")
    adapter_path = run_spec.get("adapter_path")
    if not base_model or not adapter_path:
        raise ValueError(f"Run '{name}': mode=finetuned requires base_model and adapter_path.")
    cmd = [
        sys.executable, str(FINETUNED_SCRIPT),
        "--dataset", dataset,
        "--base-model", base_model,
        "--adapter-path", adapter_path,
        "--metadata-strategy", metadata_strategy,
    ]
    cmd += _few_shot_cli(run_spec)
    cmd += _extra_args_to_cli(extra_args)
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a sequence of evaluation experiments from a JSON sweep config.")
    parser.add_argument("--config", type=str, required=True, help="Path to sweep config JSON")
    parser.add_argument("--dry-run", action="store_true", help="Print/record commands without executing them")
    parser.add_argument(
        "--manifest-dir",
        type=str,
        default=str(PROJECT_ROOT / "results" / "sweeps"),
        help="Directory to write the sweep manifest to",
    )
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Sweep config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    runs = config.get("runs", [])
    if not runs:
        raise ValueError(f"Sweep config {config_path} has no 'runs' entries.")

    tag = config.get("tag", "sweep")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 80)
    print(f"EXPERIMENT SWEEP: {tag}")
    print("=" * 80)
    print(f"Config:    {config_path}")
    print(f"Runs:      {len(runs)}")
    print(f"Dry run:   {args.dry_run}")
    print("=" * 80)

    results: List[Dict[str, Any]] = []

    for i, run_spec in enumerate(runs, 1):
        name = run_spec.get("name", f"run_{i}")
        print("-" * 80)
        print(f"[{i}/{len(runs)}] {name} (mode={run_spec.get('mode')})")

        try:
            cmd = build_command(run_spec)
        except ValueError as e:
            print(f"  SKIPPED (invalid spec): {e}")
            results.append({"name": name, "status": "invalid_spec", "error": str(e)})
            continue

        print(f"  Command: {shlex.join(cmd)}")

        if args.dry_run:
            results.append({"name": name, "status": "dry_run", "command": cmd})
            continue

        start = time.time()
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        elapsed = time.time() - start

        status = "success" if proc.returncode == 0 else "failed"
        print(f"  {status.upper()} (return code {proc.returncode}, {elapsed:.1f}s)")

        results.append(
            {
                "name": name,
                "status": status,
                "return_code": proc.returncode,
                "elapsed_sec": elapsed,
                "command": cmd,
            }
        )

    manifest_dir = Path(args.manifest_dir).expanduser().resolve()
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"sweep_{tag}_{timestamp}.json"

    succeeded = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] in ("failed", "invalid_spec"))

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "tag": tag,
                "timestamp": timestamp,
                "config_path": str(config_path),
                "dry_run": args.dry_run,
                "total_runs": len(runs),
                "succeeded": succeeded,
                "failed": failed,
                "results": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("=" * 80)
    print("SWEEP DONE")
    print(f"Succeeded: {succeeded}/{len(runs)}")
    print(f"Failed:    {failed}/{len(runs)}")
    print(f"Manifest:  {manifest_path}")
    print("=" * 80)

    if failed and not args.dry_run:
        sys.exit(1)


if __name__ == "__main__":
    main()
