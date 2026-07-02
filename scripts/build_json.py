#!/usr/bin/env python3
"""
Build a targets JSON for scripts/test_targets.py.

Output format (matches _TARGET_SPECS):
    {
      "PARP_dataset_4r6e_8GJ_A_401": ["PARP_dataset", "4r6e_8GJ_A_401"],
      ...
    }

Files are read from:
    <targets_dir>/<protein>/Test/
        <basename>.sdf
        <basename>_pocket.pdb   (or <pdbid>.pdb for raw-PDB layout)

Usage (interactive cluster session)
-----------------------------------
    python scripts/build_targets_json.py \\
        --data_glob "/data/stat-cadd/stat0548/PRISM/data/*_dataset" \\
        --targets_dir /data/stat-cadd/stat0548/PRISM/data \\
        --dry_run

    python scripts/build_targets_json.py \\
        --data_glob "/data/stat-cadd/stat0548/PRISM/data/*_dataset" \\
        --targets_dir /data/stat-cadd/stat0548/PRISM/data \\
        --output /data/stat-cadd/stat0548/PRISM/data/targets.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TEST_DIR_NAMES = ("Test", "test")


def resolve_dataset_dirs(data_glob: str) -> list[Path]:
    if any(ch in data_glob for ch in "*?[]"):
        matches = sorted(Path("/").glob(data_glob.lstrip("/")))
    else:
        p = Path(data_glob)
        matches = [p] if p.is_dir() else []

    seen: set[Path] = set()
    dirs: list[Path] = []
    for m in matches:
        m = m.resolve()
        if m.is_dir() and m not in seen:
            seen.add(m)
            dirs.append(m)
    return dirs


def find_test_dir(dataset_dir: Path) -> tuple[Path, str] | None:
    for name in TEST_DIR_NAMES:
        test_dir = dataset_dir / name
        if test_dir.is_dir():
            return test_dir, name
    return None


def find_pdb_for_sdf(test_dir: Path, sdf_path: Path) -> Path | None:
    """Match SDF to PDB (same rules as scripts/test_crossdocked.py)."""
    stem = sdf_path.stem
    pdb_token = stem.split("_", 1)[0]

    for name in (f"{stem}_pocket.pdb", f"{stem}_pocket_only.pdb"):
        candidate = test_dir / name
        if candidate.exists():
            return candidate

    matches = [p for p in test_dir.glob("*.pdb") if p.stem.lower() == pdb_token.lower()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        print(
            f"[WARN] Multiple PDBs match '{pdb_token}' for {sdf_path.name} in {test_dir}",
            file=sys.stderr,
        )
        return sorted(matches, key=lambda p: p.name)[0]
    return None


def make_target_key(protein: str, basename: str, key_style: str) -> str:
    if key_style == "basename":
        return basename
    if key_style == "pdb":
        return f"{protein}_{basename.split('_', 1)[0].upper()}"
    return f"{protein}_{basename}"


def collect_specs(
    dataset_dirs: list[Path],
    key_style: str,
) -> tuple[dict[str, list[str]], dict[str, dict[str, Path]]]:
    specs: dict[str, list[str]] = {}
    paths: dict[str, dict[str, Path]] = {}

    for dataset_dir in dataset_dirs:
        protein = dataset_dir.name
        found = find_test_dir(dataset_dir)
        if found is None:
            print(f"[WARN] No Test/ folder in {dataset_dir}", file=sys.stderr)
            continue

        test_dir, _ = found

        for sdf in sorted(test_dir.glob("*.sdf")):
            basename = sdf.stem
            pdb = find_pdb_for_sdf(test_dir, sdf)
            if pdb is None:
                print(f"[WARN] No PDB for {sdf.name} in {test_dir}", file=sys.stderr)
                continue

            key = make_target_key(protein, basename, key_style)
            if key in specs:
                print(
                    f"[WARN] Duplicate key '{key}' — keeping first occurrence",
                    file=sys.stderr,
                )
                continue

            specs[key] = [protein, basename]
            paths[key] = {"pocket": pdb.resolve(), "ligand": sdf.resolve()}

    return specs, paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build test_targets.py target specs JSON from dataset Test folders.",
    )
    parser.add_argument(
        "--data_glob",
        required=True,
        help=(
            "Glob for dataset directories, e.g. "
            "'/data/stat-cadd/stat0548/PRISM/data/*_dataset'"
        ),
    )
    parser.add_argument(
        "--targets_dir",
        type=Path,
        required=True,
        help="Root passed to test_targets --targets_dir (for reference in output)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: print to stdout)",
    )
    parser.add_argument(
        "--key_style",
        choices=("protein_basename", "basename", "pdb"),
        default="protein_basename",
        help="Target key style (default: PARP_dataset_4r6e_8GJ_A_401)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print summary only; do not write JSON",
    )
    args = parser.parse_args()

    dataset_dirs = resolve_dataset_dirs(args.data_glob)
    if not dataset_dirs:
        print(f"[ERROR] No directories matched: {args.data_glob}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(dataset_dirs)} dataset folder(s):")
    for d in dataset_dirs:
        found = find_test_dir(d)
        if found:
            test_dir, name = found
            n_sdf = len(list(test_dir.glob("*.sdf")))
            print(f"  {d}  ({name}/, sdf={n_sdf})")
        else:
            print(f"  {d}  (Test/=NOT FOUND)")

    specs, paths = collect_specs(dataset_dirs, key_style=args.key_style)
    if not specs:
        print("[ERROR] No target specs found.", file=sys.stderr)
        sys.exit(1)

    print(f"\nCollected {len(specs)} target(s):")
    for key in sorted(specs):
        protein, basename = specs[key]
        pocket = paths[key]["pocket"]
        ligand = paths[key]["ligand"]
        pocket_ok = pocket.exists()
        ligand_ok = ligand.exists()
        status = "OK" if pocket_ok and ligand_ok else "MISSING"
        print(f"  [{status}] {key}")
        print(f"    protein:  {protein}")
        print(f"    basename: {basename}")
        print(f"    pocket:   {pocket}")
        print(f"    ligand:   {ligand}")

    if args.dry_run:
        print("\n[dry_run] No file written.")
        return

    payload = json.dumps(specs, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
        print(f"\nWrote {args.output}")
    else:
        print()
        print(payload)


if __name__ == "__main__":
    main()
