from __future__ import annotations

import argparse
from pathlib import Path


IMPORT_OLD = "from .attention import flash_attention"
IMPORT_NEW = "from .attention import attention"
CALL_OLD = "flash_attention("
CALL_NEW = "attention("


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Patch the Wan2.1 upstream repo to route flash-attention call sites "
            "through the existing attention() wrapper. This keeps DPLab-style "
            "hosts usable when flash-attn cannot be built."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        required=True,
        help="Path to the cloned Wan2.1 repository root.",
    )
    parser.add_argument(
        "--backup-suffix",
        default=".codex-bak",
        help="Suffix for the one-time backup file. Set empty string to skip backups.",
    )
    return parser.parse_args()


def patch_model(model_path: Path, backup_suffix: str) -> str:
    original = model_path.read_text(encoding="utf-8")

    already_patched = IMPORT_NEW in original and CALL_OLD not in original
    if already_patched:
        return "already-patched"

    if IMPORT_OLD not in original:
        raise SystemExit(f"Expected import line not found in {model_path}")

    import_count = original.count(IMPORT_OLD)
    call_count = original.count(CALL_OLD)
    if import_count != 1 or call_count == 0:
        raise SystemExit(
            f"Unexpected Wan model.py layout in {model_path}: "
            f"import_count={import_count}, call_count={call_count}"
        )

    patched = original.replace(IMPORT_OLD, IMPORT_NEW)
    patched = patched.replace(CALL_OLD, CALL_NEW)

    if patched == original:
        return "already-patched"

    if backup_suffix:
        backup_path = model_path.with_name(model_path.name + backup_suffix)
        if not backup_path.exists():
            backup_path.write_text(original, encoding="utf-8")

    model_path.write_text(patched, encoding="utf-8")
    return f"patched {call_count} call site(s)"


def main() -> None:
    args = parse_args()
    model_path = args.repo_root / "wan" / "modules" / "model.py"
    if not model_path.exists():
        raise SystemExit(f"Wan model.py not found: {model_path}")
    result = patch_model(model_path, args.backup_suffix)
    print(result)


if __name__ == "__main__":
    main()
