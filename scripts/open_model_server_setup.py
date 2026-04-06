from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "open_model_codebases.json"
DEFAULT_PLAN_ROOT = Path("server/open-models")
DEFAULT_CODE_ROOT = Path("/u/nkp2mr/open-video-models")
DEFAULT_WEIGHTS_ROOT = Path("/bigtemp/nkp2mr/shared-benchmarks/open-video-model-weights")
DEFAULT_OUTPUT_ROOT = Path("/bigtemp/nkp2mr/shared-benchmarks/open-video-model-samples")
DEFAULT_HF_HOME = Path("/bigtemp/nkp2mr/huggingface-shared")
DEFAULT_DATASET_ROOT = Path("/bigtemp/nkp2mr/shared-benchmarks/mmvgm-open-video-models")


@dataclass(frozen=True)
class SampleScript:
    filename: str
    title: str
    commands: List[str]

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SampleScript":
        return cls(**payload)


@dataclass(frozen=True)
class CodebaseEntry:
    slug: str
    name: str
    repo_dir: str
    clone_url: str
    repo_url: str
    setup_reference: str
    python_version: str
    venv_dir: str
    recommended: bool
    models_supported: List[str]
    install_steps: List[str]
    notes: List[str]
    download_steps: List[str]
    sample_scripts: List[SampleScript]

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "CodebaseEntry":
        payload = dict(payload)
        payload["sample_scripts"] = [SampleScript.from_dict(item) for item in payload["sample_scripts"]]
        return cls(**payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize a server-side playbook for modern open video model repos, "
            "and optionally clone the upstream codebases without downloading weights."
        )
    )
    parser.add_argument("--plan-root", type=Path, default=DEFAULT_PLAN_ROOT)
    parser.add_argument("--code-root", type=Path, default=DEFAULT_CODE_ROOT)
    parser.add_argument("--weights-root", type=Path, default=DEFAULT_WEIGHTS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--hf-home", type=Path, default=DEFAULT_HF_HOME)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--clone-repos",
        action="store_true",
        help="Clone or fetch the upstream repositories under --code-root.",
    )
    parser.add_argument(
        "--recommended-only",
        action="store_true",
        help="Only clone the recommended codebases when used with --clone-repos.",
    )
    return parser.parse_args()


def load_codebases() -> List[CodebaseEntry]:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return [CodebaseEntry.from_dict(item) for item in payload]


def shell_header(args: argparse.Namespace) -> List[str]:
    return [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f'PLAN_ROOT="${{PLAN_ROOT:-{args.plan_root}}}"',
        'MMVGM_ROOT="${MMVGM_ROOT:-$(cd "$PLAN_ROOT/../.." && pwd)}"',
        f'CODE_ROOT="${{CODE_ROOT:-{args.code_root}}}"',
        f'WEIGHTS_ROOT="${{WEIGHTS_ROOT:-{args.weights_root}}}"',
        f'OUTPUT_ROOT="${{OUTPUT_ROOT:-{args.output_root}}}"',
        f'HF_HOME="${{HF_HOME:-{args.hf_home}}}"',
        'LOCAL_STAGE_ROOT="${LOCAL_STAGE_ROOT:-/tmp/${USER:-$(whoami)}/mmvgm-stage}"',
        'ENABLE_LOCAL_STAGE="${ENABLE_LOCAL_STAGE:-1}"',
        f'DATASET_ROOT="${{DATASET_ROOT:-{args.dataset_root}}}"',
        "export HF_HOME",
        'if [ -n "${HF_TOKEN_PATH:-}" ]; then',
        "  export HF_TOKEN_PATH",
        "fi",
        'mkdir -p "$PLAN_ROOT" "$CODE_ROOT" "$WEIGHTS_ROOT" "$OUTPUT_ROOT"',
        "",
        "ensure_stage_root() {",
        '  if [ "$ENABLE_LOCAL_STAGE" != "1" ]; then',
        "    return 1",
        "  fi",
        '  command -v rsync >/dev/null 2>&1 || { echo "ENABLE_LOCAL_STAGE=1 requires rsync." >&2; exit 1; }',
        '  mkdir -p "$LOCAL_STAGE_ROOT"',
        "}",
        "",
        "require_stage_space_mb() {",
        '  local need_mb="$1"',
        '  local free_mb',
        '  free_mb=$(df -Pm "$LOCAL_STAGE_ROOT" | awk \'NR==2 {print $4}\')',
        '  if [ -z "$free_mb" ] || [ "$free_mb" -lt "$need_mb" ]; then',
        '    echo "Local stage root $LOCAL_STAGE_ROOT only has ${free_mb:-0}MB free; need about ${need_mb}MB. Set ENABLE_LOCAL_STAGE=0 to run from shared storage or choose a roomier host." >&2',
        "    exit 1",
        "  fi",
        "}",
        "",
        "stage_dir() {",
        '  local src="$1"',
        '  local dest="$2"',
        '  if [ "$ENABLE_LOCAL_STAGE" != "1" ]; then',
        '    printf "%s\\n" "$src"',
        "    return 0",
        "  fi",
        "  ensure_stage_root",
        '  require_stage_space_mb "$(du -sm "$src" | awk \'{print $1}\')"',
        '  mkdir -p "$(dirname "$dest")"',
        '  rsync -a --delete --exclude ".cache" "$src"/ "$dest"/',
        '  printf "%s\\n" "$dest"',
        "}",
        "",
        "stage_file() {",
        '  local src="$1"',
        '  local dest="$2"',
        '  if [ "$ENABLE_LOCAL_STAGE" != "1" ]; then',
        '    printf "%s\\n" "$src"',
        "    return 0",
        "  fi",
        "  ensure_stage_root",
        '  require_stage_space_mb "$(du -sm "$src" | awk \'{print $1}\')"',
        '  mkdir -p "$(dirname "$dest")"',
        '  rsync -a "$src" "$dest"',
        '  printf "%s\\n" "$dest"',
        "}",
        "",
    ]


def write_text(path: Path, content: str, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    if executable:
        path.chmod(0o755)


def render_clone_script(args: argparse.Namespace, codebases: Sequence[CodebaseEntry]) -> str:
    lines = shell_header(args)
    lines.extend(
        [
            "clone_repo() {",
            "  local url=\"$1\"",
            "  local target=\"$2\"",
            "  if [ ! -d \"$target/.git\" ]; then",
            "    git clone --filter=blob:none \"$url\" \"$target\"",
            "  else",
            "    git -C \"$target\" remote set-url origin \"$url\"",
            "    git -C \"$target\" fetch --prune origin",
            "  fi",
            "}",
            "",
        ]
    )
    for entry in codebases:
        lines.append(f'clone_repo "{entry.clone_url}" "$CODE_ROOT/{entry.repo_dir}"')
    lines.append("")
    return "\n".join(lines)


def render_download_script(args: argparse.Namespace, codebases: Sequence[CodebaseEntry]) -> str:
    lines = shell_header(args)
    lines.append('echo "Weights are large. Review this file before running it."')
    lines.append("")
    for entry in codebases:
        lines.append(f"# {entry.name}")
        for command in entry.download_steps:
            lines.append(command)
        lines.append("")
    return "\n".join(lines)


def render_sample_script(args: argparse.Namespace, entry: CodebaseEntry, sample: SampleScript) -> str:
    lines = shell_header(args)
    lines.append(f'echo "Running {sample.title}"')
    lines.append("")
    lines.extend(
        [
            f'if [ ! -d "$CODE_ROOT/{entry.repo_dir}/{entry.venv_dir}" ]; then',
            f'  echo "Missing virtual environment: $CODE_ROOT/{entry.repo_dir}/{entry.venv_dir}" >&2',
            '  echo "Bootstrap the upstream repo environment first, then rerun this script." >&2',
            "  exit 1",
            "fi",
            f'source "$CODE_ROOT/{entry.repo_dir}/{entry.venv_dir}/bin/activate"',
            "",
        ]
    )
    lines.extend(sample.commands)
    lines.append("")
    return "\n".join(lines)


def render_checklist(args: argparse.Namespace, entry: CodebaseEntry) -> str:
    lines = [
        f"# {entry.name}",
        "",
        f"- Recommended first wave: {'yes' if entry.recommended else 'no'}",
        f"- Repo: {entry.repo_url}",
        f"- Clone path: {args.code_root / entry.repo_dir}",
        f"- Setup reference: {entry.setup_reference}",
        f"- Python: {entry.python_version}",
        f"- Expected venv: {args.code_root / entry.repo_dir / entry.venv_dir}",
        f"- MMVGM model slugs: {', '.join(entry.models_supported)}",
        "",
        "## Install",
    ]
    for step in entry.install_steps:
        lines.append(f"- `{step}`")
    lines.extend(["", "## Notes"])
    for note in entry.notes:
        lines.append(f"- {note}")
    lines.extend(["", "## Weight download"])
    for step in entry.download_steps:
        lines.append(f"- `{step}`")
    if entry.sample_scripts:
        lines.extend(["", "## Smoke tests"])
        for sample in entry.sample_scripts:
            lines.append(
                f"- `{args.plan_root / 'sample_runs' / sample.filename}`: {sample.title}"
            )
    return "\n".join(lines) + "\n"


def render_readme(args: argparse.Namespace, codebases: Sequence[CodebaseEntry]) -> str:
    recommended = [entry for entry in codebases if entry.recommended]
    lines = [
        "# Open Video Model Server Playbook",
        "",
        "This directory is generated by `scripts/open_model_server_setup.py`.",
        "It stages code-only upstream repos, model-family checklists, and smoke-test helpers for the MMVGM shortlist.",
        "",
        "## Paths",
        f"- Code root: `{args.code_root}`",
        f"- Weights root: `{args.weights_root}`",
        f"- Output root: `{args.output_root}`",
        f"- Shared HF cache: `{args.hf_home}`",
        f"- MMVGM dataset root: `{args.dataset_root}`",
        "",
        "## Recommended first-run stack",
    ]
    for entry in recommended:
        lines.append(f"- {entry.name}")
    lines.extend(
        [
            "",
            "## Generated files",
            "- `clone_all_repos.sh`: code-only clones or fetches for all configured upstream repos",
            "- `download_recommended_weights.sh`: optional heavyweight downloads for the recommended first-wave models",
            "- `checklists/*.md`: per-codebase install and runtime notes",
            "- `sample_runs/*.sh`: smoke-test scripts for the recommended first-wave models",
            "",
            "## Next steps",
            "1. Run `clone_all_repos.sh` to stage the upstream codebases under the code root.",
            "2. Review `download_recommended_weights.sh` before downloading any checkpoint.",
            "3. Use `sample_runs/*.sh` only after the corresponding weights exist.",
            "4. Copy generated `.mp4` files into the MMVGM dataset skeleton under the dataset root.",
        ]
    )
    return "\n".join(lines) + "\n"


def clone_repo(entry: CodebaseEntry, code_root: Path) -> None:
    target = code_root / entry.repo_dir
    if not target.exists():
        subprocess.run(
            ["git", "clone", "--filter=blob:none", entry.clone_url, str(target)],
            check=True,
        )
        return
    if not (target / ".git").exists():
        raise RuntimeError(f"{target} exists but is not a git repository")
    subprocess.run(["git", "-C", str(target), "remote", "set-url", "origin", entry.clone_url], check=True)
    subprocess.run(["git", "-C", str(target), "fetch", "--prune", "origin"], check=True)


def materialize(args: argparse.Namespace, codebases: Sequence[CodebaseEntry]) -> None:
    args.plan_root.mkdir(parents=True, exist_ok=True)
    args.code_root.mkdir(parents=True, exist_ok=True)
    args.weights_root.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)

    write_text(args.plan_root / "README.md", render_readme(args, codebases))
    write_text(args.plan_root / "clone_all_repos.sh", render_clone_script(args, codebases), executable=True)
    write_text(
        args.plan_root / "download_recommended_weights.sh",
        render_download_script(args, [entry for entry in codebases if entry.recommended]),
        executable=True,
    )

    for entry in codebases:
        write_text(args.plan_root / "checklists" / f"{entry.slug}.md", render_checklist(args, entry))
        for sample in entry.sample_scripts:
            write_text(
                args.plan_root / "sample_runs" / sample.filename,
                render_sample_script(args, entry, sample),
                executable=True,
            )

    summary_payload = {
        "plan_root": str(args.plan_root),
        "code_root": str(args.code_root),
        "weights_root": str(args.weights_root),
        "output_root": str(args.output_root),
        "hf_home": str(args.hf_home),
        "dataset_root": str(args.dataset_root),
        "codebases": [
            {
                "slug": entry.slug,
                "name": entry.name,
                "repo_dir": entry.repo_dir,
                "clone_url": entry.clone_url,
                "recommended": entry.recommended,
                "models_supported": entry.models_supported,
            }
            for entry in codebases
        ],
    }
    write_text(args.plan_root / "summary.json", json.dumps(summary_payload, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    codebases = load_codebases()
    materialize(args, codebases)

    if args.clone_repos:
        clone_targets = codebases
        if args.recommended_only:
            clone_targets = [entry for entry in codebases if entry.recommended]
        for entry in clone_targets:
            clone_repo(entry, args.code_root)


if __name__ == "__main__":
    main()
