from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scaffold a small open-model batch run: prompts manifest, a server-side "
            "batch launcher, and a staging helper that links finished videos into "
            "an MMVGM dataset directory."
        )
    )
    parser.add_argument("--model-slug", required=True, help="Model slug for metadata and file naming.")
    parser.add_argument(
        "--prompt-file",
        type=Path,
        required=True,
        help="Plain-text prompt file with one prompt per line. Blank lines and # comments are ignored.",
    )
    parser.add_argument(
        "--sample-helper",
        type=Path,
        required=True,
        help="Existing smoke helper to call for each prompt, for example server/open-models/sample_runs/wan_t2v_1_3b.sh.",
    )
    parser.add_argument(
        "--batch-root",
        type=Path,
        required=True,
        help="Directory where prompts.tsv, batch_manifest.json, and helper scripts will be written.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where generated videos should land.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="MMVGM dataset directory to populate from finished outputs.",
    )
    parser.add_argument(
        "--copy-mode",
        choices=["symlink", "copy", "hardlink"],
        default="symlink",
        help="How stage_to_dataset.sh should place finished videos into the dataset directory.",
    )
    parser.add_argument(
        "--filename-prefix",
        help="Optional filename prefix. Defaults to the model slug.",
    )
    return parser.parse_args()


def read_prompts(path: Path) -> list[str]:
    prompts: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        prompt = raw_line.strip()
        if not prompt or prompt.startswith("#"):
            continue
        prompts.append(prompt)
    if not prompts:
        raise SystemExit(f"No prompts found in {path}")
    return prompts


def quote(value: str) -> str:
    return shlex.quote(value)


def render_run_script(
    sample_helper: Path, prompts_tsv: Path, output_dir: Path, log_dir: Path
) -> str:
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            f'SAMPLE_HELPER="{sample_helper}"',
            f'PROMPTS_TSV="{prompts_tsv}"',
            f'OUTPUT_DIR="{output_dir}"',
            f'LOG_DIR="{log_dir}"',
            'GPU_LIST="${GPU_LIST:-0}"',
            'SKIP_EXISTING="${SKIP_EXISTING:-1}"',
            "",
            'mkdir -p "$OUTPUT_DIR" "$LOG_DIR"',
            'IFS="," read -r -a GPUS <<< "$GPU_LIST"',
            '[ "${#GPUS[@]}" -gt 0 ] || { echo "GPU_LIST resolved to zero GPUs." >&2; exit 1; }',
            'JOB_LIMIT="${MAX_JOBS:-${#GPUS[@]}}"',
            "",
            "launch_job() {",
            '  local sample_id="$1"',
            '  local prompt="$2"',
            '  local gpu="$3"',
            '  local output_file="$OUTPUT_DIR/${sample_id}.mp4"',
            '  local log_file="$LOG_DIR/${sample_id}.log"',
            '  if [ "$SKIP_EXISTING" = "1" ] && [ -f "$output_file" ]; then',
            '    echo "Skipping $sample_id because $output_file already exists."',
            "    return 0",
            "  fi",
            "  (",
            '    export CUDA_VISIBLE_DEVICES="$gpu"',
            '    export PROMPT="$prompt"',
            '    export OUTPUT_FILE="$output_file"',
            '    echo "[$(date)] sample_id=$sample_id gpu=$gpu output=$output_file"',
            '    bash "$SAMPLE_HELPER"',
            '  ) >"$log_file" 2>&1 &',
            "}",
            "",
            "job_count=0",
            "gpu_index=0",
            'while IFS=$\'\\t\' read -r sample_id prompt; do',
            '  [ -n "$sample_id" ] || continue',
            '  while [ "$(jobs -pr | wc -l | tr -d \" \")" -ge "$JOB_LIMIT" ]; do',
            "    sleep 5",
            "  done",
            '  gpu="${GPUS[$((gpu_index % ${#GPUS[@]}))]}"',
            '  launch_job "$sample_id" "$prompt" "$gpu"',
            "  job_count=$((job_count + 1))",
            "  gpu_index=$((gpu_index + 1))",
            'done < "$PROMPTS_TSV"',
            "",
            'wait',
            'echo "Completed $job_count batch job(s)."',
            "",
        ]
    ) + "\n"


def render_stage_script(
    prompts_tsv: Path, output_dir: Path, dataset_dir: Path, copy_mode: str
) -> str:
    if copy_mode == "copy":
        stage_line = 'cp -f "$src" "$dest"'
    elif copy_mode == "hardlink":
        stage_line = 'ln -f "$src" "$dest"'
    else:
        stage_line = 'ln -sfn "$src" "$dest"'

    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            f'PROMPTS_TSV="{prompts_tsv}"',
            f'OUTPUT_DIR="{output_dir}"',
            f'DATASET_DIR="{dataset_dir}"',
            'mkdir -p "$DATASET_DIR"',
            "",
            'while IFS=$\'\\t\' read -r sample_id prompt; do',
            '  [ -n "$sample_id" ] || continue',
            '  src="$OUTPUT_DIR/${sample_id}.mp4"',
            '  dest="$DATASET_DIR/${sample_id}.mp4"',
            '  [ -f "$src" ] || { echo "Missing output: $src" >&2; exit 1; }',
            f"  {stage_line}",
            '  echo "Staged $src -> $dest"',
            'done < "$PROMPTS_TSV"',
            "",
        ]
    ) + "\n"


def main() -> None:
    args = parse_args()
    prompts = read_prompts(args.prompt_file)
    args.batch_root.mkdir(parents=True, exist_ok=True)

    filename_prefix = args.filename_prefix or args.model_slug
    prompts_tsv = args.batch_root / "prompts.tsv"
    log_dir = args.batch_root / "logs"

    manifest_entries = []
    prompt_lines = []
    for index, prompt in enumerate(prompts, start=1):
        sample_id = f"{filename_prefix}-{index:03d}"
        output_file = args.output_dir / f"{sample_id}.mp4"
        dataset_file = args.dataset_dir / f"{sample_id}.mp4"
        manifest_entries.append(
            {
                "sample_id": sample_id,
                "prompt": prompt,
                "output_file": str(output_file),
                "dataset_file": str(dataset_file),
            }
        )
        prompt_lines.append(f"{sample_id}\t{prompt}")

    prompts_tsv.write_text("\n".join(prompt_lines) + "\n", encoding="utf-8")
    (args.batch_root / "run_batch.sh").write_text(
        render_run_script(args.sample_helper, prompts_tsv, args.output_dir, log_dir),
        encoding="utf-8",
    )
    (args.batch_root / "run_batch.sh").chmod(0o755)
    (args.batch_root / "stage_to_dataset.sh").write_text(
        render_stage_script(prompts_tsv, args.output_dir, args.dataset_dir, args.copy_mode),
        encoding="utf-8",
    )
    (args.batch_root / "stage_to_dataset.sh").chmod(0o755)
    (args.batch_root / "batch_manifest.json").write_text(
        json.dumps(
            {
                "model_slug": args.model_slug,
                "prompt_file": str(args.prompt_file),
                "sample_helper": str(args.sample_helper),
                "output_dir": str(args.output_dir),
                "dataset_dir": str(args.dataset_dir),
                "copy_mode": args.copy_mode,
                "sample_count": len(manifest_entries),
                "samples": manifest_entries,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (args.batch_root / "README.md").write_text(
        "\n".join(
            [
                f"# {args.model_slug} batch scaffold",
                "",
                f"- Prompt count: {len(manifest_entries)}",
                f"- Sample helper: `{args.sample_helper}`",
                f"- Output dir: `{args.output_dir}`",
                f"- Dataset dir: `{args.dataset_dir}`",
                "",
                "## Generated files",
                "- `prompts.tsv`: stable sample IDs plus prompts",
                "- `batch_manifest.json`: machine-readable metadata",
                "- `run_batch.sh`: launch helper for each prompt, with round-robin GPU dispatch via `GPU_LIST`",
                "- `stage_to_dataset.sh`: place finished outputs into the MMVGM dataset directory",
                "",
                "## Example",
                f"- `GPU_LIST=0,1,2,3 {quote(str(args.batch_root / 'run_batch.sh'))}`",
                f"- `{quote(str(args.batch_root / 'stage_to_dataset.sh'))}`",
                "",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
