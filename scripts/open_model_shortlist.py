from __future__ import annotations

import argparse
import json
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "open_video_models.json"
DEFAULT_DATASET_ROOT = "datasets/open-video-models"


@dataclass(frozen=True)
class ModelEntry:
    slug: str
    name: str
    family: str
    tasks: List[str]
    weights_license: str
    min_vram_gb: Optional[float]
    max_quality_vram_gb: Optional[float]
    parameter_count_b: Optional[float]
    quality_tier: str
    source_tracing_default: bool
    source_tracing_candidate: bool
    mmvgm_priority: int
    repo_url: str
    weights_url: str
    notes: str

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ModelEntry":
        return cls(**payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Curate modern open-source video generation models and emit "
            "MMVGM source-tracing command scaffolding."
        )
    )
    parser.add_argument(
        "--task",
        default="source-tracing",
        choices=["source-tracing", "any", "t2v", "i2v", "v2v", "flf2v", "editing"],
        help=(
            "Model capability to filter by. The default uses a curated "
            "source-tracing shortlist instead of the paper-era nine-model setup."
        ),
    )
    parser.add_argument(
        "--selected",
        nargs="+",
        help="Explicit model slugs to include. Overrides the default shortlist.",
    )
    parser.add_argument(
        "--include-optional",
        action="store_true",
        help="When used with --task source-tracing, include budget and optional models too.",
    )
    parser.add_argument(
        "--license",
        dest="licenses",
        action="append",
        default=[],
        help="Filter by weights license label. Can be repeated, e.g. --license apache-2.0.",
    )
    parser.add_argument(
        "--max-vram-gb",
        type=float,
        help="Only keep models with a known minimum VRAM at or below this threshold.",
    )
    parser.add_argument(
        "--sort",
        default="mmvgm",
        choices=["mmvgm", "vram", "footprint", "name", "license"],
        help="Sort order for the output table or JSON payload.",
    )
    parser.add_argument(
        "--format",
        default="table",
        choices=["table", "json", "mmvgm-json", "mmvgm-shell"],
        help="Output format.",
    )
    parser.add_argument(
        "--backbone",
        default="xclip",
        choices=["i3d", "mae", "xclip"],
        help="MMVGM backbone to use when emitting source-tracing commands.",
    )
    parser.add_argument(
        "--dataset-root",
        default=DEFAULT_DATASET_ROOT,
        help="Root directory that contains one folder per generator slug.",
    )
    parser.add_argument(
        "--materialize-root",
        type=Path,
        help=(
            "Create empty dataset directories plus helper files under this root. "
            "If --dataset-root is left at its default, this path also becomes the emitted dataset root."
        ),
    )
    parser.add_argument(
        "--write-json",
        type=Path,
        help="Write the JSON payload to a file. Valid with --format json or --format mmvgm-json.",
    )
    return parser.parse_args()


def load_models() -> List[ModelEntry]:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return [ModelEntry.from_dict(entry) for entry in payload]


def resolve_selected(models: Sequence[ModelEntry], slugs: Sequence[str]) -> List[ModelEntry]:
    by_slug = {model.slug: model for model in models}
    resolved: List[ModelEntry] = []
    missing: List[str] = []
    for slug in slugs:
        model = by_slug.get(slug)
        if model is None:
            missing.append(slug)
            continue
        resolved.append(model)
    if missing:
        raise SystemExit("Unknown model slug(s): " + ", ".join(missing))
    return resolved


def select_models(models: Sequence[ModelEntry], args: argparse.Namespace) -> List[ModelEntry]:
    if args.selected:
        chosen = resolve_selected(models, args.selected)
    elif args.task == "source-tracing":
        if args.include_optional:
            chosen = [model for model in models if model.source_tracing_candidate]
        else:
            chosen = [model for model in models if model.source_tracing_default]
    elif args.task == "any":
        chosen = list(models)
    else:
        chosen = [model for model in models if args.task in model.tasks]

    if args.licenses:
        allowed = set(args.licenses)
        chosen = [model for model in chosen if model.weights_license in allowed]

    if args.max_vram_gb is not None:
        chosen = [
            model
            for model in chosen
            if model.min_vram_gb is not None and model.min_vram_gb <= args.max_vram_gb
        ]

    if not chosen:
        raise SystemExit("No models matched the current filters.")
    return sort_models(chosen, args.sort)


def sort_models(models: Sequence[ModelEntry], sort_key: str) -> List[ModelEntry]:
    def key(model: ModelEntry) -> Any:
        if sort_key == "vram":
            return (model.min_vram_gb is None, model.min_vram_gb or sys.maxsize, model.slug)
        if sort_key == "footprint":
            return (
                model.parameter_count_b is None,
                model.parameter_count_b or sys.maxsize,
                model.slug,
            )
        if sort_key == "license":
            return (model.weights_license, model.slug)
        if sort_key == "name":
            return (model.name.lower(), model.slug)
        return (model.mmvgm_priority, model.slug)

    return sorted(models, key=key)


def format_number(value: Optional[float], suffix: str) -> str:
    if value is None:
        return "unknown"
    if abs(value - round(value)) < 0.01:
        return f"{int(round(value))}{suffix}"
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{text}{suffix}"


def render_vram(model: ModelEntry) -> str:
    if model.min_vram_gb is None:
        return "unknown"
    if model.max_quality_vram_gb and model.max_quality_vram_gb > model.min_vram_gb:
        return f"{format_number(model.min_vram_gb, 'GB')}-{format_number(model.max_quality_vram_gb, 'GB')}"
    return format_number(model.min_vram_gb, "GB")


def render_table(models: Sequence[ModelEntry]) -> str:
    rows: List[Dict[str, str]] = []
    for model in models:
        rows.append(
            {
                "slug": model.slug,
                "tasks": ",".join(model.tasks),
                "license": model.weights_license,
                "min_vram": render_vram(model),
                "params": format_number(model.parameter_count_b, "B"),
                "tier": model.quality_tier,
                "default": "yes" if model.source_tracing_default else "no",
            }
        )

    columns = ["slug", "tasks", "license", "min_vram", "params", "tier", "default"]
    widths = {
        column: max(len(column), max(len(row[column]) for row in rows))
        for column in columns
    }

    def make_row(row: Dict[str, str]) -> str:
        return "  ".join(row[column].ljust(widths[column]) for column in columns)

    header = make_row({column: column for column in columns})
    separator = "  ".join("-" * widths[column] for column in columns)
    body = [make_row(row) for row in rows]
    return "\n".join([header, separator] + body)


def shell_join(args: Sequence[str]) -> str:
    return " \\\n  ".join(shlex.quote(part) for part in args)


def build_training_command(backbone: str, dataset_paths: Sequence[str]) -> List[str]:
    args = [
        "python",
        f"detection_and_source_tracing/{backbone}.py",
        "--train",
        "True",
        "--task",
        "source_tracing",
        "--epoch",
        "20",
        "--learning_rate",
        "1e-5",
    ]
    if backbone == "i3d":
        args.extend(["--pre_trained_I3D_model", "models/rgb_imagenet.pt"])
    args.extend(
        [
            "--fake_videos_path",
            *dataset_paths,
            "--label_number",
            str(len(dataset_paths)),
            "--save_checkpoint_dir",
            f"checkpoints/{backbone}_open_models_source_tracing.pt",
        ]
    )
    return args


def build_eval_command(backbone: str, dataset_paths: Sequence[str]) -> List[str]:
    return [
        "python",
        f"detection_and_source_tracing/{backbone}.py",
        "--train",
        "False",
        "--task",
        "source_tracing",
        "--load_pre_trained_model_state",
        "path/to/checkpoint.pt",
        "--fake_videos_path",
        *dataset_paths,
        "--label_number",
        str(len(dataset_paths)),
    ]


def build_mmvgm_manifest(
    models: Sequence[ModelEntry], dataset_root: str, backbone: str
) -> Dict[str, Any]:
    root = Path(dataset_root)
    dataset_paths = [str(root / model.slug) for model in models]
    label_map = []
    for label, (model, dataset_path) in enumerate(zip(models, dataset_paths)):
        label_map.append(
            {
                "label": label,
                "slug": model.slug,
                "name": model.name,
                "tasks": model.tasks,
                "weights_license": model.weights_license,
                "dataset_path": dataset_path,
                "repo_url": model.repo_url,
                "weights_url": model.weights_url,
            }
        )

    train_command = build_training_command(backbone, dataset_paths)
    eval_command = build_eval_command(backbone, dataset_paths)
    return {
        "dataset_root": str(root),
        "backbone": backbone,
        "label_number": len(label_map),
        "fake_videos_path": dataset_paths,
        "label_map": label_map,
        "train_command": train_command,
        "train_command_shell": shell_join(train_command),
        "eval_command": eval_command,
        "eval_command_shell": shell_join(eval_command),
    }


def build_json_payload(models: Sequence[ModelEntry]) -> Dict[str, Any]:
    return {
        "config_path": str(CONFIG_PATH),
        "model_count": len(models),
        "models": [model.__dict__ for model in models],
    }


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def materialize_root(root: Path, manifest: Dict[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)

    for entry in manifest["label_map"]:
        Path(entry["dataset_path"]).mkdir(parents=True, exist_ok=True)

    write_json(root / "manifest.json", manifest)
    write_json(root / "label_map.json", {"label_map": manifest["label_map"]})

    train_script = root / f"train_{manifest['backbone']}.sh"
    train_script.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n\n"
        + manifest["train_command_shell"]
        + "\n",
        encoding="utf-8",
    )
    train_script.chmod(0o755)

    eval_script = root / f"eval_{manifest['backbone']}.sh"
    eval_script.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n\n"
        + manifest["eval_command_shell"]
        + "\n",
        encoding="utf-8",
    )
    eval_script.chmod(0o755)

    lines = [
        "# MMVGM Open-Model Staging",
        "",
        "This directory contains an empty dataset skeleton for MMVGM source tracing.",
        "Populate each model directory with generated `.mp4` files before training or evaluation.",
        "No checkpoints or generator weights are downloaded by this script.",
        "",
        "## Labels",
    ]
    for entry in manifest["label_map"]:
        lines.append(
            f"- {entry['label']}: {entry['slug']} ({entry['weights_license']}) -> {entry['dataset_path']}"
        )
    lines.extend(
        [
            "",
            "## Helper Files",
            f"- `manifest.json`: full MMVGM manifest for this layout",
            f"- `label_map.json`: compact label mapping",
            f"- `train_{manifest['backbone']}.sh`: training command skeleton",
            f"- `eval_{manifest['backbone']}.sh`: evaluation command skeleton",
        ]
    )
    (root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_mmvgm_shell(manifest: Dict[str, Any]) -> str:
    label_lines = [
        f"# {entry['label']}: {entry['slug']} -> {entry['dataset_path']}"
        for entry in manifest["label_map"]
    ]
    sections = [
        "# MMVGM source-tracing label map",
        *label_lines,
        "",
        "# Training command",
        manifest["train_command_shell"],
        "",
        "# Evaluation command",
        manifest["eval_command_shell"],
    ]
    return "\n".join(sections)


def main() -> None:
    args = parse_args()
    if args.write_json and args.format not in {"json", "mmvgm-json"}:
        raise SystemExit("--write-json is only valid with --format json or --format mmvgm-json.")

    models = select_models(load_models(), args)

    dataset_root = args.dataset_root
    if args.materialize_root and dataset_root == DEFAULT_DATASET_ROOT:
        dataset_root = str(args.materialize_root)

    if args.format == "table":
        if args.materialize_root:
            manifest = build_mmvgm_manifest(models, dataset_root, args.backbone)
            materialize_root(args.materialize_root, manifest)
        print(render_table(models))
        return

    if args.format == "json":
        payload = build_json_payload(models)
        if args.write_json:
            write_json(args.write_json, payload)
        print(json.dumps(payload, indent=2))
        return

    manifest = build_mmvgm_manifest(models, dataset_root, args.backbone)
    if args.materialize_root:
        materialize_root(args.materialize_root, manifest)
    if args.format == "mmvgm-json":
        if args.write_json:
            write_json(args.write_json, manifest)
        print(json.dumps(manifest, indent=2))
        return

    print(render_mmvgm_shell(manifest))


if __name__ == "__main__":
    main()
