from __future__ import annotations

import sys

import argparse
import importlib.util
import json
import logging
import shlex
import struct
from dataclasses import dataclass, replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
TWO_DGS_ROOT = ROOT / "2d-gaussian-splatting"

MIPNERF360_OUTDOOR = ("bicycle", "flowers", "garden", "stump", "treehill")
MIPNERF360_INDOOR = ("room", "counter", "kitchen", "bonsai")
DEEP_BLENDING = ("drjohnson", "playroom")
DTU = (
    "scan24",
    "scan37",
    "scan40",
    "scan55",
    "scan63",
    "scan65",
    "scan69",
    "scan83",
    "scan97",
    "scan105",
    "scan106",
    "scan110",
    "scan114",
    "scan118",
    "scan122",
)
TANKS_AND_TEMPLES = (
    "Barn",
    "Caterpillar",
    "Courthouse",
    "Ignatius",
    "Meetingroom",
    "Truck",
    "Church",
)
METHODS = (
    "hexels_v1",
    "hexels_v2",
    "hexels_v3",
    "2dgs_native",
    "2dgs_equal_primitives",
    "2dgs_equal_v2_memory",
    "2dgs_equal_v1_compact_memory",
    "2dgs_equal_v2_compact_memory",
    "2dgs_equal_v3_compact_memory",
)

# Current FP32 resident scene tensors, excluding optimizer state and shared code.
HEXELS_RESIDENT_BYTES_PER_PRIMITIVE = {"v1": 169, "v2": 169, "v3": 313}
HEXELS_ACTIVE_FLOATS_PER_PRIMITIVE = {"v1": 20, "v2": 29, "v3": 65}
HEXELS_COMPACT_BYTES_PER_PRIMITIVE = {"v1": 80, "v2": 116, "v3": 260}
TWO_DGS_BYTES_PER_PRIMITIVE = 232

LOGGER = logging.getLogger("minimum_3d_benchmark")

HEXEL_TRAINING_CONFIG = ROOT / "configs/train.json"
HEXEL_TRAINING_KEYS = frozenset({
    "camera_sampling", "init_scale_mode", "init_knn_scale_multiplier", "spatial_lr_scale", "lr_means3d_final",
})


def hexel_training_defaults(path: Path = HEXEL_TRAINING_CONFIG) -> dict[str, object]:
    """Read the five shared Hexel training settings from their single source, the release train config."""
    defaults = json.loads(path.read_text()).get("hexels")
    if not isinstance(defaults, dict) or defaults.keys() != HEXEL_TRAINING_KEYS:
        raise ValueError(f"{path} must set exactly these shared Hexel training settings under 'hexels': "
                         f"{sorted(HEXEL_TRAINING_KEYS)}")
    return defaults


@dataclass(frozen=True)
class SceneSpec:
    dataset: str
    scene: str
    source: Path
    task: str
    images: str | None
    eval: bool
    resolution: int


def _scene_specs(
    *,
    mipnerf360_root: Path,
    dtu_root: Path,
    deep_blending_root: Path,
    tanks_and_temples_root: Path,
    dtu_nvs: bool = False,
) -> tuple[SceneSpec, ...]:
    specs: list[SceneSpec] = []
    for scene in MIPNERF360_OUTDOOR:
        specs.append(
            SceneSpec("mipnerf360", scene, mipnerf360_root / scene, "nvs", "images_4", True, -1)
        )
    for scene in MIPNERF360_INDOOR:
        specs.append(
            SceneSpec("mipnerf360", scene, mipnerf360_root / scene, "nvs", "images_2", True, -1)
        )
    for scene in DTU:
        specs.append(
            SceneSpec(
                "dtu",
                scene,
                dtu_root / scene,
                "nvs" if dtu_nvs else "geometry",
                None,
                dtu_nvs,
                2,
            )
        )
    for scene in DEEP_BLENDING:
        specs.append(
            SceneSpec("deep_blending", scene, deep_blending_root / scene, "nvs", None, True, -1)
        )
    for scene in TANKS_AND_TEMPLES:
        specs.append(
            SceneSpec(
                "tanks_and_temples",
                scene,
                tanks_and_temples_root / scene,
                "nvs",
                None,
                True,
                -1 if scene == "Church" else 2,
            )
        )
    return tuple(specs)


def read_colmap_point_count(source: Path) -> int:
    """Read the sparse COLMAP point count without loading point records."""
    sparse = source / "sparse/0"
    ply_path = sparse / "points3D.ply"
    if ply_path.is_file():
        with ply_path.open("rb") as handle:
            for raw_line in handle:
                line = raw_line.decode("ascii").strip()
                if line.startswith("element vertex "):
                    return int(line.rsplit(" ", 1)[1])
                if line == "end_header":
                    break
        raise ValueError(f"PLY header has no vertex count: {ply_path}")

    binary_path = sparse / "points3D.bin"
    if binary_path.is_file():
        with binary_path.open("rb") as handle:
            count_bytes = handle.read(8)
        if len(count_bytes) != 8:
            raise ValueError(f"COLMAP points file is truncated: {binary_path}")
        return int(struct.unpack("<Q", count_bytes)[0])

    text_path = sparse / "points3D.txt"
    if text_path.is_file():
        with text_path.open() as handle:
            return sum(1 for line in handle if line.strip() and not line.lstrip().startswith("#"))

    raise FileNotFoundError(f"No sparse COLMAP points3D file under {sparse}")


def image_holdout_split(source: Path, hold: int) -> dict[str, object]:
    image_dir = source / "images"
    if not image_dir.is_dir():
        return {"status": "missing_image_directory", "path": str(image_dir)}
    names = sorted(
        path.stem
        for path in image_dir.iterdir()
        if not path.name.startswith(".") and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    test_names = names[::hold]
    test_set = set(test_names)
    return {
        "total_views": len(names),
        "train_views": len(names) - len(test_names),
        "test_views": len(test_names),
        "test_camera_names": test_names,
        "train_camera_names": [name for name in names if name not in test_set],
    }


def matched_2dgs_cap(
    hexel_primitives: int,
    *,
    hexel_bytes_per_primitive: int = HEXELS_RESIDENT_BYTES_PER_PRIMITIVE["v2"],
    hexel_external_scene_bytes: int = 0,
    two_dgs_external_scene_bytes: int = 0,
) -> tuple[int, int]:
    """Return a 2DGS cap and total byte budget for equal deployable scene state."""
    if hexel_primitives < 1:
        raise ValueError("hexel_primitives must be positive")
    byte_budget = hexel_primitives * hexel_bytes_per_primitive + hexel_external_scene_bytes
    available = byte_budget - two_dgs_external_scene_bytes
    if available < TWO_DGS_BYTES_PER_PRIMITIVE:
        raise ValueError("The scene byte budget cannot hold one 2DGS primitive")
    return available // TWO_DGS_BYTES_PER_PRIMITIVE, byte_budget


def _hexels_dataset_args(scene: SceneSpec) -> list[str]:
    args: list[str] = []
    if scene.images is not None:
        args.extend(["--images", scene.images])
    if scene.eval:
        args.extend(["--eval", "--llffhold", "8"])
    if scene.resolution > 0:
        args.extend(["--resolution", str(scene.resolution)])
    return args


def _two_dgs_dataset_args(scene: SceneSpec) -> list[str]:
    args: list[str] = []
    if scene.images is not None:
        args.extend(["--images", scene.images])
    if scene.eval:
        args.append("--eval")
    if scene.resolution > 0:
        args.extend(["--resolution", str(scene.resolution)])
    if scene.dataset == "tanks_and_temples":
        args.extend(["--depth_ratio", "1.0"])
    return args


def _mesh_args(scene: SceneSpec) -> list[str]:
    """Use the extraction scales in the original 2DGS geometry protocols."""
    large = scene.dataset == "tanks_and_temples" and scene.scene in {"Meetingroom", "Courthouse"}
    return [
        "--voxel_size", "0.006" if large else "0.004",
        "--sdf_trunc", "0.024" if large else "0.016",
        "--depth_trunc", "4.5" if large else "3.0",
        "--num_cluster", "1",
    ]


def _hexels_stage_commands(
    scene: SceneSpec,
    *,
    variant: str,
    output_root: Path,
    python_bin: Path = PYTHON,
    seed: int = 0,
    training_defaults: dict[str, object] | None = None,
) -> tuple[list[str], ...]:
    if variant not in {"v1", "v2", "v3"}:
        raise ValueError(f"Unknown Hexel variant: {variant}")
    view_model = {"v1": "none", "v2": "linear", "v3": "sh3"}[variant]
    output_base = output_root / f"hexels_{variant}" / scene.dataset / scene.scene
    commands: list[list[str]] = []
    defaults = hexel_training_defaults() | (training_defaults or {})
    unknown = defaults.keys() - HEXEL_TRAINING_KEYS
    if unknown:
        raise ValueError(f"Unknown shared Hexel training settings: {sorted(unknown)}")
    shared_args = [
        value
        for name, setting in defaults.items()
        for value in (f"--{name}", str(setting))
    ]
    parent: Path | None = None
    for iteration in (15_000, 20_000, 25_000, 30_000):
        model_path = output_base / f"{iteration // 1000}k"
        command = [
            str(python_bin),
            "-m",
            "hexels.train",
            "--source_path",
            str(scene.source),
            "--model_path",
            str(model_path),
            "--seed",
            str(seed),
            "--iterations",
            str(iteration),
            *shared_args,
            "--color_model",
            "linear",
            "--view_model",
            view_model,
            "--use_view_grads",
            "False" if variant == "v1" else "True",
            "--save_every",
            str(iteration),
            "--render_every",
            str(iteration),
            *_hexels_dataset_args(scene),
        ]
        if variant == "v3":
            command.extend(["--lr_sh_coeffs", "0.0005", "--sh_progressive_interval", "0"])
        if parent is not None:
            command.extend(
                [
                    "--resume_checkpoint",
                    str(parent),
                    "--resume_geometry_lr_scale",
                    "0.5",
                    "--resume_appearance_lr_scale",
                    "0.5",
                ]
            )
        if iteration < 30_000 or scene.task == "geometry":
            command.append("--skip_postprocess")
        commands.append(command)
        parent = model_path / "checkpoints" / f"ckpt_{iteration:06d}.pt"

    if scene.task == "geometry":
        commands.append(
            [
                str(python_bin),
                str(ROOT / "scripts/extract_hexels_dtu_mesh.py"),
                "--checkpoint",
                str(parent),
                "--model_path",
                str(output_base / "30k"),
                *_mesh_args(scene),
                # Explicit, so this staged protocol keeps expected-depth fusion whatever the extractor default.
                "--depth_mode",
                "expected",
            ]
        )
    return tuple(commands)


def _two_dgs_commands(
    scene: SceneSpec,
    *,
    method: str,
    output_root: Path,
    primitive_cap: int | None,
    port: int,
    dtu_official_root: Path | None,
    dtu_dataset_root: Path,
    python_bin: Path = PYTHON,
    seed: int = 0,
) -> tuple[list[str], ...]:
    model_path = output_root / method / scene.dataset / scene.scene
    train = [
        str(python_bin),
        "train.py",
        "-s",
        str(scene.source),
        "-m",
        str(model_path),
        "--iterations",
        "30000",
        "--position_lr_max_steps",
        "30000",
        "--save_iterations",
        "30000",
        "--test_iterations",
        "-1",
        "--seed",
        str(seed),
        "--port",
        str(port),
        "--quiet",
        *_two_dgs_dataset_args(scene),
    ]
    if primitive_cap is not None:
        train.extend(["--max_primitives", str(primitive_cap)])
    if scene.dataset == "dtu":
        train.extend(["--depth_ratio", "1.0", "--lambda_dist", "1000"])

    commands = [train]
    if scene.task == "nvs":
        render = [
            str(python_bin),
            "render.py",
            "--iteration",
            "30000",
            "-s",
            str(scene.source),
            "-m",
            str(model_path),
            "--skip_train",
            "--skip_mesh",
            "--quiet",
            *_two_dgs_dataset_args(scene),
        ]
        if scene.dataset == "dtu":
            render.extend(["--depth_ratio", "1.0"])
            metrics = [
                str(python_bin),
                str(ROOT / "hexels/metrics.py"),
                "--model_path",
                str(model_path),
                "--iteration",
                "30000",
                "--split",
                "test",
                "--render_dir",
                str(model_path / "test/ours_30000"),
            ]
        else:
            metrics = [str(python_bin), "metrics.py", "-m", str(model_path)]
        commands.extend([render, metrics])
    else:
        render = [
            str(python_bin),
            "render.py",
            "--iteration",
            "30000",
            "-s",
            str(scene.source),
            "-m",
            str(model_path),
            "--skip_test",
            "--quiet",
            "--depth_ratio",
            "1.0",
            *_mesh_args(scene),
        ]
        render.append("--skip_train")
        commands.append(render)
        if scene.dataset == "dtu" and dtu_official_root is not None:
            commands.append(
                _dtu_evaluation_command(
                    mesh_path=model_path / "train/ours_30000/fuse_post.ply",
                    scene=scene,
                    output_dir=model_path / "dtu_eval",
                    dtu_official_root=dtu_official_root,
                    dtu_dataset_root=dtu_dataset_root,
                    python_bin=python_bin,
                )
            )
    return tuple(commands)


def _tnt_evaluation_command(
    scene: SceneSpec, model_path: Path, official_root: Path, python_bin: Path,
) -> list[str]:
    """Run the pinned 2DGS T&T alignment, crop and F-score evaluator."""
    return [
        str(python_bin), str(TWO_DGS_ROOT / "scripts/eval_tnt/run.py"),
        "--dataset-dir", str(official_root / scene.scene),
        "--traj-path", str(scene.source / f"{scene.scene}_COLMAP_SfM.log"),
        "--ply-path", str(model_path / "train/ours_30000/fuse_post.ply"),
        "--out-dir", str(model_path / "tnt_eval"),
    ]


def _dtu_evaluation_command(
    *,
    mesh_path: Path,
    scene: SceneSpec,
    output_dir: Path,
    dtu_official_root: Path,
    dtu_dataset_root: Path,
    python_bin: Path = PYTHON,
) -> list[str]:
    return [
        str(python_bin),
        str(TWO_DGS_ROOT / "scripts/eval_dtu/evaluate_single_scene.py"),
        "--input_mesh",
        str(mesh_path),
        "--scan_id",
        scene.scene.removeprefix("scan"),
        "--output_dir",
        str(output_dir),
        "--mask_dir",
        str(dtu_dataset_root),
        "--DTU",
        str(dtu_official_root),
    ]


def build_manifest(
    *,
    mipnerf360_root: Path,
    dtu_root: Path,
    deep_blending_root: Path,
    tanks_and_temples_root: Path,
    output_root: Path,
    dtu_official_root: Path | None,
    dtu_nvs: bool = False,
    python_bin: Path = PYTHON,
    seed: int = 0,
    dataset_defaults: dict[str, dict[str, object]] | None = None,
    hexel_defaults: dict[str, object] | None = None,
) -> dict[str, object]:
    scenes = _scene_specs(
        mipnerf360_root=mipnerf360_root,
        dtu_root=dtu_root,
        deep_blending_root=deep_blending_root,
        tanks_and_temples_root=tanks_and_temples_root,
        dtu_nvs=dtu_nvs,
    )
    dtu_camera_splits = (
        {
            scene.scene: image_holdout_split(scene.source, 8)
            for scene in scenes
            if scene.dataset == "dtu"
        }
        if dtu_nvs
        else {}
    )
    jobs: list[dict[str, object]] = []
    for scene_index, scene in enumerate(scenes):
        settings = (dataset_defaults or {}).get(scene.dataset, {})
        if "scenes" in settings and scene.scene not in settings["scenes"]:
            continue
        scene = replace(scene, resolution=int(settings.get("resolution", scene.resolution)))
        if "task" in settings:
            scene = replace(scene, task=settings["task"], eval=settings["task"] == "nvs")
        try:
            primitive_count = read_colmap_point_count(scene.source)
            data_status = "ready"
            data_error = None
        except (FileNotFoundError, ValueError) as exc:
            primitive_count = None
            data_status = "blocked_missing_or_invalid_data"
            data_error = str(exc)

        for method_index, method in enumerate(METHODS):
            primitive_cap: int | None = None
            byte_budget: int | None = None
            if primitive_count is not None and method == "2dgs_equal_primitives":
                primitive_cap = primitive_count
            elif primitive_count is not None and method == "2dgs_equal_v2_memory":
                primitive_cap, byte_budget = matched_2dgs_cap(primitive_count)
            elif primitive_count is not None and method == "2dgs_equal_v1_compact_memory":
                primitive_cap, byte_budget = matched_2dgs_cap(
                    primitive_count,
                    hexel_bytes_per_primitive=HEXELS_COMPACT_BYTES_PER_PRIMITIVE["v1"],
                )
            elif primitive_count is not None and method == "2dgs_equal_v2_compact_memory":
                primitive_cap, byte_budget = matched_2dgs_cap(
                    primitive_count,
                    hexel_bytes_per_primitive=HEXELS_COMPACT_BYTES_PER_PRIMITIVE["v2"],
                )
            elif primitive_count is not None and method == "2dgs_equal_v3_compact_memory":
                primitive_cap, byte_budget = matched_2dgs_cap(
                    primitive_count,
                    hexel_bytes_per_primitive=HEXELS_COMPACT_BYTES_PER_PRIMITIVE["v3"],
                )

            if primitive_count is None and method in {
                "2dgs_equal_primitives",
                "2dgs_equal_v2_memory",
                "2dgs_equal_v1_compact_memory",
                "2dgs_equal_v2_compact_memory",
                "2dgs_equal_v3_compact_memory",
            }:
                commands = []
            elif method.startswith("hexels_"):
                commands = list(
                    _hexels_stage_commands(
                        scene,
                        variant=method.removeprefix("hexels_"),
                        output_root=output_root,
                        python_bin=python_bin,
                        seed=seed,
                        training_defaults=(hexel_defaults or {}) | settings.get("hexels", {}),
                    )
                )
                if scene.dataset == "dtu" and scene.task == "geometry" and dtu_official_root is not None:
                    model_path = output_root / method / scene.dataset / scene.scene / "30k"
                    commands.append(
                        _dtu_evaluation_command(
                            mesh_path=model_path / "train/ours_30000/fuse_post.ply",
                            scene=scene,
                            output_dir=model_path / "dtu_eval",
                            dtu_official_root=dtu_official_root,
                            dtu_dataset_root=dtu_root,
                            python_bin=python_bin,
                        )
                    )
            else:
                commands = _two_dgs_commands(
                    scene,
                    method=method,
                    output_root=output_root,
                    primitive_cap=primitive_cap,
                    port=6100 + scene_index * 10 + method_index,
                    dtu_official_root=dtu_official_root,
                    dtu_dataset_root=dtu_root,
                    python_bin=python_bin,
                    seed=seed,
                )

            evaluation_root = (
                settings.get("official_root") if scene.dataset == "tanks_and_temples"
                else dtu_official_root
            )
            if scene.dataset == "tanks_and_temples" and scene.task == "geometry" and evaluation_root and commands:
                model_path = output_root / method / scene.dataset / scene.scene
                if method.startswith("hexels_"):
                    model_path = model_path / "30k"
                commands = [*commands, _tnt_evaluation_command(
                    scene, model_path, Path(evaluation_root),
                    Path(settings.get("evaluation_python", python_bin)),
                )]

            jobs.append(
                {
                    "dataset": scene.dataset,
                    "scene": scene.scene,
                    "task": scene.task,
                    "method": method,
                    "status": data_status,
                    "data_error": data_error,
                    "evaluation_status": (
                        "included"
                        if scene.task == "nvs" or evaluation_root is not None
                        else f"blocked_missing_{scene.dataset}_official_root"
                    ),
                    "source": str(scene.source),
                    "initial_hexel_primitives": primitive_count,
                    "two_dgs_primitive_cap": primitive_cap,
                    "deployable_scene_byte_budget": byte_budget,
                    "cwd": str(ROOT if method.startswith("hexels_") else TWO_DGS_ROOT),
                    "commands": list(commands),
                }
            )

    scene_inventory: dict[str, list[str]] = {}
    for job in jobs:
        names = scene_inventory.setdefault(job["dataset"], [])
        if job["scene"] not in names:
            names.append(job["scene"])
    return {
        "protocol": (
            "minimum_3d_benchmark_dtu_nvs_v1"
            if dtu_nvs
            else "minimum_3d_benchmark_v1"
        ),
        "scene_count": sum(len(names) for names in scene_inventory.values()),
        "method_count": len(METHODS),
        "job_count": len(jobs),
        "datasets": scene_inventory,
        "methods": list(METHODS),
        "runtime_preflight": {
            "python": str(python_bin),
            "python_exists": python_bin.is_file(),
            "two_dgs_root": str(TWO_DGS_ROOT),
            "two_dgs_root_exists": TWO_DGS_ROOT.is_dir(),
            "open3d_available": importlib.util.find_spec("open3d") is not None,
            "open3d_required_for": ["dtu_mesh_fusion", "dtu_official_2dgs_render"],
        },
        "memory_accounting": {
            "compact_controls": {
                "hexels_v1": "2dgs_equal_v1_compact_memory",
                "hexels_v2": "2dgs_equal_v2_compact_memory",
                "hexels_v3": "2dgs_equal_v3_compact_memory",
            },
            "conservative_resident_control": "2dgs_equal_v2_memory",
            "budget_definition": "total active deployable learned scene parameter bytes",
            "hexels_resident_bytes_per_primitive": HEXELS_RESIDENT_BYTES_PER_PRIMITIVE,
            "hexels_active_floats_per_primitive": HEXELS_ACTIVE_FLOATS_PER_PRIMITIVE,
            "hexels_compact_bytes_per_primitive": HEXELS_COMPACT_BYTES_PER_PRIMITIVE,
            "two_dgs_bytes_per_primitive": TWO_DGS_BYTES_PER_PRIMITIVE,
            "included_external_scene_state": [
                "learned textures",
                "scene-specific grids",
                "scene-specific decoder or MLP weights",
                "codebooks",
                "other scene-specific tensor metadata",
            ],
            "current_external_scene_bytes": {
                "hexels_v1": 0,
                "hexels_v2": 0,
                "hexels_v3": 0,
                "2dgs": 0,
            },
            "separately_report": [
                "serialized artifact bytes",
                "compressed artifact bytes",
                "persistent render VRAM",
                "peak render VRAM",
                "peak training VRAM",
            ],
        },
        "dtu_image_metrics": {
            "enabled": dtu_nvs,
            "split": "sorted camera index modulo 8",
            "per_scene_camera_splits": dtu_camera_splits,
            "resolution_factor": 2,
            "metrics": ["PSNR", "SSIM", "LPIPS"],
            "background": "unmasked RGB",
            "official_geometry_protocol_reused": False,
        },
        "jobs": jobs,
    }


def _filter_manifest(
    manifest: dict[str, object],
    *,
    datasets: set[str],
    scenes: set[str],
    methods: set[str],
) -> dict[str, object]:
    filtered = dict(manifest)
    jobs = [
        job
        for job in manifest["jobs"]
        if (not datasets or job["dataset"] in datasets)
        and (not scenes or job["scene"] in scenes)
        and (not methods or job["method"] in methods)
    ]
    filtered["jobs"] = jobs
    filtered["job_count"] = len(jobs)
    filtered["methods"] = list(dict.fromkeys(job["method"] for job in jobs))
    filtered["method_count"] = len(filtered["methods"])
    filtered_datasets: dict[str, list[str]] = {}
    for job in jobs:
        scene_names = filtered_datasets.setdefault(job["dataset"], [])
        if job["scene"] not in scene_names:
            scene_names.append(job["scene"])
    filtered["datasets"] = filtered_datasets
    filtered["scene_count"] = sum(len(scene_names) for scene_names in filtered_datasets.values())
    return filtered


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the minimum multi-dataset 3D benchmark.")
    parser.add_argument("--mipnerf360-root", type=Path, default=ROOT / "data/mipnerf360")
    parser.add_argument("--dtu-root", type=Path, default=ROOT / "data/dtu")
    parser.add_argument("--deep-blending-root", type=Path, default=ROOT / "data/deep_blending")
    parser.add_argument(
        "--tanks-and-temples-root",
        type=Path,
        default=ROOT / "data/tanks_and_temples",
    )
    parser.add_argument("--dtu-official-root", type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs/minimum_3d",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "outputs/minimum_3d/manifests/jobs.json",
    )
    parser.add_argument("--datasets", nargs="*", default=[])
    parser.add_argument("--scenes", nargs="*", default=[])
    parser.add_argument("--methods", nargs="*", choices=METHODS, default=[])
    parser.add_argument("--dtu-nvs", action="store_true")
    parser.add_argument("--print-commands", action="store_true")
    args = parser.parse_args()

    dtu_official_root = (
        args.dtu_official_root.resolve() if args.dtu_official_root is not None else None
    )
    manifest = build_manifest(
        mipnerf360_root=args.mipnerf360_root.resolve(),
        dtu_root=args.dtu_root.resolve(),
        deep_blending_root=args.deep_blending_root.resolve(),
        tanks_and_temples_root=args.tanks_and_temples_root.resolve(),
        output_root=args.output_root.resolve(),
        dtu_official_root=dtu_official_root,
        dtu_nvs=args.dtu_nvs,
    )
    manifest = _filter_manifest(
        manifest,
        datasets=set(args.datasets),
        scenes=set(args.scenes),
        methods=set(args.methods),
    )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    LOGGER.info("Wrote %d jobs to %s", manifest["job_count"], args.manifest)

    if args.print_commands:
        for job in manifest["jobs"]:
            print(f"# {job['dataset']}/{job['scene']} {job['method']} [{job['status']}]")
            for command in job["commands"]:
                print(shlex.join(command))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
