"""Export raw or contact-welded colored Hexel/2DGS patches without TSDF or images."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.special import expit

from hexels.contact_mesh import (
    ContactOptions, find_contacts, nondegenerate_patches, primitive_polygons, triangulate_patches, weld_contacts,
)

LOGGER = logging.getLogger(__name__)


def load_primitives(model_path: Path, family: str, iteration: int) -> tuple[dict[str, np.ndarray], Path]:
    """Load learned geometry and view-independent color, without loading camera images."""
    if family == "2dgs":
        checkpoint = model_path / f"point_cloud/iteration_{iteration}/point_cloud.ply"
        data = PlyData.read(checkpoint)["vertex"].data
        fields = lambda names: np.stack([data[name] for name in names], axis=1).astype(np.float64)
        return {
            "centers": fields(["x", "y", "z"]),
            "scales": np.exp(fields(["scale_0", "scale_1"])),
            "rotations": fields([f"rot_{index}" for index in range(4)]),
            "colors": np.clip(0.5 + 0.28209479177387814 * fields([f"f_dc_{index}" for index in range(3)]), 0, 1),
            "opacities": expit(np.asarray(data["opacity"], dtype=np.float64)),
        }, checkpoint
    import torch
    from hexels.model import Hexels3DGeometry, Hexels3DModel, validate_active_mask_state
    from hexels.render import RenderOptions, _checkpoint_view_basis_dim, _prepare_render_model_state

    checkpoint = model_path / f"checkpoints/ckpt_{iteration:06d}.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = payload["model"]
    opt = RenderOptions()
    opt.checkpoint = str(checkpoint)
    opt.model_path = str(model_path)
    opt.process_args()
    geometry = Hexels3DGeometry(state["means3d"], state["log_scales"].exp(), state["rotation_quats"])
    model = Hexels3DModel(geometry, state["colors"], 0.1, opt.softness,
                          softness_mode=opt.softness_mode,
                          view_basis_dim=_checkpoint_view_basis_dim(state, opt.view_model))
    model.allocate_children(int(state.get("child_parent_idx", torch.empty(0)).numel()))
    state = _prepare_render_model_state(payload, model, opt)
    validate_active_mask_state(state, model.num_base)
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys or set(incompatible.missing_keys) - {"active_mask", "child_rotvec", "child_layer"}:
        raise ValueError(f"Unsupported Hexel state: {incompatible}")
    if model.num_children and not model.child_layer.numel():
        model.child_layer = torch.ones_like(model.child_parent_idx)
    with torch.no_grad():
        packed = model.pack_for_rasterizer(
            child_offset_frac=opt.child_offset_frac, child_scale_ratio=opt.child_scale_ratio,
            child_rotation_delta=opt.child_rotation_delta,
        )
    result = {target: packed[source].detach().numpy().astype(np.float64)
              for target, source in {"centers": "means3d", "scales": "scales",
                                     "rotations": "rotations", "colors": "colors",
                                     "opacities": "opacities"}.items()}
    result["scales"] *= opt.scale_modifier
    result["colors"] = np.clip(result["colors"], 0, 1)
    return result, checkpoint


def write_colored_mesh(
    path: Path, vertices: np.ndarray, colors: np.ndarray, faces: np.ndarray, edges: np.ndarray,
) -> None:
    """Write binary PLY faces and a separate colored boundary-edge PLY."""
    vertex_data = np.empty(len(vertices), dtype=[(name, "f8") for name in ("x", "y", "z")]
                           + [(name, "u1") for name in ("red", "green", "blue")])
    for axis, name in enumerate(("x", "y", "z")):
        vertex_data[name] = vertices[:, axis]
    for axis, name in enumerate(("red", "green", "blue")):
        vertex_data[name] = np.round(np.clip(colors[:, axis], 0, 1) * 255).astype(np.uint8)
    face_data = np.empty(len(faces), dtype=[("vertex_indices", "i4", (3,))])
    face_data["vertex_indices"] = faces
    vertex_element = PlyElement.describe(vertex_data, "vertex")
    PlyData([vertex_element, PlyElement.describe(face_data, "face")], text=False).write(path)
    edge_data = np.empty(len(edges), dtype=[("vertex1", "i4"), ("vertex2", "i4")]
                         + [(name, "u1") for name in ("red", "green", "blue")])
    edge_data["vertex1"], edge_data["vertex2"] = edges.T
    for axis, name in enumerate(("red", "green", "blue")):
        edge_data[name] = np.round(np.clip(colors[edges, axis].mean(axis=1), 0, 1) * 255).astype(np.uint8)
    PlyData([vertex_element, PlyElement.describe(edge_data, "edge")], text=False).write(path.with_name("wireframe.ply"))


def extract(args: argparse.Namespace) -> dict[str, object]:
    """Measure load, polygon/contact work and serialization separately."""
    if not 0 <= args.opacity_threshold < 1 or args.gaussian_sides < 6:
        raise ValueError("Invalid opacity threshold or polygon resolution")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    primitives, checkpoint = load_primitives(args.model_path, args.family, args.iteration)
    loaded = time.perf_counter()
    total_count = len(primitives["centers"])
    retained = primitives["opacities"] >= args.opacity_threshold
    if not retained.any():
        raise ValueError("Opacity filter removed every primitive")
    primitive_ids = np.flatnonzero(retained)
    primitives = {name: values[retained] for name, values in primitives.items()}
    corners, normals = primitive_polygons(
        primitives["centers"], primitives["scales"], primitives["rotations"],
        family=args.family, gaussian_sides=args.gaussian_sides,
    )
    valid = nondegenerate_patches(corners, primitives["centers"])
    degenerate_count = int((~valid).sum())
    if degenerate_count:
        LOGGER.warning("Excluding %d numerically zero-area patches", degenerate_count)
        corners, normals = corners[valid], normals[valid]
        primitive_ids = primitive_ids[valid]
        primitives = {name: values[valid] for name, values in primitives.items()}
    if not len(corners):
        raise ValueError("No positive-area patches remain")
    colors = primitives["colors"]
    options = ContactOptions()
    diagnostics = {}
    contacts = np.empty((0, 2), dtype=np.int64)
    vertices = corners.reshape(-1, 3)
    vertex_colors = np.repeat(colors, corners.shape[1], axis=0)
    corner_ids = np.arange(len(vertices)).reshape(corners.shape[:2])
    if args.mode == "contact":
        contacts, endpoints, diagnostics = find_contacts(corners, normals, options, workers=args.workers)
        vertices, vertex_colors, corner_ids, weld_stats = weld_contacts(
            corners, colors, endpoints, max_shift_fraction=options.max_vertex_shift_fraction,
        )
        diagnostics.update(weld_stats)
        first, second = contacts.T
        sides = corners.shape[1]
        first_edge = np.sort(np.stack([corner_ids.ravel()[first],
                                       corner_ids[first // sides, (first + 1) % sides]], axis=1), axis=1)
        second_edge = np.sort(np.stack([corner_ids.ravel()[second],
                                        corner_ids[second // sides, (second + 1) % sides]], axis=1), axis=1)
        accepted = np.all(first_edge == second_edge, axis=1)
        diagnostics["welded_contacts"] = int(accepted.sum())
        diagnostics["isolated_primitive_fraction"] = float(
            1 - len(np.unique(contacts[accepted] // sides)) / len(corners))
    vertices, vertex_colors, faces, edges = triangulate_patches(
        vertices, vertex_colors, corner_ids, primitives["centers"], colors,
    )
    areas = np.linalg.norm(np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                                    vertices[faces[:, 2]] - vertices[faces[:, 0]]), axis=1)
    if not np.isfinite(vertices).all() or np.any(areas <= 0):
        raise ValueError("Export produced nonfinite vertices or degenerate triangles")
    constructed = time.perf_counter()
    write_colored_mesh(output / "mesh.ply", vertices, vertex_colors, faces, edges)
    if args.mode == "contact":
        np.savez_compressed(output / "contacts.npz", side_pairs=contacts, corner_ids=corner_ids,
                            primitive_ids=primitive_ids)
    finished = time.perf_counter()
    result = {
        "family": args.family, "mode": args.mode, "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "primitives": total_count, "retained_primitives": len(primitive_ids),
        "opacity_filtered_primitives": int((~retained).sum()), "zero_area_primitives": degenerate_count,
        "opacity_threshold": args.opacity_threshold, "gaussian_sides": args.gaussian_sides,
        "contour": "hexel_sdf_zero" if args.family == "hexels" else "gaussian_half_peak",
        "color": "view_independent_constant_rgb", "contacts": asdict(options),
        "vertices": len(vertices), "triangles": len(faces), "edges": len(edges),
        "load_seconds": loaded - started, "construction_seconds": constructed - loaded,
        "write_seconds": finished - constructed, "extraction_seconds": finished - loaded,
        "total_seconds": finished - started, "file_bytes": (output / "mesh.ply").stat().st_size,
        "workers": args.workers, "diagnostics": diagnostics,
    }
    (output / "extraction.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    LOGGER.info("Exported %s", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("hexels", "2dgs"), required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("raw", "contact"), required=True)
    parser.add_argument("--opacity-threshold", type=float, default=0.1)
    parser.add_argument("--gaussian-sides", type=int, default=12)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    extract(parser.parse_args())


if __name__ == "__main__":
    main()
