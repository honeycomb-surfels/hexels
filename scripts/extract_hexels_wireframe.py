from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from plyfile import PlyData, PlyElement

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hexels.wireframe import HexelWireframe, extract_hexel_wireframe


LOGGER = logging.getLogger("hexels_wireframe")
ColorMode = Literal["none", "base", "spatial"]


def write_wireframe_ply(path: Path, wireframe: HexelWireframe) -> None:
    """Write colored cells and indexed sides with source Hexel IDs."""
    positions = wireframe.vertices.detach().cpu().numpy().astype(np.float32, copy=False)
    indices = wireframe.edges.detach().cpu().numpy().astype(np.int32, copy=False)
    primitive_ids = (
        wireframe.primitive_ids.detach().cpu().numpy().astype(np.int32, copy=False)
    )
    side_ids = wireframe.side_ids.detach().cpu().numpy().astype(np.uint8, copy=False)
    faces = wireframe.faces.detach().cpu().numpy().astype(np.int32, copy=False)
    colors = None
    if wireframe.vertex_colors is not None:
        colors = (
            wireframe.vertex_colors.detach()
            .clamp(0.0, 1.0)
            .mul(255)
            .round()
            .byte()
            .cpu()
            .numpy()
        )

    vertex_dtype = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    if colors is not None:
        vertex_dtype.extend([("red", "u1"), ("green", "u1"), ("blue", "u1")])
    vertices = np.empty(len(positions), dtype=vertex_dtype)
    vertices["x"], vertices["y"], vertices["z"] = positions.T
    if colors is not None:
        vertices["red"], vertices["green"], vertices["blue"] = colors.T
    edge_dtype = [
        ("vertex1", "i4"),
        ("vertex2", "i4"),
        ("primitive_id", "i4"),
        ("side_id", "u1"),
    ]
    if colors is not None:
        edge_dtype.extend([("red", "u1"), ("green", "u1"), ("blue", "u1")])
    edges = np.empty(
        len(indices),
        dtype=edge_dtype,
    )
    edges["vertex1"], edges["vertex2"] = indices.T
    edges["primitive_id"] = primitive_ids
    edges["side_id"] = side_ids
    if colors is not None:
        edge_colors = np.rint(colors[indices].mean(axis=1)).astype(np.uint8)
        edges["red"], edges["green"], edges["blue"] = edge_colors.T

    face_colors = (
        None
        if colors is None
        else np.rint(colors[faces].mean(axis=1)).astype(np.uint8)
    )
    face_dtype = [("vertex_indices", "O"), ("primitive_id", "i4")]
    if face_colors is not None:
        face_dtype.extend([("red", "u1"), ("green", "u1"), ("blue", "u1")])
    face_elements = np.empty(len(faces), dtype=face_dtype)
    face_elements["vertex_indices"] = [face for face in faces]
    face_elements["primitive_id"] = primitive_ids[faces[:, 0]]
    if face_colors is not None:
        face_elements["red"], face_elements["green"], face_elements["blue"] = (
            face_colors.T
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData(
        [
            PlyElement.describe(vertices, "vertex"),
            PlyElement.describe(edges, "edge"),
            PlyElement.describe(face_elements, "face"),
        ],
        text=False,
    ).write(path)


@torch.no_grad()
def extract_checkpoint_wireframe(
    checkpoint_path: Path,
    output_path: Path,
    *,
    scale_modifier: float,
    min_opacity: float,
    include_inactive: bool,
    device: torch.device,
    color_mode: ColorMode = "none",
) -> HexelWireframe:
    if not 0.0 <= min_opacity <= 1.0:
        raise ValueError("min_opacity must be between 0 and 1")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state = checkpoint.get("model")
    if not isinstance(state, Mapping):
        raise ValueError("Checkpoint is missing its model state")
    child_parent_idx = state.get("child_parent_idx")
    if isinstance(child_parent_idx, torch.Tensor) and child_parent_idx.numel() > 0:
        raise ValueError("Wireframe export currently supports base Hexels only")

    means3d = state["means3d"].to(device)
    log_scales = state["log_scales"].to(device)
    rotations = state["rotation_quats"].to(device)
    colors = None if color_mode == "none" else state["colors"].to(device)
    color_grads = state["color_grads"].to(device) if color_mode == "spatial" else None
    active_mask = torch.ones(len(means3d), device=device, dtype=torch.bool)
    if not include_inactive and "active_mask" in state:
        active_mask &= state["active_mask"].to(device)
    if min_opacity > 0.0:
        active_mask &= torch.sigmoid(state["opacity_logits"].to(device)) >= min_opacity
    if not bool(active_mask.any()):
        raise ValueError("No Hexels satisfy the export filters")

    wireframe = extract_hexel_wireframe(
        means3d,
        log_scales,
        rotations,
        scale_modifier=scale_modifier,
        active_mask=active_mask,
        colors=colors,
        color_grads=color_grads,
    )
    write_wireframe_ply(output_path, wireframe)
    return wireframe


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export analytic Hexel perimeters without mesh extraction."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scale-modifier", type=float, default=1.0)
    parser.add_argument("--min-opacity", type=float, default=0.0)
    parser.add_argument("--include-inactive", action="store_true")
    parser.add_argument(
        "--color-mode",
        choices=("none", "base", "spatial"),
        default="spatial",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    wireframe = extract_checkpoint_wireframe(
        args.checkpoint,
        args.output,
        scale_modifier=args.scale_modifier,
        min_opacity=args.min_opacity,
        include_inactive=args.include_inactive,
        device=torch.device(args.device),
        color_mode=args.color_mode,
    )
    LOGGER.info(
        "Wrote %d Hexels, %d vertices, and %d line segments to %s",
        len(wireframe.plane_normals),
        len(wireframe.vertices),
        len(wireframe.edges),
        args.output,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
