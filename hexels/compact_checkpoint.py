from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from tap import Tap


FORMAT_NAME = "hexels.compact_appearance"
FORMAT_VERSION = 1
_LEGACY_FORMAT_NAME = "hexels3dgs.compact_appearance"
METADATA_KEY = "deployment"

# Training keeps slots for every appearance variant. Export omits inactive
# fields, so a training checkpoint's size is not the learned representation size.
_APPEARANCE_SHAPES: dict[str, tuple[int, ...]] = {
    "color_grads": (2, 3),
    "color_quads": (3, 3),
    "view_grads": (3, 3),
    "specular_ks": (3,),
    "specular_log_s": (),
}


@dataclass(frozen=True)
class AppearanceLayout:
    name: str
    view_model: str
    view_basis_dim: int
    active_optional_fields: frozenset[str]


V1_SPATIAL = AppearanceLayout(
    name="v1_spatial",
    view_model="none",
    view_basis_dim=3,
    active_optional_fields=frozenset({"color_grads"}),
)
V2_LINEAR_VIEW = AppearanceLayout(
    name="v2_linear_view",
    view_model="linear",
    view_basis_dim=3,
    active_optional_fields=frozenset({"color_grads", "view_grads"}),
)
V3_SH3 = AppearanceLayout(
    name="v3_sh3",
    view_model="sh3",
    view_basis_dim=15,
    active_optional_fields=frozenset({"color_grads", "view_grads"}),
)


def infer_appearance_layout(config: Mapping[str, object]) -> AppearanceLayout | None:
    if config.get("color_model") != "linear":
        return None
    view_model = config.get("view_model")
    if view_model == "none":
        return V1_SPATIAL
    if view_model == "linear" and config.get("use_view_grads") is True:
        return V2_LINEAR_VIEW
    if view_model == "sh3" and config.get("use_view_grads") is True:
        return V3_SH3
    return None


def _state_tensor(state: Mapping[str, object], name: str) -> torch.Tensor:
    value = state.get(name)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"Checkpoint model state is missing tensor {name!r}")
    return value


def _appearance_shapes(layout: AppearanceLayout) -> dict[str, tuple[int, ...]]:
    return {
        **_APPEARANCE_SHAPES,
        "view_grads": (layout.view_basis_dim, 3),
    }


def _inactive_fields(layout: AppearanceLayout) -> tuple[str, ...]:
    return tuple(
        name for name in _appearance_shapes(layout) if name not in layout.active_optional_fields
    )


def validate_appearance_state(
    state: Mapping[str, object],
    layout: AppearanceLayout,
    *,
    inactive_storage: Literal["full", "compact"],
) -> tuple[int, int]:
    means3d = _state_tensor(state, "means3d")
    if means3d.ndim != 2 or means3d.shape[1] != 3 or means3d.shape[0] < 1:
        raise ValueError(f"means3d must have non-empty shape (P,3), got {tuple(means3d.shape)}")
    num_base = int(means3d.shape[0])

    child_parent_idx = _state_tensor(state, "child_parent_idx")
    if child_parent_idx.ndim != 1:
        raise ValueError(f"child_parent_idx must be 1D, got {tuple(child_parent_idx.shape)}")
    num_children = int(child_parent_idx.numel())

    for name, count, trailing_shape in (
        ("colors", num_base, (3,)),
        ("opacity_logits", num_base, ()),
        ("child_colors", num_children, (3,)),
        ("child_opacity_logits", num_children, ()),
    ):
        value = _state_tensor(state, name)
        expected = (count, *trailing_shape)
        if value.shape != expected:
            raise ValueError(f"Active appearance tensor {name!r} must have shape {expected}, got {tuple(value.shape)}")

    log_softness = _state_tensor(state, "log_softness")
    if log_softness.shape not in {(num_base,), (num_base, 3)}:
        raise ValueError(
            "Active appearance tensor 'log_softness' must have shape "
            f"({num_base},) or ({num_base}, 3), got {tuple(log_softness.shape)}"
        )
    child_log_softness = _state_tensor(state, "child_log_softness")
    expected_child_softness = (num_children, *log_softness.shape[1:])
    if child_log_softness.shape != expected_child_softness:
        raise ValueError(
            "Active appearance tensor 'child_log_softness' must have shape "
            f"{expected_child_softness}, got {tuple(child_log_softness.shape)}"
        )

    inactive_count = num_base if inactive_storage == "full" else 0
    inactive_child_count = num_children if inactive_storage == "full" else 0
    for name, trailing_shape in _appearance_shapes(layout).items():
        active = name in layout.active_optional_fields
        value = _state_tensor(state, name)
        expected_count = num_base if active else inactive_count
        expected = (expected_count, *trailing_shape)
        if value.shape != expected:
            kind = "Active" if active else "Inactive"
            raise ValueError(f"{kind} appearance tensor {name!r} must have shape {expected}, got {tuple(value.shape)}")
        if not active and value.numel() > 0 and torch.count_nonzero(value).item() != 0:
            raise ValueError(f"Inactive appearance tensor {name!r} must be exactly zero")

        child_name = f"child_{name}"
        child_value = _state_tensor(state, child_name)
        expected_child_count = num_children if active else inactive_child_count
        expected_child = (expected_child_count, *trailing_shape)
        if child_value.shape != expected_child:
            kind = "Active" if active else "Inactive"
            raise ValueError(
                f"{kind} appearance tensor {child_name!r} must have shape {expected_child}, "
                f"got {tuple(child_value.shape)}"
            )
        if not active and child_value.numel() > 0 and torch.count_nonzero(child_value).item() != 0:
            raise ValueError(f"Inactive appearance tensor {child_name!r} must be exactly zero")

    return num_base, num_children


def compact_active_layout_state(
    state: Mapping[str, object],
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Remove inactive base rows and return compact-to-original IDs."""
    means3d = _state_tensor(state, "means3d")
    if means3d.ndim != 2 or means3d.shape[1] != 3:
        raise ValueError(f"means3d must have shape (P,3), got {tuple(means3d.shape)}")
    num_base = int(means3d.shape[0])
    active_mask = _state_tensor(state, "active_mask")
    if active_mask.dtype != torch.bool or active_mask.shape != (num_base,):
        raise ValueError(f"active_mask must be bool with shape ({num_base},)")
    current_to_original_ids = torch.nonzero(
        active_mask.detach().cpu(),
        as_tuple=False,
    ).flatten()
    if current_to_original_ids.numel() == 0:
        raise ValueError("Active-layout compaction requires at least one active primitive")

    child_parent_idx = _state_tensor(state, "child_parent_idx")
    if child_parent_idx.dtype != torch.long or child_parent_idx.ndim != 1:
        raise ValueError("child_parent_idx must be a one-dimensional long tensor")
    inverse = torch.full((num_base,), -1, dtype=torch.long)
    inverse[current_to_original_ids] = torch.arange(current_to_original_ids.numel())
    child_parents = child_parent_idx.detach().cpu()
    if child_parents.numel() > 0:
        if int(child_parents.min()) < 0 or int(child_parents.max()) >= num_base:
            raise ValueError("child_parent_idx contains an out-of-range base ID")
        if bool((inverse[child_parents] < 0).any()):
            raise ValueError("Active-layout compaction cannot remove a child parent")

    compact: dict[str, torch.Tensor] = {}
    for name, value in state.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise ValueError("Checkpoint model state must map string names to tensors")
        if name == "child_parent_idx":
            compact[name] = inverse[child_parents].to(device=value.device)
        elif name.startswith("child_"):
            compact[name] = value.detach().clone()
        else:
            if value.ndim == 0 or value.shape[0] != num_base:
                raise ValueError(
                    f"Base tensor {name!r} must have first dimension {num_base}, "
                    f"got {tuple(value.shape)}"
                )
            ids = current_to_original_ids.to(device=value.device)
            compact[name] = value.detach().index_select(0, ids).clone()
    if not bool(compact["active_mask"].all()):
        raise RuntimeError("Compacted active mask is not all true")
    return compact, current_to_original_ids


def compact_appearance_state(
    state: Mapping[str, object],
    layout: AppearanceLayout,
) -> dict[str, object]:
    validate_appearance_state(state, layout, inactive_storage="full")
    compact_state = dict(state)
    for name in _inactive_fields(layout):
        trailing_shape = _appearance_shapes(layout)[name]
        for state_name in (name, f"child_{name}"):
            compact_state[state_name] = _state_tensor(state, state_name).new_empty((0, *trailing_shape))
    validate_appearance_state(compact_state, layout, inactive_storage="compact")
    return compact_state


def expand_compact_appearance_state(
    state: Mapping[str, object],
    reference_state: Mapping[str, object],
    layout: AppearanceLayout,
) -> dict[str, object]:
    compact_counts = validate_appearance_state(state, layout, inactive_storage="compact")
    reference_counts = validate_appearance_state(reference_state, layout, inactive_storage="full")
    if compact_counts != reference_counts:
        raise ValueError(
            "Compact and render model primitive counts differ: "
            f"checkpoint={compact_counts}, render_model={reference_counts}"
        )
    expanded_state = dict(state)
    for name in _inactive_fields(layout):
        for state_name in (name, f"child_{name}"):
            expanded_state[state_name] = torch.zeros_like(_state_tensor(reference_state, state_name))
    return expanded_state


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deployment_metadata(parent_sha256: str, layout: AppearanceLayout) -> dict[str, object]:
    return {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "parent_sha256": parent_sha256,
        "layout": layout.name,
        "color_model": "linear",
        "view_model": layout.view_model,
        "use_view_grads": "view_grads" in layout.active_optional_fields,
    }


def validate_deployment_checkpoint(
    checkpoint: Mapping[str, object],
    layout: AppearanceLayout,
) -> None:
    metadata = checkpoint.get(METADATA_KEY)
    if not isinstance(metadata, Mapping):
        raise ValueError(f"Compact checkpoint is missing {METADATA_KEY!r} metadata")
    expected = deployment_metadata(str(metadata.get("parent_sha256", "")), layout)
    if metadata.get("format") == _LEGACY_FORMAT_NAME:
        expected["format"] = _LEGACY_FORMAT_NAME
    for name in ("format", "version", "layout", "color_model", "view_model", "use_view_grads"):
        actual_value = metadata.get(name)
        expected_value = expected[name]
        if type(actual_value) is not type(expected_value) or actual_value != expected_value:
            raise ValueError(
                f"Compact checkpoint metadata mismatch for {name}: "
                f"got {actual_value!r}, expected {expected_value!r}"
            )
    parent_sha256 = metadata.get("parent_sha256")
    if not isinstance(parent_sha256, str) or len(parent_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in parent_sha256
    ):
        raise ValueError("Compact checkpoint parent_sha256 must be a lowercase SHA-256 digest")
    if checkpoint.get("resume_compatible") is not False:
        raise ValueError("Compact deployment checkpoints must set resume_compatible=false")
    if "optimizer" in checkpoint:
        raise ValueError("Compact deployment checkpoints must not contain optimizer state")
    iteration = checkpoint.get("iteration")
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
        raise ValueError(f"Compact deployment checkpoint has invalid iteration: {iteration!r}")
    active_layout = metadata.get("active_layout")
    if active_layout is not None:
        if not isinstance(active_layout, Mapping):
            raise ValueError("Compact checkpoint active_layout metadata must be a mapping")
        original_num_base = active_layout.get("original_num_base")
        current_num_base = active_layout.get("current_num_base")
        current_to_original_ids = active_layout.get("current_to_original_ids")
        if (
            isinstance(original_num_base, bool)
            or not isinstance(original_num_base, int)
            or isinstance(current_num_base, bool)
            or not isinstance(current_num_base, int)
            or not 0 < current_num_base <= original_num_base
        ):
            raise ValueError("Compact checkpoint active-layout counts are invalid")
        if (
            not isinstance(current_to_original_ids, torch.Tensor)
            or current_to_original_ids.dtype != torch.long
            or current_to_original_ids.shape != (current_num_base,)
            or int(current_to_original_ids.min()) < 0
            or int(current_to_original_ids.max()) >= original_num_base
            or not bool(torch.all(current_to_original_ids[1:] > current_to_original_ids[:-1]))
        ):
            raise ValueError("Compact checkpoint active-layout ID mapping is invalid")
        if active_layout.get("training_resume_supported") is not False:
            raise ValueError("Compact active-layout checkpoints must not claim training resume support")
        model_state = checkpoint.get("model")
        if not isinstance(model_state, Mapping):
            raise ValueError("Compact checkpoint is missing its model state")
        means3d = _state_tensor(model_state, "means3d")
        active_mask = _state_tensor(model_state, "active_mask")
        if means3d.shape[0] != current_num_base or active_mask.shape != (current_num_base,):
            raise ValueError("Compact checkpoint model count differs from active-layout metadata")
        if active_mask.dtype != torch.bool or not bool(active_mask.all()):
            raise ValueError("Compact checkpoint active mask must be all true")


def export_compact_checkpoint(
    checkpoint_path: Path,
    output_path: Path,
    *,
    active_only: bool = False,
) -> dict[str, object]:
    checkpoint_path = checkpoint_path.resolve()
    output_path = output_path.absolute()
    if not checkpoint_path.is_file():
        raise ValueError(f"Checkpoint does not exist: {checkpoint_path}")
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")

    config_path = checkpoint_path.parent.parent / "config.json"
    if not config_path.is_file():
        raise ValueError(f"Checkpoint is missing its adjacent config: {config_path}")
    config = json.loads(config_path.read_text())
    if not isinstance(config, Mapping):
        raise ValueError(f"Config must contain a mapping: {config_path}")
    layout = infer_appearance_layout(config)
    if layout is None:
        raise ValueError("Deployment compaction supports V1, V2, and V3 configurations")

    parent_sha256 = _sha256(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Checkpoint must contain a mapping: {checkpoint_path}")
    iteration = checkpoint.get("iteration")
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
        raise ValueError(f"Checkpoint has invalid iteration: {iteration!r}")
    model_state = checkpoint.get("model")
    if not isinstance(model_state, Mapping):
        raise ValueError("Checkpoint is missing its model state")

    if active_only:
        deployment_state, current_to_original_ids = compact_active_layout_state(model_state)
    else:
        deployment_state = dict(model_state)
        current_to_original_ids = None
    compact_state = compact_appearance_state(deployment_state, layout)
    metadata = deployment_metadata(parent_sha256, layout)
    if current_to_original_ids is not None:
        metadata["active_layout"] = {
            "original_num_base": int(_state_tensor(model_state, "means3d").shape[0]),
            "current_num_base": int(current_to_original_ids.numel()),
            "current_to_original_ids": current_to_original_ids,
            "training_resume_supported": False,
        }
    deployment_checkpoint = {
        "iteration": iteration,
        "model": compact_state,
        "resume_compatible": False,
        METADATA_KEY: metadata,
    }
    validate_deployment_checkpoint(deployment_checkpoint, layout)
    if _sha256(checkpoint_path) != parent_sha256:
        raise RuntimeError(f"Source checkpoint changed during export: {checkpoint_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as handle:
        torch.save(deployment_checkpoint, handle)
    return metadata


class CompactCheckpointOptions(Tap):
    checkpoint: str = ""
    output: str = ""
    active_only: bool = False

    def process_args(self) -> None:
        if not self.checkpoint:
            raise ValueError("--checkpoint is required")
        if not self.output:
            raise ValueError("--output is required")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    opt = CompactCheckpointOptions().parse_args()
    metadata = export_compact_checkpoint(
        Path(opt.checkpoint),
        Path(opt.output),
        active_only=opt.active_only,
    )
    logging.getLogger(__name__).info(
        "Exported %s version %d deployment checkpoint to %s",
        metadata["layout"],
        metadata["version"],
        opt.output,
    )


if __name__ == "__main__":
    main()
