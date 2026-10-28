from __future__ import annotations

import hashlib
import json
import logging
import random
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from hexels.mcmc import init_relocation_buffers
from hexels.metrics import _evaluation_paths, _iteration_from_render_dir, _paired_png_names
from hexels.model import Hexels3DGeometry, Hexels3DModel
from hexels.options import Hexels3DOptions
from hexels.render import (
    RenderOptions,
    _box_downsample,
    _checkpoint_view_basis_dim,
    _render_output_name,
    _render_views,
    _scaled_camera,
    _select_split_cameras,
    _supersample_output_dir,
)
from hexels.renderer import Hexels3DRenderer, _eval_linear_view_residual, _eval_sh3_residual
from hexels.trainer import (
    _RESUME_CONFIG_FIELDS,
    _DensifyShadowEvent,
    Hexels3DTrainer,
    _balanced_densify_shadow_groups,
    _compare_densify_shadow_renders,
    _coverage_weighted_alpha_loss,
    _knn_initial_scales,
    _json_sha256,
    _quat_to_normal_z,
    _quat_to_tangent_xy,
    _residual_mass_or_density_mask,
    _sh_degree_for_iteration,
    _should_render_progress,
    _select_resume_train_camera_manifest,
    _split_resume_train_cameras,
)


def test_replacement_rows_move_and_keep_optimizer_registration() -> None:
    trainer = _make_densify_trainer(num_primitives=2, cap_max=0)
    trainer.opt.densify_opacity_split = False
    original = trainer.model.means3d.detach().clone()
    opacity = trainer.model.opacity_logits.detach()[0].clone()
    trainer._bisplit_primitives(torch.tensor([0]), iteration=1000)
    assert trainer.model.num_base == 3
    assert not torch.equal(trainer.model.means3d[0], trainer.model.means3d[2])
    assert torch.equal(trainer.model.means3d[1], original[1])
    torch.testing.assert_close(trainer.model.opacity_logits[[0, 2]], opacity.expand(2))
    positions = trainer.model.means3d.detach().clone()
    for group in trainer.optimizer.param_groups:
        assert group["params"][0] is getattr(trainer.model, group["name"])
    trainer.optimizer.zero_grad(set_to_none=True)
    trainer.model.means3d[[0, 2]].sum().backward()
    trainer.optimizer.step()
    assert torch.all(trainer.model.means3d[[0, 2]] != positions[[0, 2]])
    assert torch.equal(trainer.model.means3d[1], positions[1])


def test_density_projected_support_gate_is_not_a_population_cap() -> None:
    trainer = _make_densify_trainer(num_primitives=2, cap_max=0)
    trainer.opt.densify_min_projected_radius = 4.0
    trainer.densify_max_radii2d.copy_(torch.tensor([2.0, 6.0]))
    trainer._densify_and_prune_hexels_safe(size_threshold=0, iteration=1000)
    assert trainer.model.num_base == 3
    torch.testing.assert_close(trainer.model.means3d[2], trainer.model.means3d[1])


def _make_model(num_primitives: int = 2, view_basis_dim: int = 3) -> Hexels3DModel:
    geometry = Hexels3DGeometry(
        means3d=torch.zeros((num_primitives, 3)),
        scales=torch.ones((num_primitives, 2)),
        rotations=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(num_primitives, 1),
    )
    return Hexels3DModel(
        geometry=geometry,
        colors_init=torch.zeros((num_primitives, 3)),
        init_opacity=0.1,
        init_softness=10.0,
        view_basis_dim=view_basis_dim,
    )


def _make_densify_trainer(*, num_primitives: int, cap_max: int) -> Hexels3DTrainer:
    trainer = Hexels3DTrainer.__new__(Hexels3DTrainer)
    trainer.model = _make_model(num_primitives=num_primitives)
    with torch.no_grad():
        trainer.model.means3d[:, 0] = torch.arange(num_primitives, dtype=torch.float32)
    trainer.device = torch.device("cpu")
    trainer.scene_extent = 1.0
    trainer.spatial_lr_scale = 1.0
    trainer.opt = SimpleNamespace(
        photometric_loss="l1",
        densify_grad_threshold=0.1,
        densify_min_denom=1,
        densify_max_add=0,
        densify_cap_max=cap_max,
        densify_action="clone",
        densify_residual_topk_views=0,
        densify_residual_persistent_filter=False,
        densify_attributed_weak_power=0,
        densify_covariance_children_per_parent=2,
        densify_covariance_min_views=8,
        densify_covariance_min_anisotropy=0.15,
        densify_covariance_min_axis_coherence=0.50,
        densify_covariance_min_side_balance=0.30,
        densify_covariance_min_delta=0.10,
        densify_joint_coplanar_steps=0,
        densify_shadow_steps=0,
        densify_clone_jitter=0.0,
        densify_clone_init_opacity=0.0,
        densify_opacity_split=True,
        opacity_cull=0.0,
        percent_dense=0.01,
        scale_modifier=1.0,
        enable_children=False,
        lr_means3d=0.00016,
        lr_scales=0.005,
        lr_rotations=0.001,
        lr_colors=0.01,
        lr_color_grads=0.005,
        lr_view_grads=0.005,
        lr_opacity=0.1,
        lr_softness=0.01,
    )
    trainer.optimize_geometry = True
    trainer.optimize_softness = True
    trainer.optimize_color_grads = True
    trainer.optimize_color_quads = False
    trainer.optimize_view_grads = True
    trainer.optimize_specular = False
    trainer.view_model = "linear"
    trainer.model.color_quads.requires_grad_(False)
    trainer.model.specular_ks.requires_grad_(False)
    trainer.model.specular_log_s.requires_grad_(False)
    trainer.enable_densify = True
    trainer.densify_grad_accum = torch.arange(1, num_primitives + 1, dtype=torch.float32)[:, None]
    trainer.densify_denom = torch.ones((num_primitives, 1))
    trainer.densify_max_radii2d = torch.zeros(num_primitives)
    trainer.densify_local_grad_abs_accum = None
    trainer.densify_local_grad_signed_accum = None
    trainer.densify_residual_mass_accum = None
    trainer.densify_centroid_mass_accum = None
    trainer.densify_direction_vote_accum = None
    trainer.densify_moment_vote_count = None
    trainer.densify_mv_edge_top_scores = None
    trainer.densify_mv_edge_top_centroids = None
    trainer.densify_mv_edge_vote_counts = None
    trainer.densify_attributed_edge_mass = None
    trainer.densify_attributed_edge_centroid_mass = None
    trainer.densify_attributed_edge_vote_counts = None
    trainer.densify_covariance_coverage = None
    trainer.densify_covariance_energy = None
    trainer.densify_covariance_robust_weight = None
    trainer.densify_covariance_first_moment = None
    trainer.densify_covariance_second_moment = None
    trainer.densify_covariance_double_angle = None
    trainer.densify_covariance_side_mass = None
    trainer.densify_covariance_material_view_count = None
    trainer.densify_covariance_axis_view_count = None
    trainer.densify_moment_camera_names = set()
    trainer.densify_contrib_accum = None
    trainer.densify_contrib_camera_names = set()
    trainer.densify_birth_iteration = None
    trainer.densify_has_spawned_child = torch.zeros(num_primitives, dtype=torch.bool)
    trainer.densify_joint_coplanar_event = None
    trainer.densify_joint_coplanar_last_completed = None
    trainer.train_cameras = [SimpleNamespace(image_name="cam0")]
    trainer.contrib_ema = None
    trainer.optimizer = trainer._build_optimizer()
    trainer.logger = logging.getLogger("test_densify")
    return trainer


def _enable_residual_topk_densification(
    trainer: Hexels3DTrainer,
    *,
    topk: int = 3,
    camera_names: tuple[str, ...] = ("cam0", "cam1", "cam2"),
) -> None:
    population = trainer.model.num_base
    trainer.opt.densify_action = "edge_child"
    trainer.opt.densify_direction_signal = "residual_moment"
    trainer.opt.densify_residual_topk_views = topk
    trainer.opt.densify_min_denom = topk
    trainer.opt.densify_moment_interval = 1
    trainer.opt.densify_moment_min_radius = 0.25
    trainer.opt.densify_direction_consistency = 0.6
    trainer.opt.densify_child_offset_fraction = 1.0
    trainer.opt.densify_newborn_grace = 50
    trainer.opt.densify_prune_interval = 0
    trainer.densify_grad_accum.fill_(1.0)
    trainer.densify_denom.fill_(float(topk))
    trainer.densify_residual_mass_accum = torch.zeros(population)
    trainer.densify_centroid_mass_accum = torch.zeros((population, 2))
    trainer.densify_direction_vote_accum = torch.zeros((population, 2))
    trainer.densify_moment_vote_count = torch.zeros(population)
    trainer.densify_mv_edge_top_scores = torch.zeros(
        (population, 6, topk), dtype=torch.float64
    )
    trainer.densify_mv_edge_top_centroids = torch.zeros(
        (population, 6, topk, 2), dtype=torch.float64
    )
    trainer.densify_mv_edge_vote_counts = torch.zeros(
        (population, 6), dtype=torch.int32
    )
    trainer.densify_birth_iteration = torch.zeros(population, dtype=torch.long)
    trainer.densify_contrib_accum = torch.ones(population)
    trainer.train_cameras = [SimpleNamespace(image_name=name) for name in camera_names]
    trainer.densify_shadow_proposal_camera_names = set(camera_names)
    trainer.densify_moment_camera_names = set(camera_names)


def _enable_attributed_densification(
    trainer: Hexels3DTrainer,
    *,
    camera_names: tuple[str, ...] = ("cam0", "cam1"),
    complete: bool = True,
) -> None:
    population = trainer.model.num_base
    trainer.opt.densify_action = "edge_child"
    trainer.opt.densify_direction_signal = "attributed_residual"
    trainer.opt.densify_attributed_weak_power = 0
    trainer.opt.densify_min_denom = len(camera_names)
    trainer.opt.densify_moment_interval = 1
    trainer.opt.densify_moment_min_radius = 0.25
    trainer.opt.densify_direction_consistency = 0.6
    trainer.opt.densify_child_offset_fraction = 1.0
    trainer.opt.densify_newborn_grace = 50
    trainer.opt.densify_prune_interval = 0
    trainer.densify_grad_accum.fill_(1.0)
    trainer.densify_denom.fill_(float(len(camera_names)))
    trainer.densify_attributed_edge_mass = torch.zeros(
        (population, 6), dtype=torch.float64
    )
    trainer.densify_attributed_edge_centroid_mass = torch.zeros(
        (population, 6, 2), dtype=torch.float64
    )
    trainer.densify_attributed_edge_vote_counts = torch.zeros(
        (population, 6), dtype=torch.int32
    )
    trainer.densify_birth_iteration = torch.zeros(population, dtype=torch.long)
    trainer.densify_contrib_accum = torch.ones(population)
    trainer.train_cameras = [SimpleNamespace(image_name=name) for name in camera_names]
    trainer.densify_shadow_proposal_camera_names = set(camera_names)
    trainer.densify_moment_camera_names = set(camera_names) if complete else set()


def _make_attributed_options() -> Hexels3DOptions:
    opt = Hexels3DOptions()
    opt.source_path = "family"
    opt.model_path = "output"
    opt.enable_densify = True
    opt.densify_impl = "hexels_safe"
    opt.densify_action = "edge_child"
    opt.densify_direction_signal = "attributed_residual"
    opt.camera_sampling = "epoch"
    opt.densify_moment_interval = 1
    opt.densify_prune_interval = 0
    opt.opacity_reset_interval = 0
    return opt


def _enable_covariance_densification(
    trainer: Hexels3DTrainer,
    model_path: Path,
    *,
    children_per_parent: int = 2,
    camera_names: tuple[str, ...] = ("cam0", "cam1"),
    complete: bool = True,
) -> None:
    population = trainer.model.num_base
    trainer.opt.model_path = str(model_path)
    trainer.opt.densify_action = "edge_child"
    trainer.opt.densify_direction_signal = "attributed_covariance"
    trainer.opt.densify_attributed_weak_power = 0
    trainer.opt.densify_covariance_children_per_parent = children_per_parent
    trainer.opt.densify_covariance_min_views = len(camera_names)
    trainer.opt.densify_covariance_min_anisotropy = 0.15
    trainer.opt.densify_covariance_min_axis_coherence = 0.50
    trainer.opt.densify_covariance_min_side_balance = 0.30
    trainer.opt.densify_covariance_min_delta = 0.10
    trainer.opt.densify_max_add = children_per_parent
    trainer.opt.densify_child_scale_ratio = 0.5
    trainer.opt.densify_child_init_opacity = 0.01
    trainer.opt.densify_child_extent_cap_fraction = 0.0
    trainer.opt.densify_child_parent_opacity_fraction = (
        0.05 if children_per_parent == 2 else 0.0
    )
    trainer.opt.densify_newborn_grace = 50
    trainer.opt.densify_prune_interval = 0
    trainer.densify_grad_accum.zero_()
    trainer.densify_denom.zero_()
    trainer.densify_covariance_coverage = torch.zeros(
        population, dtype=torch.float64
    )
    trainer.densify_covariance_energy = torch.zeros(
        population, dtype=torch.float64
    )
    trainer.densify_covariance_robust_weight = torch.zeros(
        population, dtype=torch.float64
    )
    trainer.densify_covariance_first_moment = torch.zeros(
        (population, 2), dtype=torch.float64
    )
    trainer.densify_covariance_second_moment = torch.zeros(
        (population, 2, 2), dtype=torch.float64
    )
    trainer.densify_covariance_double_angle = torch.zeros(
        (population, 2), dtype=torch.float64
    )
    trainer.densify_covariance_side_mass = torch.zeros(
        (population, 2), dtype=torch.float64
    )
    trainer.densify_covariance_material_view_count = torch.zeros(
        population, dtype=torch.int32
    )
    trainer.densify_covariance_axis_view_count = torch.zeros(
        population, dtype=torch.int32
    )
    trainer.densify_birth_iteration = torch.zeros(population, dtype=torch.long)
    trainer.densify_contrib_accum = torch.ones(population)
    trainer.train_cameras = [SimpleNamespace(image_name=name) for name in camera_names]
    trainer.densify_shadow_proposal_camera_names = set(camera_names)
    trainer.densify_moment_camera_names = set(camera_names) if complete else set()


def _seed_covariance_pair(
    trainer: Hexels3DTrainer,
    parent_id: int,
    *,
    view_count: int = 2,
) -> None:
    trainer.densify_covariance_coverage[parent_id] = 8.0
    trainer.densify_covariance_energy[parent_id] = 8.0
    trainer.densify_covariance_robust_weight[parent_id] = 16.0
    trainer.densify_covariance_second_moment[parent_id] = torch.tensor(
        [[4.0, 0.0], [0.0, 0.16]], dtype=torch.float64
    )
    trainer.densify_covariance_double_angle[parent_id] = torch.tensor(
        [float(view_count), 0.0], dtype=torch.float64
    )
    trainer.densify_covariance_side_mass[parent_id] = torch.tensor(
        [8.0, 8.0], dtype=torch.float64
    )
    trainer.densify_covariance_material_view_count[parent_id] = view_count
    trainer.densify_covariance_axis_view_count[parent_id] = view_count


def _make_covariance_options(*, children_per_parent: int = 2) -> Hexels3DOptions:
    opt = Hexels3DOptions()
    opt.source_path = "family"
    opt.model_path = "output"
    opt.enable_densify = True
    opt.densify_impl = "hexels_safe"
    opt.densify_action = "edge_child"
    opt.densify_direction_signal = "attributed_covariance"
    opt.camera_sampling = "epoch"
    opt.densify_moment_interval = 1
    opt.densify_prune_interval = 0
    opt.opacity_reset_interval = 0
    opt.densify_covariance_children_per_parent = children_per_parent
    opt.densify_max_add = children_per_parent
    opt.densify_child_parent_opacity_fraction = (
        0.05 if children_per_parent == 2 else 0.0
    )
    return opt


def test_zero_render_interval_disables_periodic_and_final_progress_renders() -> None:
    assert not _should_render_progress(50, iterations=100, render_every=0)
    assert not _should_render_progress(100, iterations=100, render_every=0)
    assert _should_render_progress(50, iterations=100, render_every=50)
    assert _should_render_progress(100, iterations=100, render_every=60)
    assert not _should_render_progress(49, iterations=100, render_every=50)


def test_render_options_restore_view_model(tmp_path: Path) -> None:
    model_path = tmp_path / "model"
    (model_path / "checkpoints").mkdir(parents=True)
    (model_path / "config.json").write_text(
        json.dumps(
            {
                "source_path": "family",
                "view_model": "lobe",
                "model_path": "stale/model/path",
                "data_device": "cuda",
            }
        )
    )

    opt = RenderOptions()
    opt.checkpoint = str(model_path / "checkpoints" / "iteration_10.pt")
    opt.model_path = str(model_path)
    opt.data_device = "cpu"
    opt.process_args()

    assert opt.view_model == "lobe"
    assert opt.model_path == str(model_path)
    assert opt.data_device == "cpu"


def test_render_command_line_overrides_config(tmp_path: Path) -> None:
    model_path = tmp_path / "model"
    (model_path / "checkpoints").mkdir(parents=True)
    (model_path / "config.json").write_text(
        json.dumps({"source_path": "data/dtu/scan24", "resolution": 2, "view_model": "sh3"})
    )

    opt = RenderOptions().parse_args(
        [
            "--checkpoint", str(model_path / "checkpoints" / "ckpt_030000.pt"),
            "--source_path", "elsewhere/scan24",
            "--resolution", "4",
        ]
    )

    assert opt.source_path == "elsewhere/scan24"
    assert opt.resolution == 4
    assert opt.view_model == "sh3"


def test_supersample_camera_and_output_name() -> None:
    world_transform = torch.eye(4)
    projection_transform = torch.eye(4)
    camera = SimpleNamespace(
        image_width=4,
        image_height=3,
        FoVx=0.7,
        FoVy=0.6,
        world_view_transform=world_transform,
        full_proj_transform=projection_transform,
    )

    scaled = _scaled_camera(camera, 3)

    assert (scaled.image_width, scaled.image_height) == (12, 9)
    assert (camera.image_width, camera.image_height) == (4, 3)
    assert scaled.FoVx == camera.FoVx
    assert scaled.FoVy == camera.FoVy
    assert scaled.world_view_transform is world_transform
    assert scaled.full_proj_transform is projection_transform
    assert _render_output_name(50_000, 1) == "ours_50000"
    assert _render_output_name(50_000, 3) == "ours_50000_ss3"
    baseline_dir = Path("test/ours_50000")
    assert _supersample_output_dir(baseline_dir, 3) == Path("test/ours_50000_ss3")
    assert _supersample_output_dir(baseline_dir, 1) == baseline_dir
    assert _supersample_output_dir(Path("test/ours_50000_ss3"), 3) == Path("test/ours_50000_ss3")


def test_supersample_box_filter_preserves_alpha_weighted_depth() -> None:
    color = torch.arange(12, dtype=torch.float32).reshape(3, 2, 2)
    alpha = torch.tensor([[1.0, 0.0], [0.5, 0.5]])
    depth = torch.tensor([[2.0, 100.0], [6.0, 10.0]])
    premultiplied_depth = alpha * depth

    native_color = _box_downsample(color, 2)
    native_alpha = _box_downsample(alpha, 2)
    native_depth_numerator = _box_downsample(premultiplied_depth, 2)

    assert torch.equal(native_color[:, 0, 0], torch.tensor([1.5, 5.5, 9.5]))
    assert native_alpha.item() == pytest.approx(0.5)
    assert (native_depth_numerator / native_alpha).item() == pytest.approx(5.0)


def test_alternate_metrics_paths_are_isolated(tmp_path: Path) -> None:
    alternate = tmp_path / "test" / "ours_50000_ss3"

    render_dir, output_dir = _evaluation_paths(tmp_path, "test", 50_000, alternate)
    baseline_render_dir, baseline_output_dir = _evaluation_paths(tmp_path, "test", 50_000, None)

    assert render_dir == alternate
    assert output_dir == alternate
    assert baseline_render_dir == tmp_path / "test" / "ours_50000"
    assert baseline_output_dir == tmp_path
    assert _iteration_from_render_dir(alternate) == 50_000


def test_render_selection_rejects_empty_test_split() -> None:
    scene = SimpleNamespace(train_cameras=[object()], test_cameras=[])
    with pytest.raises(ValueError, match="No test cameras"):
        _select_split_cameras(scene, "test")


def test_render_views_remove_stale_outputs(tmp_path: Path) -> None:
    class Renderer:
        def render(self, camera: object, model: object, background: torch.Tensor, **kwargs: object) -> dict[str, torch.Tensor]:
            del camera, model, background, kwargs
            return {
                "image": torch.full((3, 2, 2), 0.5),
                "surf_depth": torch.ones((2, 2)),
                "surf_alpha": torch.ones((2, 2)),
            }

    camera = SimpleNamespace(original_image=torch.ones((3, 2, 2)), zfar=10.0)
    for name, filename in (("renders", "stale.png"), ("gt", "stale.png"), ("vis", "stale.tiff")):
        directory = tmp_path / name
        directory.mkdir()
        (directory / filename).write_bytes(b"stale")
    for filename in ("results.json", "per_view.json"):
        (tmp_path / filename).write_text("stale")

    _render_views(
        cams=[camera],
        renderer=Renderer(),
        model=object(),
        bg_color=torch.zeros(3),
        out_dir=tmp_path,
        save_gt=True,
    )

    assert {p.name for p in (tmp_path / "renders").iterdir()} == {"00000.png"}
    assert {p.name for p in (tmp_path / "gt").iterdir()} == {"00000.png"}
    assert {p.name for p in (tmp_path / "vis").iterdir()} == {"depth_00000.tiff"}
    assert not (tmp_path / "results.json").exists()
    assert not (tmp_path / "per_view.json").exists()


def test_metrics_require_exact_filename_pairs(tmp_path: Path) -> None:
    renders = tmp_path / "renders"
    gt = tmp_path / "gt"
    renders.mkdir()
    gt.mkdir()
    (renders / "00000.png").touch()
    (gt / "00001.png").touch()

    with pytest.raises(ValueError, match="missing_gt=.*00000.png.*missing_renders=.*00001.png"):
        _paired_png_names(renders, gt)

    (renders / "00000.png").unlink()
    (gt / "00001.png").unlink()
    (renders / "00002.png").touch()
    (gt / "00002.png").touch()
    assert _paired_png_names(renders, gt) == ["00002.png"]


def test_child_checkpoint_round_trip_and_legacy_defaults() -> None:
    model = _make_model()
    model.append_children(torch.tensor([0]), sides="both")
    with torch.no_grad():
        model.child_rotvec.copy_(torch.arange(6, dtype=torch.float32).reshape(2, 3))
        model.child_specular_log_s.copy_(torch.tensor([0.5, 1.5]))
    state = model.state_dict()

    restored = _make_model()
    restored.allocate_children(2)
    restored.load_state_dict(state)
    for name, value in state.items():
        assert torch.equal(restored.state_dict()[name], value)

    legacy_state = {name: value for name, value in state.items() if name not in {"child_layer", "child_rotvec"}}
    legacy = _make_model()
    legacy.allocate_children(2)
    incompatible = legacy.load_state_dict(legacy_state, strict=False)
    assert set(incompatible.missing_keys) == {"child_layer", "child_rotvec"}
    assert torch.equal(legacy.child_layer, torch.ones(2, dtype=torch.long))
    assert torch.equal(legacy.child_rotvec, torch.zeros((2, 3)))


def test_mcmc_noise_uses_current_position_lr_and_window(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _make_model()
    with torch.no_grad():
        model.opacity_logits.fill_(torch.logit(torch.tensor(0.001)))
    trainer = Hexels3DTrainer.__new__(Hexels3DTrainer)
    trainer.opt = SimpleNamespace(
        enable_mcmc=True,
        mcmc_from_iter=5,
        mcmc_until_iter=10,
        mcmc_noise_lr=2.0,
        mcmc_normal_noise_frac=0.0,
        lr_means3d=999.0,
    )
    trainer.optimize_geometry = True
    trainer.model = model
    trainer.device = torch.device("cpu")
    trainer.optimizer = torch.optim.Adam([{"name": "means3d", "params": [model.means3d], "lr": 0.25}])
    monkeypatch.setattr(
        torch,
        "randn",
        lambda shape, **kwargs: torch.ones(shape, device=kwargs["device"], dtype=kwargs["dtype"]),
    )

    trainer._mcmc_noise_step(4)
    assert torch.equal(model.means3d, torch.zeros_like(model.means3d))

    trainer._mcmc_noise_step(5)
    gate = torch.sigmoid(torch.tensor(0.4))
    expected = torch.tensor([0.5 * gate, 0.5 * gate, 0.0]).repeat(2, 1)
    assert torch.allclose(model.means3d, expected)

    before = model.means3d.detach().clone()
    trainer._mcmc_noise_step(11)
    assert torch.equal(model.means3d, before)


def test_mcmc_relocation_copies_specular_attributes() -> None:
    model = _make_model()
    with torch.no_grad():
        model.opacity_logits.copy_(torch.logit(torch.tensor([0.001, 0.5])))
        model.specular_ks.copy_(torch.tensor([[9.0, 9.0, 9.0], [1.0, 2.0, 3.0]]))
        model.specular_log_s.copy_(torch.tensor([9.0, 2.0]))

    trainer = Hexels3DTrainer.__new__(Hexels3DTrainer)
    trainer.opt = SimpleNamespace(
        enable_children=False,
        mcmc_cap_max=0,
        mcmc_dead_threshold=0.01,
        mcmc_mode="opacity",
        mcmc_reorient=False,
        mcmc_spawn_jitter=0.0,
    )
    trainer.model = model
    trainer.device = torch.device("cpu")
    trainer.mcmc_mode = "opacity"
    trainer.mcmc_binoms = init_relocation_buffers(torch.device("cpu"), torch.float32)
    trainer.contrib_ema = None
    trainer.surface_knn_idx = None
    trainer.optimize_color_grads = True
    trainer.optimize_color_quads = True
    trainer.optimize_view_grads = True
    trainer.optimize_specular = True
    trainer.optimize_geometry = True
    trainer.optimize_softness = True
    trainer.optimizer = torch.optim.Adam([model.means3d])
    trainer.logger = logging.getLogger("test_mcmc_relocation")

    trainer._mcmc_step()

    assert torch.equal(model.specular_ks[0], torch.tensor([1.0, 2.0, 3.0]))
    assert model.specular_log_s[0].item() == 2.0


def test_knn_initial_scales_match_rms_definition_and_cap() -> None:
    points = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [6.0, 0.0, 0.0], [10.0, 0.0, 0.0]]
    )
    scales = _knn_initial_scales(points, k=3, multiplier=0.5, max_scale=2.0)
    expected_first = 0.5 * ((1.0**2 + 3.0**2 + 6.0**2) / 3.0) ** 0.5

    assert scales.shape == (5, 2)
    assert torch.all(scales.isfinite())
    assert torch.all(scales > 0.0)
    assert scales[0, 0].item() == pytest.approx(expected_first)
    assert torch.equal(scales[:, 0], scales[:, 1])
    assert scales.max().item() == 2.0


def test_clone_only_densification_respects_total_cap_and_preserves_optimizer_state() -> None:
    trainer = _make_densify_trainer(num_primitives=3, cap_max=4)
    trainer.model.means3d.grad = torch.ones_like(trainer.model.means3d)
    trainer.optimizer.step()
    trainer.optimizer.zero_grad(set_to_none=True)

    for index, group in enumerate(trainer.optimizer.param_groups):
        group["lr"] = 1e-5 * (index + 1)
        group["betas"] = (0.7, 0.8)
        group["eps"] = 1e-6
        group["weight_decay"] = 0.03
    old_group_options = [
        {key: value for key, value in group.items() if key != "params"}
        for group in trainer.optimizer.param_groups
    ]

    old_means = trainer.model.means3d.detach().clone()
    old_state = trainer.optimizer.state[trainer.model.means3d]["exp_avg"].clone()
    trainer._densify_and_prune_hexels_safe(size_threshold=0)

    assert trainer.model.num_base == 4
    assert torch.equal(trainer.model.means3d[:3], old_means)
    assert torch.equal(trainer.model.means3d[3], old_means[2])
    assert torch.equal(trainer.model.scales, torch.ones((4, 2)))
    assert torch.equal(trainer.model.active_mask, torch.ones(4, dtype=torch.bool))

    means_group = next(group for group in trainer.optimizer.param_groups if group["name"] == "means3d")
    means_state = trainer.optimizer.state[means_group["params"][0]]["exp_avg"]
    assert means_state.shape == (4, 3)
    assert torch.equal(means_state[:3], old_state)
    assert torch.equal(means_state[3], torch.zeros(3))
    for group, expected_options in zip(trainer.optimizer.param_groups, old_group_options):
        assert {key: value for key, value in group.items() if key != "params"} == expected_options

    trainer.densify_grad_accum.fill_(1.0)
    trainer.densify_denom.fill_(1.0)
    trainer._densify_and_prune_hexels_safe(size_threshold=0)
    assert trainer.model.num_base == 4


def test_default_densification_keeps_attributed_buffers_disabled() -> None:
    trainer = _make_densify_trainer(num_primitives=3, cap_max=4)

    trainer._densify_and_prune_hexels_safe(size_threshold=0, iteration=100)

    assert trainer.model.num_base == 4
    assert trainer.densify_attributed_edge_mass is None
    assert trainer.densify_attributed_edge_centroid_mass is None
    assert trainer.densify_attributed_edge_vote_counts is None
    assert trainer.densify_covariance_coverage is None
    assert trainer.densify_covariance_energy is None
    assert trainer.densify_covariance_robust_weight is None
    assert trainer.densify_covariance_first_moment is None
    assert trainer.densify_covariance_second_moment is None
    assert trainer.densify_covariance_double_angle is None
    assert trainer.densify_covariance_side_mass is None
    assert trainer.densify_covariance_material_view_count is None
    assert trainer.densify_covariance_axis_view_count is None


def test_optimizer_step_precedes_density_mutation_and_opacity_reset() -> None:
    trainer = Hexels3DTrainer.__new__(Hexels3DTrainer)
    events: list[object] = []
    trainer.optimizer = SimpleNamespace(step=lambda: events.append("step"), param_groups=[])
    trainer._densify_and_prune = lambda *, size_threshold, iteration: events.append(
        ("densify", size_threshold, iteration)
    )
    trainer._reset_opacity = lambda: events.append("reset")

    trainer._step_optimizer_and_density(
        iteration=23,
        run_densify=True,
        size_threshold=17,
        reset_opacity=True,
    )

    assert events == ["step", ("densify", 17, 23), "reset"]


def test_bisplit_uses_local_gradient_axis_and_resets_parent_state() -> None:
    trainer = _make_densify_trainer(num_primitives=3, cap_max=4)
    trainer.opt.densify_action = "bisplit"
    trainer.opt.densify_newborn_grace = 50
    trainer.opt.densify_split_offset_fraction = 0.25
    trainer.opt.densify_split_scale_ratio = 0.6
    trainer.opt.densify_prune_interval = 1000
    trainer.densify_local_grad_abs_accum = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.5, 4.0]]
    )
    trainer.densify_local_grad_signed_accum = torch.zeros((3, 2))
    trainer.densify_contrib_accum = torch.ones(3)
    trainer.densify_birth_iteration = torch.zeros(3, dtype=torch.long)
    loss = sum(group["params"][0].sum() * 0.0 for group in trainer.optimizer.param_groups)
    loss.backward()
    trainer.optimizer.step()
    trainer.optimizer.zero_grad(set_to_none=True)
    original_parent = trainer.model.means3d.detach()[2].clone()

    trainer._densify_and_prune_hexels_safe(size_threshold=0, iteration=100)

    assert trainer.model.num_base == 4
    assert torch.allclose(trainer.model.means3d[2], original_parent - torch.tensor([0.0, 0.25, 0.0]))
    assert torch.allclose(trainer.model.means3d[3], original_parent + torch.tensor([0.0, 0.25, 0.0]))
    assert torch.allclose(trainer.model.scales[2:], torch.tensor([[1.0, 0.6], [1.0, 0.6]]))
    combined_opacity = 1.0 - (1.0 - trainer.model.opacities[2]) * (1.0 - trainer.model.opacities[3])
    assert combined_opacity.item() == pytest.approx(0.1)
    assert torch.equal(trainer.densify_birth_iteration, torch.tensor([0, 0, 100, 100]))
    means_group = next(group for group in trainer.optimizer.param_groups if group["name"] == "means3d")
    means_state = trainer.optimizer.state[means_group["params"][0]]["exp_avg"]
    assert torch.count_nonzero(means_state[2:]) == 0


def test_bisplit_pruning_protects_newborns_and_keeps_contributors() -> None:
    trainer = _make_densify_trainer(num_primitives=4, cap_max=4)
    trainer.opt.densify_action = "bisplit"
    trainer.opt.densify_prune_interval = 1000
    trainer.opt.densify_newborn_grace = 100
    trainer.opt.densify_prune_contrib_threshold = 0.0
    trainer.densify_local_grad_abs_accum = torch.zeros((4, 2))
    trainer.densify_local_grad_signed_accum = torch.zeros((4, 2))
    trainer.densify_grad_accum.zero_()
    trainer.densify_contrib_accum = torch.tensor([0.0, 1.0, 0.0, 0.0])
    trainer.densify_contrib_camera_names = {"cam0"}
    trainer.densify_birth_iteration = torch.tensor([0, 0, 950, 0])

    trainer._densify_and_prune_hexels_safe(size_threshold=0, iteration=1000)

    assert trainer.model.num_base == 2
    assert torch.equal(trainer.model.means3d[:, 0], torch.tensor([1.0, 2.0]))
    assert torch.equal(trainer.densify_birth_iteration, torch.tensor([0, 950]))
    assert torch.count_nonzero(trainer.densify_contrib_accum) == 0


def test_directional_pruning_runs_without_a_growth_event() -> None:
    trainer = _make_densify_trainer(num_primitives=3, cap_max=3)
    trainer.opt.densify_action = "edge_child"
    trainer.opt.densify_prune_interval = 1000
    trainer.opt.densify_newborn_grace = 100
    trainer.opt.densify_prune_contrib_threshold = 0.0
    trainer.densify_contrib_accum = torch.tensor([0.0, 1.0, 0.0])
    trainer.densify_contrib_camera_names = {"cam0"}
    trainer.densify_birth_iteration = torch.tensor([0, 0, 950])

    trainer._prune_directional_primitives(iteration=1000)

    assert trainer.model.num_base == 3
    assert trainer.model.num_active_base == 2
    assert torch.equal(trainer.model.active_mask, torch.tensor([False, True, True]))
    assert torch.count_nonzero(trainer.densify_contrib_accum) == 0


def test_directional_pruning_waits_for_a_complete_camera_sweep() -> None:
    trainer = _make_densify_trainer(num_primitives=3, cap_max=3)
    trainer.opt.densify_action = "edge_child"
    trainer.opt.densify_prune_interval = 1000
    trainer.opt.densify_newborn_grace = 100
    trainer.opt.densify_prune_contrib_threshold = 0.0
    trainer.train_cameras = [
        SimpleNamespace(image_name="cam0"),
        SimpleNamespace(image_name="cam1"),
    ]
    trainer.densify_contrib_accum = torch.zeros(3)
    trainer.densify_contrib_camera_names = {"cam0"}
    trainer.densify_birth_iteration = torch.zeros(3, dtype=torch.long)

    trainer._prune_directional_primitives(iteration=1000)

    assert trainer.model.num_active_base == 3
    assert trainer.densify_contrib_camera_names == {"cam0"}


def test_full_prune_sweep_collects_each_training_camera() -> None:
    trainer = _make_densify_trainer(num_primitives=3, cap_max=3)
    trainer.train_cameras = [
        SimpleNamespace(image_name="cam0"),
        SimpleNamespace(image_name="cam1"),
    ]
    contributions = {
        "cam0": torch.tensor([1.0, 0.0, 0.0]),
        "cam1": torch.tensor([0.0, 2.0, 0.0]),
    }
    trainer.renderer = SimpleNamespace(
        render=lambda camera, model, background, return_contrib: {
            "contrib": contributions[camera.image_name]
        }
    )
    trainer._render_model = lambda: trainer.model
    trainer.bg_color = torch.zeros(3)
    trainer.densify_contrib_accum = torch.full((3,), 9.0)

    trainer._collect_full_prune_contribution()

    assert torch.equal(trainer.densify_contrib_accum, torch.tensor([1.0, 2.0, 0.0]))
    assert trainer.densify_contrib_camera_names == {"cam0", "cam1"}


def test_full_prune_sweep_rechecks_a_complete_stale_window() -> None:
    trainer = _make_densify_trainer(num_primitives=3, cap_max=3)
    trainer.opt.densify_action = "edge_child"
    trainer.opt.densify_prune_interval = 1000
    trainer.opt.densify_prune_full_sweep = True
    trainer.opt.densify_newborn_grace = 100
    trainer.opt.densify_prune_contrib_threshold = 0.0
    trainer.train_cameras = [
        SimpleNamespace(image_name="cam0"),
        SimpleNamespace(image_name="cam1"),
    ]
    contributions = {
        "cam0": torch.tensor([1.0, 0.0, 0.0]),
        "cam1": torch.tensor([0.0, 1.0, 0.0]),
    }
    calls: list[str] = []

    def render(
        camera: object,
        model: object,
        background: torch.Tensor,
        return_contrib: bool,
    ) -> dict[str, torch.Tensor]:
        calls.append(str(camera.image_name))
        return {"contrib": contributions[str(camera.image_name)]}

    trainer.renderer = SimpleNamespace(render=render)
    trainer._render_model = lambda: trainer.model
    trainer.bg_color = torch.zeros(3)
    trainer.densify_contrib_accum = torch.tensor([0.0, 0.0, 1.0])
    trainer.densify_contrib_camera_names = {"cam0", "cam1"}
    trainer.densify_birth_iteration = torch.zeros(3, dtype=torch.long)

    trainer._prune_directional_primitives(iteration=1000)

    assert calls == ["cam0", "cam1"]
    assert torch.equal(trainer.model.active_mask, torch.tensor([True, True, False]))


def test_coverage_weighted_alpha_loss_targets_low_alpha_error() -> None:
    prediction = torch.zeros((3, 1, 2), requires_grad=True)
    target = torch.tensor([[[1.0, 0.0]], [[1.0, 0.0]], [[1.0, 0.0]]])
    alpha = torch.tensor([[0.25, 0.25]], requires_grad=True)

    loss = _coverage_weighted_alpha_loss(prediction, target, alpha)
    loss.backward()

    assert loss.item() == pytest.approx(0.28125)
    assert prediction.grad is None
    assert torch.equal(alpha.grad, torch.tensor([[-0.75, 0.0]]))


def test_residual_material_mask_keeps_high_mass_or_high_density() -> None:
    mass = torch.ones(30)
    mass[:3] = torch.tensor([10.0, 9.0, 8.0])
    mass[29] = 0.01
    contribution = torch.ones(30)
    contribution[29] = 1e-6

    material = _residual_mass_or_density_mask(mass, contribution)

    expected = torch.zeros(30, dtype=torch.bool)
    expected[[0, 1, 2, 29]] = True
    assert torch.equal(material, expected)


def test_residual_moment_stats_keep_continuous_centroid_and_unique_camera_votes() -> None:
    trainer = _make_densify_trainer(num_primitives=3, cap_max=4)
    trainer.opt.densify_moment_min_radius = 0.25
    trainer.densify_residual_mass_accum = torch.zeros(3)
    trainer.densify_centroid_mass_accum = torch.zeros((3, 2))
    trainer.densify_direction_vote_accum = torch.zeros((3, 2))
    trainer.densify_moment_vote_count = torch.zeros(3)
    prediction = (
        trainer.model.colors[0] + 0.5 * trainer.model.color_grads[0, 0]
    ).reshape(3, 1, 1).expand(-1, 2, 2)
    target = torch.ones_like(prediction)
    trainer.densify_shadow_selector_camera_names = {"selector"}

    trainer._densify_add_moment_stats(
        prediction,
        target,
        torch.tensor([4.0, 0.0, 0.0]),
        camera_name="selector",
    )

    trainer._densify_add_moment_stats(
        prediction,
        target,
        torch.tensor([4.0, 0.0, 0.0]),
        camera_name="cam0",
    )
    trainer._densify_add_moment_stats(
        prediction,
        target,
        torch.tensor([4.0, 0.0, 0.0]),
        camera_name="cam0",
    )

    assert trainer.densify_moment_vote_count[0] == 1.0
    assert "selector" not in trainer.densify_moment_camera_names
    assert trainer.densify_residual_mass_accum[0] == pytest.approx(1.0)
    assert torch.allclose(trainer.densify_centroid_mass_accum[0], torch.tensor([0.5, 0.0]))
    assert torch.equal(trainer.densify_direction_vote_accum[0], torch.tensor([1.0, 0.0]))
    assert torch.count_nonzero(trainer.densify_moment_vote_count[1:]) == 0
    assert trainer.densify_mv_edge_top_scores is None
    assert trainer.densify_mv_edge_top_centroids is None
    assert trainer.densify_mv_edge_vote_counts is None


def test_residual_moment_persistent_filter_uses_per_channel_minimum_error() -> None:
    trainer = _make_densify_trainer(num_primitives=3, cap_max=4)
    trainer.opt.densify_moment_min_radius = 0.25
    trainer.densify_residual_mass_accum = torch.zeros(3)
    trainer.densify_centroid_mass_accum = torch.zeros((3, 2))
    trainer.densify_direction_vote_accum = torch.zeros((3, 2))
    trainer.densify_moment_vote_count = torch.zeros(3)
    prediction = (
        trainer.model.colors[0] + 0.5 * trainer.model.color_grads[0, 0]
    ).reshape(3, 1, 1).expand(-1, 2, 2)
    target = torch.ones_like(prediction)
    filtered_prediction = torch.tensor([0.5, 0.75, 1.0]).reshape(3, 1, 1).expand_as(
        prediction
    )

    trainer._densify_add_moment_stats(
        prediction,
        target,
        torch.tensor([4.0, 0.0, 0.0]),
        camera_name="cam0",
        filtered_prediction=filtered_prediction,
    )

    persistent_error = torch.tensor([0.5, 0.25, 0.0])
    expected_mass = persistent_error.square().sum() / 3.0
    assert trainer.densify_residual_mass_accum[0] == pytest.approx(expected_mass)
    assert torch.allclose(
        trainer.densify_centroid_mass_accum[0],
        expected_mass * torch.tensor([0.5, 0.0]),
    )


def test_residual_topk_stats_keep_unique_camera_edge_evidence() -> None:
    trainer = _make_densify_trainer(num_primitives=3, cap_max=4)
    _enable_residual_topk_densification(trainer)
    trainer.densify_moment_camera_names.clear()
    prediction = (
        trainer.model.colors[0] + 0.5 * trainer.model.color_grads[0, 0]
    ).reshape(3, 1, 1).expand(-1, 2, 2)
    target = torch.ones_like(prediction)

    for camera_name in ("cam0", "cam0", "cam1"):
        trainer._densify_add_moment_stats(
            prediction,
            target,
            torch.tensor([4.0, 0.0, 0.0]),
            camera_name=camera_name,
        )

    assert torch.equal(
        trainer.densify_mv_edge_top_scores[0, 0],
        torch.tensor([1.0, 1.0, 0.0], dtype=torch.float64),
    )
    assert torch.equal(
        trainer.densify_mv_edge_top_centroids[0, 0, :2],
        torch.tensor([[0.5, 0.0], [0.5, 0.0]], dtype=torch.float64),
    )
    assert trainer.densify_mv_edge_vote_counts[0, 0] == 2
    assert torch.count_nonzero(trainer.densify_residual_mass_accum) == 0
    assert trainer.densify_moment_camera_names == {"cam0", "cam1"}


def test_residual_topk_selector_rejects_one_view_spike_and_uses_agreeing_centroid() -> None:
    trainer = _make_densify_trainer(num_primitives=3, cap_max=4)
    _enable_residual_topk_densification(trainer)
    trainer.densify_mv_edge_top_scores[1, 0] = torch.tensor([100.0, 0.0, 0.0])
    trainer.densify_mv_edge_vote_counts[1, 0] = 1
    trainer.densify_mv_edge_top_scores[2, 0] = torch.tensor([3.0, 2.0, 1.0])
    trainer.densify_mv_edge_top_centroids[2, 0] = torch.tensor(
        [[0.9, -0.1], [0.6, -0.2], [0.3, -0.3]],
        dtype=torch.float64,
    )
    trainer.densify_mv_edge_vote_counts[2, 0] = 3
    original = trainer.model.means3d.detach().clone()

    trainer._densify_and_prune_hexels_safe(size_threshold=0, iteration=100)

    assert trainer.model.num_base == 4
    assert torch.equal(
        trainer.densify_has_spawned_child,
        torch.tensor([False, False, True, False]),
    )
    assert torch.allclose(
        trainer.model.means3d[3],
        original[2] + torch.tensor([0.7, -1.0 / 6.0, 0.0]),
    )


def test_residual_topk_shadow_event_preserves_selected_evidence() -> None:
    trainer = _make_densify_trainer(num_primitives=3, cap_max=4)
    _enable_shadow_densification(trainer)
    _enable_residual_topk_densification(trainer)
    trainer.densify_mv_edge_top_scores[2, 0] = torch.tensor([3.0, 2.0, 1.0])
    trainer.densify_mv_edge_top_centroids[2, 0] = torch.tensor(
        [[0.9, -0.1], [0.6, -0.2], [0.3, -0.3]],
        dtype=torch.float64,
    )
    trainer.densify_mv_edge_vote_counts[2, 0] = 3

    trainer._densify_and_prune_hexels_safe(size_threshold=0, iteration=100)

    event = trainer.densify_shadow_event
    assert event is not None
    evidence = event.residual_topk_evidence
    assert evidence is not None
    assert evidence["k"] == 3
    assert evidence["observed_camera_count"] == 3
    assert evidence["dominant_edge_ids"] == [0]
    assert evidence["view_counts"] == [3]
    assert evidence["dominant_edge_vote_counts"] == [3]
    assert evidence["consistency"] == [1.0]
    assert evidence["kth_scores"] == [1.0]
    assert evidence["centroids"][0] == pytest.approx([0.7, -1.0 / 6.0])
    assert evidence["canonical_radii"] == pytest.approx([0.6895511159824405])


def test_residual_topk_selector_rejects_conflicting_edges() -> None:
    camera_names = tuple(f"cam{index}" for index in range(6))
    trainer = _make_densify_trainer(num_primitives=3, cap_max=4)
    _enable_residual_topk_densification(trainer, camera_names=camera_names)
    trainer.densify_denom.fill_(6.0)
    trainer.densify_mv_edge_top_scores[2, 0] = torch.tensor([3.0, 2.0, 1.0])
    trainer.densify_mv_edge_top_centroids[2, 0] = torch.tensor(
        [[0.9, -0.1], [0.6, -0.2], [0.3, -0.3]],
        dtype=torch.float64,
    )
    trainer.densify_mv_edge_vote_counts[2, 0:2] = 3

    trainer._densify_and_prune_hexels_safe(size_threshold=0, iteration=100)

    assert trainer.model.num_base == 3


def test_residual_topk_selector_breaks_score_ties_by_parent_id() -> None:
    trainer = _make_densify_trainer(num_primitives=4, cap_max=5)
    _enable_residual_topk_densification(trainer)
    trainer.opt.densify_max_add = 1
    for parent_id in (1, 3):
        trainer.densify_mv_edge_top_scores[parent_id, 0] = torch.tensor([3.0, 2.0, 1.0])
        trainer.densify_mv_edge_top_centroids[parent_id, 0] = torch.tensor(
            [[0.9, -0.1], [0.6, -0.2], [0.3, -0.3]],
            dtype=torch.float64,
        )
        trainer.densify_mv_edge_vote_counts[parent_id, 0] = 3

    trainer._densify_and_prune_hexels_safe(size_threshold=0, iteration=100)

    assert trainer.densify_has_spawned_child[1]
    assert not trainer.densify_has_spawned_child[3]


def test_residual_topk_incomplete_camera_sweep_defers_without_mutation() -> None:
    trainer = _make_densify_trainer(num_primitives=3, cap_max=4)
    _enable_residual_topk_densification(trainer)
    trainer.densify_moment_camera_names = {"cam0", "cam1"}
    trainer.opt.densify_prune_interval = 100
    trainer.densify_contrib_accum = torch.tensor([1.0, 1.0, 0.0])
    trainer.densify_contrib_camera_names = {"cam0", "cam1", "cam2"}
    trainer.densify_mv_edge_top_scores[2, 0] = torch.tensor([3.0, 2.0, 1.0])
    trainer.densify_mv_edge_top_centroids[2, 0] = torch.tensor(
        [[0.9, -0.1], [0.6, -0.2], [0.3, -0.3]],
        dtype=torch.float64,
    )
    trainer.densify_mv_edge_vote_counts[2, 0] = 3
    scores = trainer.densify_mv_edge_top_scores.clone()
    centroids = trainer.densify_mv_edge_top_centroids.clone()
    votes = trainer.densify_mv_edge_vote_counts.clone()

    trainer._densify_and_prune_hexels_safe(size_threshold=0, iteration=100)

    assert trainer.model.num_active_base == 3
    assert torch.equal(trainer.densify_mv_edge_top_scores, scores)
    assert torch.equal(trainer.densify_mv_edge_top_centroids, centroids)
    assert torch.equal(trainer.densify_mv_edge_vote_counts, votes)
    assert trainer.densify_moment_camera_names == {"cam0", "cam1"}


def test_attributed_stats_use_persistent_error_and_one_vote_per_camera() -> None:
    trainer = _make_densify_trainer(num_primitives=3, cap_max=4)
    _enable_attributed_densification(trainer, complete=False)
    trainer.densify_shadow_selector_camera_names = {"withheld"}
    prediction = torch.tensor(
        [
            [[2.0, 3.0, 4.0]],
            [[0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0]],
        ]
    )
    filtered_prediction = torch.tensor(
        [
            [[1.0, 2.0, 1.0]],
            [[0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0]],
        ]
    )
    owner = {
        "image": prediction.clone(),
        "owner_id": torch.tensor([[0, 1, 2]]),
        "owner_weight": torch.full((1, 3), 0.5),
        "owner_uv": torch.tensor(
            [
                [[0.0, 1.0, 0.0]],
                [[-1.0, 0.0, 1.0]],
            ]
        ),
        "alpha": torch.full((1, 3), 0.75),
    }
    args = (
        prediction,
        filtered_prediction,
        torch.zeros_like(prediction),
        torch.tensor([10.0, 10.0, 0.05]),
        owner,
    )

    trainer._densify_add_attributed_stats(*args, camera_name="withheld")

    assert torch.count_nonzero(trainer.densify_attributed_edge_mass) == 0
    assert trainer.densify_moment_camera_names == set()

    trainer._densify_add_attributed_stats(*args, camera_name="cam0")

    assert trainer.densify_attributed_edge_mass[1, 0] == pytest.approx(2.0)
    assert torch.equal(
        trainer.densify_attributed_edge_centroid_mass[1, 0],
        torch.tensor([2.0, 0.0], dtype=torch.float64),
    )
    assert trainer.densify_attributed_edge_mass[2, 4] == pytest.approx(0.5)
    assert torch.equal(
        trainer.densify_attributed_edge_centroid_mass[2, 4],
        torch.tensor([0.0, 0.5], dtype=torch.float64),
    )
    assert torch.count_nonzero(trainer.densify_attributed_edge_mass[0]) == 0
    assert torch.equal(
        trainer.densify_attributed_edge_vote_counts.sum(dim=1),
        torch.tensor([0, 1, 1], dtype=torch.int32),
    )
    assert trainer.densify_moment_camera_names == {"cam0"}
    expected_mass = trainer.densify_attributed_edge_mass.clone()
    expected_centroid_mass = trainer.densify_attributed_edge_centroid_mass.clone()
    expected_votes = trainer.densify_attributed_edge_vote_counts.clone()

    trainer._densify_add_attributed_stats(*args, camera_name="cam0")

    assert torch.equal(trainer.densify_attributed_edge_mass, expected_mass)
    assert torch.equal(
        trainer.densify_attributed_edge_centroid_mass,
        expected_centroid_mass,
    )
    assert torch.equal(trainer.densify_attributed_edge_vote_counts, expected_votes)


def test_attributed_incomplete_camera_sweep_defers_without_mutation() -> None:
    trainer = _make_densify_trainer(num_primitives=3, cap_max=4)
    _enable_attributed_densification(trainer, complete=False)
    trainer.densify_moment_camera_names = {"cam0"}
    trainer.opt.densify_prune_interval = 100
    trainer.densify_contrib_accum = torch.tensor([1.0, 1.0, 0.0])
    trainer.densify_contrib_camera_names = {"cam0", "cam1"}
    trainer.densify_attributed_edge_mass[2, 0] = 6.0
    trainer.densify_attributed_edge_centroid_mass[2, 0] = torch.tensor(
        [4.2, -1.0], dtype=torch.float64
    )
    trainer.densify_attributed_edge_vote_counts[2, 0] = 2
    active = trainer.model.active_mask.clone()
    mass = trainer.densify_attributed_edge_mass.clone()
    centroid_mass = trainer.densify_attributed_edge_centroid_mass.clone()
    votes = trainer.densify_attributed_edge_vote_counts.clone()

    trainer._densify_and_prune_hexels_safe(size_threshold=0, iteration=100)

    assert trainer.model.num_base == 3
    assert torch.equal(trainer.model.active_mask, active)
    assert torch.equal(trainer.densify_attributed_edge_mass, mass)
    assert torch.equal(trainer.densify_attributed_edge_centroid_mass, centroid_mass)
    assert torch.equal(trainer.densify_attributed_edge_vote_counts, votes)
    assert trainer.densify_moment_camera_names == {"cam0"}


def test_attributed_selected_centroid_reaches_child_birth() -> None:
    trainer = _make_densify_trainer(num_primitives=3, cap_max=4)
    _enable_attributed_densification(trainer)
    trainer.densify_attributed_edge_mass[2, 0] = 6.0
    trainer.densify_attributed_edge_centroid_mass[2, 0] = torch.tensor(
        [4.2, -1.0], dtype=torch.float64
    )
    trainer.densify_attributed_edge_vote_counts[2, 0] = 2
    original = trainer.model.means3d.detach().clone()

    trainer._densify_and_prune_hexels_safe(size_threshold=0, iteration=100)

    assert trainer.model.num_base == 4
    assert torch.equal(trainer.model.means3d[:3], original)
    assert torch.equal(
        trainer.densify_has_spawned_child,
        torch.tensor([False, False, True, False]),
    )
    assert torch.allclose(
        trainer.model.means3d[3],
        original[2] + torch.tensor([0.7, -1.0 / 6.0, 0.0]),
    )


def test_covariance_stats_accumulate_material_owner_local_moments(
    tmp_path: Path,
) -> None:
    trainer = _make_densify_trainer(num_primitives=3, cap_max=4)
    _enable_covariance_densification(trainer, tmp_path, complete=False)
    prediction = torch.tensor(
        [
            [[2.0, 3.0, 4.0]],
            [[0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0]],
        ]
    )
    filtered_prediction = torch.tensor(
        [
            [[1.0, 2.0, 1.0]],
            [[0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0]],
        ]
    )
    owner = {
        "image": prediction.clone(),
        "owner_id": torch.tensor([[0, 1, 2]]),
        "owner_weight": torch.full((1, 3), 0.5),
        "owner_uv": torch.tensor(
            [
                [[0.0, 1.0, 0.0]],
                [[-1.0, 0.0, 1.0]],
            ]
        ),
        "alpha": torch.full((1, 3), 0.75),
    }
    args = (
        prediction,
        filtered_prediction,
        torch.zeros_like(prediction),
        torch.tensor([10.0, 10.0, 0.05]),
        owner,
    )

    trainer._densify_add_covariance_stats(*args, camera_name="cam0")

    assert trainer.densify_covariance_coverage.tolist() == [0.0, 0.5, 0.5]
    assert trainer.densify_covariance_energy.tolist() == [0.0, 2.0, 0.5]
    assert trainer.densify_covariance_robust_weight.tolist() == [0.0, 0.5, 0.5]
    assert torch.equal(
        trainer.densify_covariance_first_moment,
        torch.tensor(
            [[0.0, 0.0], [0.5, 0.0], [0.0, 0.5]], dtype=torch.float64
        ),
    )
    assert torch.equal(
        trainer.densify_covariance_material_view_count,
        torch.tensor([0, 1, 1], dtype=torch.int32),
    )
    assert torch.equal(
        trainer.densify_covariance_axis_view_count,
        torch.tensor([0, 1, 1], dtype=torch.int32),
    )
    assert torch.equal(
        trainer.densify_covariance_side_mass,
        torch.tensor(
            [[0.0, 0.0], [0.0, 0.5], [0.0, 0.5]], dtype=torch.float64
        ),
    )
    assert trainer.densify_moment_camera_names == {"cam0"}
    snapshot = tuple(
        buffer.clone()
        for buffer in (
            trainer.densify_covariance_coverage,
            trainer.densify_covariance_energy,
            trainer.densify_covariance_first_moment,
            trainer.densify_covariance_material_view_count,
        )
    )

    trainer._densify_add_covariance_stats(*args, camera_name="cam0")

    for buffer, expected in zip(
        (
            trainer.densify_covariance_coverage,
            trainer.densify_covariance_energy,
            trainer.densify_covariance_first_moment,
            trainer.densify_covariance_material_view_count,
        ),
        snapshot,
    ):
        assert torch.equal(buffer, expected)


def test_covariance_incomplete_camera_sweep_defers_without_mutation(
    tmp_path: Path,
) -> None:
    trainer = _make_densify_trainer(num_primitives=3, cap_max=5)
    _enable_covariance_densification(trainer, tmp_path, complete=False)
    _seed_covariance_pair(trainer, 2)
    trainer.densify_moment_camera_names = {"cam0"}
    before = tuple(
        buffer.clone()
        for buffer in (
            trainer.densify_covariance_coverage,
            trainer.densify_covariance_energy,
            trainer.densify_covariance_second_moment,
            trainer.densify_covariance_material_view_count,
        )
    )

    trainer._densify_and_prune_hexels_safe(size_threshold=0, iteration=100)

    assert trainer.model.num_base == 3
    for buffer, expected in zip(
        (
            trainer.densify_covariance_coverage,
            trainer.densify_covariance_energy,
            trainer.densify_covariance_second_moment,
            trainer.densify_covariance_material_view_count,
        ),
        before,
    ):
        assert torch.equal(buffer, expected)
    assert not list(tmp_path.glob("densify_covariance_birth_*.json"))


def test_covariance_pair_selection_is_centered_and_writes_birth_sidecar(
    tmp_path: Path,
) -> None:
    trainer = _make_densify_trainer(num_primitives=3, cap_max=5)
    _enable_covariance_densification(trainer, tmp_path)
    _seed_covariance_pair(trainer, 2)
    original = trainer.model.means3d.detach().clone()

    trainer._densify_and_prune_hexels_safe(size_threshold=0, iteration=100)

    assert trainer.model.num_base == 5
    assert torch.equal(trainer.model.means3d[:3], original)
    assert torch.allclose(
        trainer.model.means3d[3:],
        torch.tensor([[1.6, 0.0, 0.0], [2.4, 0.0, 0.0]]),
    )
    assert torch.equal(
        trainer.densify_has_spawned_child,
        torch.tensor([False, False, True, False, False]),
    )
    sidecar = json.loads(
        (tmp_path / "densify_covariance_birth_000100.json").read_text()
    )
    assert sidecar["schema"] == "hexels3d_covariance_birth_v1"
    assert sidecar["parent_ids"] == [2]
    assert sidecar["child_parent_ids"] == [2, 2]
    assert sidecar["child_ids"] == [3, 4]
    assert torch.allclose(
        torch.tensor(sidecar["child_local_offsets"]),
        torch.tensor([[-0.4, 0.0], [0.4, 0.0]]),
    )
    assert sidecar["selected_evidence"]["center"] == [[0.0, 0.0]]
    assert sidecar["selected_evidence"]["material_view_count"] == [2]
    assert sidecar["eligible_candidate_count"] == 1
    assert sidecar["actual_child_count"] == 2
    assert sidecar["population"] == {
        "active_after": 5,
        "active_before": 3,
        "appended_rows": 2,
        "capacity_after": 5,
        "capacity_before": 3,
        "reused_inactive_rows": 0,
    }
    assert len(sidecar["selection_sha256"]) == 64
    assert len(sidecar["evidence_sha256"]) == 64
    assert set(sidecar["source_hashes"]) == {
        "hexels/attributed_residual.py",
        "hexels/edge_shift.py",
        "hexels/options.py",
        "hexels/refinement.py",
        "hexels/trainer.py",
    }


def test_covariance_one_child_uses_center_without_pair_gates(tmp_path: Path) -> None:
    trainer = _make_densify_trainer(num_primitives=3, cap_max=4)
    _enable_covariance_densification(
        trainer,
        tmp_path,
        children_per_parent=1,
    )
    trainer.densify_covariance_coverage[2] = 2.0
    trainer.densify_covariance_energy[2] = 4.0
    trainer.densify_covariance_robust_weight[2] = 2.0
    trainer.densify_covariance_first_moment[2] = torch.tensor(
        [1.0, 0.0], dtype=torch.float64
    )
    trainer.densify_covariance_second_moment[2] = torch.tensor(
        [[0.7, 0.0], [0.0, 0.2]], dtype=torch.float64
    )
    trainer.densify_covariance_material_view_count[2] = 2

    trainer._densify_and_prune_hexels_safe(size_threshold=0, iteration=100)

    assert trainer.model.num_base == 4
    assert torch.allclose(trainer.model.means3d[3], torch.tensor([2.3, 0.0, 0.0]))
    sidecar = json.loads(
        (tmp_path / "densify_covariance_birth_000100.json").read_text()
    )
    assert sidecar["children_per_parent"] == 1
    assert sidecar["child_parent_ids"] == [2]
    assert torch.allclose(
        torch.tensor(sidecar["child_local_offsets"]),
        torch.tensor([[0.3, 0.0]]),
    )
    assert sidecar["selected_evidence"]["axis_coherence"] == [0.0]


def test_covariance_pair_cap_counts_child_rows(tmp_path: Path) -> None:
    trainer = _make_densify_trainer(num_primitives=3, cap_max=4)
    _enable_covariance_densification(trainer, tmp_path)
    _seed_covariance_pair(trainer, 2)

    trainer._densify_and_prune_hexels_safe(size_threshold=0, iteration=100)

    assert trainer.model.num_base == 3
    assert trainer.model.num_active_base == 3
    assert not list(tmp_path.glob("densify_covariance_birth_*.json"))


def test_persistent_residual_moment_requires_complete_camera_sweep() -> None:
    trainer = _make_densify_trainer(num_primitives=3, cap_max=4)
    trainer.opt.densify_action = "edge_child"
    trainer.opt.densify_direction_signal = "residual_moment"
    trainer.opt.densify_residual_persistent_filter = True
    trainer.opt.densify_newborn_grace = 50
    trainer.opt.densify_prune_interval = 0
    trainer.densify_residual_mass_accum = torch.tensor([0.0, 0.0, 4.0])
    trainer.densify_centroid_mass_accum = torch.tensor(
        [[0.0, 0.0], [0.0, 0.0], [4.0, 0.0]]
    )
    trainer.densify_direction_vote_accum = torch.tensor(
        [[0.0, 0.0], [0.0, 0.0], [2.0, 0.0]]
    )
    trainer.densify_moment_vote_count = torch.tensor([0.0, 0.0, 2.0])
    trainer.densify_birth_iteration = torch.zeros(3, dtype=torch.long)
    trainer.densify_shadow_proposal_camera_names = {"cam0", "cam1"}
    trainer.densify_moment_camera_names = {"cam0"}

    trainer._densify_and_prune_hexels_safe(size_threshold=0, iteration=100)

    assert trainer.model.num_base == 3
    assert trainer.densify_moment_camera_names == {"cam0"}


def test_edge_child_keeps_parent_and_uses_consistent_descent_direction() -> None:
    trainer = _make_densify_trainer(num_primitives=3, cap_max=4)
    trainer.opt.densify_action = "edge_child"
    trainer.opt.densify_child_offset_fraction = 0.75
    trainer.opt.densify_child_scale_ratio = 0.5
    trainer.opt.densify_child_softness_ratio = 0.5
    trainer.opt.densify_child_init_opacity = 0.01
    trainer.opt.densify_direction_consistency = 0.25
    trainer.opt.densify_newborn_grace = 50
    trainer.opt.densify_prune_interval = 1000
    trainer.densify_local_grad_abs_accum = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.5, 4.0]]
    )
    trainer.densify_local_grad_signed_accum = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 3.0]]
    )
    trainer.densify_contrib_accum = torch.ones(3)
    trainer.densify_birth_iteration = torch.zeros(3, dtype=torch.long)
    trainer.densify_has_spawned_child = torch.zeros(3, dtype=torch.bool)
    original = trainer.model.means3d.detach().clone()

    trainer._densify_and_prune_hexels_safe(size_threshold=0, iteration=100)

    assert trainer.model.num_base == 4
    assert torch.equal(trainer.model.means3d[:3], original)
    assert torch.allclose(trainer.model.means3d[3], original[2] - torch.tensor([0.0, 0.75, 0.0]))
    assert torch.allclose(trainer.model.scales[3], torch.tensor([0.5, 0.5]))
    assert torch.allclose(
        trainer.model.log_softness[3].exp(),
        trainer.model.log_softness[2].exp() * 0.25,
    )
    assert trainer.model.opacities[3].item() == pytest.approx(0.01)
    assert torch.equal(trainer.densify_birth_iteration, torch.tensor([0, 0, 0, 100]))
    assert torch.equal(trainer.densify_has_spawned_child, torch.tensor([False, False, True, False]))


def test_edge_child_reorientation_preserves_tangent_color_gradient() -> None:
    trainer = _make_densify_trainer(num_primitives=3, cap_max=4)
    trainer.opt.densify_action = "edge_child"
    trainer.opt.densify_child_reorient = True
    trainer.opt.densify_child_offset_fraction = 0.75
    trainer.opt.densify_child_scale_ratio = 0.5
    trainer.opt.densify_child_init_opacity = 0.01
    trainer.opt.densify_direction_consistency = 0.25
    trainer.opt.densify_newborn_grace = 50
    trainer.opt.densify_prune_interval = 0
    trainer.densify_local_grad_abs_accum = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.5, 4.0]]
    )
    trainer.densify_local_grad_signed_accum = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 3.0]]
    )
    trainer.densify_contrib_accum = torch.ones(3)
    trainer.densify_birth_iteration = torch.zeros(3, dtype=torch.long)
    trainer.densify_has_spawned_child = torch.zeros(3, dtype=torch.bool)
    trainer.model.color_grads.data[2, 0, 0] = 1.0
    trainer._estimate_pca_normals = lambda parent_ids: torch.tensor(
        [[0.0, 1.0, 0.0]]
    ).expand(parent_ids.numel(), -1)

    trainer._densify_and_prune_hexels_safe(size_threshold=0, iteration=100)

    child_rotation = trainer.model.rotations.detach()[3:4]
    child_normal = _quat_to_normal_z(child_rotation)[0]
    child_tx, child_ty = _quat_to_tangent_xy(child_rotation)
    child_scales = trainer.model.scales.detach()[3]
    child_grads = trainer.model.color_grads.detach()[3]
    world_gradient = (
        child_tx[0] * child_grads[0, 0] / child_scales[0]
        + child_ty[0] * child_grads[1, 0] / child_scales[1]
    )

    assert torch.allclose(
        child_normal,
        torch.tensor([0.0, 1.0, 0.0]),
        atol=1e-6,
    )
    assert torch.allclose(
        world_gradient,
        torch.tensor([1.0, 0.0, 0.0]),
        atol=1e-6,
    )


def test_pca_planarity_rejects_line_neighborhoods() -> None:
    trainer = _make_densify_trainer(num_primitives=10, cap_max=10)
    trainer.model.means3d.data.copy_(
        torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [-1.0, -1.0, 0.0],
                [-1.0, 1.0, 0.0],
                [1.0, -1.0, 0.0],
                [1.0, 1.0, 0.0],
                [-3.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ]
        )
    )
    trainer.surface_knn_idx = torch.zeros((10, 4), dtype=torch.long)
    trainer.surface_knn_idx[0] = torch.tensor([2, 3, 4, 5])
    trainer.surface_knn_idx[1] = torch.tensor([6, 7, 8, 9])

    planarity = trainer._estimate_pca_planarity(torch.tensor([0, 1]))

    assert torch.allclose(planarity, torch.tensor([1.0, 0.0]))


def test_edge_child_wires_scene_extent_caps_and_parent_opacity() -> None:
    trainer = _make_densify_trainer(num_primitives=3, cap_max=4)
    trainer.scene_extent = 2.0
    trainer.opt.densify_action = "edge_child"
    trainer.opt.densify_child_offset_fraction = 1.0
    trainer.opt.densify_child_scale_ratio = 0.5
    trainer.opt.densify_child_init_opacity = 0.05
    trainer.opt.densify_child_extent_cap_fraction = 0.1
    trainer.opt.densify_child_parent_opacity_fraction = 0.25
    trainer.opt.densify_direction_consistency = 0.25
    trainer.opt.densify_newborn_grace = 50
    trainer.opt.densify_prune_interval = 0
    trainer.densify_local_grad_abs_accum = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [4.0, 0.5]]
    )
    trainer.densify_local_grad_signed_accum = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [3.0, 0.0]]
    )
    trainer.densify_contrib_accum = torch.ones(3)
    trainer.densify_birth_iteration = torch.zeros(3, dtype=torch.long)
    trainer.densify_has_spawned_child = torch.zeros(3, dtype=torch.bool)
    original = trainer.model.means3d.detach().clone()

    trainer._densify_and_prune_hexels_safe(size_threshold=0, iteration=100)

    assert torch.equal(trainer.model.means3d[:3], original)
    assert torch.allclose(trainer.model.means3d[3], original[2] - torch.tensor([0.2, 0.0, 0.0]))
    assert torch.allclose(trainer.model.scales[3], torch.full((2,), 0.2))
    assert trainer.model.opacities[3].item() == pytest.approx(0.025)


def test_edge_child_parent_spawns_once_and_child_can_refine_after_grace() -> None:
    trainer = _make_densify_trainer(num_primitives=3, cap_max=6)
    trainer.opt.densify_action = "edge_child"
    trainer.opt.densify_max_add = 1
    trainer.opt.densify_child_offset_fraction = 0.75
    trainer.opt.densify_child_scale_ratio = 0.5
    trainer.opt.densify_child_init_opacity = 0.01
    trainer.opt.densify_direction_consistency = 0.25
    trainer.opt.densify_newborn_grace = 50
    trainer.opt.densify_prune_interval = 0
    trainer.densify_local_grad_abs_accum = torch.ones((3, 2))
    trainer.densify_local_grad_signed_accum = torch.ones((3, 2))
    trainer.densify_contrib_accum = torch.ones(3)
    trainer.densify_birth_iteration = torch.zeros(3, dtype=torch.long)
    trainer.densify_has_spawned_child = torch.zeros(3, dtype=torch.bool)

    trainer._densify_and_prune_hexels_safe(size_threshold=0, iteration=100)
    assert torch.equal(trainer.densify_has_spawned_child, torch.tensor([False, False, True, False]))

    trainer.densify_grad_accum[:] = torch.tensor([[1.0], [2.0], [10.0], [0.5]])
    trainer.densify_denom.fill_(1.0)
    trainer.densify_local_grad_abs_accum.fill_(1.0)
    trainer.densify_local_grad_signed_accum.fill_(1.0)
    trainer._densify_and_prune_hexels_safe(size_threshold=0, iteration=200)
    assert torch.equal(
        trainer.densify_has_spawned_child,
        torch.tensor([False, True, True, False, False]),
    )

    trainer.densify_grad_accum.zero_()
    trainer.densify_grad_accum[3] = 10.0
    trainer.densify_denom.fill_(1.0)
    trainer.densify_local_grad_abs_accum.fill_(1.0)
    trainer.densify_local_grad_signed_accum.fill_(1.0)
    trainer._densify_and_prune_hexels_safe(size_threshold=0, iteration=300)
    assert torch.equal(
        trainer.densify_has_spawned_child,
        torch.tensor([False, True, True, True, False, False]),
    )


def test_edge_child_rejects_cross_view_direction_cancellation() -> None:
    trainer = _make_densify_trainer(num_primitives=3, cap_max=4)
    trainer.opt.densify_action = "edge_child"
    trainer.opt.densify_direction_consistency = 0.25
    trainer.opt.densify_newborn_grace = 50
    trainer.opt.densify_prune_interval = 1000
    trainer.densify_local_grad_abs_accum = torch.ones((3, 2))
    trainer.densify_local_grad_signed_accum = torch.zeros((3, 2))
    trainer.densify_contrib_accum = torch.ones(3)
    trainer.densify_birth_iteration = torch.zeros(3, dtype=torch.long)

    trainer._densify_and_prune_hexels_safe(size_threshold=0, iteration=100)

    assert trainer.model.num_base == 3


def test_residual_moment_child_uses_continuous_centroid() -> None:
    trainer = _make_densify_trainer(num_primitives=3, cap_max=4)
    trainer.opt.densify_action = "edge_child"
    trainer.opt.densify_direction_signal = "residual_moment"
    trainer.opt.densify_direction_consistency = 2.0 / 3.0
    trainer.opt.densify_child_offset_fraction = 1.0
    trainer.opt.densify_child_scale_ratio = 0.5
    trainer.opt.densify_child_init_opacity = 0.01
    trainer.opt.densify_newborn_grace = 50
    trainer.opt.densify_prune_interval = 1000
    trainer.densify_residual_mass_accum = torch.zeros(3)
    trainer.densify_centroid_mass_accum = torch.zeros((3, 2))
    trainer.densify_direction_vote_accum = torch.zeros((3, 2))
    trainer.densify_moment_vote_count = torch.zeros(3)
    trainer.densify_residual_mass_accum[2] = 5.0
    trainer.densify_centroid_mass_accum[2] = torch.tensor([3.0, 4.0])
    trainer.densify_direction_vote_accum[2] = 5.0 * torch.tensor([0.6, 0.8])
    trainer.densify_moment_vote_count[2] = 5.0
    trainer.densify_contrib_accum = torch.ones(3)
    trainer.densify_birth_iteration = torch.zeros(3, dtype=torch.long)
    original = trainer.model.means3d.detach().clone()

    trainer._densify_and_prune_hexels_safe(size_threshold=0, iteration=100)

    assert trainer.model.num_base == 4
    assert torch.allclose(trainer.model.means3d[3], original[2] + torch.tensor([0.6, 0.8, 0.0]))


def test_residual_moment_child_requires_noncentral_aligned_evidence() -> None:
    trainer = _make_densify_trainer(num_primitives=3, cap_max=4)
    trainer.opt.densify_action = "edge_child"
    trainer.opt.densify_direction_signal = "residual_moment"
    trainer.opt.densify_direction_consistency = 0.6
    trainer.opt.densify_moment_min_radius = 0.25
    trainer.opt.densify_newborn_grace = 50
    trainer.opt.densify_prune_interval = 0
    trainer.densify_residual_mass_accum = torch.tensor([0.0, 5.0, 5.0])
    trainer.densify_centroid_mass_accum = torch.tensor(
        [[0.0, 0.0], [0.0, 0.0], [3.0, 0.0]]
    )
    trainer.densify_direction_vote_accum = torch.tensor(
        [[0.0, 0.0], [5.0, 0.0], [-5.0, 0.0]]
    )
    trainer.densify_moment_vote_count = torch.tensor([0.0, 5.0, 5.0])
    trainer.densify_contrib_accum = torch.ones(3)
    trainer.densify_birth_iteration = torch.zeros(3, dtype=torch.long)

    trainer._densify_and_prune_hexels_safe(size_threshold=0, iteration=100)

    assert trainer.model.num_base == 3


def test_residual_moment_child_requires_a_material_camera_vote() -> None:
    trainer = _make_densify_trainer(num_primitives=3, cap_max=4)
    trainer.opt.densify_min_denom = 0
    trainer.opt.densify_action = "edge_child"
    trainer.opt.densify_direction_signal = "residual_moment"
    trainer.opt.densify_direction_consistency = 0.0
    trainer.opt.densify_moment_min_radius = 0.0
    trainer.opt.densify_newborn_grace = 50
    trainer.opt.densify_prune_interval = 0
    trainer.densify_residual_mass_accum = torch.zeros(3)
    trainer.densify_centroid_mass_accum = torch.zeros((3, 2))
    trainer.densify_direction_vote_accum = torch.zeros((3, 2))
    trainer.densify_moment_vote_count = torch.zeros(3)
    trainer.densify_contrib_accum = torch.ones(3)
    trainer.densify_birth_iteration = torch.zeros(3, dtype=torch.long)

    trainer._densify_and_prune_hexels_safe(size_threshold=0, iteration=100)

    assert trainer.model.num_base == 3


def test_directional_densify_does_not_resplit_newborns() -> None:
    trainer = _make_densify_trainer(num_primitives=3, cap_max=4)
    trainer.opt.densify_action = "edge_child"
    trainer.opt.densify_newborn_grace = 100
    trainer.opt.densify_direction_consistency = 0.25
    trainer.opt.densify_prune_interval = 1000
    trainer.densify_local_grad_abs_accum = torch.ones((3, 2))
    trainer.densify_local_grad_signed_accum = torch.ones((3, 2))
    trainer.densify_contrib_accum = torch.ones(3)
    trainer.densify_birth_iteration = torch.tensor([0, 0, 150], dtype=torch.long)

    trainer._densify_and_prune_hexels_safe(size_threshold=0, iteration=200)

    assert trainer.model.num_base == 4
    assert torch.allclose(
        trainer.model.means3d[3],
        trainer.model.means3d[1] - torch.tensor([0.75, 0.0, 0.0]),
    )
    assert torch.equal(trainer.densify_birth_iteration, torch.tensor([0, 0, 150, 200]))


def test_directional_pruning_refills_capacity_in_the_same_event() -> None:
    trainer = _make_densify_trainer(num_primitives=3, cap_max=3)
    trainer.opt.densify_action = "edge_child"
    trainer.opt.densify_newborn_grace = 100
    trainer.opt.densify_direction_consistency = 0.25
    trainer.opt.densify_prune_interval = 1000
    trainer.densify_local_grad_abs_accum = torch.ones((3, 2))
    trainer.densify_local_grad_signed_accum = torch.ones((3, 2))
    trainer.densify_contrib_accum = torch.tensor([1.0, 1.0, 0.0])
    trainer.densify_contrib_camera_names = {"cam0"}
    trainer.densify_birth_iteration = torch.zeros(3, dtype=torch.long)

    trainer._densify_and_prune_hexels_safe(size_threshold=0, iteration=1000)

    assert trainer.model.num_base == 3
    assert trainer.model.num_active_base == 3
    assert trainer.densify_birth_iteration[-1] == 1000


def test_directional_densify_reuses_inactive_capacity() -> None:
    trainer = _make_densify_trainer(num_primitives=4, cap_max=4)
    trainer.opt.densify_action = "edge_child"
    trainer.opt.densify_child_offset_fraction = 0.75
    trainer.opt.densify_child_scale_ratio = 0.5
    trainer.opt.densify_child_init_opacity = 0.01
    trainer.opt.densify_direction_consistency = 0.25
    trainer.opt.densify_newborn_grace = 50
    trainer.opt.densify_prune_interval = 0
    trainer.model.active_mask[3] = False
    trainer.densify_local_grad_abs_accum = torch.ones((4, 2))
    trainer.densify_local_grad_signed_accum = torch.ones((4, 2))
    trainer.densify_contrib_accum = torch.ones(4)
    trainer.densify_birth_iteration = torch.zeros(4, dtype=torch.long)
    optimizer = trainer.optimizer

    trainer._densify_and_prune_hexels_safe(size_threshold=0, iteration=100)

    assert trainer.model.num_base == 4
    assert trainer.model.num_active_base == 4
    assert trainer.optimizer is optimizer
    assert trainer.densify_birth_iteration[3] == 100
    assert torch.equal(trainer.densify_has_spawned_child, torch.tensor([False, False, True, False]))
    assert torch.allclose(trainer.model.means3d[3], torch.tensor([1.25, 0.0, 0.0]))


def test_replacing_a_reservoir_row_clears_optimizer_gradients_and_statistics() -> None:
    trainer = _make_densify_trainer(num_primitives=3, cap_max=3)
    loss = sum(group["params"][0].sum() for group in trainer.optimizer.param_groups)
    loss.backward()
    trainer.optimizer.step()

    trainer.densify_grad_accum.fill_(1.0)
    trainer.densify_denom.fill_(2.0)
    trainer.densify_max_radii2d.fill_(3.0)
    trainer.densify_local_grad_abs_accum = torch.full((3, 2), 4.0)
    trainer.densify_local_grad_signed_accum = torch.full((3, 2), 5.0)
    trainer.densify_residual_mass_accum = torch.full((3,), 6.0)
    trainer.densify_centroid_mass_accum = torch.full((3, 2), 7.0)
    trainer.densify_direction_vote_accum = torch.full((3, 2), 8.0)
    trainer.densify_moment_vote_count = torch.full((3,), 9.0)
    trainer.densify_contrib_accum = torch.full((3,), 10.0)
    trainer.densify_birth_iteration = torch.zeros(3, dtype=torch.long)
    trainer.densify_has_spawned_child = torch.tensor([False, True, True])
    row_names = (
        "means3d",
        "log_scales",
        "rotation_quats",
        "colors",
        "color_grads",
        "color_quads",
        "view_grads",
        "specular_ks",
        "specular_log_s",
        "opacity_logits",
        "log_softness",
    )
    rows = {
        name: getattr(trainer.model, name).detach()[0:1].clone()
        for name in row_names
    }

    trainer._replace_primitive_rows(
        torch.tensor([1]),
        rows,
        birth_iteration=17,
    )

    for group in trainer.optimizer.param_groups:
        parameter = group["params"][0]
        if parameter.grad is not None:
            assert torch.count_nonzero(parameter.grad[1]) == 0
        for name in ("exp_avg", "exp_avg_sq"):
            assert torch.count_nonzero(trainer.optimizer.state[parameter][name][1]) == 0
    for buffer in (
        trainer.densify_grad_accum,
        trainer.densify_denom,
        trainer.densify_max_radii2d,
        trainer.densify_local_grad_abs_accum,
        trainer.densify_local_grad_signed_accum,
        trainer.densify_residual_mass_accum,
        trainer.densify_centroid_mass_accum,
        trainer.densify_direction_vote_accum,
        trainer.densify_moment_vote_count,
    ):
        assert buffer is not None
        assert torch.count_nonzero(buffer[1]) == 0
    assert trainer.densify_birth_iteration[1] == 17
    assert not trainer.densify_has_spawned_child[1]


def test_residual_topk_buffers_follow_primitive_lifecycle() -> None:
    trainer = _make_densify_trainer(num_primitives=3, cap_max=4)
    _enable_residual_topk_densification(
        trainer,
        topk=2,
        camera_names=("cam0", "cam1"),
    )
    trainer.densify_mv_edge_top_scores.copy_(
        torch.arange(36, dtype=torch.float64).reshape(3, 6, 2) + 1.0
    )
    trainer.densify_mv_edge_top_centroids.copy_(
        torch.arange(72, dtype=torch.float64).reshape(3, 6, 2, 2) + 1.0
    )
    trainer.densify_mv_edge_vote_counts.copy_(
        torch.arange(18, dtype=torch.int32).reshape(3, 6) + 1
    )
    original = {
        "scores": trainer.densify_mv_edge_top_scores.clone(),
        "centroids": trainer.densify_mv_edge_top_centroids.clone(),
        "votes": trainer.densify_mv_edge_vote_counts.clone(),
    }
    row_names = (
        "means3d",
        "log_scales",
        "rotation_quats",
        "colors",
        "color_grads",
        "color_quads",
        "view_grads",
        "specular_ks",
        "specular_log_s",
        "opacity_logits",
        "log_softness",
    )
    rows = {
        name: getattr(trainer.model, name).detach()[0:1].clone()
        for name in row_names
    }

    trainer._append_primitives(rows, birth_iteration=10)

    for name, expected in original.items():
        buffer = getattr(trainer, f"densify_mv_edge_top_{name}", None)
        if name == "votes":
            buffer = trainer.densify_mv_edge_vote_counts
        assert torch.equal(buffer[:3], expected)
        assert torch.count_nonzero(buffer[3]) == 0

    trainer._replace_primitive_rows(
        torch.tensor([1]),
        rows,
        birth_iteration=11,
    )
    for buffer in (
        trainer.densify_mv_edge_top_scores,
        trainer.densify_mv_edge_top_centroids,
        trainer.densify_mv_edge_vote_counts,
    ):
        assert torch.count_nonzero(buffer[1]) == 0

    before_prune = (
        trainer.densify_mv_edge_top_scores.clone(),
        trainer.densify_mv_edge_top_centroids.clone(),
        trainer.densify_mv_edge_vote_counts.clone(),
    )
    keep_mask = torch.tensor([False, True, True, True])
    trainer._prune_primitives(~keep_mask)
    for buffer, expected in zip(
        (
            trainer.densify_mv_edge_top_scores,
            trainer.densify_mv_edge_top_centroids,
            trainer.densify_mv_edge_vote_counts,
        ),
        before_prune,
    ):
        assert torch.equal(buffer, expected[keep_mask])

    trainer._deactivate_primitives(torch.tensor([False, True, False]))
    for buffer in (
        trainer.densify_mv_edge_top_scores,
        trainer.densify_mv_edge_top_centroids,
        trainer.densify_mv_edge_vote_counts,
    ):
        assert torch.count_nonzero(buffer[1]) == 0
        buffer.fill_(1)
    trainer.densify_moment_camera_names = {"cam0", "cam1"}

    trainer._reset_densify_stats()

    for buffer in (
        trainer.densify_mv_edge_top_scores,
        trainer.densify_mv_edge_top_centroids,
        trainer.densify_mv_edge_vote_counts,
    ):
        assert torch.count_nonzero(buffer) == 0
    assert trainer.densify_moment_camera_names == set()


def test_attributed_buffers_follow_primitive_lifecycle() -> None:
    trainer = _make_densify_trainer(num_primitives=3, cap_max=4)
    _enable_attributed_densification(trainer)
    trainer.densify_attributed_edge_mass.copy_(
        torch.arange(18, dtype=torch.float64).reshape(3, 6) + 1.0
    )
    trainer.densify_attributed_edge_centroid_mass.copy_(
        torch.arange(36, dtype=torch.float64).reshape(3, 6, 2) + 1.0
    )
    trainer.densify_attributed_edge_vote_counts.copy_(
        torch.arange(18, dtype=torch.int32).reshape(3, 6) + 1
    )
    original = tuple(
        buffer.clone()
        for buffer in (
            trainer.densify_attributed_edge_mass,
            trainer.densify_attributed_edge_centroid_mass,
            trainer.densify_attributed_edge_vote_counts,
        )
    )
    row_names = (
        "means3d",
        "log_scales",
        "rotation_quats",
        "colors",
        "color_grads",
        "color_quads",
        "view_grads",
        "specular_ks",
        "specular_log_s",
        "opacity_logits",
        "log_softness",
    )
    rows = {
        name: getattr(trainer.model, name).detach()[0:1].clone()
        for name in row_names
    }

    trainer._append_primitives(rows, birth_iteration=10)

    for buffer, expected in zip(
        (
            trainer.densify_attributed_edge_mass,
            trainer.densify_attributed_edge_centroid_mass,
            trainer.densify_attributed_edge_vote_counts,
        ),
        original,
    ):
        assert torch.equal(buffer[:3], expected)
        assert torch.count_nonzero(buffer[3]) == 0

    trainer._replace_primitive_rows(
        torch.tensor([1]),
        rows,
        birth_iteration=11,
    )
    for buffer in (
        trainer.densify_attributed_edge_mass,
        trainer.densify_attributed_edge_centroid_mass,
        trainer.densify_attributed_edge_vote_counts,
    ):
        assert torch.count_nonzero(buffer[1]) == 0

    before_prune = tuple(
        buffer.clone()
        for buffer in (
            trainer.densify_attributed_edge_mass,
            trainer.densify_attributed_edge_centroid_mass,
            trainer.densify_attributed_edge_vote_counts,
        )
    )
    keep_mask = torch.tensor([False, True, True, True])
    trainer._prune_primitives(~keep_mask)
    for buffer, expected in zip(
        (
            trainer.densify_attributed_edge_mass,
            trainer.densify_attributed_edge_centroid_mass,
            trainer.densify_attributed_edge_vote_counts,
        ),
        before_prune,
    ):
        assert torch.equal(buffer, expected[keep_mask])

    trainer._deactivate_primitives(torch.tensor([False, True, False]))
    for buffer in (
        trainer.densify_attributed_edge_mass,
        trainer.densify_attributed_edge_centroid_mass,
        trainer.densify_attributed_edge_vote_counts,
    ):
        assert torch.count_nonzero(buffer[1]) == 0
        buffer.fill_(1)
    trainer.densify_moment_camera_names = {"cam0", "cam1"}

    trainer._reset_densify_stats()

    for buffer in (
        trainer.densify_attributed_edge_mass,
        trainer.densify_attributed_edge_centroid_mass,
        trainer.densify_attributed_edge_vote_counts,
    ):
        assert torch.count_nonzero(buffer) == 0
    assert trainer.densify_moment_camera_names == set()


def test_covariance_buffers_follow_primitive_lifecycle(tmp_path: Path) -> None:
    trainer = _make_densify_trainer(num_primitives=3, cap_max=5)
    _enable_covariance_densification(trainer, tmp_path)
    buffer_names = (
        "densify_covariance_coverage",
        "densify_covariance_energy",
        "densify_covariance_robust_weight",
        "densify_covariance_first_moment",
        "densify_covariance_second_moment",
        "densify_covariance_double_angle",
        "densify_covariance_side_mass",
        "densify_covariance_material_view_count",
        "densify_covariance_axis_view_count",
    )
    for name in buffer_names:
        getattr(trainer, name).fill_(1)
    row_names = (
        "means3d",
        "log_scales",
        "rotation_quats",
        "colors",
        "color_grads",
        "color_quads",
        "view_grads",
        "specular_ks",
        "specular_log_s",
        "opacity_logits",
        "log_softness",
    )
    rows = {
        name: getattr(trainer.model, name).detach()[0:1].clone()
        for name in row_names
    }

    trainer._append_primitives(rows, birth_iteration=10)

    for name in buffer_names:
        buffer = getattr(trainer, name)
        assert buffer.shape[0] == 4
        assert torch.count_nonzero(buffer[3]) == 0

    trainer._replace_primitive_rows(torch.tensor([1]), rows, birth_iteration=11)

    for name in buffer_names:
        assert torch.count_nonzero(getattr(trainer, name)[1]) == 0

    before_prune = {name: getattr(trainer, name).clone() for name in buffer_names}
    keep_mask = torch.tensor([False, True, True, True])
    trainer._prune_primitives(~keep_mask)

    for name in buffer_names:
        assert torch.equal(getattr(trainer, name), before_prune[name][keep_mask])

    trainer._deactivate_primitives(torch.tensor([False, True, False]))
    for name in buffer_names:
        buffer = getattr(trainer, name)
        assert torch.count_nonzero(buffer[1]) == 0
        buffer.fill_(1)
    trainer.densify_moment_camera_names = {"cam0", "cam1"}

    trainer._reset_densify_stats()

    for name in buffer_names:
        assert torch.count_nonzero(getattr(trainer, name)) == 0
    assert trainer.densify_moment_camera_names == set()


def test_pruning_preserves_optimizer_hyperparameters_and_slices_state() -> None:
    trainer = _make_densify_trainer(num_primitives=3, cap_max=3)
    loss = sum(group["params"][0].sum() for group in trainer.optimizer.param_groups)
    loss.backward()
    trainer.optimizer.step()
    trainer.optimizer.zero_grad(set_to_none=True)

    for index, group in enumerate(trainer.optimizer.param_groups):
        group["lr"] = 1e-5 * (index + 1)
        group["betas"] = (0.7, 0.8)
        group["eps"] = 1e-6
        group["weight_decay"] = 0.03

    old_optimizer = trainer.optimizer
    old_group_options = [
        {key: value for key, value in group.items() if key != "params"}
        for group in old_optimizer.param_groups
    ]
    old_states = [
        {
            key: value.clone() if isinstance(value, torch.Tensor) else value
            for key, value in old_optimizer.state[group["params"][0]].items()
        }
        for group in old_optimizer.param_groups
    ]
    keep_mask = torch.tensor([True, False, True])
    trainer.model.active_mask.copy_(torch.tensor([False, True, True]))
    trainer.densify_has_spawned_child.copy_(torch.tensor([True, False, True]))

    trainer._prune_primitives(~keep_mask)

    assert trainer.model.num_base == 2
    assert torch.equal(trainer.model.active_mask, torch.tensor([False, True]))
    assert torch.equal(trainer.densify_has_spawned_child, torch.tensor([True, True]))
    for group, expected_options, old_state in zip(
        trainer.optimizer.param_groups,
        old_group_options,
        old_states,
    ):
        assert {key: value for key, value in group.items() if key != "params"} == expected_options
        new_state = trainer.optimizer.state[group["params"][0]]
        for key, old_value in old_state.items():
            if isinstance(old_value, torch.Tensor) and old_value.ndim > 0:
                assert torch.equal(new_state[key], old_value[keep_mask])
            elif isinstance(old_value, torch.Tensor):
                assert torch.equal(new_state[key], old_value)
            else:
                assert new_state[key] == old_value


def _enable_joint_coplanar_densification(
    trainer: Hexels3DTrainer,
    *,
    steps: int = 2,
) -> None:
    trainer.opt.densify_action = "edge_child"
    trainer.opt.densify_joint_coplanar_steps = steps
    trainer.opt.densify_shadow_steps = 0
    trainer.opt.densify_child_parent_opacity_fraction = 0.25
    trainer.densify_joint_coplanar_event = None
    trainer.densify_joint_coplanar_last_completed = None


def test_joint_coplanar_default_is_off() -> None:
    opt = Hexels3DOptions()
    assert opt.densify_joint_coplanar_steps == 0

    trainer = _make_densify_trainer(num_primitives=2, cap_max=2)
    trainer._start_densify_joint_coplanar_event(
        torch.tensor([0]),
        torch.tensor([1]),
        iteration=10,
    )

    assert trainer.densify_joint_coplanar_event is None


def test_joint_coplanar_birth_tracks_recycled_and_appended_rows_until_clear() -> None:
    trainer = _make_densify_trainer(num_primitives=5, cap_max=7)
    _enable_joint_coplanar_densification(trainer)
    trainer.model.active_mask[1] = False
    trainer.densify_local_grad_abs_accum = torch.ones((5, 2))
    trainer.densify_local_grad_signed_accum = torch.ones((5, 2))

    trainer._edge_child_primitives(torch.tensor([3, 4]), iteration=100)

    event = trainer.densify_joint_coplanar_event
    assert event is not None
    assert torch.equal(event.parent_ids, torch.tensor([3, 4]))
    assert torch.equal(event.child_ids, torch.tensor([1, 5]))
    assert event.birth_iteration == 100
    assert trainer.model.num_base == 6
    audit = trainer._densify_joint_coplanar_audit()
    assert audit["pending"]["parent_ids"] == [3, 4]
    assert audit["pending"]["child_ids"] == [1, 5]
    with pytest.raises(RuntimeError, match="Only one joint coplanar"):
        trainer._edge_child_primitives(torch.tensor([3]), iteration=101)
    assert trainer.model.num_base == 6
    with pytest.raises(RuntimeError, match="blocked while joint coplanar probation"):
        trainer._deactivate_primitives(torch.tensor([False, True, False, False, False, False]))

    trainer._step_optimizer_and_density(
        iteration=101,
        run_densify=False,
        size_threshold=0,
        reset_opacity=False,
    )
    assert trainer.densify_joint_coplanar_event is event
    assert event.steps_completed == 1

    trainer._step_optimizer_and_density(
        iteration=102,
        run_densify=False,
        size_threshold=0,
        reset_opacity=False,
    )
    assert trainer.densify_joint_coplanar_event is None
    completed = trainer._densify_joint_coplanar_audit()["last_completed"]
    assert completed["mapping_count"] == 2
    assert completed["unique_parent_count"] == 2
    assert completed["steps_completed"] == completed["configured_steps"] == 2
    assert completed["first_constrained_iteration"] == 101
    assert completed["last_constrained_iteration"] == 102
    assert completed["expected_first_constrained_iteration"] == 101
    assert completed["expected_last_constrained_iteration"] == 102
    assert completed["exact_constrained_step_count"] is True


def test_joint_coplanar_completed_audit_persists_in_final_checkpoint(
    tmp_path: Path,
) -> None:
    trainer = _make_densify_trainer(num_primitives=3, cap_max=3)
    _enable_joint_coplanar_densification(trainer, steps=2)
    trainer._start_densify_joint_coplanar_event(
        torch.tensor([0]),
        torch.tensor([2]),
        iteration=10,
    )
    for iteration in (11, 12):
        trainer._step_optimizer_and_density(
            iteration=iteration,
            run_densify=False,
            size_threshold=0,
            reset_opacity=False,
        )

    audit = trainer._densify_joint_coplanar_audit()
    assert audit["pending"] is None
    assert audit["last_completed"]["parent_ids"] == [0]
    assert audit["last_completed"]["child_ids"] == [2]
    assert audit["last_completed"]["exact_constrained_step_count"] is True
    json.dumps(audit)

    trainer.model_path = tmp_path
    (tmp_path / "checkpoints").mkdir()
    trainer.resume_checkpoint_edge_transfer = None
    trainer._save_checkpoint(12)
    checkpoint = torch.load(
        tmp_path / "checkpoints" / "ckpt_000012.pt",
        map_location="cpu",
        weights_only=True,
    )

    assert checkpoint["densification_joint_coplanar_audit"] == audit


def test_joint_coplanar_completion_clears_newborn_mean_moments_before_release() -> None:
    trainer = _make_densify_trainer(num_primitives=4, cap_max=4)
    _enable_joint_coplanar_densification(trainer, steps=1)
    trainer._start_densify_joint_coplanar_event(
        torch.tensor([1]),
        torch.tensor([3]),
        iteration=10,
    )
    trainer.model.means3d.grad = torch.ones_like(trainer.model.means3d)

    trainer._step_optimizer_and_density(
        iteration=11,
        run_densify=False,
        size_threshold=0,
        reset_opacity=False,
    )

    state = trainer.optimizer.state[trainer.model.means3d]
    assert trainer.densify_joint_coplanar_event is None
    assert torch.count_nonzero(state["exp_avg"][3]) == 0
    assert torch.count_nonzero(state["exp_avg_sq"][3]) == 0
    assert torch.count_nonzero(state["exp_avg"][1]) > 0
    assert torch.count_nonzero(state["exp_avg_sq"][1]) > 0

    child_before_release = trainer.model.means3d.detach()[3].clone()
    trainer.optimizer.zero_grad(set_to_none=True)
    trainer.model.means3d.grad = torch.zeros_like(trainer.model.means3d)
    trainer._step_optimizer_and_density(
        iteration=12,
        run_densify=False,
        size_threshold=0,
        reset_opacity=False,
    )

    assert torch.equal(trainer.model.means3d.detach()[3], child_before_release)


def test_joint_coplanar_covariance_pair_keeps_repeated_parent_mapping(
    tmp_path: Path,
) -> None:
    trainer = _make_densify_trainer(num_primitives=3, cap_max=5)
    _enable_covariance_densification(
        trainer,
        tmp_path,
        children_per_parent=2,
    )
    trainer.opt.densify_joint_coplanar_steps = 2
    _seed_covariance_pair(trainer, 2)

    trainer._covariance_child_primitives(
        torch.tensor([2]),
        iteration=100,
        eligible_candidate_count=1,
    )

    event = trainer.densify_joint_coplanar_event
    assert event is not None
    assert torch.equal(event.parent_ids, torch.tensor([2, 2]))
    assert torch.equal(event.child_ids, torch.tensor([3, 4]))


def test_joint_coplanar_event_rejects_invalid_parent_child_mapping() -> None:
    trainer = _make_densify_trainer(num_primitives=3, cap_max=3)
    _enable_joint_coplanar_densification(trainer)

    with pytest.raises(ValueError, match="positive parent-child mapping"):
        trainer._start_densify_joint_coplanar_event(
            torch.empty((0,), dtype=torch.long),
            torch.empty((0,), dtype=torch.long),
            iteration=10,
        )
    with pytest.raises(ValueError, match="must be distinct"):
        trainer._start_densify_joint_coplanar_event(
            torch.tensor([1]),
            torch.tensor([1]),
            iteration=10,
        )
    trainer.model.active_mask[2] = False
    with pytest.raises(ValueError, match="must be active"):
        trainer._start_densify_joint_coplanar_event(
            torch.tensor([0]),
            torch.tensor([2]),
            iteration=10,
        )


def test_joint_coplanar_probation_constrains_only_newborn_gradients() -> None:
    trainer = _make_densify_trainer(num_primitives=5, cap_max=5)
    _enable_joint_coplanar_densification(trainer)
    trainer._start_densify_joint_coplanar_event(
        torch.tensor([1, 1]),
        torch.tensor([3, 4]),
        iteration=10,
    )
    trainer.model.means3d.grad = torch.arange(15, dtype=torch.float32).reshape(5, 3)
    trainer.model.rotation_quats.grad = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    trainer.model.view_grads.grad = torch.arange(45, dtype=torch.float32).reshape(5, 3, 3)
    untouched_ids = torch.tensor([0, 1, 2])
    untouched = {
        "means3d": trainer.model.means3d.grad[untouched_ids].clone(),
        "rotation_quats": trainer.model.rotation_quats.grad[untouched_ids].clone(),
        "view_grads": trainer.model.view_grads.grad[untouched_ids].clone(),
    }

    trainer._constrain_densify_joint_coplanar_gradients()

    assert torch.equal(
        trainer.model.means3d.grad[3:5, 2],
        torch.zeros(2),
    )
    assert torch.count_nonzero(trainer.model.rotation_quats.grad[3:5]) == 0
    assert torch.count_nonzero(trainer.model.view_grads.grad[3:5]) == 0
    for name, expected in untouched.items():
        assert torch.equal(getattr(trainer.model, name).grad[untouched_ids], expected)


def test_joint_coplanar_step_jointly_updates_and_reconciles_newborn_rows() -> None:
    trainer = _make_densify_trainer(num_primitives=4, cap_max=4)
    _enable_joint_coplanar_densification(trainer)
    trainer._start_densify_joint_coplanar_event(
        torch.tensor([1]),
        torch.tensor([3]),
        iteration=10,
    )
    root_half = 2.0**-0.5
    with torch.no_grad():
        trainer.model.means3d[1] = torch.tensor([1.0, 2.0, 3.0])
        trainer.model.means3d[3] = torch.tensor([4.0, 7.0, 9.0])
        trainer.model.rotation_quats[1] = torch.tensor([root_half, root_half, 0.0, 0.0])
        trainer.model.rotation_quats[3] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        trainer.model.view_grads[1] = torch.arange(9, dtype=torch.float32).reshape(3, 3)
        trainer.model.view_grads[3].fill_(-1.0)
    parent_mean_before = trainer.model.means3d.detach()[1].clone()
    child_color_before = trainer.model.colors.detach()[3].clone()
    trainer.model.means3d.grad = torch.zeros_like(trainer.model.means3d)
    trainer.model.means3d.grad[1, 1] = 1.0
    trainer.model.means3d.grad[3] = torch.tensor([1.0, 2.0, 3.0])
    trainer.model.rotation_quats.grad = torch.zeros_like(trainer.model.rotation_quats)
    trainer.model.rotation_quats.grad[1, 2] = 1.0
    trainer.model.rotation_quats.grad[3] = 1.0
    trainer.model.view_grads.grad = torch.zeros_like(trainer.model.view_grads)
    trainer.model.view_grads.grad[1] = 1.0
    trainer.model.view_grads.grad[3] = 2.0
    trainer.model.colors.grad = torch.zeros_like(trainer.model.colors)
    trainer.model.colors.grad[3] = 1.0

    trainer._step_optimizer_and_density(
        iteration=11,
        run_densify=False,
        size_threshold=0,
        reset_opacity=False,
    )

    assert not torch.equal(trainer.model.means3d.detach()[1], parent_mean_before)
    assert not torch.equal(trainer.model.colors.detach()[3], child_color_before)
    parent_normal = _quat_to_normal_z(trainer.model.rotations.detach()[1:2])[0]
    child_offset = trainer.model.means3d[3] - trainer.model.means3d[1]
    assert torch.abs(child_offset @ parent_normal) < 1e-5
    assert torch.equal(trainer.model.rotation_quats[3], trainer.model.rotation_quats[1])
    assert torch.equal(trainer.model.view_grads[3], trainer.model.view_grads[1])
    assert trainer.densify_joint_coplanar_event.steps_completed == 1


def _enable_shadow_densification(
    trainer: Hexels3DTrainer,
    *,
    steps: int = 2,
    permanent_residual: bool = True,
) -> None:
    trainer.opt.densify_action = "edge_child"
    trainer.opt.densify_shadow_steps = steps
    trainer.opt.densify_shadow_child_probation = "unrestricted"
    trainer.opt.densify_shadow_permanent_residual = permanent_residual
    trainer.opt.densify_shadow_gate_stride = 0
    trainer.opt.densify_shadow_parent_view_anchor = False
    trainer.opt.densify_shadow_salvage_groups = 0
    trainer.opt.densify_shadow_salvage_selector_stride = 0
    trainer.opt.densify_shadow_defer_inner_gate = False
    trainer.opt.densify_shadow_salvage_allow_partial_cohort = False
    trainer.opt.densify_shadow_min_pooled_sse_reduction = 0.005
    trainer.opt.densify_shadow_min_mean_psnr_delta = 0.02
    trainer.opt.densify_shadow_min_win_fraction = 0.6
    trainer.opt.densify_shadow_min_worst_psnr_delta = -0.15
    trainer.opt.densify_repeat_parent = False
    trainer.opt.densify_prune_interval = 0
    trainer.opt.iterations = 1_000
    trainer.densify_shadow_child_mask = torch.zeros(trainer.model.num_base, dtype=torch.bool)
    trainer.densify_shadow_attempted_parent_mask = torch.zeros(
        trainer.model.num_base, dtype=torch.bool
    )
    trainer.densify_shadow_event = None
    trainer.densify_shadow_counts = {
        "proposed_events": 0,
        "accepted_events": 0,
        "rejected_events": 0,
        "proposed_children": 0,
        "accepted_children": 0,
        "rejected_children": 0,
    }
    trainer.densify_shadow_last_decision = None
    trainer.densify_shadow_decisions = []
    trainer.densify_shadow_gate_cameras = []
    trainer.densify_shadow_gate_camera_names = set()
    trainer.densify_shadow_fit_cameras = []
    trainer.densify_shadow_fit_camera_names = set()
    trainer.densify_shadow_proposal_cameras = []
    trainer.densify_shadow_proposal_camera_names = set()
    trainer.densify_shadow_selector_cameras = []
    trainer.densify_shadow_selector_camera_names = set()


def _enable_shadow_birth_envelope(trainer: Hexels3DTrainer) -> None:
    _enable_shadow_densification(trainer)
    trainer.opt.densify_shadow_enforce_birth_envelope = True
    trainer.opt.densify_child_extent_cap_fraction = 0.3
    trainer.opt.densify_child_parent_opacity_fraction = 0.25
    trainer.opt.densify_child_scale_ratio = 0.5


def test_shadow_birth_envelope_snapshots_bounded_child_state() -> None:
    trainer = _make_densify_trainer(num_primitives=3, cap_max=4)
    _enable_shadow_birth_envelope(trainer)
    trainer.scene_extent = 2.0
    trainer.opt.densify_child_extent_cap_fraction = 0.1
    trainer.opt.densify_child_offset_fraction = 1.0
    trainer.opt.densify_child_init_opacity = 0.05
    trainer.densify_local_grad_abs_accum = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [4.0, 0.5]]
    )
    trainer.densify_local_grad_signed_accum = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [3.0, 0.0]]
    )
    trainer.densify_birth_iteration = torch.zeros(3, dtype=torch.long)

    trainer._edge_child_primitives(torch.tensor([2]), iteration=100)

    event = trainer.densify_shadow_event
    assert event is not None
    assert torch.equal(event.parent_ids, torch.tensor([2]))
    assert torch.equal(event.child_ids, torch.tensor([3]))
    assert torch.allclose(event.birth_parent_distances, torch.tensor([0.2]))
    assert torch.allclose(event.birth_physical_max_scales, torch.tensor([0.2]))
    assert torch.allclose(torch.exp(event.birth_log_opacities), torch.tensor([0.025]))
    audit = trainer._densify_shadow_checkpoint_audit()["pending"]["birth_envelope"]
    assert torch.equal(audit["birth_parent_distances"], event.birth_parent_distances)
    assert audit["projection_steps"] == 0


def test_shadow_birth_envelope_projects_pending_rows_and_clears_only_matching_adam() -> None:
    trainer = _make_densify_trainer(num_primitives=5, cap_max=5)
    _enable_shadow_birth_envelope(trainer)
    loss = sum(group["params"][0].sum() for group in trainer.optimizer.param_groups)
    loss.backward()
    trainer.optimizer.step()
    trainer.optimizer.zero_grad(set_to_none=True)
    for parameter in (
        trainer.model.means3d,
        trainer.model.log_scales,
        trainer.model.opacity_logits,
    ):
        state = trainer.optimizer.state[parameter]
        state["exp_avg"].fill_(1.0)
        state["exp_avg_sq"].fill_(1.0)

    parent_ids = torch.tensor([0, 1])
    child_ids = torch.tensor([2, 3])
    with torch.no_grad():
        trainer.model.means3d[parent_ids] = torch.tensor(
            [[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]]
        )
        trainer.model.means3d[child_ids] = torch.tensor(
            [[11.0, 0.0, 0.0], [22.0, 0.0, 0.0]]
        )
        trainer.model.log_scales[parent_ids] = torch.log(
            torch.tensor([[0.4, 0.2], [1.0, 0.5]])
        )
        trainer.model.log_scales[child_ids] = torch.log(
            torch.tensor([[1.0, 0.5], [1.0, 0.25]])
        )
        trainer.model.opacity_logits[parent_ids] = torch.logit(torch.tensor([0.2, 0.2]))
        trainer.model.opacity_logits[child_ids] = torch.logit(torch.tensor([0.5, 0.5]))
    untouched_ids = torch.tensor([0, 1, 4])
    untouched = {
        name: getattr(trainer.model, name).detach()[untouched_ids].clone()
        for name in ("means3d", "log_scales", "opacity_logits")
    }
    trainer.densify_shadow_event = _DensifyShadowEvent(
        parent_ids=parent_ids,
        child_ids=child_ids,
        birth_iteration=10,
        birth_parent_distances=torch.tensor([0.25, 0.5]),
        birth_physical_max_scales=torch.tensor([0.4, 0.4]),
        birth_log_opacities=torch.log(torch.tensor([0.08, 0.04])),
    )

    trainer._project_densify_shadow_birth_envelope()

    assert torch.allclose(
        trainer.model.means3d[child_ids],
        torch.tensor([[10.25, 0.0, 0.0], [20.5, 0.0, 0.0]]),
    )
    assert torch.allclose(
        trainer.model.scales[child_ids],
        torch.tensor([[0.2, 0.1], [0.3, 0.075]]),
    )
    assert torch.allclose(trainer.model.opacities[child_ids], torch.tensor([0.05, 0.04]))
    for name, expected in untouched.items():
        parameter = getattr(trainer.model, name)
        assert torch.equal(parameter.detach()[untouched_ids], expected)
        state = trainer.optimizer.state[parameter]
        assert torch.count_nonzero(state["exp_avg"][child_ids]) == 0
        assert torch.count_nonzero(state["exp_avg_sq"][child_ids]) == 0
        assert torch.equal(state["exp_avg"][untouched_ids], torch.ones_like(expected))
        assert torch.equal(state["exp_avg_sq"][untouched_ids], torch.ones_like(expected))
    event = trainer.densify_shadow_event
    assert event is not None
    assert event.envelope_projection_steps == 1
    assert event.envelope_projected_mean_rows == 2
    assert event.envelope_projected_scale_rows == 2
    assert event.envelope_projected_opacity_rows == 2


def test_shadow_birth_envelope_projects_before_probation_decision() -> None:
    trainer = _make_densify_trainer(num_primitives=3, cap_max=3)
    _enable_shadow_birth_envelope(trainer)
    trainer.opt.densify_shadow_steps = 1
    trainer.densify_shadow_event = _DensifyShadowEvent(
        parent_ids=torch.tensor([0]),
        child_ids=torch.tensor([2]),
        birth_iteration=10,
        birth_parent_distances=torch.tensor([0.25]),
        birth_physical_max_scales=torch.tensor([0.3]),
        birth_log_opacities=torch.log(torch.tensor([0.05])),
    )
    with torch.no_grad():
        trainer.model.means3d[2] = torch.tensor([1.0, 0.0, 0.0])
    checked: list[int] = []

    def finish(*, iteration: int) -> None:
        event = trainer.densify_shadow_event
        assert event is not None
        assert event.steps_completed == 1
        assert event.envelope_projection_steps == 1
        distance = torch.linalg.vector_norm(
            trainer.model.means3d[2] - trainer.model.means3d[0]
        )
        assert distance <= event.birth_parent_distances[0]
        checked.append(iteration)
        trainer.densify_shadow_event = None

    trainer._finish_densify_shadow_event = finish
    trainer._step_optimizer_and_density(
        iteration=11,
        run_densify=False,
        size_threshold=0,
        reset_opacity=False,
    )

    assert checked == [11]


def test_shadow_birth_envelope_default_allows_events_without_bounds() -> None:
    trainer = _make_densify_trainer(num_primitives=3, cap_max=3)
    _enable_shadow_densification(trainer, steps=2)
    trainer.opt.densify_shadow_enforce_birth_envelope = False
    trainer.densify_shadow_event = _DensifyShadowEvent(
        parent_ids=torch.tensor([0]),
        child_ids=torch.tensor([2]),
        birth_iteration=10,
    )

    trainer._step_optimizer_and_density(
        iteration=11,
        run_densify=False,
        size_threshold=0,
        reset_opacity=False,
    )

    assert trainer.densify_shadow_event.steps_completed == 1


def test_shadow_coplanar_spatial_probation_constrains_pending_child_gradients() -> None:
    trainer = _make_densify_trainer(num_primitives=4, cap_max=4)
    _enable_shadow_densification(trainer)
    trainer.opt.densify_shadow_child_probation = "coplanar_spatial"
    trainer.densify_shadow_event = _DensifyShadowEvent(
        parent_ids=torch.tensor([1]),
        child_ids=torch.tensor([3]),
        birth_iteration=10,
    )
    trainer.model.means3d.grad = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ]
    )
    trainer.model.rotation_quats.grad = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    trainer.model.view_grads.grad = torch.arange(36, dtype=torch.float32).reshape(4, 3, 3)
    untouched_ids = torch.tensor([0, 1, 2])
    untouched = {
        "means3d": trainer.model.means3d.grad[untouched_ids].clone(),
        "rotation_quats": trainer.model.rotation_quats.grad[untouched_ids].clone(),
        "view_grads": trainer.model.view_grads.grad[untouched_ids].clone(),
    }

    trainer._constrain_densify_shadow_child_gradients()

    assert torch.equal(trainer.model.means3d.grad[3], torch.tensor([10.0, 11.0, 0.0]))
    assert torch.count_nonzero(trainer.model.rotation_quats.grad[3]) == 0
    assert torch.count_nonzero(trainer.model.view_grads.grad[3]) == 0
    for name, expected in untouched.items():
        assert torch.equal(getattr(trainer.model, name).grad[untouched_ids], expected)


def test_shadow_coplanar_spatial_probation_projects_and_ties_after_step() -> None:
    trainer = _make_densify_trainer(num_primitives=4, cap_max=4)
    _enable_shadow_densification(trainer)
    trainer.opt.densify_shadow_child_probation = "coplanar_spatial"
    trainer.densify_shadow_event = _DensifyShadowEvent(
        parent_ids=torch.tensor([1]),
        child_ids=torch.tensor([3]),
        birth_iteration=10,
    )
    root_half = 2.0**-0.5
    with torch.no_grad():
        trainer.model.means3d[1] = torch.tensor([1.0, 2.0, 3.0])
        trainer.model.means3d[3] = torch.tensor([4.0, 7.0, 9.0])
        trainer.model.rotation_quats[1] = torch.tensor([root_half, root_half, 0.0, 0.0])
        trainer.model.rotation_quats[3] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        trainer.model.view_grads[1] = torch.arange(9, dtype=torch.float32).reshape(3, 3)
        trainer.model.view_grads[3].fill_(-1.0)
    parent_before = {
        name: getattr(trainer.model, name).detach()[1].clone()
        for name in ("means3d", "rotation_quats", "view_grads")
    }
    trainer.model.means3d.grad = torch.zeros_like(trainer.model.means3d)
    trainer.model.means3d.grad[1, 1] = 1.0
    trainer.model.rotation_quats.grad = torch.zeros_like(trainer.model.rotation_quats)
    trainer.model.rotation_quats.grad[1, 2] = 1.0
    trainer.model.view_grads.grad = torch.zeros_like(trainer.model.view_grads)
    trainer.model.view_grads.grad[1] = 1.0

    trainer._step_optimizer_and_density(
        iteration=11,
        run_densify=False,
        size_threshold=0,
        reset_opacity=False,
    )

    for name, value in parent_before.items():
        assert not torch.equal(getattr(trainer.model, name).detach()[1], value)
    parent_normal = _quat_to_normal_z(trainer.model.rotations.detach()[1:2])[0]
    child_offset = trainer.model.means3d[3] - trainer.model.means3d[1]
    assert torch.abs(child_offset @ parent_normal) < 1e-5
    assert torch.equal(trainer.model.rotation_quats[3], trainer.model.rotation_quats[1])
    assert torch.equal(trainer.model.view_grads[3], trainer.model.view_grads[1])
    assert trainer.densify_shadow_event.steps_completed == 1


def test_shadow_unrestricted_probation_leaves_pending_child_rows_exact() -> None:
    trainer = _make_densify_trainer(num_primitives=4, cap_max=4)
    _enable_shadow_densification(trainer)
    assert trainer.opt.densify_shadow_child_probation == "unrestricted"
    trainer.densify_shadow_event = _DensifyShadowEvent(
        parent_ids=torch.tensor([1]),
        child_ids=torch.tensor([3]),
        birth_iteration=10,
    )
    with torch.no_grad():
        trainer.model.means3d[3] = torch.tensor([4.0, 7.0, 9.0])
        trainer.model.rotation_quats[3] = torch.tensor([0.5, 0.5, 0.5, 0.5])
        trainer.model.view_grads[3].fill_(-1.0)
    expected = {
        name: getattr(trainer.model, name).detach()[3].clone()
        for name in ("means3d", "rotation_quats", "view_grads")
    }

    trainer._step_optimizer_and_density(
        iteration=11,
        run_densify=False,
        size_threshold=0,
        reset_opacity=False,
    )

    for name, value in expected.items():
        assert torch.equal(getattr(trainer.model, name).detach()[3], value)


def test_shadow_permanent_residual_matches_base_control_and_updates_all_children() -> None:
    shadow = _make_densify_trainer(num_primitives=4, cap_max=8)
    control = _make_densify_trainer(num_primitives=2, cap_max=8)
    _enable_shadow_densification(shadow, permanent_residual=True)
    shadow.densify_shadow_child_mask.copy_(torch.tensor([False, False, True, True]))
    shadow.opt.lambda_dssim = 0.0
    control.opt.lambda_dssim = 0.0
    shadow.bg_color = torch.zeros(3)
    control.bg_color = torch.zeros(3)
    shadow.edge_transfer_controller = None
    control.edge_transfer_controller = None
    squared_loss = lambda prediction, target: (prediction - target).square().mean()
    shadow.l1_loss = squared_loss
    control.l1_loss = squared_loss
    shadow.ssim = lambda prediction, target: 1.0 - squared_loss(prediction, target)
    control.ssim = shadow.ssim
    parameter_names = shadow._base_optimizer_param_names()

    class Renderer:
        def render(self, camera: object, model: object, background: torch.Tensor, **kwargs: object) -> dict[str, torch.Tensor]:
            active = model.active_mask
            total = sum(
                (index + 1) * 1e-3 * getattr(model, name)[active].sum()
                for index, name in enumerate(parameter_names)
            )
            return {"image": torch.sigmoid(total).expand(3, 1, 1)}

    shadow.renderer = Renderer()
    control.renderer = Renderer()
    target = torch.full((3, 1, 1), 0.25)
    camera = object()
    initial_children = {
        name: getattr(shadow.model, name).detach()[2:].clone()
        for name in parameter_names
    }

    for iteration in (1, 2):
        shadow.optimizer.zero_grad(set_to_none=True)
        proposal = shadow.renderer.render(camera, shadow.model, shadow.bg_color)["image"]
        squared_loss(proposal, target).backward()
        proposal_child_grads = {
            name: getattr(shadow.model, name).grad[2:].clone()
            for name in parameter_names
        }
        shadow._densify_shadow_control_backward(
            camera,
            target,
            shadow.bg_color,
            sh_degree=3,
        )

        control.optimizer.zero_grad(set_to_none=True)
        control_prediction = control.renderer.render(camera, control.model, control.bg_color)["image"]
        squared_loss(control_prediction, target).backward()
        for name in parameter_names:
            assert torch.equal(
                getattr(shadow.model, name).grad[:2],
                getattr(control.model, name).grad,
            )
            assert torch.equal(
                getattr(shadow.model, name).grad[2:],
                proposal_child_grads[name],
            )

        shadow._step_optimizer_and_density(
            iteration=iteration,
            run_densify=False,
            size_threshold=0,
            reset_opacity=False,
        )
        control.optimizer.step()
        for shadow_group, control_group in zip(
            shadow.optimizer.param_groups,
            control.optimizer.param_groups,
        ):
            shadow_parameter = shadow_group["params"][0]
            control_parameter = control_group["params"][0]
            assert torch.equal(shadow_parameter[:2], control_parameter)
            shadow_state = shadow.optimizer.state[shadow_parameter]
            control_state = control.optimizer.state[control_parameter]
            assert torch.equal(shadow_state["exp_avg"][:2], control_state["exp_avg"])
            assert torch.equal(shadow_state["exp_avg_sq"][:2], control_state["exp_avg_sq"])
            assert shadow_state["step"] == control_state["step"] == iteration

    assert any(
        not torch.equal(getattr(shadow.model, name).detach()[2:], initial_children[name])
        for name in parameter_names
    )


def test_shadow_rejection_keeps_prior_child_on_without_pending_trajectory() -> None:
    shadow = _make_densify_trainer(num_primitives=4, cap_max=8)
    base_control = _make_densify_trainer(num_primitives=2, cap_max=8)
    prior_child_control = _make_densify_trainer(num_primitives=3, cap_max=8)
    _enable_shadow_densification(shadow, steps=1, permanent_residual=True)
    shadow.densify_shadow_child_mask.copy_(torch.tensor([False, False, True, True]))
    shadow.densify_shadow_attempted_parent_mask[1] = True
    shadow.densify_shadow_event = _DensifyShadowEvent(
        parent_ids=torch.tensor([1]),
        child_ids=torch.tensor([3]),
        birth_iteration=10,
    )
    shadow.densify_shadow_counts.update(
        {
            "proposed_events": 2,
            "accepted_events": 1,
            "proposed_children": 2,
            "accepted_children": 1,
        }
    )
    for trainer in (shadow, base_control, prior_child_control):
        trainer.opt.lambda_dssim = 0.0
        trainer.bg_color = torch.zeros(3)
        trainer.edge_transfer_controller = None
    squared_loss = lambda prediction, target: (prediction - target).square().mean()
    for trainer in (shadow, base_control, prior_child_control):
        trainer.l1_loss = squared_loss
        trainer.ssim = lambda prediction, target: 1.0 - squared_loss(prediction, target)
    parameter_names = shadow._base_optimizer_param_names()

    class Renderer:
        def render(self, camera: object, model: object, background: torch.Tensor, **kwargs: object) -> dict[str, torch.Tensor]:
            active = model.active_mask
            total = sum(
                (index + 1) * 1e-3 * getattr(model, name)[active].sum()
                for index, name in enumerate(parameter_names)
            )
            return {"image": torch.sigmoid(total).expand(3, 1, 1)}

    for trainer in (shadow, base_control, prior_child_control):
        trainer.renderer = Renderer()
    target = torch.full((3, 1, 1), 0.25)
    camera = object()

    shadow.optimizer.zero_grad(set_to_none=True)
    proposal = shadow.renderer.render(camera, shadow.model, shadow.bg_color)["image"]
    squared_loss(proposal, target).backward()
    pending_gradients = {
        name: getattr(shadow.model, name).grad[3].clone()
        for name in parameter_names
    }
    shadow._densify_shadow_control_backward(
        camera,
        target,
        shadow.bg_color,
        sh_degree=3,
    )

    for trainer in (base_control, prior_child_control):
        trainer.optimizer.zero_grad(set_to_none=True)
        prediction = trainer.renderer.render(camera, trainer.model, trainer.bg_color)["image"]
        squared_loss(prediction, target).backward()
    for name in parameter_names:
        assert torch.equal(
            getattr(shadow.model, name).grad[:2],
            getattr(base_control.model, name).grad,
        )
        assert torch.equal(
            getattr(shadow.model, name).grad[2],
            getattr(prior_child_control.model, name).grad[2],
        )
        assert torch.equal(getattr(shadow.model, name).grad[3], pending_gradients[name])

    shadow.optimizer.step()
    base_control.optimizer.step()
    prior_child_control.optimizer.step()
    prior_child_values = {
        name: getattr(shadow.model, name).detach()[2].clone()
        for name in parameter_names
    }
    shadow.densify_shadow_event.steps_completed = 1
    shadow._evaluate_densify_shadow_event = lambda: {
        "finite": True,
        "accepted": False,
        "num_training_cameras": 1,
        "pooled_sse_reduction": -0.1,
        "mean_psnr_delta": -0.1,
        "win_fraction": 0.0,
        "worst_psnr_delta": -0.1,
    }
    shadow._finish_densify_shadow_event(iteration=11)

    assert shadow.model.num_base == 3
    for name in parameter_names:
        assert torch.equal(getattr(shadow.model, name).detach()[2], prior_child_values[name])
        assert torch.equal(
            getattr(shadow.model, name).detach()[2],
            getattr(prior_child_control.model, name).detach()[2],
        )


def test_shadow_gate_is_conditional_and_uses_only_optimizer_training_cameras() -> None:
    trainer = _make_densify_trainer(num_primitives=4, cap_max=8)
    _enable_shadow_densification(trainer)
    trainer.bg_color = torch.zeros(3)
    trainer.edge_transfer_controller = None
    trainer.densify_shadow_child_mask.copy_(torch.tensor([False, False, True, True]))
    trainer.densify_shadow_event = _DensifyShadowEvent(
        parent_ids=torch.tensor([1]),
        child_ids=torch.tensor([3]),
        birth_iteration=10,
        steps_completed=2,
    )
    trainer.train_cameras = [
        SimpleNamespace(image_name="train_a", original_image=torch.zeros((3, 1, 1))),
        SimpleNamespace(image_name="train_b", original_image=torch.zeros((3, 1, 1))),
    ]
    trainer.test_cameras = [
        SimpleNamespace(image_name="official", original_image=torch.zeros((3, 1, 1)))
    ]
    calls: list[tuple[str, torch.Tensor]] = []

    class Renderer:
        def render(self, camera: object, model: object, background: torch.Tensor, **kwargs: object) -> dict[str, torch.Tensor]:
            calls.append((camera.image_name, model.active_mask.clone()))
            value = 0.0 if bool(model.active_mask[3]) else 0.2
            return {"image": torch.full((3, 1, 1), value)}

    trainer.renderer = Renderer()
    result = trainer._evaluate_densify_shadow_event()

    assert result["accepted"] is True
    assert result["num_training_cameras"] == 2
    assert [name for name, _ in calls] == ["train_a", "train_a", "train_b", "train_b"]
    for _, active in calls[1::2]:
        assert active[2]
        assert not active[3]
    assert torch.all(trainer.model.active_mask)


def test_shadow_gate_uses_reserved_cameras_and_records_worst_views() -> None:
    trainer = _make_densify_trainer(num_primitives=3, cap_max=8)
    _enable_shadow_densification(trainer)
    trainer.opt.densify_shadow_gate_stride = 2
    trainer.bg_color = torch.zeros(3)
    trainer.edge_transfer_controller = None
    trainer.densify_shadow_child_mask.copy_(torch.tensor([False, False, True]))
    trainer.densify_shadow_event = _DensifyShadowEvent(
        parent_ids=torch.tensor([1]),
        child_ids=torch.tensor([2]),
        birth_iteration=10,
        steps_completed=2,
    )
    trainer.train_cameras = [
        SimpleNamespace(image_name="fit", original_image=torch.zeros((3, 1, 1)))
    ]
    trainer.densify_shadow_gate_cameras = [
        SimpleNamespace(image_name="gate_b", original_image=torch.zeros((3, 1, 1))),
        SimpleNamespace(image_name="gate_a", original_image=torch.zeros((3, 1, 1))),
    ]
    trainer.densify_shadow_gate_camera_names = {"gate_a", "gate_b"}
    calls: list[str] = []

    class Renderer:
        def render(self, camera: object, model: object, background: torch.Tensor, **kwargs: object) -> dict[str, torch.Tensor]:
            calls.append(camera.image_name)
            value = 0.0 if bool(model.active_mask[2]) else 0.1
            return {"image": torch.full((3, 1, 1), value)}

    trainer.renderer = Renderer()
    result = trainer._evaluate_densify_shadow_event()

    assert calls == ["gate_b", "gate_b", "gate_a", "gate_a"]
    assert result["num_training_cameras"] == 2
    assert result["decision_camera_names"] == ["gate_b", "gate_a"]
    assert [record["image_name"] for record in result["worst_views"]] == [
        "gate_b",
        "gate_a",
    ]


def test_shadow_gate_camera_supplies_only_base_control_gradients() -> None:
    trainer = _make_densify_trainer(num_primitives=4, cap_max=8)
    _enable_shadow_densification(trainer, permanent_residual=True)
    trainer.opt.lambda_dssim = 0.0
    trainer.bg_color = torch.zeros(3)
    trainer.edge_transfer_controller = None
    trainer.densify_shadow_child_mask.copy_(torch.tensor([False, False, True, True]))
    trainer.densify_shadow_event = _DensifyShadowEvent(
        parent_ids=torch.tensor([1]),
        child_ids=torch.tensor([3]),
        birth_iteration=10,
    )
    trainer.densify_shadow_gate_camera_names = {"gate"}
    trainer.model.colors.data.fill_(0.2)
    squared_loss = lambda prediction, target: (prediction - target).square().mean()
    trainer.l1_loss = squared_loss
    trainer.ssim = lambda prediction, target: 1.0 - squared_loss(prediction, target)

    class Renderer:
        def render(self, camera: object, model: object, background: torch.Tensor, **kwargs: object) -> dict[str, torch.Tensor]:
            return {"image": model.colors[model.active_mask].sum().reshape(1, 1, 1)}

    trainer.renderer = Renderer()
    camera = SimpleNamespace(image_name="gate")
    target = torch.zeros((1, 1, 1))
    proposal = trainer.renderer.render(camera, trainer.model, trainer.bg_color)["image"]
    squared_loss(proposal, target).backward()
    trainer._densify_shadow_control_backward(
        camera,
        target,
        trainer.bg_color,
        sh_degree=0,
    )

    assert torch.count_nonzero(trainer.model.colors.grad[:2]) > 0
    assert torch.count_nonzero(trainer.model.colors.grad[2:]) == 0


def test_shadow_selector_camera_supplies_no_adaptive_child_gradients() -> None:
    trainer = _make_densify_trainer(num_primitives=4, cap_max=8)
    _enable_shadow_densification(trainer, permanent_residual=True)
    trainer.opt.lambda_dssim = 0.0
    trainer.bg_color = torch.zeros(3)
    trainer.edge_transfer_controller = None
    trainer.densify_shadow_child_mask.copy_(torch.tensor([False, False, True, True]))
    trainer.densify_shadow_event = _DensifyShadowEvent(
        parent_ids=torch.tensor([1]),
        child_ids=torch.tensor([3]),
        birth_iteration=10,
    )
    trainer.densify_shadow_selector_camera_names = {"selector"}
    trainer.model.colors.data.fill_(0.2)
    squared_loss = lambda prediction, target: (prediction - target).square().mean()
    trainer.l1_loss = squared_loss
    trainer.ssim = lambda prediction, target: 1.0 - squared_loss(prediction, target)

    class Renderer:
        def render(self, camera: object, model: object, background: torch.Tensor, **kwargs: object) -> dict[str, torch.Tensor]:
            return {"image": model.colors[model.active_mask].sum().reshape(1, 1, 1)}

    trainer.renderer = Renderer()
    camera = SimpleNamespace(image_name="selector")
    target = torch.zeros((1, 1, 1))
    proposal = trainer.renderer.render(camera, trainer.model, trainer.bg_color)["image"]
    squared_loss(proposal, target).backward()
    trainer._densify_shadow_control_backward(
        camera,
        target,
        trainer.bg_color,
        sh_degree=0,
    )

    assert torch.count_nonzero(trainer.model.colors.grad[:2]) > 0
    assert torch.count_nonzero(trainer.model.colors.grad[2:]) == 0


@pytest.mark.parametrize(
    ("held_out_names", "camera_name"),
    [
        ("densify_shadow_selector_camera_names", "selector"),
        ("densify_shadow_gate_camera_names", "inner"),
    ],
)
def test_held_out_shadow_step_preserves_adaptive_rows_and_adam_moments(
    held_out_names: str,
    camera_name: str,
) -> None:
    trainer = _make_densify_trainer(num_primitives=4, cap_max=8)
    _enable_shadow_densification(trainer, steps=10, permanent_residual=True)
    trainer.opt.lambda_dssim = 0.0
    trainer.bg_color = torch.zeros(3)
    trainer.edge_transfer_controller = None
    trainer.densify_shadow_child_mask.copy_(torch.tensor([False, False, True, True]))
    trainer.densify_shadow_event = _DensifyShadowEvent(
        parent_ids=torch.tensor([1]),
        child_ids=torch.tensor([3]),
        birth_iteration=10,
    )
    setattr(trainer, held_out_names, {camera_name})
    trainer.model.colors.data.fill_(0.2)
    squared_loss = lambda prediction, target: (prediction - target).square().mean()
    trainer.l1_loss = squared_loss
    trainer.ssim = lambda prediction, target: 1.0 - squared_loss(prediction, target)

    class Renderer:
        def render(self, camera: object, model: object, background: torch.Tensor, **kwargs: object) -> dict[str, torch.Tensor]:
            return {"image": model.colors[model.active_mask].sum().reshape(1, 1, 1)}

    trainer.renderer = Renderer()
    target = torch.zeros((1, 1, 1))

    def backward(camera: object) -> None:
        trainer.optimizer.zero_grad(set_to_none=True)
        prediction = trainer.renderer.render(camera, trainer.model, trainer.bg_color)["image"]
        squared_loss(prediction, target).backward()
        trainer._densify_shadow_control_backward(
            camera,
            target,
            trainer.bg_color,
            sh_degree=0,
        )

    backward(SimpleNamespace(image_name="proposal"))
    trainer._step_optimizer_and_density(
        iteration=1,
        run_densify=False,
        size_threshold=0,
        reset_opacity=False,
    )
    adaptive_ids = torch.tensor([2, 3])
    parameter_before = trainer.model.colors.detach()[adaptive_ids].clone()
    base_before = trainer.model.colors.detach()[0].clone()
    state = trainer.optimizer.state[trainer.model.colors]
    state_before = {
        key: value.detach()[adaptive_ids].clone()
        for key, value in state.items()
        if isinstance(value, torch.Tensor)
        and value.ndim > 0
        and value.shape[0] == trainer.model.num_base
    }
    assert torch.count_nonzero(state_before["exp_avg"]) > 0

    backward(SimpleNamespace(image_name=camera_name))
    trainer._step_optimizer_and_density(
        iteration=2,
        run_densify=False,
        size_threshold=0,
        reset_opacity=False,
    )

    assert torch.equal(trainer.model.colors.detach()[adaptive_ids], parameter_before)
    assert not torch.equal(trainer.model.colors.detach()[0], base_before)
    for key, rows in state_before.items():
        assert torch.equal(state[key][adaptive_ids], rows)
    assert trainer.densify_shadow_frozen_step_ids is None


def test_shadow_parent_view_anchor_releases_to_standard_inference() -> None:
    trainer = _make_densify_trainer(num_primitives=3, cap_max=8)
    _enable_shadow_densification(trainer, steps=4)
    trainer.opt.densify_shadow_parent_view_anchor = True
    trainer.edge_transfer_controller = None
    trainer.model.means3d.data.copy_(
        torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    )
    trainer.densify_shadow_event = _DensifyShadowEvent(
        parent_ids=torch.tensor([0]),
        child_ids=torch.tensor([2]),
        birth_iteration=10,
    )

    anchored = trainer._render_model().pack_for_rasterizer(
        child_offset_frac=0.0,
        child_scale_ratio=1.0,
    )
    assert torch.equal(anchored["view_anchor3d"][2], trainer.model.means3d[0])

    trainer.model.active_mask[2] = False
    assert trainer._render_model() is trainer.model
    trainer.model.active_mask[2] = True

    trainer.densify_shadow_event.steps_completed = 3
    releasing = trainer._render_model().pack_for_rasterizer(
        child_offset_frac=0.0,
        child_scale_ratio=1.0,
    )
    assert torch.equal(releasing["view_anchor3d"][2], torch.tensor([1.0, 0.0, 0.0]))

    trainer.densify_shadow_event.steps_completed = 4
    released = trainer._render_model().pack_for_rasterizer(
        child_offset_frac=0.0,
        child_scale_ratio=1.0,
    )
    assert "view_anchor3d" not in released


def test_shadow_salvage_groups_are_balanced_and_parent_stable() -> None:
    parents = torch.tensor([9, 1, 7, 3, 5, 11])
    children = torch.tensor([19, 13, 17, 14, 15, 21])
    groups = _balanced_densify_shadow_groups(parents, children, 3)

    assignment = {
        int(parent): group_id
        for group_id, group_parents, _ in groups
        for parent in group_parents
    }
    assert assignment == {1: 0, 3: 1, 5: 2, 7: 0, 9: 1, 11: 2}
    assert [int(group_children.numel()) for _, _, group_children in groups] == [2, 2, 2]

    order = torch.tensor([5, 2, 0, 4, 1, 3])
    permuted = _balanced_densify_shadow_groups(parents[order], children[order], 3)
    permuted_assignment = {
        int(parent): group_id
        for group_id, group_parents, _ in permuted
        for parent in group_parents
    }
    assert permuted_assignment == assignment


def test_shadow_render_comparison_uses_pooled_sse_and_ordered_views() -> None:
    reference = [
        {"image_name": "a", "sse": 12.0, "count": 3},
        {"image_name": "b", "sse": 3.0, "count": 3},
    ]
    candidate = [
        {"image_name": "a", "sse": 3.0, "count": 3},
        {"image_name": "b", "sse": 12.0, "count": 3},
    ]

    comparison = _compare_densify_shadow_renders(reference, candidate)

    assert comparison["pooled_sse_reduction"] == 0.0
    assert comparison["mean_psnr_delta"] == pytest.approx(0.0)
    assert comparison["win_fraction"] == 0.5
    assert comparison["worst_psnr_delta"] == pytest.approx(-10.0 * torch.log10(torch.tensor(4.0)).item())
    with pytest.raises(ValueError, match="camera order differs"):
        _compare_densify_shadow_renders(reference, list(reversed(candidate)))


def test_shadow_salvage_selection_uses_only_fit_cameras() -> None:
    trainer = _make_densify_trainer(num_primitives=4, cap_max=8)
    _enable_shadow_densification(trainer, steps=2)
    trainer.opt.densify_max_add = 2
    trainer.opt.densify_shadow_salvage_groups = 2
    trainer.opt.densify_shadow_min_pooled_sse_reduction = 0.0
    trainer.opt.densify_shadow_min_mean_psnr_delta = 0.0
    trainer.opt.densify_shadow_min_win_fraction = 0.5
    trainer.bg_color = torch.zeros(3)
    trainer.edge_transfer_controller = None
    trainer.densify_shadow_child_mask.copy_(torch.tensor([False, False, True, True]))
    trainer.densify_shadow_event = _DensifyShadowEvent(
        parent_ids=torch.tensor([0, 1]),
        child_ids=torch.tensor([2, 3]),
        birth_iteration=10,
        steps_completed=2,
        residual_topk_evidence={"k": 3, "kth_scores": [2.0, 1.0]},
    )
    trainer.densify_shadow_fit_cameras = [
        SimpleNamespace(image_name="fit_b", original_image=torch.zeros((3, 1, 1))),
        SimpleNamespace(image_name="fit_a", original_image=torch.zeros((3, 1, 1))),
    ]
    trainer.densify_shadow_fit_camera_names = {"fit_a", "fit_b"}
    trainer.densify_shadow_gate_cameras = [
        SimpleNamespace(image_name="gate", original_image=torch.zeros((3, 1, 1)))
    ]
    calls: list[tuple[str, tuple[bool, bool]]] = []

    class Renderer:
        def render(
            self,
            camera: object,
            model: object,
            background: torch.Tensor,
            **kwargs: object,
        ) -> dict[str, torch.Tensor]:
            state = tuple(bool(value) for value in model.active_mask[2:4])
            calls.append((camera.image_name, state))
            value = {
                (False, False): 1.0,
                (True, False): 0.5,
                (False, True): 1.0,
                (True, True): 0.6,
            }[state]
            return {"image": torch.full((3, 1, 1), value)}

    trainer.renderer = Renderer()
    selection, selected_ids = trainer._select_densify_shadow_salvage()

    assert torch.equal(selected_ids, torch.tensor([2]))
    assert selection["chosen_prefix"]["group_ids"] == [0]
    assert selection["chosen_prefix"]["parent_ids"] == [0]
    assert len(selection["chosen_prefix"]["parent_ids_sha256"]) == 64
    assert all("child_state" in record for record in selection["group_records"])
    assert len(selection["child_state"]["child_parameter_rows_sha256"]) == 64
    assert selection["residual_topk_evidence"] == {
        "k": 3,
        "kth_scores": [2.0, 1.0],
    }
    assert set(selection["child_state"]["max_scale"]) == {
        "min",
        "p05",
        "median",
        "p95",
        "max",
    }
    assert all(name in {"fit_a", "fit_b"} for name, _ in calls)
    assert len(calls) == 14
    assert torch.all(trainer.model.active_mask)


def test_nested_shadow_selector_scores_proposal_then_selects_on_selector() -> None:
    trainer = _make_densify_trainer(num_primitives=8, cap_max=12)
    _enable_shadow_densification(trainer, steps=2)
    trainer.opt.densify_max_add = 4
    trainer.opt.densify_shadow_salvage_groups = 2
    trainer.opt.densify_shadow_salvage_selector_stride = 6
    trainer.opt.densify_shadow_salvage_prefix_policy = "first_qualifying"
    trainer.opt.densify_shadow_min_pooled_sse_reduction = 0.0
    trainer.opt.densify_shadow_min_mean_psnr_delta = 0.0
    trainer.opt.densify_shadow_min_win_fraction = 0.5
    trainer.opt.densify_shadow_min_worst_psnr_delta = 0.0
    trainer.densify_shadow_event = _DensifyShadowEvent(
        parent_ids=torch.arange(4),
        child_ids=torch.arange(4, 8),
        birth_iteration=10,
        steps_completed=2,
    )
    proposal = [SimpleNamespace(image_name="proposal")]
    selector = [SimpleNamespace(image_name="selector")]
    trainer.densify_shadow_fit_cameras = proposal + selector
    trainer.densify_shadow_fit_camera_names = {"proposal", "selector"}
    trainer.densify_shadow_proposal_cameras = proposal
    trainer.densify_shadow_proposal_camera_names = {"proposal"}
    trainer.densify_shadow_selector_cameras = selector
    trainer.densify_shadow_selector_camera_names = {"selector"}
    trainer.densify_shadow_gate_cameras = [SimpleNamespace(image_name="inner")]
    calls: list[tuple[str, frozenset[int]]] = []

    proposal_sse = {
        frozenset(): 100.0,
        frozenset({4, 5, 6, 7}): 60.0,
        frozenset({4, 6}): 70.0,
        frozenset({5, 7}): 80.0,
    }

    def render_subset(
        cameras: list[object], child_ids: torch.Tensor
    ) -> list[dict[str, object]]:
        key = frozenset(int(value) for value in child_ids.tolist())
        camera_name = str(cameras[0].image_name)
        calls.append((camera_name, key))
        sse = proposal_sse[key] if camera_name == "proposal" else {frozenset(): 100.0, frozenset({4, 6}): 90.0}[key]
        return [{"image_name": camera_name, "sse": sse, "count": 1}]

    trainer._render_densify_shadow_subset = render_subset
    selection, selected_ids = trainer._select_densify_shadow_salvage()

    assert torch.equal(selected_ids, torch.tensor([4, 6]))
    assert selection["chosen_prefix"]["group_ids"] == [0]
    assert selection["nested_selector"]["proposal_camera_names"] == ["proposal"]
    assert selection["nested_selector"]["selector_camera_names"] == ["selector"]
    for key in ("proposal_camera_names", "selector_camera_names"):
        payload = json.dumps(
            selection["nested_selector"][key],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        assert selection["nested_selector"][f"{key}_sha256"] == hashlib.sha256(
            payload
        ).hexdigest()
    assert [(name, ids) for name, ids in calls if name == "selector"] == [
        ("selector", frozenset()),
        ("selector", frozenset({4, 6})),
    ]
    assert all(name != "inner" for name, _ in calls)


def test_shadow_salvage_partial_cohort_balances_modulo_groups() -> None:
    trainer = _make_densify_trainer(num_primitives=10, cap_max=15)
    _enable_shadow_densification(trainer, steps=2)
    trainer.opt.densify_max_add = 8
    trainer.opt.densify_shadow_salvage_groups = 3
    trainer.opt.densify_shadow_salvage_allow_partial_cohort = True
    trainer.opt.densify_shadow_salvage_prefix_policy = "first_qualifying"
    trainer.opt.densify_shadow_min_pooled_sse_reduction = 0.0
    trainer.opt.densify_shadow_min_mean_psnr_delta = 0.0
    trainer.opt.densify_shadow_min_win_fraction = 0.5
    trainer.opt.densify_shadow_min_worst_psnr_delta = 0.0
    trainer.densify_shadow_event = _DensifyShadowEvent(
        parent_ids=torch.arange(5),
        child_ids=torch.arange(5, 10),
        birth_iteration=10,
        steps_completed=2,
    )
    trainer.densify_shadow_fit_cameras = [SimpleNamespace(image_name="fit")]
    trainer.densify_shadow_fit_camera_names = {"fit"}

    def render_subset(
        cameras: list[object], child_ids: torch.Tensor
    ) -> list[dict[str, object]]:
        return [
            {
                "image_name": "fit",
                "sse": 100.0 - 5.0 * int(child_ids.numel()),
                "count": 1,
            }
        ]

    trainer._render_densify_shadow_subset = render_subset
    selection, selected_ids = trainer._select_densify_shadow_salvage()

    assert selection["partial_cohort"] == {
        "enabled": True,
        "maximum_child_count": 8,
        "actual_child_count": 5,
        "group_sizes": [2, 2, 1],
    }
    assert selected_ids.numel() == 2


def _shadow_salvage_ordering_selection(
    order: str | None,
    *,
    prefix_policy: str | None = None,
    safe_single_qualifies: bool = False,
) -> tuple[dict, torch.Tensor]:
    trainer = _make_densify_trainer(num_primitives=16, cap_max=24)
    _enable_shadow_densification(trainer, steps=2)
    trainer.opt.densify_max_add = 8
    trainer.opt.densify_shadow_salvage_groups = 8
    if order is not None:
        trainer.opt.densify_shadow_salvage_order = order
    if prefix_policy is not None:
        trainer.opt.densify_shadow_salvage_prefix_policy = prefix_policy
    trainer.opt.densify_shadow_min_pooled_sse_reduction = 0.0
    trainer.opt.densify_shadow_min_mean_psnr_delta = 0.02
    trainer.opt.densify_shadow_min_win_fraction = 0.6
    trainer.opt.densify_shadow_min_worst_psnr_delta = -0.15
    trainer.densify_shadow_event = _DensifyShadowEvent(
        parent_ids=torch.arange(8),
        child_ids=torch.arange(8, 16),
        birth_iteration=10,
        steps_completed=2,
    )
    trainer.densify_shadow_fit_cameras = [
        SimpleNamespace(image_name="fit_a"),
        SimpleNamespace(image_name="fit_b"),
    ]
    trainer.densify_shadow_fit_camera_names = {"fit_a", "fit_b"}

    base = (100.0, 100.0)
    full = (80.0, 80.0)
    safe_single = (90.0, 90.0) if safe_single_qualifies else (90.0, 102.0)
    unsafe_high_score = (60.0, 110.0)
    inactive = (110.0, 110.0)
    subsets: dict[frozenset[int], tuple[float, float]] = {
        frozenset(): base,
        frozenset(range(8, 16)): full,
        frozenset({8}): safe_single,
        frozenset({13}): safe_single,
        frozenset({14}): unsafe_high_score,
        frozenset({8, 13}): (70.0, 70.0),
    }
    for child_id in range(8, 16):
        subsets.setdefault(frozenset({child_id}), inactive)
        subsets[frozenset(set(range(8, 16)) - {child_id})] = (90.0, 90.0)

    def render_subset(
        cameras: list[object], child_ids: torch.Tensor
    ) -> list[dict[str, object]]:
        key = frozenset(int(value) for value in child_ids.tolist())
        sse = subsets.get(key, unsafe_high_score)
        return [
            {"image_name": camera.image_name, "sse": value, "count": 1}
            for camera, value in zip(cameras, sse)
        ]

    trainer._render_densify_shadow_subset = render_subset
    return trainer._select_densify_shadow_salvage()


def test_shadow_salvage_safe_worst_tests_robust_pair_missed_by_score() -> None:
    legacy, legacy_ids = _shadow_salvage_ordering_selection("score")
    safe, safe_ids = _shadow_salvage_ordering_selection("safe_worst")

    assert [record["group_ids"] for record in legacy["prefix_records"]] == [
        [6],
        [6, 0],
        [6, 0, 5],
    ]
    assert legacy["chosen_prefix"] is None
    assert legacy_ids.numel() == 0
    assert [record["group_ids"] for record in safe["prefix_records"]] == [
        [0],
        [0, 5],
        [0, 5, 6],
    ]
    assert safe["chosen_prefix"]["group_ids"] == [0, 5]
    assert torch.equal(safe_ids, torch.tensor([8, 13]))
    assert safe["ordering"] == "safe_worst"


def test_shadow_salvage_default_order_matches_explicit_score() -> None:
    implicit, implicit_ids = _shadow_salvage_ordering_selection(None)
    explicit, explicit_ids = _shadow_salvage_ordering_selection("score")

    assert implicit == explicit
    assert torch.equal(implicit_ids, explicit_ids)
    assert "ordering" not in implicit


def test_shadow_salvage_default_prefix_policy_matches_explicit_best_pooled() -> None:
    implicit, implicit_ids = _shadow_salvage_ordering_selection("safe_worst")
    explicit, explicit_ids = _shadow_salvage_ordering_selection(
        "safe_worst",
        prefix_policy="best_pooled",
    )

    assert implicit == explicit
    assert torch.equal(implicit_ids, explicit_ids)
    assert "prefix_policy" not in implicit
    assert "nested_selector" not in implicit
    assert "partial_cohort" not in implicit


def test_shadow_salvage_first_qualifying_stops_at_first_safe_prefix() -> None:
    best, best_ids = _shadow_salvage_ordering_selection(
        "safe_worst",
        prefix_policy="best_pooled",
        safe_single_qualifies=True,
    )
    first, first_ids = _shadow_salvage_ordering_selection(
        "safe_worst",
        prefix_policy="first_qualifying",
        safe_single_qualifies=True,
    )

    assert best["chosen_prefix"]["group_ids"] == [0, 5]
    assert torch.equal(best_ids, torch.tensor([8, 13]))
    assert [record["group_ids"] for record in first["prefix_records"]] == [[0]]
    assert first["chosen_prefix"]["group_ids"] == [0]
    assert torch.equal(first_ids, torch.tensor([8]))
    assert first["prefix_policy"] == "first_qualifying"


def _prepare_shadow_salvage_finish(
    tmp_path: Path,
) -> Hexels3DTrainer:
    trainer = _make_densify_trainer(num_primitives=8, cap_max=12)
    _enable_shadow_densification(trainer, steps=2)
    trainer.opt.densify_max_add = 4
    trainer.opt.densify_shadow_salvage_groups = 2
    trainer.model_path = tmp_path
    trainer.resume_checkpoint_path = Path("parent.pt")
    trainer.resume_checkpoint_sha256 = "a" * 64
    trainer.densify_shadow_child_mask.copy_(
        torch.tensor([False, False, False, False, True, True, True, True])
    )
    trainer.densify_shadow_attempted_parent_mask.copy_(
        torch.tensor([True, True, True, True, False, False, False, False])
    )
    trainer.densify_shadow_event = _DensifyShadowEvent(
        parent_ids=torch.tensor([0, 1, 2, 3]),
        child_ids=torch.tensor([4, 5, 6, 7]),
        birth_iteration=10,
        steps_completed=2,
    )
    trainer.densify_shadow_counts.update(
        {"proposed_events": 1, "proposed_children": 4}
    )
    loss = sum(group["params"][0].sum() for group in trainer.optimizer.param_groups)
    loss.backward()
    trainer.optimizer.step()
    trainer.optimizer.zero_grad(set_to_none=True)
    return trainer


def test_shadow_salvage_precommits_then_partially_compacts(
    tmp_path: Path,
) -> None:
    trainer = _prepare_shadow_salvage_finish(tmp_path)
    trainer.opt.densify_shadow_enforce_birth_envelope = True
    trainer.opt.densify_child_extent_cap_fraction = 0.3
    trainer.opt.densify_child_parent_opacity_fraction = 0.25
    trainer.opt.densify_child_scale_ratio = 0.5
    event = trainer.densify_shadow_event
    assert event is not None
    event.birth_parent_distances = torch.linalg.vector_norm(
        trainer.model.means3d[event.child_ids] - trainer.model.means3d[event.parent_ids],
        dim=1,
    ).detach().clone()
    event.birth_physical_max_scales = (
        trainer.model.scales[event.child_ids].max(dim=1).values.detach().clone()
    )
    event.birth_log_opacities = (
        -torch.nn.functional.softplus(-trainer.model.opacity_logits[event.child_ids])
    ).detach().clone()
    event.envelope_projection_steps = 2
    event.envelope_projected_mean_rows = 3
    original_means = trainer.model.means3d.detach().clone()
    original_state = trainer.optimizer.state[trainer.model.means3d]["exp_avg"].clone()
    keep_mask = torch.tensor([True, True, True, True, True, False, True, False])
    trainer._select_densify_shadow_salvage = lambda: (
        {"schema_version": 1, "chosen_prefix": {"group_ids": [0]}},
        torch.tensor([4, 6]),
    )

    def evaluate(child_ids: torch.Tensor) -> dict[str, object]:
        paths = list(tmp_path.glob("densify_shadow_salvage_selection_*.json"))
        assert len(paths) == 1
        assert torch.equal(child_ids, torch.tensor([4, 6]))
        return {
            "finite": True,
            "accepted": True,
            "num_training_cameras": 2,
            "reference_sse": 2.0,
            "candidate_sse": 1.0,
            "pooled_sse_reduction": 0.5,
            "mean_psnr_delta": 1.0,
            "win_fraction": 1.0,
            "worst_psnr_delta": 0.5,
        }

    trainer._evaluate_densify_shadow_subset = evaluate
    trainer._finish_densify_shadow_event(iteration=12)

    assert trainer.model.num_base == 6
    assert torch.equal(trainer.model.means3d, original_means[keep_mask])
    means_state = trainer.optimizer.state[trainer.model.means3d]["exp_avg"]
    assert torch.equal(means_state, original_state[keep_mask])
    assert torch.equal(
        trainer.densify_shadow_child_mask,
        torch.tensor([False, False, False, False, True, True]),
    )
    assert torch.equal(
        trainer.densify_has_spawned_child,
        torch.tensor([True, False, True, False, False, False]),
    )
    assert trainer.densify_shadow_counts == {
        "proposed_events": 1,
        "accepted_events": 1,
        "rejected_events": 0,
        "proposed_children": 4,
        "accepted_children": 2,
        "rejected_children": 2,
    }
    decision = trainer.densify_shadow_last_decision
    assert decision["gate_evaluated"] is True
    assert decision["selected_parent_ids"] == [0, 2]
    assert decision["selected_child_ids"] == [4, 6]
    assert len(decision["selected_parent_ids_sha256"]) == 64
    assert len(decision["selected_child_ids_sha256"]) == 64
    assert decision["retained_child_ids_before_compaction"] == [4, 6]
    assert decision["retained_child_ids_after_compaction"] == [4, 5]
    assert decision["birth_envelope"]["projection_counts"] == {
        "steps": 2,
        "means3d_rows": 3,
        "log_scales_rows": 0,
        "opacity_logits_rows": 0,
    }
    sidecar = tmp_path / decision["selection_sidecar"]
    assert sidecar.exists()
    assert hashlib.sha256(sidecar.read_bytes()).hexdigest() == decision["selection_sidecar_sha256"]
    assert "birth_envelope" in json.loads(sidecar.read_text())
    assert trainer._densify_shadow_summary()["adaptive_rows"] == 2


def test_nested_shadow_selector_can_accept_without_rendering_inner_gate(
    tmp_path: Path,
) -> None:
    trainer = _prepare_shadow_salvage_finish(tmp_path)
    trainer.opt.densify_shadow_salvage_selector_stride = 6
    trainer.opt.densify_shadow_salvage_prefix_policy = "first_qualifying"
    trainer.opt.densify_shadow_defer_inner_gate = True
    trainer.densify_shadow_selector_camera_names = {"selector"}
    metrics = {
        "finite": True,
        "num_training_cameras": 1,
        "reference_sse": 2.0,
        "candidate_sse": 1.0,
        "pooled_sse_reduction": 0.5,
        "mean_psnr_delta": 1.0,
        "win_fraction": 1.0,
        "worst_psnr_delta": 1.0,
        "per_view": [{"image_name": "selector", "psnr_delta": 1.0}],
    }
    trainer._select_densify_shadow_salvage = lambda: (
        {
            "schema_version": 1,
            "chosen_prefix": {"group_ids": [0], "metrics": metrics},
        },
        torch.tensor([4, 6]),
    )
    trainer._evaluate_densify_shadow_subset = lambda child_ids: pytest.fail(
        "deferred selector acceptance must not render the inner gate"
    )

    trainer._finish_densify_shadow_event(iteration=12)

    assert trainer.model.num_base == 6
    decision = trainer.densify_shadow_last_decision
    assert decision["accepted"] is True
    assert decision["gate_evaluated"] is False
    assert decision["selector_evaluated"] is True
    assert decision["inner_gate_deferred"] is True
    assert decision["decision_camera_names"] == ["selector"]
    assert decision["retained_child_count"] == 2


def test_shadow_salvage_without_fit_prefix_skips_gate_and_rolls_back(
    tmp_path: Path,
) -> None:
    trainer = _prepare_shadow_salvage_finish(tmp_path)
    trainer._select_densify_shadow_salvage = lambda: (
        {"schema_version": 1, "chosen_prefix": None},
        torch.empty((0,), dtype=torch.long),
    )
    trainer._evaluate_densify_shadow_subset = lambda child_ids: pytest.fail(
        "gate must not be evaluated without a fit-selected prefix"
    )

    trainer._finish_densify_shadow_event(iteration=12)

    assert trainer.model.num_base == 4
    assert trainer.densify_shadow_last_decision["gate_evaluated"] is False
    assert trainer.densify_shadow_counts["rejected_events"] == 1
    assert trainer.densify_shadow_counts["rejected_children"] == 4
    sidecar = tmp_path / trainer.densify_shadow_last_decision["selection_sidecar"]
    assert json.loads(sidecar.read_text())["chosen_prefix"] is None
    assert trainer._densify_shadow_summary()["adaptive_rows"] == 0


def test_shadow_salvage_gate_failure_rolls_back_selected_prefix(
    tmp_path: Path,
) -> None:
    trainer = _prepare_shadow_salvage_finish(tmp_path)
    trainer._select_densify_shadow_salvage = lambda: (
        {"schema_version": 1, "chosen_prefix": {"group_ids": [0]}},
        torch.tensor([4, 6]),
    )

    def reject(child_ids: torch.Tensor) -> dict[str, object]:
        assert list(tmp_path.glob("densify_shadow_salvage_selection_*.json"))
        assert torch.equal(child_ids, torch.tensor([4, 6]))
        return {
            "finite": True,
            "accepted": False,
            "num_training_cameras": 2,
            "reference_sse": 1.0,
            "candidate_sse": 2.0,
            "pooled_sse_reduction": -1.0,
            "mean_psnr_delta": -1.0,
            "win_fraction": 0.0,
            "worst_psnr_delta": -1.0,
        }

    trainer._evaluate_densify_shadow_subset = reject
    trainer._finish_densify_shadow_event(iteration=12)

    assert trainer.model.num_base == 4
    assert not bool(trainer.densify_has_spawned_child.any().item())
    assert bool(trainer.densify_shadow_attempted_parent_mask.all().item())
    assert trainer.densify_shadow_counts["accepted_children"] == 0
    assert trainer.densify_shadow_counts["rejected_children"] == 4
    assert trainer.densify_shadow_last_decision["gate_evaluated"] is True
    assert trainer.densify_shadow_last_decision["retained_child_count"] == 0


@pytest.mark.parametrize("accepted", [False, True])
def test_shadow_decision_commits_or_physically_rolls_back_tail(accepted: bool) -> None:
    trainer = _make_densify_trainer(num_primitives=3, cap_max=8)
    _enable_shadow_densification(trainer)
    trainer.densify_local_grad_abs_accum = torch.ones((3, 2))
    trainer.densify_local_grad_signed_accum = torch.ones((3, 2))
    trainer.densify_birth_iteration = torch.zeros(3, dtype=torch.long)
    trainer._edge_child_primitives(torch.tensor([2]), iteration=100)

    assert trainer.model.num_base == 4
    assert trainer.densify_shadow_event is not None
    assert trainer.densify_shadow_attempted_parent_mask[2]
    assert not trainer.densify_has_spawned_child[2]
    trainer.densify_shadow_event.steps_completed = 2
    trainer._evaluate_densify_shadow_event = lambda: {
        "finite": True,
        "accepted": accepted,
        "num_training_cameras": 1,
        "pooled_sse_reduction": 0.1 if accepted else -0.1,
        "mean_psnr_delta": 0.1 if accepted else -0.1,
        "win_fraction": 1.0 if accepted else 0.0,
        "worst_psnr_delta": 0.1 if accepted else -0.1,
    }
    trainer._finish_densify_shadow_event(iteration=102)

    assert trainer.densify_shadow_event is None
    assert trainer.densify_shadow_attempted_parent_mask[2]
    assert bool(trainer.densify_has_spawned_child[2]) is accepted
    assert trainer.model.num_base == (4 if accepted else 3)
    assert trainer.densify_shadow_child_mask.shape == (trainer.model.num_base,)
    assert int(trainer.densify_shadow_child_mask.sum().item()) == (1 if accepted else 0)
    assert len(trainer.densify_shadow_decisions) == 1
    assert trainer.densify_shadow_decisions[0]["accepted"] is accepted
    assert (
        trainer.densify_shadow_counts["accepted_events"]
        + trainer.densify_shadow_counts["rejected_events"]
        == len(trainer.densify_shadow_decisions)
    )
    audit = trainer._densify_shadow_checkpoint_audit()
    assert audit["child_mask"].shape == (trainer.model.num_base,)
    assert audit["attempted_parent_mask"].shape == (trainer.model.num_base,)
    assert len(audit["decisions"]) == 1
    if not accepted:
        trainer.densify_grad_accum.zero_()
        trainer.densify_grad_accum[2] = 1.0
        trainer.densify_denom.fill_(1.0)
        trainer.densify_local_grad_abs_accum = torch.ones((3, 2))
        trainer.densify_local_grad_signed_accum = torch.ones((3, 2))
        trainer._densify_and_prune_hexels_safe(size_threshold=0, iteration=200)
        assert trainer.densify_shadow_event is None
        assert trainer.model.num_base == 3


def test_shadow_probation_timing_and_mutation_block() -> None:
    trainer = _make_densify_trainer(num_primitives=3, cap_max=8)
    _enable_shadow_densification(trainer, steps=2)
    trainer.densify_shadow_event = _DensifyShadowEvent(
        parent_ids=torch.tensor([0]),
        child_ids=torch.tensor([2]),
        birth_iteration=10,
    )
    decisions: list[int] = []

    def finish(*, iteration: int) -> None:
        decisions.append(iteration)
        trainer.densify_shadow_event = None

    trainer._finish_densify_shadow_event = finish
    with pytest.raises(RuntimeError, match="blocked during shadow probation"):
        trainer._step_optimizer_and_density(
            iteration=11,
            run_densify=True,
            size_threshold=0,
            reset_opacity=False,
        )
    trainer._step_optimizer_and_density(
        iteration=11,
        run_densify=False,
        size_threshold=0,
        reset_opacity=False,
    )
    assert decisions == []
    trainer._step_optimizer_and_density(
        iteration=12,
        run_densify=False,
        size_threshold=0,
        reset_opacity=False,
    )
    assert decisions == [12]


def test_shadow_probation_only_separates_pending_cohort() -> None:
    trainer = _make_densify_trainer(num_primitives=4, cap_max=8)
    _enable_shadow_densification(trainer, permanent_residual=False)
    trainer.densify_shadow_child_mask.copy_(torch.tensor([False, False, True, True]))
    trainer.densify_shadow_event = _DensifyShadowEvent(
        parent_ids=torch.tensor([1]),
        child_ids=torch.tensor([3]),
        birth_iteration=10,
    )
    assert torch.equal(trainer._densify_shadow_gradient_ids(), torch.tensor([3]))
    trainer.densify_shadow_event = None
    assert trainer._densify_shadow_gradient_ids().numel() == 0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("densify_shadow_steps", -1, "densify_shadow_steps must be >= 0"),
        (
            "densify_shadow_child_probation",
            "unsafe",
            "must be one of: unrestricted, coplanar_spatial",
        ),
        ("densify_shadow_gate_stride", 1, "must be 0 or >= 2"),
        ("densify_shadow_salvage_groups", 1, "must be 0 or >= 2"),
        ("densify_shadow_salvage_selector_stride", 1, "must be 0 or >= 2"),
        (
            "densify_shadow_salvage_order",
            "unsafe",
            "must be one of: score, safe_worst",
        ),
        (
            "densify_shadow_salvage_prefix_policy",
            "unsafe",
            "must be one of: best_pooled, first_qualifying",
        ),
        ("densify_shadow_min_win_fraction", 1.1, "must be in \\[0,1\\]"),
        ("densify_action", "clone", "requires hexels_safe edge_child"),
        ("densify_prune_interval", 100, "requires --densify_prune_interval 0"),
        ("densify_repeat_parent", True, "requires one proposal attempt per parent"),
        ("random_background", True, "requires plain photometric optimization"),
        ("repulsion_screen_weight", 0.1, "requires plain photometric optimization"),
    ],
)
def test_shadow_option_validation(field: str, value: object, message: str) -> None:
    opt = Hexels3DOptions()
    opt.source_path = "family"
    opt.model_path = "output"
    opt.enable_densify = True
    opt.densify_impl = "hexels_safe"
    opt.densify_action = "edge_child"
    opt.densify_shadow_steps = 2
    opt.densify_prune_interval = 0
    opt.opacity_reset_interval = 0
    setattr(opt, field, value)

    with pytest.raises(ValueError, match=message):
        opt.process_args()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "densify_child_extent_cap_fraction",
            float("inf"),
            "extent_cap_fraction must be finite and >= 0",
        ),
        (
            "densify_child_extent_cap_fraction",
            -0.1,
            "extent_cap_fraction must be finite and >= 0",
        ),
        (
            "densify_child_parent_opacity_fraction",
            1.1,
            "parent_opacity_fraction must be finite and in \\[0,1\\]",
        ),
    ],
)
def test_bounded_child_option_value_validation(
    field: str,
    value: float,
    message: str,
) -> None:
    opt = Hexels3DOptions()
    opt.source_path = "family"
    opt.model_path = "output"
    setattr(opt, field, value)

    with pytest.raises(ValueError, match=message):
        opt.process_args()


@pytest.mark.parametrize(
    ("enable_densify", "densify_impl", "densify_action", "message"),
    [
        (False, "2dgs", "mixed", "requires --enable_densify"),
        (True, "2dgs", "mixed", "requires hexels_safe edge_child"),
        (True, "hexels_safe", "clone", "requires hexels_safe edge_child"),
    ],
)
def test_bounded_child_options_require_edge_child_densification(
    enable_densify: bool,
    densify_impl: str,
    densify_action: str,
    message: str,
) -> None:
    opt = Hexels3DOptions()
    opt.source_path = "family"
    opt.model_path = "output"
    opt.enable_densify = enable_densify
    opt.densify_impl = densify_impl
    opt.densify_action = densify_action
    opt.densify_child_extent_cap_fraction = 0.05

    with pytest.raises(ValueError, match=message):
        opt.process_args()


def test_bounded_child_options_accept_safe_edge_child_configuration() -> None:
    opt = Hexels3DOptions()
    opt.source_path = "family"
    opt.model_path = "output"
    opt.enable_densify = True
    opt.densify_impl = "hexels_safe"
    opt.densify_action = "edge_child"
    opt.densify_child_extent_cap_fraction = 0.05
    opt.densify_child_parent_opacity_fraction = 0.25
    opt.opacity_reset_interval = 0

    opt.process_args()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("densify_shadow_steps", 0, "requires shadow densification"),
        (
            "densify_child_extent_cap_fraction",
            0.0,
            "requires a positive child extent cap",
        ),
        (
            "densify_child_parent_opacity_fraction",
            0.0,
            "requires a positive parent opacity fraction",
        ),
    ],
)
def test_shadow_birth_envelope_option_dependencies(
    field: str,
    value: object,
    message: str,
) -> None:
    opt = Hexels3DOptions()
    opt.source_path = "family"
    opt.model_path = "output"
    opt.enable_densify = True
    opt.densify_impl = "hexels_safe"
    opt.densify_action = "edge_child"
    opt.densify_shadow_steps = 2
    opt.densify_shadow_enforce_birth_envelope = True
    opt.densify_child_extent_cap_fraction = 0.05
    opt.densify_child_parent_opacity_fraction = 0.25
    opt.densify_prune_interval = 0
    opt.opacity_reset_interval = 0
    setattr(opt, field, value)

    with pytest.raises(ValueError, match=message):
        opt.process_args()


def test_shadow_birth_envelope_accepts_bounded_shadow_edge_children() -> None:
    opt = Hexels3DOptions()
    opt.source_path = "family"
    opt.model_path = "output"
    opt.enable_densify = True
    opt.densify_impl = "hexels_safe"
    opt.densify_action = "edge_child"
    opt.densify_shadow_steps = 2
    opt.densify_shadow_enforce_birth_envelope = True
    opt.densify_child_extent_cap_fraction = 0.05
    opt.densify_child_parent_opacity_fraction = 0.25
    opt.densify_prune_interval = 0
    opt.opacity_reset_interval = 0

    opt.process_args()


def test_shadow_steps_require_densification() -> None:
    opt = Hexels3DOptions()
    opt.source_path = "family"
    opt.model_path = "output"
    opt.densify_shadow_steps = 2

    with pytest.raises(ValueError, match="requires --enable_densify"):
        opt.process_args()


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        (
            {"densify_joint_coplanar_steps": -1},
            "densify_joint_coplanar_steps must be >= 0",
        ),
        (
            {"enable_densify": False},
            "requires --enable_densify",
        ),
        (
            {"densify_impl": "2dgs"},
            "requires hexels_safe edge_child",
        ),
        (
            {"densify_action": "clone"},
            "requires hexels_safe edge_child",
        ),
        (
            {"densify_shadow_steps": 1},
            "does not support shadow probation",
        ),
        (
            {"densify_child_parent_opacity_fraction": 0.0},
            "requires a positive child parent opacity fraction",
        ),
    ],
)
def test_joint_coplanar_option_dependencies(
    updates: dict[str, object],
    message: str,
) -> None:
    opt = Hexels3DOptions()
    opt.source_path = "family"
    opt.model_path = "output"
    opt.enable_densify = True
    opt.densify_impl = "hexels_safe"
    opt.densify_action = "edge_child"
    opt.densify_joint_coplanar_steps = 2
    opt.densify_child_parent_opacity_fraction = 0.25
    opt.opacity_reset_interval = 0
    for name, value in updates.items():
        setattr(opt, name, value)

    with pytest.raises(ValueError, match=message):
        opt.process_args()


def test_joint_coplanar_options_accept_direct_edge_children() -> None:
    opt = Hexels3DOptions()
    opt.source_path = "family"
    opt.model_path = "output"
    opt.enable_densify = True
    opt.densify_impl = "hexels_safe"
    opt.densify_action = "edge_child"
    opt.densify_joint_coplanar_steps = 2
    opt.densify_child_parent_opacity_fraction = 0.25
    opt.opacity_reset_interval = 0

    opt.process_args()


def test_coplanar_spatial_probation_requires_shadow_steps() -> None:
    opt = Hexels3DOptions()
    opt.source_path = "family"
    opt.model_path = "output"
    opt.densify_shadow_child_probation = "coplanar_spatial"

    with pytest.raises(ValueError, match="requires --densify_shadow_steps"):
        opt.process_args()


def test_coplanar_spatial_probation_accepts_shadow_edge_children() -> None:
    opt = Hexels3DOptions()
    opt.source_path = "family"
    opt.model_path = "output"
    opt.enable_densify = True
    opt.densify_impl = "hexels_safe"
    opt.densify_action = "edge_child"
    opt.densify_shadow_steps = 2
    opt.densify_shadow_child_probation = "coplanar_spatial"
    opt.densify_prune_interval = 0
    opt.opacity_reset_interval = 0

    opt.process_args()


def test_shadow_gate_requires_permanent_residual_training() -> None:
    opt = Hexels3DOptions()
    opt.source_path = "family"
    opt.model_path = "output"
    opt.enable_densify = True
    opt.densify_impl = "hexels_safe"
    opt.densify_action = "edge_child"
    opt.densify_shadow_steps = 2
    opt.densify_shadow_gate_stride = 7
    opt.densify_shadow_permanent_residual = False
    opt.densify_prune_interval = 0
    opt.opacity_reset_interval = 0

    with pytest.raises(ValueError, match="require permanent residual"):
        opt.process_args()


def test_shadow_salvage_requires_reserved_gate() -> None:
    opt = Hexels3DOptions()
    opt.source_path = "family"
    opt.model_path = "output"
    opt.enable_densify = True
    opt.densify_impl = "hexels_safe"
    opt.densify_action = "edge_child"
    opt.densify_shadow_steps = 2
    opt.densify_shadow_salvage_groups = 2
    opt.densify_prune_interval = 0
    opt.opacity_reset_interval = 0

    with pytest.raises(ValueError, match="requires a shadow gate"):
        opt.process_args()


def test_shadow_first_qualifying_prefix_requires_salvage_groups() -> None:
    opt = Hexels3DOptions()
    opt.source_path = "family"
    opt.model_path = "output"
    opt.enable_densify = True
    opt.densify_impl = "hexels_safe"
    opt.densify_action = "edge_child"
    opt.densify_shadow_steps = 2
    opt.densify_shadow_salvage_prefix_policy = "first_qualifying"
    opt.densify_prune_interval = 0
    opt.opacity_reset_interval = 0

    with pytest.raises(ValueError, match="requires --densify_shadow_salvage_groups"):
        opt.process_args()


def test_shadow_salvage_requires_fixed_divisible_cohort() -> None:
    opt = Hexels3DOptions()
    opt.source_path = "family"
    opt.model_path = "output"
    opt.enable_densify = True
    opt.densify_impl = "hexels_safe"
    opt.densify_action = "edge_child"
    opt.densify_shadow_steps = 2
    opt.densify_shadow_gate_stride = 7
    opt.densify_shadow_salvage_groups = 3
    opt.densify_prune_interval = 0
    opt.opacity_reset_interval = 0

    with pytest.raises(ValueError, match="requires --densify_max_add > 0"):
        opt.process_args()

    opt.densify_max_add = 4
    with pytest.raises(ValueError, match="must be divisible"):
        opt.process_args()


def test_shadow_salvage_rejects_short_runtime_cohort_before_render() -> None:
    trainer = _make_densify_trainer(num_primitives=4, cap_max=8)
    _enable_shadow_densification(trainer, steps=2)
    trainer.opt.densify_max_add = 4
    trainer.opt.densify_shadow_salvage_groups = 2
    trainer.densify_shadow_event = _DensifyShadowEvent(
        parent_ids=torch.tensor([0, 1]),
        child_ids=torch.tensor([2, 3]),
        birth_iteration=10,
        steps_completed=2,
    )
    trainer._render_densify_shadow_subset = lambda *_: pytest.fail(
        "fit render must not run for a short cohort"
    )

    with pytest.raises(RuntimeError, match="expected 4, got 2"):
        trainer._select_densify_shadow_salvage()


def test_shadow_partial_cohort_skips_undersized_proposal_before_birth() -> None:
    trainer = _make_densify_trainer(num_primitives=4, cap_max=8)
    _enable_shadow_densification(trainer, steps=2)
    trainer.opt.densify_max_add = 4
    trainer.opt.densify_shadow_salvage_groups = 3
    trainer.opt.densify_shadow_salvage_allow_partial_cohort = True
    trainer.opt.densify_newborn_grace = 1
    trainer.densify_grad_accum = torch.tensor([[1.0], [1.0], [0.0], [0.0]])
    trainer.densify_denom = torch.ones((4, 1))
    trainer.densify_local_grad_abs_accum = torch.ones((4, 2))
    trainer.densify_local_grad_signed_accum = torch.ones((4, 2))
    trainer.densify_birth_iteration = torch.zeros(4, dtype=torch.long)
    trainer._edge_child_primitives = lambda *args, **kwargs: pytest.fail(
        "undersized partial cohort must be skipped before child birth"
    )

    trainer._densify_and_prune_hexels_safe(size_threshold=0, iteration=100)

    assert trainer.model.num_base == 4
    assert trainer.densify_shadow_event is None
    assert trainer.densify_shadow_counts["proposed_events"] == 0
    assert torch.count_nonzero(trainer.densify_grad_accum) == 0


def test_shadow_salvage_accepts_valid_group_gate_configuration() -> None:
    opt = Hexels3DOptions()
    opt.source_path = "family"
    opt.model_path = "output"
    opt.enable_densify = True
    opt.densify_impl = "hexels_safe"
    opt.densify_action = "edge_child"
    opt.densify_shadow_steps = 2
    opt.densify_shadow_gate_stride = 7
    opt.densify_shadow_salvage_groups = 2
    opt.densify_shadow_salvage_order = "safe_worst"
    opt.densify_shadow_salvage_prefix_policy = "first_qualifying"
    opt.densify_max_add = 4
    opt.densify_prune_interval = 0
    opt.opacity_reset_interval = 0

    opt.process_args()


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        (
            {"densify_shadow_salvage_selector_stride": 6},
            "requires --densify_shadow_salvage_groups",
        ),
        (
            {
                "densify_shadow_salvage_groups": 2,
                "densify_max_add": 4,
                "densify_shadow_salvage_selector_stride": 6,
            },
            "requires --densify_shadow_salvage_prefix_policy first_qualifying",
        ),
        (
            {"densify_shadow_defer_inner_gate": True},
            "requires --densify_shadow_salvage_selector_stride",
        ),
        (
            {"densify_shadow_salvage_allow_partial_cohort": True},
            "requires --densify_shadow_salvage_groups",
        ),
        (
            {
                "densify_shadow_salvage_groups": 3,
                "densify_max_add": 2,
                "densify_shadow_salvage_allow_partial_cohort": True,
            },
            "must be at least --densify_shadow_salvage_groups",
        ),
    ],
)
def test_nested_shadow_selector_option_dependencies(
    updates: dict[str, object],
    message: str,
) -> None:
    opt = Hexels3DOptions()
    opt.source_path = "family"
    opt.model_path = "output"
    opt.enable_densify = True
    opt.densify_impl = "hexels_safe"
    opt.densify_action = "edge_child"
    opt.densify_shadow_steps = 2
    opt.densify_shadow_gate_stride = 7
    opt.densify_prune_interval = 0
    opt.opacity_reset_interval = 0
    for name, value in updates.items():
        setattr(opt, name, value)

    with pytest.raises(ValueError, match=message):
        opt.process_args()


def test_nested_shadow_selector_accepts_deferred_partial_configuration() -> None:
    opt = Hexels3DOptions()
    opt.source_path = "family"
    opt.model_path = "output"
    opt.enable_densify = True
    opt.densify_impl = "hexels_safe"
    opt.densify_action = "edge_child"
    opt.densify_shadow_steps = 2
    opt.densify_shadow_gate_stride = 7
    opt.densify_shadow_salvage_groups = 3
    opt.densify_shadow_salvage_prefix_policy = "first_qualifying"
    opt.densify_shadow_salvage_selector_stride = 6
    opt.densify_shadow_defer_inner_gate = True
    opt.densify_shadow_salvage_allow_partial_cohort = True
    opt.densify_max_add = 5
    opt.densify_prune_interval = 0
    opt.opacity_reset_interval = 0

    opt.process_args()


def test_nested_shadow_selector_rejects_parent_relative_birth_envelope() -> None:
    opt = Hexels3DOptions()
    opt.source_path = "family"
    opt.model_path = "output"
    opt.enable_densify = True
    opt.densify_impl = "hexels_safe"
    opt.densify_action = "edge_child"
    opt.densify_shadow_steps = 2
    opt.densify_shadow_gate_stride = 7
    opt.densify_shadow_salvage_groups = 2
    opt.densify_shadow_salvage_prefix_policy = "first_qualifying"
    opt.densify_shadow_salvage_selector_stride = 6
    opt.densify_max_add = 4
    opt.densify_shadow_enforce_birth_envelope = True
    opt.densify_child_extent_cap_fraction = 0.05
    opt.densify_child_parent_opacity_fraction = 0.25
    opt.densify_prune_interval = 0
    opt.opacity_reset_interval = 0

    with pytest.raises(ValueError, match="incompatible with"):
        opt.process_args()


def test_shadow_view_anchor_requires_view_gradients() -> None:
    opt = Hexels3DOptions()
    opt.source_path = "family"
    opt.model_path = "output"
    opt.enable_densify = True
    opt.densify_impl = "hexels_safe"
    opt.densify_action = "edge_child"
    opt.densify_shadow_steps = 2
    opt.densify_shadow_parent_view_anchor = True
    opt.densify_prune_interval = 0
    opt.opacity_reset_interval = 0
    opt.use_view_grads = False

    with pytest.raises(ValueError, match="requires view gradients"):
        opt.process_args()


@pytest.mark.parametrize(
    ("view_model", "prefilter_edge_on"),
    [("none", False), ("sh3", False), ("lobe", False), ("linear", True)],
)
def test_shadow_view_anchor_requires_linear_fast_path(
    view_model: str,
    prefilter_edge_on: bool,
) -> None:
    opt = Hexels3DOptions()
    opt.source_path = "family"
    opt.model_path = "output"
    opt.enable_densify = True
    opt.densify_impl = "hexels_safe"
    opt.densify_action = "edge_child"
    opt.densify_shadow_steps = 2
    opt.densify_shadow_parent_view_anchor = True
    opt.densify_prune_interval = 0
    opt.opacity_reset_interval = 0
    opt.view_model = view_model
    opt.prefilter_edge_on = prefilter_edge_on

    with pytest.raises(ValueError, match="requires the linear view fast path"):
        opt.process_args()


def test_residual_moment_children_allow_zero_legacy_gradient_threshold() -> None:
    opt = Hexels3DOptions()
    opt.source_path = "family"
    opt.model_path = "output"
    opt.enable_densify = True
    opt.train_mode = "full"
    opt.densify_impl = "hexels_safe"
    opt.densify_action = "edge_child"
    opt.densify_direction_signal = "residual_moment"
    opt.densify_grad_threshold = 0.0
    opt.densify_prune_interval = 0
    opt.opacity_reset_interval = 0

    opt.process_args()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("densify_residual_topk_views", -1, "must be >= 0"),
        (
            "densify_direction_signal",
            "position",
            "requires residual-moment hexels_safe edge-child densification",
        ),
        ("camera_sampling", "random", "requires --camera_sampling epoch"),
        ("densify_moment_interval", 2, "requires --densify_moment_interval 1"),
        (
            "densify_min_denom",
            2,
            "must be at least --densify_residual_topk_views",
        ),
    ],
)
def test_residual_topk_option_dependencies(
    field: str,
    value: object,
    message: str,
) -> None:
    opt = Hexels3DOptions()
    opt.source_path = "family"
    opt.model_path = "output"
    opt.enable_densify = True
    opt.densify_impl = "hexels_safe"
    opt.densify_action = "edge_child"
    opt.densify_direction_signal = "residual_moment"
    opt.densify_residual_topk_views = 3
    opt.camera_sampling = "epoch"
    opt.densify_moment_interval = 1
    opt.densify_min_denom = 3
    opt.densify_prune_interval = 0
    opt.opacity_reset_interval = 0
    setattr(opt, field, value)

    with pytest.raises(ValueError, match=message):
        opt.process_args()


def test_residual_topk_option_accepts_camera_complete_residual_children() -> None:
    opt = Hexels3DOptions()
    opt.source_path = "family"
    opt.model_path = "output"
    opt.enable_densify = True
    opt.densify_impl = "hexels_safe"
    opt.densify_action = "edge_child"
    opt.densify_direction_signal = "residual_moment"
    opt.densify_residual_topk_views = 3
    opt.camera_sampling = "epoch"
    opt.densify_moment_interval = 1
    opt.densify_min_denom = 3
    opt.densify_prune_interval = 0
    opt.opacity_reset_interval = 0

    opt.process_args()


def test_attributed_option_defaults_preserve_historical_path() -> None:
    opt = Hexels3DOptions()
    opt.source_path = "family"
    opt.model_path = "output"

    assert opt.densify_direction_signal == "position"
    assert opt.densify_attributed_weak_power == 0
    assert not opt.densify_residual_persistent_filter
    assert opt.densify_covariance_children_per_parent == 2
    assert opt.densify_covariance_min_views == 8
    assert opt.densify_covariance_min_anisotropy == pytest.approx(0.15)
    assert opt.densify_covariance_min_axis_coherence == pytest.approx(0.50)
    assert opt.densify_covariance_min_side_balance == pytest.approx(0.30)
    assert opt.densify_covariance_min_delta == pytest.approx(0.10)
    opt.process_args()


def test_attributed_options_accept_native_camera_complete_mode() -> None:
    opt = _make_attributed_options()
    opt.densify_attributed_weak_power = 2

    opt.process_args()


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        (
            {"densify_attributed_weak_power": 1},
            "densify_attributed_weak_power must be 0 or 2",
        ),
        (
            {
                "densify_direction_signal": "position",
                "densify_attributed_weak_power": 2,
            },
            "requires an attributed direction",
        ),
        (
            {"densify_residual_persistent_filter": True},
            "requires residual-moment direction",
        ),
        (
            {"enable_children": True},
            "does not support learned children",
        ),
    ],
)
def test_attributed_option_dependencies(
    updates: dict[str, object],
    message: str,
) -> None:
    opt = _make_attributed_options()
    for name, value in updates.items():
        setattr(opt, name, value)

    with pytest.raises(ValueError, match=message):
        opt.process_args()


@pytest.mark.parametrize("children_per_parent", [1, 2])
def test_covariance_options_accept_direct_native_mode(
    children_per_parent: int,
) -> None:
    opt = _make_covariance_options(children_per_parent=children_per_parent)
    opt.densify_attributed_weak_power = 2

    opt.process_args()


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        (
            {"densify_covariance_children_per_parent": 3},
            "densify_covariance_children_per_parent must be 1 or 2",
        ),
        (
            {"densify_covariance_min_views": 0},
            "densify_covariance_min_views must be positive",
        ),
        (
            {"densify_covariance_min_anisotropy": 1.1},
            "densify_covariance_min_anisotropy must be in",
        ),
        (
            {"densify_covariance_min_axis_coherence": -0.1},
            "densify_covariance_min_axis_coherence must be in",
        ),
        (
            {"densify_covariance_min_side_balance": 1.1},
            "densify_covariance_min_side_balance must be in",
        ),
        (
            {"densify_covariance_min_delta": -0.1},
            "densify_covariance_min_delta must be non-negative",
        ),
        (
            {"densify_max_add": 3},
            "densify_max_add must be divisible by 2",
        ),
        (
            {"densify_child_parent_opacity_fraction": 0.0},
            "covariance-pair births require a positive",
        ),
        (
            {"camera_sampling": "random"},
            "requires epoch sampling",
        ),
        (
            {"prefilter_edge_on": True},
            "requires native edge rendering",
        ),
        (
            {"enable_children": True},
            "does not support learned children",
        ),
        (
            {"densify_shadow_steps": 1},
            "does not support shadow probation",
        ),
    ],
)
def test_covariance_option_dependencies(
    updates: dict[str, object],
    message: str,
) -> None:
    opt = _make_covariance_options()
    for name, value in updates.items():
        setattr(opt, name, value)

    with pytest.raises(ValueError, match=message):
        opt.process_args()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("densify_cap_max", -1, "densify_cap_max must be >= 0"),
        ("densify_action", "split", "densify_action must be one of"),
    ],
)
def test_densify_budget_option_validation(field: str, value: object, message: str) -> None:
    opt = Hexels3DOptions()
    opt.source_path = "family"
    opt.model_path = "output"
    opt.enable_densify = True
    opt.densify_impl = "hexels_safe"
    setattr(opt, field, value)
    with pytest.raises(ValueError, match=message):
        opt.process_args()


def test_clone_densify_rejects_immediate_opacity_pruning() -> None:
    opt = Hexels3DOptions()
    opt.source_path = "family"
    opt.model_path = "output"
    opt.enable_densify = True
    opt.densify_impl = "hexels_safe"
    opt.densify_action = "clone"
    opt.densify_clone_jitter = 0.25
    opt.densify_clone_init_opacity = 0.01
    opt.opacity_cull = 0.05
    opt.opacity_reset_interval = 0

    with pytest.raises(ValueError, match="opacity_cull must be smaller"):
        opt.process_args()

    opt.densify_action = "mixed"
    with pytest.raises(ValueError, match="opacity_cull must be smaller"):
        opt.process_args()


def test_directional_densify_requires_newborn_grace() -> None:
    opt = Hexels3DOptions()
    opt.source_path = "family"
    opt.model_path = "output"
    opt.enable_densify = True
    opt.densify_impl = "hexels_safe"
    opt.densify_action = "edge_child"
    opt.densify_newborn_grace = 0
    opt.opacity_reset_interval = 0

    with pytest.raises(ValueError, match="densify_newborn_grace > 0"):
        opt.process_args()


def test_directional_densify_rejects_positive_contribution_pruning() -> None:
    opt = Hexels3DOptions()
    opt.source_path = "family"
    opt.model_path = "output"
    opt.enable_densify = True
    opt.densify_impl = "hexels_safe"
    opt.densify_action = "edge_child"
    opt.densify_prune_contrib_threshold = 0.01
    opt.opacity_reset_interval = 0

    with pytest.raises(ValueError, match="exact-zero contribution pruning only"):
        opt.process_args()


@pytest.mark.parametrize(
    "field",
    [
        "repulsion_knn_weight",
        "surface_coplanar_weight",
        "surface_normal_smooth_weight",
    ],
)
def test_edge_child_pruning_rejects_inactive_unsafe_knn_losses(field: str) -> None:
    opt = Hexels3DOptions()
    opt.source_path = "family"
    opt.model_path = "output"
    opt.enable_densify = True
    opt.densify_impl = "hexels_safe"
    opt.densify_action = "edge_child"
    opt.opacity_reset_interval = 0
    setattr(opt, field, 0.1)

    with pytest.raises(ValueError, match="incompatible with KNN repulsion/surface losses"):
        opt.process_args()


def test_safe_densify_rejects_reset_cull_roundoff_boundary() -> None:
    opt = Hexels3DOptions()
    opt.source_path = "family"
    opt.model_path = "output"
    opt.enable_densify = True
    opt.densify_impl = "hexels_safe"
    opt.densify_clone_jitter = 0.25
    opt.densify_clone_init_opacity = 0.02
    opt.opacity_reset_interval = 3000
    opt.opacity_cull = 0.01

    with pytest.raises(ValueError, match="opacity resets require --opacity_cull < 0.01"):
        opt.process_args()


def test_sh3_residual_matches_2dgs_basis_and_gradcheck() -> None:
    reference = runpy.run_path(str(Path(__file__).parents[1] / "2d-gaussian-splatting" / "utils" / "sh_utils.py"))
    eval_sh = reference["eval_sh"]
    coeffs = torch.randn((4, 15, 3), dtype=torch.double, requires_grad=True)
    raw_dirs = torch.randn((4, 3), dtype=torch.double, requires_grad=True)
    dirs = raw_dirs / torch.linalg.norm(raw_dirs, dim=-1, keepdim=True)
    full_coeffs = torch.cat(
        [torch.zeros((4, 3, 1), dtype=torch.double), coeffs.permute(0, 2, 1)],
        dim=-1,
    )

    expected = eval_sh(3, full_coeffs, dirs)
    assert torch.allclose(_eval_sh3_residual(coeffs, dirs), expected)
    assert torch.allclose(_eval_sh3_residual(coeffs, dirs, degree=3), expected)
    assert torch.autograd.gradcheck(_eval_sh3_residual, (coeffs, dirs))


def test_sh3_degree_masking_zeros_inactive_coefficient_gradients() -> None:
    coeffs = torch.ones((2, 15, 3), requires_grad=True)
    dirs = torch.nn.functional.normalize(
        torch.tensor([[1.0, 2.0, 3.0], [-2.0, 1.0, 4.0]]),
        dim=-1,
    )

    degree_zero = _eval_sh3_residual(coeffs, dirs, degree=0)
    assert torch.equal(degree_zero, torch.zeros_like(degree_zero))
    degree_zero.sum().backward()
    assert torch.equal(coeffs.grad, torch.zeros_like(coeffs.grad))

    coeffs.grad = None
    _eval_sh3_residual(coeffs, dirs, degree=2).sum().backward()
    assert torch.count_nonzero(coeffs.grad[:, :8]) > 0
    assert torch.equal(coeffs.grad[:, 8:], torch.zeros_like(coeffs.grad[:, 8:]))


def test_progressive_sh_schedule_boundaries_and_transition_logging() -> None:
    assert [_sh_degree_for_iteration(i, 1000) for i in (1, 999, 1000, 1999, 2000, 2999, 3000)] == [
        0,
        0,
        1,
        1,
        2,
        2,
        3,
    ]
    assert _sh_degree_for_iteration(1, 0) == 3

    messages: list[tuple[object, ...]] = []
    trainer = Hexels3DTrainer.__new__(Hexels3DTrainer)
    trainer.view_model = "sh3"
    trainer.sh_progressive_interval = 1000
    trainer.active_sh_degree = None
    trainer.logger = SimpleNamespace(info=lambda *args: messages.append(args))
    for iteration in (1, 2, 999, 1000, 1001, 2000, 3000, 4000):
        trainer._update_sh_degree(iteration)
    assert len(messages) == 4


def test_sh_progressive_interval_validation() -> None:
    opt = Hexels3DOptions()
    opt.source_path = "family"
    opt.model_path = "output"
    opt.sh_progressive_interval = -1
    with pytest.raises(ValueError, match="sh_progressive_interval must be >= 0"):
        opt.process_args()


def test_sh3_model_children_and_checkpoint_contract() -> None:
    model = _make_model(view_basis_dim=15)
    assert model.view_grads.shape == (2, 15, 3)
    model.append_children(torch.tensor([0]), sides="both")
    assert model.child_view_grads.shape == (2, 15, 3)

    restored = _make_model(view_basis_dim=15)
    restored.allocate_children(2)
    restored.load_state_dict(model.state_dict())
    assert torch.equal(restored.child_view_grads, model.child_view_grads)
    assert _checkpoint_view_basis_dim(model.state_dict(), "sh3") == 15
    with pytest.raises(ValueError, match="incompatible with view_model='linear'"):
        _checkpoint_view_basis_dim(model.state_dict(), "linear")


def test_sh3_uses_native_no_view_model() -> None:
    class Rasterizer:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        def __call__(self, **kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
            colors = kwargs["colors"]
            assert isinstance(colors, torch.Tensor)
            return colors, torch.zeros(colors.shape[0])

    renderer = Hexels3DRenderer.__new__(Hexels3DRenderer)
    renderer.opt = SimpleNamespace(
        softness=10.0,
        scale_modifier=1.0,
        color_model="linear",
        view_model="sh3",
        use_view_grads=True,
        debug=False,
    )
    renderer._rasterizer_cls = Rasterizer
    renderer._cache = {}
    camera = SimpleNamespace(
        image_width=16,
        image_height=8,
        FoVx=1.0,
        FoVy=0.8,
        camera_center=torch.tensor([0.0, 0.0, -1.0]),
        world_view_transform=torch.eye(4),
        full_proj_transform=torch.eye(4),
    )

    rasterizer = renderer._get_rasterizer(camera)
    assert rasterizer.kwargs["view_model"] == "none"
    assert rasterizer.kwargs["use_view_grads"] is False

    means3d = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    pack = {
        "means3d": means3d,
        "scales": torch.ones((2, 2)),
        "rotations": torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1),
        "colors": torch.zeros((2, 3)),
        "color_grads": torch.zeros((2, 2, 3)),
        "color_quads": torch.zeros((2, 3, 3)),
        "view_grads": torch.ones((2, 15, 3)),
        "specular_ks": torch.zeros((2, 3)),
        "specular_log_s": torch.zeros(2),
        "opacities": torch.ones(2),
        "softnesses": torch.ones((2, 3)),
    }
    model = SimpleNamespace(pack_for_rasterizer=lambda **kwargs: pack, num_base=2)
    view_delta = means3d - camera.camera_center.unsqueeze(0)
    view_dirs = view_delta * torch.rsqrt(torch.sum(view_delta * view_delta, dim=-1, keepdim=True) + 1e-8)

    default_image = renderer.render(camera, model, torch.zeros(3))["image"]
    degree_one_image = renderer.render(camera, model, torch.zeros(3), sh_degree=1)["image"]
    assert torch.allclose(default_image, _eval_sh3_residual(pack["view_grads"], view_dirs, degree=3))
    assert torch.allclose(degree_one_image, _eval_sh3_residual(pack["view_grads"], view_dirs, degree=1))


def test_linear_view_residual_uses_color_direction_layout() -> None:
    view_grads = torch.tensor(
        [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]],
        requires_grad=True,
    )
    directions = torch.tensor([[2.0, -1.0, 0.5]], requires_grad=True)

    residual = _eval_linear_view_residual(view_grads, directions)
    assert torch.equal(residual, torch.tensor([[1.5, 6.0, 10.5]]))

    residual.sum().backward()
    assert torch.equal(view_grads.grad, directions.detach().unsqueeze(1).expand_as(view_grads))
    assert torch.equal(directions.grad, torch.tensor([[12.0, 15.0, 18.0]]))


def test_linear_view_fast_path_dispatch_and_render() -> None:
    class Rasterizer:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        def __call__(self, **kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
            colors = kwargs["colors"]
            assert isinstance(colors, torch.Tensor)
            return colors, torch.zeros(colors.shape[0])

    opt = SimpleNamespace(
        softness=10.0,
        scale_modifier=1.0,
        color_model="linear",
        view_model="linear",
        use_view_grads=True,
        prefilter_edge_on=False,
        debug=False,
    )
    renderer = Hexels3DRenderer.__new__(Hexels3DRenderer)
    renderer.opt = opt
    renderer._rasterizer_cls = Rasterizer
    renderer._cache = {}
    camera = SimpleNamespace(
        image_width=16,
        image_height=8,
        FoVx=1.0,
        FoVy=0.8,
        camera_center=torch.tensor([0.0, 0.0, -1.0]),
        world_view_transform=torch.eye(4),
        full_proj_transform=torch.eye(4),
    )
    rasterizer = renderer._get_rasterizer(camera)
    assert rasterizer.kwargs["view_model"] == "none"
    assert rasterizer.kwargs["use_view_grads"] is False

    means3d = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    pack = {
        "means3d": means3d,
        "scales": torch.ones((2, 2)),
        "rotations": torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1),
        "colors": torch.full((2, 3), 0.25),
        "color_grads": torch.zeros((2, 2, 3)),
        "color_quads": torch.zeros((2, 3, 3)),
        "view_grads": torch.arange(18, dtype=torch.float32).reshape(2, 3, 3) / 10.0,
        "specular_ks": torch.zeros((2, 3)),
        "specular_log_s": torch.zeros(2),
        "opacities": torch.ones(2),
        "softnesses": torch.ones((2, 3)),
    }
    model = SimpleNamespace(pack_for_rasterizer=lambda **kwargs: pack, num_base=2)
    view_delta = means3d - camera.camera_center.unsqueeze(0)
    view_dirs = view_delta * torch.rsqrt(torch.sum(view_delta.square(), dim=-1, keepdim=True) + 1e-8)
    expected = pack["colors"] + _eval_linear_view_residual(pack["view_grads"], view_dirs)
    assert torch.allclose(renderer.render(camera, model, torch.zeros(3))["image"], expected)

    opt.prefilter_edge_on = True
    renderer._cache = {}
    rasterizer = renderer._get_rasterizer(camera)
    assert rasterizer.kwargs["view_model"] == "linear"
    assert rasterizer.kwargs["use_view_grads"] is True


def test_epoch_camera_sampling_exhausts_each_camera_once() -> None:
    trainer = Hexels3DTrainer.__new__(Hexels3DTrainer)
    trainer.train_cameras = list(range(5))
    trainer.camera_sampling = "epoch"
    trainer._camera_stack = []
    random.seed(7)

    first_epoch = [trainer._next_train_camera() for _ in range(5)]
    second_epoch = [trainer._next_train_camera() for _ in range(5)]
    assert sorted(first_epoch) == list(range(5))
    assert sorted(second_epoch) == list(range(5))


def test_sh3_uses_mode_specific_coefficient_lr() -> None:
    trainer = Hexels3DTrainer.__new__(Hexels3DTrainer)
    trainer.opt = SimpleNamespace(
        lr_colors=0.01,
        lr_opacity=0.1,
        lr_view_grads=0.005,
        lr_sh_coeffs=0.0005,
        enable_children=False,
    )
    trainer.optimize_color_grads = False
    trainer.optimize_color_quads = False
    trainer.optimize_view_grads = True
    trainer.optimize_specular = False
    trainer.optimize_geometry = False
    trainer.optimize_softness = False

    trainer.view_model = "sh3"
    assert dict(trainer._optimizer_group_specs())["view_grads"] == 0.0005
    trainer.view_model = "linear"
    assert dict(trainer._optimizer_group_specs())["view_grads"] == 0.005


def _make_resume_trainer(
    *,
    iterations: int = 12,
    geometry_lr_scale: float = 1.0,
    appearance_lr_scale: float = 1.0,
    optimize_geometry: bool = True,
    lambda_dssim: float = 0.2,
    allow_lambda_dssim_change: bool = False,
    photometric_loss: str = "l1",
    allow_photometric_loss_change: bool = False,
    camera_sampling: str = "epoch",
    allow_camera_sampling_change: bool = False,
) -> Hexels3DTrainer:
    trainer = Hexels3DTrainer.__new__(Hexels3DTrainer)
    trainer.model = _make_model()
    trainer.device = torch.device("cpu")
    trainer.spatial_lr_scale = 1.0
    trainer.opt = SimpleNamespace(
        source_path="scene",
        images=None,
        eval=False,
        llffhold=8,
        resolution=1,
        iterations=iterations,
        seed=17,
        camera_sampling=camera_sampling,
        resume_geometry_lr_scale=geometry_lr_scale,
        resume_appearance_lr_scale=appearance_lr_scale,
        resume_allow_lambda_dssim_change=allow_lambda_dssim_change,
        resume_allow_photometric_loss_change=(
            allow_photometric_loss_change
        ),
        resume_allow_camera_sampling_change=allow_camera_sampling_change,
        resume_train_holdout_stride=0,
        resume_train_camera_manifest=None,
        skip_postprocess=True,
        test_iterations=[],
        enable_children=False,
        train_mode="full",
        softness_mode="iso",
        init_normal_mode="identity",
        spatial_appearance_start=0,
        scale_modifier=1.0,
        color_model="linear",
        use_view_grads=True,
        view_model="linear",
        sh_progressive_interval=0,
        lambda_dssim=lambda_dssim,
        photometric_loss=photometric_loss,
        white_background=False,
        prefilter_edge_on=False,
        repulsion_screen_weight=0.0,
        repulsion_screen_target=1.0,
        repulsion_knn_weight=0.0,
        surface_coplanar_weight=0.0,
        surface_normal_smooth_weight=0.0,
        surface_normal_depth_weight=0.0,
        surface_normal_depth_start=0,
        surface_normal_depth_alpha_thresh=0.01,
        surface_normal_objective="legacy",
        surface_normal_depth_mode="expected",
        lr_means3d=0.01,
        lr_means3d_final=0.0,
        lr_means3d_max_steps=30_000,
        lr_means3d_delay_steps=0,
        lr_means3d_delay_mult=0.01,
        spatial_lr_scale=False,
        lr_scales=0.02,
        lr_rotations=0.03,
        lr_colors=0.04,
        lr_color_grads=0.05,
        lr_view_grads=0.06,
        lr_opacity=0.07,
        lr_softness=0.08,
    )
    trainer.optimize_geometry = optimize_geometry
    trainer.optimize_softness = True
    trainer.optimize_color_grads = True
    trainer.optimize_color_quads = False
    trainer.optimize_view_grads = True
    trainer.optimize_specular = False
    trainer.view_model = "linear"
    trainer.optimizer = trainer._build_optimizer()
    trainer.position_lr_scheduler = None
    trainer.resume_geometry_lr_scale = geometry_lr_scale
    trainer.resume_appearance_lr_scale = appearance_lr_scale
    trainer.resume_allow_lambda_dssim_change = allow_lambda_dssim_change
    trainer.resume_allow_photometric_loss_change = (
        allow_photometric_loss_change
    )
    trainer.resume_allow_camera_sampling_change = allow_camera_sampling_change
    trainer.resume_parent_lambda_dssim = None
    trainer.resume_parent_photometric_loss = None
    trainer.resume_parent_camera_sampling = None
    trainer.resume_parent_train_holdout_stride = 0
    trainer.resume_train_holdout_stride = 0
    trainer.resume_train_camera_manifest_path = None
    trainer.resume_train_camera_manifest_sha256 = None
    trainer.resume_train_camera_names = []
    trainer.resume_train_camera_names_sha256 = None
    trainer.resume_train_camera_trace_names = []
    trainer.resume_train_holdout_cameras = []
    trainer.resume_train_holdout_camera_names = []
    trainer.position_lr_resume_scale = 1.0
    trainer.start_iteration = 0
    trainer.resume_checkpoint_path = None
    trainer.train_cameras = list(range(5))
    trainer.camera_sampling = camera_sampling
    trainer._camera_stack = []
    trainer.logger = logging.getLogger("test_resume")
    return trainer


def _write_resume_checkpoint(path: Path, trainer: Hexels3DTrainer, iteration: int = 5) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    source_config = {
        name: getattr(trainer.opt, name)
        for name in ("source_path", *_RESUME_CONFIG_FIELDS)
    }
    (path.parent.parent / "config.json").write_text(json.dumps(source_config))
    loss = sum(group["params"][0].sum() for group in trainer.optimizer.param_groups)
    loss.backward()
    trainer.optimizer.step()
    for index, group in enumerate(trainer.optimizer.param_groups, start=1):
        group["lr"] = 0.001 * index
    torch.save(
        {
            "iteration": iteration,
            "model": trainer.model.state_dict(),
            "optimizer": trainer.optimizer.state_dict(),
        },
        path,
    )


def _resume_checkpoint_path(tmp_path: Path, name: str = "checkpoint.pt") -> Path:
    return tmp_path / "run" / "checkpoints" / name


def _write_resume_camera_contract(
    run_path: Path,
    manifest_path: Path,
    *,
    resumed_parent: bool,
) -> None:
    names = ["camera_000", "camera_001"]
    names_sha256 = _json_sha256(names)
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "camera_names": names,
                "camera_names_sha256": names_sha256,
            }
        )
    )
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if resumed_parent:
        summary = {
            "resume": {
                "train_camera_manifest_sha256": manifest_sha256,
                "train_camera_names_sha256": names_sha256,
            }
        }
    else:
        summary = {
            "train_camera_selection": {
                "manifest_sha256": manifest_sha256,
                "camera_names_sha256": names_sha256,
            }
        }
    (run_path / "summary.json").write_text(json.dumps(summary))


def test_resume_train_holdout_matches_sorted_stride_gate() -> None:
    cameras = [
        SimpleNamespace(image_name=f"camera_{index:03d}")
        for index in reversed(range(133))
    ]

    train, holdout = _split_resume_train_cameras(cameras, holdout_stride=8)

    assert len(train) == 116
    assert [camera.image_name for camera in holdout] == [
        f"camera_{index:03d}" for index in range(0, 133, 8)
    ]
    assert [camera.image_name for camera in train] == [
        f"camera_{index:03d}" for index in range(133) if index % 8 != 0
    ]
    trainer = _make_resume_trainer()
    trainer.resume_train_holdout_stride = 8
    trainer.resume_train_holdout_camera_names = [camera.image_name for camera in holdout]
    metadata = trainer._resume_metadata()
    assert metadata["train_holdout_stride"] == 8
    assert metadata["train_holdout_camera_names"] == [
        f"camera_{index:03d}" for index in range(0, 133, 8)
    ]


def test_resume_train_camera_manifest_selects_exact_sorted_names(tmp_path: Path) -> None:
    cameras = [
        SimpleNamespace(image_name=name)
        for name in ["camera_003", "camera_001", "camera_002", "camera_000"]
    ]
    names = ["camera_000", "camera_002"]
    manifest_path = tmp_path / "proposal.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "camera_names": names,
                "camera_names_sha256": _json_sha256(names),
            }
        )
    )

    selected, excluded, selected_names, names_sha256, file_sha256 = (
        _select_resume_train_camera_manifest(cameras, manifest_path=manifest_path)
    )

    assert [camera.image_name for camera in selected] == names
    assert [camera.image_name for camera in excluded] == ["camera_001", "camera_003"]
    assert selected_names == names
    assert names_sha256 == _json_sha256(names)
    assert file_sha256 == hashlib.sha256(manifest_path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    ("names", "digest", "message"),
    [
        (["camera_002", "camera_000"], None, "sorted and unique"),
        (["camera_000", "camera_000"], None, "sorted and unique"),
        (["camera_000"], "0" * 64, "camera_names_sha256 mismatch"),
        (["camera_999"], None, "unknown camera names"),
    ],
)
def test_resume_train_camera_manifest_rejects_invalid_names(
    tmp_path: Path,
    names: list[str],
    digest: str | None,
    message: str,
) -> None:
    manifest_path = tmp_path / "proposal.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "camera_names": names,
                "camera_names_sha256": digest or _json_sha256(names),
            }
        )
    )
    cameras = [SimpleNamespace(image_name=f"camera_{index:03d}") for index in range(4)]

    with pytest.raises(ValueError, match=message):
        _select_resume_train_camera_manifest(cameras, manifest_path=manifest_path)


def test_resume_train_camera_manifest_metadata_and_trace(tmp_path: Path) -> None:
    trainer = _make_resume_trainer()
    names = ["camera_000", "camera_002"]
    trainer.resume_train_camera_manifest_path = (tmp_path / "proposal.json").resolve()
    trainer.resume_train_camera_manifest_sha256 = "a" * 64
    trainer.resume_train_camera_names = names
    trainer.resume_train_camera_names_sha256 = _json_sha256(names)
    trainer._record_resume_train_camera(SimpleNamespace(image_name="camera_002"))

    metadata = trainer._resume_metadata()

    assert metadata["train_camera_manifest"] == str(trainer.resume_train_camera_manifest_path)
    assert metadata["train_camera_manifest_sha256"] == "a" * 64
    assert metadata["train_camera_count"] == 2
    assert metadata["train_camera_names"] == names
    assert metadata["train_camera_names_sha256"] == _json_sha256(names)
    assert metadata["train_camera_trace_count"] == 1
    assert metadata["train_camera_trace_sha256"] == _json_sha256(["camera_002"])
    with pytest.raises(RuntimeError, match="outside"):
        trainer._record_resume_train_camera(SimpleNamespace(image_name="camera_001"))


def test_fresh_train_camera_trace_is_recorded() -> None:
    trainer = _make_resume_trainer()
    trainer.train_camera_trace_names = []
    trainer._record_train_camera(SimpleNamespace(image_name="camera_002"))

    assert trainer.train_camera_trace_names == ["camera_002"]
    assert trainer.resume_train_camera_trace_names == []


def test_progress_render_keeps_resume_holdout_unopened() -> None:
    trainer = Hexels3DTrainer.__new__(Hexels3DTrainer)
    train_camera = object()
    test_camera = object()
    holdout_camera = object()
    trainer.train_cameras = [train_camera]
    trainer.test_cameras = [test_camera]
    trainer.resume_train_holdout_cameras = []

    assert trainer._progress_render_camera() is test_camera
    trainer.test_cameras = []
    assert trainer._progress_render_camera() is train_camera

    trainer.test_cameras = [test_camera]
    trainer.resume_train_holdout_cameras = [holdout_camera]
    assert trainer._progress_render_camera() is train_camera

@pytest.mark.parametrize(
    ("skip_postprocess", "test_iterations", "message"),
    [
        (False, [], "requires skip_postprocess=True"),
        (True, [10], "requires empty test_iterations"),
    ],
)
def test_resume_loader_enforces_blind_holdout_policy(
    tmp_path: Path,
    skip_postprocess: bool,
    test_iterations: list[int],
    message: str,
) -> None:
    source = _make_resume_trainer()
    checkpoint_path = _resume_checkpoint_path(tmp_path)
    _write_resume_checkpoint(checkpoint_path, source)
    resumed = _make_resume_trainer()
    resumed.resume_train_holdout_stride = 8
    resumed.opt.resume_train_holdout_stride = 8
    resumed.opt.skip_postprocess = skip_postprocess
    resumed.opt.test_iterations = test_iterations

    with pytest.raises(ValueError, match=message):
        resumed._load_resume_checkpoint(checkpoint_path)


@pytest.mark.parametrize("source_stride", [None, 0])
def test_resume_allows_first_holdout_opt_in(tmp_path: Path, source_stride: int | None) -> None:
    source = _make_resume_trainer()
    checkpoint_path = _resume_checkpoint_path(tmp_path)
    _write_resume_checkpoint(checkpoint_path, source)
    config_path = checkpoint_path.parent.parent / "config.json"
    source_config = json.loads(config_path.read_text())
    if source_stride is not None:
        source_config["resume_train_holdout_stride"] = source_stride
    config_path.write_text(json.dumps(source_config))

    resumed = _make_resume_trainer()
    resumed.resume_train_holdout_stride = 8
    resumed.opt.resume_train_holdout_stride = 8
    resumed._load_resume_checkpoint(checkpoint_path)
    assert resumed._resume_metadata()["parent_train_holdout_stride"] == 0


@pytest.mark.parametrize("current_stride", [0, 4])
def test_resume_rejects_disabling_or_changing_sticky_holdout(
    tmp_path: Path,
    current_stride: int,
) -> None:
    source = _make_resume_trainer()
    checkpoint_path = _resume_checkpoint_path(tmp_path)
    _write_resume_checkpoint(checkpoint_path, source)
    config_path = checkpoint_path.parent.parent / "config.json"
    source_config = json.loads(config_path.read_text())
    source_config["resume_train_holdout_stride"] = 8
    config_path.write_text(json.dumps(source_config))

    resumed = _make_resume_trainer()
    resumed.resume_train_holdout_stride = current_stride
    resumed.opt.resume_train_holdout_stride = current_stride
    with pytest.raises(
        ValueError,
        match=rf"holdout stride is sticky once enabled: checkpoint=8, current={current_stride}",
    ):
        resumed._load_resume_checkpoint(checkpoint_path)


def test_resume_preserves_sticky_holdout_stride(tmp_path: Path) -> None:
    source = _make_resume_trainer()
    checkpoint_path = _resume_checkpoint_path(tmp_path)
    _write_resume_checkpoint(checkpoint_path, source)
    config_path = checkpoint_path.parent.parent / "config.json"
    source_config = json.loads(config_path.read_text())
    source_config["resume_train_holdout_stride"] = 8
    config_path.write_text(json.dumps(source_config))

    resumed = _make_resume_trainer()
    resumed.resume_train_holdout_stride = 8
    resumed.opt.resume_train_holdout_stride = 8
    resumed._load_resume_checkpoint(checkpoint_path)
    assert resumed._resume_metadata()["parent_train_holdout_stride"] == 8


def test_resume_restores_model_adam_and_scales_geometry_and_appearance_lrs(tmp_path: Path) -> None:
    source = _make_resume_trainer()
    checkpoint_path = _resume_checkpoint_path(tmp_path)
    _write_resume_checkpoint(checkpoint_path, source)

    resumed = _make_resume_trainer(geometry_lr_scale=0.25, appearance_lr_scale=0.4)
    resumed._load_resume_checkpoint(checkpoint_path)

    for name, value in source.model.state_dict().items():
        assert torch.equal(resumed.model.state_dict()[name], value)
    for source_group, resumed_group in zip(source.optimizer.param_groups, resumed.optimizer.param_groups):
        source_param = source_group["params"][0]
        resumed_param = resumed_group["params"][0]
        for state_name, source_value in source.optimizer.state[source_param].items():
            resumed_value = resumed.optimizer.state[resumed_param][state_name]
            if isinstance(source_value, torch.Tensor):
                assert torch.equal(resumed_value, source_value)
            else:
                assert resumed_value == source_value

    source_lrs = source._optimizer_lr_map()
    resumed_lrs = resumed._optimizer_lr_map()
    geometry_names = {"means3d", "log_scales", "rotation_quats"}
    for name, source_lr in source_lrs.items():
        scale = 0.25 if name in geometry_names else 0.4
        assert resumed_lrs[name] == pytest.approx(source_lr * scale)


def test_resume_loads_legacy_checkpoint_as_all_active(tmp_path: Path) -> None:
    source = _make_resume_trainer()
    checkpoint_path = _resume_checkpoint_path(tmp_path)
    _write_resume_checkpoint(checkpoint_path, source)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint["model"].pop("active_mask")
    torch.save(checkpoint, checkpoint_path)

    resumed = _make_resume_trainer()
    resumed._load_resume_checkpoint(checkpoint_path)

    assert torch.equal(resumed.model.active_mask, torch.ones_like(resumed.model.active_mask))


def test_resume_preserves_active_mask(tmp_path: Path) -> None:
    source = _make_resume_trainer()
    source.model.active_mask[1] = False
    checkpoint_path = _resume_checkpoint_path(tmp_path)
    _write_resume_checkpoint(checkpoint_path, source)

    resumed = _make_resume_trainer()
    resumed._load_resume_checkpoint(checkpoint_path)

    assert torch.equal(resumed.model.active_mask, source.model.active_mask)


def test_resumed_save_preserves_active_pruning_lineage(tmp_path: Path) -> None:
    source = _make_resume_trainer()
    checkpoint_path = _resume_checkpoint_path(tmp_path)
    _write_resume_checkpoint(checkpoint_path, source)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint["schema_version"] = 1
    checkpoint["checkpoint_type"] = "hexels3d_active_mask"
    checkpoint["edge_transfer"] = {
        "complete": True,
        "parent_ids": [0],
        "child_ids": [1],
    }
    checkpoint["lineage"] = {
        "kind": "hexels3d_phase1_active_prune",
        "plan_manifest_sha256": "a" * 64,
        "candidate_original_ids": torch.tensor([1]),
    }
    torch.save(checkpoint, checkpoint_path)

    resumed = _make_resume_trainer()
    resumed._load_resume_checkpoint(checkpoint_path)
    resumed.model_path = tmp_path / "continued"
    (resumed.model_path / "checkpoints").mkdir(parents=True)
    resumed._save_checkpoint(6)

    saved = torch.load(
        resumed.model_path / "checkpoints" / "ckpt_000006.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert saved["schema_version"] == 1
    assert saved["checkpoint_type"] == "hexels3d_active_mask"
    assert saved["edge_transfer"] == checkpoint["edge_transfer"]
    assert saved["lineage"]["kind"] == "hexels3d_phase1_active_prune"
    assert saved["lineage"]["plan_manifest_sha256"] == "a" * 64
    assert torch.equal(saved["lineage"]["candidate_original_ids"], torch.tensor([1]))
    continuation = saved["lineage"]["continuations"][-1]
    assert continuation["parent_checkpoint"] == str(checkpoint_path.resolve())
    assert continuation["parent_checkpoint_sha256"] == resumed.resume_checkpoint_sha256
    assert continuation["start_iteration"] == 5
    assert continuation["saved_iteration"] == 6


def test_resumed_save_creates_lineage_for_plain_parent(tmp_path: Path) -> None:
    source = _make_resume_trainer()
    checkpoint_path = _resume_checkpoint_path(tmp_path)
    _write_resume_checkpoint(checkpoint_path, source)

    resumed = _make_resume_trainer()
    resumed.resume_geometry_policy = "inherit"
    resumed.resume_scale_lower_log = None
    resumed.resume_scale_upper_log = None
    resumed.resume_scale_constraint_metadata = None
    resumed._load_resume_checkpoint(checkpoint_path)
    resumed.model_path = tmp_path / "plain_continuation"
    (resumed.model_path / "checkpoints").mkdir(parents=True)
    resumed._save_checkpoint(6)

    saved = torch.load(
        resumed.model_path / "checkpoints" / "ckpt_000006.pt",
        map_location="cpu",
        weights_only=True,
    )
    lineage = saved["lineage"]
    assert lineage["kind"] == "hexels3d_resume_root"
    assert lineage["parent_checkpoint"] == str(checkpoint_path.resolve())
    assert lineage["parent_checkpoint_sha256"] == resumed.resume_checkpoint_sha256
    continuation = lineage["continuations"][-1]
    assert continuation["parent_checkpoint"] == str(checkpoint_path.resolve())
    assert continuation["start_iteration"] == 5
    assert continuation["saved_iteration"] == 6


def test_mvs_surface_metadata_is_sticky_across_second_resume(
    tmp_path: Path,
) -> None:
    names = ["camera"]
    sidecar = tmp_path / "mvs.pt"
    torch.save(
        {
            "schema_version": 1,
            "kind": "mvs_surface_targets",
            "camera_names": names,
            "depth": torch.ones((1, 2, 2)),
            "source_count": torch.ones((1, 2, 2), dtype=torch.uint8),
            "metadata": {
                "camera_names_sha256": hashlib.sha256(
                    json.dumps(names, separators=(",", ":")).encode()
                ).hexdigest(),
                "minimum_source_count": 1,
                "source_count_cap": 20,
                "maximum_camera_depth": 100.0,
                "resize_mode": "camera-intrinsics-nearest",
            },
        },
        sidecar,
    )
    source = _make_resume_trainer()
    parent = _resume_checkpoint_path(tmp_path)
    _write_resume_checkpoint(parent, source)

    first = _make_resume_trainer()
    first.resume_mvs_surface_path = sidecar.resolve()
    first.resume_mvs_surface_weight = 0.05
    first.resume_mvs_surface_targets = None
    first.resume_mvs_surface_metadata = None
    first.resume_checkpoint_mvs_surface = None
    first.train_cameras = [SimpleNamespace(image_name="camera")]
    first._load_resume_checkpoint(parent)
    first._configure_resume_mvs_surface()
    first.model_path = tmp_path / "mvs_first"
    (first.model_path / "checkpoints").mkdir(parents=True)
    first._save_checkpoint(6)
    first_checkpoint = first.model_path / "checkpoints/ckpt_000006.pt"
    continued_config = {
        name: getattr(first.opt, name)
        for name in ("source_path", *_RESUME_CONFIG_FIELDS)
    }
    continued_config.update(
        {
            "resume_mvs_surface_path": str(sidecar.resolve()),
            "resume_mvs_surface_weight": 0.05,
        }
    )
    (first.model_path / "config.json").write_text(json.dumps(continued_config))

    second = _make_resume_trainer(iterations=8)
    second.resume_mvs_surface_path = sidecar.resolve()
    second.resume_mvs_surface_weight = 0.05
    second.resume_mvs_surface_targets = None
    second.resume_mvs_surface_metadata = None
    second.resume_checkpoint_mvs_surface = None
    second.train_cameras = [SimpleNamespace(image_name="camera")]
    second._load_resume_checkpoint(first_checkpoint)
    second._configure_resume_mvs_surface()
    second.model_path = tmp_path / "mvs_second"
    (second.model_path / "checkpoints").mkdir(parents=True)
    second._save_checkpoint(7)

    saved = torch.load(
        second.model_path / "checkpoints/ckpt_000007.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert saved["resume_mvs_surface"] == second._resume_mvs_surface_summary()
    assert len(saved["lineage"]["continuations"]) == 2
    assert all(
        continuation["mvs_surface"] == saved["resume_mvs_surface"]
        for continuation in saved["lineage"]["continuations"]
    )


@pytest.mark.parametrize("complete", [False, True])
def test_edge_transfer_checkpoint_resume_contract(tmp_path: Path, complete: bool) -> None:
    source = _make_resume_trainer()
    checkpoint_path = _resume_checkpoint_path(tmp_path)
    _write_resume_checkpoint(checkpoint_path, source)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint["lineage"] = {"kind": "hexels3d_edge_transfer_birth"}
    torch.save(checkpoint, checkpoint_path)

    resumed = _make_resume_trainer()
    resumed._load_resume_checkpoint(checkpoint_path)
    resumed.model_path = tmp_path / f"edge_{complete}"
    (resumed.model_path / "checkpoints").mkdir(parents=True)

    class Controller:
        def is_complete(self) -> bool:
            return complete

        def state(self) -> dict[str, object]:
            return {"complete": complete}

    resumed.edge_transfer_controller = Controller()
    resumed._save_checkpoint(6)

    saved = torch.load(
        resumed.model_path / "checkpoints" / "ckpt_000006.pt",
        map_location="cpu",
        weights_only=True,
    )
    suffix = "complete" if complete else "incomplete"
    assert saved["resume_compatible"] is complete
    assert saved["checkpoint_type"] == f"hexels3d_edge_transfer_{suffix}"
    assert saved["lineage"]["kind"] == saved["checkpoint_type"]
    assert saved["lineage"]["status"] == "ungated"
    assert saved["lineage"]["parent_lineage"]["kind"] == "hexels3d_edge_transfer_birth"


def test_resume_preserves_and_compounds_scheduled_position_lr_scale(tmp_path: Path) -> None:
    source = _make_resume_trainer()
    checkpoint_path = _resume_checkpoint_path(tmp_path)
    _write_resume_checkpoint(checkpoint_path, source)

    resumed = _make_resume_trainer(geometry_lr_scale=0.25)
    resumed.position_lr_scheduler = lambda iteration: 0.004 if iteration == 5 else 0.003
    resumed._load_resume_checkpoint(checkpoint_path)
    assert resumed.position_lr_resume_scale == pytest.approx(0.0625)
    resumed._update_position_lr(6)
    assert resumed._optimizer_lr_map()["means3d"] == pytest.approx(0.0001875)

    chained_path = checkpoint_path.parent / "chained.pt"
    torch.save(
        {
            "iteration": 6,
            "model": resumed.model.state_dict(),
            "optimizer": resumed.optimizer.state_dict(),
            "resume_compatible": True,
        },
        chained_path,
    )
    chained = _make_resume_trainer(geometry_lr_scale=0.5)
    chained.position_lr_scheduler = lambda iteration: 0.003 if iteration == 6 else 0.002
    chained._load_resume_checkpoint(chained_path)
    assert chained.position_lr_resume_scale == pytest.approx(0.03125)
    chained._update_position_lr(7)
    assert chained._optimizer_lr_map()["means3d"] == pytest.approx(0.0000625)


def test_scheduled_base_mode_resume_has_no_position_group(tmp_path: Path) -> None:
    source = _make_resume_trainer(optimize_geometry=False)
    checkpoint_path = _resume_checkpoint_path(tmp_path)
    _write_resume_checkpoint(checkpoint_path, source)

    resumed = _make_resume_trainer(optimize_geometry=False, appearance_lr_scale=0.5)
    resumed.position_lr_scheduler = lambda iteration: 0.01
    resumed._load_resume_checkpoint(checkpoint_path)

    assert "means3d" not in resumed._optimizer_lr_map()
    resumed._update_position_lr(6)
    for name, lr in resumed._optimizer_lr_map().items():
        assert lr == pytest.approx(source._optimizer_lr_map()[name] * 0.5)


def test_resume_rejects_optimizer_groups_model_shape_and_nonpositive_target(tmp_path: Path) -> None:
    source = _make_resume_trainer()
    checkpoint_path = _resume_checkpoint_path(tmp_path)
    _write_resume_checkpoint(checkpoint_path, source)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    checkpoint["optimizer"]["param_groups"][0]["name"] = "wrong"
    bad_groups_path = checkpoint_path.parent / "bad_groups.pt"
    torch.save(checkpoint, bad_groups_path)
    with pytest.raises(ValueError, match="optimizer groups do not match in order"):
        _make_resume_trainer()._load_resume_checkpoint(bad_groups_path)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint["model"]["colors"] = checkpoint["model"]["colors"][:1]
    bad_shape_path = checkpoint_path.parent / "bad_shape.pt"
    torch.save(checkpoint, bad_shape_path)
    with pytest.raises(ValueError, match="fixed-population/shape-compatible"):
        _make_resume_trainer()._load_resume_checkpoint(bad_shape_path)

    with pytest.raises(ValueError, match="must exceed checkpoint iteration"):
        _make_resume_trainer(iterations=5)._load_resume_checkpoint(checkpoint_path)


def test_resume_rejects_source_config_mismatch(tmp_path: Path) -> None:
    source_dir = tmp_path / "source_run"
    checkpoint_dir = source_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    source = _make_resume_trainer()
    checkpoint_path = checkpoint_dir / "checkpoint.pt"
    _write_resume_checkpoint(checkpoint_path, source)
    source_config = {
        name: getattr(source.opt, name)
        for name in ("source_path", *_RESUME_CONFIG_FIELDS)
    }
    source_config["source_path"] = "different_scene"
    (source_dir / "config.json").write_text(json.dumps(source_config))

    resumed = _make_resume_trainer()
    resumed.opt.source_path = "current_scene"
    with pytest.raises(ValueError, match="source_path does not match"):
        resumed._load_resume_checkpoint(checkpoint_path)


def test_resume_rejects_lambda_dssim_change_by_default(tmp_path: Path) -> None:
    source = _make_resume_trainer()
    checkpoint_path = _resume_checkpoint_path(tmp_path)
    _write_resume_checkpoint(checkpoint_path, source)

    resumed = _make_resume_trainer(lambda_dssim=0.35)
    with pytest.raises(ValueError, match=r"lambda_dssim: checkpoint=0\.2, current=0\.35"):
        resumed._load_resume_checkpoint(checkpoint_path)


def test_resume_allows_only_explicit_lambda_dssim_change(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    source = _make_resume_trainer()
    checkpoint_path = _resume_checkpoint_path(tmp_path)
    _write_resume_checkpoint(checkpoint_path, source)

    resumed = _make_resume_trainer(
        lambda_dssim=0.35,
        allow_lambda_dssim_change=True,
    )
    with caplog.at_level(logging.INFO, logger="test_resume"):
        resumed._load_resume_checkpoint(checkpoint_path)

    metadata = resumed._resume_metadata()
    assert metadata["allow_lambda_dssim_change"] is True
    assert metadata["parent_lambda_dssim"] == pytest.approx(0.2)
    assert metadata["current_lambda_dssim"] == pytest.approx(0.35)
    assert "lambda_dssim(parent=0.2 current=0.35 change_allowed=True)" in caplog.text


def test_resume_rejects_photometric_loss_change_by_default(
    tmp_path: Path,
) -> None:
    source = _make_resume_trainer()
    checkpoint_path = _resume_checkpoint_path(tmp_path)
    _write_resume_checkpoint(checkpoint_path, source)

    resumed = _make_resume_trainer(photometric_loss="l2")
    with pytest.raises(
        ValueError,
        match="photometric_loss mismatch",
    ):
        resumed._load_resume_checkpoint(checkpoint_path)


def test_resume_records_explicit_photometric_loss_change(
    tmp_path: Path,
) -> None:
    source = _make_resume_trainer()
    checkpoint_path = _resume_checkpoint_path(tmp_path)
    _write_resume_checkpoint(checkpoint_path, source)

    resumed = _make_resume_trainer(
        photometric_loss="l2",
        allow_photometric_loss_change=True,
    )
    resumed._load_resume_checkpoint(checkpoint_path)

    metadata = resumed._resume_metadata()
    assert metadata["allow_photometric_loss_change"] is True
    assert metadata["parent_photometric_loss"] == "l1"
    assert metadata["current_photometric_loss"] == "l2"


def test_resume_rejects_camera_sampling_change_by_default(tmp_path: Path) -> None:
    source = _make_resume_trainer(camera_sampling="random")
    checkpoint_path = _resume_checkpoint_path(tmp_path)
    _write_resume_checkpoint(checkpoint_path, source)

    resumed = _make_resume_trainer(camera_sampling="epoch")
    with pytest.raises(
        ValueError,
        match=r"camera_sampling: checkpoint='random', current='epoch'",
    ):
        resumed._load_resume_checkpoint(checkpoint_path)


def test_resume_allows_camera_sampling_change_and_replays_current_sampler(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    source = _make_resume_trainer(camera_sampling="random")
    checkpoint_path = _resume_checkpoint_path(tmp_path)
    _write_resume_checkpoint(checkpoint_path, source)

    resumed = _make_resume_trainer(
        camera_sampling="epoch",
        allow_camera_sampling_change=True,
    )
    with caplog.at_level(logging.INFO, logger="test_resume"):
        resumed._load_resume_checkpoint(checkpoint_path)

    metadata = resumed._resume_metadata()
    assert metadata["allow_camera_sampling_change"] is True
    assert metadata["parent_camera_sampling"] == "random"
    assert metadata["current_camera_sampling"] == "epoch"
    assert (
        "camera_sampling(parent=random current=epoch change_allowed=True)"
        in caplog.text
    )

    reference = _make_resume_trainer(camera_sampling="epoch")
    reference._replay_camera_sampler(resumed.start_iteration)
    expected = [reference._next_train_camera() for _ in range(10)]
    resumed._replay_camera_sampler(resumed.start_iteration)
    actual = [resumed._next_train_camera() for _ in range(10)]
    assert actual == expected


def test_resume_camera_sampling_override_rejects_other_mismatch(tmp_path: Path) -> None:
    source = _make_resume_trainer(camera_sampling="random")
    checkpoint_path = _resume_checkpoint_path(tmp_path)
    _write_resume_checkpoint(checkpoint_path, source)

    resumed = _make_resume_trainer(
        camera_sampling="epoch",
        allow_camera_sampling_change=True,
    )
    resumed.opt.white_background = True
    with pytest.raises(ValueError, match=r"white_background: checkpoint=False, current=True"):
        resumed._load_resume_checkpoint(checkpoint_path)


def test_resume_lambda_override_rejects_any_other_config_mismatch(tmp_path: Path) -> None:
    source = _make_resume_trainer()
    checkpoint_path = _resume_checkpoint_path(tmp_path)
    _write_resume_checkpoint(checkpoint_path, source)

    resumed = _make_resume_trainer(
        lambda_dssim=0.35,
        allow_lambda_dssim_change=True,
    )
    resumed.opt.white_background = True
    with pytest.raises(ValueError, match=r"white_background: checkpoint=False, current=True"):
        resumed._load_resume_checkpoint(checkpoint_path)


def test_resume_requires_complete_source_config(tmp_path: Path) -> None:
    source = _make_resume_trainer()
    checkpoint_path = _resume_checkpoint_path(tmp_path)
    _write_resume_checkpoint(checkpoint_path, source)
    (checkpoint_path.parent.parent / "config.json").write_text(
        json.dumps({"source_path": source.opt.source_path})
    )

    with pytest.raises(ValueError, match="missing compatibility fields"):
        _make_resume_trainer()._load_resume_checkpoint(checkpoint_path)


def test_geometry_start_iteration_is_resume_sticky(tmp_path: Path) -> None:
    source = _make_resume_trainer()
    checkpoint_path = _resume_checkpoint_path(tmp_path)
    _write_resume_checkpoint(checkpoint_path, source)
    config_path = checkpoint_path.parent.parent / "config.json"
    source_config = json.loads(config_path.read_text())
    source_config["geometry_start_iteration"] = 10
    config_path.write_text(json.dumps(source_config))
    resumed = _make_resume_trainer()
    resumed.geometry_start_iteration = 11

    with pytest.raises(ValueError, match="sticky across resume"):
        resumed._load_resume_checkpoint(checkpoint_path)


def test_constrained_resume_requires_parent_camera_manifest(tmp_path: Path) -> None:
    source = _make_resume_trainer()
    checkpoint_path = _resume_checkpoint_path(tmp_path)
    _write_resume_checkpoint(checkpoint_path, source)
    config_path = checkpoint_path.parent.parent / "config.json"
    source_config = json.loads(config_path.read_text())
    source_config["train_camera_manifest"] = str(tmp_path / "proposal.json")
    config_path.write_text(json.dumps(source_config))

    resumed = _make_resume_trainer()
    resumed.resume_geometry_policy = "frozen"

    with pytest.raises(ValueError, match="same training camera manifest"):
        resumed._load_resume_checkpoint(checkpoint_path)


def test_constrained_resume_rejects_changed_camera_manifest_bytes(tmp_path: Path) -> None:
    source = _make_resume_trainer()
    checkpoint_path = _resume_checkpoint_path(tmp_path)
    _write_resume_checkpoint(checkpoint_path, source)
    manifest_path = tmp_path / "proposal.json"
    _write_resume_camera_contract(
        checkpoint_path.parent.parent,
        manifest_path,
        resumed_parent=False,
    )
    config_path = checkpoint_path.parent.parent / "config.json"
    source_config = json.loads(config_path.read_text())
    source_config["train_camera_manifest"] = str(manifest_path)
    config_path.write_text(json.dumps(source_config))
    manifest_path.write_text(manifest_path.read_text() + "\n")

    resumed = _make_resume_trainer()
    resumed.resume_geometry_policy = "frozen"
    resumed.resume_train_camera_manifest_path = manifest_path.resolve()

    with pytest.raises(ValueError, match="content differs"):
        resumed._load_resume_checkpoint(checkpoint_path)


def test_first_bounded_resume_clears_inherited_scale_adam_state(tmp_path: Path) -> None:
    source = _make_resume_trainer()
    checkpoint_path = _resume_checkpoint_path(tmp_path)
    _write_resume_checkpoint(checkpoint_path, source)
    manifest_path = tmp_path / "proposal.json"
    _write_resume_camera_contract(
        checkpoint_path.parent.parent,
        manifest_path,
        resumed_parent=False,
    )
    config_path = checkpoint_path.parent.parent / "config.json"
    source_config = json.loads(config_path.read_text())
    source_config["train_camera_manifest"] = str(manifest_path)
    config_path.write_text(json.dumps(source_config))

    resumed = _make_resume_trainer()
    resumed.resume_geometry_policy = "bounded_isotropic_shrink"
    resumed.resume_train_camera_manifest_path = manifest_path.resolve()
    resumed.geometry_start_iteration = 0
    resumed._load_resume_checkpoint(checkpoint_path)

    assert resumed.optimizer.state.get(resumed.model.log_scales, {}) == {}


def test_bounded_resume_keeps_original_bounds_across_chain(tmp_path: Path) -> None:
    source = _make_resume_trainer()
    checkpoint_path = _resume_checkpoint_path(tmp_path)
    _write_resume_checkpoint(checkpoint_path, source)
    parent = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = parent["model"]
    upper_scales = torch.exp(state["log_scales"])
    lower_scales = upper_scales * 0.5
    lower_path = tmp_path / "lower.pt"
    upper_path = tmp_path / "upper.pt"
    for path, scales in ((lower_path, lower_scales), (upper_path, upper_scales)):
        torch.save(
            {
                "means3d": state["means3d"],
                "scales": scales,
                "rotations": state["rotation_quats"],
                "colors": torch.full((state["means3d"].shape[0], 3), 0.5),
            },
            path,
        )
    manifest_path = tmp_path / "proposal.json"
    _write_resume_camera_contract(
        checkpoint_path.parent.parent,
        manifest_path,
        resumed_parent=False,
    )
    config_path = checkpoint_path.parent.parent / "config.json"
    source_config = json.loads(config_path.read_text())
    source_config["train_camera_manifest"] = str(manifest_path)
    config_path.write_text(json.dumps(source_config))

    first = _make_resume_trainer()
    first.resume_geometry_policy = "bounded_isotropic_shrink"
    first.resume_scale_lower_geometry_path = lower_path.resolve()
    first.resume_scale_upper_geometry_path = upper_path.resolve()
    first.resume_scale_anchor_weight = 1e-3
    first.resume_scale_lower_log = None
    first.resume_scale_upper_log = None
    first.resume_scale_constraint_metadata = None
    first.resume_checkpoint_scale_constraint = None
    first.resume_train_camera_manifest_path = manifest_path.resolve()
    first.geometry_start_iteration = 0
    first._load_resume_checkpoint(checkpoint_path)
    first._configure_resume_scale_constraint()
    with torch.no_grad():
        midpoint = 0.5 * (first.resume_scale_lower_log + first.resume_scale_upper_log)
        first.model.log_scales.copy_(midpoint.expand_as(first.model.log_scales))
    first.model_path = tmp_path / "bounded_first"
    (first.model_path / "checkpoints").mkdir(parents=True)
    first._save_checkpoint(6)

    continued_config = {
        name: getattr(first.opt, name)
        for name in ("source_path", *_RESUME_CONFIG_FIELDS)
    }
    continued_config.update(
        {
            "geometry_start_iteration": 0,
            "train_camera_manifest": None,
            "resume_train_camera_manifest": str(manifest_path),
            "resume_geometry_policy": "bounded_isotropic_shrink",
            "resume_scale_lower_geometry_path": str(lower_path),
            "resume_scale_upper_geometry_path": str(upper_path),
            "resume_scale_anchor_weight": 1e-3,
        }
    )
    (first.model_path / "config.json").write_text(json.dumps(continued_config))
    _write_resume_camera_contract(
        first.model_path,
        manifest_path,
        resumed_parent=True,
    )

    second = _make_resume_trainer(iterations=8)
    second.resume_geometry_policy = "bounded_isotropic_shrink"
    second.resume_scale_lower_geometry_path = lower_path.resolve()
    second.resume_scale_upper_geometry_path = upper_path.resolve()
    second.resume_scale_anchor_weight = 1e-3
    second.resume_scale_lower_log = None
    second.resume_scale_upper_log = None
    second.resume_scale_constraint_metadata = None
    second.resume_checkpoint_scale_constraint = None
    second.resume_train_camera_manifest_path = manifest_path.resolve()
    second.geometry_start_iteration = 0
    second.opt.resume_scale_lower_geometry_path = str(lower_path)
    second.opt.resume_scale_upper_geometry_path = str(upper_path)
    child_path = first.model_path / "checkpoints" / "ckpt_000006.pt"
    second._load_resume_checkpoint(child_path)
    second._configure_resume_scale_constraint()

    expected_upper = torch.log(upper_scales).mean(dim=1, keepdim=True)
    assert torch.allclose(second.resume_scale_upper_log, expected_upper)
    assert torch.allclose(
        second.model.log_scales,
        midpoint.expand_as(second.model.log_scales),
    )

    corrupted = torch.load(child_path, map_location="cpu", weights_only=True)
    corrupted.pop("resume_scale_constraint")
    corrupt_path = child_path.with_name("ckpt_000006_missing_constraint.pt")
    torch.save(corrupted, corrupt_path)
    third = _make_resume_trainer(iterations=8)
    third.resume_geometry_policy = "bounded_isotropic_shrink"
    third.resume_scale_lower_geometry_path = lower_path.resolve()
    third.resume_scale_upper_geometry_path = upper_path.resolve()
    third.resume_scale_anchor_weight = 1e-3
    third.resume_train_camera_manifest_path = manifest_path.resolve()
    third.geometry_start_iteration = 0
    third.opt.resume_scale_lower_geometry_path = str(lower_path)
    third.opt.resume_scale_upper_geometry_path = str(upper_path)
    with pytest.raises(ValueError, match="missing scale constraint metadata"):
        third._load_resume_checkpoint(corrupt_path)


def test_resume_options_reject_missing_path_dynamic_modes_and_nonpositive_scale(tmp_path: Path) -> None:
    def make_options() -> Hexels3DOptions:
        opt = Hexels3DOptions()
        opt.source_path = "scene"
        opt.model_path = str(tmp_path / "output")
        return opt

    opt = make_options()
    opt.resume_checkpoint = str(tmp_path / "missing.pt")
    with pytest.raises(ValueError, match="does not exist"):
        opt.process_args()

    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.touch()
    opt = make_options()
    opt.resume_checkpoint = str(checkpoint_path)
    opt.enable_children = True
    with pytest.raises(ValueError, match="fixed-population deterministic"):
        opt.process_args()

    opt = make_options()
    opt.resume_checkpoint = str(checkpoint_path)
    opt.enable_densify = True
    opt.densify_impl = "hexels_safe"
    opt.densify_action = "edge_child"
    opt.opacity_reset_interval = 0
    opt.process_args()

    opt.edge_transfer_manifest = str(tmp_path / "manifest.pt")
    with pytest.raises(ValueError, match="incompatible with --enable_densify"):
        opt.process_args()

    opt = make_options()
    opt.resume_geometry_lr_scale = 0.0
    with pytest.raises(ValueError, match="must be > 0"):
        opt.process_args()

    opt = make_options()
    opt.resume_appearance_lr_scale = 0.0
    with pytest.raises(ValueError, match="must be > 0"):
        opt.process_args()

    opt = make_options()
    opt.resume_allow_lambda_dssim_change = True
    with pytest.raises(ValueError, match="requires --resume_checkpoint"):
        opt.process_args()

    opt = make_options()
    opt.resume_allow_camera_sampling_change = True
    with pytest.raises(ValueError, match="requires --resume_checkpoint"):
        opt.process_args()

    opt = make_options()
    opt.resume_train_holdout_stride = 8
    with pytest.raises(ValueError, match="requires --resume_checkpoint"):
        opt.process_args()

    opt = make_options()
    opt.resume_train_holdout_stride = 1
    with pytest.raises(ValueError, match="must be 0 or at least 2"):
        opt.process_args()

    opt = make_options()
    opt.resume_checkpoint = str(checkpoint_path)
    opt.resume_train_holdout_stride = 8
    with pytest.raises(ValueError, match="requires --skip_postprocess"):
        opt.process_args()

    opt.skip_postprocess = True
    opt.test_iterations = [10]
    with pytest.raises(ValueError, match="requires an empty --test_iterations"):
        opt.process_args()

    opt.test_iterations = []
    opt.process_args()

    manifest_path = tmp_path / "proposal.json"
    manifest_path.write_text("{}")
    opt = make_options()
    opt.resume_train_camera_manifest = str(manifest_path)
    with pytest.raises(ValueError, match="requires --resume_checkpoint"):
        opt.process_args()

    opt.resume_checkpoint = str(checkpoint_path)
    with pytest.raises(ValueError, match="requires --skip_postprocess"):
        opt.process_args()

    opt.skip_postprocess = True
    opt.test_iterations = [10]
    with pytest.raises(ValueError, match="requires an empty --test_iterations"):
        opt.process_args()

    opt.test_iterations = []
    opt.resume_train_holdout_stride = 8
    with pytest.raises(ValueError, match="incompatible with --resume_train_holdout_stride"):
        opt.process_args()

    opt = make_options()
    opt.resume_checkpoint = str(checkpoint_path)
    opt.surface_coplanar_weight = 0.1
    with pytest.raises(ValueError, match="surface_coplanar_weight"):
        opt.process_args()


@pytest.mark.parametrize("sampling", ["random", "epoch"])
def test_resume_replays_camera_sampler(sampling: str) -> None:
    reference = _make_resume_trainer()
    reference.camera_sampling = sampling
    random.seed(reference.opt.seed)
    for _ in range(13):
        reference._next_train_camera()
    expected = [reference._next_train_camera() for _ in range(10)]

    resumed = _make_resume_trainer()
    resumed.camera_sampling = sampling
    resumed._replay_camera_sampler(13)
    actual = [resumed._next_train_camera() for _ in range(10)]
    assert actual == expected
