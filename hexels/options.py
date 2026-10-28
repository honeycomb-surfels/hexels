from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Optional

from tap import Tap


_COLOR_MODELS = frozenset({"flat", "linear", "quadratic", "quadratic_bounded"})
_QUADRATIC_COLOR_MODELS = frozenset({"quadratic", "quadratic_bounded"})


def color_model_uses_quadratic_coefficients(color_model: str) -> bool:
    return color_model in _QUADRATIC_COLOR_MODELS


class Hexels3DOptions(Tap):
    source_path: str = ""
    images: Optional[str] = None
    eval: bool = False
    llffhold: int = 8
    resolution: int = -1
    data_device: str = "cuda"

    model_path: Optional[str] = None
    seed: int = 42
    camera_sampling: str = "random"  # "random" | "epoch" | "weighted"
    camera_sampling_weights_path: Optional[str] = None
    train_camera_manifest: Optional[str] = None

    iterations: int = 10_000
    max_points: int = 0
    init_geometry_path: Optional[str] = None
    init_normal_mode: str = "identity"
    spatial_appearance_start: int = 0
    geometry_start_iteration: int = 0
    resume_checkpoint: Optional[str] = None
    resume_geometry_lr_scale: float = 1.0
    resume_appearance_lr_scale: float = 1.0
    lr_decay_milestones: list[int] = []
    lr_decay_factor: float = 0.5
    # "inherit" | "appearance_only" | "view_only"
    resume_parameter_policy: str = "inherit"
    # "inherit" | "frozen" | "bounded_isotropic_shrink"
    resume_geometry_policy: str = "inherit"
    resume_scale_lower_geometry_path: Optional[str] = None
    resume_scale_upper_geometry_path: Optional[str] = None
    resume_scale_anchor_weight: float = 0.0
    resume_normal_trust_path: Optional[str] = None
    resume_normal_trust_min_degrees: float = 5.0
    resume_normal_trust_max_degrees: float = 25.0
    resume_surface_anchor_path: Optional[str] = None
    resume_surface_anchor_weight: float = 0.0
    resume_mvs_surface_path: Optional[str] = None
    resume_mvs_surface_weight: float = 0.0
    resume_allow_lambda_dssim_change: bool = False
    resume_allow_photometric_loss_change: bool = False
    resume_allow_camera_sampling_change: bool = False
    resume_train_holdout_stride: int = 0
    resume_train_camera_manifest: Optional[str] = None
    edge_transfer_manifest: Optional[str] = None

    # Which attributes to optimize:
    # - "base": colors, color_grads, view_grads, opacities
    # - "geom": base + means3d/scales/rotations
    # - "full": geom + per-primitive softness
    train_mode: str = "full"

    enable_mcmc: bool = False
    mcmc_mode: str = "contrib"  # "contrib" | "opacity"
    mcmc_interval: int = 100
    mcmc_dead_threshold: float = 0.005
    mcmc_cap_max: int = 0
    mcmc_grow_rate: float = 0.005
    mcmc_ema_decay: float = 0.9
    mcmc_reorient: bool = False
    mcmc_noise_lr: float = 5e4
    mcmc_spawn_jitter: float = 0.0  # 3D position jitter for spawned/relocated primitives (scale-relative)
    mcmc_normal_noise_frac: float = 0.0  # fraction of normal-direction noise relative to tangent noise
    mcmc_from_iter: int = 500
    mcmc_until_iter: int = 15_000

    init_scale_frac: float = 0.002
    init_scale_mode: str = "uniform"  # "uniform" | "knn"
    init_knn_k: int = 3
    init_knn_scale_multiplier: float = 0.5
    init_knn_max_scale_frac: float = 0.05
    init_opacity: float = 0.1
    softness: float = 10.0
    softness_mode: str = "iso"  # "iso" | "aniso" (per-primitive 3-edge softness)
    scale_modifier: float = 1.0

    color_model: str = "linear"  # "flat" | "linear" | "quadratic" | "quadratic_bounded"
    use_view_grads: Optional[bool] = True
    view_model: str = "linear"  # "none" | "linear" | "sh3" | "lobe"
    sh_progressive_interval: int = 0  # 0 disables the curriculum

    repulsion_screen_weight: float = 0.0
    repulsion_screen_target: float = 1.0

    repulsion_knn_weight: float = 0.0
    repulsion_knn_k: int = 8
    repulsion_knn_interval: int = 500
    repulsion_knn_radius_frac: float = 1.0

    track_overdraw: bool = False

    conservative_bounds: bool = False  # Restore support coverage omitted by legacy tile bounds.

    prefilter_edge_on: bool = False

    surface_knn_k: int = 8
    surface_knn_interval: int = 500
    surface_coplanar_weight: float = 0.0
    surface_normal_smooth_weight: float = 0.0

    surface_normal_depth_weight: float = 0.0
    surface_normal_depth_start: int = 0
    surface_normal_depth_alpha_thresh: float = 0.01
    surface_normal_objective: str = "legacy"
    surface_normal_depth_mode: str = "expected"
    ray_distortion_weight: float = 0.0
    ray_distortion_start: int = 0
    ray_distortion_ramp_steps: int = 0

    enable_children: bool = False
    child_interval: int = 500
    child_parents_per_interval: int = 128
    child_offset_frac: float = 0.5
    child_scale_ratio: float = 0.5
    child_cap_max: int = 50_000
    child_max_layers_per_parent: int = 1
    child_sides: str = "both"  # "both" | "plus" | "minus"
    children_detach_geometry: Optional[bool] = True
    child_rotation_delta: bool = False
    child_rotation_delta_weight: float = 0.001

    lr_means3d: float = 0.00016
    lr_means3d_final: float = 0.0000016  # final LR after exponential decay (0 = no decay)
    lr_means3d_max_steps: int = 30_000
    lr_means3d_delay_steps: int = 0
    lr_means3d_delay_mult: float = 0.01
    spatial_lr_scale: Optional[bool] = True  # multiply position LR by cameras_extent (like 2DGS/3DGS)
    lr_scales: float = 0.005
    lr_rotations: float = 0.001
    lr_colors: float = 0.01
    lr_color_grads: float = 0.005
    lr_color_quads: Optional[float] = None
    lr_view_grads: float = 0.005
    lr_sh_coeffs: float = 0.0005
    lr_specular_ks: float = 0.005
    lr_specular_s: float = 0.001
    lr_opacity: float = 0.1
    lr_softness: float = 0.01

    lambda_dssim: float = 0.2
    use_fused_ssim: Optional[bool] = False
    resident_training_targets: Optional[bool] = False
    memory_poll_interval: int = 1
    photometric_loss: str = "l1"  # "l1" | "l2"
    white_background: bool = False
    random_background: bool = False

    log_every: int = 100
    render_every: int = 5000
    save_every: int = 10000
    test_iterations: list[int] = []
    test_train_views: int = 10
    wandb_test_images: int = 3

    enable_densify: bool = False
    densify_impl: str = "2dgs"  # "2dgs" | "hexels_safe"
    densification_interval: int = 100
    densify_from_iter: int = 500
    densify_until_iter: int = 15_000
    densify_grad_threshold: float = 0.0002
    densify_max_add: int = 0
    densify_cap_max: int = 0
    densify_action: str = "mixed"  # "mixed" | "clone" | "bisplit" | "edge_child"
    densify_min_denom: int = 0
    densify_clone_jitter: float = 0.0
    densify_split_jitter: float = 1.0
    densify_split_offset_fraction: float = 0.25
    densify_split_scale_ratio: float = 0.75
    densify_preserve_linear_field: bool = False
    densify_absgrad_rank: bool = False
    densify_min_projected_radius: float = 0.0
    densify_child_offset_fraction: float = 0.75
    densify_child_scale_ratio: float = 0.5
    densify_child_softness_ratio: float = 1.0  # multiplier after scale-matched child initialization
    densify_child_reorient: bool = False
    densify_child_reorient_min_planarity: float = 0.0
    densify_child_reorient_max_degrees: float = 180.0
    densify_child_init_opacity: float = 0.01
    densify_child_extent_cap_fraction: float = 0.0
    densify_child_parent_opacity_fraction: float = 0.0
    densify_direction_consistency: float = 0.25
    densify_direction_signal: str = "position"  # "position" | "residual_moment" | "attributed_residual" | "attributed_covariance"
    densify_moment_interval: int = 10
    densify_moment_min_radius: float = 0.25
    densify_residual_topk_views: int = 0
    densify_residual_persistent_filter: bool = False
    densify_attributed_weak_power: int = 0
    densify_covariance_children_per_parent: int = 2
    densify_covariance_min_views: int = 8
    densify_covariance_min_anisotropy: float = 0.15
    densify_covariance_min_axis_coherence: float = 0.50
    densify_covariance_min_side_balance: float = 0.30
    densify_covariance_min_delta: float = 0.10
    densify_repeat_parent: bool = False
    densify_joint_coplanar_steps: int = 0
    densify_shadow_steps: int = 0
    densify_shadow_child_probation: str = "unrestricted"  # "unrestricted" | "coplanar_spatial"
    densify_shadow_permanent_residual: bool = True
    densify_shadow_gate_stride: int = 0
    densify_shadow_parent_view_anchor: bool = False
    densify_shadow_enforce_birth_envelope: bool = False
    densify_shadow_salvage_groups: int = 0
    densify_shadow_salvage_order: str = "score"  # "score" | "safe_worst"
    densify_shadow_salvage_prefix_policy: str = "best_pooled"  # "best_pooled" | "first_qualifying"
    densify_shadow_salvage_selector_stride: int = 0
    densify_shadow_defer_inner_gate: bool = False
    densify_shadow_salvage_allow_partial_cohort: bool = False
    densify_shadow_min_pooled_sse_reduction: float = 0.005
    densify_shadow_min_mean_psnr_delta: float = 0.02
    densify_shadow_min_win_fraction: float = 0.6
    densify_shadow_min_worst_psnr_delta: float = -0.15
    densify_accept_parent_transfer_fraction: float = 0.0
    densify_accept_transfer_min_pooled_sse_reduction: float = -0.0001
    densify_accept_transfer_min_mean_psnr_delta: float = -0.001
    densify_accept_transfer_min_worst_psnr_delta: float = -0.01
    densify_opacity_split: bool = True
    densify_clone_init_opacity: float = 0.0  # 0 => inherit parent opacity
    densify_coverage_weight: float = 0.0
    densify_newborn_grace: int = 500
    densify_prune_interval: int = 1000
    densify_prune_full_sweep: bool = False
    densify_prune_contrib_threshold: float = 0.0
    densify_coordinated_prune: bool = False
    densify_prune_mean_threshold: float = 0.0
    densify_prune_max_view_threshold: float = -1.0
    densify_prune_max_fraction: float = 0.02
    densify_merge_interval: int = 0
    densify_merge_distance_scale_ratio: float = 0.02
    densify_merge_max_pairs: int = 256
    densify_merge_min_pooled_sse_reduction: float = 0.0
    densify_merge_min_mean_psnr_delta: float = -0.0005
    densify_merge_min_worst_psnr_delta: float = -0.005
    opacity_reset_interval: int = 3000  # 0 disables
    opacity_cull: float = 0.05
    percent_dense: float = 0.01
    densify_max_screen_size: int = 20

    skip_postprocess: bool = False
    quality_safe_retry: bool = False

    wandb: bool = False
    wandb_project: str = "hexels3d"
    wandb_entity: Optional[str] = None
    wandb_tags: list[str] = []

    debug: bool = False

    def process_args(self) -> None:
        if self.init_normal_mode not in {"identity", "pca_confident"}:
            raise ValueError("init_normal_mode must be identity or pca_confident")
        if self.spatial_appearance_start < 0:
            raise ValueError("spatial_appearance_start must be nonnegative")
        if not self.source_path:
            raise ValueError("--source_path is required")

        if self.train_mode not in {"base", "geom", "full"}:
            raise ValueError("--train_mode must be one of: base, geom, full")
        if self.photometric_loss not in {"l1", "l2"}:
            raise ValueError("--photometric_loss must be one of: l1, l2")
        if self.memory_poll_interval < 1:
            raise ValueError("--memory_poll_interval must be positive")
        if self.camera_sampling not in {"random", "epoch", "weighted"}:
            raise ValueError(
                "--camera_sampling must be one of: random, epoch, weighted"
            )
        if (
            self.camera_sampling == "weighted"
        ) != (self.camera_sampling_weights_path is not None):
            raise ValueError(
                "--camera_sampling weighted requires exactly one "
                "--camera_sampling_weights_path"
            )
        if self.camera_sampling_weights_path is not None:
            weights_path = Path(self.camera_sampling_weights_path)
            if not weights_path.is_file():
                raise ValueError(
                    "--camera_sampling_weights_path does not exist: "
                    f"{weights_path}"
                )
        if self.geometry_start_iteration < 0:
            raise ValueError("--geometry_start_iteration must be >= 0")
        if self.init_geometry_path is not None:
            geometry_path = Path(self.init_geometry_path)
            if not geometry_path.is_file():
                raise ValueError(f"--init_geometry_path does not exist: {geometry_path}")
            if self.max_points > 0:
                raise ValueError("--init_geometry_path is incompatible with --max_points")

        if (
            not math.isfinite(self.densify_child_extent_cap_fraction)
            or self.densify_child_extent_cap_fraction < 0.0
        ):
            raise ValueError("--densify_child_extent_cap_fraction must be finite and >= 0")
        if (
            not math.isfinite(self.densify_child_parent_opacity_fraction)
            or not (0.0 <= self.densify_child_parent_opacity_fraction <= 1.0)
        ):
            raise ValueError("--densify_child_parent_opacity_fraction must be finite and in [0,1]")
        bounded_child_birth = (
            self.densify_child_extent_cap_fraction > 0.0
            or self.densify_child_parent_opacity_fraction > 0.0
        )
        if bounded_child_birth and not self.enable_densify:
            raise ValueError("bounded child birth requires --enable_densify")
        if bounded_child_birth and (
            self.densify_impl != "hexels_safe" or self.densify_action != "edge_child"
        ):
            raise ValueError("bounded child birth requires hexels_safe edge_child")

        if self.densify_residual_topk_views < 0:
            raise ValueError("--densify_residual_topk_views must be >= 0")
        if self.densify_residual_topk_views > 0:
            if not (
                self.enable_densify
                and self.densify_impl == "hexels_safe"
                and self.densify_action == "edge_child"
                and self.densify_direction_signal == "residual_moment"
            ):
                raise ValueError(
                    "--densify_residual_topk_views requires residual-moment "
                    "hexels_safe edge-child densification"
                )
            if self.camera_sampling != "epoch":
                raise ValueError(
                    "--densify_residual_topk_views requires --camera_sampling epoch"
                )
            if self.densify_moment_interval != 1:
                raise ValueError(
                    "--densify_residual_topk_views requires --densify_moment_interval 1"
                )
            if self.densify_min_denom < self.densify_residual_topk_views:
                raise ValueError(
                    "--densify_min_denom must be at least --densify_residual_topk_views"
                )
        if self.densify_residual_persistent_filter and (
            self.densify_direction_signal != "residual_moment"
        ):
            raise ValueError(
                "--densify_residual_persistent_filter requires residual-moment direction"
            )
        if self.densify_attributed_weak_power not in {0, 2}:
            raise ValueError("--densify_attributed_weak_power must be 0 or 2")
        if self.densify_attributed_weak_power > 0 and self.densify_direction_signal not in {
            "attributed_residual",
            "attributed_covariance",
        }:
            raise ValueError(
                "--densify_attributed_weak_power requires an attributed direction"
            )
        if self.densify_direction_signal == "attributed_residual":
            if not (
                self.enable_densify
                and self.densify_impl == "hexels_safe"
                and self.densify_action == "edge_child"
            ):
                raise ValueError(
                    "attributed-residual direction requires hexels_safe edge-child densification"
                )
            if self.camera_sampling != "epoch" or self.densify_moment_interval != 1:
                raise ValueError(
                    "attributed-residual direction requires epoch sampling and moment interval 1"
                )
            if self.prefilter_edge_on:
                raise ValueError("attributed-residual direction requires native edge rendering")
            if self.enable_children:
                raise ValueError(
                    "attributed-residual direction does not support learned children"
                )
        if self.densify_covariance_children_per_parent not in {1, 2}:
            raise ValueError("--densify_covariance_children_per_parent must be 1 or 2")
        if self.densify_covariance_min_views < 1:
            raise ValueError("--densify_covariance_min_views must be positive")
        covariance_thresholds = (
            self.densify_covariance_min_anisotropy,
            self.densify_covariance_min_axis_coherence,
            self.densify_covariance_min_side_balance,
            self.densify_covariance_min_delta,
        )
        if not all(math.isfinite(value) for value in covariance_thresholds):
            raise ValueError("densify covariance thresholds must be finite")
        if not 0.0 <= self.densify_covariance_min_anisotropy <= 1.0:
            raise ValueError("--densify_covariance_min_anisotropy must be in [0,1]")
        if not 0.0 <= self.densify_covariance_min_axis_coherence <= 1.0:
            raise ValueError("--densify_covariance_min_axis_coherence must be in [0,1]")
        if not 0.0 <= self.densify_covariance_min_side_balance <= 1.0:
            raise ValueError("--densify_covariance_min_side_balance must be in [0,1]")
        if self.densify_covariance_min_delta < 0.0:
            raise ValueError("--densify_covariance_min_delta must be non-negative")
        if self.densify_direction_signal == "attributed_covariance":
            if not (
                self.enable_densify
                and self.densify_impl == "hexels_safe"
                and self.densify_action == "edge_child"
            ):
                raise ValueError(
                    "attributed-covariance direction requires hexels_safe edge-child densification"
                )
            if self.camera_sampling != "epoch" or self.densify_moment_interval != 1:
                raise ValueError(
                    "attributed-covariance direction requires epoch sampling and moment interval 1"
                )
            if self.prefilter_edge_on:
                raise ValueError("attributed-covariance direction requires native edge rendering")
            if self.enable_children:
                raise ValueError(
                    "attributed-covariance direction does not support learned children"
                )
            if self.densify_shadow_steps != 0:
                raise ValueError("attributed-covariance direction does not support shadow probation")
            if (
                self.densify_covariance_children_per_parent == 2
                and self.densify_max_add % 2 != 0
            ):
                raise ValueError(
                    "--densify_max_add must be divisible by 2 for covariance-pair births"
                )
            if (
                self.densify_covariance_children_per_parent == 2
                and self.densify_child_parent_opacity_fraction <= 0.0
            ):
                raise ValueError(
                    "covariance-pair births require a positive child parent opacity fraction"
                )

        if self.densify_joint_coplanar_steps < 0:
            raise ValueError("--densify_joint_coplanar_steps must be >= 0")
        if self.densify_joint_coplanar_steps > 0:
            if not self.enable_densify:
                raise ValueError(
                    "--densify_joint_coplanar_steps requires --enable_densify"
                )
            if self.densify_impl != "hexels_safe" or self.densify_action != "edge_child":
                raise ValueError(
                    "--densify_joint_coplanar_steps requires hexels_safe edge_child"
                )
            if self.densify_shadow_steps != 0:
                raise ValueError(
                    "--densify_joint_coplanar_steps does not support shadow probation"
                )
            if self.densify_child_parent_opacity_fraction <= 0.0:
                raise ValueError(
                    "--densify_joint_coplanar_steps requires a positive "
                    "child parent opacity fraction"
                )

        if self.densify_shadow_steps < 0:
            raise ValueError("--densify_shadow_steps must be >= 0")
        if self.densify_shadow_child_probation not in {
            "unrestricted",
            "coplanar_spatial",
        }:
            raise ValueError(
                "--densify_shadow_child_probation must be one of: "
                "unrestricted, coplanar_spatial"
            )
        if (
            self.densify_shadow_child_probation != "unrestricted"
            and self.densify_shadow_steps == 0
        ):
            raise ValueError(
                "--densify_shadow_child_probation requires --densify_shadow_steps"
            )
        if self.densify_shadow_gate_stride == 1 or self.densify_shadow_gate_stride < 0:
            raise ValueError("--densify_shadow_gate_stride must be 0 or >= 2")
        if self.densify_shadow_gate_stride > 0 and self.densify_shadow_steps == 0:
            raise ValueError("--densify_shadow_gate_stride requires --densify_shadow_steps")
        if self.densify_shadow_parent_view_anchor and self.densify_shadow_steps == 0:
            raise ValueError("--densify_shadow_parent_view_anchor requires --densify_shadow_steps")
        if self.densify_shadow_enforce_birth_envelope:
            if not self.enable_densify or self.densify_shadow_steps == 0:
                raise ValueError(
                    "--densify_shadow_enforce_birth_envelope requires shadow densification"
                )
            if self.densify_impl != "hexels_safe" or self.densify_action != "edge_child":
                raise ValueError(
                    "--densify_shadow_enforce_birth_envelope requires hexels_safe edge_child"
                )
            if self.densify_child_extent_cap_fraction <= 0.0:
                raise ValueError(
                    "--densify_shadow_enforce_birth_envelope requires a positive child extent cap"
                )
            if self.densify_child_parent_opacity_fraction <= 0.0:
                raise ValueError(
                    "--densify_shadow_enforce_birth_envelope requires a positive parent opacity fraction"
                )
        if self.densify_shadow_salvage_groups == 1 or self.densify_shadow_salvage_groups < 0:
            raise ValueError("--densify_shadow_salvage_groups must be 0 or >= 2")
        if (
            self.densify_shadow_salvage_selector_stride == 1
            or self.densify_shadow_salvage_selector_stride < 0
        ):
            raise ValueError(
                "--densify_shadow_salvage_selector_stride must be 0 or >= 2"
            )
        if self.densify_shadow_salvage_order not in {"score", "safe_worst"}:
            raise ValueError(
                "--densify_shadow_salvage_order must be one of: score, safe_worst"
            )
        if (
            self.densify_shadow_salvage_order != "score"
            and self.densify_shadow_salvage_groups == 0
        ):
            raise ValueError(
                "--densify_shadow_salvage_order requires --densify_shadow_salvage_groups"
            )
        if self.densify_shadow_salvage_prefix_policy not in {
            "best_pooled",
            "first_qualifying",
        }:
            raise ValueError(
                "--densify_shadow_salvage_prefix_policy must be one of: "
                "best_pooled, first_qualifying"
            )
        if (
            self.densify_shadow_salvage_prefix_policy != "best_pooled"
            and self.densify_shadow_salvage_groups == 0
        ):
            raise ValueError(
                "--densify_shadow_salvage_prefix_policy requires "
                "--densify_shadow_salvage_groups"
            )
        if (
            self.densify_shadow_salvage_selector_stride > 0
            and self.densify_shadow_salvage_groups == 0
        ):
            raise ValueError(
                "--densify_shadow_salvage_selector_stride requires "
                "--densify_shadow_salvage_groups"
            )
        if (
            self.densify_shadow_salvage_selector_stride > 0
            and self.densify_shadow_salvage_prefix_policy != "first_qualifying"
        ):
            raise ValueError(
                "--densify_shadow_salvage_selector_stride requires "
                "--densify_shadow_salvage_prefix_policy first_qualifying"
            )
        if (
            self.densify_shadow_salvage_selector_stride > 0
            and self.densify_shadow_enforce_birth_envelope
        ):
            raise ValueError(
                "--densify_shadow_salvage_selector_stride is incompatible with "
                "--densify_shadow_enforce_birth_envelope"
            )
        if (
            self.densify_shadow_defer_inner_gate
            and self.densify_shadow_salvage_selector_stride == 0
        ):
            raise ValueError(
                "--densify_shadow_defer_inner_gate requires "
                "--densify_shadow_salvage_selector_stride"
            )
        if (
            self.densify_shadow_salvage_allow_partial_cohort
            and self.densify_shadow_salvage_groups == 0
        ):
            raise ValueError(
                "--densify_shadow_salvage_allow_partial_cohort requires "
                "--densify_shadow_salvage_groups"
            )
        if self.densify_shadow_salvage_groups > 0 and self.densify_shadow_steps == 0:
            raise ValueError("--densify_shadow_salvage_groups requires --densify_shadow_steps")
        if self.densify_shadow_salvage_groups > 0 and self.densify_shadow_gate_stride == 0:
            raise ValueError("--densify_shadow_salvage_groups requires a shadow gate")
        if self.densify_shadow_salvage_groups > 0 and self.densify_max_add <= 0:
            raise ValueError("--densify_shadow_salvage_groups requires --densify_max_add > 0")
        if (
            self.densify_shadow_salvage_groups > 0
            and not self.densify_shadow_salvage_allow_partial_cohort
            and self.densify_max_add % self.densify_shadow_salvage_groups != 0
        ):
            raise ValueError(
                "--densify_max_add must be divisible by --densify_shadow_salvage_groups"
            )
        if (
            self.densify_shadow_salvage_allow_partial_cohort
            and self.densify_max_add < self.densify_shadow_salvage_groups
        ):
            raise ValueError(
                "--densify_max_add must be at least --densify_shadow_salvage_groups "
                "when partial cohorts are enabled"
            )
        shadow_thresholds = (
            self.densify_shadow_min_pooled_sse_reduction,
            self.densify_shadow_min_mean_psnr_delta,
            self.densify_shadow_min_win_fraction,
            self.densify_shadow_min_worst_psnr_delta,
        )
        if not all(math.isfinite(value) for value in shadow_thresholds):
            raise ValueError("densify shadow thresholds must be finite")
        if not (0.0 <= self.densify_shadow_min_win_fraction <= 1.0):
            raise ValueError("--densify_shadow_min_win_fraction must be in [0,1]")
        if not 0.0 <= self.densify_accept_parent_transfer_fraction <= 1.0:
            raise ValueError(
                "--densify_accept_parent_transfer_fraction must be in [0,1]"
            )
        if (
            self.densify_accept_parent_transfer_fraction > 0.0
            and self.densify_shadow_steps == 0
        ):
            raise ValueError(
                "--densify_accept_parent_transfer_fraction requires shadow densification"
            )
        transfer_thresholds = (
            self.densify_accept_transfer_min_pooled_sse_reduction,
            self.densify_accept_transfer_min_mean_psnr_delta,
            self.densify_accept_transfer_min_worst_psnr_delta,
        )
        if not all(math.isfinite(value) for value in transfer_thresholds):
            raise ValueError("densify accepted-transfer thresholds must be finite")
        if self.densify_shadow_steps > 0 and not self.enable_densify:
            raise ValueError("--densify_shadow_steps requires --enable_densify")

        if self.resume_geometry_lr_scale <= 0.0:
            raise ValueError("--resume_geometry_lr_scale must be > 0")
        if self.resume_appearance_lr_scale <= 0.0:
            raise ValueError("--resume_appearance_lr_scale must be > 0")
        if self.lr_decay_milestones != sorted(set(self.lr_decay_milestones)):
            raise ValueError("--lr_decay_milestones must be strictly increasing and unique")
        if any(
            milestone <= 0 or milestone > self.iterations
            for milestone in self.lr_decay_milestones
        ):
            raise ValueError("--lr_decay_milestones must be within the training iteration range")
        if not math.isfinite(self.lr_decay_factor) or not 0.0 < self.lr_decay_factor <= 1.0:
            raise ValueError("--lr_decay_factor must be finite and in (0,1]")
        if self.resume_parameter_policy not in {
            "inherit",
            "appearance_only",
            "view_only",
        }:
            raise ValueError(
                "--resume_parameter_policy must be one of: "
                "inherit, appearance_only, view_only"
            )
        if self.resume_parameter_policy != "inherit" and self.resume_checkpoint is None:
            raise ValueError("--resume_parameter_policy requires --resume_checkpoint")
        if self.resume_parameter_policy == "view_only" and (
            self.view_model != "sh3" or not self.use_view_grads
        ):
            raise ValueError("--resume_parameter_policy view_only requires V3 SH")
        if self.resume_geometry_policy not in {
            "inherit",
            "frozen",
            "bounded_isotropic_shrink",
        }:
            raise ValueError(
                "--resume_geometry_policy must be one of: "
                "inherit, frozen, bounded_isotropic_shrink"
            )
        if self.resume_geometry_policy != "inherit" and self.resume_checkpoint is None:
            raise ValueError("--resume_geometry_policy requires --resume_checkpoint")
        if (
            not math.isfinite(self.resume_scale_anchor_weight)
            or self.resume_scale_anchor_weight < 0.0
        ):
            raise ValueError("--resume_scale_anchor_weight must be finite and >= 0")
        if self.resume_geometry_policy == "bounded_isotropic_shrink":
            if (
                self.resume_scale_lower_geometry_path is None
                or self.resume_scale_upper_geometry_path is None
            ):
                raise ValueError(
                    "bounded_isotropic_shrink requires lower and upper geometry paths"
                )
            for name, value in (
                (
                    "--resume_scale_lower_geometry_path",
                    self.resume_scale_lower_geometry_path,
                ),
                (
                    "--resume_scale_upper_geometry_path",
                    self.resume_scale_upper_geometry_path,
                ),
            ):
                if not Path(value).is_file():
                    raise ValueError(f"{name} does not exist: {value}")
        elif (
            self.resume_scale_lower_geometry_path is not None
            or self.resume_scale_upper_geometry_path is not None
        ):
            raise ValueError(
                "resume scale geometry paths require bounded_isotropic_shrink"
            )
        elif self.resume_scale_anchor_weight != 0.0:
            raise ValueError("--resume_scale_anchor_weight requires bounded_isotropic_shrink")
        normal_trust_values = (
            self.resume_normal_trust_min_degrees,
            self.resume_normal_trust_max_degrees,
        )
        if (
            not all(math.isfinite(value) for value in normal_trust_values)
            or not (
                0.0
                < self.resume_normal_trust_min_degrees
                <= self.resume_normal_trust_max_degrees
                < 90.0
            )
        ):
            raise ValueError(
                "resume normal trust degrees must satisfy 0 < min <= max < 90"
            )
        if self.resume_normal_trust_path is not None:
            if self.resume_checkpoint is None:
                raise ValueError(
                    "--resume_normal_trust_path requires --resume_checkpoint"
                )
            if self.resume_geometry_policy != "inherit":
                raise ValueError(
                    "--resume_normal_trust_path requires --resume_geometry_policy inherit"
                )
            if self.train_mode not in {"geom", "full"}:
                raise ValueError(
                    "--resume_normal_trust_path requires --train_mode {geom,full}"
                )
            if not Path(self.resume_normal_trust_path).is_file():
                raise ValueError("--resume_normal_trust_path does not exist")
        if (
            not math.isfinite(self.resume_surface_anchor_weight)
            or self.resume_surface_anchor_weight < 0.0
        ):
            raise ValueError(
                "--resume_surface_anchor_weight must be finite and >= 0"
            )
        if self.resume_surface_anchor_path is not None:
            if self.resume_checkpoint is None:
                raise ValueError(
                    "--resume_surface_anchor_path requires --resume_checkpoint"
                )
            if self.resume_geometry_policy != "inherit":
                raise ValueError(
                    "--resume_surface_anchor_path requires --resume_geometry_policy inherit"
                )
            if self.train_mode not in {"geom", "full"}:
                raise ValueError(
                    "--resume_surface_anchor_path requires --train_mode {geom,full}"
                )
            if not Path(self.resume_surface_anchor_path).is_file():
                raise ValueError("--resume_surface_anchor_path does not exist")
        elif self.resume_surface_anchor_weight != 0.0:
            raise ValueError(
                "--resume_surface_anchor_weight requires --resume_surface_anchor_path"
            )
        if (
            not math.isfinite(self.resume_mvs_surface_weight)
            or self.resume_mvs_surface_weight < 0.0
        ):
            raise ValueError(
                "--resume_mvs_surface_weight must be finite and >= 0"
            )
        if self.resume_mvs_surface_path is not None:
            if self.resume_checkpoint is None:
                raise ValueError(
                    "--resume_mvs_surface_path requires --resume_checkpoint"
                )
            if self.resume_geometry_policy != "inherit":
                raise ValueError(
                    "--resume_mvs_surface_path requires --resume_geometry_policy inherit"
                )
            if self.train_mode not in {"geom", "full"}:
                raise ValueError(
                    "--resume_mvs_surface_path requires --train_mode {geom,full}"
                )
            if not Path(self.resume_mvs_surface_path).is_file():
                raise ValueError("--resume_mvs_surface_path does not exist")
        elif self.resume_mvs_surface_weight != 0.0:
            raise ValueError(
                "--resume_mvs_surface_weight requires --resume_mvs_surface_path"
            )
        if self.resume_train_holdout_stride < 0 or self.resume_train_holdout_stride == 1:
            raise ValueError("--resume_train_holdout_stride must be 0 or at least 2")
        if self.resume_allow_lambda_dssim_change and self.resume_checkpoint is None:
            raise ValueError("--resume_allow_lambda_dssim_change requires --resume_checkpoint")
        if (
            self.resume_allow_photometric_loss_change
            and self.resume_checkpoint is None
        ):
            raise ValueError(
                "--resume_allow_photometric_loss_change requires --resume_checkpoint"
            )
        if self.resume_allow_camera_sampling_change and self.resume_checkpoint is None:
            raise ValueError("--resume_allow_camera_sampling_change requires --resume_checkpoint")
        if self.resume_train_holdout_stride > 0 and self.resume_checkpoint is None:
            raise ValueError("--resume_train_holdout_stride requires --resume_checkpoint")
        if self.resume_train_holdout_stride > 0 and not self.skip_postprocess:
            raise ValueError(
                "--resume_train_holdout_stride requires --skip_postprocess to keep the official test split blind"
            )
        if self.resume_train_holdout_stride > 0 and self.test_iterations:
            raise ValueError(
                "--resume_train_holdout_stride requires an empty --test_iterations to keep the official test split blind"
            )
        if self.resume_train_camera_manifest is not None:
            if self.resume_checkpoint is None:
                raise ValueError("--resume_train_camera_manifest requires --resume_checkpoint")
            if self.resume_train_holdout_stride > 0:
                raise ValueError(
                    "--resume_train_camera_manifest is incompatible with --resume_train_holdout_stride"
                )
            manifest = Path(self.resume_train_camera_manifest)
            if not manifest.is_file():
                raise ValueError(f"--resume_train_camera_manifest does not exist: {manifest}")
            if not self.skip_postprocess:
                raise ValueError(
                    "--resume_train_camera_manifest requires --skip_postprocess to keep excluded views blind"
                )
            if self.test_iterations:
                raise ValueError(
                    "--resume_train_camera_manifest requires an empty --test_iterations to keep excluded views blind"
                )
        if self.train_camera_manifest is not None:
            if self.resume_checkpoint is not None:
                raise ValueError("--train_camera_manifest is only supported for fresh training")
            manifest = Path(self.train_camera_manifest)
            if not manifest.is_file():
                raise ValueError(f"--train_camera_manifest does not exist: {manifest}")
        if self.resume_checkpoint is not None:
            checkpoint = Path(self.resume_checkpoint)
            if not checkpoint.is_file():
                raise ValueError(f"--resume_checkpoint does not exist: {checkpoint}")
            constrained_resume = (
                self.resume_geometry_policy != "inherit"
                or self.resume_parameter_policy != "inherit"
            )
            if constrained_resume and (
                self.enable_mcmc
                or self.enable_densify
                or self.enable_children
                or self.edge_transfer_manifest is not None
            ):
                raise ValueError(
                    "constrained resume is incompatible with topology-changing methods"
                )
            incompatible = [
                name
                for name, enabled in (
                    ("--enable_mcmc", self.enable_mcmc),
                    (
                        "--enable_densify",
                        self.enable_densify
                        and not (
                            self.densify_impl == "hexels_safe"
                            and self.densify_action == "edge_child"
                        ),
                    ),
                    ("--enable_children", self.enable_children),
                    ("--random_background", self.random_background),
                    ("--repulsion_knn_weight", self.repulsion_knn_weight > 0.0),
                    ("--surface_coplanar_weight", self.surface_coplanar_weight > 0.0),
                    ("--surface_normal_smooth_weight", self.surface_normal_smooth_weight > 0.0),
                )
                if enabled
            ]
            if incompatible:
                raise ValueError(
                    "--resume_checkpoint supports fixed-population deterministic training only; "
                    f"disable {', '.join(incompatible)}"
                )
        if self.edge_transfer_manifest is not None:
            if self.resume_checkpoint is None:
                raise ValueError("--edge_transfer_manifest requires --resume_checkpoint")
            if self.enable_densify:
                raise ValueError("--edge_transfer_manifest is incompatible with --enable_densify")
            if not Path(self.edge_transfer_manifest).is_file():
                raise ValueError("--edge_transfer_manifest does not exist")
            if self.resume_train_holdout_stride == 0:
                raise ValueError("--edge_transfer_manifest requires a resume training holdout")
            if self.train_mode != "full":
                raise ValueError("--edge_transfer_manifest requires --train_mode full")
            if self.view_model != "linear" or not self.use_view_grads or self.prefilter_edge_on:
                raise ValueError(
                    "--edge_transfer_manifest requires the linear-view fast path"
                )

        if self.init_scale_mode not in {"uniform", "knn"}:
            raise ValueError("--init_scale_mode must be one of: uniform, knn")
        if self.init_knn_k < 1:
            raise ValueError("--init_knn_k must be >= 1")
        if self.init_knn_scale_multiplier <= 0.0:
            raise ValueError("--init_knn_scale_multiplier must be > 0")
        if self.init_knn_max_scale_frac < 0.0:
            raise ValueError("--init_knn_max_scale_frac must be >= 0")

        if self.enable_mcmc:
            if self.mcmc_mode not in {"contrib", "opacity"}:
                raise ValueError("--mcmc_mode must be one of: contrib, opacity")
            if self.mcmc_interval <= 0:
                raise ValueError("--mcmc_interval must be > 0")
            if not (0.0 < self.mcmc_ema_decay < 1.0):
                raise ValueError("--mcmc_ema_decay must be in (0,1)")
            if self.mcmc_dead_threshold <= 0:
                raise ValueError("--mcmc_dead_threshold must be > 0")
            if self.mcmc_cap_max < 0:
                raise ValueError("--mcmc_cap_max must be >= 0")
            if self.mcmc_grow_rate < 0:
                raise ValueError("--mcmc_grow_rate must be >= 0")
            if self.mcmc_noise_lr < 0:
                raise ValueError("--mcmc_noise_lr must be >= 0")
            if self.mcmc_from_iter < 0:
                raise ValueError("--mcmc_from_iter must be >= 0")
            if self.mcmc_until_iter < -1:
                raise ValueError("--mcmc_until_iter must be >= -1")
            if self.mcmc_until_iter != -1 and self.mcmc_until_iter < self.mcmc_from_iter:
                raise ValueError("--mcmc_until_iter must be >= --mcmc_from_iter (or -1)")

        if self.softness_mode not in {"iso", "aniso"}:
            raise ValueError("--softness_mode must be one of: iso, aniso")

        if self.color_model not in _COLOR_MODELS:
            raise ValueError(
                "--color_model must be one of: flat, linear, quadratic, quadratic_bounded"
            )

        if self.view_model not in {"none", "linear", "sh3", "lobe"}:
            raise ValueError("--view_model must be one of: none, linear, sh3, lobe")
        if self.sh_progressive_interval < 0:
            raise ValueError("--sh_progressive_interval must be >= 0")

        if self.repulsion_screen_weight < 0:
            raise ValueError("--repulsion_screen_weight must be >= 0")
        if self.repulsion_screen_target <= 0:
            raise ValueError("--repulsion_screen_target must be > 0")

        if self.repulsion_knn_weight < 0:
            raise ValueError("--repulsion_knn_weight must be >= 0")
        if self.repulsion_knn_weight > 0 and self.train_mode == "base":
            raise ValueError("--repulsion_knn_weight requires --train_mode {geom,full} (needs geometry gradients)")
        if self.repulsion_knn_k <= 0:
            raise ValueError("--repulsion_knn_k must be > 0")
        if self.repulsion_knn_interval <= 0:
            raise ValueError("--repulsion_knn_interval must be > 0")
        if self.repulsion_knn_radius_frac <= 0:
            raise ValueError("--repulsion_knn_radius_frac must be > 0")

        if self.surface_knn_k <= 0:
            raise ValueError("--surface_knn_k must be > 0")
        if self.surface_knn_interval <= 0:
            raise ValueError("--surface_knn_interval must be > 0")
        if self.surface_coplanar_weight < 0:
            raise ValueError("--surface_coplanar_weight must be >= 0")
        if self.surface_normal_smooth_weight < 0:
            raise ValueError("--surface_normal_smooth_weight must be >= 0")
        if self.surface_normal_depth_weight < 0:
            raise ValueError("--surface_normal_depth_weight must be >= 0")
        if self.surface_normal_depth_start < 0:
            raise ValueError("--surface_normal_depth_start must be >= 0")
        if self.surface_normal_depth_alpha_thresh < 0:
            raise ValueError("--surface_normal_depth_alpha_thresh must be >= 0")
        if self.surface_normal_objective not in {"legacy", "2dgs"}:
            raise ValueError("--surface_normal_objective must be legacy or 2dgs")
        if self.surface_normal_depth_mode not in {"expected", "median"}:
            raise ValueError("--surface_normal_depth_mode must be expected or median")
        if (
            not math.isfinite(self.ray_distortion_weight)
            or self.ray_distortion_weight < 0
        ):
            raise ValueError("--ray_distortion_weight must be finite and >= 0")
        if self.ray_distortion_start < 0:
            raise ValueError("--ray_distortion_start must be >= 0")
        if self.ray_distortion_ramp_steps < 0:
            raise ValueError("--ray_distortion_ramp_steps must be >= 0")

        if (
            self.surface_coplanar_weight > 0
            or self.surface_normal_smooth_weight > 0
            or self.surface_normal_depth_weight > 0
            or self.ray_distortion_weight > 0
        ) and self.train_mode == "base":
            raise ValueError(
                "surface and distortion losses require --train_mode {geom,full}"
            )

        if self.mcmc_reorient and not self.enable_mcmc:
            raise ValueError("--mcmc_reorient requires --enable_mcmc")

        if self.enable_children:
            if self.train_mode == "base":
                raise ValueError("--enable_children requires --train_mode {geom,full} (needs geometry gradients)")
            if self.child_interval <= 0:
                raise ValueError("--child_interval must be > 0")
            if self.child_parents_per_interval <= 0:
                raise ValueError("--child_parents_per_interval must be > 0")
            if self.child_offset_frac < 0:
                raise ValueError("--child_offset_frac must be >= 0")
            if not (0.0 < self.child_scale_ratio <= 1.0):
                raise ValueError("--child_scale_ratio must be in (0,1]")
            if self.child_cap_max < 0:
                raise ValueError("--child_cap_max must be >= 0")
            if self.child_max_layers_per_parent <= 0:
                raise ValueError("--child_max_layers_per_parent must be > 0")
            if self.child_sides not in {"both", "plus", "minus"}:
                raise ValueError("--child_sides must be one of: both, plus, minus")
            if self.child_rotation_delta_weight < 0:
                raise ValueError("--child_rotation_delta_weight must be >= 0")
        else:
            if self.child_rotation_delta:
                raise ValueError("--child_rotation_delta requires --enable_children")

        if self.densify_child_reorient and (
            not self.enable_densify
            or self.densify_impl != "hexels_safe"
            or self.densify_action != "edge_child"
        ):
            raise ValueError(
                "--densify_child_reorient requires hexels_safe edge-child densification"
            )

        if self.enable_densify:
            if self.enable_mcmc:
                raise ValueError("--enable_densify is not supported with --enable_mcmc (for clean ablations)")
            if self.enable_children:
                raise ValueError("--enable_densify is not supported with --enable_children (children would lose anchors)")
            if self.train_mode == "base":
                raise ValueError("--enable_densify requires --train_mode {geom,full} (needs geometry gradients)")
            if self.densify_impl not in {"2dgs", "hexels_safe"}:
                raise ValueError("--densify_impl must be one of: 2dgs, hexels_safe")
            if self.densification_interval <= 0:
                raise ValueError("--densification_interval must be > 0")
            if self.densify_from_iter < 0:
                raise ValueError("--densify_from_iter must be >= 0")
            if self.densify_until_iter < 0:
                raise ValueError("--densify_until_iter must be >= 0")
            if self.densify_until_iter < self.densify_from_iter:
                raise ValueError("--densify_until_iter must be >= --densify_from_iter")
            residual_only = (
                self.densify_impl == "hexels_safe"
                and self.densify_action == "edge_child"
                and self.densify_direction_signal
                in {"residual_moment", "attributed_residual", "attributed_covariance"}
            )
            if self.densify_grad_threshold < 0 or (
                self.densify_grad_threshold == 0 and not residual_only
            ):
                raise ValueError(
                    "--densify_grad_threshold must be > 0, or 0 for residual-driven edge children"
                )
            if self.densify_max_add < 0:
                raise ValueError("--densify_max_add must be >= 0 (0 disables cap)")
            if self.densify_cap_max < 0:
                raise ValueError("--densify_cap_max must be >= 0 (0 disables cap)")
            if self.densify_absgrad_rank and (
                self.densify_impl != "hexels_safe" or self.densify_action != "mixed"
                or self.densify_direction_signal != "position" or self.view_model != "none" or self.color_model != "linear"
            ):
                raise ValueError("Absolute-gradient ranking requires safe mixed V1 position-driven growth")
            if not math.isfinite(self.densify_prune_max_view_threshold) or (
                self.densify_prune_max_view_threshold != -1 and self.densify_prune_max_view_threshold < 0
            ):
                raise ValueError("Prune max-view threshold must be -1 (disabled) or nonnegative")
            if self.densify_prune_max_view_threshold >= 0 and not self.densify_coordinated_prune:
                raise ValueError("Max-view protection requires coordinated pruning")
            if self.densify_coordinated_prune:
                if self.densify_impl != "hexels_safe" or self.densify_action not in {"mixed", "clone", "bisplit"}:
                    raise ValueError("Coordinated pruning requires safe mixed/clone/bisplit growth")
                if self.densify_prune_interval <= 0 or self.enable_children:
                    raise ValueError("Coordinated pruning requires a positive interval and no child layers")
                if not math.isfinite(self.densify_prune_mean_threshold) or self.densify_prune_mean_threshold < 0:
                    raise ValueError("Prune mean threshold must be finite and nonnegative")
                if not 0 < self.densify_prune_max_fraction < 1:
                    raise ValueError("Prune max fraction must be between zero and one")
            if self.densify_action not in {"mixed", "clone", "bisplit", "edge_child"}:
                raise ValueError("--densify_action must be one of: mixed, clone, bisplit, edge_child")
            if self.densify_impl != "hexels_safe" and (
                self.densify_cap_max > 0 or self.densify_action != "mixed"
            ):
                raise ValueError("--densify_cap_max and Hexel-specific actions require --densify_impl hexels_safe")
            if self.densify_min_denom < 0:
                raise ValueError("--densify_min_denom must be >= 0")
            if not math.isfinite(self.densify_min_projected_radius) or self.densify_min_projected_radius < 0:
                raise ValueError("--densify_min_projected_radius must be finite and nonnegative")
            if self.densify_clone_jitter < 0:
                raise ValueError("--densify_clone_jitter must be >= 0")
            if self.densify_split_jitter < 0:
                raise ValueError("--densify_split_jitter must be >= 0")
            if self.densify_preserve_linear_field and (self.color_model != "linear" or self.view_model != "none"):
                raise ValueError("--densify_preserve_linear_field currently requires V1 linear color without view terms")
            if self.densify_split_offset_fraction < 0:
                raise ValueError("--densify_split_offset_fraction must be >= 0")
            if not (0.0 < self.densify_split_scale_ratio < 1.0):
                raise ValueError("--densify_split_scale_ratio must be in (0,1)")
            if self.densify_child_offset_fraction < 0:
                raise ValueError("--densify_child_offset_fraction must be >= 0")
            if not (0.0 < self.densify_child_scale_ratio < 1.0):
                raise ValueError("--densify_child_scale_ratio must be in (0,1)")
            if self.densify_child_softness_ratio <= 0.0:
                raise ValueError("--densify_child_softness_ratio must be positive")
            if not (
                0.0 <= self.densify_child_reorient_min_planarity <= 1.0
            ):
                raise ValueError(
                    "--densify_child_reorient_min_planarity must be in [0,1]"
                )
            if not (
                0.0 < self.densify_child_reorient_max_degrees <= 180.0
            ):
                raise ValueError(
                    "--densify_child_reorient_max_degrees must be in (0,180]"
                )
            if not (0.0 < self.densify_child_init_opacity < 1.0):
                raise ValueError("--densify_child_init_opacity must be in (0,1)")
            if not (0.0 <= self.densify_direction_consistency <= 1.0):
                raise ValueError("--densify_direction_consistency must be in [0,1]")
            if self.densify_direction_signal not in {
                "position",
                "residual_moment",
                "attributed_residual",
                "attributed_covariance",
            }:
                raise ValueError(
                    "--densify_direction_signal must be one of: position, "
                    "residual_moment, attributed_residual, attributed_covariance"
                )
            if self.densify_moment_interval <= 0:
                raise ValueError("--densify_moment_interval must be > 0")
            if self.densify_moment_min_radius < 0.0:
                raise ValueError("--densify_moment_min_radius must be >= 0")
            if self.densify_coverage_weight < 0:
                raise ValueError("--densify_coverage_weight must be >= 0")
            if self.densify_newborn_grace < 0:
                raise ValueError("--densify_newborn_grace must be >= 0")
            if self.densify_prune_interval < 0:
                raise ValueError("--densify_prune_interval must be >= 0")
            if self.densify_prune_contrib_threshold < 0:
                raise ValueError("--densify_prune_contrib_threshold must be >= 0")
            if self.densify_merge_interval < 0:
                raise ValueError("--densify_merge_interval must be >= 0")
            if self.densify_merge_distance_scale_ratio <= 0.0:
                raise ValueError(
                    "--densify_merge_distance_scale_ratio must be positive"
                )
            if self.densify_merge_max_pairs < 1:
                raise ValueError("--densify_merge_max_pairs must be positive")
            merge_thresholds = (
                self.densify_merge_min_pooled_sse_reduction,
                self.densify_merge_min_mean_psnr_delta,
                self.densify_merge_min_worst_psnr_delta,
            )
            if not all(math.isfinite(value) for value in merge_thresholds):
                raise ValueError("densify merge thresholds must be finite")
            if self.densify_merge_interval > 0 and self.densify_action != "edge_child":
                raise ValueError(
                    "--densify_merge_interval requires --densify_action edge_child"
                )
            if self.densify_action in {"bisplit", "edge_child"}:
                if self.densify_prune_contrib_threshold != 0.0:
                    raise ValueError("directional densification supports exact-zero contribution pruning only")
                if color_model_uses_quadratic_coefficients(self.color_model):
                    raise ValueError("directional densification does not support quadratic color")
                if self.opacity_reset_interval != 0:
                    raise ValueError("directional densification requires --opacity_reset_interval 0")
                if self.densify_newborn_grace == 0:
                    raise ValueError("directional densification requires --densify_newborn_grace > 0")
            if self.densify_shadow_steps > 0:
                if self.densify_impl != "hexels_safe" or self.densify_action != "edge_child":
                    raise ValueError("shadow densification requires hexels_safe edge_child")
                if self.densify_prune_interval != 0:
                    raise ValueError("shadow densification requires --densify_prune_interval 0")
                if self.densify_repeat_parent:
                    raise ValueError("shadow densification requires one proposal attempt per parent")
                if self.densify_shadow_gate_stride > 0 and not self.densify_shadow_permanent_residual:
                    raise ValueError("shadow gate cameras require permanent residual training")
                if (
                    self.densify_shadow_salvage_groups > 0
                    and not self.densify_shadow_permanent_residual
                ):
                    raise ValueError("shadow group salvage requires permanent residual training")
                if self.densify_shadow_parent_view_anchor and not self.use_view_grads:
                    raise ValueError("shadow parent-view anchoring requires view gradients")
                if self.densify_shadow_parent_view_anchor and (
                    self.view_model != "linear" or self.prefilter_edge_on
                ):
                    raise ValueError(
                        "shadow parent-view anchoring requires the linear view fast path"
                    )
                incompatible_shadow = [
                    name
                    for name, enabled in (
                        ("--random_background", self.random_background),
                        ("--densify_coverage_weight", self.densify_coverage_weight > 0.0),
                        ("--repulsion_screen_weight", self.repulsion_screen_weight > 0.0),
                        ("--repulsion_knn_weight", self.repulsion_knn_weight > 0.0),
                        ("--surface_coplanar_weight", self.surface_coplanar_weight > 0.0),
                        ("--surface_normal_smooth_weight", self.surface_normal_smooth_weight > 0.0),
                        ("--surface_normal_depth_weight", self.surface_normal_depth_weight > 0.0),
                        ("--ray_distortion_weight", self.ray_distortion_weight > 0.0),
                    )
                    if enabled
                ]
                if incompatible_shadow:
                    raise ValueError(
                        "shadow densification requires plain photometric optimization; disable "
                        + ", ".join(incompatible_shadow)
                    )
            if (
                self.densify_action == "edge_child"
                and self.densify_prune_interval > 0
                and (
                    self.repulsion_knn_weight > 0.0
                    or self.surface_coplanar_weight > 0.0
                    or self.surface_normal_smooth_weight > 0.0
                )
            ):
                raise ValueError(
                    "edge-child pruning is incompatible with KNN repulsion/surface losses"
                )
            if self.densify_direction_signal == "residual_moment":
                if self.densify_action != "edge_child":
                    raise ValueError("residual-moment direction requires --densify_action edge_child")
                if self.color_model != "linear":
                    raise ValueError("residual-moment direction requires --color_model linear")
            if not (0.0 <= self.densify_clone_init_opacity < 1.0):
                raise ValueError("--densify_clone_init_opacity must be in [0,1) (0 inherits parent)")
            if self.densify_impl == "hexels_safe" and self.densify_action in {"mixed", "clone"}:
                if self.densify_clone_init_opacity == 0.0:
                    raise ValueError("hexels_safe clones require faint positive initialization")
                if self.densify_clone_jitter == 0.0:
                    raise ValueError("hexels_safe clones require nonzero tangent jitter")
            if (
                self.densify_impl == "hexels_safe"
                and self.densify_clone_init_opacity > 0.0
                and self.opacity_cull >= self.densify_clone_init_opacity
            ):
                raise ValueError("--opacity_cull must be smaller than --densify_clone_init_opacity")
            if self.opacity_reset_interval < 0:
                raise ValueError("--opacity_reset_interval must be >= 0 (0 disables)")
            if (
                self.densify_impl == "hexels_safe"
                and self.opacity_reset_interval > 0
                and self.opacity_cull >= 0.01
            ):
                raise ValueError("hexels_safe opacity resets require --opacity_cull < 0.01")
            if not (0.0 <= self.opacity_cull <= 1.0):
                raise ValueError("--opacity_cull must be in [0,1]")
            if self.percent_dense <= 0:
                raise ValueError("--percent_dense must be > 0")
        if self.densify_max_screen_size < 0:
            raise ValueError("--densify_max_screen_size must be >= 0 (0 disables)")

        if self.lr_means3d_delay_steps < 0:
            raise ValueError("--lr_means3d_delay_steps must be >= 0")
        if not (0.0 < self.lr_means3d_delay_mult <= 1.0):
            raise ValueError("--lr_means3d_delay_mult must be in (0,1]")

        if self.model_path is None:
            scene_name = Path(self.source_path).name
            ts = time.strftime("%Y-%m-%d_%H-%M-%S")
            self.model_path = str(Path("outputs") / "hexels3d" / scene_name / ts)

        if self.test_train_views <= 0:
            raise ValueError("--test_train_views must be > 0")
        if self.wandb_test_images < 0:
            raise ValueError("--wandb_test_images must be >= 0")
