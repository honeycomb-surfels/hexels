from types import SimpleNamespace

import pytest
import torch

from test_hexels3d_correctness import _make_densify_trainer
from hexels.refinement import rank_absolute_candidates
from hexels.options import Hexels3DOptions


def test_capped_random_split_never_appends_above_budget() -> None:
    trainer = _make_densify_trainer(num_primitives=3, cap_max=5)
    trainer.opt.densify_action = "mixed"
    trainer.opt.densify_split_jitter = 0.5
    trainer.densify_birth_iteration = torch.zeros(3, dtype=torch.long)
    populations = []
    append = trainer._append_primitives

    def track_append(rows: dict, **kwargs) -> None:
        populations.append(trainer.model.num_base + rows["means3d"].shape[0])
        append(rows, **kwargs)

    trainer._append_primitives = track_append
    trainer._densify_and_prune_hexels_safe(size_threshold=0, iteration=1000)
    assert trainer.model.num_base == 5
    assert max(populations) == trainer.densify_peak_population == 5
    assert (trainer.densify_birth_iteration == 1000).sum() == 4
    assert trainer.optimizer.param_groups[0]["params"][0] is trainer.model.means3d


def test_append_rejects_budget_overflow_before_mutation() -> None:
    trainer = _make_densify_trainer(num_primitives=3, cap_max=3)
    original = trainer.model.means3d
    with pytest.raises(ValueError, match="exceed budget"):
        trainer._append_primitives({"means3d": torch.zeros(1, 3)})
    assert trainer.model.means3d is original
    assert trainer.model.num_base == 3


def make_prune_trainer():
    trainer = _make_densify_trainer(num_primitives=4, cap_max=4)
    trainer.opt.densify_coordinated_prune = True
    trainer.opt.densify_prune_interval = 100
    trainer.opt.densify_prune_mean_threshold = 0.02
    trainer.opt.densify_prune_max_fraction = 0.5
    trainer.opt.densify_newborn_grace = 20
    trainer.densify_birth_iteration = torch.tensor([0, 0, 95, 0])
    trainer.densify_prune_audit = []
    trainer.bg_color = torch.zeros(3)
    trainer.train_cameras = [SimpleNamespace(image_name=name, image_height=10, image_width=10)
                             for name in ("first", "second")]
    trainer._render_model = lambda: trainer.model
    visited = []

    def render(camera, model, background, **kwargs):
        visited.append(camera.image_name)
        return {"contrib": torch.tensor([0.0, 1.0, 0.0, 20.0])}

    trainer.renderer = SimpleNamespace(render=render)
    return trainer, visited


def test_prune_sweeps_all_views_and_protects_newborns() -> None:
    trainer, visited = make_prune_trainer()
    trainer._prune_coordinated_primitives(iteration=100)
    assert visited == ["first", "second"]
    assert trainer.model.means3d[:, 0].tolist() == [2.0, 3.0]
    assert trainer.densify_prune_audit[-1]["pruned"] == 2


def test_prune_recovers_growth_space_at_full_budget() -> None:
    trainer, visited = make_prune_trainer()
    trainer._densify_and_prune_hexels_safe(size_threshold=0, iteration=100)
    assert len(visited) == 2
    assert trainer.model.num_base == 3
    assert trainer.densify_peak_population <= 4


def test_prune_does_not_render_between_scheduled_events() -> None:
    trainer, visited = make_prune_trainer()
    trainer._prune_coordinated_primitives(iteration=99)
    assert visited == []
    assert trainer.model.num_base == 4


def test_max_view_guard_preserves_a_locally_important_row() -> None:
    trainer, visited = make_prune_trainer()
    trainer.opt.densify_prune_mean_threshold = 0.11
    trainer.opt.densify_prune_max_view_threshold = 0.15

    def render(camera, model, background, **kwargs):
        visited.append(camera.image_name)
        return {"contrib": torch.tensor([0.0, 20.0 if camera.image_name == "first" else 0.0, 0.0, 40.0])}

    trainer.renderer.render = render
    trainer._prune_coordinated_primitives(iteration=100)
    assert trainer.model.means3d[:, 0].tolist() == [1.0, 2.0, 3.0]
    assert trainer.densify_prune_audit[-1]["max_view_protected"] == 1
    assert trainer.densify_prune_audit[-1]["pruned"] == 1


def test_absolute_ranking_changes_selection_not_candidate_count() -> None:
    signed = torch.tensor([0.2, 0.3, 0.01, 0.01])
    absolute = torch.tensor([0.2, 0.3, 2.0, 10.0])
    eligible = torch.tensor([True, True, True, False])
    selected = rank_absolute_candidates(signed, absolute, eligible, 0.1)
    assert set(selected.tolist()) == {1, 2}
    assert rank_absolute_candidates(signed, absolute, eligible, 3.0).numel() == 0


def test_absolute_stats_survive_population_changes() -> None:
    trainer = _make_densify_trainer(num_primitives=3, cap_max=5)
    trainer.opt.densify_action = "mixed"
    trainer.opt.densify_absgrad_rank = True
    trainer.opt.densify_split_jitter = 0.5
    trainer.densify_birth_iteration = torch.zeros(3, dtype=torch.long)
    trainer.densify_abs_grad_accum = torch.ones(3, 1)
    trainer.densify_score_audit = []
    proxy = torch.zeros(3, 4, requires_grad=True)
    proxy.grad = torch.tensor([[3.0, 4.0, 0.0, 9.0]] * 3)
    trainer._densify_add_stats(proxy, torch.ones(3, dtype=torch.bool))
    assert trainer.densify_grad_accum[:, 0].tolist() == [6.0, 7.0, 8.0]
    assert trainer.densify_abs_grad_accum[:, 0].tolist() == [10.0] * 3
    trainer._densify_and_prune_hexels_safe(size_threshold=0, iteration=1000)
    assert trainer.model.num_base == 5
    assert trainer.densify_abs_grad_accum.shape == (5, 1)
    assert trainer.densify_abs_grad_accum.count_nonzero() == 0
    trainer._prune_primitives(torch.tensor([True, False, False, False, False]))
    assert trainer.densify_abs_grad_accum.shape == (4, 1)


@pytest.mark.parametrize("basis,view_model", [(3, "linear"), (15, "sh3")])
@pytest.mark.parametrize("action", ["clone", "mixed"])
def test_growth_preserves_view_coefficients_and_optimizer(basis: int, view_model: str, action: str) -> None:
    trainer = _make_densify_trainer(num_primitives=3, cap_max=5)
    trainer.opt.densify_action = action
    trainer.opt.densify_split_jitter = 0.5
    trainer.opt.lr_sh_coeffs = 0.0005
    trainer.view_model = view_model
    trainer.densify_birth_iteration = torch.zeros(3, dtype=torch.long)
    original = torch.arange(3 * basis * 3, dtype=torch.float32).reshape(3, basis, 3)
    trainer.model.view_grads = torch.nn.Parameter(original.clone())
    trainer.optimizer = trainer._build_optimizer()
    trainer._densify_and_prune_hexels_safe(size_threshold=0, iteration=1000)
    assert trainer.model.view_grads.shape == (5, basis, 3)
    for row in trainer.model.view_grads:
        assert any(torch.equal(row, parent) for parent in original)
    assert any(group["params"][0] is trainer.model.view_grads for group in trainer.optimizer.param_groups)
    before = trainer.model.view_grads.detach().clone()
    trainer.optimizer.zero_grad(set_to_none=True)
    trainer.model.view_grads.sum().backward()
    trainer.optimizer.step()
    assert torch.all(trainer.model.view_grads.detach() < before)


@pytest.mark.parametrize("threshold", [-0.5, float("nan"), float("inf")])
def test_invalid_max_view_threshold_is_rejected(threshold: float) -> None:
    options = Hexels3DOptions()
    options.source_path = "garden"
    options.model_path = "output"
    options.enable_densify = True
    options.densify_impl = "hexels_safe"
    options.densify_prune_max_view_threshold = threshold
    with pytest.raises(ValueError, match="Prune max-view threshold"):
        options.process_args()


def test_absolute_ranking_rejects_unsupported_view_model() -> None:
    options = Hexels3DOptions()
    options.source_path = "garden"
    options.model_path = "output"
    options.enable_densify = True
    options.densify_impl = "hexels_safe"
    options.densify_absgrad_rank = True
    options.view_model = "sh3"
    with pytest.raises(ValueError, match="Absolute-gradient ranking"):
        options.process_args()
