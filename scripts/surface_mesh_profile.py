"""Published MeshSplatting weights with explicit normal-loss activation."""
from __future__ import annotations

from argparse import Namespace


def apply_paper_profile(options: Namespace, dataset: str) -> dict[str, float | int]:
    """Apply published weights/start; align normal activation with mesh conversion.

    ``iteration_mesh`` gates normal losses. Actual mesh conversion already uses
    ``densify_until_iter + 1000`` in the released training loop.
    """
    if dataset not in {"dtu", "tanks_and_temples"}:
        raise ValueError(f"Unsupported surface dataset: {dataset}")
    values = {"densify_from_iter": 500, "iteration_mesh": 11000,
              "lambda_normals": 0.0001, "lambda_vertex": 0.00025}
    if dataset == "dtu":
        values.update(depth_lambda_init=0.0, depth_lambda_final=0.0,
                      lambda_normals_super=0.0, lambda_weight=0.0)
    else:
        values.update(lambda_weight=0.000002)
    for name, value in values.items():
        setattr(options, name, value)
    return values
