"""Native capacity controls for measured per-scene learned-state budgets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

PRIMITIVE_BYTES = {"2dgs": 232, "3dgs": 236, "3d_convex": 276, "triangle": 236, "bbsplat": 4324}


def resolve(run: dict, roots: list[Path]) -> dict:
    """Resolve a completed sparse Hexel anchor without accepting smoke models."""
    paths = [root / folder / run["budget_anchor"] / "metrics.json" for root in roots for folder in ("accepted", "runs")]
    paths = [path for path in paths if path.exists() and json.loads(path.read_text()).get("complete")]
    if not paths:
        raise ValueError(f"No complete anchor for {run['id']}")
    path = paths[0]
    job = json.loads(path.with_name("job.json").read_text())
    result = json.loads(path.read_text())
    if job["smoke"] or job["seed"] != 0 or job["iteration"] != 30000:
        raise ValueError("Budget anchors must be 30K seed-zero production models")
    if result["id"] != run["budget_anchor"]:
        raise ValueError("Anchor identity differs")
    population = result["artifact"]["primitives"]
    if not 0 < population <= 2000000:
        raise ValueError("Invalid Hexel population anchor")
    budget = None
    method = run["method"]
    architecture = {}
    if "_count_" in run["arm"]:
        cap = population
    else:
        variant = run["arm"].split("_")[-1]
        budget = population * {"v1": 80, "v2": 116, "v3": 260}[variant]
        if method == "mesh_splatting":
            initial_points = budget // 636
            if initial_points < 4:
                raise ValueError("Budget cannot fit four MeshSplatting initialization points")
            cap = budget // 12
            architecture = {"initial_sfm_points": initial_points, "initialization_seed": 0,
                            "disable_densification": True, "initial_state_bytes": initial_points * 636}
            if "_initial_bytes_" in run["arm"]:
                architecture["budget_scope"] = "initialization"
        elif method == "nexels":
            candidates = [size for size in range(10, 21) if 128 * 2**size + 37568 <= budget // 2]
            if not candidates:
                raise ValueError(f"Budget cannot fit the minimum Nexels architecture: {budget}")
            size = max(candidates)
            fixed_bytes = 128 * 2**size + 37568
            cap = (budget - fixed_bytes) // 256
            architecture = {"log_hash_table_size": size, "shared_state_bytes": fixed_bytes}
        else:
            cap = budget // PRIMITIVE_BYTES[method]
    if cap < 1:
        raise ValueError("Budget cannot fit a primitive")
    policy = ("SfM subsampling and no densification; native pruning/remeshing; measured byte gates without face caps"
              if architecture.get("disable_densification") else "native adaptive population under a ceiling; no forced final count")
    if architecture.get("budget_scope") == "initialization":
        policy = "initialization-matched state bytes; no densification; later byte excess is reported without stopping or face caps"
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "primitive_ceiling": cap, "active_learned_byte_budget": budget, **architecture,
            "policy": policy}


def apply(command: list[str], job: dict, evidence: dict) -> None:
    """Set exposed native controls without disabling pruning or opacity resets."""
    from scripts.run_reproduction import set_option

    method, cap = job["method"], evidence["primitive_ceiling"]
    if method == "mesh_splatting" and evidence.get("disable_densification"):
        for name in ("initial_sfm_points", "initialization_seed"):
            set_option(command, f"--{name}", evidence[name])
        set_option(command, "--audit_scene_bytes", evidence["active_learned_byte_budget"])
        command.append("--disable_densification")
        if evidence.get("budget_scope") == "initialization":
            command.append("--initial_memory_only")
    elif method in {"2dgs", "3dgs", "3d_convex", "mesh_splatting"}:
        set_option(command, "--max_primitives", cap)
    elif method == "triangle":
        set_option(command, "--max_shapes", cap)
    elif method == "bbsplat":
        sky = 0 if job["dataset"] == "dtu" else max(1, int(cap * 0.05))
        for name, value in {"cap_max": cap, "max_read_points": cap - sky, "sphere_point": sky}.items():
            set_option(command, f"--{name}", value)
    elif method == "nexels":
        controls = {"cap_max_init": min(cap, 200000), "cap_max_final": cap}
        if "log_hash_table_size" in evidence:
            controls["log_hash_table_size"] = evidence["log_hash_table_size"]
        for name, value in controls.items():
            set_option(command, f"--{name}", value)
    else:
        raise ValueError(f"No approved budget control for {method}")


def validate(artifact: dict, evidence: dict) -> None:
    """Reject realized models that exceed their declared ceiling or state budget."""
    initial_only = evidence.get("budget_scope") == "initialization"
    if not initial_only and artifact["primitives"] > evidence["primitive_ceiling"]:
        raise ValueError("Final primitive count exceeds capacity budget")
    budget = evidence["active_learned_byte_budget"]
    if not initial_only and budget is not None and artifact["compact_learned_bytes"] > budget:
        raise ValueError(f"Realized learned state exceeds budget: {artifact['compact_learned_bytes']} > {budget}")
    if evidence.get("disable_densification"):
        audit = artifact["initialization_budget_audit"]
        stages = audit["stages"]
        if audit["budget_bytes"] != budget or [stage["stage"] for stage in stages] != ["initialization", "remeshing", "final"]:
            raise ValueError("Missing or mismatched no-growth memory audit")
        if stages[0]["state_bytes"] != evidence["initial_state_bytes"] or stages[0]["vertices"] != 3 * evidence["initial_sfm_points"]:
            raise ValueError("Initialization differs from the declared SfM subset")
        enforced_stages = stages[:1] if initial_only else stages
        if any(stage["vertices"] > stages[0]["vertices"] for stage in stages) or any(stage["state_bytes"] > budget for stage in enforced_stages):
            raise ValueError("No-growth model exceeded its memory or vertex budget")
        if stages[-1]["state_bytes"] != artifact["compact_learned_bytes"] or stages[-1]["faces"] != artifact["primitives"]:
            raise ValueError("Final model differs from its memory audit")
        if artifact["topology"]["unreferenced_vertices"]:
            raise ValueError("Final mesh contains unreferenced vertices")
