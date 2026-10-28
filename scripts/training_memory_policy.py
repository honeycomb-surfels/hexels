"""Keep training RGB caches on CPU while retaining native GPU targets and appearance."""

from __future__ import annotations

import argparse
import ast
from functools import wraps
import importlib
import inspect
import json
import logging
from pathlib import Path
import runpy
import sys
import textwrap
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
GPU_RGB_METHODS = {"3dgs", "3d_convex", "triangle", "bbsplat", "mesh_splatting", "nexels", "milo"}
TRAINING_METHODS = GPU_RGB_METHODS | {"hexels", "2dgs"}
TRAINING_MEMORY_POLICY = "cpu_rgb_only_no_nexels_validation_vgg"
AUXILIARY_FIELDS = ("alpha_mask", "gt_alpha_mask", "gt_mask", "invdepthmap", "depth_mask")
VGG_STATEMENTS = (
    "lpips_test = 0.0",
    "lpips_test += lpips(image, gt_image, net_type='vgg').mean().double()",
    "lpips_test /= len(config['cameras'])",
)
LOGGER = logging.getLogger(__name__)


def is_camera_field(node: ast.AST, name: str) -> bool:
    return (isinstance(node, ast.Attribute) and node.attr == name
            and isinstance(node.value, ast.Name) and node.value.id == "self")


class RgbOnly(ast.NodeTransformer):
    def __init__(self) -> None:
        self.transfers = 0
        self.multiplications = 0

    def visit_Assign(self, node: ast.Assign) -> ast.Assign:
        if any(is_camera_field(target, "original_image") for target in node.targets):
            value = node.value
            if (not isinstance(value, ast.Call) or not isinstance(value.func, ast.Attribute)
                    or value.func.attr != "to" or len(value.args) != 1
                    or not is_camera_field(value.args[0], "data_device") or value.keywords):
                raise ValueError("Unexpected native RGB device transfer")
            node.value = value.func.value
            self.transfers += 1
        return self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> ast.AugAssign:
        if is_camera_field(node.target, "original_image"):
            value = node.value
            if (not isinstance(value, ast.Call) or not isinstance(value.func, ast.Attribute)
                    or value.func.attr != "ones"):
                raise ValueError("Unexpected native RGB multiplication")
            devices = [keyword for keyword in value.keywords if keyword.arg == "device"]
            if len(devices) != 1 or not is_camera_field(devices[0].value, "data_device"):
                raise ValueError("Unexpected native RGB multiplier device")
            devices[0].value = ast.Attribute(
                value=ast.Attribute(value=ast.Name(id="self", ctx=ast.Load()),
                                    attr="original_image", ctx=ast.Load()),
                attr="device", ctx=ast.Load())
            self.multiplications += 1
        return self.generic_visit(node)


def camera_tree(source: str, filename: str) -> tuple[ast.Module, dict[str, int]]:
    transformer = RgbOnly()
    tree = transformer.visit(ast.parse(textwrap.dedent(source), filename=filename))
    if transformer.transfers != 1:
        raise ValueError(f"Expected one RGB transfer in {filename}")
    ast.fix_missing_locations(tree)
    return tree, {"rgb_transfers_removed": transformer.transfers,
                  "rgb_multiplier_devices_changed": transformer.multiplications}


class CameraAudit:
    """Keep cached RGB on CPU without moving masks, depth or camera transforms."""

    def __init__(self) -> None:
        self.cameras = 0
        self.rgb_bytes = 0
        self.auxiliary: dict[str, dict[str, int]] = {}
        self.changes: dict[str, int] = {}

    def install(self, camera_class: type, *, move_rgb: bool) -> None:
        original = camera_class.__init__
        initialize_rgb_cpu = original
        if move_rgb:
            tree, self.changes = camera_tree(inspect.getsource(original), original.__code__.co_filename)
            namespace = dict(original.__globals__)
            exec(compile(tree, original.__code__.co_filename, "exec"), namespace)
            initialize_rgb_cpu = namespace[original.__name__]

        @wraps(original)
        def initialize(camera: Any, *args: Any, **kwargs: Any) -> None:
            initialize_rgb_cpu(camera, *args, **kwargs)
            if camera.original_image.device.type != "cpu":
                raise RuntimeError("RGB cache is not on CPU")
            self.cameras += 1
            self.rgb_bytes += camera.original_image.numel() * camera.original_image.element_size()
            for name in AUXILIARY_FIELDS:
                tensor = getattr(camera, name, None)
                if tensor is not None:
                    devices = self.auxiliary.setdefault(name, {})
                    device = tensor.device.type
                    devices[device] = devices.get(device, 0) + tensor.numel() * tensor.element_size()

        camera_class.__init__ = initialize

    def record(self) -> dict[str, Any]:
        return {"camera_count": self.cameras, "rgb_device": "cpu", "rgb_cpu_bytes": self.rgb_bytes,
                "rgb_cuda_cached_bytes": 0, "auxiliary_cache_bytes_by_device": self.auxiliary,
                "constructor_overrides": self.changes}


def training_tree(source: str, filename: str) -> tuple[ast.Module, dict[str, int]]:
    """Remove only Nexels' three validation VGG statements, preserving its loss."""
    tree = ast.parse(source, filename=filename)
    reporters = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == "training_report"]
    if len(reporters) != 1:
        raise ValueError(f"Expected one Nexels training_report in {filename}")
    patterns = {ast.dump(ast.parse(statement).body[0]): 0 for statement in VGG_STATEMENTS}

    class WithoutVgg(ast.NodeTransformer):
        def remove_metric(self, node: ast.AST) -> ast.AST | None:
            key = ast.dump(node)
            if key in patterns:
                patterns[key] += 1
                return None
            return self.generic_visit(node)

        visit_Assign = remove_metric
        visit_AugAssign = remove_metric

    WithoutVgg().visit(reporters[0])
    if set(patterns.values()) != {1}:
        raise ValueError(f"Unexpected Nexels VGG validation statements in {filename}: {patterns}")
    ast.fix_missing_locations(tree)
    return tree, {"validation_vgg_statements_removed": sum(patterns.values()),
                  "validation_vgg_calls_removed": 1}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=sorted(TRAINING_METHODS), required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise ValueError("A native training command is required")
    if "--data_device" in command and command[command.index("--data_device") + 1].split(":")[0] != "cuda":
        raise ValueError("RGB-only CPU caching requires native --data_device cuda for other targets")
    sys.path.insert(0, str(ROOT))
    if args.method == "hexels":
        from hexels.colmap import _add_repo_deps_to_path
        _add_repo_deps_to_path()
    sys.path.insert(0, str(Path.cwd()))
    audit = CameraAudit()
    audit.install(importlib.import_module("scene.cameras").Camera,
                  move_rgb=args.method in GPU_RGB_METHODS)
    if args.method == "nexels":
        script = Path(command[0]).resolve()
        tree, changes = training_tree(script.read_text(), str(script))
        sys.argv = command
        namespace = {"__name__": "__main__", "__file__": str(script), "__package__": None}
        LOGGER.info("Nexels reporting policy: %s", changes)
        exec(compile(tree, str(script), "exec"), namespace)
    elif command[0] == "-m":
        sys.argv = [command[1], *command[2:]]
        runpy.run_module(command[1], run_name="__main__", alter_sys=True)
    else:
        sys.argv = command
        runpy.run_path(command[0], run_name="__main__")
    if audit.cameras == 0:
        raise RuntimeError("No RGB camera-cache observations were recorded")
    LOGGER.info("Training RGB cache: %s", json.dumps(audit.record(), sort_keys=True))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
