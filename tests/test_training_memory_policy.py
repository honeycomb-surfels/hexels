import ast
import inspect
import json

import pytest
import torch

from scripts.training_memory_policy import CameraAudit, camera_tree, training_tree


def test_only_rgb_residency_changes() -> None:
    class Camera:
        def __init__(self, image, mask, data_device="cuda"):
            self.data_device = torch.device(data_device)
            self.original_image = image.clamp(0, 1).to(self.data_device)
            self.original_image *= torch.ones((1, image.shape[1], image.shape[2]), device=self.data_device)
            self.gt_alpha_mask = mask

    signature = inspect.signature(Camera)
    rng = torch.random.get_rng_state().clone()
    audit = CameraAudit()
    audit.install(Camera, move_rgb=True)
    image = torch.arange(12, dtype=torch.float32).reshape(3, 2, 2) / 12
    mask = torch.ones((1, 2, 2), device="meta")
    camera = Camera(image, mask)
    assert torch.equal(camera.original_image, image)
    assert camera.original_image.device.type == "cpu"
    assert camera.data_device.type == "cuda"
    assert camera.gt_alpha_mask is mask
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert inspect.signature(Camera) == signature
    assert audit.record()["rgb_cpu_bytes"] == 48
    assert audit.record()["rgb_cuda_cached_bytes"] == 0
    assert audit.record()["auxiliary_cache_bytes_by_device"] == {"gt_alpha_mask": {"meta": 16}}


def test_existing_cpu_camera_is_audited_without_rewriting_its_constructor() -> None:
    class Camera:
        def __init__(self, image):
            self.original_image = image.clamp(0, 1)

    original = Camera.__init__
    audit = CameraAudit()
    audit.install(Camera, move_rgb=False)
    Camera(torch.ones((3, 2, 2)))
    assert Camera.__init__.__wrapped__ is original
    assert audit.record()["constructor_overrides"] == {}
    assert audit.record()["camera_count"] == 1


def test_camera_audit_rejects_non_cpu_rgb() -> None:
    class Camera:
        def __init__(self):
            self.original_image = torch.ones((3, 2, 2), device="meta")

    audit = CameraAudit()
    audit.install(Camera, move_rgb=False)
    with pytest.raises(RuntimeError, match="RGB cache is not on CPU"):
        Camera()


def test_unknown_rgb_constructor_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unexpected native RGB"):
        camera_tree("def __init__(self, image):\n    self.original_image = image.cpu()", "camera.py")


NEXELS_SOURCE = """
def training():
    loss = l1_loss(image, target)
    loss.backward()
    appearance_optimizer.step()

def training_report(config, image, gt_image):
    lpips_test = 0.0
    l1_test = l1_loss(image, gt_image)
    psnr_test = psnr(image, gt_image)
    lpips_test += lpips(image, gt_image, net_type='vgg').mean().double()
    lpips_test /= len(config['cameras'])
    return l1_test, psnr_test
"""


def test_only_nexels_validation_vgg_statements_change() -> None:
    before = ast.parse(NEXELS_SOURCE)
    after, changes = training_tree(NEXELS_SOURCE, "train.py")
    assert ast.dump(before.body[0]) == ast.dump(after.body[0])
    assert changes == {"validation_vgg_statements_removed": 3, "validation_vgg_calls_removed": 1}
    calls = {node.func.id for node in ast.walk(after.body[1])
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert calls == {"l1_loss", "psnr"}
    compile(after, "train.py", "exec")


def test_unexpected_nexels_scoring_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unexpected Nexels VGG"):
        training_tree(NEXELS_SOURCE.replace("net_type='vgg'", "net_type='alex'"), "train.py")


@pytest.mark.parametrize("method", ["3dgs", "3d_convex", "triangle", "bbsplat", "mesh_splatting", "nexels", "milo"])
def test_external_baselines_keep_non_rgb_targets_on_cuda(method: str) -> None:
    from scripts.nvs_external import training_command

    job = {"method": method, "dataset": "dtu", "scene": "scan24", "model_path": "/runs/model",
           "iteration": 30000, "seed": 0}
    settings = {"source_path": "/data/dtu/scan24", "images": "images", "resolution": 2}
    command, _ = training_command(job, settings, False)
    assert command[command.index("--data_device") + 1] == "cuda"
    assert command[command.index("--iterations") + 1] == "30000"
    assert command[command.index("--test_iterations") + 1] == "-1"


def test_nvs_and_capacity_defaults_request_five_ms() -> None:
    from scripts.nvs_campaign import ROOT

    for name in ("nvs.json", "capacity.json"):
        config = json.loads((ROOT / "configs" / name).read_text())
        assert config["measurement"]["resource_sample_seconds"] == .005
