"""Restore audited external sources and build isolated A5000 baseline environments."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METHODS = {
    "3dgs": ("gaussian-splatting", "3.11", "2.4.0", "0.19.0", "cu124", ["diff-gaussian-rasterization", "simple-knn", "fused-ssim"]),
    "3d_convex": ("convex-splatting", "3.11", "2.4.0", "0.19.0", "cu124", ["diff-convex-rasterization", "simple-knn"]),
    "triangle": ("triangle-splatting", "3.11", "2.4.0", "0.19.0", "cu124", ["diff-triangle-rasterization", "simple-knn"]),
    "bbsplat": ("BBSplat", "3.8", "2.0.0", "0.15.1", "cu118", ["diff-bbsplat-rasterization", "simple-knn"]),
    "nexels": ("nexels", "3.8", "2.1.2", "0.16.2", "cu118", ["diff-nexel-rasterization", "simple-knn", "fused-ssim"]),
    "mesh_splatting": ("mesh-splatting", "3.11", "2.7.1", "0.22.1", "cu126", ["diff-triangle-mesh-rasterization", "simple-knn", "effrdel"]),
    "milo": ("MILo", "3.9", "2.3.1", "0.18.1", "cu118", ["diff-gaussian-rasterization_ms", "diff-gaussian-rasterization", "diff-gaussian-rasterization_gof", "simple-knn", "fused-ssim"]),
}
EXTRAS = {
    "bbsplat": [
        "git+https://github.com/facebookresearch/pytorch3d.git@9381c4016376345bb795b97c45a6c2de66db354a",
        "git+https://github.com/NVlabs/nvdiffrast.git@253ac4fcea7de5f396371124af597e6cc957bfae",
    ],
    "nexels": ["git+https://github.com/NVlabs/tiny-cuda-nn.git@749dd70c5afc5a9dadb85e5652ed65d55e0ba187#subdirectory=bindings/torch"],
}


def environment_prefix(method: str) -> Path:
    name = "nvs-milo-v2" if method == "milo" else f"nvs-{method}"
    return ROOT / ".envs" / name


def run(command: list[str], *, cwd: Path = ROOT, environment: dict[str, str] | None = None) -> None:
    logging.info("Running %s in %s", command, cwd)
    subprocess.run(command, cwd=cwd, env=environment, check=True)


def cuda_architectures(environment: dict[str, str]) -> None:
    """Keep user build targets; default to the tested A5000 (8.6), with TCNN/CMake following PyTorch's list."""
    torch_list = environment.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")
    capabilities = ";".join(item.split("+")[0].replace(".", "")
                            for item in torch_list.replace(" ", ";").split(";") if item)
    environment.setdefault("TCNN_CUDA_ARCHITECTURES", capabilities)
    environment.setdefault("CMAKE_CUDA_ARCHITECTURES", capabilities)
    logging.info("CUDA architectures: PyTorch %s, TCNN %s, CMake %s", environment["TORCH_CUDA_ARCH_LIST"],
                 environment["TCNN_CUDA_ARCHITECTURES"], environment["CMAKE_CUDA_ARCHITECTURES"])


def restore(method: str, output: Path) -> Path:
    """Fetch the pinned, attributed source and local adaptation patch."""
    from scripts.fetch_baselines import fetch

    repository = fetch(METHODS[method][0])
    output.mkdir(parents=True, exist_ok=True)
    specification = ROOT / "baselines" / repository.name / "source.json"
    (output / f"{method}_source.json").write_bytes(specification.read_bytes())
    return repository


def install(method: str, output: Path, conda: str) -> None:
    """Install native packages only inside an allocated GPU batch job."""
    _, version, torch_version, vision_version, cuda, packages = METHODS[method]
    if cuda != "cu118" and not os.environ.get("CUDA_HOME"):
        raise ValueError(f"{method} ({cuda}) compiles against a local CUDA 12.x toolkit: "
                         "export CUDA_HOME=/path/to/that/toolkit (containing bin/nvcc) and rerun")
    devices = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True)
    repository = restore(method, output)
    prefix = environment_prefix(method)
    python = prefix / "bin/python"
    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    environment["CONDA_PKGS_DIRS"] = str(output.resolve() / "package_cache" / method)
    environment["CONDA_ANACONDA_ANON_USAGE"] = "false"
    if not python.exists():
        run([conda, "create", "--prefix", str(prefix), "--yes", f"python={version}", "pip", "--override-channels", "-c", "conda-forge"], environment=environment)
    environment.update(PATH=f"{prefix / 'bin'}:{environment['PATH']}", MAX_JOBS="4",
                       CONDA_PREFIX=str(prefix), TORCH_HOME=str(ROOT / "data/torch_cache"), MPLBACKEND="Agg")
    cuda_architectures(environment)
    if cuda == "cu118":
        if not (prefix / "bin/nvcc").exists():
            run([conda, "install", "--prefix", str(prefix), "--yes", "cuda-toolkit=11.8.0",
                 "--override-channels", "-c", "nvidia/label/cuda-11.8.0", "-c", "conda-forge"], environment=environment)
        if method == "milo":
            run([conda, "install", "--prefix", str(prefix), "--yes", "--freeze-installed",
                 "--override-channels", "-c", "nvidia/label/cuda-11.8.0", "-c", "conda-forge",
                 "cuda-toolkit=11.8.0", "cmake<4", "gmp", "cgal"], environment=environment)
        environment["CUDA_HOME"] = str(prefix)
    else:
        environment["CUDA_HOME"] = os.environ["CUDA_HOME"]
    environment["PATH"] = f"{environment['CUDA_HOME']}/bin:{environment['PATH']}"
    environment["LD_LIBRARY_PATH"] = f"{prefix / 'lib'}:{environment['CUDA_HOME']}/lib64:{environment.get('LD_LIBRARY_PATH', '')}"
    pip = [str(python), "-m", "pip", "install"]
    run([*pip, "setuptools<76", "wheel", "ninja", "packaging"], environment=environment)
    run([*pip, f"torch=={torch_version}", f"torchvision=={vision_version}",
         "--index-url", f"https://download.pytorch.org/whl/{cuda}"], environment=environment)
    numpy = "numpy==1.24.4" if version == "3.8" else "numpy==1.26.4"
    common = [numpy, "scipy", "pillow", "tqdm", "plyfile", "tensorboard", "lpips", "psutil",
              "nvidia-ml-py", "imageio", "imageio-ffmpeg", "opencv-python==4.9.0.80", "matplotlib",
              "scikit-image", "trimesh", "pyyaml", "open3d==0.18.0", "easydict", "rich", "termcolor", "pybind11"]
    if method == "bbsplat":
        common += ["iopath", "xatlas", "mediapy"]
    if method == "nexels":
        common += ["wandb", "gitpython", "jaxtyping", "viser", "fpsample==1.0.1"]
    if method == "mesh_splatting":
        common += ["timm", "yapf", "gstaichi==4.6.0", "mmengine==0.10.7", "libigl", "mediapy"]
    if method == "triangle":
        common += ["mediapy==1.2.7"]
    run([*pip, *common], environment=environment)
    for package in packages:
        editable = ["-e"] if package == "effrdel" else []
        run([*pip, "--no-build-isolation", "--no-deps", *editable, str(repository / "submodules" / package)], environment=environment)
    for package in EXTRAS.get(method, []):
        run([*pip, "--no-build-isolation", "--no-deps", package], environment=environment)
    if method == "milo":
        tetra = repository / "submodules/tetra_triangulation"
        torch_cmake = subprocess.check_output([str(python), "-c", "import torch; print(torch.utils.cmake_prefix_path)"], env=environment, text=True).strip()
        environment["CMAKE_ARGS"] = (f"-DCMAKE_PREFIX_PATH={prefix};{torch_cmake} "
                                     f"-DCMAKE_CUDA_ARCHITECTURES={environment['CMAKE_CUDA_ARCHITECTURES']} -DCONDA_PREFIX={prefix}")
        run(["cmake", ".", *environment["CMAKE_ARGS"].split(), f"-DPYTHON_EXECUTABLE={python}"], cwd=tetra, environment=environment)
        run(["cmake", "--build", ".", "--parallel", "4"], cwd=tetra, environment=environment)
        run([*pip, "--no-build-isolation", "--no-deps", "-e", str(tetra)], environment=environment)
        run([*pip, "--no-build-isolation", "--no-deps", str(repository / "submodules/nvdiffrast")], environment=environment)
        run([str(python), "-c", "import torch; from tetranerf.utils.extension import cpp; print(cpp.__file__)"], environment=environment)
    run([str(python), "-c", "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"], environment=environment)
    (output / f"{method}_pip_freeze.txt").write_bytes(subprocess.check_output([str(python), "-m", "pip", "freeze"], env=environment))
    (output / f"{method}_build.json").write_text(json.dumps({
        "method": method, "status": "packages_installed_smoke_required", "python": str(python),
        "cuda_home": environment["CUDA_HOME"], "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cuda_architectures": {name: environment[name] for name in
                               ("TORCH_CUDA_ARCH_LIST", "TCNN_CUDA_ARCHITECTURES", "CMAKE_CUDA_ARCHITECTURES")},
    }, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["source", "install"])
    parser.add_argument("--methods", choices=METHODS, nargs="+", default=list(METHODS))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conda", default="conda")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    for method in args.methods:
        if args.action == "source":
            restore(method, args.output)
        else:
            install(method, args.output, args.conda)


if __name__ == "__main__":
    main()
