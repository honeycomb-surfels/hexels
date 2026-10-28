#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000M
#SBATCH --time=24:00:00

set -euo pipefail

root="${HEXELS_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "$root"
python_bin="${PYTHON_BIN:-$root/.envs/hexels311/bin/python}"
manifest="${1:?Pass a prepared train.json or evaluate.json}"
state_root="${2:?Pass the matching runner state directory}"
export PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export MPLBACKEND=Agg
export TORCH_HOME="$root/data/torch_cache"
export QT_QPA_PLATFORM=offscreen
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,pci.bus_id --format=csv

if [[ -n "${3:-}" ]]; then
    exec "$python_bin" -m scripts.run_minimum_3d_benchmark \
        --manifest "$manifest" --state-root "$state_root" \
        --job-name "$3" --gpu "${CUDA_VISIBLE_DEVICES:-0}"
fi

"$python_bin" -m scripts.run_minimum_3d_benchmark \
    --manifest "$manifest" --state-root "$state_root" \
    --worker-index "${SLURM_ARRAY_TASK_ID:?Use a zero-based contiguous Slurm array}" \
    --worker-count "${HEXELS_WORKER_COUNT:-${SLURM_ARRAY_TASK_COUNT:?}}" \
    --gpu "${CUDA_VISIBLE_DEVICES:-0}"
