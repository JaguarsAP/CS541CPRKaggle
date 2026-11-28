#!/bin/bash -l
#$ -P cs541
#$ -N cpr_gpu_test
#$ -l gpus=1
#$ -l gpu_c=7.5
#$ -pe omp 4
#$ -l h_rt=00:15:00
#$ -cwd
#$ -j y

set -e

echo "========================================"
echo "GPU diagnostic test"
echo "Start date: $(date)"
echo "Job ID: $JOB_ID"
echo "Node: $(hostname)"
echo "========================================"

echo "Checking module environment"
echo "MODULEPATH: $MODULEPATH"

module load miniconda academic-ml/fall-2026
conda activate fall-2026-pyt

export HF_HOME=/projectnb/cs541/students/$USER/.cache/huggingface
export TORCH_HOME=/projectnb/cs541/students/$USER/.cache/torch

mkdir -p "$HF_HOME" "$TORCH_HOME"

python -u - <<'PY'
import torch
import torchvision

print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())

assert torch.cuda.is_available(), "CUDA unavailable!"

device = torch.device("cuda:0")

print("GPU:", torch.cuda.get_device_name(0))
print("Compute capability:", torch.cuda.get_device_capability(0))
print("GPU memory (GB):",
      torch.cuda.get_device_properties(0).total_memory / 1024**3)

x = torch.randn(1024, 1024, device=device)
y = x @ x
torch.cuda.synchronize()

print("CUDA matrix multiplication successful!")
print("ALL GPU TESTS PASSED!")
PY

echo "End date: $(date)"
