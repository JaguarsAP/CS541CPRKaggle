#!/bin/bash -l
#$ -P cs541
#$ -N cpr_features
#$ -l gpus=1
#$ -l gpu_c=7.5
#$ -pe omp 4
#$ -l h_rt=08:00:00
#$ -cwd
#$ -j y

set -e

echo "========================================"
echo "CPR RGB Feature Extraction"
echo "Start date: $(date)"
echo "Job ID: $JOB_ID"
echo "Node: $(hostname)"
echo "========================================"

module load miniconda academic-ml/fall-2026
conda activate fall-2026-pyt

export HF_HOME=/projectnb/cs541/students/$USER/.cache/huggingface
export TORCH_HOME=/projectnb/cs541/students/$USER/.cache/torch

mkdir -p "$HF_HOME" "$TORCH_HOME"

cd /projectnb/cs541/students/$USER/CS541CPRKaggle

python -c "import torch; assert torch.cuda.is_available(); print('GPU:', torch.cuda.get_device_name(0))"

echo "Starting training feature extraction"
date

python -u scripts/extract_features.py \
    --split train \
    --batch-size 2 \
    --num-workers 2

echo "Training feature extraction complete"
date

echo "Starting validation feature extraction"

python -u scripts/extract_features.py \
    --split val \
    --batch-size 2 \
    --num-workers 2

echo "========================================"
echo "ALL FEATURE EXTRACTION COMPLETE"
echo "End date: $(date)"
echo "========================================"
