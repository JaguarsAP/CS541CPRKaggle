#!/bin/bash -l
#$ -P cs541
#$ -N cpr_predict
#$ -l gpus=1
#$ -l gpu_c=7.5
#$ -pe omp 4
#$ -l h_rt=02:00:00
#$ -cwd
#$ -j y

set -e

module load miniconda academic-ml/fall-2026
conda activate fall-2026-pyt

export HF_HOME=/projectnb/cs541/students/$USER/.cache/huggingface
export TORCH_HOME=/projectnb/cs541/students/$USER/.cache/torch

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

cd /projectnb/cs541/students/$USER/CS541CPRKaggle

echo "========================================"
echo "CPR Test Extraction and Prediction"
echo "Start date: $(date)"
echo "Job ID: $JOB_ID"
echo "Node: $(hostname)"
echo "========================================"

python -c "import torch; assert torch.cuda.is_available(); print('GPU:', torch.cuda.get_device_name(0))"

echo "Starting test feature extraction"

python -u scripts/extract_test_features.py \
    --batch-size 2 \
    --num-workers 2

echo "Test feature extraction finished"
date

echo "Generating Kaggle submissions"

python -u scripts/make_submission.py

echo "========================================"
echo "ALL PREDICTION EXPERIMENTS COMPLETE"
echo "End date: $(date)"
echo "========================================"
