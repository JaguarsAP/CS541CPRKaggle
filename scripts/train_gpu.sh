#!/bin/bash -l
#$ -P cs541
#$ -N cpr_train
#$ -l gpus=1
#$ -l gpu_c=7.5
#$ -pe omp 4
#$ -l h_rt=02:00:00
#$ -cwd
#$ -j y

set -e

module load miniconda academic-ml/fall-2026
conda activate fall-2026-pyt

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

cd /projectnb/cs541/students/$USER/CS541CPRKaggle

echo "========================================"
echo "CPR Classifier Experiments"
echo "Start date: $(date)"
echo "Job ID: $JOB_ID"
echo "========================================"

python -c "import torch; assert torch.cuda.is_available(); print('GPU:', torch.cuda.get_device_name(0))"

for MODEL in linear mlp statistical lstm temporal_cnn
do

    echo "========================================"
    echo "Training model: $MODEL"
    date

    python -u scripts/train_classifier.py \
        --model "$MODEL" \
        --epochs 100 \
        --batch-size 32 \
        --lr 0.001 \
        --patience 15 \
        --threshold 0.5

    echo "Finished model: $MODEL"
    date

done

echo "ALL TRAINING EXPERIMENTS COMPLETE"
echo "End date: $(date)"
