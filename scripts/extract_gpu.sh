
#!/bin/bash
#$ -S /bin/bash
#$ -P cs541
#$ -l gpus=1
#$ -l h_rt=04:00:00
#$ -pe omp 4
#$ -N cpr_features
#$ -j y
#$ -cwd

set -e

# Initialize SCC's Lmod environment.
source /usr/local/lmod/8.7.12/init/bash

module load miniconda academic-ml/fall-2026
conda activate fall-2026-pyt

cd /projectnb/cs541/students/alexpfau/CS541CPRKaggle

echo "Starting feature extraction"
date
nvidia-smi

python -u scripts/extract_features.py \
    --split train \
    --batch-size 2 \
    --num-workers 2

python -u scripts/extract_features.py \
    --split val \
    --batch-size 2 \
    --num-workers 2

echo "Feature extraction complete"
date
