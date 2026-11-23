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

# Initialize SCC's modules system.
source /usr/share/Modules/init/bash

module load miniconda academic-ml/fall-2026
conda activate fall-2026-pyt

cd /projectnb/cs541/students/alexpfau/CS541CPRKaggle

echo "Starting feature extraction"
date
nvidia-smi