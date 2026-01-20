#!/bin/bash

#SBATCH --partition=main
#SBATCH --time=0-10:00:00
#SBATCH --job-name=biomarker_run
#SBATCH --output=logs_%J.out
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

module load anaconda
source activate biomarkers-discovery-env

python -u run_all.py