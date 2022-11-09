#!/bin/bash
#SBATCH --job-name=mimic_cxr_cs334
#SBATCH --output=out_slur
#SBATCH --gres=gpu:1
#SBATCH -o %A.out
#SBATCH -e %A.err

cd /local/scratch/zzaiman/CS334FinalProject/zach/trainlib
source ../venv/bin/activate
python train.py
