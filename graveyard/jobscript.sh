#!/bin/sh
#SBATCH --account=e32706
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1 
#SBATCH --time=01:00:00 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=20G
#SBATCH --output=/home/qfx2034/roadsigns/training.log

module purge
eval "$(conda shell.bash hook)"
conda activate /home/qfx2034/.conda/envs/roadsigns

python quest_train.py