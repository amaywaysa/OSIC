#!/bin/bash
#SBATCH --job-name=multi_all_features
#SBATCH --nodes=1
#SBATCH --partition=gpu                    # You're missing this!
#SBATCH --gres=gpu:4                       # Use this instead of --gpus-per-node
#SBATCH --cpus-per-task=12                 # Changed from --cpus-per-gpu
#SBATCH --ntasks=4                         # One task per GPU
#SBATCH --mem=64GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=emma.chanut@boehringer-ingelheim.com
#SBATCH --time=300:00:00

# Install requirements
#pip install -r requirements.txt

python training_multi_one_hot_cv.py 
