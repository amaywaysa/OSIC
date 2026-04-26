#!/bin/bash
#SBATCH --job-name=xai_multimodal_explainability
#SBATCH --nodes=1
#SBATCH --partition=gpu                    # You're missing this!
#SBATCH --gres=gpu:4                       # Use this instead of --gpus-per-node
#SBATCH --cpus-per-task=12                 # Changed from --cpus-per-gpu
#SBATCH --ntasks=4                         # One task per GPU
#SBATCH --mem=64GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=emma.chanut@boehringer-ingelheim.com # add your email here
#SBATCH --time=300:00:00

# Install requirements
#pip install -r requirements.txt
pip install shap

# python explainability_tabular_mlp_rf.py \
#   --mlp-run-dir cv_tabular_one_hot_2026-03-31_15-22-26 \
#   --mlp-timestamp 2026-03-31_15-22-26 \
#   --seed 42 \
#   --num-folds 5

# python explainability_multimodal_permutation.py \
#   --model-dir cv_multi_one_hot_5_cv_scaled_ES2026-03-31_08-58-23 \
#   --model-timestamp 2026-03-31_08-58-23 \
#   --perm-repeats 30 \
#   --hidden-dim 256

python smoothgrad_image_only_selected_patients.py
#python smoothgrad_multimodal_selected_patients.py
#python multimodal_prediction_5fold.py
#python image_only_prediction.py