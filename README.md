# Tracked Files Quick Guide

This is a fast map of tracked files in this repository: what each one does and how they connect.

## 1) End-to-End Flow

Data preparation and exploration
-> Split setup
-> Model training (image-only, multimodal, tabular)
-> Evaluation and result analysis
-> Explainability and SmoothGrad visual outputs

Main dependencies used across most scripts:
- cleaned_collected_cases_final_with_FEV1.csv
- Patient_Splits_Seed*/ files
- model checkpoint .pth files produced by training

## 2) Tracked Python Scripts

### Core training

- training_image_only_cv.py
  - 5-fold CV for image-only model.
  - Input: image volumes + fold splits.
  - Output: fold checkpoints, loss/eval outputs.

- training_image_only_cv_fc.py
  - Variant of image-only CV training (FC head variation).
  - Output compatible with image-only evaluation/explainability scripts.

- training_image_only_head.py
  - Image-only model head-focused training variant.

- training_multi_one_hot_cv.py
  - 5-fold CV multimodal training (image + one-hot tabular).
  - Input: imaging + tabular features + fold splits.
  - Output: multimodal fold checkpoints used by multimodal explainability.

- training_tabular_one_hot.py
  - Tabular-only training with one-hot categorical features.
  - Output: tabular model artifacts and metrics.

- training_RF_regressor.py
  - Random Forest tabular regression with one-hot categorical features.
  - Includes grid search, repeated seed/fold evaluation, bootstrap confidence intervals, and feature-importance/SHAP outputs.
  - Uses predefined split files in Patient_Splits_Seed*/ and writes results to a timestamped cv_rf_one_hot_* directory.

### Explainability and interpretation
- explainability_multimodal_permutation.py
  - Permutation-based feature importance for multimodal setup.
  - Consumes trained multimodal checkpoints and tabular/image inputs.

- explainability_tabular_mlp_rf.py
  - Explainability for tabular models (MLP/RF comparison style workflow).

- smoothgrad_image_only_selected_patients.py
  - SmoothGrad saliency generation for selected patients in image-only pipeline.
  - Uses trained image-only checkpoints.

- smoothgrad_multimodal_selected_patients.py
  - SmoothGrad saliency generation for selected patients in multimodal pipeline.
  - Uses trained multimodal checkpoints and tabular inputs.
  - Produces per-patient plots plus fold-by-patient grid overlays.

## 3) Job Script

- job_script_cv_multi.sh
  - Batch scheduler helper to run multimodal CV training (cluster/HPC style).
  - Typically launches training_multi_one_hot_cv.py or related CV training flow.

## 4) Notebooks

- data_exploration.ipynb
  - General data profiling and inspection.

- DICOM_exploration.ipynb
  - DICOM/image structure exploration.

- Subset_data_exploration.ipynb
  - Focused EDA on a subset cohort.

- classification_exploration.ipynb
  - Classification-oriented exploratory analysis.

- patient_split.ipynb
  - Patient split strategy exploration and validation.

- figure_creation.ipynb
  - Figure assembly for reports/presentations.

- regression_baseline_v2.ipynb
  - Baseline regression experimentation and sanity checks.

- mask_exploration_preprocessing.ipynb
  - Mask/scan quality exploration and preprocessing prototyping (crop, resize, depth handling, outlier checks).
  - Produces cleaned cohort files used downstream by training scripts.

## Config/Dependency File

- requirements.txt
  - Python dependencies used by training, notebook, and explainability workflows.

## 5) Selected CV Hyperparameters

Common to all selected model variants:
- Optimizer: Adam
- Loss: MSE

### Image-only models
| Parameter | Linear Head | MLP Head |
|---|---|---|
| Run timestamp | 2026-03-20_09-45-24 | 2026-04-03_21-07-05 |
| Head architecture | 512 -> 1 | 512 -> 128 -> 64 -> 1 |
| Activation | - | ReLU |
| Dropout | - | 0.2 / 0.1 |
| Batch size | 5 | 5 |
| Learning rate | 5e-5 | 3e-5 |
| Weight decay | 0.005 | 2e-5 |
| Early stopping | 15 epochs | 15 epochs |
| LR scheduler | ReduceLROnPlateau (patience=7, factor=0.6) | ReduceLROnPlateau (patience=7, factor=0.6) |

### Multimodal models
| Parameter | MM-1 | MM-2 | MM-3 |
|---|---|---|---|
| Run timestamp | 2026-03-20_21-02-47 | 2026-03-31_08-58-23 | 2026-04-02_09-52-03 |
| Tabular hidden dim | 128 | 256 | 256 |
| Fusion MLP | 576 -> 128 -> 64 -> 1 | 640 -> 256 -> 128 -> 1 | 640 -> 256 -> 128 -> 1 |
| Batch size | 4 | 8 | 4 |
| Learning rate | 5e-3 | 3e-3 | 3e-3 |
| Weight decay | 1e-7 | 1e-7 | 1e-7 |
| Early stopping | 22 epochs | 20 epochs | 20 epochs |
| LR scheduler | ReduceLROnPlateau (patience=6, factor=0.6) | ReduceLROnPlateau (patience=6, factor=0.6) | ReduceLROnPlateau (patience=6, factor=0.6) |

### Output and Figures Location

Outputs and figures for the selected multimodal runs are stored in the MAGIC HUB SharePoint:
- Dokumente/General/Interns/Emma Chanut/OSIC Masterthesis

## 6) How Files Connect

### Multimodal branch
1. patient_split.ipynb (and/or precomputed split CSVs) define folds.
2. training_multi_one_hot_cv.py trains fold models.
3. Checkpoints are consumed by explainability_multimodal_permutation.py and smoothgrad_multimodal_selected_patients.py.
4. Generated plots and summaries are used in figure_creation.ipynb and reporting.

### Image-only branch
1. training_image_only_cv.py or training_image_only_cv_fc.py trains image-only folds.
2. smoothgrad_image_only_selected_patients.py generates saliency visualizations.
3. figure_creation.ipynb consolidates outputs.

### Tabular branch
1. training_tabular_one_hot.py trains tabular model(s).
2. training_RF_regressor.py trains and evaluates a Random Forest baseline on the same tabular feature space.
3. explainability_tabular_mlp_rf.py analyzes feature contribution and behavior.

### Preprocessing and QC branch
1. mask_exploration_preprocessing.ipynb explores mask coverage, outliers, and resizing/cropping behavior.
2. The notebook exports cleaned tables (for example cleaned_collected_cases.csv and derived variants) that feed training pipelines.

<!-- ## 7) Notes

- This guide describes tracked source files and their logical links.
- The repository also tracks many generated artifacts (plots, checkpoints, summaries) that are outputs of the scripts above.
- If needed, this can be expanded into a strict file-by-file catalog generated directly from git ls-files. -->
