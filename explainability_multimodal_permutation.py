#!/usr/bin/env python
"""
Permutation feature importance for the multimodal 3D-CNN + tabular models.

Strategy
--------
For each cross-validation fold:
  1. Load the fold model and fit the fold-specific StandardScaler on training
     patients.
  2. Iterate over test patients: load the 3-D CT volume, pre-process it
     identically to training, run the CNN backbone to produce a 512-d feature
     vector, then discard the image tensor.  All CNN embeddings are cached.
  3. Assemble the full tabular feature matrix (scaled continuous + one-hot).
  4. Compute the baseline test-set MAE.
  5. Image modality permutation: shuffle the cached CNN embeddings across
     patients (n_repeats times); for each shuffle run combined_fc with the
     shuffled embeddings and the original tabular-MLP outputs → record MAE
     increase.
  6. Tabular column permutation: for each column index, shuffle that column
     across patients (n_repeats times); re-run tabular_mlp + combined_fc →
     record MAE increase.
  7. Importance for a feature = mean MAE increase over repeats.

Results are averaged across folds and saved as a CSV and bar chart.

Image pre-processing
--------------------
Identical to the SmoothGrad script:
  remove_zero_slice_hight_width → resize_scan_height_width → resize_scan_depth
"""

import argparse
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from acsconv.converters import ACSConverter
from skimage.transform import resize
from sklearn.preprocessing import OneHotEncoder, StandardScaler


plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})


# ============================================================
# Model definition (must match training exactly)
# ============================================================

class MultiInputModel(nn.Module):
    def __init__(self, cnn_backbone, cnn_output_dim=512, tabular_input_dim=5, hidden_dim=256):
        super().__init__()
        self.cnn_backbone = cnn_backbone
        self.cnn_backbone.fc = nn.Identity()

        self.tabular_mlp = nn.Sequential(
            nn.Linear(tabular_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        combined_input_dim = cnn_output_dim + (hidden_dim // 2)
        self.combined_fc = nn.Sequential(
            nn.Linear(combined_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, image, tabular_features):
        cnn_features = self.cnn_backbone(image)
        if cnn_features.dim() > 2:
            cnn_features = cnn_features.view(cnn_features.size(0), -1)
        if tabular_features.dim() == 1:
            tabular_features = tabular_features.unsqueeze(0)
        tabular_processed = self.tabular_mlp(tabular_features)
        combined = torch.cat([cnn_features, tabular_processed], dim=1)
        return self.combined_fc(combined)


# ============================================================
# Image pre-processing helpers (identical to training / SmoothGrad)
# ============================================================

def remove_zero_slice_hight_width(cases_df, patient_id, method="zscore", hu_clip=(-1000, 400), eps=1e-8):
    mask_nifty_path = cases_df[cases_df["PatientID"] == patient_id]["Mask"].values[0]
    image_nifty_path = cases_df[cases_df["PatientID"] == patient_id]["Image"].values[0]

    mask = nib.load(mask_nifty_path).get_fdata()
    scan = nib.load(image_nifty_path).get_fdata()

    lungs = mask * scan

    non_zero_slices = [i for i in range(mask.shape[0]) if np.sum(mask[i, :, :]) > 0]
    non_zero_widths = [i for i in range(mask.shape[1]) if np.sum(mask[:, i, :]) > 0]
    non_zero_heights = [i for i in range(mask.shape[2]) if np.sum(mask[:, :, i]) > 0]

    lungs_crop = lungs[np.ix_(non_zero_slices, non_zero_widths, non_zero_heights)]
    mask_crop = mask[np.ix_(non_zero_slices, non_zero_widths, non_zero_heights)]

    lung_voxels = lungs_crop[mask_crop > 0]

    if lung_voxels.size == 0:
        return lungs_crop

    if method.lower() == "zscore":
        mean_val = float(np.mean(lung_voxels))
        std_val = float(np.std(lung_voxels))
        out = (lungs_crop - mean_val) / (std_val + eps)
        out[mask_crop == 0] = 0.0
    elif method.lower() == "minmax":
        min_val, max_val = float(np.min(lung_voxels)), float(np.max(lung_voxels))
        out = (lungs_crop - min_val) / (max_val - min_val + eps)
        out[mask_crop == 0] = 0.0
    elif method.lower() == "hu":
        lo, hi = hu_clip
        out = (np.clip(lungs_crop, lo, hi) - lo) / (hi - lo + eps)
        out[mask_crop == 0] = 0.0
    else:
        raise ValueError("Unsupported method. Use 'zscore', 'minmax', or 'hu'.")

    return out


def resize_scan_height_width(scan, new_height=256, new_width=256):
    depth, height, width = scan.shape

    if width == new_width and height == new_height:
        return scan

    if width != height:
        size = max(width, height)
        padded = np.zeros((depth, size, size), dtype=np.float32)
        h_pad = (size - height) // 2
        w_pad = (size - width) // 2
        for i in range(depth):
            padded[i, h_pad:h_pad + height, w_pad:w_pad + width] = scan[i]
        scan = padded
        height, width = size, size

    resized = np.zeros((depth, new_height, new_width), dtype=np.float32)
    for i in range(depth):
        resized[i] = resize(scan[i], (new_height, new_width), mode="constant", preserve_range=True)
    return resized


def resize_scan_depth(scan, new_depth=200):
    if scan.shape[0] == new_depth:
        return scan
    resized = np.zeros((new_depth, scan.shape[1], scan.shape[2]), dtype=np.float32)
    for i in range(scan.shape[1]):
        for j in range(scan.shape[2]):
            resized[:, i, j] = resize(scan[:, i, j], (new_depth,), mode="constant", preserve_range=True)
    return resized


def process_full_scan(cases_df, patient_id, new_height=256, new_width=256, new_depth=200):
    volume = remove_zero_slice_hight_width(cases_df, patient_id)
    volume = resize_scan_height_width(volume, new_height=new_height, new_width=new_width)
    volume = resize_scan_depth(volume, new_depth=new_depth)
    return volume


# ============================================================
# Argument parsing
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Permutation feature importance for multimodal 3D-CNN + tabular models."
    )
    parser.add_argument("--csv-path", default="cleaned_collected_cases_final_with_FEV1.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-folds", type=int, default=5)

    parser.add_argument("--model-dir", default="cv_multi_one_hot_5_cv_scaled_ES2026-03-20_21-02-47")
    parser.add_argument("--model-timestamp", default="2026-03-20_21-02-47")

    parser.add_argument("--hidden-dim", type=int, default=128,
                        help="Hidden dim of the tabular MLP (must match training).")
    parser.add_argument("--cnn-output-dim", type=int, default=512,
                        help="Dimension of the CNN backbone output (512 for ResNet-18).")

    parser.add_argument("--image-height", type=int, default=256)
    parser.add_argument("--image-width", type=int, default=256)
    parser.add_argument("--image-depth", type=int, default=200)

    parser.add_argument("--perm-repeats", type=int, default=30,
                        help="Number of random shuffles per feature per fold.")

    parser.add_argument("--output-dir", default=None,
                        help="Output directory. Defaults to Explainability_Multimodal_<timestamp>.")

    return parser.parse_args()


# ============================================================
# Feature construction
# ============================================================

def build_features(csv_path):
    df = pd.read_csv(csv_path)
    df["PatientID"] = df["PatientID"].astype(str)

    if df["FEV1 Volume L"].isna().sum() > 0:
        med = df["FEV1 Volume L"].median()
        df["FEV1 Volume L"] = df["FEV1 Volume L"].fillna(med)
        print(f"Imputed missing FEV1 values with median={med:.4f}")

    enc_diagnosis = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    enc_sex = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    enc_smoking = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

    diagnosis_encoded = enc_diagnosis.fit_transform(df[["Primary Diagnosis"]].fillna("Unknown"))
    sex_encoded = enc_sex.fit_transform(df[["Sex"]].fillna("Unknown"))
    smoking_encoded = enc_smoking.fit_transform(df[["Smoking History"]].fillna("Unknown"))

    baseline = df["Baseline FVC Volume L"].to_numpy(dtype=np.float32).reshape(-1, 1)
    fev1 = df["FEV1 Volume L"].to_numpy(dtype=np.float32).reshape(-1, 1)
    age = df["Age"].to_numpy(dtype=np.float32).reshape(-1, 1)
    y_all = df["Followup FVC Volume L"].to_numpy(dtype=np.float32)

    # Full feature matrix: [baseline, fev1, age, diagnosis_OH, sex_OH, smoking_OH]
    x_cont = np.concatenate([baseline, fev1, age], axis=1)          # columns 0, 1, 2
    x_all = np.concatenate([x_cont, diagnosis_encoded, sex_encoded, smoking_encoded], axis=1).astype(np.float32)

    feature_names = (
        ["Baseline FVC", "FEV1 Volume L", "Age"]
        + [f"Diagnosis_{cat}" for cat in enc_diagnosis.categories_[0]]
        + [f"Sex_{cat}" for cat in enc_sex.categories_[0]]
        + [f"Smoking_{cat}" for cat in enc_smoking.categories_[0]]
    )

    patient_ids = df["PatientID"].tolist()
    patient_id_to_idx = {pid: i for i, pid in enumerate(patient_ids)}

    return df, x_all, y_all, feature_names, patient_id_to_idx


# ============================================================
# Model loading
# ============================================================

def build_and_load_model(model_path, tabular_input_dim, cnn_output_dim, hidden_dim, device):
    backbone = models.resnet18(weights="IMAGENET1K_V1")
    backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    backbone_3d = ACSConverter(backbone).to(device)

    model = MultiInputModel(
        backbone_3d,
        cnn_output_dim=cnn_output_dim,
        tabular_input_dim=tabular_input_dim,
        hidden_dim=hidden_dim,
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


# ============================================================
# CNN feature extraction
# ============================================================

@torch.no_grad()
def extract_cnn_features(model, cases_df, patient_ids, device, image_height, image_width, image_depth):
    """Load each patient's CT, pre-process and extract CNN backbone features.

    Returns
    -------
    cnn_features : np.ndarray, shape (N, cnn_output_dim)
    valid_mask   : list[bool], True for patients whose image was found / loaded
    """
    features = []
    valid = []

    for i, pid in enumerate(patient_ids):
        if pid not in cases_df["PatientID"].values:
            print(f"  [WARNING] PatientID {pid} not in CSV; skipping.")
            valid.append(False)
            features.append(None)
            continue

        try:
            volume = process_full_scan(
                cases_df, pid,
                new_height=image_height,
                new_width=image_width,
                new_depth=image_depth,
            )
        except Exception as exc:
            print(f"  [WARNING] Could not load image for {pid}: {exc}")
            valid.append(False)
            features.append(None)
            continue

        img_t = torch.from_numpy(volume).float().unsqueeze(0).unsqueeze(0).to(device)
        cnn_feat = model.cnn_backbone(img_t)
        if cnn_feat.dim() > 2:
            cnn_feat = cnn_feat.view(cnn_feat.size(0), -1)
        features.append(cnn_feat.squeeze(0).cpu().numpy())
        valid.append(True)

        if (i + 1) % 10 == 0 or (i + 1) == len(patient_ids):
            print(f"  Extracted CNN features: {i + 1}/{len(patient_ids)}")

        del img_t, cnn_feat

    valid_features = [f for f, v in zip(features, valid) if v]
    return np.stack(valid_features, axis=0), valid


# ============================================================
# Inference helpers
# ============================================================

@torch.no_grad()
def predict_from_parts(model, cnn_feats_np, tab_mat_np, device):
    """Run tabular_mlp + combined_fc given cached CNN features and a full
    tabular feature matrix (already scaled & one-hot encoded).

    Parameters
    ----------
    cnn_feats_np : (N, cnn_output_dim) numpy array
    tab_mat_np   : (N, tabular_input_dim) numpy array

    Returns
    -------
    preds : (N,) numpy array
    """
    cnn_t = torch.tensor(cnn_feats_np, dtype=torch.float32, device=device)
    tab_t = torch.tensor(tab_mat_np, dtype=torch.float32, device=device)

    tab_proc = model.tabular_mlp(tab_t)
    combined = torch.cat([cnn_t, tab_proc], dim=1)
    preds = model.combined_fc(combined).squeeze(1).cpu().numpy()
    return preds


@torch.no_grad()
def predict_from_embeddings(model, cnn_feats_np, tab_proc_np, device):
    """Run only combined_fc given pre-computed CNN and tabular-MLP embeddings.

    Parameters
    ----------
    cnn_feats_np : (N, cnn_output_dim) numpy array
    tab_proc_np  : (N, hidden_dim // 2) numpy array

    Returns
    -------
    preds : (N,) numpy array
    """
    cnn_t = torch.tensor(cnn_feats_np, dtype=torch.float32, device=device)
    tab_t = torch.tensor(tab_proc_np, dtype=torch.float32, device=device)
    combined = torch.cat([cnn_t, tab_t], dim=1)
    preds = model.combined_fc(combined).squeeze(1).cpu().numpy()
    return preds


@torch.no_grad()
def compute_tab_mlp_outputs(model, tab_mat_np, device):
    """Pre-compute tabular MLP outputs for caching."""
    tab_t = torch.tensor(tab_mat_np, dtype=torch.float32, device=device)
    return model.tabular_mlp(tab_t).cpu().numpy()


# ============================================================
# Permutation importance for one fold
# ============================================================

def permutation_importance_fold(
    model,
    cnn_feats,           # (N, cnn_output_dim) numpy
    tab_mat,             # (N, tabular_input_dim) numpy, scaled + OHE
    y_true,              # (N,) numpy
    feature_names,       # list[str], tabular
    n_repeats,
    rng,
    device,
):
    """Return importance scores for all features (tabular + image modality).

    Returns
    -------
    dict mapping feature_name -> list of MAE increases over n_repeats shuffles
    """
    N = len(y_true)

    # Baseline predictions
    baseline_preds = predict_from_parts(model, cnn_feats, tab_mat, device)
    baseline_mae = np.mean(np.abs(baseline_preds - y_true))

    # Pre-compute tabular MLP outputs once (for image permutation efficiency)
    tab_proc = compute_tab_mlp_outputs(model, tab_mat, device)

    importance = {}

    # --- Image modality permutation ---
    image_increases = []
    for _ in range(n_repeats):
        perm_idx = rng.permutation(N)
        shuffled_cnn = cnn_feats[perm_idx]
        preds = predict_from_embeddings(model, shuffled_cnn, tab_proc, device)
        image_increases.append(np.mean(np.abs(preds - y_true)) - baseline_mae)
    importance["CT Image (3D-CNN)"] = image_increases

    # --- Per-column tabular permutation ---
    for col_idx, feat_name in enumerate(feature_names):
        increases = []
        for _ in range(n_repeats):
            tab_perm = tab_mat.copy()
            tab_perm[:, col_idx] = rng.permutation(tab_mat[:, col_idx])
            preds = predict_from_parts(model, cnn_feats, tab_perm, device)
            increases.append(np.mean(np.abs(preds - y_true)) - baseline_mae)
        importance[feat_name] = increases

    return baseline_mae, importance


# ============================================================
# Plotting helper
# ============================================================

def plot_importance_bar(mean_vals, std_vals, feature_names, out_path, title, top_n=25, color="#DD8452"):
    idx_sorted = np.argsort(mean_vals)[::-1][:top_n]
    top_names = [feature_names[i] for i in idx_sorted][::-1]
    top_means = mean_vals[idx_sorted][::-1]
    top_stds = std_vals[idx_sorted][::-1]

    plt.figure(figsize=(12, max(6, top_n * 0.38)))
    plt.barh(
        top_names, top_means,
        xerr=top_stds,
        color=color,
        edgecolor="black",
        alpha=0.85,
        capsize=3,
    )
    plt.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    plt.title(title)
    plt.xlabel("Mean MAE increase (L)")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()
    print(f"Saved: {out_path}")


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    if args.output_dir is None:
        run_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        args.output_dir = f"Explainability_Multimodal_Permutation_{run_ts}"
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Output directory: {args.output_dir}")

    # Build full dataset feature matrix
    cases_df, x_all, y_all, feature_names, patient_id_to_idx = build_features(args.csv_path)
    tabular_input_dim = x_all.shape[1]
    split_dir = f"Patient_Splits_Seed{args.seed}"

    # Load test patients (same test set across all folds)
    test_csv = os.path.join(split_dir, "test_patients.csv")
    test_ids_all = pd.read_csv(test_csv)["PatientID"].astype(str).tolist()
    test_ids = [pid for pid in test_ids_all if pid in patient_id_to_idx]
    test_idx = np.array([patient_id_to_idx[pid] for pid in test_ids], dtype=np.int64)
    y_test = y_all[test_idx]

    print(f"Test patients: {len(test_ids)}")

    # Accumulate per-fold results
    all_fold_importances = {feat: [] for feat in ["CT Image (3D-CNN)"] + feature_names}
    all_fold_baselines = []
    fold_baseline_rows = []

    rng = np.random.default_rng(args.seed)

    for fold_num in range(1, args.num_folds + 1):
        print(f"\n{'=' * 70}")
        print(f"Fold {fold_num} / {args.num_folds}")
        print(f"{'=' * 70}")

        model_path = os.path.join(
            args.model_dir,
            f"best_model_seed{args.seed}_fold{fold_num}_lr_multi_all_{args.model_timestamp}.pth",
        )
        if not os.path.exists(model_path):
            print(f"  Model not found, skipping: {model_path}")
            continue

        # Fit fold-specific scaler on training patients
        train_csv = os.path.join(split_dir, f"train_fold_{fold_num - 1}.csv")
        if not os.path.exists(train_csv):
            raise FileNotFoundError(f"Train fold file not found: {train_csv}")

        train_ids = pd.read_csv(train_csv)["PatientID"].astype(str).tolist()
        train_ids = [pid for pid in train_ids if pid in patient_id_to_idx]
        train_idx = np.array([patient_id_to_idx[pid] for pid in train_ids], dtype=np.int64)

        # Scale only the three continuous columns (indices 0, 1, 2)
        fold_scaler = StandardScaler()
        fold_scaler.fit(x_all[train_idx, :3])

        # Assemble scaled test tabular matrix
        x_test_raw = x_all[test_idx]
        x_test_scaled = x_test_raw.copy()
        x_test_scaled[:, :3] = fold_scaler.transform(x_test_raw[:, :3])

        # Build and load model
        print(f"  Loading model: {model_path}")
        model = build_and_load_model(
            model_path,
            tabular_input_dim=tabular_input_dim,
            cnn_output_dim=args.cnn_output_dim,
            hidden_dim=args.hidden_dim,
            device=device,
        )

        # Extract CNN features for all test patients (images loaded once & discarded)
        print(f"  Extracting CNN features for {len(test_ids)} test patients...")
        cnn_feats, valid_mask = extract_cnn_features(
            model, cases_df, test_ids, device,
            args.image_height, args.image_width, args.image_depth,
        )

        # Filter out patients whose image couldn't be loaded
        valid_idx = [i for i, v in enumerate(valid_mask) if v]
        y_fold = y_test[valid_idx]
        tab_fold = x_test_scaled[valid_idx]

        n_valid = len(valid_idx)
        if n_valid == 0:
            print("  No valid test patients; skipping fold.")
            continue
        print(f"  Valid patients for permutation: {n_valid}")

        # Run permutation importance for this fold
        print(f"  Running permutation importance ({args.perm_repeats} repeats per feature)...")
        baseline_mae, fold_importance = permutation_importance_fold(
            model=model,
            cnn_feats=cnn_feats,
            tab_mat=tab_fold,
            y_true=y_fold,
            feature_names=feature_names,
            n_repeats=args.perm_repeats,
            rng=rng,
            device=device,
        )
        print(f"  Baseline MAE (fold {fold_num}): {baseline_mae:.4f} L")
        all_fold_baselines.append(baseline_mae)
        fold_baseline_rows.append({"Fold": fold_num, "Baseline_MAE": baseline_mae})

        for feat, increases in fold_importance.items():
            all_fold_importances[feat].append(np.mean(increases))

        # Per-fold CSV
        fold_rows = []
        for feat, increases in fold_importance.items():
            fold_rows.append({
                "Feature": feat,
                "Mean_MAE_Increase": np.mean(increases),
                "Std_MAE_Increase": np.std(increases),
            })
        fold_df = pd.DataFrame(fold_rows).sort_values("Mean_MAE_Increase", ascending=False)
        fold_csv = os.path.join(args.output_dir, f"fold{fold_num}_permutation_importance.csv")
        fold_df.to_csv(fold_csv, index=False)
        print(f"  Saved fold CSV: {fold_csv}")

        # Cleanup model from GPU memory before next fold
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ============================================================
    # Aggregate across folds
    # ============================================================
    all_features = ["CT Image (3D-CNN)"] + feature_names
    mean_vals = np.array([np.mean(all_fold_importances[f]) if all_fold_importances[f] else 0.0
                          for f in all_features])
    std_vals  = np.array([np.std(all_fold_importances[f])  if all_fold_importances[f] else 0.0
                          for f in all_features])

    summary_df = pd.DataFrame({
        "Feature": all_features,
        "Mean_MAE_Increase": mean_vals,
        "Std_MAE_Increase": std_vals,
    }).sort_values("Mean_MAE_Increase", ascending=False)

    summary_csv = os.path.join(args.output_dir, "multimodal_permutation_importance.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSaved summary CSV: {summary_csv}")

    # Baseline CSV
    baseline_df = pd.DataFrame(fold_baseline_rows)
    baseline_df.loc["mean"] = ["Mean", np.mean(all_fold_baselines)]
    baseline_df.to_csv(os.path.join(args.output_dir, "fold_baseline_mae.csv"), index=False)

    # Bar chart (all features)
    plot_importance_bar(
        mean_vals, std_vals, all_features,
        out_path=os.path.join(args.output_dir, "multimodal_permutation_importance_all.png"),
        title="Multimodal Model — Permutation Feature Importance\n(mean MAE increase ± std across folds)",
        top_n=len(all_features),
        color="#DD8452",
    )

    # Bar chart (top 20)
    plot_importance_bar(
        mean_vals, std_vals, all_features,
        out_path=os.path.join(args.output_dir, "multimodal_permutation_importance_top20.png"),
        title="Multimodal Model — Top-20 Permutation Feature Importance\n(mean MAE increase ± std across folds)",
        top_n=20,
        color="#DD8452",
    )

    print("\n" + "=" * 70)
    print("Top-10 most important features:")
    print(summary_df.head(10).to_string(index=False))
    print("=" * 70)
    print(f"All outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
