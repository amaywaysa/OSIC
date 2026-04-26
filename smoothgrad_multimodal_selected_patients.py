import os
import numpy as np
import pandas as pd
import nibabel as nib

import torch
import torch.nn as nn
import torchvision.models as models
from acsconv.converters import ACSConverter
from skimage.transform import resize
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt


plt.rcParams.update({
    "font.family": "serif",      # serif to match LaTeX article style
    "font.size": 14,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "legend.fontsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
})

# ==============================
# CONFIG
# ==============================
# Run saliency + SmoothGrad for all selected patients across all folds
model_dir = 'cv_multi_one_hot_5_cv_scaled_ES2026-04-02_09-52-03'
model_timestamp = '2026-04-02_09-52-03'
#/cv_multi_one_hot_5_cv_scaled_ES2026-03-20_21-02-47/best_model_seed42_fold1_lr_multi_all_2026-03-20_21-02-47.pth
#model_dir = 'cv_multi_one_hot_5_cv_scaled_ES2026-03-20_21-02-47'
# model_dir = 'cv_multi_one_hot_5_cv_scaled_ES2026-03-31_08-58-23'
# model_timestamp = '2026-03-31_08-58-23'
#model_timestamp = '2026-03-20_21-02-47'
n_folds = 5
csv_path = 'cleaned_collected_cases_final_with_FEV1.csv'
seed_to_use = 42
split_dir = f'Patient_Splits_Seed{seed_to_use}'
targets_shape = (256, 256, 200)
hidden_dim = 256
cnn_output_dim = 512

smoothgrad_n_samples = 24
smoothgrad_noise_std_ratio = 0.10
overlay_threshold = 5e-5 

# selected_patients = {
#     '247524': 'strong_error',
#     '374153': 'strong_error',
#     '1001405': 'good_prediction',
#     '1000788': 'good_prediction',
# }

# other patients 1001148, 687751 good
# 1001201, 469929 bad
selected_patients = {
    '1001201': 'strong_error',
    '469929': 'strong_error',
    '1001148': 'good_prediction',
    '687751': 'good_prediction',
}

# ==============================
# Helpers
# ==============================
def normalize_slice(x):
    x_min, x_max = np.min(x), np.max(x)
    if x_max - x_min < 1e-8:
        return np.zeros_like(x)
    return (x - x_min) / (x_max - x_min)


def remove_zero_slice_hight_width(cases_df, patient_id, method='zscore', hu_clip=(-1000, 400), eps=1e-8):
    mask_nifty_path = cases_df[cases_df['PatientID'] == patient_id]['Mask'].values[0]
    image_nifty_path = cases_df[cases_df['PatientID'] == patient_id]['Image'].values[0]

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

    if method.lower() == 'zscore':
        mean_val = float(np.mean(lung_voxels))
        std_val = float(np.std(lung_voxels))
        only_lungs_normalized = (lungs_crop - mean_val) / (std_val + eps)
        only_lungs_normalized[mask_crop == 0] = 0.0
    elif method.lower() == 'minmax':
        min_val = float(np.min(lung_voxels))
        max_val = float(np.max(lung_voxels))
        range_val = max_val - min_val
        only_lungs_normalized = (lungs_crop - min_val) / (range_val + eps)
        only_lungs_normalized[mask_crop == 0] = 0.0
    elif method.lower() == 'hu':
        lo, hi = hu_clip
        clipped = np.clip(lungs_crop, lo, hi)
        only_lungs_normalized = (clipped - lo) / (hi - lo + eps)
        only_lungs_normalized[mask_crop == 0] = 0.0
    else:
        raise ValueError("Unsupported method. Use 'zscore', 'minmax', or 'hu'.")

    return only_lungs_normalized


def resize_scan_height_width(scan, new_height=256, new_width=256):
    depth = scan.shape[0]
    width = scan.shape[2]
    height = scan.shape[1]

    if width == new_width and height == new_height:
        return scan

    if width != height:
        size = max(width, height)
        padded_scan = np.zeros((depth, size, size), dtype=np.float32)

        height_pad = (size - height) // 2
        width_pad = (size - width) // 2

        for i in range(depth):
            padded_scan[i, height_pad:height_pad + height, width_pad:width_pad + width] = scan[i]

        resized_scan = np.zeros((depth, new_height, new_width), dtype=np.float32)
        for i in range(depth):
            resized_scan[i] = resize(padded_scan[i], (new_height, new_width), mode='constant', preserve_range=True)
        return resized_scan

    resized_scan = np.zeros((depth, new_height, new_width), dtype=np.float32)
    for i in range(depth):
        resized_scan[i] = resize(scan[i], (new_height, new_width), mode='constant', preserve_range=True)
    return resized_scan


def resize_scan_depth(scan, new_depth=200):
    current_depth = scan.shape[0]
    if current_depth == new_depth:
        return scan

    resized_scan = np.zeros((new_depth, scan.shape[1], scan.shape[2]), dtype=np.float32)
    for i in range(scan.shape[1]):
        for j in range(scan.shape[2]):
            resized_scan[:, i, j] = resize(scan[:, i, j], (new_depth,), mode='constant', preserve_range=True)
    return resized_scan


def process_full_scan(cases_df, patient_id, new_height=256, new_width=256, new_depth=200):
    lung_scan = remove_zero_slice_hight_width(cases_df, patient_id)
    resized_hw_scan = resize_scan_height_width(lung_scan, new_height=new_height, new_width=new_width)
    resized_full_scan = resize_scan_depth(resized_hw_scan, new_depth=new_depth)
    return resized_full_scan


class MultiInputModel(nn.Module):
    def __init__(self, cnn_backbone, cnn_output_dim=512, tabular_input_dim=5, hidden_dim=256):
        super(MultiInputModel, self).__init__()

        self.cnn_backbone = cnn_backbone
        self.cnn_backbone.fc = nn.Identity()

        self.tabular_mlp = nn.Sequential(
            nn.Linear(tabular_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        combined_input_dim = cnn_output_dim + (hidden_dim // 2)
        self.combined_fc = nn.Sequential(
            nn.Linear(combined_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
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


def compute_smoothgrad_3d(model, image_tensor, tabular_tensor, n_samples=24, noise_std_ratio=0.10):
    """Compute SmoothGrad saliency for a 3D input volume (multimodal model)."""
    model.eval()
    image_base = image_tensor.detach()
    signal_std = torch.std(image_base).item()
    noise_std = max(signal_std * noise_std_ratio, 1e-8)

    grad_accumulator = torch.zeros_like(image_base)

    for _ in range(n_samples):
        noise = torch.randn_like(image_base) * noise_std
        noisy_image = (image_base + noise).detach().requires_grad_(True)

        model.zero_grad(set_to_none=True)
        pred = model(noisy_image, tabular_tensor)
        pred.backward(torch.ones_like(pred))

        grad_accumulator += noisy_image.grad.detach().abs()

        del noise, noisy_image, pred

    smoothgrad = grad_accumulator / n_samples
    smoothgrad_np = smoothgrad.squeeze(0).squeeze(0).cpu().numpy()
    smoothgrad_np = normalize_slice(smoothgrad_np)
    return smoothgrad_np


def compute_gradient_saliency_3d(model, image_tensor, tabular_tensor):
    """Compute vanilla gradient saliency for a 3D input volume (multimodal model)."""
    model.eval()
    image_var = image_tensor.clone().detach().requires_grad_(True)

    model.zero_grad(set_to_none=True)
    pred = model(image_var, tabular_tensor)
    pred.backward(torch.ones_like(pred))

    grad_saliency = image_var.grad.detach().abs().squeeze(0).squeeze(0).cpu().numpy()
    grad_saliency = normalize_slice(grad_saliency)

    del image_var, pred

    return grad_saliency


# ==============================
# Main
# ==============================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    cases_df = pd.read_csv(csv_path)
    cases_df['PatientID'] = cases_df['PatientID'].astype(str)

    fev1_missing_count = cases_df['FEV1 Volume L'].isna().sum()
    if fev1_missing_count > 0:
        fev1_median = cases_df['FEV1 Volume L'].median()
        cases_df['FEV1 Volume L'] = cases_df['FEV1 Volume L'].fillna(fev1_median)
        print(f'Imputed {fev1_missing_count} missing FEV1 values using median: {fev1_median:.4f}')

    patient_id_to_idx = {pid: idx for idx, pid in enumerate(cases_df['PatientID'].tolist())}

    # Encode categoricals exactly as training/XAI (fit once on full dataset)
    enc_diagnosis = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    enc_sex = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    enc_smoking = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    diagnosis_encoded = enc_diagnosis.fit_transform(cases_df[['Primary Diagnosis']].fillna('Unknown'))
    sex_encoded = enc_sex.fit_transform(cases_df[['Sex']].fillna('Unknown'))
    smoking_encoded = enc_smoking.fit_transform(cases_df[['Smoking History']].fillna('Unknown'))

    total_categorical_dims = (
        len(enc_diagnosis.categories_[0]) +
        len(enc_sex.categories_[0]) +
        len(enc_smoking.categories_[0])
    )
    tabular_input_dim = 3 + total_categorical_dims

    all_summary_rows = []

    for fold_num in range(1, n_folds + 1):
        print(f'\n===== Fold {fold_num} / {n_folds} =====')

        model_path = os.path.join(
            model_dir,
            f'best_model_seed{seed_to_use}_fold{fold_num}_lr_multi_all_{model_timestamp}.pth'
        )
        if not os.path.exists(model_path):
            print(f'  Model not found, skipping: {model_path}')
            continue

        # Fit fold scaler on this fold's training patients
        train_fold_path = os.path.join(split_dir, f'train_fold_{fold_num - 1}.csv')
        if not os.path.exists(train_fold_path):
            raise FileNotFoundError(f'Train fold file not found: {train_fold_path}')

        train_patient_ids = pd.read_csv(train_fold_path)['PatientID'].astype(str).tolist()
        train_patient_ids = [pid for pid in train_patient_ids if pid in patient_id_to_idx]

        train_baseline = [cases_df.iloc[patient_id_to_idx[pid]]['Baseline FVC Volume L'] for pid in train_patient_ids]
        train_fev1 = [cases_df.iloc[patient_id_to_idx[pid]]['FEV1 Volume L'] for pid in train_patient_ids]
        train_ages = [cases_df.iloc[patient_id_to_idx[pid]]['Age'] for pid in train_patient_ids]

        train_cont = np.column_stack([train_baseline, train_fev1, train_ages]).astype(np.float32)
        fold_scaler = StandardScaler()
        fold_scaler.fit(train_cont)

        # Build and load model for this fold
        backbone = models.resnet18(weights='IMAGENET1K_V1')
        backbone.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        backbone_3d = ACSConverter(backbone).to(device)

        model = MultiInputModel(
            backbone_3d,
            cnn_output_dim=cnn_output_dim,
            tabular_input_dim=tabular_input_dim,
            hidden_dim=hidden_dim
        ).to(device)

        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f'  Loaded model: {model_path}')

        output_dir = os.path.join(model_dir, f'{model_timestamp}_pred', f'fold{fold_num}_xai_selected_patients')
        os.makedirs(output_dir, exist_ok=True)

        summary_rows = []

        for pid, patient_group in selected_patients.items():
            if pid not in patient_id_to_idx:
                print(f'  Skipping missing patient ID: {pid}')
                continue

            idx = patient_id_to_idx[pid]

            # Image input
            volume = process_full_scan(
                cases_df,
                pid,
                new_height=targets_shape[0],
                new_width=targets_shape[1],
                new_depth=targets_shape[2]
            )
            image_tensor = torch.from_numpy(volume).float().unsqueeze(0).unsqueeze(0).to(device)

            # Tabular input (continuous scaled by fold scaler + one-hot as-is)
            baseline_val = float(cases_df.iloc[idx]['Baseline FVC Volume L'])
            fev1_val = float(cases_df.iloc[idx]['FEV1 Volume L'])
            age_val = float(cases_df.iloc[idx]['Age'])

            cont_arr = np.array([[baseline_val, fev1_val, age_val]], dtype=np.float32)
            cont_scaled = fold_scaler.transform(cont_arr)[0]

            diagnosis_vec = diagnosis_encoded[idx]
            sex_vec = sex_encoded[idx]
            smoking_vec = smoking_encoded[idx]

            tab_vec = np.concatenate([cont_scaled, diagnosis_vec, sex_vec, smoking_vec]).astype(np.float32)
            tabular_tensor = torch.tensor(tab_vec, dtype=torch.float32, device=device).unsqueeze(0)

            # Prediction, saliency, and SmoothGrad
            with torch.no_grad():
                pred_val = float(model(image_tensor, tabular_tensor).squeeze().cpu().item())

            grad_saliency_3d = compute_gradient_saliency_3d(model, image_tensor, tabular_tensor)

            smoothgrad_3d = compute_smoothgrad_3d(
                model,
                image_tensor,
                tabular_tensor,
                n_samples=smoothgrad_n_samples,
                noise_std_ratio=smoothgrad_noise_std_ratio
            )

            image_np = image_tensor.detach().squeeze(0).squeeze(0).cpu().numpy()
            true_val = float(cases_df.iloc[idx]['Followup FVC Volume L'])
            abs_err = abs(true_val - pred_val)

            d, h, w = grad_saliency_3d.shape
            d_mid, h_mid, w_mid = d // 2, h // 2, 2 * w // 3

            img_axial = normalize_slice(image_np[d_mid, :, :])
            img_coronal = normalize_slice(image_np[:, h_mid, :])
            img_sagittal = normalize_slice(image_np[:, :, w_mid])

            sal_axial = normalize_slice(grad_saliency_3d[d_mid, :, :])
            sal_coronal = normalize_slice(grad_saliency_3d[:, h_mid, :])
            sal_sagittal = normalize_slice(grad_saliency_3d[:, :, w_mid])

            diagnosis = str(cases_df.iloc[idx]['Primary Diagnosis'])
            suptitle_base = (
                f'Fold {fold_num} | {patient_group} - Patient {pid}\n'
                f'Diagnosis: {diagnosis} | Baseline FVC: {baseline_val:.3f} L\n'
                f'True Follow-up FVC: {true_val:.3f} L | Pred FVC: {pred_val:.3f} L | Abs Error: {abs_err:.3f} L'
            )

            # ---- Gradient saliency plot ----
            plt.figure(figsize=(12, 8))

            plt.subplot(2, 3, 1)
            plt.imshow(img_axial, cmap='gray')
            plt.title('Image Axial')
            plt.axis('off')

            plt.subplot(2, 3, 2)
            plt.imshow(img_coronal, cmap='gray')
            plt.title('Image Coronal')
            plt.axis('off')

            plt.subplot(2, 3, 3)
            plt.imshow(img_sagittal, cmap='gray')
            plt.title('Image Sagittal')
            plt.axis('off')

            plt.subplot(2, 3, 4)
            plt.imshow(sal_axial, cmap='hot')
            plt.title('Saliency Axial')
            plt.axis('off')

            plt.subplot(2, 3, 5)
            plt.imshow(sal_coronal, cmap='hot')
            plt.title('Saliency Coronal')
            plt.axis('off')

            plt.subplot(2, 3, 6)
            plt.imshow(sal_sagittal, cmap='hot')
            plt.title('Saliency Sagittal')
            plt.axis('off')

            plt.suptitle(f'Image Importance (Gradient Saliency)\n{suptitle_base}')
            plt.tight_layout()

            saliency_path = os.path.join(output_dir, f'saliency_{patient_group}_patient_{pid}.png')
            plt.savefig(saliency_path)
            plt.close()
            print(f'  Saved saliency plot: {saliency_path}')

            # ---- SmoothGrad overlay plot ----
            sg_axial = np.ma.masked_where(smoothgrad_3d[d_mid, :, :] <= overlay_threshold, smoothgrad_3d[d_mid, :, :])
            sg_coronal = np.ma.masked_where(smoothgrad_3d[:, h_mid, :] <= overlay_threshold, smoothgrad_3d[:, h_mid, :])
            sg_sagittal = np.ma.masked_where(smoothgrad_3d[:, :, w_mid] <= overlay_threshold, smoothgrad_3d[:, :, w_mid])

            cmap = plt.colormaps['turbo'].copy()
            cmap.set_bad(alpha=0)

            plt.figure(figsize=(14, 8))

            plt.subplot(2, 3, 1)
            plt.imshow(img_axial, cmap='gray')
            plt.title('Image Axial')
            plt.axis('off')

            plt.subplot(2, 3, 2)
            plt.imshow(img_coronal, cmap='gray')
            plt.title('Image Coronal')
            plt.axis('off')

            plt.subplot(2, 3, 3)
            plt.imshow(img_sagittal, cmap='gray')
            plt.title('Image Sagittal')
            plt.axis('off')

            plt.subplot(2, 3, 4)
            plt.imshow(img_axial, cmap='gray')
            plt.imshow(sg_axial, cmap=cmap, alpha=0.55)
            plt.title('SmoothGrad Axial')
            plt.axis('off')

            plt.subplot(2, 3, 5)
            plt.imshow(img_coronal, cmap='gray')
            plt.imshow(sg_coronal, cmap=cmap, alpha=0.55)
            plt.title('SmoothGrad Coronal')
            plt.axis('off')

            plt.subplot(2, 3, 6)
            plt.imshow(img_sagittal, cmap='gray')
            plt.imshow(sg_sagittal, cmap=cmap, alpha=0.55)
            plt.title('SmoothGrad Sagittal')
            plt.axis('off')

            plt.suptitle(f'SmoothGrad\n{suptitle_base}')
            plt.tight_layout()

            out_path = os.path.join(output_dir, f'smoothgrad_{patient_group}_patient_{pid}.png')
            plt.savefig(out_path)
            plt.close()
            print(f'  Saved SmoothGrad plot: {out_path}')

            summary_rows.append({
                'Fold': fold_num,
                'PatientID': pid,
                'Patient_Group': patient_group,
                'Baseline_FVC': baseline_val,
                'True_Followup_FVC': true_val,
                'Predicted_FVC': pred_val,
                'Abs_Error': abs_err,
                'Diagnosis': diagnosis,
                'Model_Path': model_path,
                'Train_Fold_Path': train_fold_path,
            })

        # Per-fold summary CSV
        fold_summary_df = pd.DataFrame(summary_rows)
        fold_summary_csv = os.path.join(output_dir, f'fold{fold_num}_xai_summary.csv')
        fold_summary_df.to_csv(fold_summary_csv, index=False)
        print(f'  Saved fold summary CSV: {fold_summary_csv}')

        all_summary_rows.extend(summary_rows)

    # Combined summary across all folds
    combined_summary_df = pd.DataFrame(all_summary_rows)
    combined_summary_csv = os.path.join(model_dir, f'{model_timestamp}_pred', 'all_folds_xai_summary.csv')
    combined_summary_df.to_csv(combined_summary_csv, index=False)
    print(f'\nSaved combined summary CSV: {combined_summary_csv}')


if __name__ == '__main__':
    main()
