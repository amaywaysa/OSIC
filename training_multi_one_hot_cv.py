import numpy as np
import nibabel as nib
import pandas as pd
import os

import torchvision.models as models  
from acsconv.converters import ACSConverter  

import torch  
import torch.nn as nn  
import torch.optim as optim  
from skimage.transform import resize
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import shap


# 1. Load data paths and targets from CSV
cases_df = pd.read_csv('cleaned_collected_cases_final_with_FEV1.csv')
cases_df['PatientID'] = cases_df['PatientID'].astype(str)

fev1_missing_count = cases_df['FEV1 Volume L'].isna().sum()
if fev1_missing_count > 0:
    fev1_median = cases_df['FEV1 Volume L'].median()
    cases_df['FEV1 Volume L'] = cases_df['FEV1 Volume L'].fillna(fev1_median)
    print(f"Imputed {fev1_missing_count} missing FEV1 values using median: {fev1_median:.4f}")
else:
    print("No missing FEV1 values found.")

# Prepare categorical features with OneHotEncoder
enc_diagnosis = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
enc_sex = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
enc_smoking = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# Fit and transform categorical features
diagnosis_encoded = enc_diagnosis.fit_transform(cases_df[['Primary Diagnosis']].fillna('Unknown'))
sex_encoded = enc_sex.fit_transform(cases_df[['Sex']].fillna('Unknown'))
smoking_encoded = enc_smoking.fit_transform(cases_df[['Smoking History']].fillna('Unknown'))

# Store encoded features in dataframe
for i, cat in enumerate(enc_diagnosis.categories_[0]):
    cases_df[f'Primary Diagnosis_{cat}'] = diagnosis_encoded[:, i]
for i, cat in enumerate(enc_sex.categories_[0]):
    cases_df[f'Sex_{cat}'] = sex_encoded[:, i]
for i, cat in enumerate(enc_smoking.categories_[0]):
    cases_df[f'Smoking History_{cat}'] = smoking_encoded[:, i]

# Store encoders for later use
print(f"Primary Diagnosis classes: {list(enc_diagnosis.categories_[0])}")
print(f"Sex classes: {list(enc_sex.categories_[0])}")
print(f"Smoking History classes: {list(enc_smoking.categories_[0])}")

# Calculate total one-hot dimensions
n_diagnosis_cats = len(enc_diagnosis.categories_[0])
n_sex_cats = len(enc_sex.categories_[0])
n_smoking_cats = len(enc_smoking.categories_[0])
total_categorical_dims = n_diagnosis_cats + n_sex_cats + n_smoking_cats
print(f"Total categorical dimensions: {total_categorical_dims} (diagnosis: {n_diagnosis_cats}, sex: {n_sex_cats}, smoking: {n_smoking_cats})")

timestamp = pd.to_datetime("today").strftime("%Y-%m-%d_%H-%M-%S")
print(f"Data loaded. Total cases: {len(cases_df)} at {timestamp}.")

# Scan Preprocessing Function

def remove_zero_slice_hight_width(
    patient_id,
    method: str = "zscore",            # 'zscore' | 'minmax' | 'hu'
    hu_clip: tuple = (-1000, 400),     # used when method='hu'
    eps: float = 1e-8                  # small constant to avoid divide-by-zero
):
    """
    Load mask & scan for a patient, crop to non-zero mask extents, and normalize intensities
    within the lung mask region. Preserves zeros outside the mask.

    Parameters
    ----------
    patient_id : Any
        Key used to look up paths in cases_df.
    method : str
        Normalization method:
          - 'zscore': (x - mean_lung) / std_lung
          - 'minmax': (x - min_lung) / (max_lung - min_lung)
          - 'hu'    : clip to hu_clip (e.g., [-1000, 400]) then scale to [0, 1]
    hu_clip : tuple(int, int)
        Lower and upper bounds for HU clipping, only used when method='hu'.
    eps : float
        Small value added to denominators to avoid instability.

    Returns
    -------
    only_lungs_normalized : np.ndarray
        Cropped 3D volume containing the lung region, normalized according to `method`.
        Voxels outside the mask are set to 0.
    """

    # Resolve paths from the dataframe (assumes cases_df exists in the scope)
    mask_nifty_path = cases_df[cases_df['PatientID'] == patient_id]['Mask'].values[0]
    image_nifty_path = cases_df[cases_df['PatientID'] == patient_id]['Image'].values[0]

    # Load volumes
    mask = nib.load(mask_nifty_path).get_fdata()
    scan = nib.load(image_nifty_path).get_fdata()

    # Apply mask (preserve scan values in lungs, zeros elsewhere)
    lungs = mask * scan

    # Compute non-zero extents along each axis using the mask
    non_zero_slices = [i for i in range(mask.shape[0]) if np.sum(mask[i, :, :]) > 0]
    non_zero_widths = [i for i in range(mask.shape[1]) if np.sum(mask[:, i, :]) > 0]
    non_zero_heights = [i for i in range(mask.shape[2]) if np.sum(mask[:, :, i]) > 0]

    # Crop to the minimal bounding box with non-zero mask along each axis
    lungs_crop = lungs[np.ix_(non_zero_slices, non_zero_widths, non_zero_heights)]
    mask_crop  = mask[np.ix_(non_zero_slices, non_zero_widths, non_zero_heights)]

    # Collect lung voxels (exclude zeros outside mask)
    lung_voxels = lungs_crop[mask_crop > 0]

    # If the mask is empty after cropping (edge case), return the crop as-is
    if lung_voxels.size == 0:
        # No lung voxels found; just return the cropped lungs (all zeros)
        return lungs_crop

    # Normalize based on selected method
    if method.lower() == "zscore":
        mean_val = float(np.mean(lung_voxels))
        std_val  = float(np.std(lung_voxels))
        only_lungs_normalized = (lungs_crop - mean_val) / (std_val + eps)
        # Preserve zeros outside mask
        only_lungs_normalized[mask_crop == 0] = 0.0

    elif method.lower() == "minmax":
        min_val = float(np.min(lung_voxels))
        max_val = float(np.max(lung_voxels))
        range_val = (max_val - min_val)
        only_lungs_normalized = (lungs_crop - min_val) / (range_val + eps)
        only_lungs_normalized[mask_crop == 0] = 0.0

    elif method.lower() == "hu":
        lo, hi = hu_clip
        # Clip within the crop, scale to [0,1], then zero-out non-mask voxels
        clipped = np.clip(lungs_crop, lo, hi)
        only_lungs_normalized = (clipped - lo) / (hi - lo + eps)
        only_lungs_normalized[mask_crop == 0] = 0.0

    else:
        raise ValueError("Unsupported method. Use 'zscore', 'minmax', or 'hu'.")

    return only_lungs_normalized

# Resizing scan to fixed size function

def resize_scan_height_width(scan, new_height=256, new_width=256):
    depth = scan.shape[0]
    width = scan.shape[2]
    height = scan.shape[1]
    if width == new_width and height == new_height:
        return scan
    if width != height:
        # zero pad to make square - pad equally from both sides
        size = max(width, height)
        padded_scan = np.zeros((depth, size, size))
        
        # Calculate padding for height and width
        height_pad = (size - height) // 2
        width_pad = (size - width) // 2
        
        for i in range(depth):
            padded_scan[i, height_pad:height_pad+height, width_pad:width_pad+width] = scan[i]
        resized_scan = np.zeros((depth, new_height, new_width))
        for i in range(depth):
            resized_scan[i] = resize(padded_scan[i], (new_height, new_width), mode='constant', preserve_range=True)
        return resized_scan
    else:
        resized_scan = np.zeros((depth, new_height, new_width))
        for i in range(depth):
            resized_scan[i] = resize(scan[i], (new_height, new_width), mode='constant', preserve_range=True)
        return resized_scan
    
def resize_scan_depth(scan, new_depth=400):
    current_depth = scan.shape[0]
    if current_depth == new_depth:
        return scan
    else:
        resized_scan = np.zeros((new_depth, scan.shape[1], scan.shape[2]))
        for i in range(scan.shape[1]):
            for j in range(scan.shape[2]):
                resized_scan[:, i, j] = resize(scan[:, i, j], (new_depth,), mode='constant', preserve_range=True)
        return resized_scan
    
def process_full_scan(patient_id, new_height=256, new_width=256, new_depth=200):
    lung_scan = remove_zero_slice_hight_width(patient_id)
    resized_hw_scan = resize_scan_height_width(lung_scan, new_height=new_height, new_width=new_width)
    resized_full_scan = resize_scan_depth(resized_hw_scan, new_depth=new_depth)
    return resized_full_scan

class FixedSizeDataset(torch.utils.data.Dataset):
    def __init__(self, patient_ids, targets, baseline_fvcs, fev1s, ages, diagnosis_encoded, sex_encoded, smoking_encoded, target_shape=(256, 256, 200)):
        self.patient_ids = patient_ids
        self.targets = targets
        self.baseline_fvcs = baseline_fvcs
        self.fev1s = fev1s
        self.ages = ages
        self.diagnosis_encoded = diagnosis_encoded  # Now a 2D array (n_samples, n_diagnosis_categories)
        self.sex_encoded = sex_encoded  # Now a 2D array (n_samples, n_sex_categories)
        self.smoking_encoded = smoking_encoded  # Now a 2D array (n_samples, n_smoking_categories)
        self.target_shape = target_shape
    
    def __getitem__(self, idx):
        # Use process_full_scan instead of preprocess_volume_fixed_size
        volume = process_full_scan(
            self.patient_ids[idx], 
            new_height=self.target_shape[0],
            new_width=self.target_shape[1],
            new_depth=self.target_shape[2]
        )
        
        # Convert to tensor: (C, D, H, W) format for 3D CNNs
        volume_tensor = torch.from_numpy(volume).float()
        volume_tensor = volume_tensor.unsqueeze(0)  # Add channel dimension        
        
        # Get additional features
        baseline_fvc = torch.tensor(self.baseline_fvcs[idx]).float()
        fev1 = torch.tensor(self.fev1s[idx]).float()
        age = torch.tensor(self.ages[idx]).float()
        diagnosis = torch.tensor(self.diagnosis_encoded[idx]).float()  # Already one-hot encoded array
        sex = torch.tensor(self.sex_encoded[idx]).float()  # Already one-hot encoded array
        smoking = torch.tensor(self.smoking_encoded[idx]).float()  # Already one-hot encoded array
        target = torch.tensor(self.targets[idx]).float()
        
        # Combine tabular features (baseline_fvc + fev1 + age + one-hot encoded categoricals)
        tabular_features = torch.cat([
            baseline_fvc.unsqueeze(0),  # 1D
            fev1.unsqueeze(0),  # 1D
            age.unsqueeze(0),  # 1D
            diagnosis,  # Already 1D one-hot vector
            sex,  # Already 1D one-hot vector
            smoking  # Already 1D one-hot vector
        ])
        
        # Check tensor for NaN
        if torch.any(torch.isnan(volume_tensor)) or torch.any(torch.isnan(tabular_features)) or torch.any(torch.isnan(target)):
            print(f"Warning: NaN in tensor for patient {self.patient_ids[idx]}")
            volume_tensor = torch.nan_to_num(volume_tensor)
            tabular_features = torch.nan_to_num(tabular_features)
            target = torch.nan_to_num(target)
            
        return volume_tensor, tabular_features, target
    
    def __len__(self):
        return len(self.patient_ids)
    

targets_shape = (256, 256, 200)

print(f'Target shape for volumes: {targets_shape}')
batch_size = 4
print(f"Batch size set to {batch_size}.")

# Cross-validation and output directory configuration
# Use predefined split files created in patient_split.ipynb
split_seeds = [42]#, 24, 7]
seed_to_use = split_seeds[-1]
num_folds = 5
total_folds = len(split_seeds) * num_folds
output_dir = f"cv_multi_one_hot_5_cv_scaled_ES{timestamp}"
os.makedirs(output_dir, exist_ok=True)
print(f"Cross-validation folds: {num_folds}")
print(f"Split seeds: {split_seeds}")
print(f"Total folds across seeds: {total_folds}")
print(f"Output directory: {output_dir}")

patients_ids = cases_df['PatientID'].tolist()
patient_id_to_idx = {pid: idx for idx, pid in enumerate(patients_ids)}


def load_fold_patient_ids(seed_value, fold_idx_zero_based):
    split_dir = f"Patient_Splits_Seed{seed_value}"
    train_fold_path = os.path.join(split_dir, f"train_fold_{fold_idx_zero_based}.csv")
    val_fold_path = os.path.join(split_dir, f"val_fold_{fold_idx_zero_based}.csv")

    if not os.path.exists(train_fold_path) or not os.path.exists(val_fold_path):
        raise FileNotFoundError(
            f"Could not find split files for seed {seed_value}, fold {fold_idx_zero_based}: "
            f"{train_fold_path}, {val_fold_path}"
        )

    train_ids = pd.read_csv(train_fold_path)['PatientID'].astype(str).tolist()
    val_ids = pd.read_csv(val_fold_path)['PatientID'].astype(str).tolist()
    return train_ids, val_ids, train_fold_path, val_fold_path


def load_test_patient_ids(seed_value):
    split_dir = f"Patient_Splits_Seed{seed_value}"
    test_path = os.path.join(split_dir, 'test_patients.csv')
    if not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Could not find test patient file for seed {seed_value}: {test_path}"
        )

    test_ids = pd.read_csv(test_path)['PatientID'].astype(str).tolist()
    return test_ids, test_path

# Hyperparameters
lr = 3e-3 
weight_decay = 1e-7
early_stopping_patience = 20
epochs = 100
cnn_output_dim = 512  # ResNet18 final feature dimension after removing FC layer
hidden_dim = 256 
# tabular_input_dim = 3 (baseline_fvc, fev1, age) + total one-hot encoded dimensions
tabular_input_dim = 3 + total_categorical_dims

def build_tabular_arrays(patient_ids_subset):
    subset_targets = [cases_df.iloc[patient_id_to_idx[pid]]['Followup FVC Volume L'] for pid in patient_ids_subset]
    subset_baseline_fvcs = [cases_df.iloc[patient_id_to_idx[pid]]['Baseline FVC Volume L'] for pid in patient_ids_subset]
    subset_fev1s = [cases_df.iloc[patient_id_to_idx[pid]]['FEV1 Volume L'] for pid in patient_ids_subset]
    subset_ages = [cases_df.iloc[patient_id_to_idx[pid]]['Age'] for pid in patient_ids_subset]
    subset_diagnosis = [diagnosis_encoded[patient_id_to_idx[pid]] for pid in patient_ids_subset]
    subset_sex = [sex_encoded[patient_id_to_idx[pid]] for pid in patient_ids_subset]
    subset_smoking = [smoking_encoded[patient_id_to_idx[pid]] for pid in patient_ids_subset]

    return (
        subset_targets,
        subset_baseline_fvcs,
        subset_fev1s,
        subset_ages,
        subset_diagnosis,
        subset_sex,
        subset_smoking
    )


def build_dataloader(
    patient_ids_subset,
    targets,
    baseline_fvcs,
    fev1s,
    ages,
    diagnosis_encoded_subset,
    sex_encoded_subset,
    smoking_encoded_subset,
    shuffle=False
):

    dataset = FixedSizeDataset(
        patient_ids=patient_ids_subset,
        targets=targets,
        baseline_fvcs=baseline_fvcs,
        fev1s=fev1s,
        ages=ages,
        diagnosis_encoded=diagnosis_encoded_subset,
        sex_encoded=sex_encoded_subset,
        smoking_encoded=smoking_encoded_subset,
        target_shape=targets_shape
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2
    )
    return dataset, loader

class MultiInputModel(nn.Module):
    def __init__(self, cnn_backbone, cnn_output_dim=512, tabular_input_dim=5, hidden_dim=256):
        super(MultiInputModel, self).__init__()
        
        # 3D CNN backbone for processing images
        self.cnn_backbone = cnn_backbone
        
        # Remove the final FC layer from the backbone
        self.cnn_backbone.fc = nn.Identity()
        
        # MLP for processing tabular features (baseline FVC + FEV1 + Age + categoricals)
        self.tabular_mlp = nn.Sequential(
            nn.Linear(tabular_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Combined layer
        combined_input_dim = cnn_output_dim + (hidden_dim // 2)
        self.combined_fc = nn.Sequential(
            nn.Linear(combined_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),  # Reduced from 0.3 to avoid underfitting
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),  # Reduced from 0.2 to avoid underfitting
            nn.Linear(hidden_dim // 2, 1)
        )
        

    def forward(self, image, tabular_features):
        # Process image through CNN
        cnn_features = self.cnn_backbone(image)

        # Flatten if needed
        if cnn_features.dim() > 2:
            cnn_features = cnn_features.view(cnn_features.size(0), -1)

        # Process tabular features
        if tabular_features.dim() == 1:
            tabular_features = tabular_features.unsqueeze(0)
        
        tabular_processed = self.tabular_mlp(tabular_features)

        # Concatenate features
        combined = torch.cat([cnn_features, tabular_processed], dim=1)

        # Output
        return self.combined_fc(combined)

# Move model to device and ensure correct data type
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ============================================================================
# TRAINING CONFIGURATION SUMMARY
# ============================================================================
print("\n" + "="*80)
print("TRAINING CONFIGURATION SUMMARY")
print("="*80)
print(f"\nRun Timestamp: {timestamp}")
print(f"\nDATA CONFIGURATION:")
print(f"  Total Patients: {len(patients_ids)}")
print(f"  Split Seeds: {split_seeds}")
print(f"  Number of Folds per Seed: {num_folds}")
print(f"  Total Folds: {total_folds}")
print(f"  Target Volume Shape (H, W, D): {targets_shape}")
print(f"\nMODEL ARCHITECTURE:")
print(f"  Backbone: ResNet18 + ACS 3D Conversion")
print(f"  CNN Output Dimension: {cnn_output_dim}")
print(f"  Tabular Input Dimension: {tabular_input_dim} (Baseline FVC, FEV1, Age, One-Hot: Diagnosis={n_diagnosis_cats}, Sex={n_sex_cats}, Smoking={n_smoking_cats})")
print(f"  Hidden Dimension: {hidden_dim}")
print(f"\nTRAINING HYPERPARAMETERS:")
print(f"  Batch Size: {batch_size}")
print(f"  Learning Rate: {lr}")
print(f"  Weight Decay: {weight_decay}")
print(f"  Optimizer: Adam")
print(f"  Loss Function: MSE")
print(f"  Max Epochs: {epochs}")
print(f"  Early Stopping Patience: {early_stopping_patience}")
print(f"  Learning Rate Scheduler: ReduceLROnPlateau (patience=6, factor=0.6)")
print(f"\nCOMPUTE:")
print(f"  Device: {device}")
print(f"  DataLoader Workers: 2")
print(f"  Output Directory: {output_dir}")
print("="*80 + "\n")

fold_results = []
global_fold_idx = 0

for seed_value in split_seeds:
    print("\n" + "="*80)
    print(f"Starting seed {seed_value}")
    print("="*80)

    test_patient_ids, test_path = load_test_patient_ids(seed_value)
    missing_test_ids = [pid for pid in test_patient_ids if pid not in patient_id_to_idx]
    if missing_test_ids:
        print(
            f"Warning: {len(missing_test_ids)} test IDs from {test_path} were not found in cases_df and will be skipped."
        )
    test_patient_ids = [pid for pid in test_patient_ids if pid in patient_id_to_idx]
    if len(test_patient_ids) == 0:
        raise ValueError(f"No valid test IDs available for seed {seed_value} after filtering.")

    for fold_idx in range(1, num_folds + 1):
        global_fold_idx += 1
        print("\n" + "-"*80)
        print(
            f"Starting Seed {seed_value} Fold {fold_idx}/{num_folds} "
            f"(Global Fold {global_fold_idx}/{total_folds})"
        )
        print("-"*80)

        fold_idx_zero_based = fold_idx - 1
        train_patient_ids, val_patient_ids, train_fold_path, val_fold_path = load_fold_patient_ids(seed_value, fold_idx_zero_based)

        print(f"Train fold file: {train_fold_path}")
        print(f"Validation fold file: {val_fold_path}")
        print(f"Train patients in fold: {len(train_patient_ids)}")
        print(f"Validation patients in fold: {len(val_patient_ids)}")
        print(f"Test split file: {test_path}")
        print(f"Test patients in seed {seed_value}: {len(test_patient_ids)}")

        (
            train_targets,
            train_baseline,
            train_fev1,
            train_ages,
            train_diag,
            train_sex,
            train_smoking
        ) = build_tabular_arrays(train_patient_ids)

        (
            val_targets,
            val_baseline,
            val_fev1,
            val_ages,
            val_diag,
            val_sex,
            val_smoking
        ) = build_tabular_arrays(val_patient_ids)

        (
            test_targets,
            test_baseline,
            test_fev1,
            test_ages,
            test_diag,
            test_sex,
            test_smoking
        ) = build_tabular_arrays(test_patient_ids)

        # Scale only continuous tabular features (Baseline FVC, FEV1, Age) using train-fold stats.
        train_cont = np.column_stack([train_baseline, train_fev1, train_ages]).astype(np.float32)
        val_cont = np.column_stack([val_baseline, val_fev1, val_ages]).astype(np.float32)
        test_cont = np.column_stack([test_baseline, test_fev1, test_ages]).astype(np.float32)

        fold_scaler = StandardScaler()
        fold_scaler.fit(train_cont)
        train_cont_scaled = fold_scaler.transform(train_cont)
        val_cont_scaled = fold_scaler.transform(val_cont)
        test_cont_scaled = fold_scaler.transform(test_cont)

        train_dataset, train_loader = build_dataloader(
            train_patient_ids,
            train_targets,
            train_cont_scaled[:, 0].tolist(),
            train_cont_scaled[:, 1].tolist(),
            train_cont_scaled[:, 2].tolist(),
            train_diag,
            train_sex,
            train_smoking,
            shuffle=True
        )
        val_dataset, val_loader = build_dataloader(
            val_patient_ids,
            val_targets,
            val_cont_scaled[:, 0].tolist(),
            val_cont_scaled[:, 1].tolist(),
            val_cont_scaled[:, 2].tolist(),
            val_diag,
            val_sex,
            val_smoking,
            shuffle=False
        )
        test_dataset, test_loader = build_dataloader(
            test_patient_ids,
            test_targets,
            test_cont_scaled[:, 0].tolist(),
            test_cont_scaled[:, 1].tolist(),
            test_cont_scaled[:, 2].tolist(),
            test_diag,
            test_sex,
            test_smoking,
            shuffle=False
        )

        print(f'Seed {seed_value} Fold {fold_idx}: Training dataset created with {len(train_dataset)} samples.')
        print(f'Seed {seed_value} Fold {fold_idx}: Validation dataset created with {len(val_dataset)} samples.')
        print(f'Seed {seed_value} Fold {fold_idx}: Test dataset created with {len(test_dataset)} samples.')

        # Define fresh model per fold
        backbone = models.resnet18(weights='IMAGENET1K_V1')
        backbone.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        backbone_3d = ACSConverter(backbone).to(device)
        model_3d = MultiInputModel(backbone_3d, cnn_output_dim=cnn_output_dim, tabular_input_dim=tabular_input_dim, hidden_dim=hidden_dim)
        model_3d = model_3d.type(torch.FloatTensor).to(device)

        optimizer = optim.Adam(model_3d.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=6)
        criterion = nn.MSELoss()

        print("Model summary:")
        print(model_3d)

        best_val_loss = float('inf')
        epochs_no_improve = 0
        loss_history = {'train_loss': [], 'val_loss': []}
        best_model_path = os.path.join(output_dir, f'best_model_seed{seed_value}_fold{fold_idx}_lr_multi_all_{timestamp}.pth')

        print(f"Seed {seed_value} Fold {fold_idx}: Starting Training Loop...")
        for epoch in range(epochs):
            model_3d.train()
            train_loss = 0.0

            for i, (images, tabular_features, targets) in enumerate(train_loader):
                optimizer.zero_grad()

                images = images.type(torch.FloatTensor).to(device, non_blocking=True)
                tabular_features = tabular_features.type(torch.FloatTensor).to(device, non_blocking=True)
                targets = targets.type(torch.FloatTensor).to(device, non_blocking=True)

                outputs = model_3d(images, tabular_features)
                loss = criterion(outputs, targets.unsqueeze(1))

                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                if i % 10 == 0:
                    torch.cuda.empty_cache()

                del images, tabular_features, targets, outputs, loss

            model_3d.eval()
            val_loss = 0.0

            with torch.no_grad():
                for i, (images, tabular_features, targets) in enumerate(val_loader):
                    images = images.type(torch.FloatTensor).to(device, non_blocking=True)
                    tabular_features = tabular_features.type(torch.FloatTensor).to(device, non_blocking=True)
                    targets = targets.type(torch.FloatTensor).to(device, non_blocking=True)

                    outputs = model_3d(images, tabular_features)
                    loss = criterion(outputs, targets.unsqueeze(1))

                    val_loss += loss.item()

                    if i % 10 == 0:
                        torch.cuda.empty_cache()

                    del images, tabular_features, targets, outputs, loss

            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)

            loss_history['train_loss'].append(avg_train_loss)
            loss_history['val_loss'].append(avg_val_loss)

            current_lr = optimizer.param_groups[0]['lr']
            print(f"Seed {seed_value} Fold {fold_idx} | Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, LR: {current_lr:.2e}")

            scheduler.step(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                torch.save(model_3d.state_dict(), best_model_path)
                print(f"Seed {seed_value} Fold {fold_idx}: New best model saved with validation loss: {best_val_loss:.6f}")
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= early_stopping_patience:
                print(f"Seed {seed_value} Fold {fold_idx}: Early stopping triggered after {epoch+1} epochs")
                break

            torch.cuda.empty_cache()

        print(f"Seed {seed_value} Fold {fold_idx}: Training complete.")
        print(f"Seed {seed_value} Fold {fold_idx}: Best validation loss: {best_val_loss:.6f}")

        # Evaluate on validation set
        model_3d.load_state_dict(torch.load(best_model_path))
        model_3d = model_3d.type(torch.FloatTensor).to(device)
        model_3d.eval()

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for images, tabular_features, targets in val_loader:
                images = images.type(torch.FloatTensor).to(device, non_blocking=True)
                tabular_features = tabular_features.type(torch.FloatTensor).to(device, non_blocking=True)
                targets = targets.type(torch.FloatTensor).to(device, non_blocking=True)

                outputs = model_3d(images, tabular_features)
                predictions = outputs.squeeze().cpu().numpy()
                targets_np = targets.cpu().numpy()

                if predictions.ndim == 0:
                    predictions = np.array([predictions])
                if targets_np.ndim == 0:
                    targets_np = np.array([targets_np])

                all_predictions.extend(predictions)
                all_targets.extend(targets_np)

                del images, tabular_features, targets, outputs

        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)

        val_mse = mean_squared_error(all_targets, all_predictions)
        val_rmse = np.sqrt(val_mse)
        val_mae = mean_absolute_error(all_targets, all_predictions)
        val_r2 = r2_score(all_targets, all_predictions)

        print(f"\nSeed {seed_value} Fold {fold_idx} Validation Metrics:")
        print(f"MSE: {val_mse:.6f}")
        print(f"RMSE: {val_rmse:.6f}")
        print(f"MAE: {val_mae:.6f}")
        print(f"R² Score: {val_r2:.6f}")
        print(f"Best Validation Loss: {best_val_loss:.6f}")
        print(f"Total Epochs Trained: {len(loss_history['train_loss'])}")

        validation_results = {
            'Timestamp': [timestamp],
            'Seed': [seed_value],
            'Fold': [fold_idx],
            'Global_Fold': [global_fold_idx],
            'Batch_Size': [batch_size],
            'Learning_Rate': [lr],
            'Weight_Decay': [weight_decay],
            'Early_Stopping_Patience': [early_stopping_patience],
            'Total_Patients': [len(patients_ids)],
            'Train_Samples': [len(train_patient_ids)],
            'Val_Samples': [len(val_patient_ids)],
            'MSE': [val_mse],
            'RMSE': [val_rmse],
            'MAE': [val_mae],
            'R2_Score': [val_r2],
            'Best_Val_Loss': [best_val_loss],
            'Total_Epochs': [len(loss_history['train_loss'])],
            'CNN_Output_Dim': [cnn_output_dim],
            'Hidden_Dim': [hidden_dim],
            'Best_Model_Path': [best_model_path]
        }
        validation_df = pd.DataFrame(validation_results)
        fold_validation_path = os.path.join(output_dir, f'validation_metrics_seed{seed_value}_fold{fold_idx}_multi_all_{timestamp}.csv')
        validation_df.to_csv(fold_validation_path, index=False)
        print(f"Seed {seed_value} Fold {fold_idx}: Validation metrics saved to: {fold_validation_path}")

        # Test evaluation
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for images, tabular_features, targets in test_loader:
                images = images.type(torch.FloatTensor).to(device, non_blocking=True)
                tabular_features = tabular_features.type(torch.FloatTensor).to(device, non_blocking=True)
                targets = targets.type(torch.FloatTensor).to(device, non_blocking=True)

                outputs = model_3d(images, tabular_features)
                predictions = outputs.squeeze().cpu().numpy()
                targets_np = targets.cpu().numpy()

                if predictions.ndim == 0:
                    predictions = np.array([predictions])
                if targets_np.ndim == 0:
                    targets_np = np.array([targets_np])

                all_predictions.extend(predictions)
                all_targets.extend(targets_np)

                del images, tabular_features, targets, outputs

        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)

        test_mse = mean_squared_error(all_targets, all_predictions)
        test_rmse = np.sqrt(test_mse)
        test_mae = mean_absolute_error(all_targets, all_predictions)
        test_r2 = r2_score(all_targets, all_predictions)

        print(f"\nSeed {seed_value} Fold {fold_idx} Test Metrics:")
        print(f"MSE: {test_mse:.6f}")
        print(f"RMSE: {test_rmse:.6f}")
        print(f"MAE: {test_mae:.6f}")
        print(f"R² Score: {test_r2:.6f}")
        print(f"Best Validation Loss: {best_val_loss:.6f}")
        print(f"Total Epochs Trained: {len(loss_history['train_loss'])}")

        eval_results = {
            'Timestamp': [timestamp],
            'Seed': [seed_value],
            'Fold': [fold_idx],
            'Global_Fold': [global_fold_idx],
            'Batch_Size': [batch_size],
            'Learning_Rate': [lr],
            'Weight_Decay': [weight_decay],
            'Early_Stopping_Patience': [early_stopping_patience],
            'Total_Patients': [len(patients_ids)],
            'Train_Samples': [len(train_patient_ids)],
            'Val_Samples': [len(val_patient_ids)],
            'Test_Samples': [len(test_patient_ids)],
            'MSE': [test_mse],
            'RMSE': [test_rmse],
            'MAE': [test_mae],
            'R2_Score': [test_r2],
            'Best_Val_Loss': [best_val_loss],
            'Total_Epochs': [len(loss_history['train_loss'])],
            'CNN_Output_Dim': [cnn_output_dim],
            'Hidden_Dim': [hidden_dim],
            'Best_Model_Path': [best_model_path]
        }

        eval_df = pd.DataFrame(eval_results)
        fold_eval_path = os.path.join(output_dir, f'evaluation_metrics_seed{seed_value}_fold{fold_idx}_multi_all_{timestamp}.csv')
        eval_df.to_csv(fold_eval_path, index=False)
        print(f"Seed {seed_value} Fold {fold_idx}: Evaluation metrics saved to: {fold_eval_path}")

        loss_history['epoch'] = list(range(1, len(loss_history['train_loss']) + 1))
        loss_df = pd.DataFrame(loss_history)
        loss_history_path = os.path.join(output_dir, f'loss_history_seed{seed_value}_fold{fold_idx}_multi_all_{timestamp}.csv')
        loss_df.to_csv(loss_history_path, index=False)
        print(f"Seed {seed_value} Fold {fold_idx}: Loss history saved to: {loss_history_path}")

        model_name = os.path.basename(best_model_path)

        plt.rcParams.update({
            "font.family": "serif",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        })

        plt.figure(figsize=(10, 6))
        plt.plot(loss_history['epoch'], loss_history['train_loss'], label='Train Loss')
        plt.plot(loss_history['epoch'], loss_history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training and Validation Loss Curves - Fold {fold_idx}\nModel: {model_name}\nBS: {batch_size}, LR: {lr}, WD: {weight_decay}, ES Patience: {early_stopping_patience}, Samples: {len(train_patient_ids)}/{len(val_patient_ids)}')
        plt.legend()
        plt.grid()
        loss_curve_path = os.path.join(output_dir, f'loss_curves_seed{seed_value}_fold{fold_idx}_multi_all_{timestamp}.png')
        plt.savefig(loss_curve_path)
        plt.close()

        plt.figure(figsize=(8, 8))
        plt.scatter(all_targets, all_predictions, alpha=0.7)
        plt.plot([all_targets.min(), all_targets.max()], [all_targets.min(), all_targets.max()], 'r--')
        plt.xlabel('True FVC Volume L')
        plt.ylabel('Predicted FVC Volume L')
        plt.title(f'Predicted vs True FVC Volume L - Fold {fold_idx}\nModel: {model_name}\nMSE: {test_mse:.6f}, RMSE: {test_rmse:.6f}, MAE: {test_mae:.6f}, R²: {test_r2:.6f}')
        plt.grid()
        pred_scatter_path = os.path.join(output_dir, f'predicted_vs_true_seed{seed_value}_fold{fold_idx}_multi_all_{timestamp}.png')
        plt.savefig(pred_scatter_path)
        plt.close()

        residuals = all_targets - all_predictions
        plt.figure(figsize=(10, 6))
        plt.scatter(all_predictions, residuals, alpha=0.7)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted FVC Volume L')
        plt.ylabel('Residual (True - Predicted)')
        plt.title(f'Residual Plot - Fold {fold_idx}\nModel: {model_name}\nMSE: {test_mse:.6f}, RMSE: {test_rmse:.6f}, MAE: {test_mae:.6f}, R²: {test_r2:.6f}')
        plt.grid()
        residual_plot_path = os.path.join(output_dir, f'residual_plot_seed{seed_value}_fold{fold_idx}_multi_all_{timestamp}.png')
        plt.savefig(residual_plot_path)
        plt.close()

        print(f"Seed {seed_value} Fold {fold_idx}: Predicted vs True scatter plot saved to: {pred_scatter_path}")
        print(f"Seed {seed_value} Fold {fold_idx}: Loss curves plot saved to: {loss_curve_path}")
        print(f"Seed {seed_value} Fold {fold_idx}: Residual plot saved to: {residual_plot_path}")
        print(f"Seed {seed_value} Fold {fold_idx}: Model saved to: {best_model_path}")

        fold_results.append({
            'Seed': seed_value,
            'Fold': fold_idx,
            'Global_Fold': global_fold_idx,
            'Train_Samples': len(train_patient_ids),
            'Val_Samples': len(val_patient_ids),
            'Test_Samples': len(test_patient_ids),
            'MSE': val_mse,
            'RMSE': val_rmse,
            'MAE': val_mae,
            'R2_Score': val_r2,
            'Test_MSE': test_mse,
            'Test_RMSE': test_rmse,
            'Test_MAE': test_mae,
            'Test_R2_Score': test_r2,
            'Best_Val_Loss': best_val_loss,
            'Total_Epochs': len(loss_history['train_loss']),
            'Best_Model_Path': best_model_path
        })

# Keep references from the last fold for explainability
explain_model = model_3d
explain_val_loader = test_loader

# Save fold metrics and CV summary
fold_results_df = pd.DataFrame(fold_results)
fold_results_path = os.path.join(output_dir, f'cv_fold_metrics_multi_all_all_seeds_fold_{total_folds}_{timestamp}.csv')
fold_results_df.to_csv(fold_results_path, index=False)

cv_summary = {
    'Timestamp': [timestamp],
    'Split_Seeds': [str(split_seeds)],
    'Num_Seeds': [len(split_seeds)],
    'Num_Folds_Per_Seed': [num_folds],
    'Total_Folds': [total_folds],
    'MSE_Mean': [fold_results_df['MSE'].mean()],
    'MSE_Std': [fold_results_df['MSE'].std()],
    'RMSE_Mean': [fold_results_df['RMSE'].mean()],
    'RMSE_Std': [fold_results_df['RMSE'].std()],
    'MAE_Mean': [fold_results_df['MAE'].mean()],
    'MAE_Std': [fold_results_df['MAE'].std()],
    'R2_Mean': [fold_results_df['R2_Score'].mean()],
    'R2_Std': [fold_results_df['R2_Score'].std()],
    'Test_MSE_Mean': [fold_results_df['Test_MSE'].mean()],
    'Test_MSE_Std': [fold_results_df['Test_MSE'].std()],
    'Test_RMSE_Mean': [fold_results_df['Test_RMSE'].mean()],
    'Test_RMSE_Std': [fold_results_df['Test_RMSE'].std()],
    'Test_MAE_Mean': [fold_results_df['Test_MAE'].mean()],
    'Test_MAE_Std': [fold_results_df['Test_MAE'].std()],
    'Test_R2_Mean': [fold_results_df['Test_R2_Score'].mean()],
    'Test_R2_Std': [fold_results_df['Test_R2_Score'].std()],
    'Best_Val_Loss_Mean': [fold_results_df['Best_Val_Loss'].mean()],
    'Best_Val_Loss_Std': [fold_results_df['Best_Val_Loss'].std()]
}
cv_summary_df = pd.DataFrame(cv_summary)
cv_summary_path = os.path.join(output_dir, f'cv_summary_multi_all_all_seeds_fold_{total_folds}_{timestamp}.csv')
cv_summary_df.to_csv(cv_summary_path, index=False)

print("\n" + "="*80)
print("CROSS-VALIDATION COMPLETE")
print("="*80)
print(f"Fold metrics saved to: {fold_results_path}")
print(f"CV summary saved to: {cv_summary_path}")
print(f"All artifacts saved under: {output_dir}")
print(f"\nCV Mean Metrics:")
print(f"  MSE:  {cv_summary['MSE_Mean'][0]:.6f} ± {cv_summary['MSE_Std'][0]:.6f}")
print(f"  RMSE: {cv_summary['RMSE_Mean'][0]:.6f} ± {cv_summary['RMSE_Std'][0]:.6f}")
print(f"  MAE:  {cv_summary['MAE_Mean'][0]:.6f} ± {cv_summary['MAE_Std'][0]:.6f}")
print(f"  R²:   {cv_summary['R2_Mean'][0]:.6f} ± {cv_summary['R2_Std'][0]:.6f}")
print(f"\nCV Mean Test Metrics:")
print(f"  Test MSE:  {cv_summary['Test_MSE_Mean'][0]:.6f} ± {cv_summary['Test_MSE_Std'][0]:.6f}")
print(f"  Test RMSE: {cv_summary['Test_RMSE_Mean'][0]:.6f} ± {cv_summary['Test_RMSE_Std'][0]:.6f}")
print(f"  Test MAE:  {cv_summary['Test_MAE_Mean'][0]:.6f} ± {cv_summary['Test_MAE_Std'][0]:.6f}")
print(f"  Test R²:   {cv_summary['Test_R2_Mean'][0]:.6f} ± {cv_summary['Test_R2_Std'][0]:.6f}")

# # ============================================================================
# # EXPLAINABILITY: TABULAR SHAP + IMAGE IMPORTANCE (LAST FOLD MODEL)
# # ============================================================================
# try:
#     print("\n" + "="*80)
#     print("GENERATING EXPLAINABILITY OUTPUTS")
#     print("="*80)

#     explain_model.eval()

#     collected_images = []
#     collected_tabular = []
#     collected_targets = []
#     for images, tabular_features, targets in explain_val_loader:
#         collected_images.append(images)
#         collected_tabular.append(tabular_features)
#         collected_targets.append(targets)
#         if sum(batch.shape[0] for batch in collected_tabular) >= 32:
#             break

#     if len(collected_tabular) == 0:
#         raise RuntimeError("Validation loader is empty. Cannot generate explainability outputs.")

#     val_images = torch.cat(collected_images, dim=0)[:32].type(torch.FloatTensor).to(device)
#     val_tabular = torch.cat(collected_tabular, dim=0)[:32].type(torch.FloatTensor).to(device)
#     val_targets = torch.cat(collected_targets, dim=0)[:32].type(torch.FloatTensor).to(device)

#     # -------------------------
#     # 1) TABULAR SHAP (conditioned on one image)
#     # -------------------------
#     fixed_image = val_images[0:1].detach()
#     background_size = min(20, val_tabular.shape[0])
#     explain_size = min(10, val_tabular.shape[0])

#     background_tab = val_tabular[:background_size].detach().cpu().numpy()
#     explain_tab = val_tabular[:explain_size].detach().cpu().numpy()

#     def predict_with_fixed_image(tab_array_np):
#         tab_tensor = torch.tensor(tab_array_np, dtype=torch.float32, device=device)
#         image_batch = fixed_image.repeat(tab_tensor.shape[0], 1, 1, 1, 1)
#         with torch.no_grad():
#             preds = explain_model(image_batch, tab_tensor).squeeze(1)
#         return preds.detach().cpu().numpy()

#     explainer = shap.KernelExplainer(predict_with_fixed_image, background_tab)
#     shap_values = explainer.shap_values(explain_tab, nsamples=100)

#     if isinstance(shap_values, list):
#         shap_values = shap_values[0]
#     shap_values = np.array(shap_values)

#     tabular_feature_names = (
#         ['Baseline FVC Volume L', 'FEV1 Volume L', 'Age'] +
#         [f'Primary Diagnosis_{cat}' for cat in enc_diagnosis.categories_[0]] +
#         [f'Sex_{cat}' for cat in enc_sex.categories_[0]] +
#         [f'Smoking History_{cat}' for cat in enc_smoking.categories_[0]]
#     )

#     mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
#     shap_importance_df = pd.DataFrame({
#         'Feature': tabular_feature_names,
#         'MeanAbsSHAP': mean_abs_shap
#     }).sort_values('MeanAbsSHAP', ascending=False)

#     shap_csv_path = os.path.join(output_dir, f'tabular_shap_importance_seed{seed_to_use}_{timestamp}.csv')
#     shap_importance_df.to_csv(shap_csv_path, index=False)

#     top_k = min(15, len(shap_importance_df))
#     plt.figure(figsize=(10, 6))
#     plt.barh(shap_importance_df['Feature'].head(top_k)[::-1], shap_importance_df['MeanAbsSHAP'].head(top_k)[::-1])
#     plt.xlabel('Mean |SHAP value|')
#     plt.ylabel('Feature')
#     plt.title(f'Tabular SHAP Importance (conditioned on fixed image) - Top {top_k}')
#     plt.tight_layout()
#     shap_plot_path = os.path.join(output_dir, f'tabular_shap_importance_seed{seed_to_use}_{timestamp}.png')
#     plt.savefig(shap_plot_path)
#     plt.close()

#     print(f"Tabular SHAP importance saved to: {shap_csv_path}")
#     print(f"Tabular SHAP plot saved to: {shap_plot_path}")

#     # -------------------------
#     # 2) IMAGE IMPORTANCE (gradient saliency)
#     # -------------------------
#     sample_image = val_images[0:1].clone().detach().requires_grad_(True)
#     sample_tabular = val_tabular[0:1].clone().detach()

#     explain_model.zero_grad(set_to_none=True)
#     pred = explain_model(sample_image, sample_tabular)
#     pred.backward(torch.ones_like(pred))

#     sample_pred_fvc = float(pred.detach().squeeze().cpu().item())
#     sample_true_fvc = float(val_targets[0].detach().cpu().item())
#     sample_baseline_fvc = float(sample_tabular[0, 0].detach().cpu().item())
#     sample_fev1 = float(sample_tabular[0, 1].detach().cpu().item())

#     grad_saliency = sample_image.grad.detach().abs().squeeze(0).squeeze(0).cpu().numpy()  # (D, H, W)
#     image_np = sample_image.detach().squeeze(0).squeeze(0).cpu().numpy()  # (D, H, W)

#     d, h, w = grad_saliency.shape
#     d_mid, h_mid, w_mid = d // 2, h // 2, w // 2

#     def normalize_slice(x):
#         x_min, x_max = np.min(x), np.max(x)
#         if x_max - x_min < 1e-8:
#             return np.zeros_like(x)
#         return (x - x_min) / (x_max - x_min)

#     img_axial = normalize_slice(image_np[d_mid, :, :])
#     img_coronal = normalize_slice(image_np[:, h_mid, :])
#     img_sagittal = normalize_slice(image_np[:, :, w_mid])

#     sal_axial = normalize_slice(grad_saliency[d_mid, :, :])
#     sal_coronal = normalize_slice(grad_saliency[:, h_mid, :])
#     sal_sagittal = normalize_slice(grad_saliency[:, :, w_mid])

#     plt.figure(figsize=(12, 8))
#     plt.subplot(2, 3, 1)
#     plt.imshow(img_axial, cmap='gray')
#     plt.title('Image Axial')
#     plt.axis('off')

#     plt.subplot(2, 3, 2)
#     plt.imshow(img_coronal, cmap='gray')
#     plt.title('Image Coronal')
#     plt.axis('off')

#     plt.subplot(2, 3, 3)
#     plt.imshow(img_sagittal, cmap='gray')
#     plt.title('Image Sagittal')
#     plt.axis('off')

#     plt.subplot(2, 3, 4)
#     plt.imshow(sal_axial, cmap='hot')
#     plt.title('Saliency Axial')
#     plt.axis('off')

#     plt.subplot(2, 3, 5)
#     plt.imshow(sal_coronal, cmap='hot')
#     plt.title('Saliency Coronal')
#     plt.axis('off')

#     plt.subplot(2, 3, 6)
#     plt.imshow(sal_sagittal, cmap='hot')
#     plt.title('Saliency Sagittal')
#     plt.axis('off')

#     plt.suptitle(
#         f'Image Importance (Gradient Saliency)\n'
#         f'Baseline FVC: {sample_baseline_fvc:.3f} L | '
#         f'FEV1: {sample_fev1:.3f} L | '
#         f'True Follow-up FVC: {sample_true_fvc:.3f} L | '
#         f'Pred Follow-up FVC: {sample_pred_fvc:.3f} L'
#     )
#     plt.tight_layout()
#     image_imp_path = os.path.join(output_dir, f'image_importance_saliency_seed{seed_to_use}_{timestamp}.png')
#     plt.savefig(image_imp_path)
#     plt.close()

#     print(f"Image saliency plot saved to: {image_imp_path}")

# except Exception as explainability_error:
#     print(f"Explainability step skipped due to error: {explainability_error}")
