import os
import numpy as np
from scipy.ndimage import zoom
import nibabel as nib
import pandas as pd

import torchvision.models as models  
from acsconv.converters import ACSConverter  

import torch  
import torch.nn as nn  
import torch.optim as optim  
from skimage.transform import resize
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Load data paths and targets from CSV
cases_df = pd.read_csv('cleaned_collected_cases_final.csv')
cases_df['PatientID'] = cases_df['PatientID'].astype(str)

images_path = cases_df['Image'].tolist()
masks_path = cases_df['Mask'].tolist()

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
    def __init__(self, patient_ids, targets, target_shape=(256, 256, 200)):
        self.patient_ids = patient_ids
        self.targets = targets
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
        
        target = torch.tensor(self.targets[idx]).float()

        # Check tensor for NaN
        if torch.any(torch.isnan(volume_tensor)) or torch.any(torch.isnan(target)):
            print(f"Warning: NaN in tensor for patient {self.patient_ids[idx]}")
            volume_tensor = torch.nan_to_num(volume_tensor)
            target = torch.nan_to_num(target)

        return volume_tensor, target
    
    def __len__(self):
        return len(self.patient_ids)
    

targets_shape = (256, 256, 200)#(384, 384, 300)

split_seeds = [42]
seed_to_use = split_seeds[-1]
fold_to_use = 1
split_dir = f'Patient_Splits_Seed{seed_to_use}'

patients_ids = cases_df['PatientID'].tolist()
patient_id_to_idx = {pid: idx for idx, pid in enumerate(patients_ids)}


def load_fold_patient_ids(seed_value, fold_idx_zero_based):
    train_fold_path = os.path.join(f'Patient_Splits_Seed{seed_value}', f'train_fold_{fold_idx_zero_based}.csv')
    val_fold_path = os.path.join(f'Patient_Splits_Seed{seed_value}', f'val_fold_{fold_idx_zero_based}.csv')

    if not os.path.exists(train_fold_path) or not os.path.exists(val_fold_path):
        raise FileNotFoundError(
            f"Could not find split files for seed {seed_value}, fold {fold_idx_zero_based}: "
            f"{train_fold_path}, {val_fold_path}"
        )

    train_ids = pd.read_csv(train_fold_path)['PatientID'].astype(str).tolist()
    val_ids = pd.read_csv(val_fold_path)['PatientID'].astype(str).tolist()
    return train_ids, val_ids, train_fold_path, val_fold_path


def build_dataloader(patient_ids_subset, shuffle=False):
    subset_targets = [cases_df.iloc[patient_id_to_idx[pid]]['Followup FVC Volume L'] for pid in patient_ids_subset]

    dataset = FixedSizeDataset(
        patient_ids=patient_ids_subset,
        targets=subset_targets,
        target_shape=targets_shape
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2
    )
    return dataset, loader


print(f'Target shape for volumes: {targets_shape}')
batch_size = 4
print(f"Batch size set to {batch_size}.")
print(f"Using predefined split seed: {seed_to_use}")
print(f"Using fold: {fold_to_use}")
print(f"Split directory: {split_dir}")

output_dir = os.path.join(
    'image_only_head_runs',
    f'seed{seed_to_use}_fold{fold_to_use}_{timestamp}'
)
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory: {output_dir}")

fold_idx_zero_based = fold_to_use - 1
train_patient_ids, val_patient_ids, train_fold_path, val_fold_path = load_fold_patient_ids(
    seed_to_use,
    fold_idx_zero_based
)

missing_train_ids = [pid for pid in train_patient_ids if pid not in patient_id_to_idx]
missing_val_ids = [pid for pid in val_patient_ids if pid not in patient_id_to_idx]
if missing_train_ids:
    print(
        f"Warning: {len(missing_train_ids)} train IDs from {train_fold_path} were not found in cases_df and will be skipped."
    )
if missing_val_ids:
    print(
        f"Warning: {len(missing_val_ids)} validation IDs from {val_fold_path} were not found in cases_df and will be skipped."
    )

train_patient_ids = [pid for pid in train_patient_ids if pid in patient_id_to_idx]
val_patient_ids = [pid for pid in val_patient_ids if pid in patient_id_to_idx]

if len(train_patient_ids) == 0:
    raise ValueError(f'No valid train IDs available for seed {seed_to_use} fold {fold_to_use}.')
if len(val_patient_ids) == 0:
    raise ValueError(f'No valid validation IDs available for seed {seed_to_use} fold {fold_to_use}.')

train_dataset, train_loader = build_dataloader(train_patient_ids, shuffle=True)
val_dataset, val_loader = build_dataloader(val_patient_ids, shuffle=False)

print(f'Train fold file: {train_fold_path}')
print(f'Validation fold file: {val_fold_path}')
print(f'Training dataset created with {len(train_dataset)} samples.')
print(f'Validation dataset created with {len(val_dataset)} samples.')
print(f"Training set size: {len(train_patient_ids)}")
print(f"Validation set size: {len(val_patient_ids)}")

# Hyperparameters
lr = 1e-03
weight_decay = 1e-7
early_stopping_patience = 15
epochs = 100
cnn_output_dim = 512
hidden_dim = 128

class ImageOnlyModel(nn.Module):
    def __init__(self, cnn_backbone, cnn_output_dim=512, hidden_dim=128):
        super(ImageOnlyModel, self).__init__()
        
        # 3D CNN backbone for processing images
        self.cnn_backbone = cnn_backbone
        
        # Remove the final FC layer from the backbone
        self.cnn_backbone.fc = nn.Identity()

        # Combined layer
        self.combined_fc = nn.Sequential(
            nn.Linear(cnn_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),  # Reduced from 0.3 to avoid underfitting
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),  # Reduced from 0.2 to avoid underfitting
            nn.Linear(hidden_dim // 2, 1)
        )

        # Head-only training: freeze CNN backbone parameters
        self.freeze_backbone = True
        for param in self.cnn_backbone.parameters():
            param.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep backbone in eval mode when training head only
        if self.freeze_backbone:
            self.cnn_backbone.eval()
        return self
        

    def forward(self, image):
        # Process image through CNN
        if self.freeze_backbone:
            with torch.no_grad():
                cnn_features = self.cnn_backbone(image)
        else:
            cnn_features = self.cnn_backbone(image)

        # Flatten if needed
        if cnn_features.dim() > 2:
            cnn_features = cnn_features.view(cnn_features.size(0), -1)

        # Output
        return self.combined_fc(cnn_features)

# Move model to device and ensure correct data type
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

## Define Model
backbone = models.resnet18(weights='IMAGENET1K_V1')  
backbone.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  
backbone_3d = ACSConverter(backbone).to(device)
model_3d = ImageOnlyModel(backbone_3d, cnn_output_dim=cnn_output_dim, hidden_dim=hidden_dim)

model_3d = model_3d.type(torch.FloatTensor).to(device)

## Training Loop (simplified)
#optimizer = optim.SGD(model_3d.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
head_parameters = [p for p in model_3d.combined_fc.parameters() if p.requires_grad]
optimizer = optim.Adam(head_parameters, lr=lr, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=7)

criterion = nn.MSELoss()

# ============================================================================
# TRAINING CONFIGURATION SUMMARY
# ============================================================================
print("\n" + "="*80)
print("TRAINING CONFIGURATION SUMMARY")
print("="*80)
print(f"\nRun Timestamp: {timestamp}")
print(f"\nDATA CONFIGURATION:")
print(f"  Total Patients: {len(patients_ids)}")
print(f"  Split Seed: {seed_to_use}")
print(f"  Fold: {fold_to_use}")
print(f"  Train Fold File: {train_fold_path}")
print(f"  Validation Fold File: {val_fold_path}")
print(f"  Training Samples: {len(train_patient_ids)}")
print(f"  Validation Samples: {len(val_patient_ids)}")
print(f"  Target Volume Shape (H, W, D): {targets_shape}")
print(f"\nMODEL ARCHITECTURE:")
print(f"  Backbone: ResNet18 + ACS 3D Conversion")
print(f"  CNN Output Dimension: {cnn_output_dim}")
print(f"  Input Type: Image only")
print(f"  Output Target: Followup FVC Volume L")
print(f"  Hidden Dimension: {hidden_dim}")
print(f"\nTRAINING HYPERPARAMETERS:")
print(f"  Batch Size: {batch_size}")
print(f"  Learning Rate: {lr}")
print(f"  Weight Decay: {weight_decay}")
print(f"  Optimizer: Adam")
print(f"  Loss Function: MSE")
print(f"  Max Epochs: {epochs}")
print(f"  Early Stopping Patience: {early_stopping_patience}")
print(f"  Learning Rate Scheduler: ReduceLROnPlateau (patience=7, factor=0.6)")
print(f"\nCOMPUTE:")
print(f"  Device: {device}")
print(f"  DataLoader Workers: 2")
print(f"  Output Directory: {output_dir}")
print("="*80 + "\n")
print(f"Model summary:")
print(model_3d)

# add early stopping mechanism
best_val_loss = float('inf')
epochs_no_improve = 0
best_model_path = os.path.join(
    output_dir,
    f'best_model_seed{seed_to_use}_fold{fold_to_use}_lr_image_only_head_bs{batch_size}_pt{len(patients_ids)}_{timestamp}.pth'
)

print("Starting Training Loop...") 
loss_history = {'train_loss': [], 'val_loss': []}

for epoch in range(epochs):  
    # Training phase
    model_3d.train()
    train_loss = 0.0
    
    for i, (images, targets) in enumerate(train_loader):
        # Clear gradients
        optimizer.zero_grad()
        
        # Move data to device and ensure correct data type
        images = images.type(torch.FloatTensor).to(device, non_blocking=True)
        targets = targets.type(torch.FloatTensor).to(device, non_blocking=True)
        
        # Forward pass
        outputs = model_3d(images)
        loss = criterion(outputs, targets.unsqueeze(1))
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Accumulate loss
        train_loss += loss.item()
        
        # Clear GPU cache periodically
        if i % 10 == 0:
            torch.cuda.empty_cache()
        
        # Delete tensors to free memory
        del images, targets, outputs, loss
    
    # Validation phase
    model_3d.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for i, (images, targets) in enumerate(val_loader):
            # Move data to device and ensure correct data type
            images = images.type(torch.FloatTensor).to(device, non_blocking=True)
            targets = targets.type(torch.FloatTensor).to(device, non_blocking=True)
            
            outputs = model_3d(images)
            loss = criterion(outputs, targets.unsqueeze(1))
            
            val_loss += loss.item()
            
            # Clear GPU cache periodically
            if i % 10 == 0:
                torch.cuda.empty_cache()
            
            del images, targets, outputs, loss

    # Calculate average losses
    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    
    # Store loss history
    loss_history['train_loss'].append(avg_train_loss)
    loss_history['val_loss'].append(avg_val_loss)
    
    current_lr = optimizer.param_groups[0]['lr']
    
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, LR: {current_lr:.2e}")
    
    # Step the scheduler
    scheduler.step(avg_val_loss)
    
    # Early stopping check
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        # Save best model
        torch.save(model_3d.state_dict(), best_model_path)
        print(f"New best model saved with validation loss: {best_val_loss:.6f}")
    else:
        epochs_no_improve += 1
    
    if epochs_no_improve >= early_stopping_patience:
        print(f"Early stopping triggered after {epoch+1} epochs")
        break
    
    # Clear cache at end of each epoch
    torch.cuda.empty_cache()

print("Training complete.")
print(f"Best validation loss: {best_val_loss:.6f}")

#Evaluate on validation with different metrics for comparison
print("Evaluating model on validation set...")
# Load best model and ensure correct data type
model_3d.load_state_dict(torch.load(best_model_path, map_location=device))
model_3d = model_3d.type(torch.FloatTensor).to(device)
model_3d.eval()
all_predictions = []
all_targets = []

with torch.no_grad():
    for images, targets in val_loader:
        # Move data to device and ensure correct data type
        images = images.type(torch.FloatTensor).to(device, non_blocking=True)
        targets = targets.type(torch.FloatTensor).to(device, non_blocking=True)
        
        outputs = model_3d(images)
        
        predictions = outputs.squeeze().cpu().numpy()
        targets_np = targets.cpu().numpy()
        
        if predictions.ndim == 0:
            predictions = np.array([predictions])
        if targets_np.ndim == 0:
            targets_np = np.array([targets_np])
            
        all_predictions.extend(predictions)
        all_targets.extend(targets_np)
        
        del images, targets, outputs

# Convert to numpy arrays
all_predictions = np.array(all_predictions)
all_targets = np.array(all_targets)

# Calculate metrics
mse = mean_squared_error(all_targets, all_predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(all_targets, all_predictions)
r2 = r2_score(all_targets, all_predictions)

print("\nValidation Metrics:")
print(f"MSE: {mse:.6f}")
print(f"RMSE: {rmse:.6f}")
print(f"MAE: {mae:.6f}")
print(f"R² Score: {r2:.6f}")

# Save evaluation results with configuration
eval_results = {
    'Timestamp': [timestamp],
    'Seed': [seed_to_use],
    'Fold': [fold_to_use],
    'Batch_Size': [batch_size],
    'Learning_Rate': [lr],
    'Weight_Decay': [weight_decay],
    'Early_Stopping_Patience': [early_stopping_patience],
    'Total_Patients': [len(patients_ids)],
    'Train_Samples': [len(train_patient_ids)],
    'Val_Samples': [len(val_patient_ids)],
    'MSE': [mse],
    'RMSE': [rmse],
    'MAE': [mae],
    'R2_Score': [r2],
    'Best_Val_Loss': [best_val_loss],
    'Total_Epochs': [len(loss_history['train_loss'])],
    'CNN_Output_Dim': [cnn_output_dim],
    'Hidden_Dim': [hidden_dim],
    'Best_Model_Path': [best_model_path]
}
eval_df = pd.DataFrame(eval_results)
eval_metrics_path = os.path.join(
    output_dir,
    f'evaluation_metrics_seed{seed_to_use}_fold{fold_to_use}_image_only_head_bs{batch_size}_pt{len(patients_ids)}_{timestamp}.csv'
)
eval_df.to_csv(eval_metrics_path, index=False)
print(f"\nEvaluation metrics saved to: {eval_metrics_path}")

# Save patient split information in a separate file
patient_split_data = []
for pid in train_patient_ids:
    patient_split_data.append({'PatientID': pid, 'Split': 'train'})
for pid in val_patient_ids:
    patient_split_data.append({'PatientID': pid, 'Split': 'validation'})
patient_split_df = pd.DataFrame(patient_split_data)
patient_split_path = os.path.join(
    output_dir,
    f'patient_split_seed{seed_to_use}_fold{fold_to_use}_image_only_head_bs{batch_size}_pt{len(patients_ids)}_{timestamp}.csv'
)
patient_split_df.to_csv(patient_split_path, index=False)
print(f"Patient split information saved to: {patient_split_path}")

#save loss history to a csv file
# Add epoch numbers to the existing loss_history dictionary
loss_history['epoch'] = list(range(1, len(loss_history['train_loss']) + 1))
loss_df = pd.DataFrame(loss_history)
loss_history_path = os.path.join(
    output_dir,
    f'loss_history_seed{seed_to_use}_fold{fold_to_use}_image_only_head_bs{batch_size}_pt{len(patients_ids)}_{timestamp}.csv'
)
loss_df.to_csv(loss_history_path, index=False)
print(f"Loss history saved to: {loss_history_path}")

# plot training and validation loss curves and save to a png file
# hyperparameters in legend
import matplotlib.pyplot as plt

model_name = os.path.basename(best_model_path)

plt.figure(figsize=(10, 6))
plt.plot(loss_history['epoch'], loss_history['train_loss'], label='Train Loss')
plt.plot(loss_history['epoch'], loss_history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f'Training and Validation Loss Curves - Image-Only Model\nSeed {seed_to_use} Fold {fold_to_use} | Model: {model_name}\nBS: {batch_size}, LR: {lr}, WD: {weight_decay}, ES Patience: {early_stopping_patience}, Samples: {len(train_patient_ids)}/{len(val_patient_ids)}')
plt.legend()
plt.grid()
loss_curve_path = os.path.join(
    output_dir,
    f'loss_curves_seed{seed_to_use}_fold{fold_to_use}_image_only_head_bs{batch_size}_pt{len(patients_ids)}_{timestamp}.png'
)
plt.savefig(loss_curve_path)
plt.close()

# plot prediction vs target scatter plot and save to a png file
plt.figure(figsize=(8, 8))
plt.scatter(all_targets, all_predictions, alpha=0.7)
plt.plot([all_targets.min(), all_targets.max()], [all_targets.min(), all_targets.max()], 'r--')  # Line for perfect predictions
plt.xlabel('True FVC Volume L')
plt.ylabel('Predicted FVC Volume L')
plt.title(f'Predicted vs True FVC Volume L - Image-Only Model\nSeed {seed_to_use} Fold {fold_to_use} | Model: {model_name}\nMSE: {mse:.6f}, RMSE: {rmse:.6f}, MAE: {mae:.6f}, R²: {r2:.6f}')
plt.grid()
pred_scatter_path = os.path.join(
    output_dir,
    f'predicted_vs_true_seed{seed_to_use}_fold{fold_to_use}_image_only_head_bs{batch_size}_pt{len(patients_ids)}_{timestamp}.png'
)
plt.savefig(pred_scatter_path)
plt.close() 

print(f"Predicted vs True scatter plot saved to: {pred_scatter_path}")
print(f"Loss curves plot saved to: {loss_curve_path}")
print(f"\nModel saved to: {model_name}")