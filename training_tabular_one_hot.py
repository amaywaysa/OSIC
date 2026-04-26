import numpy as np
import pandas as pd
import os
import json
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


pd.set_option('display.max_columns', None)
plt.style.use('default')
#sns.set_palette("husl") 
# set color to '#5F8FDC'
#sns.set_color_codes("pastel")

plt.rcParams.update({
    "font.family": "serif",      # serif to match LaTeX article style
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

# 1. Load data
cases_df = pd.read_csv('cleaned_collected_cases_final_with_FEV1.csv')

# 2. Prepare categorical features with OneHotEncoder
enc_diagnosis = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
enc_sex = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
enc_smoking = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

diagnosis_encoded = enc_diagnosis.fit_transform(cases_df[['Primary Diagnosis']].fillna('Unknown'))
sex_encoded = enc_sex.fit_transform(cases_df[['Sex']].fillna('Unknown'))
smoking_encoded = enc_smoking.fit_transform(cases_df[['Smoking History']].fillna('Unknown'))

# Store encoded features in dataframe for traceability
for i, cat in enumerate(enc_diagnosis.categories_[0]):
    cases_df[f'Primary Diagnosis_{cat}'] = diagnosis_encoded[:, i]
for i, cat in enumerate(enc_sex.categories_[0]):
    cases_df[f'Sex_{cat}'] = sex_encoded[:, i]
for i, cat in enumerate(enc_smoking.categories_[0]):
    cases_df[f'Smoking History_{cat}'] = smoking_encoded[:, i]

print(f"Primary Diagnosis classes: {list(enc_diagnosis.categories_[0])}")
print(f"Sex classes: {list(enc_sex.categories_[0])}")
print(f"Smoking History classes: {list(enc_smoking.categories_[0])}")

n_diagnosis_cats = len(enc_diagnosis.categories_[0])
n_sex_cats = len(enc_sex.categories_[0])
n_smoking_cats = len(enc_smoking.categories_[0])
total_categorical_dims = n_diagnosis_cats + n_sex_cats + n_smoking_cats
print(
    f"Total categorical dimensions: {total_categorical_dims} "
    f"(diagnosis: {n_diagnosis_cats}, sex: {n_sex_cats}, smoking: {n_smoking_cats})"
)

timestamp = pd.to_datetime("today").strftime("%Y-%m-%d_%H-%M-%S")
print(f"Data loaded. Total cases: {len(cases_df)} at {timestamp}.")

# 3. Build tabular arrays
baseline_fvc = cases_df['Baseline FVC Volume L'].to_numpy(dtype=np.float32).reshape(-1, 1)
fev1_missing_count = cases_df['FEV1 Volume L'].isna().sum()
if fev1_missing_count > 0:
    fev1_median = cases_df['FEV1 Volume L'].median()
    cases_df['FEV1 Volume L'] = cases_df['FEV1 Volume L'].fillna(fev1_median)
    print(f"Imputed {fev1_missing_count} missing FEV1 values using median: {fev1_median:.4f}")
else:
    print("No missing FEV1 values found.")

fev1 = cases_df['FEV1 Volume L'].to_numpy(dtype=np.float32).reshape(-1, 1)
age = cases_df['Age'].to_numpy(dtype=np.float32).reshape(-1, 1)
y_all = cases_df['Followup FVC Volume L'].to_numpy(dtype=np.float32)

X_all = np.concatenate([baseline_fvc, fev1, age, diagnosis_encoded, sex_encoded, smoking_encoded], axis=1).astype(np.float32)
continuous_feature_indices = [0, 1, 2]  # Baseline FVC, FEV1 Volume L, Age

patient_ids = cases_df['PatientID'].tolist()
patient_ids_str = [str(pid) for pid in patient_ids]
patient_id_to_idx = {pid: idx for idx, pid in enumerate(patient_ids_str)}

# 4. Config
batch_size = 16
split_seeds = [42]#, 24, 7] #, 100, 123]
num_folds = 5
total_folds = len(split_seeds) * num_folds
output_dir = f"cv_tabular_one_hot_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

lr = 1e-3
weight_decay = 1e-6
early_stopping_patience = 20
epochs = 200
hidden_dim = 128
dropout = 0.2
input_dim = X_all.shape[1]

# Hyperparameter search config (runs before the main repeated CV)
search_n_splits = 3
search_epochs = 80
search_patience = 18
max_search_configs = 100
param_grid = {
    'batch_size': [4],#, 8, 16, 32],
    'lr': [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
    'weight_decay': [1e-7, 0, 1e-3, 1e-1, 1e-5],
    'hidden_dim': [ 96, 128, 256, 384, 512],
    'dropout': [0, 0.1, 0.2]#, 0.3] 0.4]
}

print(f"Using predefined split seeds: {split_seeds}")
print(f"Folds per seed: {num_folds}")
print(f"Total folds across all seeds: {total_folds}")
print(f"Output directory: {output_dir}")
print(f"Input dimension: {input_dim}")

bootstrap_n_resamples = 1000
bootstrap_ci_alpha = 0.95


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


def scale_continuous_by_train(train_features, val_features, continuous_indices, scaler=None):
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(train_features[:, continuous_indices])

    train_scaled = train_features.copy()
    val_scaled = val_features.copy()

    train_scaled[:, continuous_indices] = scaler.transform(train_features[:, continuous_indices])
    val_scaled[:, continuous_indices] = scaler.transform(val_features[:, continuous_indices])

    return train_scaled.astype(np.float32), val_scaled.astype(np.float32), scaler


def bootstrap_metric_cis(y_true, y_pred, n_resamples=1000, alpha=0.95, seed=42):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    n = len(y_true)
    if n == 0:
        raise ValueError("Cannot compute bootstrap CI on empty arrays.")

    mse_samples = []
    rmse_samples = []
    mae_samples = []
    r2_samples = []

    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        yb_true = y_true[idx]
        yb_pred = y_pred[idx]

        mse_b = mean_squared_error(yb_true, yb_pred)
        rmse_b = np.sqrt(mse_b)
        mae_b = mean_absolute_error(yb_true, yb_pred)

        mse_samples.append(mse_b)
        rmse_samples.append(rmse_b)
        mae_samples.append(mae_b)

        # R2 can be undefined for degenerate bootstrap samples; keep as NaN and ignore in percentiles.
        try:
            r2_b = r2_score(yb_true, yb_pred)
        except ValueError:
            r2_b = np.nan
        r2_samples.append(r2_b)

    lower_q = (1.0 - alpha) / 2.0
    upper_q = 1.0 - lower_q

    return {
        'MSE_CI_Lower': float(np.quantile(mse_samples, lower_q)),
        'MSE_CI_Upper': float(np.quantile(mse_samples, upper_q)),
        'RMSE_CI_Lower': float(np.quantile(rmse_samples, lower_q)),
        'RMSE_CI_Upper': float(np.quantile(rmse_samples, upper_q)),
        'MAE_CI_Lower': float(np.quantile(mae_samples, lower_q)),
        'MAE_CI_Upper': float(np.quantile(mae_samples, upper_q)),
        'R2_CI_Lower': float(np.nanquantile(r2_samples, lower_q)),
        'R2_CI_Upper': float(np.nanquantile(r2_samples, upper_q))
    }

feature_names = ['Baseline FVC', 'FEV1 Volume L', 'Age'] + \
                [f'Primary Diagnosis_{cat}' for cat in enc_diagnosis.categories_[0]] + \
                [f'Sex_{cat}' for cat in enc_sex.categories_[0]] + \
                [f'Smoking Status_{cat}' for cat in enc_smoking.categories_[0]]


class TabularDataset(torch.utils.data.Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.targets[idx]

        if torch.any(torch.isnan(x)) or torch.any(torch.isnan(y)):
            x = torch.nan_to_num(x)
            y = torch.nan_to_num(y)

        return x, y

    def __len__(self):
        return len(self.targets)


class TabularMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        return self.net(x)


# 5. Training setup summary
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# print("\n" + "=" * 80)
# print("HYPERPARAMETER SEARCH (PRE-CV)")
# print("=" * 80)

# all_search_configs = list(ParameterGrid(param_grid))
# if max_search_configs is not None and len(all_search_configs) > max_search_configs:
#     rng = np.random.default_rng(42)
#     chosen_indices = rng.choice(len(all_search_configs), size=max_search_configs, replace=False)
#     search_configs = [all_search_configs[i] for i in chosen_indices]
# else:
#     search_configs = all_search_configs

# print(f"Candidate configurations available: {len(all_search_configs)}")
# print(f"Candidate configurations evaluated: {len(search_configs)}")
# print(
#     f"Inner CV for search: KFold(n_splits={search_n_splits}, shuffle=True, random_state=42), "
#     f"max epochs={search_epochs}, early stopping patience={search_patience}"
# )

# search_kf = KFold(n_splits=search_n_splits, shuffle=True, random_state=42)
# search_results = []
# best_search_loss = float('inf')
# best_params = None

# for config_idx, config in enumerate(search_configs, start=1):
#     print("\n" + "-" * 80)
#     print(f"Search config {config_idx}/{len(search_configs)}: {config}")
#     print("-" * 80)

#     fold_best_losses = []
#     fold_best_mse = []
#     fold_best_r2 = []

#     for inner_fold_idx, (inner_train_idx, inner_val_idx) in enumerate(search_kf.split(X_all), start=1):
#         X_inner_train, X_inner_val = X_all[inner_train_idx], X_all[inner_val_idx]
#         y_inner_train, y_inner_val = y_all[inner_train_idx], y_all[inner_val_idx]

#         X_inner_train, X_inner_val, _ = scale_continuous_by_train(
#             X_inner_train,
#             X_inner_val,
#             continuous_feature_indices
#         )

#         inner_train_dataset = TabularDataset(X_inner_train, y_inner_train)
#         inner_val_dataset = TabularDataset(X_inner_val, y_inner_val)

#         inner_train_loader = torch.utils.data.DataLoader(
#             inner_train_dataset,
#             batch_size=config['batch_size'],
#             shuffle=True,
#             num_workers=0
#         )
#         inner_val_loader = torch.utils.data.DataLoader(
#             inner_val_dataset,
#             batch_size=config['batch_size'],
#             shuffle=False,
#             num_workers=0
#         )

#         inner_model = TabularMLP(
#             input_dim=input_dim,
#             hidden_dim=config['hidden_dim'],
#             dropout=config['dropout']
#         ).to(device)
#         inner_model = inner_model.type(torch.FloatTensor).to(device)

#         inner_optimizer = optim.Adam(
#             inner_model.parameters(),
#             lr=config['lr'],
#             weight_decay=config['weight_decay']
#         )
#         inner_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#             inner_optimizer,
#             mode='min',
#             factor=0.6,
#             patience=5
#         )
#         inner_criterion = nn.MSELoss()

#         inner_best_val_loss = float('inf')
#         inner_best_state_dict = None
#         inner_no_improve = 0

#         for _ in range(search_epochs):
#             inner_model.train()
#             for features, targets in inner_train_loader:
#                 inner_optimizer.zero_grad()

#                 features = features.type(torch.FloatTensor).to(device, non_blocking=True)
#                 targets = targets.type(torch.FloatTensor).to(device, non_blocking=True)

#                 outputs = inner_model(features)
#                 loss = inner_criterion(outputs, targets.unsqueeze(1))
#                 loss.backward()
#                 inner_optimizer.step()

#                 del features, targets, outputs, loss

#             inner_model.eval()
#             val_loss_total = 0.0
#             with torch.no_grad():
#                 for features, targets in inner_val_loader:
#                     features = features.type(torch.FloatTensor).to(device, non_blocking=True)
#                     targets = targets.type(torch.FloatTensor).to(device, non_blocking=True)

#                     outputs = inner_model(features)
#                     loss = inner_criterion(outputs, targets.unsqueeze(1))
#                     val_loss_total += loss.item()

#                     del features, targets, outputs, loss

#             avg_inner_val_loss = val_loss_total / len(inner_val_loader)
#             inner_scheduler.step(avg_inner_val_loss)

#             if avg_inner_val_loss < inner_best_val_loss:
#                 inner_best_val_loss = avg_inner_val_loss
#                 inner_best_state_dict = copy.deepcopy(inner_model.state_dict())
#                 inner_no_improve = 0
#             else:
#                 inner_no_improve += 1

#             if inner_no_improve >= search_patience:
#                 break

#         if inner_best_state_dict is not None:
#             inner_model.load_state_dict(inner_best_state_dict)

#         inner_model.eval()
#         inner_fold_predictions = []
#         inner_fold_targets = []
#         with torch.no_grad():
#             for features, targets in inner_val_loader:
#                 features = features.type(torch.FloatTensor).to(device, non_blocking=True)
#                 targets = targets.type(torch.FloatTensor).to(device, non_blocking=True)

#                 outputs = inner_model(features)
#                 predictions = outputs.squeeze().cpu().numpy()
#                 targets_np = targets.cpu().numpy()

#                 if predictions.ndim == 0:
#                     predictions = np.array([predictions])
#                 if targets_np.ndim == 0:
#                     targets_np = np.array([targets_np])

#                 inner_fold_predictions.extend(predictions)
#                 inner_fold_targets.extend(targets_np)

#                 del features, targets, outputs

#         inner_fold_predictions = np.array(inner_fold_predictions)
#         inner_fold_targets = np.array(inner_fold_targets)
#         inner_mse = mean_squared_error(inner_fold_targets, inner_fold_predictions)
#         inner_r2 = r2_score(inner_fold_targets, inner_fold_predictions)

#         fold_best_losses.append(inner_best_val_loss)
#         fold_best_mse.append(inner_mse)
#         fold_best_r2.append(inner_r2)
#         print(
#             f"  Inner fold {inner_fold_idx}/{search_n_splits} best val loss: "
#             f"{inner_best_val_loss:.6f}, MSE: {inner_mse:.6f}, R²: {inner_r2:.6f}"
#         )

#         del inner_model

#     mean_config_loss = float(np.mean(fold_best_losses))
#     std_config_loss = float(np.std(fold_best_losses))
#     mean_config_mse = float(np.mean(fold_best_mse))
#     std_config_mse = float(np.std(fold_best_mse))
#     mean_config_r2 = float(np.mean(fold_best_r2))
#     std_config_r2 = float(np.std(fold_best_r2))

#     search_results.append({
#         'Config_Index': config_idx,
#         'Batch_Size': config['batch_size'],
#         'Learning_Rate': config['lr'],
#         'Weight_Decay': config['weight_decay'],
#         'Hidden_Dim': config['hidden_dim'],
#         'Dropout': config['dropout'],
#         'Inner_Best_Val_Loss_Mean': mean_config_loss,
#         'Inner_Best_Val_Loss_Std': std_config_loss,
#         'Inner_Best_Val_MSE_Mean': mean_config_mse,
#         'Inner_Best_Val_MSE_Std': std_config_mse,
#         'Inner_Best_Val_R2_Mean': mean_config_r2,
#         'Inner_Best_Val_R2_Std': std_config_r2
#     })

#     print(
#         f"Config {config_idx} summary -> Mean best val loss: {mean_config_loss:.6f}, "
#         f"Std: {std_config_loss:.6f}, Mean MSE: {mean_config_mse:.6f}, Mean R²: {mean_config_r2:.6f}"
#     )

#     if mean_config_loss < best_search_loss:
#         best_search_loss = mean_config_loss
#         best_params = config.copy()
#         print(f"New best config found with mean val loss: {best_search_loss:.6f}")

# if best_params is None:
#     raise RuntimeError("Hyperparameter search did not produce any valid configuration.")

# search_results_df = pd.DataFrame(search_results).sort_values('Inner_Best_Val_Loss_Mean')
# search_results_path = os.path.join(
#     output_dir,
#     f'hyperparameter_search_results_tabular_one_hot_pt{len(patient_ids)}_{timestamp}.csv'
# )
# search_results_df.to_csv(search_results_path, index=False)

# # Plot hyperparameter value vs performance (MSE and R²) aggregated across evaluated configs.
# for hyperparam in ['batch_size', 'lr', 'weight_decay', 'hidden_dim', 'dropout']:
#     hp_col = {
#         'batch_size': 'Batch_Size',
#         'lr': 'Learning_Rate',
#         'weight_decay': 'Weight_Decay',
#         'hidden_dim': 'Hidden_Dim',
#         'dropout': 'Dropout'
#     }[hyperparam]

#     hp_perf_df = (
#         search_results_df[[hp_col, 'Inner_Best_Val_MSE_Mean', 'Inner_Best_Val_R2_Mean']]
#         .groupby(hp_col, as_index=False)
#         .mean()
#         .sort_values(hp_col)
#     )

#     plt.figure(figsize=(8, 5))
#     plt.plot(hp_perf_df[hp_col], hp_perf_df['Inner_Best_Val_MSE_Mean'], marker='o')
#     plt.xlabel(hp_col)
#     plt.ylabel('Mean Inner-Fold Best MSE')
#     plt.title(f'Hyperparameter vs MSE: {hp_col}')
#     plt.grid(True, alpha=0.4)
#     plt.tight_layout()
#     hp_mse_plot_path = os.path.join(
#         output_dir,
#         f'hyperparam_vs_mse_{hyperparam}_tabular_one_hot_pt{len(patient_ids)}_{timestamp}.png'
#     )
#     plt.savefig(hp_mse_plot_path, dpi=200)
#     plt.close()

#     plt.figure(figsize=(8, 5))
#     plt.plot(hp_perf_df[hp_col], hp_perf_df['Inner_Best_Val_R2_Mean'], marker='o')
#     plt.xlabel(hp_col)
#     plt.ylabel('Mean Inner-Fold Best R²')
#     plt.title(f'Hyperparameter vs R²: {hp_col}')
#     plt.grid(True, alpha=0.4)
#     plt.tight_layout()
#     hp_r2_plot_path = os.path.join(
#         output_dir,
#         f'hyperparam_vs_r2_{hyperparam}_tabular_one_hot_pt{len(patient_ids)}_{timestamp}.png'
#     )
#     plt.savefig(hp_r2_plot_path, dpi=200)
#     plt.close()

#     print(f"Hyperparameter-vs-MSE plot saved to: {hp_mse_plot_path}")
#     print(f"Hyperparameter-vs-R² plot saved to: {hp_r2_plot_path}")


# def plot_interaction_curves(
#     df,
#     x_col,
#     group_col,
#     metric_col,
#     metric_label,
#     title,
#     output_path,
#     x_is_log=False,
#     group_sort_numeric=False
# ):
#     interaction_df = df[[x_col, group_col, metric_col]].copy()
#     grouped = interaction_df.groupby([group_col, x_col], as_index=False).mean()

#     unique_groups = grouped[group_col].drop_duplicates().tolist()
#     if group_sort_numeric:
#         unique_groups = sorted(unique_groups)

#     plt.figure(figsize=(9, 6))
#     for group_value in unique_groups:
#         group_slice = grouped[grouped[group_col] == group_value].sort_values(x_col)
#         if len(group_slice) < 2:
#             continue

#         plt.plot(
#             group_slice[x_col],
#             group_slice[metric_col],
#             marker='o',
#             linewidth=1.8,
#             label=f"{group_col}={group_value}"
#         )

#     if x_is_log:
#         plt.xscale('log')

#     plt.xlabel(x_col)
#     plt.ylabel(metric_label)
#     plt.title(title)
#     plt.grid(True, alpha=0.35)
#     plt.legend(ncol=2, frameon=True)
#     plt.tight_layout()
#     plt.savefig(output_path, dpi=220)
#     plt.close()


# interaction_specs = [
#     ('Hidden_Dim', 'Batch_Size', True, True),
#     ('Learning_Rate', 'Batch_Size', True, True),
#     ('Dropout', 'Batch_Size', False, True),
#     ('Weight_Decay', 'Batch_Size', True, True),
#     ('Hidden_Dim', 'Dropout', True, False),
#     ('Learning_Rate', 'Hidden_Dim', True, True)
# ]

# metric_specs = [
#     ('Inner_Best_Val_MSE_Mean', 'Mean Inner-Fold Best MSE', 'mse'),
#     ('Inner_Best_Val_R2_Mean', 'Mean Inner-Fold Best R²', 'r2')
# ]

# for x_col, group_col, x_is_log, group_sort_numeric in interaction_specs:
#     for metric_col, metric_label, metric_slug in metric_specs:
#         interaction_plot_path = os.path.join(
#             output_dir,
#             f'hyperparam_interaction_{metric_slug}_{x_col}_by_{group_col}_tabular_one_hot_pt{len(patient_ids)}_{timestamp}.png'
#         )
#         plot_interaction_curves(
#             df=search_results_df,
#             x_col=x_col,
#             group_col=group_col,
#             metric_col=metric_col,
#             metric_label=metric_label,
#             title=f'{metric_label}: {x_col} with curves by {group_col}',
#             output_path=interaction_plot_path,
#             x_is_log=x_is_log,
#             group_sort_numeric=group_sort_numeric
#         )
#         print(f"Hyperparameter interaction plot saved to: {interaction_plot_path}")


# # Heatmaps: each pair of hyperparameters, one heatmap per metric.
# # Cells show the mean metric value across configs that share those two hyperparameter values.
# heatmap_pairs = [
#     ('Hidden_Dim', 'Batch_Size'),
#     ('Hidden_Dim', 'Dropout'),
#     ('Hidden_Dim', 'Learning_Rate'),
#     ('Dropout', 'Batch_Size'),
#     ('Learning_Rate', 'Batch_Size'),
#     ('Weight_Decay', 'Learning_Rate'),
# ]

# for metric_col, metric_label, metric_slug in metric_specs:
#     for row_col, col_col in heatmap_pairs:
#         pivot = (
#             search_results_df
#             .groupby([row_col, col_col], as_index=False)[metric_col]
#             .mean()
#             .pivot(index=row_col, columns=col_col, values=metric_col)
#         )

#         if pivot.empty:
#             continue

#         fig, ax = plt.subplots(figsize=(max(6, pivot.shape[1] * 1.2), max(4, pivot.shape[0] * 0.9)))
#         im = ax.imshow(pivot.values, aspect='auto', cmap='viridis_r' if 'MSE' in metric_label else 'viridis')

#         ax.set_xticks(range(pivot.shape[1]))
#         ax.set_xticklabels([str(v) for v in pivot.columns], rotation=45, ha='right')
#         ax.set_yticks(range(pivot.shape[0]))
#         ax.set_yticklabels([str(v) for v in pivot.index])
#         ax.set_xlabel(col_col)
#         ax.set_ylabel(row_col)
#         ax.set_title(f'{metric_label}: {row_col} × {col_col}')

#         # Write the numeric value in each cell.
#         for i in range(pivot.shape[0]):
#             for j in range(pivot.shape[1]):
#                 val = pivot.values[i, j]
#                 if not np.isnan(val):
#                     ax.text(j, i, f'{val:.4f}', ha='center', va='center',
#                             fontsize=7, color='white')

#         plt.colorbar(im, ax=ax, label=metric_label)
#         plt.tight_layout()

#         heatmap_path = os.path.join(
#             output_dir,
#             f'hyperparam_heatmap_{metric_slug}_{row_col}_x_{col_col}_tabular_one_hot_pt{len(patient_ids)}_{timestamp}.png'
#         )
#         plt.savefig(heatmap_path, dpi=220)
#         plt.close()
#         print(f"Hyperparameter heatmap saved to: {heatmap_path}")


# best_params_path = os.path.join(
#     output_dir,
#     f'best_hyperparameters_tabular_one_hot_pt{len(patient_ids)}_{timestamp}.json'
# )
# with open(best_params_path, 'w') as f:
#     json.dump(best_params, f, indent=2)

# # Use best hyperparameters for the main repeated CV stage
# batch_size = int(best_params['batch_size'])
# lr = float(best_params['lr'])
# weight_decay = float(best_params['weight_decay'])
# hidden_dim = int(best_params['hidden_dim'])
# dropout = float(best_params['dropout'])

# print("\nHyperparameter search complete.")
# print(f"Best params: {best_params}")
# print(f"Best mean inner-fold val loss: {best_search_loss:.6f}")
# print(f"Search results saved to: {search_results_path}")
# print(f"Best hyperparameters saved to: {best_params_path}")

# print("\n" + "=" * 80)
# print("TRAINING CONFIGURATION SUMMARY")
# print("=" * 80)
# print(f"\nRun Timestamp: {timestamp}")
# print("\nDATA CONFIGURATION:")
# print(f"  Total Patients: {len(patient_ids)}")
# print(f"  Seed Splits: {split_seeds}")
# print(f"  Folds per Seed: {num_folds}")
# print(f"  Total Folds: {total_folds}")
# print(f"  Input Features: {input_dim}")
# print("\nMODEL ARCHITECTURE:")
# print("  Model: Tabular MLP")
# print(
#     f"  Input Features: Baseline FVC, FEV1 Volume L, Age, One-Hot "
#     f"(Diagnosis={n_diagnosis_cats}, Sex={n_sex_cats}, Smoking={n_smoking_cats})"
# )
# print(f"  Hidden Dimension: {hidden_dim}")
# print(f"  Dropout: {dropout}")
# print("\nTRAINING HYPERPARAMETERS:")
# print(f"  Batch Size: {batch_size}")
# print(f"  Learning Rate: {lr}")
# print(f"  Weight Decay: {weight_decay}")
# print("  Optimizer: Adam")
# print("  Loss Function: MSE")
# print(f"  Max Epochs: {epochs}")
# print(f"  Early Stopping Patience: {early_stopping_patience}")
# print("  Learning Rate Scheduler: ReduceLROnPlateau (patience=7, factor=0.6)")
# print("\nCOMPUTE:")
# print(f"  Device: {device}")
# print("  DataLoader Workers: 2")
# print(f"  Output Directory: {output_dir}")
# print(f"  Hyperparameter Search Results: {search_results_path}")
# print(f"  Best Hyperparameters File: {best_params_path}")
# print("=" * 80 + "\n")


# 6. Repeated K-Fold CV
fold_results = []
last_fold_artifacts = {}
global_fold_idx = 0

batch_size = 4
lr = 0.0005
dropout = 0.2
weight_decay = 1e-5
hidden_dim = 256



for seed_value in split_seeds:
    print("\n" + "=" * 80)
    print(f"Starting seed {seed_value}")
    print("=" * 80)

    test_ids, test_path = load_test_patient_ids(seed_value)
    missing_test_ids = [pid for pid in test_ids if pid not in patient_id_to_idx]
    if missing_test_ids:
        print(
            f"Warning: {len(missing_test_ids)} test IDs from {test_path} were not found in cases_df and will be skipped."
        )
    test_ids = [pid for pid in test_ids if pid in patient_id_to_idx]

    if len(test_ids) == 0:
        raise ValueError(
            f"No valid test IDs available for seed {seed_value} after filtering against cases_df."
        )

    test_idx = np.array([patient_id_to_idx[pid] for pid in test_ids], dtype=np.int64)
    X_test_seed = X_all[test_idx]
    y_test_seed = y_all[test_idx]
    print(f"Test split file: {test_path}")
    print(f"Test patients available for seed {seed_value}: {len(test_ids)}")

    for fold_idx in range(1, num_folds + 1):
        global_fold_idx += 1
        fold_idx_zero_based = fold_idx - 1

        train_ids, val_ids, train_fold_path, val_fold_path = load_fold_patient_ids(seed_value, fold_idx_zero_based)
        train_idx = np.array([patient_id_to_idx[pid] for pid in train_ids], dtype=np.int64)
        val_idx = np.array([patient_id_to_idx[pid] for pid in val_ids], dtype=np.int64)

        print("\n" + "-" * 80)
        print(
            f"Starting Seed {seed_value} Fold {fold_idx}/{num_folds} "
            f"(Global Fold {global_fold_idx}/{total_folds})"
        )
        print("-" * 80)
        print(f"Train split file: {train_fold_path}")
        print(f"Validation split file: {val_fold_path}")

        X_train, X_val = X_all[train_idx], X_all[val_idx]
        y_train, y_val = y_all[train_idx], y_all[val_idx]

        X_train_raw = X_train.copy()
        X_test = X_test_seed.copy()
        y_test = y_test_seed.copy()

        X_train, X_val, fold_scaler = scale_continuous_by_train(
            X_train_raw,
            X_val,
            continuous_feature_indices
        )

        # Apply train-fold scaler to the external test set to avoid leakage.
        _, X_test, _ = scale_continuous_by_train(
            X_train_raw,
            X_test,
            continuous_feature_indices,
            scaler=fold_scaler
        )

        train_dataset = TabularDataset(X_train, y_train)
        val_dataset = TabularDataset(X_val, y_val)
        test_dataset = TabularDataset(X_test, y_test)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )

        model = TabularMLP(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout).to(device)
        model = model.type(torch.FloatTensor).to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=7)
        criterion = nn.MSELoss()

        best_val_loss = float('inf')
        epochs_no_improve = 0
        loss_history = {'train_loss': [], 'val_loss': []}
        best_model_path = os.path.join(
            output_dir,
            f'best_model_seed{seed_value}_fold{fold_idx}_tabular_one_hot_bs{batch_size}_pt{len(patient_ids)}_{timestamp}.pth'
        )

        print(f"Seed {seed_value} Fold {fold_idx}: Starting Training Loop...")
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0

            for features, targets in train_loader:
                optimizer.zero_grad()

                features = features.type(torch.FloatTensor).to(device, non_blocking=True)
                targets = targets.type(torch.FloatTensor).to(device, non_blocking=True)

                outputs = model(features)
                loss = criterion(outputs, targets.unsqueeze(1))

                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                del features, targets, outputs, loss

            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for features, targets in val_loader:
                    features = features.type(torch.FloatTensor).to(device, non_blocking=True)
                    targets = targets.type(torch.FloatTensor).to(device, non_blocking=True)

                    outputs = model(features)
                    loss = criterion(outputs, targets.unsqueeze(1))

                    val_loss += loss.item()

                    del features, targets, outputs, loss

            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)

            loss_history['train_loss'].append(avg_train_loss)
            loss_history['val_loss'].append(avg_val_loss)

            current_lr = optimizer.param_groups[0]['lr']
            print(
                f"Seed {seed_value} Fold {fold_idx} | Epoch {epoch+1}/{epochs}, "
                f"Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, LR: {current_lr:.2e}"
            )

            scheduler.step(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), best_model_path)
                print(
                    f"Seed {seed_value} Fold {fold_idx}: "
                    f"New best model saved with validation loss: {best_val_loss:.6f}"
                )
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= early_stopping_patience:
                print(f"Seed {seed_value} Fold {fold_idx}: Early stopping triggered after {epoch+1} epochs")
                break

        print(f"Seed {seed_value} Fold {fold_idx}: Training complete.")
        print(f"Seed {seed_value} Fold {fold_idx}: Best validation loss: {best_val_loss:.6f}")

        # Evaluate on validation set
        model.load_state_dict(torch.load(best_model_path))
        model = model.type(torch.FloatTensor).to(device)
        model.eval()

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for features, targets in val_loader:
                features = features.type(torch.FloatTensor).to(device, non_blocking=True)
                targets = targets.type(torch.FloatTensor).to(device, non_blocking=True)

                outputs = model(features)
                predictions = outputs.squeeze().cpu().numpy()
                targets_np = targets.cpu().numpy()

                if predictions.ndim == 0:
                    predictions = np.array([predictions])
                if targets_np.ndim == 0:
                    targets_np = np.array([targets_np])

                all_predictions.extend(predictions)
                all_targets.extend(targets_np)

                del features, targets, outputs

        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)

        mse = mean_squared_error(all_targets, all_predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(all_targets, all_predictions)
        r2 = r2_score(all_targets, all_predictions)

        print(f"\nSeed {seed_value} Fold {fold_idx} Validation Metrics:")
        print(f"MSE: {mse:.6f}")
        print(f"RMSE: {rmse:.6f}")
        print(f"MAE: {mae:.6f}")
        print(f"R² Score: {r2:.6f}")

        eval_results = {
            'Timestamp': [timestamp],
            'Seed': [seed_value],
            'Fold': [fold_idx],
            'Global_Fold': [global_fold_idx],
            'Batch_Size': [batch_size],
            'Learning_Rate': [lr],
            'Weight_Decay': [weight_decay],
            'Early_Stopping_Patience': [early_stopping_patience],
            'Total_Patients': [len(patient_ids)],
            'Train_Samples': [len(train_idx)],
            'Val_Samples': [len(val_idx)],
            'MSE': [mse],
            'RMSE': [rmse],
            'MAE': [mae],
            'R2_Score': [r2],
            'Best_Val_Loss': [best_val_loss],
            'Total_Epochs': [len(loss_history['train_loss'])],
            'Input_Dim': [input_dim],
            'Hidden_Dim': [hidden_dim],
            'Best_Model_Path': [best_model_path]#,
         #   'Selected_Params': [json.dumps(best_params)]
        }
        eval_df = pd.DataFrame(eval_results)
        fold_eval_path = os.path.join(
            output_dir,
            f'evaluation_metrics_seed{seed_value}_fold{fold_idx}_tabular_one_hot_bs{batch_size}_pt{len(patient_ids)}_{timestamp}.csv'
        )
        eval_df.to_csv(fold_eval_path, index=False)
        print(f"Seed {seed_value} Fold {fold_idx}: Evaluation metrics saved to: {fold_eval_path}")

        # Evaluate on external test set for this seed
        test_predictions = []
        test_targets = []

        with torch.no_grad():
            for features, targets in test_loader:
                features = features.type(torch.FloatTensor).to(device, non_blocking=True)
                targets = targets.type(torch.FloatTensor).to(device, non_blocking=True)

                outputs = model(features)
                predictions = outputs.squeeze().cpu().numpy()
                targets_np = targets.cpu().numpy()

                if predictions.ndim == 0:
                    predictions = np.array([predictions])
                if targets_np.ndim == 0:
                    targets_np = np.array([targets_np])

                test_predictions.extend(predictions)
                test_targets.extend(targets_np)

                del features, targets, outputs

        test_predictions = np.array(test_predictions)
        test_targets = np.array(test_targets)

        test_mse = mean_squared_error(test_targets, test_predictions)
        test_rmse = np.sqrt(test_mse)
        test_mae = mean_absolute_error(test_targets, test_predictions)
        test_r2 = r2_score(test_targets, test_predictions)
        test_bootstrap_ci = bootstrap_metric_cis(
            test_targets,
            test_predictions,
            n_resamples=bootstrap_n_resamples,
            alpha=bootstrap_ci_alpha,
            seed=42 + global_fold_idx
        )

        print(f"\nSeed {seed_value} Fold {fold_idx} Test Metrics:")
        print(f"MSE: {test_mse:.6f}")
        print(f"RMSE: {test_rmse:.6f}")
        print(f"MAE: {test_mae:.6f}")
        print(f"R² Score: {test_r2:.6f}")
        print(
            f"95% Bootstrap CI - RMSE: [{test_bootstrap_ci['RMSE_CI_Lower']:.6f}, "
            f"{test_bootstrap_ci['RMSE_CI_Upper']:.6f}]"
        )

        test_results = {
            'Timestamp': [timestamp],
            'Seed': [seed_value],
            'Fold': [fold_idx],
            'Global_Fold': [global_fold_idx],
            'Batch_Size': [batch_size],
            'Learning_Rate': [lr],
            'Weight_Decay': [weight_decay],
            'Early_Stopping_Patience': [early_stopping_patience],
            'Total_Patients': [len(patient_ids)],
            'Train_Samples': [len(train_idx)],
            'Val_Samples': [len(val_idx)],
            'Test_Samples': [len(test_idx)],
            'MSE': [test_mse],
            'RMSE': [test_rmse],
            'MAE': [test_mae],
            'R2_Score': [test_r2],
            'MSE_CI_Lower': [test_bootstrap_ci['MSE_CI_Lower']],
            'MSE_CI_Upper': [test_bootstrap_ci['MSE_CI_Upper']],
            'RMSE_CI_Lower': [test_bootstrap_ci['RMSE_CI_Lower']],
            'RMSE_CI_Upper': [test_bootstrap_ci['RMSE_CI_Upper']],
            'MAE_CI_Lower': [test_bootstrap_ci['MAE_CI_Lower']],
            'MAE_CI_Upper': [test_bootstrap_ci['MAE_CI_Upper']],
            'R2_CI_Lower': [test_bootstrap_ci['R2_CI_Lower']],
            'R2_CI_Upper': [test_bootstrap_ci['R2_CI_Upper']],
            'Bootstrap_Resamples': [bootstrap_n_resamples],
            'Bootstrap_CI_Alpha': [bootstrap_ci_alpha],
            'Best_Val_Loss': [best_val_loss],
            'Total_Epochs': [len(loss_history['train_loss'])],
            'Input_Dim': [input_dim],
            'Hidden_Dim': [hidden_dim],
            'Best_Model_Path': [best_model_path],
            #'Selected_Params': [json.dumps(best_params)]
        }
        test_df = pd.DataFrame(test_results)
        fold_test_path = os.path.join(
            output_dir,
            f'test_metrics_seed{seed_value}_fold{fold_idx}_tabular_one_hot_bs{batch_size}_pt{len(patient_ids)}_{timestamp}.csv'
        )
        test_df.to_csv(fold_test_path, index=False)
        print(f"Seed {seed_value} Fold {fold_idx}: Test metrics saved to: {fold_test_path}")

        split_df = pd.DataFrame({
            'PatientID': train_ids + val_ids,
            'Split': ['train'] * len(train_ids) + ['validation'] * len(val_ids),
            'Seed': [seed_value] * (len(train_ids) + len(val_ids)),
            'Fold': [fold_idx] * (len(train_ids) + len(val_ids)),
            'Global_Fold': [global_fold_idx] * (len(train_ids) + len(val_ids))
        })
        patient_split_path = os.path.join(
            output_dir,
            f'patient_split_seed{seed_value}_fold{fold_idx}_tabular_one_hot_bs{batch_size}_pt{len(patient_ids)}_{timestamp}.csv'
        )
        split_df.to_csv(patient_split_path, index=False)
        print(f"Seed {seed_value} Fold {fold_idx}: Patient split information saved to: {patient_split_path}")

        loss_history['epoch'] = list(range(1, len(loss_history['train_loss']) + 1))
        loss_df = pd.DataFrame(loss_history)
        loss_history_path = os.path.join(
            output_dir,
            f'loss_history_seed{seed_value}_fold{fold_idx}_tabular_one_hot_bs{batch_size}_pt{len(patient_ids)}_{timestamp}.csv'
        )
        loss_df.to_csv(loss_history_path, index=False)
        print(f"Seed {seed_value} Fold {fold_idx}: Loss history saved to: {loss_history_path}")

        model_name = os.path.basename(best_model_path)

        plt.figure(figsize=(10, 6))
        plt.plot(loss_history['epoch'], loss_history['train_loss'], label='Train Loss')
        plt.plot(loss_history['epoch'], loss_history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(
            f'Training and Validation Loss Curves - Seed {seed_value} Fold {fold_idx}\n'
            f'Model: {model_name}\n'
            f'BS: {batch_size}, LR: {lr}, WD: {weight_decay}, ES Patience: {early_stopping_patience}, '
            f'Samples: {len(train_idx)}/{len(val_idx)}'
        )
        plt.legend()
        plt.grid()
        loss_curve_path = os.path.join(
            output_dir,
            f'loss_curves_seed{seed_value}_fold{fold_idx}_tabular_one_hot_bs{batch_size}_pt{len(patient_ids)}_{timestamp}.png'
        )
        plt.savefig(loss_curve_path)
        plt.close()

        plt.figure(figsize=(8, 8))
        plt.scatter(all_targets, all_predictions, alpha=0.7)
        plt.plot([all_targets.min(), all_targets.max()], [all_targets.min(), all_targets.max()], 'r--')
        plt.xlabel('True FVC Volume L')
        plt.ylabel('Predicted FVC Volume L')
        plt.title(
            f'Predicted vs True FVC Volume L - Seed {seed_value} Fold {fold_idx}\n'
            f'Model: {model_name}\n'
            f'MSE: {mse:.6f}, RMSE: {rmse:.6f}, MAE: {mae:.6f}, R²: {r2:.6f}'
        )
        plt.grid()
        pred_scatter_path = os.path.join(
            output_dir,
            f'predicted_vs_true_seed{seed_value}_fold{fold_idx}_tabular_one_hot_bs{batch_size}_pt{len(patient_ids)}_{timestamp}.png'
        )
        plt.savefig(pred_scatter_path)
        plt.close()

        residuals = all_targets - all_predictions
        plt.figure(figsize=(10, 6))
        plt.scatter(all_predictions, residuals, alpha=0.7)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted FVC Volume L')
        plt.ylabel('Residual (True - Predicted)')
        plt.title(
            f'Residual Plot - Seed {seed_value} Fold {fold_idx}\n'
            f'Model: {model_name}\n'
            f'MSE: {mse:.6f}, RMSE: {rmse:.6f}, MAE: {mae:.6f}, R²: {r2:.6f}'
        )
        plt.grid()
        residual_plot_path = os.path.join(
            output_dir,
            f'residual_plot_seed{seed_value}_fold{fold_idx}_tabular_one_hot_bs{batch_size}_pt{len(patient_ids)}_{timestamp}.png'
        )
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
            'Train_Samples': len(train_idx),
            'Val_Samples': len(val_idx),
            'Test_Samples': len(test_idx),
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2_Score': r2,
            'Test_MSE': test_mse,
            'Test_RMSE': test_rmse,
            'Test_MAE': test_mae,
            'Test_R2_Score': test_r2,
            'Best_Val_Loss': best_val_loss,
            'Total_Epochs': len(loss_history['train_loss']),
            'Best_Model_Path': best_model_path
        })

        last_fold_artifacts = {
            'seed': seed_value,
            'fold_idx': fold_idx,
            'best_model_path': best_model_path,
            'X_train': X_train.copy(),
            'X_val': X_val.copy(),
            'y_val': y_val.copy()
        }

# 7. Save fold metrics and CV summary
fold_results_df = pd.DataFrame(fold_results)
fold_results_path = os.path.join(
    output_dir,
    f'cv_fold_metrics_tabular_one_hot_all_seeds_fold_{total_folds}_bs{batch_size}_pt{len(patient_ids)}_{timestamp}.csv'
)
fold_results_df.to_csv(fold_results_path, index=False)

cv_summary = {
    'Timestamp': [timestamp],
    'Split_Seeds': [json.dumps(split_seeds)],
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
    'Best_Val_Loss_Std': [fold_results_df['Best_Val_Loss'].std()],
    #'Selected_Params': [json.dumps(best_params)],
    'Hyperparameter_Search_Results_Path': [search_results_path],
    #'Best_Hyperparameters_Path': [best_params_path]
}
cv_summary_df = pd.DataFrame(cv_summary)
cv_summary_path = os.path.join(
    output_dir,
    f'cv_summary_tabular_one_hot_all_seeds_fold_{total_folds}_bs{batch_size}_pt{len(patient_ids)}_{timestamp}.csv'
)
cv_summary_df.to_csv(cv_summary_path, index=False)

print("\n" + "=" * 80)
print("CROSS-VALIDATION COMPLETE")
print("=" * 80)
print(f"Fold metrics saved to: {fold_results_path}")
print(f"CV summary saved to: {cv_summary_path}")
print(f"All artifacts saved under: {output_dir}")
print("\nCV Mean Metrics:")
print(f"  MSE:  {cv_summary['MSE_Mean'][0]:.6f} ± {cv_summary['MSE_Std'][0]:.6f}")
print(f"  RMSE: {cv_summary['RMSE_Mean'][0]:.6f} ± {cv_summary['RMSE_Std'][0]:.6f}")
print(f"  MAE:  {cv_summary['MAE_Mean'][0]:.6f} ± {cv_summary['MAE_Std'][0]:.6f}")
print(f"  R²:   {cv_summary['R2_Mean'][0]:.6f} ± {cv_summary['R2_Std'][0]:.6f}")
print("\nCV Mean Test Metrics:")
print(f"  Test MSE:  {cv_summary['Test_MSE_Mean'][0]:.6f} ± {cv_summary['Test_MSE_Std'][0]:.6f}")
print(f"  Test RMSE: {cv_summary['Test_RMSE_Mean'][0]:.6f} ± {cv_summary['Test_RMSE_Std'][0]:.6f}")
print(f"  Test MAE:  {cv_summary['Test_MAE_Mean'][0]:.6f} ± {cv_summary['Test_MAE_Std'][0]:.6f}")
print(f"  Test R²:   {cv_summary['Test_R2_Mean'][0]:.6f} ± {cv_summary['Test_R2_Std'][0]:.6f}")
print(
    f"  Best Val Loss: "
    f"{cv_summary['Best_Val_Loss_Mean'][0]:.6f} ± {cv_summary['Best_Val_Loss_Std'][0]:.6f}"
)

# Box plot of fold-wise metric distributions
metric_cols = ['MSE', 'RMSE', 'MAE', 'R2_Score', 'Best_Val_Loss']
metric_labels = ['MSE', 'RMSE', 'MAE', 'R²', 'Best Val Loss']

plt.figure(figsize=(10, 6))
plt.boxplot(
    [fold_results_df[col].values for col in metric_cols],
    tick_labels=metric_labels,
    showmeans=True
)
plt.title(f'Cross-Validation Metric Distribution Across {total_folds} Folds')
plt.ylabel('Metric Value')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()

cv_boxplot_path = os.path.join(
    output_dir,
    f'cv_metrics_boxplot_tabular_one_hot_all_seeds_fold_{total_folds}_bs{batch_size}_pt{len(patient_ids)}_{timestamp}.png'
)
plt.savefig(cv_boxplot_path, dpi=200)
plt.close()

print(f"CV metrics box plot saved to: {cv_boxplot_path}")

# Plot SHAP values + classical permutation feature importance for last fold's best model

try:

    import shap

    np.random.seed(42)
    torch.manual_seed(42)

    last_fold_model_path = last_fold_artifacts['best_model_path']
    last_fold_seed = last_fold_artifacts['seed']
    last_fold_idx = last_fold_artifacts['fold_idx']
    X_train_last = last_fold_artifacts['X_train']
    X_val_last = last_fold_artifacts['X_val']
    y_val_last = last_fold_artifacts['y_val']

    model = TabularMLP(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout).to(device)
    model.load_state_dict(torch.load(last_fold_model_path))
    model.eval()
    background_size = min(100, len(X_train_last))
    explain_size = min(200, len(X_val_last))

    background_array = X_train_last[:background_size]
    explain_array = X_val_last[:explain_size]

    def predict_fn(x_np):
        x_tensor = torch.tensor(x_np, dtype=torch.float32).to(device)
        with torch.no_grad():
            preds = model(x_tensor).squeeze(-1).cpu().numpy()
        return preds

    explainer = shap.KernelExplainer(predict_fn, background_array)
    shap_values = explainer.shap_values(explain_array, nsamples=200)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap.summary_plot(shap_values, features=explain_array, feature_names=feature_names, show=False)
    shap_plot_path = os.path.join(
        output_dir,
        f'shap_summary_seed{last_fold_seed}_fold{last_fold_idx}_tabular_one_hot_bs{batch_size}_pt{len(patient_ids)}_{timestamp}.png'
    )
    plt.savefig(shap_plot_path, bbox_inches='tight')
    plt.close()
    print(f"SHAP summary plot saved to: {shap_plot_path}")

    shap.summary_plot(
        shap_values,
        features=explain_array,
        feature_names=feature_names,
        plot_type='bar',
        show=False
    )
    shap_bar_plot_path = os.path.join(
        output_dir,
        f'shap_feature_importance_bar_seed{last_fold_seed}_fold{last_fold_idx}_tabular_one_hot_bs{batch_size}_pt{len(patient_ids)}_{timestamp}.png'
    )
    plt.savefig(shap_bar_plot_path, bbox_inches='tight')
    plt.close()
    print(f"SHAP bar importance plot saved to: {shap_bar_plot_path}")

    # Classical feature importance: permutation importance on validation set
    with torch.no_grad():
        base_pred = model(torch.tensor(X_val_last, dtype=torch.float32).to(device)).squeeze().cpu().numpy()
    base_mse = mean_squared_error(y_val_last, base_pred)

    permutation_importances = []
    for feat_idx, feat_name in enumerate(feature_names):
        X_perm = X_val_last.copy()
        np.random.shuffle(X_perm[:, feat_idx])

        with torch.no_grad():
            perm_pred = model(torch.tensor(X_perm, dtype=torch.float32).to(device)).squeeze().cpu().numpy()

        perm_mse = mean_squared_error(y_val_last, perm_pred)
        importance = perm_mse - base_mse
        permutation_importances.append({'Feature': feat_name, 'Importance': importance})

    permutation_df = pd.DataFrame(permutation_importances).sort_values('Importance', ascending=False)
    permutation_csv_path = os.path.join(
        output_dir,
        f'permutation_feature_importance_seed{last_fold_seed}_fold{last_fold_idx}_tabular_one_hot_bs{batch_size}_pt{len(patient_ids)}_{timestamp}.csv'
    )
    permutation_df.to_csv(permutation_csv_path, index=False)

    top_n = min(20, len(permutation_df))
    top_perm = permutation_df.head(top_n).iloc[::-1]
    plt.figure(figsize=(10, max(6, top_n * 0.35)))
    plt.barh(top_perm['Feature'], top_perm['Importance'])
    plt.xlabel('Increase in MSE after permutation')
    plt.ylabel('Feature')
    plt.title(f'Permutation Feature Importance (Top {top_n}) - Fold {last_fold_idx}')
    plt.tight_layout()
    permutation_plot_path = os.path.join(
        output_dir,
        f'permutation_feature_importance_seed{last_fold_seed}_fold{last_fold_idx}_tabular_one_hot_bs{batch_size}_pt{len(patient_ids)}_{timestamp}.png'
    )
    plt.savefig(permutation_plot_path)
    plt.close()

    print(f"Permutation feature importance CSV saved to: {permutation_csv_path}")
    print(f"Permutation feature importance plot saved to: {permutation_plot_path}")

except ImportError:
    print("SHAP library not installed. Skipping SHAP analysis.")