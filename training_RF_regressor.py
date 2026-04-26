import numpy as np
import pandas as pd
import os
import json

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
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
fev1 = cases_df['FEV1 Volume L'].to_numpy(dtype=np.float32).reshape(-1, 1)
age = cases_df['Age'].to_numpy(dtype=np.float32).reshape(-1, 1)
y_all = cases_df['Followup FVC Volume L'].to_numpy(dtype=np.float32)

X_all = np.concatenate([baseline_fvc, fev1, age, diagnosis_encoded, sex_encoded, smoking_encoded], axis=1).astype(np.float32)
patient_ids = cases_df['PatientID'].tolist()
patient_ids_str = [str(pid) for pid in patient_ids]
patient_id_to_idx = {pid: idx for idx, pid in enumerate(patient_ids_str)}

feature_names = ['Baseline FVC', 'FEV1 Volume L', 'Age'] + \
                [f'Primary Diagnosis_{cat}' for cat in enc_diagnosis.categories_[0]] + \
                [f'Sex_{cat}' for cat in enc_sex.categories_[0]] + \
                [f'Smoking Status_{cat}' for cat in enc_smoking.categories_[0]]

# 4. Config
split_seeds = [42, 24, 7]
num_folds = 5
total_folds = len(split_seeds) * num_folds

output_dir = f"cv_rf_one_hot_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

print(f"Using predefined split seeds: {split_seeds}")
print(f"Folds per seed: {num_folds}")
print(f"Total folds across all seeds: {total_folds}")
print(f"Output directory: {output_dir}")
print(f"Input dimension: {X_all.shape[1]}")

bootstrap_n_resamples = 1000
bootstrap_ci_alpha = 0.95

print("\n" + "=" * 80)
print("RANDOM FOREST TRAINING CONFIGURATION")
print("=" * 80)
print(f"Run Timestamp: {timestamp}")
print(f"Total Patients: {len(patient_ids)}")
print(f"Input Features: {X_all.shape[1]}")
print(f"Split Seeds: {split_seeds}")
print(f"Folds per Seed: {num_folds}")
print(f"Total Folds: {total_folds}")
print("=" * 80 + "\n")


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

# 5. Grid Search BEFORE cross-validation (refined around previous best)
# Previous best: {'max_depth': 30, 'max_features': None, 'min_samples_leaf': 4,
#                 'min_samples_split': 10, 'n_estimators': 400}
param_grid = {
    'n_estimators': [100, 200, 300, 400, 600],
    'max_depth': [20, 30, 40, None],
    'min_samples_split': [4,6,8, 10, 12],
    'min_samples_leaf': [1, 2, 3,  5],
    'max_features': [None, 'sqrt']
}

print("Starting GridSearchCV to find best RandomForest hyperparameters...")
base_rf = RandomForestRegressor(random_state=42, n_jobs=-1)

cv_splits = []
for seed_value in split_seeds:
    for fold_idx in range(1, num_folds + 1):
        fold_idx_zero_based = fold_idx - 1
        train_ids, val_ids, _, _ = load_fold_patient_ids(seed_value, fold_idx_zero_based)

        train_ids = [pid for pid in train_ids if pid in patient_id_to_idx]
        val_ids = [pid for pid in val_ids if pid in patient_id_to_idx]
        if len(train_ids) == 0 or len(val_ids) == 0:
            raise ValueError(
                f"Seed {seed_value} fold {fold_idx} has no valid train/val IDs after filtering."
            )

        train_idx = np.array([patient_id_to_idx[pid] for pid in train_ids], dtype=np.int64)
        val_idx = np.array([patient_id_to_idx[pid] for pid in val_ids], dtype=np.int64)
        cv_splits.append((train_idx, val_idx))

grid_search = GridSearchCV(
    estimator=base_rf,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=cv_splits,
    n_jobs=-1,
    verbose=2,
    #return_train_score=True
)
grid_search.fit(X_all, y_all)

best_params = grid_search.best_params_
best_score = -grid_search.best_score_

print("Grid search complete.")
print(f"Best params: {best_params}")
print(f"Best CV MSE from grid search: {best_score:.6f}")

grid_results_df = pd.DataFrame(grid_search.cv_results_)
grid_results_path = os.path.join(output_dir, f'grid_search_results_rf_{timestamp}.csv')
grid_results_df.to_csv(grid_results_path, index=False)

# Build plotting dataframe from GridSearchCV results.
plot_df = grid_results_df.copy()
plot_df['mean_test_mse'] = -plot_df['mean_test_score']
plot_df['std_test_mse'] = plot_df['std_test_score']

for hp in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features']:
    plot_df[hp] = plot_df[f'param_{hp}']


def _to_numeric_or_none(series):
    if series.dtype == object:
        mapped = series.map(lambda x: np.nan if x is None or str(x) == 'None' else x)
        return pd.to_numeric(mapped, errors='coerce')
    return pd.to_numeric(series, errors='coerce')


def _plot_hyperparam_vs_mse(df, hp_name, output_path):
    hp_vals = df[hp_name]
    hp_num = _to_numeric_or_none(hp_vals)

    # If values are numeric-ish, plot by numeric order; else, keep categorical labels.
    if hp_num.notna().sum() == len(hp_num):
        hp_perf = (
            pd.DataFrame({hp_name: hp_num, 'mean_test_mse': df['mean_test_mse']})
            .groupby(hp_name, as_index=False)
            .mean()
            .sort_values(hp_name)
        )

        plt.figure(figsize=(8, 5))
        plt.plot(hp_perf[hp_name], hp_perf['mean_test_mse'], marker='o')
        plt.xlabel(hp_name)
        plt.ylabel('Mean CV MSE')
        plt.title(f'Hyperparameter vs MSE: {hp_name}')
        plt.grid(True, alpha=0.4)
        plt.tight_layout()
        plt.savefig(output_path, dpi=220)
        plt.close()
        return

    hp_perf = (
        pd.DataFrame({hp_name: hp_vals.astype(str), 'mean_test_mse': df['mean_test_mse']})
        .groupby(hp_name, as_index=False)
        .mean()
    )

    # Keep a stable, meaningful order for max_features when None is present.
    if hp_name == 'max_features':
        order = [x for x in ['None', 'sqrt', 'log2'] if x in hp_perf[hp_name].tolist()]
        extras = [x for x in hp_perf[hp_name].tolist() if x not in order]
        hp_perf['__order'] = hp_perf[hp_name].map(
            {k: i for i, k in enumerate(order + sorted(extras))}
        )
        hp_perf = hp_perf.sort_values('__order').drop(columns='__order')

    x_positions = np.arange(len(hp_perf))
    plt.figure(figsize=(8, 5))
    plt.plot(x_positions, hp_perf['mean_test_mse'], marker='o')
    plt.xticks(x_positions, hp_perf[hp_name].tolist())
    plt.xlabel(hp_name)
    plt.ylabel('Mean CV MSE')
    plt.title(f'Hyperparameter vs MSE: {hp_name}')
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


# 5b. Hyperparameter search plots (similar spirit to MLP search diagnostics)
for hp_name in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features']:
    hp_plot_path = os.path.join(output_dir, f'hyperparam_vs_mse_rf_{hp_name}_{timestamp}.png')
    _plot_hyperparam_vs_mse(plot_df, hp_name, hp_plot_path)
    print(f"Hyperparameter-vs-MSE plot saved to: {hp_plot_path}")


def _plot_heatmap(df, row_hp, col_hp, metric_col, output_path):
    row_vals = df[row_hp].astype(str)
    col_vals = df[col_hp].astype(str)

    pivot = (
        pd.DataFrame({row_hp: row_vals, col_hp: col_vals, metric_col: df[metric_col]})
        .groupby([row_hp, col_hp], as_index=False)
        .mean()
        .pivot(index=row_hp, columns=col_hp, values=metric_col)
    )

    if pivot.empty:
        return

    fig, ax = plt.subplots(
        figsize=(max(6, pivot.shape[1] * 1.2), max(4.5, pivot.shape[0] * 0.9))
    )
    im = ax.imshow(pivot.values, aspect='auto', cmap='viridis_r')

    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels([str(v) for v in pivot.columns], rotation=45, ha='right')
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels([str(v) for v in pivot.index])
    ax.set_xlabel(col_hp)
    ax.set_ylabel(row_hp)
    ax.set_title(f'Mean CV MSE: {row_hp} x {col_hp}')

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.4f}', ha='center', va='center', fontsize=7, color='white')

    plt.colorbar(im, ax=ax, label='Mean CV MSE')
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


heatmap_pairs = [
    #('n_estimators', 'max_depth'),
    ('min_samples_split', 'min_samples_leaf'),
    ('n_estimators', 'min_samples_leaf'),
    ('n_estimators', 'min_samples_split'),
    #('max_depth', 'max_features'),
    ('n_estimators', 'max_features')
]

for row_hp, col_hp in heatmap_pairs:
    heatmap_path = os.path.join(
        output_dir,
        f'hyperparam_heatmap_rf_mse_{row_hp}_x_{col_hp}_{timestamp}.png'
    )
    _plot_heatmap(plot_df, row_hp, col_hp, 'mean_test_mse', heatmap_path)
    print(f"Hyperparameter heatmap saved to: {heatmap_path}")

best_params_path = os.path.join(output_dir, f'best_rf_params_{timestamp}.json')
with open(best_params_path, 'w') as f:
    json.dump(best_params, f, indent=2)

print(f"Grid search results saved to: {grid_results_path}")
print(f"Best params saved to: {best_params_path}")

# 6. Predefined seed/fold CV using best params from grid search
fold_results = []
fold_feature_importances = []
global_fold_idx = 0
last_fold_artifacts = {}

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
    X_test = X_all[test_idx]
    y_test = y_all[test_idx]

    print(f"Test split file: {test_path}")
    print(f"Test patients available for seed {seed_value}: {len(test_ids)}")

    for fold_idx in range(1, num_folds + 1):
        global_fold_idx += 1
        fold_idx_zero_based = fold_idx - 1
        train_ids, val_ids, train_fold_path, val_fold_path = load_fold_patient_ids(seed_value, fold_idx_zero_based)

        train_ids = [pid for pid in train_ids if pid in patient_id_to_idx]
        val_ids = [pid for pid in val_ids if pid in patient_id_to_idx]
        if len(train_ids) == 0 or len(val_ids) == 0:
            raise ValueError(
                f"Seed {seed_value} fold {fold_idx} has no valid train/val IDs after filtering."
            )

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

        model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        predictions = model.predict(X_val)
        test_predictions = model.predict(X_test)

        mse = mean_squared_error(y_val, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_val, predictions)
        r2 = r2_score(y_val, predictions)

        test_mse = mean_squared_error(y_test, test_predictions)
        test_rmse = np.sqrt(test_mse)
        test_mae = mean_absolute_error(y_test, test_predictions)
        test_r2 = r2_score(y_test, test_predictions)
        test_bootstrap_ci = bootstrap_metric_cis(
            y_test,
            test_predictions,
            n_resamples=bootstrap_n_resamples,
            alpha=bootstrap_ci_alpha,
            seed=42 + global_fold_idx
        )

        print(
            f"Seed {seed_value} Fold {fold_idx} Validation Metrics: "
            f"MSE={mse:.6f}, RMSE={rmse:.6f}, MAE={mae:.6f}, R²={r2:.6f}"
        )
        print(
            f"Seed {seed_value} Fold {fold_idx} Test Metrics: "
            f"MSE={test_mse:.6f}, RMSE={test_rmse:.6f}, MAE={test_mae:.6f}, R²={test_r2:.6f}"
        )
        print(
            f"Seed {seed_value} Fold {fold_idx} Test 95% Bootstrap CI - RMSE: "
            f"[{test_bootstrap_ci['RMSE_CI_Lower']:.6f}, {test_bootstrap_ci['RMSE_CI_Upper']:.6f}]"
        )

        eval_df = pd.DataFrame({
            'Timestamp': [timestamp],
            'Seed': [seed_value],
            'Fold': [fold_idx],
            'Global_Fold': [global_fold_idx],
            'Total_Patients': [len(patient_ids)],
            'Train_Samples': [len(train_idx)],
            'Val_Samples': [len(val_idx)],
            'MSE': [mse],
            'RMSE': [rmse],
            'MAE': [mae],
            'R2_Score': [r2],
            'Best_Params': [json.dumps(best_params)]
        })
        fold_eval_path = os.path.join(
            output_dir,
            f'evaluation_metrics_seed{seed_value}_fold{fold_idx}_rf_one_hot_{timestamp}.csv'
        )
        eval_df.to_csv(fold_eval_path, index=False)

        test_df = pd.DataFrame({
            'Timestamp': [timestamp],
            'Seed': [seed_value],
            'Fold': [fold_idx],
            'Global_Fold': [global_fold_idx],
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
            'Best_Params': [json.dumps(best_params)]
        })
        fold_test_path = os.path.join(
            output_dir,
            f'test_metrics_seed{seed_value}_fold{fold_idx}_rf_one_hot_{timestamp}.csv'
        )
        test_df.to_csv(fold_test_path, index=False)

        split_df = pd.DataFrame({
            'PatientID': train_ids + val_ids,
            'Split': ['train'] * len(train_ids) + ['validation'] * len(val_ids),
            'Seed': [seed_value] * (len(train_ids) + len(val_ids)),
            'Fold': [fold_idx] * (len(train_ids) + len(val_ids)),
            'Global_Fold': [global_fold_idx] * (len(train_ids) + len(val_ids))
        })
        split_path = os.path.join(
            output_dir,
            f'patient_split_seed{seed_value}_fold{fold_idx}_rf_one_hot_{timestamp}.csv'
        )
        split_df.to_csv(split_path, index=False)

    # Predicted vs true
        plt.figure(figsize=(8, 8))
        plt.scatter(y_val, predictions, alpha=0.7)
        plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
        plt.xlabel('True FVC Volume L')
        plt.ylabel('Predicted FVC Volume L')
        plt.title(
            f'RF Predicted vs True - Seed {seed_value} Fold {fold_idx}\n'
            f'MSE: {mse:.6f}, RMSE: {rmse:.6f}, MAE: {mae:.6f}, R²: {r2:.6f}'
        )
        plt.grid()
        pred_scatter_path = os.path.join(
            output_dir,
            f'predicted_vs_true_seed{seed_value}_fold{fold_idx}_rf_one_hot_{timestamp}.png'
        )
        plt.savefig(pred_scatter_path)
        plt.close()

    # Residual plot
        residuals = y_val - predictions
        plt.figure(figsize=(10, 6))
        plt.scatter(predictions, residuals, alpha=0.7)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted FVC Volume L')
        plt.ylabel('Residual (True - Predicted)')
        plt.title(f'RF Residual Plot - Seed {seed_value} Fold {fold_idx}')
        plt.grid()
        residual_plot_path = os.path.join(
            output_dir,
            f'residual_plot_seed{seed_value}_fold{fold_idx}_rf_one_hot_{timestamp}.png'
        )
        plt.savefig(residual_plot_path)
        plt.close()

    # Fold feature importance (classical for RF)
        fold_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_,
            'Seed': seed_value,
            'Fold': fold_idx,
            'Global_Fold': global_fold_idx
        }).sort_values('Importance', ascending=False)
        fold_importance_path = os.path.join(
            output_dir,
            f'feature_importance_seed{seed_value}_fold{fold_idx}_rf_one_hot_{timestamp}.csv'
        )
        fold_importance_df.to_csv(fold_importance_path, index=False)

        top_n = min(20, len(fold_importance_df))
        top_imp = fold_importance_df.head(top_n).iloc[::-1]
        plt.figure(figsize=(10, max(6, top_n * 0.35)))
        plt.barh(top_imp['Feature'], top_imp['Importance'])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(f'Random Forest Feature Importance (Top {top_n}) - Seed {seed_value} Fold {fold_idx}')
        plt.tight_layout()
        fold_importance_plot_path = os.path.join(
            output_dir,
            f'feature_importance_seed{seed_value}_fold{fold_idx}_rf_one_hot_{timestamp}.png'
        )
        plt.savefig(fold_importance_plot_path)
        plt.close()

        print(
            f"Seed {seed_value} Fold {fold_idx}: Saved validation/test metrics, split, prediction, residual, and feature-importance outputs."
        )

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
            'Test_R2_Score': test_r2
        })
        fold_feature_importances.append(model.feature_importances_)

        last_fold_artifacts = {
            'seed': seed_value,
            'fold_idx': fold_idx,
            'model': model,
            'X_train': X_train.copy(),
            'X_val': X_val.copy()
        }

# 7. Save fold metrics and CV summary
fold_results_df = pd.DataFrame(fold_results)
fold_results_path = os.path.join(output_dir, f'cv_fold_metrics_rf_one_hot_fold_{total_folds}_{timestamp}.csv')
fold_results_df.to_csv(fold_results_path, index=False)

cv_summary = {
    'Timestamp': [timestamp],
    'Split_Seeds': [json.dumps(split_seeds)],
    'Num_Seeds': [len(split_seeds)],
    'Num_Folds_Per_Seed': [num_folds],
    'Total_Folds': [total_folds],
    'Best_Params': [json.dumps(best_params)],
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
    'Test_R2_Std': [fold_results_df['Test_R2_Score'].std()]
}
cv_summary_df = pd.DataFrame(cv_summary)
cv_summary_path = os.path.join(output_dir, f'cv_summary_rf_one_hot_fold_{total_folds}_{timestamp}.csv')
cv_summary_df.to_csv(cv_summary_path, index=False)

# CV metrics boxplot
metric_cols = ['MSE', 'RMSE', 'MAE', 'R2_Score']
metric_labels = ['MSE', 'RMSE', 'MAE', 'R²']
plt.figure(figsize=(10, 6))
plt.boxplot([fold_results_df[col].values for col in metric_cols], tick_labels=metric_labels, showmeans=True)
plt.title(f'Random Forest CV Metric Distribution Across {total_folds} Folds')
plt.ylabel('Metric Value')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
boxplot_path = os.path.join(output_dir, f'cv_metrics_boxplot_rf_one_hot_fold_{total_folds}_{timestamp}.png')
plt.savefig(boxplot_path, dpi=200)
plt.close()

# Mean feature importance across folds
mean_importance = np.mean(np.vstack(fold_feature_importances), axis=0)
importance_df = pd.DataFrame({'Feature': feature_names, 'Mean_Importance': mean_importance})
importance_df = importance_df.sort_values('Mean_Importance', ascending=False)
mean_importance_csv_path = os.path.join(output_dir, f'feature_importance_mean_rf_one_hot_{timestamp}.csv')
importance_df.to_csv(mean_importance_csv_path, index=False)

top_n = min(20, len(importance_df))
top_mean = importance_df.head(top_n).iloc[::-1]
plt.figure(figsize=(10, max(6, top_n * 0.35)))
plt.barh(top_mean['Feature'], top_mean['Mean_Importance'])
plt.xlabel('Mean Importance Across Folds')
plt.ylabel('Feature')
plt.title(f'Random Forest Mean Feature Importance (Top {top_n})')
plt.tight_layout()
mean_importance_plot_path = os.path.join(output_dir, f'feature_importance_mean_rf_one_hot_{timestamp}.png')
plt.savefig(mean_importance_plot_path)
plt.close()

print("\n" + "=" * 80)
print("RANDOM FOREST TRAINING COMPLETE")
print("=" * 80)
print(f"Grid search results saved to: {grid_results_path}")
print(f"Best params saved to: {best_params_path}")
print(f"Fold metrics saved to: {fold_results_path}")
print(f"CV summary saved to: {cv_summary_path}")
print(f"CV boxplot saved to: {boxplot_path}")
print(f"Mean feature importance CSV saved to: {mean_importance_csv_path}")
print(f"Mean feature importance plot saved to: {mean_importance_plot_path}")
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

# 8. SHAP explainability (last fold model)
try:
    import shap

    last_fold_seed = last_fold_artifacts['seed']
    last_fold_idx = last_fold_artifacts['fold_idx']
    last_fold_model = last_fold_artifacts['model']
    X_train_last = last_fold_artifacts['X_train']
    X_val_last = last_fold_artifacts['X_val']

    background_size = min(200, len(X_train_last))
    explain_size = min(200, len(X_val_last))

    background_array = X_train_last[:background_size]
    explain_array = X_val_last[:explain_size]

    explainer = shap.TreeExplainer(last_fold_model)
    shap_values = explainer.shap_values(explain_array)

    shap.summary_plot(shap_values, features=explain_array, feature_names=feature_names, show=False)
    shap_plot_path = os.path.join(
        output_dir,
        f'shap_summary_seed{last_fold_seed}_fold{last_fold_idx}_rf_one_hot_{timestamp}.png'
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
        f'shap_feature_importance_bar_seed{last_fold_seed}_fold{last_fold_idx}_rf_one_hot_{timestamp}.png'
    )
    plt.savefig(shap_bar_plot_path, bbox_inches='tight')
    plt.close()
    print(f"SHAP bar importance plot saved to: {shap_bar_plot_path}")

except ImportError:
    print("SHAP library not installed. Skipping SHAP analysis.")
