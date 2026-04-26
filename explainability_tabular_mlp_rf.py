import argparse
import glob
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
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


class TabularMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x)


class TorchRegressorWrapper(BaseEstimator, RegressorMixin):
    """Minimal sklearn-compatible wrapper for permutation_importance."""

    def __init__(self, model, device):
        self.model = model
        self.device = device

    def fit(self, X, y):
        return self

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            xt = torch.tensor(X, dtype=torch.float32, device=self.device)
            preds = self.model(xt).squeeze(1).cpu().numpy()
        return preds


def parse_args():
    parser = argparse.ArgumentParser(
        description="SHAP + permutation importance for tabular MLP and RF."
    )
    parser.add_argument("--csv-path", default="cleaned_collected_cases_final_with_FEV1.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-folds", type=int, default=5)

    parser.add_argument("--mlp-run-dir", default="cv_tabular_one_hot_2026-03-31_15-22-26")
    parser.add_argument("--mlp-timestamp", default="2026-03-31_15-22-26")

    parser.add_argument("--rf-run-dir", default="cv_rf_one_hot_2026-03-10_13-23-51")
    parser.add_argument("--rf-timestamp", default="2026-03-10_13-23-51")

    parser.add_argument("--perm-repeats", type=int, default=30)
    parser.add_argument("--shap-background-size", type=int, default=120)
    parser.add_argument("--shap-explain-size", type=int, default=120)
    parser.add_argument("--shap-kernel-nsamples", type=int, default=200)

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Defaults to Explainability_Tabular_<timestamp>.",
    )

    return parser.parse_args()


def make_output_dir(output_dir):
    if output_dir is None:
        run_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = f"Explainability_Tabular_{run_ts}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


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

    x_all = np.concatenate([baseline, fev1, age, diagnosis_encoded, sex_encoded, smoking_encoded], axis=1).astype(np.float32)

    feature_names = (
        ["Baseline FVC", "FEV1 Volume L", "Age"]
        + [f"Primary Diagnosis_{cat}" for cat in enc_diagnosis.categories_[0]]
        + [f"Sex_{cat}" for cat in enc_sex.categories_[0]]
        + [f"Smoking Status_{cat}" for cat in enc_smoking.categories_[0]]
    )

    patient_ids = df["PatientID"].tolist()
    patient_id_to_idx = {pid: i for i, pid in enumerate(patient_ids)}
    return df, x_all, y_all, feature_names, patient_id_to_idx


def load_test_arrays(seed, x_all, y_all, patient_id_to_idx):
    test_path = os.path.join(f"Patient_Splits_Seed{seed}", "test_patients.csv")
    test_ids = pd.read_csv(test_path)["PatientID"].astype(str).tolist()
    test_ids = [pid for pid in test_ids if pid in patient_id_to_idx]
    test_idx = np.array([patient_id_to_idx[pid] for pid in test_ids], dtype=np.int64)
    return x_all[test_idx], y_all[test_idx], test_ids


def fit_scaler_for_mlp_fold(run_dir, seed, fold, model_timestamp, x_all, patient_id_to_idx):
    split_path = os.path.join(
        run_dir,
        f"patient_split_seed{seed}_fold{fold}_tabular_one_hot_bs4_pt556_{model_timestamp}.csv",
    )
    split_df = pd.read_csv(split_path)
    split_df["PatientID"] = split_df["PatientID"].astype(str)
    train_ids = split_df[split_df["Split"] == "train"]["PatientID"].tolist()
    train_ids = [pid for pid in train_ids if pid in patient_id_to_idx]

    train_idx = np.array([patient_id_to_idx[pid] for pid in train_ids], dtype=np.int64)
    x_train = x_all[train_idx]

    scaler = StandardScaler()
    scaler.fit(x_train[:, [0, 1, 2]])
    return scaler, x_train


def load_mlp_hidden_dims(run_dir, model_timestamp):
    pattern = os.path.join(
        run_dir,
        f"cv_fold_metrics_tabular_one_hot_all_seeds_fold_*_bs*_pt*_{model_timestamp}.csv",
    )
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"Could not find MLP fold metrics with pattern: {pattern}")

    metrics = pd.read_csv(paths[0])
    hidden_by_fold = {}
    if "Hidden_Dim" in metrics.columns and "Fold" in metrics.columns:
        for _, row in metrics.iterrows():
            hidden_by_fold[int(row["Fold"])] = int(row["Hidden_Dim"])
    return hidden_by_fold


def load_mlp_model(run_dir, seed, fold, model_timestamp, input_dim, hidden_dim, device):
    pattern = os.path.join(
        run_dir,
        f"best_model_seed{seed}_fold{fold}_tabular_one_hot_bs*_pt*_{model_timestamp}.pth",
    )
    ckpts = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not ckpts:
        raise FileNotFoundError(f"No MLP checkpoint found for fold {fold}: {pattern}")

    state_dict = torch.load(ckpts[-1], map_location=device)

    # If Hidden_Dim is unavailable in metrics, infer from first linear layer.
    if hidden_dim is None:
        if "net.0.weight" not in state_dict:
            raise KeyError("Could not infer hidden_dim: 'net.0.weight' not found in checkpoint state_dict.")
        hidden_dim = int(state_dict["net.0.weight"].shape[0])

    model = TabularMLP(input_dim=input_dim, hidden_dim=hidden_dim, dropout=0.2)
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    return model


def plot_top_importance(values, feature_names, out_path, title, top_n=20, color="#8172B3"):
    idx = np.argsort(values)[::-1][:top_n]
    top_features = [feature_names[i] for i in idx][::-1]
    top_values = values[idx][::-1]

    plt.figure(figsize=(12, max(6, top_n * 0.35)))
    plt.barh(top_features, top_values, color=color, edgecolor="black", alpha=0.85)
    plt.title(title)
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def run_mlp_explainability(args, x_all, y_all, feature_names, patient_id_to_idx, output_dir, device):
    print("\n" + "=" * 80)
    print("MLP Explainability")
    print("=" * 80)

    x_test_raw, y_test, _ = load_test_arrays(args.seed, x_all, y_all, patient_id_to_idx)
    hidden_by_fold = load_mlp_hidden_dims(args.mlp_run_dir, args.mlp_timestamp)

    shap_available = True
    try:
        import shap
    except ImportError:
        shap_available = False
        print("SHAP not installed; MLP SHAP plots will be skipped.")

    all_perm = []
    all_shap_abs = []
    all_shap_features = []
    all_shap_values = []

    for fold in range(1, args.num_folds + 1):
        print(f"Running MLP fold {fold}/{args.num_folds}...")
        scaler, x_train_raw = fit_scaler_for_mlp_fold(
            args.mlp_run_dir,
            args.seed,
            fold,
            args.mlp_timestamp,
            x_all,
            patient_id_to_idx,
        )

        x_test_scaled = x_test_raw.copy()
        x_test_scaled[:, [0, 1, 2]] = scaler.transform(x_test_raw[:, [0, 1, 2]])

        x_train_scaled = x_train_raw.copy()
        x_train_scaled[:, [0, 1, 2]] = scaler.transform(x_train_raw[:, [0, 1, 2]])

        hidden_dim = hidden_by_fold.get(fold)
        model = load_mlp_model(
            args.mlp_run_dir,
            args.seed,
            fold,
            args.mlp_timestamp,
            x_all.shape[1],
            hidden_dim,
            device,
        )

        wrapper = TorchRegressorWrapper(model, device)
        perm = permutation_importance(
            wrapper,
            x_test_scaled,
            y_test,
            scoring="neg_mean_absolute_error",
            n_repeats=args.perm_repeats,
            random_state=args.seed + fold,
            n_jobs=-1,
        )
        all_perm.append(perm.importances_mean)

        if shap_available:
            import shap

            bg_n = min(args.shap_background_size, len(x_train_scaled))
            ex_n = min(args.shap_explain_size, len(x_test_scaled))

            background = x_train_scaled[:bg_n]
            explain_x = x_test_scaled[:ex_n]

            def mlp_predict(x_np):
                with torch.no_grad():
                    xt = torch.tensor(x_np, dtype=torch.float32, device=device)
                    return model(xt).squeeze(1).cpu().numpy()

            explainer = shap.KernelExplainer(mlp_predict, background)
            shap_vals = explainer.shap_values(explain_x, nsamples=args.shap_kernel_nsamples)
            shap_vals = np.asarray(shap_vals)

            if shap_vals.ndim == 3:
                shap_vals = shap_vals[0]

            all_shap_values.append(shap_vals)
            all_shap_abs.append(np.abs(shap_vals).mean(axis=0))
            all_shap_features.append(explain_x)

    perm_mean = np.mean(np.vstack(all_perm), axis=0)
    perm_std = np.std(np.vstack(all_perm), axis=0)

    perm_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "MLP_Permutation_Mean": perm_mean,
            "MLP_Permutation_Std": perm_std,
        }
    ).sort_values("MLP_Permutation_Mean", ascending=False)

    perm_csv = os.path.join(output_dir, "mlp_permutation_importance.csv")
    perm_df.to_csv(perm_csv, index=False)

    plot_top_importance(
        perm_mean,
        feature_names,
        os.path.join(output_dir, "mlp_permutation_importance_top20.png"),
        "MLP Permutation Importance (mean across folds)",
        top_n=20,
        color="#8172B3",
    )

    if shap_available and all_shap_abs:
        import shap

        shap_abs_mean = np.mean(np.vstack(all_shap_abs), axis=0)
        shap_df = pd.DataFrame(
            {"Feature": feature_names, "MLP_SHAP_MeanAbs": shap_abs_mean}
        ).sort_values("MLP_SHAP_MeanAbs", ascending=False)
        shap_csv = os.path.join(output_dir, "mlp_shap_mean_abs.csv")
        shap_df.to_csv(shap_csv, index=False)

        plot_top_importance(
            shap_abs_mean,
            feature_names,
            os.path.join(output_dir, "mlp_shap_mean_abs_top20.png"),
            "MLP SHAP Mean |Value| (mean across folds)",
            top_n=20,
            color="#8172B3",
        )

        shap_features_concat = np.vstack(all_shap_features)
        shap_values_concat = np.vstack(all_shap_values)

        shap.summary_plot(
            shap_values_concat,
            features=shap_features_concat,
            feature_names=feature_names,
            plot_size=(12, 6),
            show=False,
        )
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "mlp_shap_summary_beeswarm.png"), dpi=220, bbox_inches="tight")
        plt.close()

    print(f"Saved MLP outputs to: {output_dir}")
    return perm_df


def train_rf_fold(seed, fold_zero_based, x_all, y_all, patient_id_to_idx, best_params):
    train_path = os.path.join(f"Patient_Splits_Seed{seed}", f"train_fold_{fold_zero_based}.csv")
    val_path = os.path.join(f"Patient_Splits_Seed{seed}", f"val_fold_{fold_zero_based}.csv")

    train_ids = pd.read_csv(train_path)["PatientID"].astype(str).tolist()
    val_ids = pd.read_csv(val_path)["PatientID"].astype(str).tolist()

    train_ids = [pid for pid in train_ids if pid in patient_id_to_idx]
    val_ids = [pid for pid in val_ids if pid in patient_id_to_idx]

    train_idx = np.array([patient_id_to_idx[pid] for pid in train_ids], dtype=np.int64)
    val_idx = np.array([patient_id_to_idx[pid] for pid in val_ids], dtype=np.int64)

    x_train = x_all[train_idx]
    y_train = y_all[train_idx]
    x_val = x_all[val_idx]

    model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
    model.fit(x_train, y_train)

    return model, x_train, x_val


def run_rf_explainability(args, x_all, y_all, feature_names, patient_id_to_idx, output_dir):
    print("\n" + "=" * 80)
    print("RF Explainability")
    print("=" * 80)

    best_params_path = os.path.join(args.rf_run_dir, f"best_rf_params_{args.rf_timestamp}.json")
    if not os.path.exists(best_params_path):
        raise FileNotFoundError(f"RF best params not found: {best_params_path}")

    with open(best_params_path, "r", encoding="utf-8") as f:
        best_params = json.load(f)

    x_test, y_test, _ = load_test_arrays(args.seed, x_all, y_all, patient_id_to_idx)

    shap_available = True
    try:
        import shap
    except ImportError:
        shap_available = False
        print("SHAP not installed; RF SHAP plots will be skipped.")

    all_perm = []
    all_shap_abs = []
    all_shap_features = []
    all_shap_values = []

    for fold in range(1, args.num_folds + 1):
        print(f"Running RF fold {fold}/{args.num_folds}...")
        model, x_train, _ = train_rf_fold(
            args.seed,
            fold - 1,
            x_all,
            y_all,
            patient_id_to_idx,
            best_params,
        )

        perm = permutation_importance(
            model,
            x_test,
            y_test,
            scoring="neg_mean_absolute_error",
            n_repeats=args.perm_repeats,
            random_state=args.seed + fold,
            n_jobs=-1,
        )
        all_perm.append(perm.importances_mean)

        if shap_available:
            import shap

            bg_n = min(args.shap_background_size, len(x_train))
            ex_n = min(args.shap_explain_size, len(x_test))
            background = x_train[:bg_n]
            explain_x = x_test[:ex_n]

            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(explain_x)
            shap_vals = np.asarray(shap_vals)
            if shap_vals.ndim == 3:
                shap_vals = shap_vals[0]

            all_shap_values.append(shap_vals)
            all_shap_abs.append(np.abs(shap_vals).mean(axis=0))
            all_shap_features.append(explain_x)

    perm_mean = np.mean(np.vstack(all_perm), axis=0)
    perm_std = np.std(np.vstack(all_perm), axis=0)

    perm_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "RF_Permutation_Mean": perm_mean,
            "RF_Permutation_Std": perm_std,
        }
    ).sort_values("RF_Permutation_Mean", ascending=False)

    perm_csv = os.path.join(output_dir, "rf_permutation_importance.csv")
    perm_df.to_csv(perm_csv, index=False)

    plot_top_importance(
        perm_mean,
        feature_names,
        os.path.join(output_dir, "rf_permutation_importance_top20.png"),
        "RF Permutation Importance (mean across folds)",
        top_n=20,
        color="#8172B3",
    )

    if shap_available and all_shap_abs:
        import shap

        shap_abs_mean = np.mean(np.vstack(all_shap_abs), axis=0)
        shap_df = pd.DataFrame(
            {"Feature": feature_names, "RF_SHAP_MeanAbs": shap_abs_mean}
        ).sort_values("RF_SHAP_MeanAbs", ascending=False)
        shap_csv = os.path.join(output_dir, "rf_shap_mean_abs.csv")
        shap_df.to_csv(shap_csv, index=False)

        plot_top_importance(
            shap_abs_mean,
            feature_names,
            os.path.join(output_dir, "rf_shap_mean_abs_top20.png"),
            "RF SHAP Mean |Value| (mean across folds)",
            top_n=20,
            color="#8172B3",
        )

        shap_features_concat = np.vstack(all_shap_features)
        shap_values_concat = np.vstack(all_shap_values)

        shap.summary_plot(
            shap_values_concat,
            features=shap_features_concat,
            feature_names=feature_names,
            plot_size=(12, 6),
            show=False,
        )
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "rf_shap_summary_beeswarm.png"), dpi=220, bbox_inches="tight")
        plt.close()

    print(f"Saved RF outputs to: {output_dir}")
    return perm_df


def main():
    args = parse_args()
    output_dir = make_output_dir(args.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Output directory: {output_dir}")

    _, x_all, y_all, feature_names, patient_id_to_idx = build_features(args.csv_path)

    mlp_perm_df = run_mlp_explainability(
        args,
        x_all,
        y_all,
        feature_names,
        patient_id_to_idx,
        output_dir,
        device,
    )

    rf_perm_df = run_rf_explainability(
        args,
        x_all,
        y_all,
        feature_names,
        patient_id_to_idx,
        output_dir,
    )

    merged = mlp_perm_df.merge(rf_perm_df, on="Feature", how="outer")
    merged = merged.sort_values(
        ["MLP_Permutation_Mean", "RF_Permutation_Mean"],
        ascending=False,
        na_position="last",
    )
    merged.to_csv(os.path.join(output_dir, "permutation_importance_mlp_vs_rf.csv"), index=False)

    print("\n" + "=" * 80)
    print("EXPLAINABILITY COMPLETE")
    print("=" * 80)
    print(f"All outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
