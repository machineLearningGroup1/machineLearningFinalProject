"""
Task 4 - Random Forest Classification (OOB tuning)
=====================================
Input:
  - ../data_preprocessing_final/result/train_70.csv
  - ../data_preprocessing_final/result/val_10.csv (merged into test below)
  - ../data_preprocessing_final/result/test_20.csv (merged into test below)
  - ../data_preprocessing_final/result/full_preprocessed_data.csv

Process:
  - Merge val_10 + test_20 as the new test set
  - Hyperparameter tuning by OOB on train_70
  - Train final RandomForest on train_70 only
  - Evaluate on train / merged-test / full and save plots

Output:
  - result/random_forest_log.txt
  - result/rf_best_params.json
  - result/rf_tuning_results.csv
  - result/rf_metrics_summary.json
  - result/rf_classification_report_{train,test,full}.txt
  - result/rf_confusion_{train,test,full}.png
  - result/rf_feature_importance.png
  - result/rf_feature_importance_permutation.png
  - result/rf_oob_vs_ntrees.png
  - result/rf_roc_curves_ovr.png
  - result/rf_decision_boundary_pca.png
  - result/rf_model.pkl
"""

from __future__ import annotations

import json
import sys
import warnings
from itertools import product
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")


# ============================================================
# 0. Paths / constants / logging
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data_preprocessing_final" / "result"
RESULT_DIR = BASE_DIR / "result" / "random_forest_triple_split"
RESULT_DIR.mkdir(parents=True, exist_ok=True, mode=0o755)

TRAIN_PATH = DATA_DIR / "train_70.csv"
VAL_PATH = DATA_DIR / "val_10.csv"
TEST_PATH = DATA_DIR / "test_20.csv"
FULL_PATH = DATA_DIR / "full_preprocessed_data.csv"

TARGET_COL = "Target"
TARGET_BIN_COL = "Target_binary"
RANDOM_STATE = 42
N_JOBS = -1

CLASS_NAMES = ["Dropout", "Enrolled", "Graduate"]
COLORS = {0: "#d73221", 1: "#fcb777", 2: "#4573b4"}
MARKERS = {0: "s", 1: "^", 2: "o"}
OOB_TREE_COUNTS = [10, 20, 30, 40, 50, 75, 100, 150, 200, 250, 300, 400, 500]


class Logger:
    def __init__(self, filename: Path):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message: str) -> None:
        self.terminal.write(message)
        self.log.write(message)

    def flush(self) -> None:
        self.terminal.flush()
        self.log.flush()


sys.stdout = Logger(RESULT_DIR / "random_forest_log.txt")

plt.rcParams.update(
    {
        "figure.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


# ============================================================
# 1. Helper functions
# ============================================================
def load_xy(csv_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)
    X = df.drop(columns=[TARGET_COL, TARGET_BIN_COL])
    y = df[TARGET_COL].astype(int)
    return X, y


def calc_metrics(y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_precision": round(
            float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4
        ),
        "macro_recall": round(
            float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4
        ),
        "macro_f1": round(
            float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4
        ),
        "weighted_f1": round(
            float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4
        ),
        "roc_auc_ovr_macro": round(
            float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")), 4
        ),
        "roc_auc_ovr_weighted": round(
            float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")), 4
        ),
    }


def save_report(y_true: pd.Series, y_pred: np.ndarray, split_name: str) -> None:
    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
    (RESULT_DIR / f"rf_classification_report_{split_name}.txt").write_text(
        report,
        encoding="utf-8",
    )


def plot_confusion(y_true: pd.Series, y_pred: np.ndarray, split_name: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=CLASS_NAMES,
        cmap="Blues",
        colorbar=False,
        ax=ax,
        values_format="d",
    )
    disp.ax_.set_title(f"Random Forest - Confusion Matrix ({split_name})")
    plt.tight_layout()
    plt.savefig(RESULT_DIR / f"rf_confusion_{split_name}.png", bbox_inches="tight")
    plt.close()


def manual_grid_search_oob_rf(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[dict, pd.DataFrame, dict]:
    grid = {
        "n_estimators": [100, 300, 500],
        "max_depth": [None, 10, 20],
        "min_samples_leaf": [1, 2, 5],
        "max_features": ["sqrt", "log2", 0.5],
        "class_weight": ["balanced"],
        "bootstrap": [True],
        "oob_score": [True],
    }

    keys = list(grid.keys())
    combos = list(product(*(grid[key] for key in keys)))
    records = []
    best_record = None

    print("=" * 70)
    print("Task 4 - Random Forest Classification")
    print("=" * 70)
    print("\n[STEP 1] Train candidates + OOB validation for hyperparameter tuning")
    print(f"  Search combinations: {len(combos)}")

    for idx, values in enumerate(combos, start=1):
        params = dict(zip(keys, values))
        model = RandomForestClassifier(
            **params,
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
        )
        model.fit(X_train, y_train)
        oob_pred = model.oob_decision_function_.argmax(axis=1)
        oob_macro_f1 = f1_score(y_train, oob_pred, average="macro", zero_division=0)
        oob_accuracy = accuracy_score(y_train, oob_pred)

        record = {
            **params,
            "oob_accuracy": float(oob_accuracy),
            "oob_macro_f1": float(oob_macro_f1),
            "oob_score_value": float(model.oob_score_),
        }
        records.append(record)

        if best_record is None:
            best_record = record
        else:
            better = (
                record["oob_macro_f1"] > best_record["oob_macro_f1"]
                or (
                    record["oob_macro_f1"] == best_record["oob_macro_f1"]
                    and record["oob_accuracy"] > best_record["oob_accuracy"]
                )
                or (
                    record["oob_macro_f1"] == best_record["oob_macro_f1"]
                    and record["oob_accuracy"] == best_record["oob_accuracy"]
                    and record["n_estimators"] < best_record["n_estimators"]
                )
            )
            if better:
                best_record = record

        if idx % 10 == 0 or idx == len(combos):
            print(
                f"  Progress: {idx:>2}/{len(combos)} | "
                f"Current best OOB macro-F1 = {best_record['oob_macro_f1']:.4f}"
            )

    result_df = pd.DataFrame(records).sort_values(
        by=["oob_macro_f1", "oob_accuracy", "oob_score_value", "n_estimators"],
        ascending=[False, False, False, True],
    )
    result_df.to_csv(RESULT_DIR / "rf_tuning_results.csv", index=False)

    best_params = {key: best_record[key] for key in keys}
    best_meta = {
        "oob_accuracy": round(float(best_record["oob_accuracy"]), 4),
        "oob_macro_f1": round(float(best_record["oob_macro_f1"]), 4),
        "oob_score_value": round(float(best_record["oob_score_value"]), 4),
    }

    print(f"  Best params: {best_params}")
    print(f"  Best OOB accuracy: {best_meta['oob_accuracy']:.4f}")
    print(f"  Best OOB macro-F1: {best_meta['oob_macro_f1']:.4f}")
    print(f"  Best OOB score during tuning: {best_meta['oob_score_value']:.4f}")
    return best_params, result_df, best_meta


def plot_feature_importance(
    importances: pd.Series,
    filename: str,
    title: str,
    top_n: int = 20,
    color: str = "#4573b4",
) -> None:
    top_features = importances.sort_values(ascending=False).head(top_n).sort_values()

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(top_features.index, top_features.values, color=color)
    ax.set_title(title)
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(RESULT_DIR / filename, bbox_inches="tight")
    plt.close()


def plot_permutation_importance_rf(
    model: RandomForestClassifier,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    top_n: int = 20,
) -> None:
    print("\n[STEP 5] Save permutation importance plot (merged test set)")
    perm = permutation_importance(
        model,
        X_eval,
        y_eval,
        scoring="f1_macro",
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )
    importances = pd.Series(perm.importances_mean, index=X_eval.columns)
    plot_feature_importance(
        importances=importances,
        filename="rf_feature_importance_permutation.png",
        title="Random Forest - Top 20 Permutation Importances (Merged Test Set)",
        top_n=top_n,
        color="#f28e2b",
    )


def plot_oob_curve(
    best_params: dict,
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
) -> None:
    print("\n[STEP 6] Save OOB error curve")

    curve_params = dict(best_params)
    curve_params["warm_start"] = True

    oob_errors = []
    oob_scores = []

    model = RandomForestClassifier(
        **curve_params,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )

    for n_trees in OOB_TREE_COUNTS:
        model.set_params(n_estimators=n_trees)
        model.fit(X_dev, y_dev)
        oob_scores.append(float(model.oob_score_))
        oob_errors.append(1 - float(model.oob_score_))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(OOB_TREE_COUNTS, oob_errors, marker="o", linewidth=1.8, color="#4573b4")
    ax.axvline(
        best_params["n_estimators"],
        linestyle="--",
        color="#d73221",
        linewidth=1.3,
        label=f"Selected n_estimators = {best_params['n_estimators']}",
    )
    ax.set_title("Random Forest - OOB Error vs Number of Trees")
    ax.set_xlabel("Number of Trees")
    ax.set_ylabel("OOB Error (1 - OOB Score)")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(framealpha=0.85)
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "rf_oob_vs_ntrees.png", bbox_inches="tight")
    plt.close()


def plot_multiclass_roc_curves(
    model: RandomForestClassifier,
    splits: dict[str, tuple[pd.DataFrame, pd.Series]],
) -> None:
    print("\n[STEP 7] Save multi-class ROC curves (OvR)")
    classes = np.array([0, 1, 2])
    n = len(splits)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.3 * nrows))
    if n == 1:
        axes = np.array([axes])
    axes = np.atleast_1d(axes).ravel()

    for ax, (split_name, (X_split, y_split)) in zip(axes, splits.items()):
        y_score = model.predict_proba(X_split)
        y_bin = label_binarize(y_split, classes=classes)

        for cls_idx, cls_id in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_bin[:, cls_idx], y_score[:, cls_idx])
            cls_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, linewidth=1.8, label=f"{CLASS_NAMES[cls_id]} AUC={cls_auc:.3f}")

        fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), y_score.ravel())
        micro_auc = auc(fpr_micro, tpr_micro)

        all_fpr = np.unique(
            np.concatenate([roc_curve(y_bin[:, idx], y_score[:, idx])[0] for idx in range(len(classes))])
        )
        mean_tpr = np.zeros_like(all_fpr)
        for idx in range(len(classes)):
            fpr, tpr, _ = roc_curve(y_bin[:, idx], y_score[:, idx])
            mean_tpr += np.interp(all_fpr, fpr, tpr)
        mean_tpr /= len(classes)
        macro_auc = auc(all_fpr, mean_tpr)

        ax.plot(all_fpr, mean_tpr, linestyle="--", linewidth=2.0, color="black", label=f"macro AUC={macro_auc:.3f}")
        ax.plot(fpr_micro, tpr_micro, linestyle=":", linewidth=2.0, color="#6a3d9a", label=f"micro AUC={micro_auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, linewidth=1)
        ax.set_title(split_name.upper())
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, framealpha=0.85, loc="lower right")

    for j in range(n, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("Random Forest - Multi-class ROC Curves (OvR)", y=0.99)
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "rf_roc_curves_ovr.png", bbox_inches="tight")
    plt.close()


def plot_decision_boundary_pca(
    best_params: dict,
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_full: pd.DataFrame,
    y_full: pd.Series,
) -> None:
    print("\n[STEP 8] PCA decision boundary visualization")
    print("  Note: this is a 2D visualization-only model, not the main evaluation model.")

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_dev_2d = pca.fit_transform(X_fit)
    X_full_2d = pca.transform(X_full)
    print(f"  PCA(2) explained variance ratio: {pca.explained_variance_ratio_.sum():.2%}")

    vis_model = RandomForestClassifier(
        **best_params,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )
    vis_model.fit(X_dev_2d, y_fit)

    x_min, x_max = X_full_2d[:, 0].min() - 0.8, X_full_2d[:, 0].max() + 0.8
    y_min, y_max = X_full_2d[:, 1].min() - 0.8, X_full_2d[:, 1].max() + 0.8
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 400),
        np.linspace(y_min, y_max, 400),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = vis_model.predict(grid).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.contourf(xx, yy, zz, alpha=0.20, levels=[-0.5, 0.5, 1.5, 2.5], cmap="RdYlBu")

    for cls in [2, 1, 0]:
        mask = y_full == cls
        ax.scatter(
            X_full_2d[mask, 0],
            X_full_2d[mask, 1],
            c=COLORS[cls],
            marker=MARKERS[cls],
            s=15,
            alpha=0.65,
            linewidths=0,
            label=CLASS_NAMES[cls],
            rasterized=True,
        )

    ax.set_title("Random Forest Decision Boundary on PCA-2D Space")
    ax.set_xlabel("PCA component 1")
    ax.set_ylabel("PCA component 2")
    ax.legend(framealpha=0.8)
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "rf_decision_boundary_pca.png", bbox_inches="tight")
    plt.close()


# ============================================================
# 2. Load datasets
# ============================================================
print("[STEP 0] Load datasets")
X_train, y_train = load_xy(TRAIN_PATH)
X_val10, y_val10 = load_xy(VAL_PATH)
X_test20, y_test20 = load_xy(TEST_PATH)
X_full, y_full = load_xy(FULL_PATH)
X_test = pd.concat([X_val10, X_test20], axis=0, ignore_index=True)
y_test = pd.concat([y_val10, y_test20], axis=0, ignore_index=True)

print(f"  Train: {X_train.shape}, y={y_train.shape}")
print(f"  Test (val+test_20): {X_test.shape}, y={y_test.shape}")
print(f"  Full:  {X_full.shape}, y={y_full.shape}")


# ============================================================
# 3. Hyperparameter search (OOB on training set)
# ============================================================
best_params, tuning_df, best_meta = manual_grid_search_oob_rf(X_train, y_train)

with open(RESULT_DIR / "rf_best_params.json", "w", encoding="utf-8") as f:
    json.dump({"best_params": best_params, "best_oob": best_meta}, f, indent=2)


# ============================================================
# 4. Final training
# ============================================================
print("\n[STEP 2] Train final model on training set only (70% split)")
X_fit = X_train
y_fit = y_train

final_model = RandomForestClassifier(
    **best_params,
    random_state=RANDOM_STATE,
    n_jobs=N_JOBS,
)
final_model.fit(X_fit, y_fit)

joblib.dump(final_model, RESULT_DIR / "rf_model.pkl")
print(f"  Final OOB score: {final_model.oob_score_:.4f}")
print(f"  Number of trees: {final_model.n_estimators}")


# ============================================================
# 5. Evaluation
# ============================================================
print("\n[STEP 3] Evaluate model on train / test (val+test_20) / full")
splits = {
    "train": (X_train, y_train),
    "test": (X_test, y_test),
    "full": (X_full, y_full),
}

metrics_summary = {
    "best_params": best_params,
    "best_oob": best_meta,
    "training_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
    "final_fit_rows": int(len(X_fit)),
    "final_oob_score": round(float(final_model.oob_score_), 4),
}

for split_name, (X_split, y_split) in splits.items():
    pred = final_model.predict(X_split)
    prob = final_model.predict_proba(X_split)
    metrics = calc_metrics(y_split, pred, prob)
    metrics_summary[split_name] = metrics
    print(f"  {split_name.upper()}: {metrics}")
    save_report(y_split, pred, split_name)
    plot_confusion(y_split, pred, split_name)

with open(RESULT_DIR / "rf_metrics_summary.json", "w", encoding="utf-8") as f:
    json.dump(metrics_summary, f, indent=2)


# ============================================================
# 6. Interpretability / visualization
# ============================================================
print("\n[STEP 4] Save impurity-based feature importance plot")
impurity_importances = pd.Series(final_model.feature_importances_, index=X_fit.columns)
plot_feature_importance(
    importances=impurity_importances,
    filename="rf_feature_importance.png",
    title="Random Forest - Top 20 Feature Importances",
    top_n=20,
)

plot_permutation_importance_rf(final_model, X_test, y_test, top_n=20)
plot_oob_curve(best_params, X_fit, y_fit)
plot_multiclass_roc_curves(final_model, splits)
plot_decision_boundary_pca(best_params, X_fit, y_fit, X_full, y_full)


# ============================================================
# 7. Final summary
# ============================================================
print("\n" + "=" * 70)
print("Random Forest task completed")
print("=" * 70)
print("Saved files:")
print("  - rf_best_params.json")
print("  - rf_tuning_results.csv")
print("  - rf_metrics_summary.json")
print("  - rf_classification_report_train.txt")
print("  - rf_classification_report_test.txt (validation + test_20 merged)")
print("  - rf_classification_report_full.txt")
print("  - rf_confusion_train.png")
print("  - rf_confusion_test.png (validation + test_20 merged)")
print("  - rf_confusion_full.png")
print("  - rf_feature_importance.png")
print("  - rf_feature_importance_permutation.png")
print("  - rf_oob_vs_ntrees.png")
print("  - rf_roc_curves_ovr.png")
print("  - rf_decision_boundary_pca.png")
print("  - rf_model.pkl")
print("  - random_forest_log.txt")
