"""
Task 4 - Decision Tree Binary Classification
============================================
Binary target: Target_binary
  - 0 = Non-Dropout (Enrolled + Graduate)
  - 1 = Dropout

Input:
  - ../data_preprocessing_final/result/train_70.csv
  - ../data_preprocessing_final/result/val_10.csv
  - ../data_preprocessing_final/result/test_20.csv
  - ../data_preprocessing_final/result/full_preprocessed_data.csv

Output (all under result/, prefixed with dt_bin_):
  - dt_bin_log.txt
  - dt_bin_best_params.json
  - dt_bin_tuning_results.csv
  - dt_bin_metrics_summary.json
  - dt_bin_classification_report_{train,val,test,full}.txt
  - dt_bin_confusion_{train,val,test,full}.png
  - dt_bin_roc_curves.png
  - dt_bin_pr_curves.png
  - dt_bin_feature_importance.png
  - dt_bin_tree_plot.png
  - dt_bin_decision_boundary_pca.png
  - dt_bin_model.pkl
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
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.tree import DecisionTreeClassifier, plot_tree

warnings.filterwarnings("ignore")


# ============================================================
# 0. Paths / constants / logging
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data_preprocessing_final" / "result"
RESULT_DIR = BASE_DIR / "result"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / "train_70.csv"
VAL_PATH = DATA_DIR / "val_10.csv"
TEST_PATH = DATA_DIR / "test_20.csv"
FULL_PATH = DATA_DIR / "full_preprocessed_data.csv"

TARGET_COL = "Target"
TARGET_BIN_COL = "Target_binary"
RANDOM_STATE = 42

# Target_binary: 0 = Non-Dropout, 1 = Dropout
CLASS_NAMES = ["Non-Dropout", "Dropout"]
COLORS = {0: "#4573b4", 1: "#d73221"}
MARKERS = {0: "o", 1: "s"}


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


sys.stdout = Logger(RESULT_DIR / "dt_bin_log.txt")

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
def load_xy(csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)
    X = df.drop(columns=[TARGET_COL, TARGET_BIN_COL])
    y = df[TARGET_BIN_COL].astype(int)
    return df, X, y


def calc_metrics(y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(
            float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)), 4
        ),
        "recall": round(
            float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)), 4
        ),
        "f1": round(
            float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)), 4
        ),
        "macro_f1": round(
            float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4
        ),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "average_precision": round(float(average_precision_score(y_true, y_prob)), 4),
    }


def save_report(y_true: pd.Series, y_pred: np.ndarray, split_name: str) -> None:
    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
    report_path = RESULT_DIR / f"dt_bin_classification_report_{split_name}.txt"
    report_path.write_text(report, encoding="utf-8")


def plot_confusion(y_true: pd.Series, y_pred: np.ndarray, split_name: str) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=CLASS_NAMES,
        cmap="Blues",
        colorbar=False,
        ax=ax,
        values_format="d",
    )
    disp.ax_.set_title(f"DT Binary - Confusion Matrix ({split_name})")
    plt.tight_layout()
    plt.savefig(RESULT_DIR / f"dt_bin_confusion_{split_name}.png", bbox_inches="tight")
    plt.close()


def manual_grid_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> tuple[dict, pd.DataFrame]:
    grid = {
        "criterion": ["gini", "entropy", "log_loss"],
        "max_depth": [4, 6, 8, 10, 12, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 5, 10],
        "ccp_alpha": [0.0, 0.001, 0.003],
        "class_weight": ["balanced"],
    }

    keys = list(grid.keys())
    combos = list(product(*(grid[k] for k in keys)))
    print("=" * 70)
    print("Task 4 - Decision Tree Binary Classification")
    print("=" * 70)
    print("\n[STEP 1] Grid search on validation set")
    print(f"  Search combinations: {len(combos)}")

    records = []
    best_record = None

    for idx, values in enumerate(combos, start=1):
        params = dict(zip(keys, values))
        model = DecisionTreeClassifier(**params, random_state=RANDOM_STATE)
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        val_prob = model.predict_proba(X_val)[:, 1]

        record = {
            **params,
            "tree_depth": model.get_depth(),
            "leaf_count": model.get_n_leaves(),
            "val_accuracy": accuracy_score(y_val, val_pred),
            "val_f1": f1_score(y_val, val_pred, pos_label=1, zero_division=0),
            "val_roc_auc": roc_auc_score(y_val, val_prob),
        }
        records.append(record)

        # Selection: highest val_f1, tie-break by roc_auc, then accuracy, then shallower tree
        if best_record is None:
            best_record = record
        else:
            better = (
                record["val_f1"] > best_record["val_f1"]
                or (
                    record["val_f1"] == best_record["val_f1"]
                    and record["val_roc_auc"] > best_record["val_roc_auc"]
                )
                or (
                    record["val_f1"] == best_record["val_f1"]
                    and record["val_roc_auc"] == best_record["val_roc_auc"]
                    and record["val_accuracy"] > best_record["val_accuracy"]
                )
                or (
                    record["val_f1"] == best_record["val_f1"]
                    and record["val_roc_auc"] == best_record["val_roc_auc"]
                    and record["val_accuracy"] == best_record["val_accuracy"]
                    and record["tree_depth"] < best_record["tree_depth"]
                )
            )
            if better:
                best_record = record

        if idx % 50 == 0 or idx == len(combos):
            print(
                f"  Progress: {idx:>3}/{len(combos)} | "
                f"Current best val F1 = {best_record['val_f1']:.4f} | "
                f"AUC = {best_record['val_roc_auc']:.4f}"
            )

    result_df = pd.DataFrame(records).sort_values(
        by=["val_f1", "val_roc_auc", "val_accuracy", "tree_depth"],
        ascending=[False, False, False, True],
    )
    result_df.to_csv(RESULT_DIR / "dt_bin_tuning_results.csv", index=False)

    best_params = {key: best_record[key] for key in keys}
    print(f"  Best params: {best_params}")
    print(f"  Best validation accuracy: {best_record['val_accuracy']:.4f}")
    print(f"  Best validation F1 (Dropout): {best_record['val_f1']:.4f}")
    print(f"  Best validation ROC-AUC: {best_record['val_roc_auc']:.4f}")
    return best_params, result_df


def plot_feature_importance(
    model: DecisionTreeClassifier, feature_names: list[str], top_n: int = 20
) -> None:
    importances = pd.Series(model.feature_importances_, index=feature_names)
    top_features = importances.sort_values(ascending=False).head(top_n).sort_values()

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(top_features.index, top_features.values, color="#4573b4")
    ax.set_title(f"DT Binary - Top {top_n} Feature Importances")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "dt_bin_feature_importance.png", bbox_inches="tight")
    plt.close()


def plot_tree_figure(model: DecisionTreeClassifier, feature_names: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(24, 12))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=CLASS_NAMES,
        filled=True,
        rounded=True,
        fontsize=8,
        max_depth=3,
        ax=ax,
    )
    ax.set_title("Decision Tree Binary (top 4 levels only)")
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "dt_bin_tree_plot.png", bbox_inches="tight")
    plt.close()


def plot_roc_curves(
    model: DecisionTreeClassifier, splits: dict
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for split_name, (X_split, y_split) in splits.items():
        y_prob = model.predict_proba(X_split)[:, 1]
        fpr, tpr, _ = roc_curve(y_split, y_prob)
        auc = roc_auc_score(y_split, y_prob)
        ax.plot(fpr, tpr, label=f"{split_name} (AUC = {auc:.4f})", linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.6, label="Chance")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("DT Binary - ROC Curves")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "dt_bin_roc_curves.png", bbox_inches="tight")
    plt.close()


def plot_pr_curves(
    model: DecisionTreeClassifier, splits: dict
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for split_name, (X_split, y_split) in splits.items():
        y_prob = model.predict_proba(X_split)[:, 1]
        precision, recall, _ = precision_recall_curve(y_split, y_prob)
        ap = average_precision_score(y_split, y_prob)
        ax.plot(recall, precision, label=f"{split_name} (AP = {ap:.4f})", linewidth=2)

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("DT Binary - Precision-Recall Curves")
    ax.legend(loc="lower left", framealpha=0.9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "dt_bin_pr_curves.png", bbox_inches="tight")
    plt.close()


def plot_decision_boundary_pca(
    best_params: dict,
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    X_full: pd.DataFrame,
    y_full: pd.Series,
) -> None:
    print("\n[STEP 7] PCA decision boundary visualization")
    print("  Note: this is a 2D visualization-only model, not the main evaluation model.")

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_dev_2d = pca.fit_transform(X_dev)
    X_full_2d = pca.transform(X_full)
    print(f"  PCA(2) explained variance ratio: {pca.explained_variance_ratio_.sum():.2%}")

    vis_model = DecisionTreeClassifier(**best_params, random_state=RANDOM_STATE)
    vis_model.fit(X_dev_2d, y_dev)

    x_min, x_max = X_full_2d[:, 0].min() - 0.8, X_full_2d[:, 0].max() + 0.8
    y_min, y_max = X_full_2d[:, 1].min() - 0.8, X_full_2d[:, 1].max() + 0.8
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 400),
        np.linspace(y_min, y_max, 400),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = vis_model.predict(grid).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.contourf(xx, yy, zz, alpha=0.22, levels=[-0.5, 0.5, 1.5], cmap="RdYlBu_r")

    for cls in [0, 1]:
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

    ax.set_title("DT Binary - Decision Boundary on PCA-2D Space")
    ax.set_xlabel("PCA component 1")
    ax.set_ylabel("PCA component 2")
    ax.legend(framealpha=0.8)
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "dt_bin_decision_boundary_pca.png", bbox_inches="tight")
    plt.close()


# ============================================================
# 2. Load datasets
# ============================================================
print("[STEP 0] Load datasets (binary target: 1 = Dropout, 0 = Non-Dropout)")
train_df, X_train, y_train = load_xy(TRAIN_PATH)
val_df, X_val, y_val = load_xy(VAL_PATH)
test_df, X_test, y_test = load_xy(TEST_PATH)
full_df, X_full, y_full = load_xy(FULL_PATH)

print(f"  Train: {X_train.shape}, pos-rate={y_train.mean():.4f}")
print(f"  Val:   {X_val.shape}, pos-rate={y_val.mean():.4f}")
print(f"  Test:  {X_test.shape}, pos-rate={y_test.mean():.4f}")
print(f"  Full:  {X_full.shape}, pos-rate={y_full.mean():.4f}")


# ============================================================
# 3. Hyperparameter search using validation set
# ============================================================
best_params, tuning_df = manual_grid_search(X_train, y_train, X_val, y_val)

with open(RESULT_DIR / "dt_bin_best_params.json", "w", encoding="utf-8") as f:
    json.dump(best_params, f, indent=2)


# ============================================================
# 4. Final training (train + validation)
# ============================================================
print("\n[STEP 2] Train final model on train + validation")
X_dev = pd.concat([X_train, X_val], axis=0, ignore_index=True)
y_dev = pd.concat([y_train, y_val], axis=0, ignore_index=True)

final_model = DecisionTreeClassifier(**best_params, random_state=RANDOM_STATE)
final_model.fit(X_dev, y_dev)

joblib.dump(final_model, RESULT_DIR / "dt_bin_model.pkl")
print(f"  Final tree depth: {final_model.get_depth()}")
print(f"  Final leaf count: {final_model.get_n_leaves()}")


# ============================================================
# 5. Evaluation
# ============================================================
print("\n[STEP 3] Evaluate model on train / val / test / full")
splits = {
    "train": (X_train, y_train),
    "val": (X_val, y_val),
    "test": (X_test, y_test),
    "full": (X_full, y_full),
}

metrics_summary = {
    "best_params": best_params,
    "training_rows": int(len(X_train)),
    "validation_rows": int(len(X_val)),
    "final_fit_rows": int(len(X_dev)),
    "positive_label": "Dropout (1)",
}

for split_name, (X_split, y_split) in splits.items():
    pred = final_model.predict(X_split)
    prob = final_model.predict_proba(X_split)[:, 1]
    metrics = calc_metrics(y_split, pred, prob)
    metrics_summary[split_name] = metrics
    print(f"  {split_name.upper()}: {metrics}")
    save_report(y_split, pred, split_name)
    plot_confusion(y_split, pred, split_name)

with open(RESULT_DIR / "dt_bin_metrics_summary.json", "w", encoding="utf-8") as f:
    json.dump(metrics_summary, f, indent=2)


# ============================================================
# 6. Interpretability / visualization
# ============================================================
print("\n[STEP 4] Save ROC curves")
plot_roc_curves(final_model, splits)

print("\n[STEP 5] Save Precision-Recall curves")
plot_pr_curves(final_model, splits)

print("\n[STEP 6] Save feature importance and tree plot")
plot_feature_importance(final_model, X_dev.columns.tolist(), top_n=20)
plot_tree_figure(final_model, X_dev.columns.tolist())

plot_decision_boundary_pca(best_params, X_dev, y_dev, X_full, y_full)


# ============================================================
# 7. Final summary
# ============================================================
print("\n" + "=" * 70)
print("Decision Tree Binary task completed")
print("=" * 70)
print("Saved files:")
print("  - dt_bin_best_params.json")
print("  - dt_bin_tuning_results.csv")
print("  - dt_bin_metrics_summary.json")
print("  - dt_bin_classification_report_train.txt")
print("  - dt_bin_classification_report_val.txt")
print("  - dt_bin_classification_report_test.txt")
print("  - dt_bin_classification_report_full.txt")
print("  - dt_bin_confusion_train.png")
print("  - dt_bin_confusion_val.png")
print("  - dt_bin_confusion_test.png")
print("  - dt_bin_confusion_full.png")
print("  - dt_bin_roc_curves.png")
print("  - dt_bin_pr_curves.png")
print("  - dt_bin_feature_importance.png")
print("  - dt_bin_tree_plot.png")
print("  - dt_bin_decision_boundary_pca.png")
print("  - dt_bin_model.pkl")
print("  - dt_bin_log.txt")
