"""
Open-ended Exploration — Feature Importance Cross-Model Analysis
================================================================
Story:
  Task 3 (clustering) found that no_eval_sem2 (Cramér's V = 0.82) and
  no_eval_sem1 (0.73) are the strongest cluster-discriminating features.
  This script asks: do supervised models agree?

  We compare feature importances from three tree-based models
  (Random Forest, XGBoost, CatBoost) against the Task 3 Cramér's V
  rankings, producing a unified "cross-validation" visualization.

Input:
  - ../data_preprocessing_final/result/train_70.csv
  - ../data_preprocessing_final/result/val_10.csv
  - ../data_preprocessing_final/result/test_20.csv

Output (all in result/open_ended_feature_importance/):
  - fig1_importance_comparison.png   ← 3-panel bar chart (RF / XGB / CatBoost)
  - fig2_top15_heatmap.png           ← rank heatmap across 4 sources
  - fig3_cramer_vs_supervised.png    ← scatter: Cramér's V vs mean supervised rank
  - feature_importance_table.csv     ← raw importance values
  - open_ended_fi_log.txt
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

# ── Try importing optional boosting libraries ────────────────────────────────
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[WARN] xgboost not installed — XGBoost panel will be skipped.")

try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
except ImportError:
    HAS_CAT = False
    print("[WARN] catboost not installed — CatBoost panel will be skipped.")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
DATA_DIR   = BASE_DIR.parent / "data_preprocessing_final" / "result"
RESULT_DIR = BASE_DIR / "result" / "open_ended_feature_importance"
RESULT_DIR.mkdir(parents=True, exist_ok=True, mode=0o755)

TRAIN_PATH = DATA_DIR / "train_70.csv"
VAL_PATH   = DATA_DIR / "val_10.csv"
TEST_PATH  = DATA_DIR / "test_20.csv"

TARGET_COL     = "Target"
TARGET_BIN_COL = "Target_binary"
RANDOM_STATE   = 42

# ── Logging ──────────────────────────────────────────────────────────────────
class Logger:
    def __init__(self, path: Path):
        self.terminal = sys.stdout
        self.log = open(path, "w", encoding="utf-8")

    def write(self, msg: str):
        self.terminal.write(msg)
        self.log.write(msg)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger(RESULT_DIR / "open_ended_fi_log.txt")

plt.rcParams.update({
    "figure.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ── Cramér's V values from Task 3 (top features, binary cross-tab result) ───
# Source: task3_binary_crosstab.py output, K-Means cluster association
CRAMERS_V = {
    "no_eval_sem2":                   0.819,
    "no_eval_sem1":                   0.734,
    "Tuition fees up to date":        0.278,
    "economic_stress":                0.241,
    "Gender":                         0.202,
    "Scholarship holder":             0.201,
    "Debtor":                         0.176,
    "Displaced":                      0.118,
    "Educational special needs":      0.085,
    "International_1":                0.070,
}

# ── Colour palette (consistent with project) ────────────────────────────────
COLORS = {
    "RF":       "#4573b4",   # blue
    "XGBoost":  "#d73221",   # red
    "CatBoost": "#f28e2b",   # orange
    "CramerV":  "#59a14f",   # green
}

# ============================================================
# 1. Load data
# ============================================================
print("=" * 65)
print("Open-ended: Feature Importance Cross-Model Analysis")
print("=" * 65)

def load_xy(path: Path):
    df = pd.read_csv(path)
    X = df.drop(columns=[TARGET_COL, TARGET_BIN_COL])
    y = df[TARGET_COL].astype(int)
    return X, y

print("\n[STEP 1] Loading data …")
X_train, y_train = load_xy(TRAIN_PATH)
X_val,   y_val   = load_xy(VAL_PATH)
X_test,  y_test  = load_xy(TEST_PATH)

# Merge val + test for evaluation (same convention as teammates)
X_eval = pd.concat([X_val, X_test], ignore_index=True)
y_eval = pd.concat([y_val, y_test], ignore_index=True)

feat_names = X_train.columns.tolist()
print(f"  Train: {X_train.shape}  |  Eval (val+test): {X_eval.shape}")

# ============================================================
# 2. Train three models with best hyperparams from Task 4/5
# ============================================================
print("\n[STEP 2] Training models …")

# ── Random Forest (best params from 04_random_forest.py OOB search) ─────────
print("  [2a] Random Forest …")
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=1,
    max_features="sqrt",
    class_weight="balanced",
    bootstrap=True,
    oob_score=True,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
rf.fit(X_train, y_train)
print(f"       OOB score = {rf.oob_score_:.4f}")

# ── XGBoost (best params from XGBoost.ipynb grid search) ────────────────────
if HAS_XGB:
    print("  [2b] XGBoost …")
    from sklearn.utils.class_weight import compute_sample_weight
    sw = compute_sample_weight("balanced", y_train)
    xgb_model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        gamma=0.1,
        learning_rate=0.05,
        max_depth=4,
        n_estimators=300,
        random_state=RANDOM_STATE,
        eval_metric="mlogloss",
        verbosity=0,
    )
    xgb_model.fit(X_train, y_train, sample_weight=sw)
    print("       Done.")

# ── CatBoost (best params from Catboost.ipynb) ───────────────────────────────
if HAS_CAT:
    print("  [2c] CatBoost …")
    cat_model = CatBoostClassifier(
        bootstrap_type="Bayesian",
        depth=6,
        iterations=500,
        learning_rate=0.05,
        auto_class_weights="Balanced",
        random_seed=RANDOM_STATE,
        verbose=0,
    )
    cat_model.fit(X_train, y_train)
    print("       Done.")

# ============================================================
# 3. Extract feature importances (top N)
# ============================================================
TOP_N = 20
print(f"\n[STEP 3] Extracting feature importances (top {TOP_N}) …")

importances: dict[str, pd.Series] = {}

# RF — impurity-based
rf_imp = pd.Series(rf.feature_importances_, index=feat_names)
importances["RF"] = rf_imp.nlargest(TOP_N)
print(f"  RF top feature: {rf_imp.idxmax()} ({rf_imp.max():.4f})")

if HAS_XGB:
    xgb_scores = xgb_model.get_booster().get_fscore()   # default: weight
    xgb_imp = pd.Series(xgb_scores).reindex(feat_names).fillna(0)
    # Normalise to sum=1 for fair cross-model comparison
    xgb_imp = xgb_imp / xgb_imp.sum()
    importances["XGBoost"] = xgb_imp.nlargest(TOP_N)
    print(f"  XGBoost top feature: {xgb_imp.idxmax()} ({xgb_imp.max():.4f})")

if HAS_CAT:
    cat_imp = pd.Series(
        cat_model.get_feature_importance(),
        index=feat_names,
    )
    cat_imp = cat_imp / cat_imp.sum()
    importances["CatBoost"] = cat_imp.nlargest(TOP_N)
    print(f"  CatBoost top feature: {cat_imp.idxmax()} ({cat_imp.max():.4f})")

# Save raw table
all_imp_df = pd.DataFrame({
    "RF_importance":       pd.Series(rf.feature_importances_, index=feat_names),
    "XGBoost_importance":  (pd.Series(xgb_scores, index=list(xgb_scores.keys())).reindex(feat_names).fillna(0) / pd.Series(xgb_scores).sum()) if HAS_XGB else np.nan,
    "CatBoost_importance": (pd.Series(cat_model.get_feature_importance(), index=feat_names) / pd.Series(cat_model.get_feature_importance()).sum()) if HAS_CAT else np.nan,
}).sort_values("RF_importance", ascending=False)
all_imp_df.to_csv(RESULT_DIR / "feature_importance_table.csv")
print("  Saved: feature_importance_table.csv")

# ============================================================
# 4. Fig 1 — Side-by-side bar charts (one panel per model)
# ============================================================
print("\n[STEP 4] Plotting fig1_importance_comparison.png …")

active_models = [m for m in ["RF", "XGBoost", "CatBoost"] if m in importances]
n_panels = len(active_models)

fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 7))
if n_panels == 1:
    axes = [axes]

for ax, model_name in zip(axes, active_models):
    series = importances[model_name].sort_values()
    color  = COLORS[model_name]

    # Highlight engineered features in a darker shade
    engineered = {
        "no_eval_sem1", "no_eval_sem2",
        "pass_rate_sem1", "pass_rate_sem2",
        "grade_trend", "economic_stress",
        "family_edu_capital", "prev_qual_ordinal",
    }
    bar_colors = [
        "#1a3a6b" if feat in engineered else color
        for feat in series.index
    ]

    bars = ax.barh(series.index, series.values, color=bar_colors)
    ax.set_title(f"{model_name}\nTop {TOP_N} Feature Importances", pad=8)
    ax.set_xlabel("Normalised Importance")
    ax.tick_params(axis="y", labelsize=8)

    # Engineered feature legend patch
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#1a3a6b", label="Engineered feature"),
        Patch(facecolor=color,     label="Original feature"),
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc="lower right")

fig.suptitle(
    "Feature Importance Comparison across Three Tree-Based Models\n"
    "(Dark bars = features engineered in Task 1 preprocessing)",
    fontsize=13, y=1.01,
)
plt.tight_layout()
plt.savefig(RESULT_DIR / "fig1_importance_comparison.png", bbox_inches="tight")
plt.close()
print("  Saved: fig1_importance_comparison.png")

# ============================================================
# 5. Fig 2 — Rank heatmap: RF / XGBoost / CatBoost / Cramér's V
# ============================================================
print("\n[STEP 5] Plotting fig2_top15_heatmap.png …")

# Build rank dataframes (rank 1 = most important)
def rank_series(s: pd.Series, n: int = 15) -> pd.Series:
    """Return rank (1-based) for top-n features; others get rank n+1."""
    top = s.nlargest(n)
    ranks = {feat: r + 1 for r, feat in enumerate(top.index)}
    return ranks

rf_ranks  = rank_series(pd.Series(rf.feature_importances_, index=feat_names))

sources = {"RF": rf_ranks}

if HAS_XGB:
    xgb_full = pd.Series(xgb_scores, index=list(xgb_scores.keys())).reindex(feat_names).fillna(0)
    sources["XGBoost"] = rank_series(xgb_full)

if HAS_CAT:
    cat_full = pd.Series(cat_model.get_feature_importance(), index=feat_names)
    sources["CatBoost"] = rank_series(cat_full)

# Cramér's V ranks
cramer_series = pd.Series(CRAMERS_V)
sources["Cramér's V\n(Task 3)"] = {feat: r + 1 for r, feat in enumerate(cramer_series.nlargest(15).index)}

# Union of top-15 features across all sources
all_top_feats = set()
for ranks in sources.values():
    all_top_feats.update(ranks.keys())

# Build matrix
PLACEHOLDER = 16   # rank for "not in top 15"
rank_df = pd.DataFrame(index=sorted(all_top_feats))
for src_name, ranks in sources.items():
    rank_df[src_name] = rank_df.index.map(lambda f, r=ranks: r.get(f, PLACEHOLDER))

# Sort by average rank
rank_df["avg"] = rank_df.mean(axis=1)
rank_df = rank_df.sort_values("avg").drop(columns="avg")

fig, ax = plt.subplots(figsize=(max(8, len(sources) * 2), len(rank_df) * 0.45 + 1.5))
im = ax.imshow(rank_df.values.T, cmap="YlOrRd_r", aspect="auto", vmin=1, vmax=PLACEHOLDER)

ax.set_xticks(range(len(rank_df)))
ax.set_xticklabels(rank_df.index, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(len(rank_df.columns)))
ax.set_yticklabels(rank_df.columns, fontsize=9)

# Annotate cells
for i in range(len(rank_df)):
    for j in range(len(rank_df.columns)):
        val = int(rank_df.values[i, j])
        label = str(val) if val < PLACEHOLDER else "–"
        ax.text(i, j, label, ha="center", va="center", fontsize=8,
                color="white" if val <= 5 else "black")

plt.colorbar(im, ax=ax, label="Rank (1 = most important, – = outside top 15)")
ax.set_title(
    "Feature Importance Rankings: Supervised Models vs Task 3 Cramér's V\n"
    "(Darker = higher rank / more important)",
    pad=12,
)
plt.tight_layout()
plt.savefig(RESULT_DIR / "fig2_top15_heatmap.png", bbox_inches="tight")
plt.close()
print("  Saved: fig2_top15_heatmap.png")

# ============================================================
# 6. Fig 3 — Scatter: Cramér's V vs mean supervised importance rank
# ============================================================
print("\n[STEP 6] Plotting fig3_cramer_vs_supervised.png …")

# Only features that appear in Cramér's V dict
cramer_feats = list(CRAMERS_V.keys())

# Mean normalised importance across available supervised models
sup_models_imp: dict[str, pd.Series] = {}
sup_models_imp["RF"] = pd.Series(rf.feature_importances_, index=feat_names)
if HAS_XGB:
    sup_models_imp["XGBoost"] = xgb_full / xgb_full.sum()
if HAS_CAT:
    sup_models_imp["CatBoost"] = cat_full / cat_full.sum()

mean_sup = pd.concat(sup_models_imp.values(), axis=1).mean(axis=1)

# Align on Cramér's V features
plot_feats = [f for f in cramer_feats if f in mean_sup.index]
x_vals = pd.Series(CRAMERS_V)[plot_feats].values
y_vals = mean_sup[plot_feats].values

fig, ax = plt.subplots(figsize=(8, 5))

scatter_colors = [
    "#1a3a6b" if f in {
        "no_eval_sem1", "no_eval_sem2",
        "economic_stress",
    } else "#4573b4"
    for f in plot_feats
]

ax.scatter(x_vals, y_vals, c=scatter_colors, s=90, zorder=3)

for feat, x, y in zip(plot_feats, x_vals, y_vals):
    ax.annotate(
        feat.replace("Tuition fees up to date", "Tuition fees\nup to date"),
        (x, y),
        textcoords="offset points",
        xytext=(6, 4),
        fontsize=7.5,
        color="#333333",
    )

# Trend line
if len(x_vals) > 2:
    z = np.polyfit(x_vals, y_vals, 1)
    p = np.poly1d(z)
    xs = np.linspace(x_vals.min(), x_vals.max(), 100)
    ax.plot(xs, p(xs), "--", color="#d73221", alpha=0.7, linewidth=1.5,
            label="Linear trend")

from scipy.stats import pearsonr, spearmanr
r, pval = pearsonr(x_vals, y_vals)
rho, _ = spearmanr(x_vals, y_vals)

ax.set_xlabel("Cramér's V  (Task 3 — unsupervised cluster association)", fontsize=10)
ax.set_ylabel("Mean Normalised Importance\n(Supervised models average)", fontsize=10)
ax.set_title(
    f"Unsupervised vs Supervised Feature Agreement\n"
    f"Pearson r = {r:.2f}  |  Spearman ρ = {rho:.2f}  (p = {pval:.3f})",
    fontsize=11,
)
ax.legend(fontsize=9)
ax.grid(alpha=0.25, linestyle="--")

from matplotlib.patches import Patch
legend_extra = [
    Patch(facecolor="#1a3a6b", label="no_eval / economic_stress (engineered)"),
    Patch(facecolor="#4573b4", label="Original feature"),
]
ax.legend(handles=legend_extra + ax.get_legend_handles_labels()[0], fontsize=8)

plt.tight_layout()
plt.savefig(RESULT_DIR / "fig3_cramer_vs_supervised.png", bbox_inches="tight")
plt.close()
print("  Saved: fig3_cramer_vs_supervised.png")

# ============================================================
# 7. Summary print
# ============================================================
print("\n" + "=" * 65)
print("Feature Importance Analysis — Key Findings")
print("=" * 65)

print("\nRF Top 10:")
rf_top10 = pd.Series(rf.feature_importances_, index=feat_names).nlargest(10)
for rank, (feat, val) in enumerate(rf_top10.items(), 1):
    tag = " ← ENGINEERED" if feat in {
        "no_eval_sem1","no_eval_sem2","pass_rate_sem1","pass_rate_sem2",
        "grade_trend","economic_stress","family_edu_capital","prev_qual_ordinal"
    } else ""
    print(f"  {rank:>2}. {feat:<45} {val:.4f}{tag}")

if HAS_XGB:
    print("\nXGBoost Top 10:")
    xgb_top10 = xgb_full.nlargest(10)
    for rank, (feat, val) in enumerate(xgb_top10.items(), 1):
        tag = " ← ENGINEERED" if feat in {
            "no_eval_sem1","no_eval_sem2","pass_rate_sem1","pass_rate_sem2",
            "grade_trend","economic_stress","family_edu_capital","prev_qual_ordinal"
        } else ""
        print(f"  {rank:>2}. {feat:<45} {val:.4f}{tag}")

print(f"\nPearson r (Cramér's V vs supervised): {r:.3f}")
print(f"Spearman ρ:                            {rho:.3f}")
print(f"\nAll outputs saved to: {RESULT_DIR}")
print("=" * 65)
