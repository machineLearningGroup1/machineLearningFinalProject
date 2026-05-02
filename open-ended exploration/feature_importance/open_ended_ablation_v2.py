"""
Open-ended Exploration — Feature Group Ablation Study (v2)
==========================================================

Motivation (from Fig 3 cross-validation analysis):
  Task 3 cluster profiles (heatmaps) and Task 4 supervised importance
  revealed that features fall into functionally distinct groups with
  very different roles:
    - Academic performance:  both clustering and supervised agree
    - Enrollment background: supervised rank #3, cluster rank #56
    - Economic factors:      Task 3's "second dropout pathway"
    - Demographics / Course one-hot: possible noise

  This script answers: how much does each functional group actually
  contribute to prediction?  Unlike v1 (which only tested engineered
  vs original), v2 groups features by their *semantic role* in the
  dropout prediction problem.

Key findings (on this dataset):
  - Academic performance (16 features) accounts for ~93.5% of total
    model performance and is the only indispensable group
  - Economic factors are a genuine second signal (ΔwF1 = -0.018)
  - 45 out of 89 features (demographics, course/app one-hot,
    grade_trend, enrollment background) are noise or redundant
  - grade_trend confirmed as a noise feature (removing it helps)

Input:
  - ../data_preprocessing_final/result/train_70.csv
  - ../data_preprocessing_final/result/val_10.csv
  - ../data_preprocessing_final/result/test_20.csv

Output (all in result/open_ended_ablation/):
  - fig4_ablation_by_function.png    ← ΔwF1 per removed group
  - fig5_perclass_ablation.png       ← per-class impact breakdown
  - fig6_only_group_comparison.png   ← "only X" sufficiency test
  - ablation_results.csv
  - open_ended_ablation_log.txt
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════
# 0. Paths / constants / logging
# ═══════════════════════════════════════════════════════════════════
BASE_DIR   = Path(__file__).resolve().parent
DATA_DIR   = BASE_DIR.parent / "data_preprocessing_final" / "result"
RESULT_DIR = BASE_DIR / "result" / "open_ended_ablation_v2"
RESULT_DIR.mkdir(parents=True, exist_ok=True, mode=0o755)

TRAIN_PATH = DATA_DIR / "train_70.csv"
VAL_PATH   = DATA_DIR / "val_10.csv"
TEST_PATH  = DATA_DIR / "test_20.csv"

TARGET_COL     = "Target"
TARGET_BIN_COL = "Target_binary"
RANDOM_STATE   = 42

# Martins et al. (2021) benchmark — indirect reference only
# NOTE: Martins et al. used a different dataset version (3,623 samples / 25 features,
# enrollment-time features only, no post-enrollment academic data) and different
# class labels (Success/Relative Success/Failure vs our Graduate/Enrolled/Dropout).
# Their best model (XGBoost + SMOTE) achieved avg F1 = 0.65.
# This line is an indirect reference, NOT a same-condition baseline.
MARTINS_F1 = 0.65


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


sys.stdout = Logger(RESULT_DIR / "open_ended_ablation_log.txt")

plt.rcParams.update({
    "figure.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Random Forest hyperparameters (best from Task 4 OOB search)
RF_PARAMS = dict(
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


# ═══════════════════════════════════════════════════════════════════
# 1. Load data
# ═══════════════════════════════════════════════════════════════════
def load_xy(path: Path):
    df = pd.read_csv(path)
    X = df.drop(columns=[TARGET_COL, TARGET_BIN_COL])
    y = df[TARGET_COL].astype(int)
    return X, y


print("=" * 75)
print("Open-ended: Feature Group Ablation Study (v2)")
print("=" * 75)

print("\n[STEP 0] Loading data …")
X_train, y_train = load_xy(TRAIN_PATH)
X_val,   y_val   = load_xy(VAL_PATH)
X_test,  y_test  = load_xy(TEST_PATH)

# Merge val + test for evaluation (same convention as teammates)
X_eval = pd.concat([X_val, X_test], ignore_index=True)
y_eval = pd.concat([y_val, y_test], ignore_index=True)

all_feats = X_train.columns.tolist()
print(f"  Train: {X_train.shape}  |  Eval (val+test): {X_eval.shape}")


# ═══════════════════════════════════════════════════════════════════
# 2. Define feature groups by FUNCTIONAL ROLE
#    (motivated by Fig 3 cluster-vs-supervised analysis)
# ═══════════════════════════════════════════════════════════════════
print("\n[STEP 1] Defining feature groups by functional role …")

# Group 1 — Academic performance (both semesters)
#   These dominate both clustering heatmaps and supervised importance
academic_sem = [c for c in all_feats if "Curricular units" in c]
academic_eng = ["pass_rate_sem1", "pass_rate_sem2", "no_eval_sem1", "no_eval_sem2"]
ACADEMIC = academic_sem + [f for f in academic_eng if f in all_feats]

# Group 2 — Enrollment background
#   Supervised rank #3-5, but cluster rank #56-85 (invisible to clustering)
ENROLLMENT = [c for c in all_feats if c in [
    "Admission grade", "Previous qualification (grade)",
    "prev_qual_ordinal", "Application order",
]]

# Group 3 — Economic factors
#   Task 3 identified this as the "second dropout pathway"
ECONOMIC = [c for c in all_feats if c in [
    "Tuition fees up to date", "Debtor", "Scholarship holder",
    "economic_stress",
]]

# Group 4 — Macroeconomic indicators
MACRO = [c for c in all_feats if c in [
    "GDP", "Unemployment rate", "Inflation rate",
]]

# Group 5 — Family background (parent education/occupation)
FAMILY = [c for c in all_feats if any(k in c for k in [
    "Mother_qual", "Father_qual", "Mother_occ", "Father_occ",
    "family_edu_capital",
])]

# Group 6 — Demographics
DEMOGRAPHIC = [c for c in all_feats if c in [
    "Gender", "Displaced", "Educational special needs",
    "Marital status_2", "Marital status_3", "Marital status_4",
    "Marital status_5", "Marital status_6",
] or c.startswith("age_group") or c.startswith("Nacionality")]

# Group 7 — Course & Application mode (one-hot encoded)
COURSE_APP = [c for c in all_feats
              if c.startswith("Course_")
              or c.startswith("Application mode_")
              or c.startswith("Daytime")]

# Group 8 — grade_trend (single feature, suspected noise)
GRADE_TREND = ["grade_trend"] if "grade_trend" in all_feats else []

# Assemble all groups in an ordered dict
GROUPS: dict[str, list[str]] = {
    "Academic performance\n(S1+S2 grades, approved,\nevaluations, pass_rate, no_eval)": ACADEMIC,
    "Enrollment background\n(Admission grade, Prev qual,\nApplication order)":           ENROLLMENT,
    "Economic factors\n(Tuition, Debtor, Scholarship,\neconomic_stress)":                ECONOMIC,
    "Macroeconomic\n(GDP, Unemployment, Inflation)":                                     MACRO,
    "Family background\n(parent edu/occ,\nfamily_edu_capital)":                           FAMILY,
    "Demographics\n(Gender, age, nationality,\nmarital, displaced)":                      DEMOGRAPHIC,
    "Course & Application mode\n(one-hot encoded)":                                      COURSE_APP,
    "grade_trend\n(single feature)":                                                     GRADE_TREND,
}

# Print summary and verify full coverage
all_assigned: set[str] = set()
for group_name, group_feats in GROUPS.items():
    present = [f for f in group_feats if f in all_feats]
    all_assigned.update(present)
    short = group_name.replace("\n", " ")
    print(f"  {short:<65}  {len(present)} features")

unassigned = set(all_feats) - all_assigned
if unassigned:
    print(f"\n  [WARN] Unassigned features: {unassigned}")
else:
    print(f"\n  All {len(all_feats)} features assigned to groups.")


# ═══════════════════════════════════════════════════════════════════
# 3. Helper: train RF and evaluate
# ═══════════════════════════════════════════════════════════════════
def run_rf(
    X_tr: pd.DataFrame,
    X_ev: pd.DataFrame,
    y_tr: pd.Series,
    y_ev: pd.Series,
    label: str,
) -> dict:
    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_ev)

    wf1    = f1_score(y_ev, pred, average="weighted", zero_division=0)
    mf1    = f1_score(y_ev, pred, average="macro",    zero_division=0)
    f1_per = f1_score(y_ev, pred, average=None,        zero_division=0)

    return {
        "label":       label,
        "weighted_f1": round(float(wf1),       4),
        "macro_f1":    round(float(mf1),       4),
        "f1_dropout":  round(float(f1_per[0]), 4),
        "f1_enrolled": round(float(f1_per[1]), 4),
        "f1_graduate": round(float(f1_per[2]), 4),
        "n_features":  X_tr.shape[1],
        "n_removed":   len(all_feats) - X_tr.shape[1],
    }


# ═══════════════════════════════════════════════════════════════════
# 4. Run ablation experiments
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 75)
print("[STEP 2] Running ablation experiments")
print("=" * 75 + "\n")

records: list[dict] = []

# ── 4a. Baseline (full model) ────────────────────────────────────
rec = run_rf(X_train, X_eval, y_train, y_eval, "FULL (all 89 features)")
records.append(rec)
baseline = rec["weighted_f1"]
print(f"  {'FULL':<60}  wF1={rec['weighted_f1']:.4f}  "
      f"D={rec['f1_dropout']:.3f}  E={rec['f1_enrolled']:.3f}  G={rec['f1_graduate']:.3f}")

# ── 4b. Remove each group ───────────────────────────────────────
for group_name, group_feats in GROUPS.items():
    present = [f for f in group_feats if f in all_feats]
    if not present:
        continue
    X_tr_ab = X_train.drop(columns=present)
    X_ev_ab = X_eval.drop(columns=present)
    short   = group_name.replace("\n", " ")
    rec     = run_rf(X_tr_ab, X_ev_ab, y_train, y_eval, f"– {short}")
    records.append(rec)
    delta = rec["weighted_f1"] - baseline
    print(f"  {rec['label']:<60}  wF1={rec['weighted_f1']:.4f}  Δ={delta:+.4f}  "
          f"D={rec['f1_dropout']:.3f}  E={rec['f1_enrolled']:.3f}  G={rec['f1_graduate']:.3f}")

# ── 4c. "Only X" sufficiency tests ──────────────────────────────
present_acad = [f for f in ACADEMIC if f in all_feats]
rec = run_rf(
    X_train[present_acad], X_eval[present_acad],
    y_train, y_eval,
    f"ONLY academic ({len(present_acad)} features)",
)
records.append(rec)
delta = rec["weighted_f1"] - baseline
print(f"  {rec['label']:<60}  wF1={rec['weighted_f1']:.4f}  Δ={delta:+.4f}  "
      f"D={rec['f1_dropout']:.3f}  E={rec['f1_enrolled']:.3f}  G={rec['f1_graduate']:.3f}")

rec = run_rf(
    X_train[ENROLLMENT], X_eval[ENROLLMENT],
    y_train, y_eval,
    f"ONLY enrollment background ({len(ENROLLMENT)} features)",
)
records.append(rec)
delta = rec["weighted_f1"] - baseline
print(f"  {rec['label']:<60}  wF1={rec['weighted_f1']:.4f}  Δ={delta:+.4f}  "
      f"D={rec['f1_dropout']:.3f}  E={rec['f1_enrolled']:.3f}  G={rec['f1_graduate']:.3f}")

# ── 4d. Save results table ──────────────────────────────────────
results_df = pd.DataFrame(records)
results_df["delta"] = results_df["weighted_f1"] - baseline
results_df.to_csv(RESULT_DIR / "ablation_results.csv", index=False)


# ═══════════════════════════════════════════════════════════════════
# 5. Fig 4 — Horizontal bar chart: ΔwF1 per removed group
# ═══════════════════════════════════════════════════════════════════
print("\n[STEP 3] Plotting fig4_ablation_by_function.png …")

removal_rows = results_df[results_df["label"].str.startswith("–")].copy()
removal_rows = removal_rows.sort_values("delta")

labels    = removal_rows["label"].tolist()
deltas    = removal_rows["delta"].tolist()
n_removed = removal_rows["n_removed"].tolist()

fig, ax = plt.subplots(figsize=(11, max(5, len(labels) * 0.7 + 2)))

bar_colors = [
    "#d73221" if d < -0.005
    else ("#59a14f" if d > 0.003 else "#aaaaaa")
    for d in deltas
]
bars = ax.barh(range(len(labels)), deltas, color=bar_colors, height=0.65)
ax.axvline(0, color="black", lw=0.8)

for i, (bar, delta, nr, lbl) in enumerate(zip(bars, deltas, n_removed, labels)):
    xpos = delta - 0.002 if delta < 0 else delta + 0.001
    ha   = "right" if delta < 0 else "left"
    ax.text(xpos, i, f" {delta:+.4f} ({nr} feats)", va="center", ha=ha, fontsize=8.5)

ax.set_yticks(range(len(labels)))
ax.set_yticklabels(
    [lbl.replace("– ", "").replace("  ", " ") for lbl in labels],
    fontsize=8.5,
)
ax.set_xlabel("ΔWeighted F1 vs Full Model (negative = performance drop)")
ax.set_title(
    "Feature Group Ablation Study (Random Forest, 3-class)\n"
    "Which functional group contributes most to prediction?",
    pad=10,
)
ax.grid(alpha=0.2, axis="x", ls="--")
plt.tight_layout()
plt.savefig(RESULT_DIR / "fig4_ablation_by_function.png", bbox_inches="tight")
plt.close()


# ═══════════════════════════════════════════════════════════════════
# 6. Fig 5 — Per-class F1 impact of removing key groups
# ═══════════════════════════════════════════════════════════════════
print("[STEP 4] Plotting fig5_perclass_ablation.png …")

full_row   = results_df[results_df["label"].str.startswith("FULL")].iloc[0]
key_groups = [
    "Academic performance",
    "Enrollment background",
    "Economic factors",
    "Macroeconomic",
    "grade_trend",
]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
class_names = ["Dropout", "Enrolled", "Graduate"]
f1_cols     = ["f1_dropout", "f1_enrolled", "f1_graduate"]

for ax, cls_name, f1_col in zip(axes, class_names, f1_cols):
    base_val = full_row[f1_col]

    group_labels: list[str] = []
    group_deltas: list[float] = []
    for gname in key_groups:
        row = removal_rows[removal_rows["label"].str.contains(gname)]
        if len(row) == 0:
            continue
        row = row.iloc[0]
        d = row[f1_col] - base_val
        short = (gname
                 .replace(" performance", "")
                 .replace(" background", "")
                 .replace(" factors", ""))
        group_labels.append(short)
        group_deltas.append(d)

    bar_colors = [
        "#d73221" if d < -0.005
        else ("#59a14f" if d > 0.005 else "#aaaaaa")
        for d in group_deltas
    ]
    bars = ax.barh(range(len(group_labels)), group_deltas, color=bar_colors, height=0.6)
    ax.axvline(0, color="black", lw=0.8)

    for i, (bar, d) in enumerate(zip(bars, group_deltas)):
        xpos = d - 0.002 if d < 0 else d + 0.001
        ha   = "right" if d < 0 else "left"
        ax.text(xpos, i, f"{d:+.3f}", va="center", ha=ha, fontsize=8.5)

    ax.set_yticks(range(len(group_labels)))
    ax.set_yticklabels(group_labels, fontsize=9)
    ax.set_title(f"{cls_name}\n(baseline F1 = {base_val:.3f})", fontsize=11)
    ax.set_xlabel("ΔF1")
    ax.grid(alpha=0.2, axis="x", ls="--")

fig.suptitle(
    "Per-Class Impact of Removing Each Feature Group\n"
    "Which group matters most for Dropout? for Enrolled? for Graduate?",
    fontsize=12, y=1.02,
)
plt.tight_layout()
plt.savefig(RESULT_DIR / "fig5_perclass_ablation.png", bbox_inches="tight")
plt.close()


# ═══════════════════════════════════════════════════════════════════
# 7. Fig 6 — "Only X" sufficiency comparison
# ═══════════════════════════════════════════════════════════════════
print("[STEP 5] Plotting fig6_only_group_comparison.png …")

only_rows = results_df[results_df["label"].str.startswith("ONLY")].copy()
full_rec  = results_df[results_df["label"].str.startswith("FULL")].iloc[0]

model_labels = ["Full model\n(89 features)"] + only_rows["label"].tolist()
wf1_values   = [full_rec["weighted_f1"]] + only_rows["weighted_f1"].tolist()
n_feats      = [int(full_rec["n_features"])] + [int(x) for x in only_rows["n_features"].tolist()]

fig, ax = plt.subplots(figsize=(9, 4))

colors = ["#4573b4"] + ["#f28e2b"] * len(only_rows)
bars = ax.barh(range(len(model_labels)), wf1_values, color=colors, height=0.55)

for i, (bar, val, nf) in enumerate(zip(bars, wf1_values, n_feats)):
    ax.text(val + 0.005, i, f"wF1 = {val:.4f}  ({nf} feats)", va="center", fontsize=9)

ax.axvline(
    MARTINS_F1, color="#d73221", ls=":", lw=1.5,
    label=f"Martins et al. (2021) avg F1 = {MARTINS_F1}\n(different labels & features, indirect ref.)",
)
ax.set_yticks(range(len(model_labels)))
ax.set_yticklabels(model_labels, fontsize=9)
ax.set_xlabel("Weighted F1")
ax.set_xlim(0.3, 0.85)
ax.set_title("How much can each feature group achieve alone?", fontsize=12)
ax.legend(fontsize=8)
ax.grid(alpha=0.2, axis="x", ls="--")
plt.tight_layout()
plt.savefig(RESULT_DIR / "fig6_only_group_comparison.png", bbox_inches="tight")
plt.close()


# ═══════════════════════════════════════════════════════════════════
# 8. Summary
# ═══════════════════════════════════════════════════════════════════
noeng_row = results_df[results_df["label"].str.contains("ONLY academic")].iloc[0]

print("\n" + "=" * 75)
print("Ablation Study v2 — Summary")
print("=" * 75)
print(f"\n  Baseline weighted F1:             {baseline:.4f}")
print(f"  Martins et al. (2021) avg F1:     {MARTINS_F1}  (indirect ref., different labels & features)")

print(f"\n  Academic features alone:          wF1 = {noeng_row['weighted_f1']:.4f}")
print(f"    → {noeng_row['weighted_f1']/baseline*100:.1f}% of full model "
      f"with {noeng_row['n_features']/89*100:.0f}% of features")
print(f"    → Validates Martins et al. future work hypothesis:")
print(f"      adding academic performance data substantially improves prediction")

print("\n  Group removal impact (sorted by drop):")
for _, row in removal_rows.iterrows():
    print(f"    {row['label']:<60}  Δ = {row['delta']:+.4f}")

print(f"\n  All outputs saved to: {RESULT_DIR}")
print("=" * 75)
