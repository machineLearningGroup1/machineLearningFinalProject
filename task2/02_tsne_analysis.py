"""
Task 2 — t-SNE Visualization
==============================
输入: full_preprocessed_data.csv
输出:
  fig1_tsne_target_vA.png        — 版本A（23列连续特征）按Target着色
  fig2_tsne_target_vB.png        — 版本B（全89列+PCA50）按Target着色
  fig3_tsne_feature_colors.png   — 版本A按4个关键特征分别着色（2×2子图）
  fig4_tsne_perplexity.png       — perplexity敏感性分析（1×3子图）
  fig5_tsne_3d.png               — 3D t-SNE（n_components=3）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import warnings, time
warnings.filterwarnings('ignore')

# ============================================================
# 0. 全局设置
# ============================================================
plt.rcParams.update({
    'figure.dpi': 300,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

RANDOM_STATE = 42
ALPHA        = 0.7   # 散点透明度
SIZE         = 12     # 散点大小

# 颜色方案（三类）
COLORS  = {0: '#d73221', 1: '#fcb777', 2: '#4573b4'}
MARKERS = {0: 's',       1: '^',       2: 'o'}
LABELS  = {0: 'Dropout', 1: 'Enrolled', 2: 'Graduate'}

# ============================================================
# 1. 读取数据
# ============================================================
print("=" * 60)
print("Task 2 — t-SNE Visualization")
print("=" * 60)

df = pd.read_csv("full_preprocessed_data.csv")
print(f"数据形状: {df.shape}")
print(f"Target分布: {df['Target'].value_counts().to_dict()}")

y = df['Target'].values   # 0=Dropout, 1=Enrolled, 2=Graduate

# ============================================================
# 2. 定义两个特征子集
# ============================================================

# 版本A：只用23个连续/有序特征（信噪比高，图形通常更清晰）
CONT_COLS = [
    'Application order',
    'Previous qualification (grade)',
    'Admission grade',
    'Curricular units 1st sem (credited)',
    'Curricular units 1st sem (enrolled)',
    'Curricular units 1st sem (evaluations)',
    'Curricular units 1st sem (approved)',
    'Curricular units 1st sem (grade)',
    'Curricular units 1st sem (without evaluations)',
    'Curricular units 2nd sem (credited)',
    'Curricular units 2nd sem (enrolled)',
    'Curricular units 2nd sem (evaluations)',
    'Curricular units 2nd sem (approved)',
    'Curricular units 2nd sem (grade)',
    'Curricular units 2nd sem (without evaluations)',
    'Unemployment rate', 'Inflation rate', 'GDP',
    'pass_rate_sem1', 'pass_rate_sem2',
    'grade_trend', 'prev_qual_ordinal', 'family_edu_capital',
]
X_vA = df[CONT_COLS].values
print(f"\n版本A 特征数: {X_vA.shape[1]}（连续/有序列）")

# 版本B：全89列，先PCA降至50维
X_all = df.drop(columns=['Target', 'Target_binary']).values
pca   = PCA(n_components=50, random_state=RANDOM_STATE)
X_vB  = pca.fit_transform(X_all)
var_50 = pca.explained_variance_ratio_.sum()
print(f"版本B 全89列 → PCA(50) 保留方差: {var_50:.1%}")

# ============================================================
# 辅助函数：绘制标准散点图
# ============================================================
def plot_tsne(ax, emb, y, title, legend=True):
    """在 ax 上画三类着色散点图"""
    for cls in [2, 1, 0]:   # Graduate最先画（底层），Dropout最后（最上层）
        mask = (y == cls)
        ax.scatter(
            emb[mask, 0], emb[mask, 1],
            c=COLORS[cls], marker=MARKERS[cls],
            s=SIZE, alpha=ALPHA,
            linewidths=0, label=LABELS[cls],
            rasterized=True,
        )
    ax.set_title(title)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.tick_params(left=False, bottom=False,
                   labelleft=False, labelbottom=False)
    if legend:
        handles = [
            mpatches.Patch(color=COLORS[c], label=LABELS[c])
            for c in [2, 1, 0]
        ]
        ax.legend(handles=handles, loc='best',
                  framealpha=0.7, fontsize=9)

# ============================================================
# 3. 图1 — 版本A：连续特征，perplexity=30
# ============================================================
print("\n[Fig 1] 版本A t-SNE (23 continuous features, perplexity=30)...")
t0 = time.time()

tsne_vA = TSNE(
    n_components=2,
    perplexity=30,
    max_iter=1000,
    init='pca',
    random_state=RANDOM_STATE,
    learning_rate='auto',
)
emb_vA = tsne_vA.fit_transform(X_vA)
print(f"  完成，耗时 {time.time()-t0:.1f}s，KL散度={tsne_vA.kl_divergence_:.4f}")

fig, ax = plt.subplots(figsize=(7, 6))
plot_tsne(ax, emb_vA, y,
          "t-SNE projection (23 continuous features, perplexity=30)")
plt.tight_layout()
plt.savefig("fig1_tsne_target_vA.png", bbox_inches='tight')
plt.close()
print("  已保存 fig1_tsne_target_vA.png")

# ============================================================
# 4. 图2 — 版本B：全89列+PCA50，perplexity=30
# ============================================================
print("\n[Fig 2] 版本B t-SNE (all 89 features + PCA50, perplexity=30)...")
t0 = time.time()

tsne_vB = TSNE(
    n_components=2,
    perplexity=30,
    max_iter=1000,
    init='pca',
    random_state=RANDOM_STATE,
    learning_rate='auto',
)
emb_vB = tsne_vB.fit_transform(X_vB)
print(f"  完成，耗时 {time.time()-t0:.1f}s，KL散度={tsne_vB.kl_divergence_:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
plot_tsne(axes[0], emb_vA, y,
          "Version A: 23 continuous features")
plot_tsne(axes[1], emb_vB, y,
          "Version B: all 89 features (PCA→50)")
fig.suptitle("t-SNE comparison: feature set A vs B (perplexity=30)",
             fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("fig2_tsne_target_vB.png", bbox_inches='tight')
plt.close()
print("  已保存 fig2_tsne_target_vB.png")

# ============================================================
# 5. 图3 — 按4个关键特征着色（2×2子图，使用版本A嵌入）
# ============================================================
print("\n[Fig 3] 多维度特征着色分析...")

feature_plots = [
    ('Curricular units 1st sem (approved)',
     '1st sem courses approved', 'viridis'),
    ('Tuition fees up to date',
     'Tuition fees up to date (1=yes)', 'RdYlGn'),
    ('Scholarship holder',
     'Scholarship holder (1=yes)', 'coolwarm'),
    ('economic_stress',
     'Economic stress (1=stressed)', 'RdYlGn_r'),
]

fig, axes = plt.subplots(2, 2, figsize=(13, 11))
axes = axes.flatten()

for i, (feat, title, cmap) in enumerate(feature_plots):
    ax = axes[i]
    vals = df[feat].values
    sc = ax.scatter(
        emb_vA[:, 0], emb_vA[:, 1],
        c=vals, cmap=cmap,
        s=SIZE, alpha=ALPHA,
        linewidths=0, rasterized=True,
    )
    plt.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
    ax.set_title(f"Colored by: {title}")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.tick_params(left=False, bottom=False,
                   labelleft=False, labelbottom=False)

fig.suptitle("t-SNE (Version A) — colored by key features",
             fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("fig3_tsne_feature_colors.png", bbox_inches='tight')
plt.close()
print("  已保存 fig3_tsne_feature_colors.png")

# ============================================================
# 6. 图4 — perplexity 敏感性分析（1×3子图）
# ============================================================
print("\n[Fig 4] Perplexity sensitivity analysis (5 / 30 / 50)...")

perplexities = [5, 30, 50]
embeddings_perp = {}

for perp in perplexities:
    print(f"  Running perplexity={perp}...", end=' ')
    t0 = time.time()
    tsne_p = TSNE(
        n_components=2,
        perplexity=perp,
        max_iter=1000,
        init='pca',
        random_state=RANDOM_STATE,
        learning_rate='auto',
    )
    emb_p = tsne_p.fit_transform(X_vA)
    embeddings_perp[perp] = emb_p
    print(f"done ({time.time()-t0:.1f}s, KL={tsne_p.kl_divergence_:.3f})")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, perp in zip(axes, perplexities):
    plot_tsne(ax, embeddings_perp[perp], y,
              f"perplexity = {perp}",
              legend=(perp == perplexities[0]))

fig.suptitle("t-SNE sensitivity to perplexity parameter",
             fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("fig4_tsne_perplexity.png", bbox_inches='tight')
plt.close()
print("  已保存 fig4_tsne_perplexity.png")

# ============================================================
# 7. 图5 — 3D t-SNE
# ============================================================
print("\n[Fig 5] 3D t-SNE...")
t0 = time.time()

tsne_3d = TSNE(
    n_components=3,
    perplexity=30,
    max_iter=1000,
    init='pca',
    random_state=RANDOM_STATE,
    learning_rate='auto',
)
emb_3d = tsne_3d.fit_transform(X_vA)
print(f"  完成，耗时 {time.time()-t0:.1f}s，KL散度={tsne_3d.kl_divergence_:.4f}")

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

for cls in [2, 1, 0]:
    mask = (y == cls)
    ax.scatter(
        emb_3d[mask, 0],
        emb_3d[mask, 1],
        emb_3d[mask, 2],
        c=COLORS[cls], marker=MARKERS[cls],
        s=SIZE, alpha=ALPHA,
        linewidths=0, label=LABELS[cls],
    )

ax.set_title("3D t-SNE projection (23 continuous features, perplexity=30)")
ax.set_xlabel("dim 1")
ax.set_ylabel("dim 2")
ax.set_zlabel("dim 3")
ax.tick_params(labelleft=False, labelbottom=False)
handles = [mpatches.Patch(color=COLORS[c], label=LABELS[c]) for c in [2, 1, 0]]
ax.legend(handles=handles, loc='upper left', fontsize=9)
plt.tight_layout()
plt.savefig("fig5_tsne_3d.png", bbox_inches='tight')
plt.close()
print("  已保存 fig5_tsne_3d.png")

# ============================================================
# 8. 总结输出
# ============================================================
print("\n" + "=" * 60)
print("全部图表已保存：")
print("  fig1_tsne_target_vA.png      — 版本A，按Target着色（主图）")
print("  fig2_tsne_target_vB.png      — 版本A vs 版本B 对比")
print("  fig3_tsne_feature_colors.png — 按4个关键特征着色（2×2）")
print("  fig4_tsne_perplexity.png     — perplexity 敏感性分析")
print("  fig5_tsne_3d.png             — 3D t-SNE")
print("=" * 60)
print("\n[KL散度参考] 版本A(perp=30):", round(tsne_vA.kl_divergence_, 4))
print("[KL散度参考] 版本B(perp=30):", round(tsne_vB.kl_divergence_, 4))
print("\n建议在报告中说明：")
print("  1. t-SNE 轴数值无物理意义，不可解读坐标绝对值")
print("  2. 簇间距离不等于真实高维距离")
print("  3. 局部邻近结构（同簇内部）是可信的")

