"""
Task 3 — Clustering Analysis (Agglomerative Hierarchical Clustering)
======================================================================
输入: X_pca_task3.npy, y_task3.npy, emb_tsne_task3.npy, km_labels_task3.npy
输出:
  fig17_dendrogram.png                   — 层次聚类树状图
  fig18_hierarchical_tsne_comparison.png — t-SNE: 真实标签 vs Hierarchical
  fig19_hierarchical_cluster_profiles.png — Hierarchical 簇特征热力图
  fig20_linkage_comparison.png            — 不同 linkage 方法对比
  fig21_all_algorithms_comparison.png    — K-Means / GMM / Spectral / Hierarchical
  task3_hierarchical_results.csv          — 完整搜索结果
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    adjusted_rand_score, normalized_mutual_info_score
)
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import warnings, time, os

os.chdir(os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '/home/claude')
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'figure.dpi': 300,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

RANDOM_STATE = 42

# Set 3 配色
COLORS_TARGET  = {0: '#a30543', 1: '#80cba4', 2: '#4965b0'}
LABELS_TARGET  = {0: 'Dropout', 1: 'Enrolled', 2: 'Graduate'}
COLORS_CLUSTER = {0: '#f36f43', 1: '#fbda83', 2: '#80cba4'}

print("=" * 65)
print("Task 3 — Clustering Analysis (Agglomerative Hierarchical)")
print("=" * 65)

# ============================================================
# 1. 加载数据
# ============================================================
print("\n[Step 1] 加载中间数据...")

X_pca     = np.load('X_pca_task3.npy')
y         = np.load('y_task3.npy')
emb_tsne  = np.load('emb_tsne_task3.npy')
km_labels = np.load('km_labels_task3.npy')

print(f"  PCA 数据: {X_pca.shape}")

# 加载 GMM 和 Spectral 标签用于最终对比
try:
    from sklearn.mixture import GaussianMixture
    gmm = GaussianMixture(n_components=3, covariance_type='full',
                          n_init=10, random_state=RANDOM_STATE)
    gmm.fit(X_pca)
    gmm_labels = gmm.predict(X_pca)
    print(f"  GMM 标签已重新生成")
except Exception as e:
    gmm_labels = None

try:
    from sklearn.cluster import SpectralClustering
    sc = SpectralClustering(n_clusters=3, affinity='nearest_neighbors',
                            n_neighbors=10, assign_labels='kmeans',
                            eigen_solver='arpack', random_state=RANDOM_STATE, n_jobs=-1)
    spectral_labels = sc.fit_predict(X_pca)
    print(f"  Spectral 标签已重新生成")
except Exception as e:
    spectral_labels = None

# 加载原始标准化数据用于热力图
df = pd.read_csv('full_preprocessed_data.csv')
X_raw = df.drop(columns=['Target', 'Target_binary']).copy()
scale_cols = [c for c in X_raw.columns if X_raw[c].nunique() > 2]
scaler = StandardScaler()
X_raw[scale_cols] = scaler.fit_transform(X_raw[scale_cols])

# ============================================================
# 2. 计算 linkage 矩阵（用于 dendrogram）
# ============================================================
print("\n[Step 2] 计算 Ward linkage 矩阵...")
print("  注意：4424 条数据 × 24 维，构建距离矩阵需要约 75MB 内存")

t0 = time.time()
linkage_matrix = linkage(X_pca, method='ward')
print(f"  完成，耗时 {time.time()-t0:.1f}s")
print(f"  linkage 矩阵形状: {linkage_matrix.shape}")

# ============================================================
# 3. Fig 17 — Dendrogram（截断显示）
# ============================================================
print("\n[Step 3] 绘制 Dendrogram...")

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# 17a: 完整 dendrogram（截断到顶部 30 层）
ax = axes[0]
dendrogram(
    linkage_matrix,
    ax=ax,
    truncate_mode='lastp',
    p=30,
    leaf_rotation=90,
    leaf_font_size=8,
    show_contracted=True,
    color_threshold=0.7 * max(linkage_matrix[:, 2]),
)
ax.set_title('Dendrogram (Top 30 merges, Ward linkage)', fontsize=13)
ax.set_xlabel('Cluster Size (or sample index)')
ax.set_ylabel('Distance (Ward)')

# 17b: 切割线展示（显示 k=2/3/4 的切割位置）
ax = axes[1]
dendrogram(
    linkage_matrix,
    ax=ax,
    truncate_mode='lastp',
    p=15,
    leaf_rotation=90,
    leaf_font_size=9,
    show_contracted=True,
    no_labels=True,
)
# 添加切割线标记不同 k
distances = linkage_matrix[:, 2]
# 倒数第 k-1 个合并的距离就是切成 k 个簇的阈值
for k_val, color, label in [(2, '#a30543', 'k=2'), (3, '#4965b0', 'k=3'), (4, '#f36f43', 'k=4')]:
    threshold = distances[-(k_val - 1)] - 1
    ax.axhline(y=threshold, color=color, linestyle='--', linewidth=2, label=label, alpha=0.8)
ax.set_title('Dendrogram with k=2/3/4 cut lines', fontsize=13)
ax.set_xlabel('Sample index (truncated)')
ax.set_ylabel('Distance (Ward)')
ax.legend(loc='upper right', fontsize=10)

plt.tight_layout()
plt.savefig('fig17_dendrogram.png', bbox_inches='tight')
plt.close()
print("  已保存 fig17_dendrogram.png")

# ============================================================
# 4. 不同 linkage 方法对比 + k 值搜索
# ============================================================
print("\n[Step 4] 不同 linkage 方法对比...")

linkage_methods = ['ward', 'complete', 'average']
results = []

for method in linkage_methods:
    for k in range(2, 9):
        # ward 只能用 euclidean 距离
        ag = AgglomerativeClustering(
            n_clusters=k,
            linkage=method,
            metric='euclidean',
        )
        labels = ag.fit_predict(X_pca)

        sil = silhouette_score(X_pca, labels)
        ch  = calinski_harabasz_score(X_pca, labels)
        db  = davies_bouldin_score(X_pca, labels)
        ari = adjusted_rand_score(y, labels)
        nmi = normalized_mutual_info_score(y, labels)

        results.append({
            'linkage': method,
            'k': k,
            'silhouette': sil,
            'calinski_harabasz': ch,
            'davies_bouldin': db,
            'ARI': ari,
            'NMI': nmi,
        })

        print(f"  {method:10s} k={k}: Sil={sil:.4f}, CH={ch:.1f}, DB={db:.4f}, "
              f"ARI={ari:.4f}, NMI={nmi:.4f}")

results_df = pd.DataFrame(results)
results_df.to_csv('task3_hierarchical_results.csv', index=False)
print("  已保存 task3_hierarchical_results.csv")

# ============================================================
# 5. 训练最终模型 (Ward, k=3)
# ============================================================
print("\n[Step 5] 训练最终 Hierarchical 模型 (Ward, k=3)...")

ag_final = AgglomerativeClustering(n_clusters=3, linkage='ward', metric='euclidean')
hier_labels = ag_final.fit_predict(X_pca)

unique, counts = np.unique(hier_labels, return_counts=True)
print(f"  簇分布: {dict(zip(unique.tolist(), counts.tolist()))}")

# ============================================================
# 6. Fig 18 — t-SNE 对比图
# ============================================================
print("\n[Step 6] 绘制 t-SNE 对比图...")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

ax = axes[0]
for cls in [2, 1, 0]:
    mask = y == cls
    ax.scatter(emb_tsne[mask, 0], emb_tsne[mask, 1],
               c=COLORS_TARGET[cls], s=12, alpha=0.8,
               linewidths=0, rasterized=True)
handles = [mpatches.Patch(color=COLORS_TARGET[c], label=LABELS_TARGET[c]) for c in [2, 1, 0]]
ax.legend(handles=handles, loc='best', framealpha=0.7, fontsize=10)
ax.set_title('Ground Truth Labels', fontsize=13)
ax.set_xlabel('t-SNE dim 1'); ax.set_ylabel('t-SNE dim 2')
ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

ax = axes[1]
for cls in sorted(np.unique(hier_labels)):
    mask = hier_labels == cls
    ax.scatter(emb_tsne[mask, 0], emb_tsne[mask, 1],
               c=COLORS_CLUSTER[cls], s=12, alpha=0.8,
               linewidths=0, rasterized=True)
handles = [mpatches.Patch(color=COLORS_CLUSTER[c], label=f'Cluster {c}')
           for c in sorted(np.unique(hier_labels))]
ax.legend(handles=handles, loc='best', framealpha=0.7, fontsize=10)
ax.set_title('Hierarchical Clusters (Ward, k=3)', fontsize=13)
ax.set_xlabel('t-SNE dim 1'); ax.set_ylabel('t-SNE dim 2')
ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

fig.suptitle('t-SNE Visualization: Ground Truth vs Hierarchical Clustering',
             fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('fig18_hierarchical_tsne_comparison.png', bbox_inches='tight')
plt.close()
print("  已保存 fig18_hierarchical_tsne_comparison.png")

# ============================================================
# 7. Fig 19 — 簇特征热力图
# ============================================================
print("\n[Step 7] 绘制簇特征热力图...")

short_names = {
    'Application order': 'App Order',
    'Previous qualification (grade)': 'Prev Qual Grade',
    'Admission grade': 'Admission Grade',
    'Curricular units 1st sem (credited)': '1st Credited',
    'Curricular units 1st sem (enrolled)': '1st Enrolled',
    'Curricular units 1st sem (evaluations)': '1st Evaluations',
    'Curricular units 1st sem (approved)': '1st Approved',
    'Curricular units 1st sem (grade)': '1st Grade',
    'Curricular units 1st sem (without evaluations)': '1st No Eval',
    'Curricular units 2nd sem (credited)': '2nd Credited',
    'Curricular units 2nd sem (enrolled)': '2nd Enrolled',
    'Curricular units 2nd sem (evaluations)': '2nd Evaluations',
    'Curricular units 2nd sem (approved)': '2nd Approved',
    'Curricular units 2nd sem (grade)': '2nd Grade',
    'Curricular units 2nd sem (without evaluations)': '2nd No Eval',
    'Unemployment rate': 'Unemployment',
    'Inflation rate': 'Inflation',
    'GDP': 'GDP',
    'pass_rate_sem1': 'Pass Rate S1',
    'pass_rate_sem2': 'Pass Rate S2',
    'grade_trend': 'Grade Trend',
    'prev_qual_ordinal': 'Prev Qual Ordinal',
    'family_edu_capital': 'Family Edu Capital',
}

X_analysis = X_raw[scale_cols].copy()
X_analysis['Cluster'] = hier_labels
cluster_means = X_analysis.groupby('Cluster')[scale_cols].mean()
cluster_means_display = cluster_means.rename(columns=short_names)

fig, ax = plt.subplots(figsize=(16, 5))
sns.heatmap(cluster_means_display, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, linewidths=0.5, ax=ax,
            yticklabels=[f'Cluster {i}' for i in range(3)])
ax.set_title('Hierarchical Clustering Cluster Feature Profiles (Ward, k=3)', fontsize=13)
ax.set_xlabel(''); ax.set_ylabel('')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('fig19_hierarchical_cluster_profiles.png', bbox_inches='tight')
plt.close()
print("  已保存 fig19_hierarchical_cluster_profiles.png")

# ============================================================
# 8. Fig 20 — 不同 linkage 方法对比
# ============================================================
print("\n[Step 8] 绘制 linkage 方法对比图...")

fig, axes = plt.subplots(1, 5, figsize=(22, 5))
metrics_plot = [
    ('silhouette', 'Silhouette ↑'),
    ('calinski_harabasz', 'Calinski-Harabasz ↑'),
    ('davies_bouldin', 'Davies-Bouldin ↓'),
    ('ARI', 'ARI ↑'),
    ('NMI', 'NMI ↑'),
]

linkage_colors = {'ward': '#4965b0', 'complete': '#f36f43', 'average': '#80cba4'}

for ax, (metric, title) in zip(axes, metrics_plot):
    for method in linkage_methods:
        subset = results_df[results_df['linkage'] == method]
        ax.plot(subset['k'], subset[metric], '-o', color=linkage_colors[method],
                label=method, linewidth=2, markersize=6)
    ax.set_xlabel('k')
    ax.set_ylabel(metric)
    ax.set_title(title, fontsize=11)
    ax.set_xticks(range(2, 9))
    ax.legend(fontsize=9)

fig.suptitle('Hierarchical Clustering: Linkage Methods Comparison',
             fontsize=14, y=1.03)
plt.tight_layout()
plt.savefig('fig20_linkage_comparison.png', bbox_inches='tight')
plt.close()
print("  已保存 fig20_linkage_comparison.png")

# ============================================================
# 9. Fig 21 — 五合一对比图（含所有算法）
# ============================================================
print("\n[Step 9] 绘制全算法对比图...")

n_plots = 2 + sum(x is not None for x in [km_labels, gmm_labels, spectral_labels, hier_labels])
fig, axes = plt.subplots(1, 5, figsize=(26, 6.5))

# Ground Truth
ax = axes[0]
for cls in [2, 1, 0]:
    mask = y == cls
    ax.scatter(emb_tsne[mask, 0], emb_tsne[mask, 1],
               c=COLORS_TARGET[cls], s=10, alpha=0.8,
               linewidths=0, rasterized=True)
handles = [mpatches.Patch(color=COLORS_TARGET[c], label=LABELS_TARGET[c]) for c in [2, 1, 0]]
ax.legend(handles=handles, loc='best', framealpha=0.7, fontsize=8)
ax.set_title('Ground Truth', fontsize=12, fontweight='bold')
ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

# K-Means
ax = axes[1]
for cls in sorted(np.unique(km_labels)):
    mask = km_labels == cls
    ax.scatter(emb_tsne[mask, 0], emb_tsne[mask, 1],
               c=COLORS_CLUSTER[cls], s=10, alpha=0.8, linewidths=0, rasterized=True)
ax.set_title('K-Means', fontsize=12, fontweight='bold')
ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

# GMM
ax = axes[2]
if gmm_labels is not None:
    for cls in sorted(np.unique(gmm_labels)):
        mask = gmm_labels == cls
        ax.scatter(emb_tsne[mask, 0], emb_tsne[mask, 1],
                   c=COLORS_CLUSTER[cls], s=10, alpha=0.8, linewidths=0, rasterized=True)
ax.set_title('GMM (full)', fontsize=12, fontweight='bold')
ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

# Spectral
ax = axes[3]
if spectral_labels is not None:
    for cls in sorted(np.unique(spectral_labels)):
        mask = spectral_labels == cls
        ax.scatter(emb_tsne[mask, 0], emb_tsne[mask, 1],
                   c=COLORS_CLUSTER[cls], s=10, alpha=0.8, linewidths=0, rasterized=True)
ax.set_title('Spectral (NN k=10)', fontsize=12, fontweight='bold')
ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

# Hierarchical
ax = axes[4]
for cls in sorted(np.unique(hier_labels)):
    mask = hier_labels == cls
    ax.scatter(emb_tsne[mask, 0], emb_tsne[mask, 1],
               c=COLORS_CLUSTER[cls], s=10, alpha=0.8, linewidths=0, rasterized=True)
ax.set_title('Hierarchical (Ward)', fontsize=12, fontweight='bold')
ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

fig.suptitle('Comparison of All Four Clustering Algorithms', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('fig21_all_algorithms_comparison.png', bbox_inches='tight')
plt.close()
print("  已保存 fig21_all_algorithms_comparison.png")

# ============================================================
# 10. 交叉分析
# ============================================================
print("\n[Step 10] 簇与真实标签交叉分析...")

cross_tab = pd.crosstab(
    pd.Series(hier_labels, name='Hierarchical Cluster'),
    pd.Series(y, name='Target').map(LABELS_TARGET),
    margins=True
)
print("\n  交叉频数表:")
print(cross_tab.to_string().replace('\n', '\n  '))

cross_pct = pd.crosstab(
    pd.Series(hier_labels, name='Hierarchical Cluster'),
    pd.Series(y, name='Target').map(LABELS_TARGET),
    normalize='index'
)
print("\n  各簇中的类别比例:")
print((cross_pct * 100).round(1).to_string().replace('\n', '\n  '))

# ============================================================
# 11. 四算法最终对比
# ============================================================
print("\n[Step 11] 四算法最终对比:")
print("=" * 80)

algos = {'K-Means': km_labels, 'GMM': gmm_labels,
         'Spectral': spectral_labels, 'Hierarchical': hier_labels}

all_metrics = {}
for name, labels in algos.items():
    if labels is not None:
        all_metrics[name] = {
            'Silhouette': silhouette_score(X_pca, labels),
            'Calinski-Harabasz': calinski_harabasz_score(X_pca, labels),
            'Davies-Bouldin': davies_bouldin_score(X_pca, labels),
            'ARI': adjusted_rand_score(y, labels),
            'NMI': normalized_mutual_info_score(y, labels),
        }

print(f"  {'Metric':<22}", end='')
for name in all_metrics:
    print(f" {name:>14}", end='')
print()
print("  " + "-" * 78)

for m in ['Silhouette', 'Calinski-Harabasz', 'Davies-Bouldin', 'ARI', 'NMI']:
    print(f"  {m:<22}", end='')
    for name in all_metrics:
        print(f" {all_metrics[name][m]:>14.4f}", end='')
    print()
print("=" * 80)

# 标记每个指标的胜者
print("\n  各指标胜者:")
directions = {'Silhouette': 'higher', 'Calinski-Harabasz': 'higher',
              'Davies-Bouldin': 'lower', 'ARI': 'higher', 'NMI': 'higher'}
for m, d in directions.items():
    if d == 'higher':
        winner = max(all_metrics, key=lambda x: all_metrics[x][m])
    else:
        winner = min(all_metrics, key=lambda x: all_metrics[x][m])
    print(f"    {m:<22} → {winner} ({all_metrics[winner][m]:.4f})")

print("\n输出文件:")
print("  fig17_dendrogram.png")
print("  fig18_hierarchical_tsne_comparison.png")
print("  fig19_hierarchical_cluster_profiles.png")
print("  fig20_linkage_comparison.png")
print("  fig21_all_algorithms_comparison.png")
print("  task3_hierarchical_results.csv")