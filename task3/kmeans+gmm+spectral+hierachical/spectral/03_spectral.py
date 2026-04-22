"""
Task 3 — Clustering Analysis (Spectral Clustering)
====================================================
输入: X_pca_task3.npy, y_task3.npy, emb_tsne_task3.npy, km_labels_task3.npy
输出:
  fig13_spectral_tsne_comparison.png   — t-SNE: 真实标签 vs Spectral
  fig14_spectral_cluster_profiles.png  — Spectral 簇特征热力图
  fig15_spectral_affinity_search.png   — 不同 affinity 参数的指标对比
  fig16_three_algo_comparison.png      — K-Means vs GMM vs Spectral 三合一
  task3_spectral_results.csv           — 完整搜索结果
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    adjusted_rand_score, normalized_mutual_info_score
)
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
print("Task 3 — Clustering Analysis (Spectral Clustering)")
print("=" * 65)

# ============================================================
# 1. 加载中间数据
# ============================================================
print("\n[Step 1] 加载中间数据...")

X_pca     = np.load('X_pca_task3.npy')
y         = np.load('y_task3.npy')
emb_tsne  = np.load('emb_tsne_task3.npy')
km_labels = np.load('km_labels_task3.npy')

print(f"  PCA 数据: {X_pca.shape}")

# 加载 GMM 标签做三方对比
try:
    # 重新跑一次 GMM k=3 来获取标签
    from sklearn.mixture import GaussianMixture
    gmm = GaussianMixture(n_components=3, covariance_type='full',
                          n_init=10, random_state=RANDOM_STATE)
    gmm.fit(X_pca)
    gmm_labels = gmm.predict(X_pca)
except Exception as e:
    print(f"  GMM 加载失败: {e}")
    gmm_labels = None

# 重新加载原始标准化数据用于热力图
df = pd.read_csv('full_preprocessed_data.csv')
X_raw = df.drop(columns=['Target', 'Target_binary']).copy()
scale_cols = [c for c in X_raw.columns if X_raw[c].nunique() > 2]
scaler = StandardScaler()
X_raw[scale_cols] = scaler.fit_transform(X_raw[scale_cols])

# ============================================================
# 2. Spectral Clustering 参数搜索
# ============================================================
print("\n[Step 2] Spectral Clustering 参数搜索...")
print("  注意：affinity='nearest_neighbors' 比 'rbf' 在高维数据上更稳健")

# 测试两种 affinity 和不同的邻居数
configs = []

# Configuration 1: nearest_neighbors with different n_neighbors
# 高维数据上 nearest_neighbors 比 rbf 更稳健
for n_neighbors in [10, 20, 50]:
    configs.append({
        'affinity': 'nearest_neighbors',
        'n_neighbors': n_neighbors,
        'gamma': None,
    })

# Configuration 2: RBF kernel
configs.append({
    'affinity': 'rbf',
    'n_neighbors': None,
    'gamma': 0.1,
})

results = []

for cfg in configs:
    label = f"{cfg['affinity']}"
    if cfg['n_neighbors'] is not None:
        label += f"_k{cfg['n_neighbors']}"
    if cfg['gamma'] is not None:
        label += f"_g{cfg['gamma']}"

    try:
        t0 = time.time()
        if cfg['affinity'] == 'nearest_neighbors':
            sc = SpectralClustering(
                n_clusters=3,
                affinity='nearest_neighbors',
                n_neighbors=cfg['n_neighbors'],
                assign_labels='kmeans',
                eigen_solver='arpack',
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        else:
            sc = SpectralClustering(
                n_clusters=3,
                affinity='rbf',
                gamma=cfg['gamma'],
                assign_labels='kmeans',
                eigen_solver='arpack',
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )

        labels = sc.fit_predict(X_pca)
        elapsed = time.time() - t0

        sil = silhouette_score(X_pca, labels)
        ch  = calinski_harabasz_score(X_pca, labels)
        db  = davies_bouldin_score(X_pca, labels)
        ari = adjusted_rand_score(y, labels)
        nmi = normalized_mutual_info_score(y, labels)

        unique, counts = np.unique(labels, return_counts=True)
        cluster_sizes = dict(zip(unique.tolist(), counts.tolist()))

        results.append({
            'config': label,
            'affinity': cfg['affinity'],
            'n_neighbors': cfg['n_neighbors'],
            'gamma': cfg['gamma'],
            'silhouette': sil,
            'calinski_harabasz': ch,
            'davies_bouldin': db,
            'ARI': ari,
            'NMI': nmi,
            'time_sec': elapsed,
            'cluster_sizes': str(cluster_sizes),
            'labels': labels,
        })

        print(f"  {label:35s}: Sil={sil:.4f}, ARI={ari:.4f}, NMI={nmi:.4f}, "
              f"sizes={cluster_sizes}, time={elapsed:.1f}s")
    except Exception as e:
        print(f"  {label}: FAILED - {str(e)[:80]}")

results_df = pd.DataFrame(results)
results_df_save = results_df.drop(columns=['labels'])
results_df_save.to_csv('task3_spectral_results.csv', index=False)
print("  已保存 task3_spectral_results.csv")

# ============================================================
# 3. 选择最佳配置（按 ARI）
# ============================================================
best_idx = results_df['ARI'].idxmax()
best_config = results_df.loc[best_idx]
spectral_labels = best_config['labels']

print(f"\n[Step 3] 最佳配置（按 ARI）: {best_config['config']}")
print(f"  Silhouette: {best_config['silhouette']:.4f}")
print(f"  ARI: {best_config['ARI']:.4f}")
print(f"  NMI: {best_config['NMI']:.4f}")
print(f"  簇分布: {best_config['cluster_sizes']}")

# ============================================================
# 4. Fig 13 — t-SNE 对比图
# ============================================================
print("\n[Step 4] 绘制 t-SNE 对比图...")

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
for cls in sorted(np.unique(spectral_labels)):
    mask = spectral_labels == cls
    ax.scatter(emb_tsne[mask, 0], emb_tsne[mask, 1],
               c=COLORS_CLUSTER[cls], s=12, alpha=0.8,
               linewidths=0, rasterized=True)
handles = [mpatches.Patch(color=COLORS_CLUSTER[c], label=f'Cluster {c}')
           for c in sorted(np.unique(spectral_labels))]
ax.legend(handles=handles, loc='best', framealpha=0.7, fontsize=10)
ax.set_title(f'Spectral Clustering ({best_config["config"]})', fontsize=13)
ax.set_xlabel('t-SNE dim 1'); ax.set_ylabel('t-SNE dim 2')
ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

fig.suptitle('t-SNE Visualization: Ground Truth vs Spectral Clustering',
             fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('fig13_spectral_tsne_comparison.png', bbox_inches='tight')
plt.close()
print("  已保存 fig13_spectral_tsne_comparison.png")

# ============================================================
# 5. Fig 14 — 簇特征热力图
# ============================================================
print("\n[Step 5] 绘制簇特征热力图...")

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
X_analysis['Cluster'] = spectral_labels
cluster_means = X_analysis.groupby('Cluster')[scale_cols].mean()
cluster_means_display = cluster_means.rename(columns=short_names)

n_clusters = len(np.unique(spectral_labels))
fig, ax = plt.subplots(figsize=(16, 5))
sns.heatmap(cluster_means_display, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, linewidths=0.5, ax=ax,
            yticklabels=[f'Cluster {i}' for i in range(n_clusters)])
ax.set_title(f'Spectral Clustering Cluster Feature Profiles (k=3, {best_config["config"]})',
             fontsize=13)
ax.set_xlabel(''); ax.set_ylabel('')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('fig14_spectral_cluster_profiles.png', bbox_inches='tight')
plt.close()
print("  已保存 fig14_spectral_cluster_profiles.png")

# ============================================================
# 6. Fig 15 — 不同参数的指标对比
# ============================================================
print("\n[Step 6] 绘制参数搜索对比图...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 三个指标的柱状图
metrics_to_plot = [('silhouette', 'Silhouette ↑'),
                   ('ARI', 'ARI ↑'),
                   ('NMI', 'NMI ↑')]

for ax, (metric, title) in zip(axes, metrics_to_plot):
    configs_short = [c.replace('nearest_neighbors', 'NN').replace('_k', ' k=').replace('_g', ' γ=')
                     for c in results_df['config']]
    colors = ['#4965b0' if 'NN' in c else '#f36f43' for c in configs_short]
    bars = ax.bar(range(len(configs_short)), results_df[metric], color=colors, alpha=0.85)
    ax.set_xticks(range(len(configs_short)))
    ax.set_xticklabels(configs_short, rotation=45, ha='right', fontsize=9)
    ax.set_title(title, fontsize=12)
    ax.set_ylabel(metric)

    # 标注数值
    for bar, val in zip(bars, results_df[metric]):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)

# 添加图例
patch1 = mpatches.Patch(color='#4965b0', label='nearest_neighbors')
patch2 = mpatches.Patch(color='#f36f43', label='rbf')
fig.legend(handles=[patch1, patch2], loc='upper right', fontsize=10,
           bbox_to_anchor=(0.99, 0.99))

fig.suptitle('Spectral Clustering: Parameter Search Comparison', fontsize=14, y=1.03)
plt.tight_layout()
plt.savefig('fig15_spectral_affinity_search.png', bbox_inches='tight')
plt.close()
print("  已保存 fig15_spectral_affinity_search.png")

# ============================================================
# 7. Fig 16 — K-Means / GMM / Spectral 三方对比
# ============================================================
print("\n[Step 7] 绘制三算法 t-SNE 对比图...")

fig, axes = plt.subplots(1, 4, figsize=(24, 6.5))

# 真实标签
ax = axes[0]
for cls in [2, 1, 0]:
    mask = y == cls
    ax.scatter(emb_tsne[mask, 0], emb_tsne[mask, 1],
               c=COLORS_TARGET[cls], s=10, alpha=0.8,
               linewidths=0, rasterized=True)
handles = [mpatches.Patch(color=COLORS_TARGET[c], label=LABELS_TARGET[c]) for c in [2, 1, 0]]
ax.legend(handles=handles, loc='best', framealpha=0.7, fontsize=9)
ax.set_title('Ground Truth', fontsize=12, fontweight='bold')
ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

# K-Means
ax = axes[1]
for cls in sorted(np.unique(km_labels)):
    mask = km_labels == cls
    ax.scatter(emb_tsne[mask, 0], emb_tsne[mask, 1],
               c=COLORS_CLUSTER[cls], s=10, alpha=0.8,
               linewidths=0, rasterized=True)
ax.set_title('K-Means (k=3)', fontsize=12, fontweight='bold')
ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

# GMM
ax = axes[2]
if gmm_labels is not None:
    for cls in sorted(np.unique(gmm_labels)):
        mask = gmm_labels == cls
        ax.scatter(emb_tsne[mask, 0], emb_tsne[mask, 1],
                   c=COLORS_CLUSTER[cls], s=10, alpha=0.8,
                   linewidths=0, rasterized=True)
ax.set_title('GMM (k=3, full)', fontsize=12, fontweight='bold')
ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

# Spectral
ax = axes[3]
for cls in sorted(np.unique(spectral_labels)):
    mask = spectral_labels == cls
    ax.scatter(emb_tsne[mask, 0], emb_tsne[mask, 1],
               c=COLORS_CLUSTER[cls], s=10, alpha=0.45,
               linewidths=0, rasterized=True)
ax.set_title(f'Spectral ({best_config["config"]})', fontsize=12, fontweight='bold')
ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

fig.suptitle('Comparison of Three Clustering Algorithms', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('fig16_three_algo_comparison.png', bbox_inches='tight')
plt.close()
print("  已保存 fig16_three_algo_comparison.png")

# ============================================================
# 8. 交叉分析与最终对比
# ============================================================
print("\n[Step 8] 交叉分析...")

cross_tab = pd.crosstab(
    pd.Series(spectral_labels, name='Spectral Cluster'),
    pd.Series(y, name='Target').map(LABELS_TARGET),
    margins=True
)
print("\n  交叉频数表:")
print(cross_tab.to_string().replace('\n', '\n  '))

cross_pct = pd.crosstab(
    pd.Series(spectral_labels, name='Spectral Cluster'),
    pd.Series(y, name='Target').map(LABELS_TARGET),
    normalize='index'
)
print("\n  各簇中的类别比例:")
print((cross_pct * 100).round(1).to_string().replace('\n', '\n  '))

# ============================================================
# 9. 三算法最终对比表
# ============================================================
print("\n[Step 9] 三算法最终对比:")
print("=" * 70)

km_metrics = {
    'Silhouette': silhouette_score(X_pca, km_labels),
    'Calinski-Harabasz': calinski_harabasz_score(X_pca, km_labels),
    'Davies-Bouldin': davies_bouldin_score(X_pca, km_labels),
    'ARI': adjusted_rand_score(y, km_labels),
    'NMI': normalized_mutual_info_score(y, km_labels),
}

if gmm_labels is not None:
    gmm_metrics = {
        'Silhouette': silhouette_score(X_pca, gmm_labels),
        'Calinski-Harabasz': calinski_harabasz_score(X_pca, gmm_labels),
        'Davies-Bouldin': davies_bouldin_score(X_pca, gmm_labels),
        'ARI': adjusted_rand_score(y, gmm_labels),
        'NMI': normalized_mutual_info_score(y, gmm_labels),
    }

spec_metrics = {
    'Silhouette': best_config['silhouette'],
    'Calinski-Harabasz': best_config['calinski_harabasz'],
    'Davies-Bouldin': best_config['davies_bouldin'],
    'ARI': best_config['ARI'],
    'NMI': best_config['NMI'],
}

print(f"  {'Metric':<22} {'K-Means':>12} {'GMM':>12} {'Spectral':>12}")
print("  " + "-" * 60)
for m in km_metrics:
    line = f"  {m:<22} {km_metrics[m]:>12.4f}"
    if gmm_labels is not None:
        line += f" {gmm_metrics[m]:>12.4f}"
    line += f" {spec_metrics[m]:>12.4f}"
    print(line)
print("=" * 70)

print("\n输出文件:")
print("  fig13_spectral_tsne_comparison.png")
print("  fig14_spectral_cluster_profiles.png")
print("  fig15_spectral_affinity_search.png")
print("  fig16_three_algo_comparison.png")
print("  task3_spectral_results.csv")