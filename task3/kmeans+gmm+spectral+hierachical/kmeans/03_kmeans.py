"""
Task 3 — Clustering Analysis (K-Means)
========================================
输入: full_preprocessed_data.csv
输出:
  fig1_elbow_method.png              — Elbow Method (Inertia)
  fig2_silhouette_scores.png         — Silhouette Score vs k
  fig3_kmeans_tsne_comparison.png    — t-SNE: 真实标签 vs K-Means 聚类标签
  fig4_kmeans_cluster_profiles.png   — 各簇特征均值热力图
  fig5_kmeans_silhouette_detail.png  — Silhouette 轮廓图（最优k）
  fig6_pca_variance.png              — PCA 累计方差解释图
  task3_kmeans_results.csv           — 各 k 值的评估指标汇总
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import (
    silhouette_score, silhouette_samples,
    calinski_harabasz_score, davies_bouldin_score,
    adjusted_rand_score, normalized_mutual_info_score,
    confusion_matrix
)
import warnings, time, os

# os.chdir(os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '/home/claude')
# warnings.filterwarnings('ignore')

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
COLORS_TARGET  = {0: '#a30543', 1: '#fbda83', 2: '#4965b0'}
LABELS_TARGET  = {0: 'Dropout', 1: 'Enrolled', 2: 'Graduate'}

print("=" * 65)
print("Task 3 — Clustering Analysis (K-Means)")
print("=" * 65)

# ============================================================
# 1. 数据加载与全量重新标准化
# ============================================================
print("\n[Step 1] 数据加载与全量标准化...")

df = pd.read_csv("full_preprocessed_data.csv")
X = df.drop(columns=['Target', 'Target_binary']).copy()
y = df['Target'].values

print(f"  数据形状: {X.shape}")
print(f"  Target 分布: { {LABELS_TARGET[k]: v for k, v in zip(*np.unique(y, return_counts=True))} }")

# 区分连续列和二值列
scale_cols  = [c for c in X.columns if X[c].nunique() > 2]   # 23列
binary_cols = [c for c in X.columns if X[c].nunique() <= 2]   # 66列

print(f"  连续特征: {len(scale_cols)} 列 → 全量 StandardScaler")
print(f"  二值特征: {len(binary_cols)} 列 → 保持 0/1")

# 全量重新标准化（聚类为无监督任务，不涉及 train/test 划分）
scaler = StandardScaler()
X[scale_cols] = scaler.fit_transform(X[scale_cols])

# 验证
print(f"  标准化后均值范围: [{X[scale_cols].mean().min():.6f}, {X[scale_cols].mean().max():.6f}]")
print(f"  标准化后标准差范围: [{X[scale_cols].std().min():.6f}, {X[scale_cols].std().max():.6f}]")

# ============================================================
# 2. PCA 降维
# ============================================================
print("\n[Step 2] PCA 降维...")

# 先做完整 PCA 分析
pca_full = PCA(random_state=RANDOM_STATE).fit(X.values)
cumvar = np.cumsum(pca_full.explained_variance_ratio_)

# 找到 90% 方差对应的维度
n_components_90 = np.argmax(cumvar >= 0.90) + 1
n_components_80 = np.argmax(cumvar >= 0.80) + 1
n_components_95 = np.argmax(cumvar >= 0.95) + 1

print(f"  保留 80% 方差: {n_components_80} 个主成分")
print(f"  保留 90% 方差: {n_components_90} 个主成分")
print(f"  保留 95% 方差: {n_components_95} 个主成分")

# 选择 90% 方差
N_PCA = n_components_90
pca = PCA(n_components=N_PCA, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X.values)
actual_var = pca.explained_variance_ratio_.sum()

print(f"  → 选择 {N_PCA} 个主成分，实际保留方差: {actual_var:.1%}")

# 绘制 PCA 累计方差图
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(range(1, len(cumvar)+1), cumvar, 'b-o', markersize=3, linewidth=1.5)
ax.axhline(y=0.90, color='r', linestyle='--', alpha=0.7, label='90% variance')
ax.axvline(x=N_PCA, color='r', linestyle='--', alpha=0.7)
ax.annotate(f'  {N_PCA} components\n  ({actual_var:.1%} variance)',
            xy=(N_PCA, 0.90), fontsize=10, color='red')
ax.set_xlabel('Number of Principal Components')
ax.set_ylabel('Cumulative Explained Variance')
ax.set_title('PCA Cumulative Explained Variance')
ax.legend(loc='lower right')
ax.set_xlim(0, 60)
ax.set_ylim(0, 1.02)
plt.tight_layout()
plt.savefig('fig6_pca_variance.png', bbox_inches='tight')
plt.close()
print("  已保存 fig6_pca_variance.png")

# ============================================================
# 3. t-SNE 嵌入（用于后续可视化）
# ============================================================
print("\n[Step 3] t-SNE 嵌入（用于聚类结果可视化）...")

t0 = time.time()
tsne = TSNE(
    n_components=2,
    perplexity=30,
    max_iter=1000,
    init='pca',
    random_state=RANDOM_STATE,
    learning_rate='auto',
)
# 用 PCA 降维后的数据做 t-SNE（与聚类输入一致）
emb_tsne = tsne.fit_transform(X_pca)
print(f"  完成，耗时 {time.time()-t0:.1f}s，KL散度={tsne.kl_divergence_:.4f}")

# ============================================================
# 4. K-Means: Elbow Method + Silhouette Score 确定最佳 k
# ============================================================
print("\n[Step 4] K-Means 参数搜索 (k=2~8)...")

K_RANGE = range(2, 9)
results = {
    'k': [], 'inertia': [], 'silhouette': [],
    'calinski_harabasz': [], 'davies_bouldin': [],
    'ARI': [], 'NMI': []
}

for k in K_RANGE:
    km = KMeans(n_clusters=k, init='k-means++', n_init=10,
                max_iter=300, random_state=RANDOM_STATE)
    labels = km.fit_predict(X_pca)

    sil = silhouette_score(X_pca, labels)
    ch  = calinski_harabasz_score(X_pca, labels)
    db  = davies_bouldin_score(X_pca, labels)
    ari = adjusted_rand_score(y, labels)
    nmi = normalized_mutual_info_score(y, labels)

    results['k'].append(k)
    results['inertia'].append(km.inertia_)
    results['silhouette'].append(sil)
    results['calinski_harabasz'].append(ch)
    results['davies_bouldin'].append(db)
    results['ARI'].append(ari)
    results['NMI'].append(nmi)

    print(f"  k={k}: Silhouette={sil:.4f}, CH={ch:.1f}, DB={db:.4f}, "
          f"ARI={ari:.4f}, NMI={nmi:.4f}, Inertia={km.inertia_:.1f}")

results_df = pd.DataFrame(results)
results_df.to_csv('task3_kmeans_results.csv', index=False)
print("  已保存 task3_kmeans_results.csv")

# ============================================================
# 5. 图1 — Elbow Method
# ============================================================
print("\n[Step 5] 绘制 Elbow Method...")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(results['k'], results['inertia'], 'b-o', linewidth=2, markersize=8)

# 标注各点的 inertia 值
for k_val, inertia_val in zip(results['k'], results['inertia']):
    ax.annotate(f'{inertia_val:.0f}', (k_val, inertia_val),
                textcoords="offset points", xytext=(0, 12),
                ha='center', fontsize=9)

ax.set_xlabel('Number of Clusters (k)')
ax.set_ylabel('Inertia (WCSS)')
ax.set_title('Elbow Method for Optimal k')
ax.set_xticks(list(K_RANGE))
plt.tight_layout()
plt.savefig('fig1_elbow_method.png', bbox_inches='tight')
plt.close()
print("  已保存 fig1_elbow_method.png")

# ============================================================
# 6. 图2 — Silhouette Score vs k
# ============================================================
print("\n[Step 6] 绘制 Silhouette Score...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 6a: Silhouette Score
ax = axes[0]
ax.plot(results['k'], results['silhouette'], 'g-o', linewidth=2, markersize=8)
best_sil_k = results['k'][np.argmax(results['silhouette'])]
best_sil_v = max(results['silhouette'])
ax.axvline(x=best_sil_k, color='red', linestyle='--', alpha=0.7)
ax.annotate(f'Best: k={best_sil_k}\n({best_sil_v:.4f})',
            xy=(best_sil_k, best_sil_v), fontsize=10, color='red',
            xytext=(15, -15), textcoords='offset points')
ax.set_xlabel('Number of Clusters (k)')
ax.set_ylabel('Silhouette Score')
ax.set_title('Silhouette Score (higher = better)')
ax.set_xticks(list(K_RANGE))

# 6b: Calinski-Harabasz Index
ax = axes[1]
ax.plot(results['k'], results['calinski_harabasz'], 'm-o', linewidth=2, markersize=8)
best_ch_k = results['k'][np.argmax(results['calinski_harabasz'])]
ax.axvline(x=best_ch_k, color='red', linestyle='--', alpha=0.7)
ax.set_xlabel('Number of Clusters (k)')
ax.set_ylabel('Calinski-Harabasz Index')
ax.set_title('Calinski-Harabasz Index (higher = better)')
ax.set_xticks(list(K_RANGE))

# 6c: Davies-Bouldin Index
ax = axes[2]
ax.plot(results['k'], results['davies_bouldin'], 'r-o', linewidth=2, markersize=8)
best_db_k = results['k'][np.argmin(results['davies_bouldin'])]
ax.axvline(x=best_db_k, color='red', linestyle='--', alpha=0.7)
ax.set_xlabel('Number of Clusters (k)')
ax.set_ylabel('Davies-Bouldin Index')
ax.set_title('Davies-Bouldin Index (lower = better)')
ax.set_xticks(list(K_RANGE))

fig.suptitle('Internal Clustering Metrics vs Number of Clusters', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('fig2_silhouette_scores.png', bbox_inches='tight')
plt.close()
print("  已保存 fig2_silhouette_scores.png")

# ============================================================
# 7. 确定最优 k 并训练最终 K-Means 模型
# ============================================================
# 综合考虑：真实类别数为3，Silhouette 和各指标
# 选择 k=3 与真实标签对齐，便于外部指标评估
BEST_K = 3
print(f"\n[Step 7] 训练最终 K-Means (k={BEST_K})...")

km_final = KMeans(n_clusters=BEST_K, init='k-means++', n_init=20,
                  max_iter=300, random_state=RANDOM_STATE)
km_labels = km_final.fit_predict(X_pca)

print(f"  簇分布: {dict(zip(*np.unique(km_labels, return_counts=True)))}")
print(f"  Inertia: {km_final.inertia_:.2f}")

# ============================================================
# 8. 图3 — t-SNE 可视化：真实标签 vs K-Means 聚类标签
# ============================================================
print("\n[Step 8] 绘制 t-SNE 对比图...")

COLORS_CLUSTER = {0: '#f8e620', 1: '#38b775', 2: '#404185'}

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# 左图：真实标签
ax = axes[0]
for cls in [2, 1, 0]:
    mask = y == cls
    ax.scatter(emb_tsne[mask, 0], emb_tsne[mask, 1],
               c=COLORS_TARGET[cls], s=12, alpha=0.7,
               linewidths=0, rasterized=True)
handles = [mpatches.Patch(color=COLORS_TARGET[c], label=LABELS_TARGET[c]) for c in [2, 1, 0]]
ax.legend(handles=handles, loc='best', framealpha=0.7, fontsize=10)
ax.set_title('Ground Truth Labels', fontsize=13)
ax.set_xlabel('t-SNE dim 1')
ax.set_ylabel('t-SNE dim 2')
ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

# 右图：K-Means 聚类标签
ax = axes[1]
for cls in sorted(np.unique(km_labels)):
    mask = km_labels == cls
    ax.scatter(emb_tsne[mask, 0], emb_tsne[mask, 1],
               c=COLORS_CLUSTER[cls], s=12, alpha=0.7,
               linewidths=0, rasterized=True)
handles = [mpatches.Patch(color=COLORS_CLUSTER[c], label=f'Cluster {c}')
           for c in sorted(np.unique(km_labels))]
ax.legend(handles=handles, loc='best', framealpha=0.7, fontsize=10)
ax.set_title(f'K-Means Clusters (k={BEST_K})', fontsize=13)
ax.set_xlabel('t-SNE dim 1')
ax.set_ylabel('t-SNE dim 2')
ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

fig.suptitle('t-SNE Visualization: Ground Truth vs K-Means Clustering',
             fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('fig3_kmeans_tsne_comparison.png', bbox_inches='tight')
plt.close()
print("  已保存 fig3_kmeans_tsne_comparison.png")

# ============================================================
# 9. 图4 — 簇特征分析热力图
# ============================================================
print("\n[Step 9] 绘制簇特征分析热力图...")

# 用原始（标准化后）的连续特征做分析，更有解释性
X_analysis = X[scale_cols].copy()
X_analysis['Cluster'] = km_labels
X_analysis['Target'] = y

# 每个簇中各连续特征的均值
cluster_means = X_analysis.groupby('Cluster')[scale_cols].mean()

# 简化列名用于显示
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
cluster_means_display = cluster_means.rename(columns=short_names)

fig, ax = plt.subplots(figsize=(16, 5))
sns.heatmap(cluster_means_display, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, linewidths=0.5, ax=ax,
            xticklabels=True, yticklabels=[f'Cluster {i}' for i in range(BEST_K)])
ax.set_title(f'K-Means Cluster Feature Profiles (Standardized Mean, k={BEST_K})', fontsize=13)
ax.set_xlabel('')
ax.set_ylabel('')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('fig4_kmeans_cluster_profiles.png', bbox_inches='tight')
plt.close()
print("  已保存 fig4_kmeans_cluster_profiles.png")

# ============================================================
# 10. 图5 — Silhouette 轮廓详细图
# ============================================================
print("\n[Step 10] 绘制 Silhouette 轮廓图...")

sil_samples = silhouette_samples(X_pca, km_labels)
sil_avg = silhouette_score(X_pca, km_labels)

fig, ax = plt.subplots(figsize=(9, 7))
y_lower = 10

for i in range(BEST_K):
    cluster_sil = sil_samples[km_labels == i]
    cluster_sil.sort()
    cluster_size = cluster_sil.shape[0]
    y_upper = y_lower + cluster_size

    color = plt.cm.Set2(float(i) / BEST_K)
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_sil,
                      facecolor=color, edgecolor=color, alpha=0.7)
    ax.text(-0.05, y_lower + 0.5 * cluster_size,
            f'Cluster {i}\n(n={cluster_size})',
            fontsize=10, va='center')
    y_lower = y_upper + 10

ax.axvline(x=sil_avg, color='red', linestyle='--', linewidth=2,
           label=f'Average Silhouette = {sil_avg:.4f}')
ax.set_xlabel('Silhouette Coefficient')
ax.set_ylabel('Samples (sorted by cluster)')
ax.set_title(f'Silhouette Plot for K-Means (k={BEST_K})')
ax.legend(loc='upper right', fontsize=11)
ax.set_yticks([])
plt.tight_layout()
plt.savefig('fig5_kmeans_silhouette_detail.png', bbox_inches='tight')
plt.close()
print("  已保存 fig5_kmeans_silhouette_detail.png")

# ============================================================
# 11. 簇与真实标签的交叉分析
# ============================================================
print("\n[Step 11] 簇与真实标签的交叉分析...")

cross_tab = pd.crosstab(
    pd.Series(km_labels, name='K-Means Cluster'),
    pd.Series(y, name='Target').map(LABELS_TARGET),
    margins=True
)
print("\n  交叉频数表:")
print(cross_tab.to_string().replace('\n', '\n  '))

# 各簇中各类别的比例
cross_pct = pd.crosstab(
    pd.Series(km_labels, name='K-Means Cluster'),
    pd.Series(y, name='Target').map(LABELS_TARGET),
    normalize='index'
)
print("\n  各簇中的类别比例:")
print((cross_pct * 100).round(1).to_string().replace('\n', '\n  '))

# ============================================================
# 12. 最终评估指标汇总
# ============================================================
print("\n[Step 12] K-Means (k=3) 最终评估指标汇总:")
print("=" * 50)

metrics = {
    'Silhouette Score': silhouette_score(X_pca, km_labels),
    'Calinski-Harabasz Index': calinski_harabasz_score(X_pca, km_labels),
    'Davies-Bouldin Index': davies_bouldin_score(X_pca, km_labels),
    'Adjusted Rand Index (ARI)': adjusted_rand_score(y, km_labels),
    'Normalized Mutual Info (NMI)': normalized_mutual_info_score(y, km_labels),
}

for name, val in metrics.items():
    direction = '↑' if 'Davies' not in name else '↓'
    print(f"  {name}: {val:.4f} ({direction} better)")

# ============================================================
# 13. 保存关键中间数据供 GMM 复用
# ============================================================
np.save('X_pca_task3.npy', X_pca)
np.save('y_task3.npy', y)
np.save('emb_tsne_task3.npy', emb_tsne)
np.save('km_labels_task3.npy', km_labels)

print("\n已保存中间数据: X_pca_task3.npy, y_task3.npy, emb_tsne_task3.npy, km_labels_task3.npy")

print("\n" + "=" * 65)
print("K-Means 聚类分析完成！")
print("=" * 65)
print("\n输出文件:")
print("  fig1_elbow_method.png              — Elbow Method")
print("  fig2_silhouette_scores.png         — 三合一内部指标对比")
print("  fig3_kmeans_tsne_comparison.png    — t-SNE 真实 vs 聚类标签")
print("  fig4_kmeans_cluster_profiles.png   — 簇特征热力图")
print("  fig5_kmeans_silhouette_detail.png  — Silhouette 轮廓图")
print("  fig6_pca_variance.png              — PCA 累计方差")
print("  task3_kmeans_results.csv           — 各 k 评估指标")