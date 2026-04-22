"""
Task 3 — Clustering Analysis (GMM)
====================================
输入: X_pca_task3.npy, y_task3.npy, emb_tsne_task3.npy, km_labels_task3.npy
     （由 task3_kmeans.py 生成的中间数据）
输出:
  fig7_bic_aic.png                   — BIC/AIC vs k（含不同协方差类型）
  fig8_gmm_tsne_comparison.png       — t-SNE: 真实标签 vs GMM 聚类标签
  fig9_gmm_cluster_profiles.png      — GMM 各簇特征均值热力图
  fig10_gmm_soft_assignment.png      — 软分配概率分布（最大概率直方图 + 边界学生分析）
  fig11_gmm_vs_kmeans_tsne.png       — K-Means vs GMM 聚类标签 t-SNE 对比
  fig12_final_comparison.png         — 最终综合评估指标对比表
  task3_gmm_results.csv              — 各 k 值 + 协方差类型的评估指标汇总
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    adjusted_rand_score, normalized_mutual_info_score
)
import warnings, os

os.chdir(os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '/home/claude')
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'figure.dpi': 150,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

RANDOM_STATE = 42

# 配色方案 (Set 3)
COLORS_TARGET  = {0: '#a30543', 1: '#80cba4', 2: '#4965b0'}
LABELS_TARGET  = {0: 'Dropout', 1: 'Enrolled', 2: 'Graduate'}
COLORS_CLUSTER = {0: '#f36f43', 1: '#fbda83', 2: '#80cba4'}

print("=" * 65)
print("Task 3 — Clustering Analysis (GMM)")
print("=" * 65)

# ============================================================
# 1. 加载 K-Means 阶段保存的中间数据
# ============================================================
print("\n[Step 1] 加载中间数据...")

X_pca     = np.load('X_pca_task3.npy')
y         = np.load('y_task3.npy')
emb_tsne  = np.load('emb_tsne_task3.npy')
km_labels = np.load('km_labels_task3.npy')

# 加载原始标准化后的连续特征（用于热力图）
df = pd.read_csv('full_preprocessed_data.csv')
from sklearn.preprocessing import StandardScaler
X_raw = df.drop(columns=['Target', 'Target_binary']).copy()
scale_cols = [c for c in X_raw.columns if X_raw[c].nunique() > 2]
scaler = StandardScaler()
X_raw[scale_cols] = scaler.fit_transform(X_raw[scale_cols])

print(f"  PCA 数据: {X_pca.shape}")
print(f"  样本数: {X_pca.shape[0]}, PCA 维度: {X_pca.shape[1]}")

# ============================================================
# 2. BIC / AIC 搜索：k × covariance_type
# ============================================================
print("\n[Step 2] BIC/AIC 搜索 (k=2~8, covariance_type=full/diag)...")

K_RANGE = range(2, 9)
COV_TYPES = ['full', 'diag']

bic_aic_results = []

for cov_type in COV_TYPES:
    for k in K_RANGE:
        gmm = GaussianMixture(
            n_components=k,
            covariance_type=cov_type,
            n_init=5,
            max_iter=300,
            random_state=RANDOM_STATE,
        )
        gmm.fit(X_pca)

        bic_val = gmm.bic(X_pca)
        aic_val = gmm.aic(X_pca)
        labels  = gmm.predict(X_pca)

        sil = silhouette_score(X_pca, labels)
        ch  = calinski_harabasz_score(X_pca, labels)
        db  = davies_bouldin_score(X_pca, labels)
        ari = adjusted_rand_score(y, labels)
        nmi = normalized_mutual_info_score(y, labels)

        bic_aic_results.append({
            'k': k, 'cov_type': cov_type,
            'BIC': bic_val, 'AIC': aic_val,
            'silhouette': sil, 'calinski_harabasz': ch,
            'davies_bouldin': db, 'ARI': ari, 'NMI': nmi,
            'converged': gmm.converged_,
            'n_iter': gmm.n_iter_,
        })

        print(f"  k={k}, cov={cov_type:5s}: BIC={bic_val:.0f}, AIC={aic_val:.0f}, "
              f"Sil={sil:.4f}, ARI={ari:.4f}, NMI={nmi:.4f}, "
              f"converged={gmm.converged_}")

bic_aic_df = pd.DataFrame(bic_aic_results)
bic_aic_df.to_csv('task3_gmm_results.csv', index=False)
print("  已保存 task3_gmm_results.csv")

# ============================================================
# 3. Fig 7 — BIC / AIC vs k
# ============================================================
print("\n[Step 3] 绘制 BIC/AIC 图...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

line_styles = {'full': '-o', 'diag': '--s'}
line_colors = {'full': '#4965b0', 'diag': '#f36f43'}

for metric_name, ax in zip(['BIC', 'AIC'], axes):
    for cov_type in COV_TYPES:
        subset = bic_aic_df[bic_aic_df['cov_type'] == cov_type]
        ax.plot(subset['k'], subset[metric_name],
                line_styles[cov_type], color=line_colors[cov_type],
                linewidth=2, markersize=7, label=f'{cov_type}')

    # 标注最优点
    best_idx = bic_aic_df[metric_name].idxmin()
    best_row = bic_aic_df.loc[best_idx]
    ax.annotate(f"Best: k={int(best_row['k'])}, {best_row['cov_type']}\n({best_row[metric_name]:.0f})",
                xy=(best_row['k'], best_row[metric_name]),
                xytext=(20, 20), textcoords='offset points',
                fontsize=10, color='red', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='red'))

    ax.set_xlabel('Number of Components (k)')
    ax.set_ylabel(metric_name)
    ax.set_title(f'{metric_name} (lower = better)')
    ax.set_xticks(list(K_RANGE))
    ax.legend(title='Covariance', fontsize=10)

fig.suptitle('GMM Model Selection: BIC and AIC', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('fig7_bic_aic.png', bbox_inches='tight')
plt.close()
print("  已保存 fig7_bic_aic.png")

# ============================================================
# 4. 确定最优 GMM 并训练最终模型
# ============================================================
# 根据 BIC 选择最优配置
best_bic_idx = bic_aic_df['BIC'].idxmin()
best_config  = bic_aic_df.loc[best_bic_idx]
BEST_K_GMM   = int(best_config['k'])
BEST_COV     = best_config['cov_type']

print(f"\n[Step 4] BIC 最优配置: k={BEST_K_GMM}, cov_type={BEST_COV}")

# 同时训练 k=3 的模型用于和 K-Means 公平对比
# 如果 BIC 最优不是 k=3，则两个都训练
gmm_models = {}

# BIC 最优模型
gmm_best = GaussianMixture(
    n_components=BEST_K_GMM,
    covariance_type=BEST_COV,
    n_init=10,
    max_iter=300,
    random_state=RANDOM_STATE,
)
gmm_best.fit(X_pca)
gmm_best_labels = gmm_best.predict(X_pca)
gmm_best_probs  = gmm_best.predict_proba(X_pca)
gmm_models['bic_best'] = (gmm_best, gmm_best_labels, gmm_best_probs, BEST_K_GMM, BEST_COV)

print(f"  BIC 最优模型簇分布: {dict(zip(*np.unique(gmm_best_labels, return_counts=True)))}")

# k=3 模型（用于和 K-Means 对比）
if BEST_K_GMM != 3:
    gmm_k3 = GaussianMixture(
        n_components=3,
        covariance_type=BEST_COV,
        n_init=10,
        max_iter=300,
        random_state=RANDOM_STATE,
    )
    gmm_k3.fit(X_pca)
    gmm_k3_labels = gmm_k3.predict(X_pca)
    gmm_k3_probs  = gmm_k3.predict_proba(X_pca)
    gmm_models['k3'] = (gmm_k3, gmm_k3_labels, gmm_k3_probs, 3, BEST_COV)
    print(f"  k=3 对比模型簇分布: {dict(zip(*np.unique(gmm_k3_labels, return_counts=True)))}")

# 选择主要分析的模型（优先用 k=3 和 K-Means 对比，除非 BIC 最优就是 k=3）
if BEST_K_GMM == 3:
    gmm_main, gmm_labels, gmm_probs = gmm_best, gmm_best_labels, gmm_best_probs
    GMM_K_MAIN = 3
else:
    gmm_main, gmm_labels, gmm_probs = gmm_models['k3'][0], gmm_models['k3'][1], gmm_models['k3'][2]
    GMM_K_MAIN = 3

print(f"\n  主分析模型: k={GMM_K_MAIN}, cov={BEST_COV}")

# ============================================================
# 5. Fig 8 — t-SNE: 真实标签 vs GMM
# ============================================================
print("\n[Step 5] 绘制 t-SNE 对比图 (Ground Truth vs GMM)...")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# 左图：真实标签
ax = axes[0]
for cls in [2, 1, 0]:
    mask = y == cls
    ax.scatter(emb_tsne[mask, 0], emb_tsne[mask, 1],
               c=COLORS_TARGET[cls], s=12, alpha=0.45,
               linewidths=0, rasterized=True)
handles = [mpatches.Patch(color=COLORS_TARGET[c], label=LABELS_TARGET[c]) for c in [2, 1, 0]]
ax.legend(handles=handles, loc='best', framealpha=0.7, fontsize=10)
ax.set_title('Ground Truth Labels', fontsize=13)
ax.set_xlabel('t-SNE dim 1'); ax.set_ylabel('t-SNE dim 2')
ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

# 右图：GMM 聚类标签
ax = axes[1]
for cls in sorted(np.unique(gmm_labels)):
    mask = gmm_labels == cls
    ax.scatter(emb_tsne[mask, 0], emb_tsne[mask, 1],
               c=COLORS_CLUSTER[cls], s=12, alpha=0.45,
               linewidths=0, rasterized=True)
handles = [mpatches.Patch(color=COLORS_CLUSTER[c], label=f'Cluster {c}')
           for c in sorted(np.unique(gmm_labels))]
ax.legend(handles=handles, loc='best', framealpha=0.7, fontsize=10)
ax.set_title(f'GMM Clusters (k={GMM_K_MAIN}, cov={BEST_COV})', fontsize=13)
ax.set_xlabel('t-SNE dim 1'); ax.set_ylabel('t-SNE dim 2')
ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

fig.suptitle('t-SNE Visualization: Ground Truth vs GMM Clustering',
             fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('fig8_gmm_tsne_comparison.png', bbox_inches='tight')
plt.close()
print("  已保存 fig8_gmm_tsne_comparison.png")

# ============================================================
# 6. Fig 9 — 簇特征热力图
# ============================================================
print("\n[Step 6] 绘制 GMM 簇特征热力图...")

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
X_analysis['Cluster'] = gmm_labels
cluster_means = X_analysis.groupby('Cluster')[scale_cols].mean()
cluster_means_display = cluster_means.rename(columns=short_names)

fig, ax = plt.subplots(figsize=(16, 5))
sns.heatmap(cluster_means_display, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, linewidths=0.5, ax=ax,
            yticklabels=[f'Cluster {i}' for i in range(GMM_K_MAIN)])
ax.set_title(f'GMM Cluster Feature Profiles (Standardized Mean, k={GMM_K_MAIN}, cov={BEST_COV})',
             fontsize=13)
ax.set_xlabel(''); ax.set_ylabel('')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('fig9_gmm_cluster_profiles.png', bbox_inches='tight')
plt.close()
print("  已保存 fig9_gmm_cluster_profiles.png")

# ============================================================
# 7. Fig 10 — 软分配概率分析
# ============================================================
print("\n[Step 7] 绘制软分配概率分析...")

max_probs = gmm_probs.max(axis=1)
entropy = -np.sum(gmm_probs * np.log(gmm_probs + 1e-10), axis=1)

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# 10a: 最大归属概率直方图
ax = axes[0]
ax.hist(max_probs, bins=50, color='#4965b0', alpha=0.75, edgecolor='white')
ax.axvline(x=0.6, color='red', linestyle='--', linewidth=2,
           label=f'Threshold 0.6')
n_uncertain = (max_probs < 0.6).sum()
ax.annotate(f'Uncertain: {n_uncertain} ({n_uncertain/len(max_probs)*100:.1f}%)',
            xy=(0.45, ax.get_ylim()[1]*0.8), fontsize=11, color='red', fontweight='bold')
ax.set_xlabel('Maximum Assignment Probability')
ax.set_ylabel('Number of Students')
ax.set_title('Distribution of Max Cluster Probability')
ax.legend(fontsize=10)

# 10b: 分配熵直方图
ax = axes[1]
ax.hist(entropy, bins=50, color='#80cba4', alpha=0.75, edgecolor='white')
ax.axvline(x=np.log(GMM_K_MAIN) * 0.5, color='red', linestyle='--', linewidth=2,
           label=f'50% max entropy')
ax.set_xlabel('Assignment Entropy')
ax.set_ylabel('Number of Students')
ax.set_title('Distribution of Assignment Entropy')
ax.legend(fontsize=10)

# 10c: 边界学生的真实标签分布
ax = axes[2]
uncertain_mask = max_probs < 0.6
if uncertain_mask.sum() > 0:
    uncertain_targets = y[uncertain_mask]
    certain_targets = y[~uncertain_mask]

    labels_list = ['Dropout', 'Enrolled', 'Graduate']
    uncertain_pcts = [np.mean(uncertain_targets == c) * 100 for c in [0, 1, 2]]
    certain_pcts   = [np.mean(certain_targets == c) * 100 for c in [0, 1, 2]]

    x_pos = np.arange(3)
    width = 0.35
    bars1 = ax.bar(x_pos - width/2, certain_pcts, width, label='Certain (max prob ≥ 0.6)',
                   color='#4965b0', alpha=0.8)
    bars2 = ax.bar(x_pos + width/2, uncertain_pcts, width, label='Uncertain (max prob < 0.6)',
                   color='#f36f43', alpha=0.8)

    # 标注百分比
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + 1,
                    f'{h:.1f}%', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels_list)
    ax.set_ylabel('Percentage (%)')
    ax.set_title(f'Target Distribution: Certain vs Uncertain Students')
    ax.legend(fontsize=9)

fig.suptitle('GMM Soft Assignment Analysis', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('fig10_gmm_soft_assignment.png', bbox_inches='tight')
plt.close()
print("  已保存 fig10_gmm_soft_assignment.png")

# ============================================================
# 8. Fig 11 — K-Means vs GMM t-SNE 对比
# ============================================================
print("\n[Step 8] 绘制 K-Means vs GMM 对比...")

fig, axes = plt.subplots(1, 3, figsize=(22, 7))

# 左图：真实标签
ax = axes[0]
for cls in [2, 1, 0]:
    mask = y == cls
    ax.scatter(emb_tsne[mask, 0], emb_tsne[mask, 1],
               c=COLORS_TARGET[cls], s=12, alpha=0.45,
               linewidths=0, rasterized=True)
handles = [mpatches.Patch(color=COLORS_TARGET[c], label=LABELS_TARGET[c]) for c in [2, 1, 0]]
ax.legend(handles=handles, loc='best', framealpha=0.7, fontsize=10)
ax.set_title('Ground Truth', fontsize=13)
ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

# 中图：K-Means
ax = axes[1]
for cls in sorted(np.unique(km_labels)):
    mask = km_labels == cls
    ax.scatter(emb_tsne[mask, 0], emb_tsne[mask, 1],
               c=COLORS_CLUSTER[cls], s=12, alpha=0.45,
               linewidths=0, rasterized=True)
handles = [mpatches.Patch(color=COLORS_CLUSTER[c], label=f'Cluster {c}')
           for c in sorted(np.unique(km_labels))]
ax.legend(handles=handles, loc='best', framealpha=0.7, fontsize=10)
ax.set_title('K-Means (k=3)', fontsize=13)
ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

# 右图：GMM
ax = axes[2]
for cls in sorted(np.unique(gmm_labels)):
    mask = gmm_labels == cls
    ax.scatter(emb_tsne[mask, 0], emb_tsne[mask, 1],
               c=COLORS_CLUSTER[cls], s=12, alpha=0.45,
               linewidths=0, rasterized=True)
handles = [mpatches.Patch(color=COLORS_CLUSTER[c], label=f'Cluster {c}')
           for c in sorted(np.unique(gmm_labels))]
ax.legend(handles=handles, loc='best', framealpha=0.7, fontsize=10)
ax.set_title(f'GMM (k={GMM_K_MAIN}, {BEST_COV})', fontsize=13)
ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

fig.suptitle('t-SNE Comparison: Ground Truth vs K-Means vs GMM',
             fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('fig11_gmm_vs_kmeans_tsne.png', bbox_inches='tight')
plt.close()
print("  已保存 fig11_gmm_vs_kmeans_tsne.png")

# ============================================================
# 9. 交叉分析
# ============================================================
print("\n[Step 9] GMM 簇与真实标签交叉分析...")

cross_tab = pd.crosstab(
    pd.Series(gmm_labels, name='GMM Cluster'),
    pd.Series(y, name='Target').map(LABELS_TARGET),
    margins=True
)
print("\n  交叉频数表:")
print(cross_tab.to_string().replace('\n', '\n  '))

cross_pct = pd.crosstab(
    pd.Series(gmm_labels, name='GMM Cluster'),
    pd.Series(y, name='Target').map(LABELS_TARGET),
    normalize='index'
)
print("\n  各簇中的类别比例:")
print((cross_pct * 100).round(1).to_string().replace('\n', '\n  '))

# ============================================================
# 10. Fig 12 — 最终综合评估对比表
# ============================================================
print("\n[Step 10] 生成最终评估对比...")

# 计算 GMM k=3 的所有指标
gmm_metrics = {
    'Silhouette Score': silhouette_score(X_pca, gmm_labels),
    'Calinski-Harabasz': calinski_harabasz_score(X_pca, gmm_labels),
    'Davies-Bouldin': davies_bouldin_score(X_pca, gmm_labels),
    'ARI': adjusted_rand_score(y, gmm_labels),
    'NMI': normalized_mutual_info_score(y, gmm_labels),
}

km_metrics = {
    'Silhouette Score': silhouette_score(X_pca, km_labels),
    'Calinski-Harabasz': calinski_harabasz_score(X_pca, km_labels),
    'Davies-Bouldin': davies_bouldin_score(X_pca, km_labels),
    'ARI': adjusted_rand_score(y, km_labels),
    'NMI': normalized_mutual_info_score(y, km_labels),
}

# 打印对比
print("\n  " + "=" * 60)
print(f"  {'Metric':<25} {'K-Means':>10} {'GMM':>10} {'Better':>10}")
print("  " + "-" * 60)

directions = {
    'Silhouette Score': 'higher',
    'Calinski-Harabasz': 'higher',
    'Davies-Bouldin': 'lower',
    'ARI': 'higher',
    'NMI': 'higher',
}

for metric in km_metrics:
    km_val  = km_metrics[metric]
    gmm_val = gmm_metrics[metric]
    d = directions[metric]
    if d == 'higher':
        winner = 'GMM' if gmm_val > km_val else 'K-Means'
    else:
        winner = 'GMM' if gmm_val < km_val else 'K-Means'
    print(f"  {metric:<25} {km_val:>10.4f} {gmm_val:>10.4f} {winner:>10}")

print("  " + "=" * 60)

# 绘制对比图
fig, axes = plt.subplots(1, 5, figsize=(22, 5))

metrics_list = list(km_metrics.keys())
for i, metric in enumerate(metrics_list):
    ax = axes[i]
    vals = [km_metrics[metric], gmm_metrics[metric]]
    colors = ['#4965b0', '#f36f43']
    bars = ax.bar(['K-Means', 'GMM'], vals, color=colors, width=0.5, alpha=0.85)

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    d = directions[metric]
    arrow = '↑' if d == 'higher' else '↓'
    ax.set_title(f'{metric}\n({arrow} better)', fontsize=11)
    ax.set_ylabel('')

fig.suptitle('K-Means vs GMM: Comprehensive Metric Comparison (k=3)',
             fontsize=14, y=1.03)
plt.tight_layout()
plt.savefig('fig12_final_comparison.png', bbox_inches='tight')
plt.close()
print("  已保存 fig12_final_comparison.png")

# ============================================================
# 11. 软聚类概率的详细统计
# ============================================================
print("\n[Step 11] 软聚类概率统计...")

max_probs = gmm_probs.max(axis=1)
print(f"\n  最大归属概率统计:")
print(f"    均值: {max_probs.mean():.4f}")
print(f"    中位数: {np.median(max_probs):.4f}")
print(f"    < 0.5 (极度不确定): {(max_probs < 0.5).sum()} ({(max_probs < 0.5).mean()*100:.1f}%)")
print(f"    0.5~0.6 (较不确定): {((max_probs >= 0.5) & (max_probs < 0.6)).sum()}")
print(f"    0.6~0.8 (中等确定): {((max_probs >= 0.6) & (max_probs < 0.8)).sum()}")
print(f"    0.8~0.95 (较确定):  {((max_probs >= 0.8) & (max_probs < 0.95)).sum()}")
print(f"    ≥ 0.95 (高度确定):  {(max_probs >= 0.95).sum()} ({(max_probs >= 0.95).mean()*100:.1f}%)")

# 边界学生分析
uncertain_mask = max_probs < 0.6
print(f"\n  边界学生 (max_prob < 0.6): {uncertain_mask.sum()} 人")
if uncertain_mask.sum() > 0:
    uncertain_y = y[uncertain_mask]
    print(f"    其中 Dropout:  {(uncertain_y == 0).sum()} ({(uncertain_y == 0).mean()*100:.1f}%)")
    print(f"    其中 Enrolled: {(uncertain_y == 1).sum()} ({(uncertain_y == 1).mean()*100:.1f}%)")
    print(f"    其中 Graduate: {(uncertain_y == 2).sum()} ({(uncertain_y == 2).mean()*100:.1f}%)")

# ============================================================
# 12. 总结
# ============================================================
print("\n" + "=" * 65)
print("GMM 聚类分析完成！")
print("=" * 65)
print("\n输出文件:")
print("  fig7_bic_aic.png                   — BIC/AIC 模型选择")
print("  fig8_gmm_tsne_comparison.png       — t-SNE 真实 vs GMM")
print("  fig9_gmm_cluster_profiles.png      — GMM 簇特征热力图")
print("  fig10_gmm_soft_assignment.png      — 软分配概率分析")
print("  fig11_gmm_vs_kmeans_tsne.png       — 三合一对比 (Truth/KM/GMM)")
print("  fig12_final_comparison.png         — 指标对比柱状图")
print("  task3_gmm_results.csv              — 完整搜索结果")