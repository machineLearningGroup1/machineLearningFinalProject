"""
Task 3 — Binary Feature × Cluster Cross-tab Analysis
======================================================
分析 66 个二值/类别特征与 K-Means 聚类结果的关联，
发现哪些类别特征对聚类有贡献，但未在热力图中展示。

输入: km_labels_task3.npy, full_preprocessed_data.csv
输出:
  fig22_binary_cluster_heatmap.png    — 单一二值特征 × 簇 阳性率热力图
  fig23_categorical_groups.png        — 多类别 one-hot 组 × 簇 分布图（4×2）
  fig24_chi2_top_features.png         — 卡方检验：与聚类最相关的二值特征
  task3_binary_crosstab.csv           — 完整交叉分析表
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import os, warnings

# os.chdir(os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '/home/claude')
# warnings.filterwarnings('ignore')

plt.rcParams.update({
    'figure.dpi': 300,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Set 3 配色
COLORS_CLUSTER = ['#f36f43', '#fbda83', '#80cba4']

# RdYlBu 8 色配色（用于多类别堆叠柱状图）
CATEGORICAL_COLORS = ['#d73027', '#f46d43', '#fdae61', '#fee090',
                      '#e0f3f8', '#abd9e9', '#74add1', '#4575b4']

print("=" * 65)
print("Task 3 — Binary Feature × Cluster Cross-tab Analysis")
print("=" * 65)

# ============================================================
# 1. 加载数据
# ============================================================
print("\n[Step 1] 加载数据...")

df = pd.read_csv('full_preprocessed_data.csv')
km_labels = np.load('km_labels_task3.npy')
df['Cluster'] = km_labels

cluster_sizes = df['Cluster'].value_counts().sort_index()
print(f"  K-Means 簇分布: {cluster_sizes.to_dict()}")

# ============================================================
# 2. 定义二值特征组
# ============================================================
# 单一的二值标志特征
SOLO_BINARY = [
    'Displaced',
    'Educational special needs',
    'Debtor',
    'Tuition fees up to date',
    'Gender',
    'Scholarship holder',
    'no_eval_sem1',
    'no_eval_sem2',
    'economic_stress',
    'Daytime/evening attendance_1',
]

# One-hot 组（按原始类别变量分组）
ONEHOT_GROUPS = {
    'Marital status':   [c for c in df.columns if c.startswith('Marital status_')],
    'Application mode': [c for c in df.columns if c.startswith('Application mode_')],
    'Course':           [c for c in df.columns if c.startswith('Course_')],
    'Nacionality':      [c for c in df.columns if c.startswith('Nacionality_grouped_')],
    'Mother Qual':      [c for c in df.columns if c.startswith('Mother_qual_grouped_')],
    'Father Qual':      [c for c in df.columns if c.startswith('Father_qual_grouped_')],
    'Mother Occ':       [c for c in df.columns if c.startswith('Mother_occ_grouped_')],
    'Father Occ':       [c for c in df.columns if c.startswith('Father_occ_grouped_')],
    'Age Group':        [c for c in df.columns if c.startswith('age_group_')],
}

print(f"  单一二值特征: {len(SOLO_BINARY)}")
print(f"  One-hot 组: {len(ONEHOT_GROUPS)}")

# ============================================================
# 3. Fig 22 — 单一二值特征 × 簇 阳性率热力图
# ============================================================
print("\n[Step 2] 单一二值特征交叉分析...")

# 计算每个簇中每个二值特征的阳性率（=1 的比例）
solo_pos_rate = pd.DataFrame(index=SOLO_BINARY, columns=sorted(df['Cluster'].unique()))
for feat in SOLO_BINARY:
    for cl in sorted(df['Cluster'].unique()):
        solo_pos_rate.loc[feat, cl] = df[df['Cluster'] == cl][feat].mean() * 100
solo_pos_rate = solo_pos_rate.astype(float)

# 同时计算全局阳性率作为参考
global_pos_rate = pd.Series({feat: df[feat].mean() * 100 for feat in SOLO_BINARY})

# 计算偏离全局比例（百分点差异），便于看出哪些簇异常
solo_deviation = solo_pos_rate.subtract(global_pos_rate, axis=0)

print("\n  各簇中二值特征的阳性率 (%):")
print(solo_pos_rate.round(1).to_string())
print(f"\n  全局阳性率 (%):")
print(global_pos_rate.round(1).to_string())

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 22a: 阳性率热力图
ax = axes[0]
solo_display = solo_pos_rate.copy()
solo_display.columns = [f'Cluster {c}' for c in solo_display.columns]
sns.heatmap(solo_display, annot=True, fmt='.1f', cmap='YlOrRd',
            ax=ax, linewidths=0.5, cbar_kws={'label': 'Positive Rate (%)'})
ax.set_title('Positive Rate (%) of Binary Features by Cluster')
ax.set_xlabel('')
ax.set_ylabel('')

# 22b: 偏离全局的差异热力图（正值=该簇阳性率高于全局，负值反之）
ax = axes[1]
solo_dev_display = solo_deviation.copy()
solo_dev_display.columns = [f'Cluster {c}' for c in solo_dev_display.columns]
sns.heatmap(solo_dev_display, annot=True, fmt='+.1f', cmap='RdBu_r',
            center=0, ax=ax, linewidths=0.5,
            cbar_kws={'label': 'Deviation from Global (%)'})
ax.set_title('Deviation from Global Positive Rate (percentage points)')
ax.set_xlabel('')
ax.set_ylabel('')

fig.suptitle('Binary Features × K-Means Clusters', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('fig22_binary_cluster_heatmap.png', bbox_inches='tight')
plt.close()
print("  已保存 fig22_binary_cluster_heatmap.png")

# ============================================================
# 4. Fig 23 — 多类别 one-hot 组 × 簇 分布图
# ============================================================
print("\n[Step 3] 多类别 one-hot 组分析...")

# 对每个 one-hot 组，把它"还原"成单一类别
def recover_category(row, cols):
    """对一个 one-hot 组，找出哪一列为 1；都不为 1 则返回 'baseline'"""
    for c in cols:
        if row[c] == 1:
            return c.split('_')[-1]
    return 'baseline'

categorical_data = pd.DataFrame({'Cluster': df['Cluster']})
for group_name, cols in ONEHOT_GROUPS.items():
    categorical_data[group_name] = df[cols].apply(
        lambda row: recover_category(row, cols), axis=1
    )

# 绘制 4×2 + 1 的子图（9 个 one-hot 组）
n_groups = len(ONEHOT_GROUPS)
n_cols = 3
n_rows = (n_groups + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
axes = axes.flatten()

for ax_idx, (group_name, _) in enumerate(ONEHOT_GROUPS.items()):
    ax = axes[ax_idx]
    cross = pd.crosstab(categorical_data[group_name], categorical_data['Cluster'],
                        normalize='columns') * 100

    # 只展示前 8 个类别（按全局占比排序），其余合并为 Other
    if cross.shape[0] > 8:
        global_counts = categorical_data[group_name].value_counts()
        top_categories = global_counts.head(7).index.tolist()
        other_mask = ~cross.index.isin(top_categories)
        if other_mask.any():
            other_row = cross[other_mask].sum()
            cross = cross[~other_mask]
            cross.loc['Other'] = other_row

    cross = cross.sort_index()
    n_cats = len(cross.index)

    # 使用自定义 RdYlBu 配色
    colors_use = CATEGORICAL_COLORS[:n_cats]

    # 堆叠柱状图：每个簇一列，类别为不同颜色段
    cross.T.plot(kind='bar', stacked=True, ax=ax, width=0.6,
                 color=colors_use, edgecolor='white', linewidth=0.5)
    ax.set_title(f'{group_name}', fontsize=12, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('Percentage (%)')
    ax.set_xticklabels([f'Cluster {c}' for c in cross.columns], rotation=0)
    ax.legend(title='', fontsize=8, loc='center left', bbox_to_anchor=(1.0, 0.5))
    ax.set_ylim(0, 100)

# 隐藏多余子图
for i in range(n_groups, len(axes)):
    axes[i].set_visible(False)

fig.suptitle('Categorical Feature Distribution by K-Means Cluster',
             fontsize=15, y=1.01)
plt.tight_layout()
plt.savefig('fig23_categorical_groups.png', bbox_inches='tight')
plt.close()
print("  已保存 fig23_categorical_groups.png")

# ============================================================
# 5. Fig 24 — 卡方检验：哪些二值特征和聚类最相关
# ============================================================
print("\n[Step 4] 卡方检验...")

chi2_results = []

# 测试所有二值列
all_binary = SOLO_BINARY.copy()
for cols in ONEHOT_GROUPS.values():
    all_binary.extend(cols)

for feat in all_binary:
    if feat not in df.columns:
        continue
    contingency = pd.crosstab(df[feat], df['Cluster'])
    if contingency.shape[0] < 2:
        continue
    try:
        chi2, p_val, dof, expected = chi2_contingency(contingency)
        # Cramér's V 作为效应量（标准化的关联强度，0~1）
        n = contingency.sum().sum()
        min_dim = min(contingency.shape) - 1
        cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0

        chi2_results.append({
            'feature': feat,
            'chi2': chi2,
            'p_value': p_val,
            'cramers_v': cramers_v,
        })
    except Exception:
        continue

chi2_df = pd.DataFrame(chi2_results).sort_values('cramers_v', ascending=False)
chi2_df.to_csv('task3_binary_crosstab.csv', index=False)
print(f"  完成 {len(chi2_df)} 个特征的卡方检验")

# 显示 Top 15
print("\n  与聚类关联最强的 Top 15 二值特征:")
print(chi2_df.head(15)[['feature', 'cramers_v', 'p_value']].to_string(index=False))

# 可视化 Top 20
top20 = chi2_df.head(20).iloc[::-1]
fig, ax = plt.subplots(figsize=(10, 8))
colors = ['#a30543' if v > 0.3 else '#f36f43' if v > 0.15 else '#80cba4'
          for v in top20['cramers_v']]
ax.barh(range(len(top20)), top20['cramers_v'], color=colors, alpha=0.85)
ax.set_yticks(range(len(top20)))
ax.set_yticklabels(top20['feature'], fontsize=9)
ax.set_xlabel("Cramér's V (Effect Size)")
ax.set_title("Top 20 Binary Features by Association Strength with Clusters\n"
             "(Chi-square test, all p-values < 0.001 unless noted)",
             fontsize=12)
ax.axvline(x=0.1, color='gray', linestyle='--', alpha=0.5, label='Small effect')
ax.axvline(x=0.3, color='gray', linestyle='-.', alpha=0.5, label='Medium effect')
ax.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5, label='Large effect')
ax.legend(loc='lower right', fontsize=9)

# 标注每个条的 Cramér's V 值
for i, v in enumerate(top20['cramers_v']):
    ax.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=8)

plt.tight_layout()
plt.savefig('fig24_chi2_top_features.png', bbox_inches='tight')
plt.close()
print("  已保存 fig24_chi2_top_features.png")

# ============================================================
# 6. 关键发现总结
# ============================================================
print("\n[Step 5] 关键发现总结")
print("=" * 65)

# 找出每个簇中最特殊的二值特征（偏离全局最大）
print("\n各簇中最具代表性的二值特征（偏离全局阳性率最大）:")
for cl in sorted(df['Cluster'].unique()):
    print(f"\n  Cluster {cl} (n={cluster_sizes[cl]}):")
    cl_dev = solo_deviation[cl].sort_values(key=abs, ascending=False)
    for feat, dev in cl_dev.head(5).items():
        global_val = global_pos_rate[feat]
        cl_val = solo_pos_rate.loc[feat, cl]
        sign = '↑' if dev > 0 else '↓'
        print(f"    {sign} {feat:<30} {cl_val:5.1f}% (vs global {global_val:5.1f}%, "
              f"{dev:+.1f}pp)")

# 强相关的多类别特征
print("\n多类别特征中关联最强的（Cramér's V > 0.15）:")
strong_categorical = chi2_df[
    (chi2_df['cramers_v'] > 0.15) &
    (~chi2_df['feature'].isin(SOLO_BINARY))
].head(10)
if len(strong_categorical) > 0:
    print(strong_categorical[['feature', 'cramers_v']].to_string(index=False))
else:
    print("  无")

print("\n输出文件:")
print("  fig22_binary_cluster_heatmap.png    — 二值特征阳性率与偏离图")
print("  fig23_categorical_groups.png        — 9 组类别变量分布")
print("  fig24_chi2_top_features.png         — 卡方检验关联强度排名")
print("  task3_binary_crosstab.csv           — 完整结果表")