import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# 中文显示
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Hiragino Sans GB', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 数据加载 + 专业标准化（连续特征Z-score，二值不动）
# ============================================================
print("[Step 1] 数据加载与标准化...")

df = pd.read_csv("full_preprocessed_data.csv")

# ✅ 聚类必须去掉标签
X = df.drop(columns=['Target', 'Target_binary']).copy()

# 区分连续特征 & 二值特征
scale_cols = [c for c in X.columns if X[c].nunique() > 2]
binary_cols = [c for c in X.columns if X[c].nunique() <= 2]

print(f"  特征总数：{X.shape[1]}")
print(f"  连续特征：{len(scale_cols)} 列 → 标准化")
print(f"  二值特征：{len(binary_cols)} 列 → 保持0/1")

# 只标准化连续特征
scaler = StandardScaler()
X[scale_cols] = scaler.fit_transform(X[scale_cols])

X_processed = X.values

# ============================================================
# 2. PCA 降维（保留95%信息）
# ============================================================
pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_processed)

# ============================================================
# 3. DBSCAN 参数
# ============================================================
eps_val = 3.5
min_val = 20
sample_size = 1000

# ============================================================
# 图1：原始特征 → DBSCAN + t-SNE
# ============================================================
print("\n[绘图 1/2] 原始特征 DBSCAN...")
dbscan_ori = DBSCAN(eps=eps_val, min_samples=min_val)
labels_ori = dbscan_ori.fit_predict(X_processed)

tsne_ori = TSNE(n_components=2, random_state=42, perplexity=30, init="pca")
X_tsne_ori = tsne_ori.fit_transform(X_processed[:sample_size])

plt.figure(figsize=(10,6))
plt.scatter(X_tsne_ori[:,0], X_tsne_ori[:,1], c=labels_ori[:sample_size], cmap="viridis", s=30, alpha=0.7)
plt.title(f"DBSCAN - 原始特征 (eps={eps_val}, min={min_val})")
plt.grid(alpha=0.3)
plt.savefig(f"dbscan_tsne_original_eps{eps_val}_min{min_val}.png", dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# 图2：PCA降维 → DBSCAN + t-SNE
# ============================================================
print("\n[绘图 2/2] PCA降维 DBSCAN...")
dbscan_pca = DBSCAN(eps=eps_val, min_samples=min_val)
labels_pca = dbscan_pca.fit_predict(X_pca)

tsne_pca = TSNE(n_components=2, random_state=42, perplexity=30, init="pca")
X_tsne_pca = tsne_pca.fit_transform(X_pca[:sample_size])

plt.figure(figsize=(10,6))
plt.scatter(X_tsne_pca[:,0], X_tsne_pca[:,1], c=labels_pca[:sample_size], cmap="viridis", s=30, alpha=0.7)
plt.title(f"DBSCAN - PCA降维 (eps={eps_val}, min={min_val})")
plt.grid(alpha=0.3)
plt.savefig(f"dbscan_tsne_pca_eps{eps_val}_min{min_val}.png", dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# 结果统计
# ============================================================
n_ori = len(set(labels_ori)) - (1 if -1 in labels_ori else 0)
n_noise_ori = list(labels_ori).count(-1)

n_pca = len(set(labels_pca)) - (1 if -1 in labels_pca else 0)
n_noise_pca = list(labels_pca).count(-1)

print("\n✅ DBSCAN 完成！")
print(f"【原始特征】聚类数：{n_ori}，噪声点：{n_noise_ori}")
print(f"【PCA降维】聚类数：{n_pca}，噪声点：{n_noise_pca}")