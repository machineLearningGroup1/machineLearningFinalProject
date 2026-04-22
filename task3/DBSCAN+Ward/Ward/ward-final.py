import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

# 中文显示
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Hiragino Sans GB', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 加载数据 + 专业标准化（连续特征Z-score，二值不动）
# ============================================================
print("[Step 1] 数据加载与标准化...")

df = pd.read_csv("full_preprocessed_data.csv")

# ✅ 关键：聚类必须去掉标签！只留特征
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
# 2. PCA 降维
# ============================================================
pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_processed)

# ============================================================
# 3. 聚类参数
# ============================================================
n_clusters = 2
sample_size = 1000

# ============================================================
# 图1：原始特征 → t-SNE
# ============================================================
print("\n[绘图 1/4] 原始特征 t-SNE...")
model_ori = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
labels_ori = model_ori.fit_predict(X_processed)

tsne_ori = TSNE(n_components=2, random_state=42, perplexity=30, init="pca")
X_tsne_ori = tsne_ori.fit_transform(X_processed[:sample_size])

plt.figure(figsize=(10,6))
plt.scatter(X_tsne_ori[:,0], X_tsne_ori[:,1], c=labels_ori[:sample_size], cmap="viridis", s=30, alpha=0.7)
plt.title(f"Ward 层次聚类 - 原始特征 (n={n_clusters})")
plt.grid(alpha=0.3)
plt.savefig(f"ward_tsne_original_n{n_clusters}.png", dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# 图2：PCA降维 → t-SNE
# ============================================================
print("\n[绘图 2/4] PCA特征 t-SNE...")
model_pca = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
labels_pca = model_pca.fit_predict(X_pca)

tsne_pca = TSNE(n_components=2, random_state=42, perplexity=30, init="pca")
X_tsne_pca = tsne_pca.fit_transform(X_pca[:sample_size])

plt.figure(figsize=(10,6))
plt.scatter(X_tsne_pca[:,0], X_tsne_pca[:,1], c=labels_pca[:sample_size], cmap="viridis", s=30, alpha=0.7)
plt.title(f"Ward 层次聚类 - PCA降维 (n={n_clusters})")
plt.grid(alpha=0.3)
plt.savefig(f"ward_tsne_pca_n{n_clusters}.png", dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# 图3：原始特征 → 树状图
# ============================================================
print("\n[绘图 3/4] 原始特征树状图...")
plt.figure(figsize=(14,6))
linked_ori = linkage(X_processed[:500], method='ward')
dendrogram(linked_ori, truncate_mode='lastp', p=10)
plt.title("树状图 - 原始特征")
plt.savefig(f"ward_dendrogram_original_n{n_clusters}.png", dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# 图4：PCA降维 → 树状图
# ============================================================
print("\n[绘图 4/4] PCA特征树状图...")
plt.figure(figsize=(14,6))
linked_pca = linkage(X_pca[:500], method='ward')
dendrogram(linked_pca, truncate_mode='lastp', p=10)
plt.title("树状图 - PCA降维特征")
plt.savefig(f"ward_dendrogram_pca_n{n_clusters}.png", dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
print("\n✅ 全部完成！生成 4 张图：")
print("1. ward_tsne_original_n2.png")
print("2. ward_tsne_pca_n2.png")
print("3. ward_dendrogram_original_n2.png")
print("4. ward_dendrogram_pca_n2.png")