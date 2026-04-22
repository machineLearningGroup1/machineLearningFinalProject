import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)
import warnings
warnings.filterwarnings('ignore')

# ======================
# 1. 读取数据
# ======================
df = pd.read_csv("full_preprocessed_data.csv")
X = df.drop(columns=['Target', 'Target_binary']).copy()
y_true = df['Target'].values

# ======================
# 2. 标准化（连续特征）
# ======================
scale_cols = [c for c in X.columns if X[c].nunique() > 2]
scaler = StandardScaler()
X[scale_cols] = scaler.fit_transform(X[scale_cols])
X_processed = X.values

# ======================
# 3. PCA降维
# ======================
pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_processed)

# ======================
# 4. Ward 层次聚类
# ======================
n_clusters = 4
ward = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
y_ward = ward.fit_predict(X_pca)

sil_ward = silhouette_score(X_pca, y_ward)
ch_ward = calinski_harabasz_score(X_pca, y_ward)
db_ward = davies_bouldin_score(X_pca, y_ward)
ari_ward = adjusted_rand_score(y_true, y_ward)
nmi_ward = normalized_mutual_info_score(y_true, y_ward)

# ======================
# 5. DBSCAN
# ======================
eps_val = 3.0
min_val = 15
dbscan = DBSCAN(eps=eps_val, min_samples=min_val)
y_db = dbscan.fit_predict(X_pca)

mask = y_db != -1
X_valid = X_pca[mask]
y_valid = y_db[mask]
n_cluster_db = len(set(y_valid))

if n_cluster_db >= 2:
    sil_db = silhouette_score(X_valid, y_valid)
    ch_db = calinski_harabasz_score(X_valid, y_valid)
    db_db = davies_bouldin_score(X_valid, y_valid)
else:
    sil_db = ch_db = db_db = np.nan

ari_db = adjusted_rand_score(y_true, y_db)
nmi_db = normalized_mutual_info_score(y_true, y_db)

# ======================
# 6. 输出到文档 txt
# ======================
with open("cluster_evaluation_result.txt", "w", encoding="utf-8") as f:
    f.write("="*60 + "\n")
    f.write("           聚类模型评估结果\n")
    f.write("="*60 + "\n\n")

    f.write("【数据设置】\n")
    f.write(f"原始特征维度：{X.shape[1]}\n")
    f.write(f"PCA 降维后维度：{X_pca.shape[1]}\n")
    f.write(f"聚类数（Ward）：{n_clusters}\n")
    f.write(f"DBSCAN 参数：eps={eps_val}, min_samples={min_val}\n\n")

    f.write("="*60 + "\n")
    f.write("一、Ward 层次聚类评估指标\n")
    f.write("="*60 + "\n")
    f.write(f"轮廓系数（Silhouette）：{sil_ward:.4f}\n")
    f.write(f"CH 分数               ：{ch_ward:.4f}\n")
    f.write(f"DB 指数               ：{db_ward:.4f}\n")
    f.write(f"ARI（外部）            ：{ari_ward:.4f}\n")
    f.write(f"NMI（外部）           ：{nmi_ward:.4f}\n\n")

    f.write("="*60 + "\n")
    f.write("二、DBSCAN 评估指标\n")
    f.write("="*60 + "\n")
    f.write(f"有效聚类数量          ：{n_cluster_db}\n")
    if np.isnan(sil_db):
        f.write("轮廓系数              ：无法计算（聚类太少）\n")
        f.write("CH 分数               ：无法计算\n")
        f.write("DB 指数               ：无法计算\n")
    else:
        f.write(f"轮廓系数              ：{sil_db:.4f}\n")
        f.write(f"CH 分数               ：{ch_db:.4f}\n")
        f.write(f"DB 指数               ：{db_db:.4f}\n")
    f.write(f"ARI（外部）           ：{ari_db:.4f}\n")
    f.write(f"NMI（外部）           ：{nmi_db:.4f}\n\n")

    f.write("="*60 + "\n")
    f.write("指标说明：\n")
    f.write("• 轮廓系数：越高越好，范围 [-1,1]\n")
    f.write("• CH 分数：越高聚类效果越好\n")
    f.write("• DB 指数：越低越好\n")
    f.write("• ARI / NMI：越接近1表示与真实标签越一致\n")
    f.write("="*60 + "\n")

print("✅ 评估完成！")
print("📄 结果已保存至：cluster_evaluation_result.txt")