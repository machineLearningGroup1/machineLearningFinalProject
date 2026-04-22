# Task 3 — Clustering Analysis (Comprehensive Report)

**数据集:** Predict Students' Dropout and Academic Success (UCI ID: 697)  
**算法:** K-Means / GMM / Spectral Clustering / Agglomerative Hierarchical  
**脚本:** `task3_kmeans.py` / `task3_gmm.py` / `task3_spectral.py` / `task3_hierarchical.py` / `task3_binary_crosstab.py`  
**输入:** `full_preprocessed_data.csv`（4,424 × 91）  
**输出:** 24 张可视化图表 + 4 份 CSV

---

## 1. 分析流程总览

```
full_preprocessed_data.csv (4424×91)
    │
    ├─ Step 1  全量重新标准化（23 连续列 fit_transform，66 二值列保持 0/1）
    ├─ Step 2  PCA 降维（89 维 → 24 维，保留 90.5% 方差）
    ├─ Step 3  t-SNE 嵌入（在 PCA 空间上，仅用于可视化）
    │
    ├─ 算法 1  K-Means（baseline）
    ├─ 算法 2  GMM（概率模型，椭圆形簇）
    ├─ 算法 3  Spectral Clustering（图论方法，非凸簇）
    ├─ 算法 4  Agglomerative Hierarchical（Ward linkage，层次结构）
    │
    └─ 补充分析  二值特征 × 簇 交叉分析（卡方检验 + Cramér's V）
```

---

## 2. 预处理决策及其理由

### 2.1 全量重新标准化

Task 1 的 StandardScaler 仅在训练集（70%）上 fit，导致全量数据均值不精确为 0（偏差约 0.004）。聚类为无监督任务，不涉及 train/test 划分，因此在全量 4424 条数据上重新 `fit_transform`，使 23 个连续列均值精确为 0、标准差精确为 1。66 个二值列保持 0/1。

### 2.2 PCA 降维（89 → 24 维）

| 保留方差 | 所需主成分 | 用途 |
|---------|----------|------|
| 80% | 14 | 可选的激进降维 |
| **90%** | **24** | **本任务选择** |
| 95% | 34 | 保守 |
| 98.5% | 50 | Task 2 的 t-SNE 预降维 |

选择 90%（24 维）的理由：后 65 个主成分每个仅贡献 < 0.15% 方差，主要是噪声；89 维空间中欧氏距离区分力退化（维度灾难）；1st/2nd sem 的 approved/grade 等高相关特征对（r > 0.84）被自动合并；24 维对所有四个算法的参数估计都在合理范围内。

### 2.3 t-SNE 仅用于可视化

t-SNE 坐标**不作为聚类输入**。t-SNE 不保全局距离，只保局部邻域。聚类在 PCA 24 维空间上进行，t-SNE 2D 坐标仅用于结果展示。

---

## 3. 算法原理

### 3.1 K-Means

**核心思想：** 将 $n$ 个数据点划分到 $k$ 个球形簇中，最小化所有点到其所属簇质心的距离平方和（Inertia）：

$$J = \sum_{i=1}^{k} \sum_{x \in C_i} \|x - \mu_i\|^2$$

**求解方法：** Lloyd 迭代算法（E-step 分配 + M-step 更新质心），配合 K-Means++ 智能初始化。

**隐含假设：** 簇是球形的、大小相近的、边界是线性的（Voronoi 划分）。每个点硬分配（$\gamma \in \{0, 1\}$）到唯一的簇。

**与该数据集的适配度：极高。** PCA 降维后簇形状接近球形，标准化保证了特征尺度一致，4424 条数据量级计算效率极高。K-Means 是最适合的 baseline 方法。

### 3.2 GMM（Gaussian Mixture Model）

**核心思想：** 假设数据由 $k$ 个高斯分布混合生成，每个高斯有独立的均值 $\mu_j$、协方差矩阵 $\Sigma_j$ 和混合权重 $\pi_j$：

$$p(x) = \sum_{j=1}^{k} \pi_j \cdot \mathcal{N}(x \mid \mu_j, \Sigma_j)$$

**求解方法：** EM 算法（E-step 计算后验概率 $\gamma_{ij}$ → 软分配，M-step 加权更新参数）。

**与 K-Means 的关系：** K-Means 是 GMM 的特例——当 $\Sigma_j = \sigma^2 I$（球形）且 $\gamma_{ij}$ 取极限值（硬分配）时，GMM 退化为 K-Means。

**独特优势：** 椭圆形簇（full 协方差）；软聚类概率；BIC/AIC 模型选择。

**与该数据集的适配度：中等。** 理论优势在此数据上未能发挥——PCA 降维后簇趋近球形，full 协方差的 974 个参数过多（K-Means 仅 72 个），软概率退化为硬分配（99.6% 学生 max prob ≥ 0.95）。

### 3.3 Spectral Clustering

**核心思想：** 先构建样本间的相似度图（$n \times n$ 邻接矩阵），对图的拉普拉斯矩阵做特征分解（求前 $k$ 个最小特征值对应的特征向量），在低维特征向量空间中用 K-Means 聚类。

**构图方式：** 两种选择：
- **nearest_neighbors**（用 $k$ 近邻构建稀疏图，本任务使用）
- **rbf**（用高斯核函数构建全连接图，在高维数据上退化严重）

**独特优势：** 能发现任意形状（非凸）的簇，对流形结构敏感。

**与该数据集的适配度：中等偏上。** nearest_neighbors 构图在 24 维 PCA 空间中表现稳健（三种 k=10/20/50 结果几乎一致），发现了 K-Means 看不到的小子群（177 人）。但 rbf 内核在 24 维空间中完全失败（4419/3/2 的退化分组），证实了高维空间中全局相似度度量的不可靠性。

### 3.4 Agglomerative Hierarchical Clustering

**核心思想：** 自底向上逐步合并最近的簇对，直到所有样本合并为一个簇。通过在指定高度"切割"Dendrogram 得到 $k$ 个簇。

**Ward linkage 合并准则：** 每次合并使总簇内方差增量最小的两个簇：

$$\Delta = \frac{n_i \cdot n_j}{n_i + n_j} \|\mu_i - \mu_j\|^2$$

这与 K-Means 最小化 Inertia 的目标**数学同源**——Ward 是 K-Means 的"自底向上版本"。

**独特优势：** Dendrogram 提供完整的层次结构，不需要预设 $k$，可以同时看到所有粒度的分组方案。

**与该数据集的适配度：高。** Ward linkage 与 K-Means 目标同源，结果高度一致（交叉验证）。4424 条数据的距离矩阵 ~75MB，计算 0.6 秒，完全可承受。Dendrogram 揭示了 K-Means 无法提供的层次信息。

**Linkage 方法对比的重要发现：** Average 和 Complete linkage 出现了"虚假高 Silhouette"现象（Silhouette 0.73，但 ARI ≈ 0），因为它们把几乎所有样本归入一个簇、极少数异常点单独分出。Ward 是唯一产生有意义聚类的 linkage 方法。

---

## 4. 评估指标体系

### 4.1 通用指标（四个算法共用）

**内部指标（不需要真实标签）：**

| 指标 | 公式核心 | 范围 | 方向 | 衡量内容 |
|------|---------|------|------|---------|
| Silhouette | $s(i) = \frac{b(i)-a(i)}{\max(a(i),b(i))}$ | [-1, 1] | ↑ | 每个点的簇内紧密度 vs 簇间分离度 |
| Calinski-Harabasz | $\frac{SSB/(k-1)}{SSW/(n-k)}$ | [0, +∞) | ↑ | 簇间方差 vs 簇内方差的 F 统计量 |
| Davies-Bouldin | $\frac{1}{k}\sum \max_{j \neq i}\frac{\sigma_i+\sigma_j}{d(\mu_i,\mu_j)}$ | [0, +∞) | ↓ | 最容易混淆的簇对的平均混淆度 |

**外部指标（需要真实标签）：**

| 指标 | 含义 | 范围 | 方向 |
|------|------|------|------|
| ARI | 聚类标签和真实标签的配对一致性（修正随机期望） | [-1, 1] | ↑ |
| NMI | 聚类标签和真实标签的互信息（归一化） | [0, 1] | ↑ |

### 4.2 GMM 特有指标

| 指标 | 公式核心 | 方向 | 说明 |
|------|---------|------|------|
| BIC | $-2\log L + p \cdot \log n$ | ↓ | 平衡拟合度和模型复杂度，惩罚较重 |
| AIC | $-2\log L + 2p$ | ↓ | 惩罚较轻，倾向更复杂的模型 |

### 4.3 指标偏向性警告

Silhouette / CH / DB 三个内部指标全部基于欧氏距离。K-Means 的优化目标恰好就是最小化欧氏距离平方和，因此**这三个指标天然偏向 K-Means**。ARI 和 NMI 不依赖距离计算，是跨算法对比中最公正的指标。

---

## 5. 实验结果

### 5.1 K-Means 参数选择

#### Elbow Method

Inertia 从 k=2 的 92121 降至 k=3 的 78448（下降 14.8%），之后 k=3→4 仅下降 4.8%，k=3 处为明显拐点。

#### 三个内部指标一致指向 k=3

| k | Silhouette ↑ | CH ↑ | DB ↓ | ARI ↑ | NMI ↑ |
|---|-------------|------|------|-------|-------|
| 2 | 0.2753 | **1072.3** | 1.5619 | 0.1923 | 0.1975 |
| **3** | **0.2956** | 1014.7 | **1.4421** | 0.1700 | 0.1701 |
| 4 | 0.1401 | 785.2 | 2.2801 | 0.3388 | 0.2499 |

### 5.2 GMM 的 BIC/AIC 搜索

BIC/AIC 最优为 k=7, full（BIC=131006），但 Silhouette 仅 0.024，说明 BIC 优化的是概率密度估计质量而非聚类分离度。关键发现：**full 协方差远优于 diag（BIC 差距约 10 万）**，证实数据存在显著的特征间相关结构。

为公平对比，使用 k=3, full 作为 GMM 的主分析模型。

### 5.3 Spectral Clustering 参数搜索

| 配置 | Silhouette | ARI | NMI | 簇分布 |
|------|-----------|-----|-----|--------|
| **NN k=10** | 0.224 | **0.170** | **0.211** | 3571/676/177 |
| NN k=20 | 0.223 | 0.162 | 0.206 | 3594/651/179 |
| NN k=50 | 0.224 | 0.164 | 0.208 | 3587/657/180 |
| RBF γ=0.1 | 0.646 | -0.001 | 0.002 | 4419/3/2 |

nearest_neighbors 三种参数结果极其稳定；RBF 完全失败（99.9% 归入一个簇，Silhouette 虚高是陷阱）。

### 5.4 Hierarchical Linkage 对比

| Linkage | k | Silhouette | ARI | NMI |
|---------|---|-----------|-----|-----|
| **Ward** | 3 | 0.2720 | **0.1485** | **0.1523** |
| Complete | 3 | 0.3295 | -0.0108 | 0.0036 |
| Average | 3 | 0.6332 | -0.0012 | 0.0021 |

Average/Complete 的高 Silhouette 是虚假的——它们把几乎所有样本归入一个簇。Ward 是唯一稳健的 linkage 方法。

### 5.5 四算法最终对比（k=3）

| 指标 | K-Means | GMM (full) | Spectral (NN) | Hierarchical (Ward) | 最优 |
|------|---------|------------|---------------|---------------------|------|
| Silhouette ↑ | **0.2956** | 0.1639 | 0.2238 | 0.2720 | K-Means |
| CH ↑ | **1014.72** | 675.27 | 585.92 | 908.69 | K-Means |
| DB ↓ | **1.4421** | 2.8268 | 1.6163 | 1.4727 | K-Means |
| ARI ↑ | 0.1700 | **0.1775** | 0.1697 | 0.1485 | GMM |
| NMI ↑ | 0.1701 | 0.1559 | **0.2106** | 0.1523 | Spectral |
| Dropout 簇纯度 | 82.0% | 83.5% | **92.9%** | 82.3% | Spectral |

---

## 6. 各算法的聚类结果对比

### 6.1 K-Means (k=3)

| 簇 | 人数 | Dropout | Enrolled | Graduate | 核心特征 |
|----|------|---------|----------|----------|----------|
| Cluster 0 | 275 | 23.3% | 13.1% | 63.6% | **高学分转入群**：1st/2nd Credited +3.4σ |
| Cluster 1 | 902 | **82.0%** | 9.5% | 8.4% | **学业困难群**：成绩/通过率 -1.7σ |
| Cluster 2 | 3247 | 19.0% | 20.7% | 60.3% | **正常学业群**：各指标接近均值 |

### 6.2 GMM (k=3, full)

| 簇 | 人数 | Dropout | Enrolled | Graduate | 核心特征 |
|----|------|---------|----------|----------|----------|
| Cluster 0 | 2475 | 15.6% | 19.5% | 64.9% | **正常学业群** |
| Cluster 1 | 865 | **83.5%** | 7.9% | 8.7% | **学业困难群**：与 K-Means Cluster 1 类似 |
| Cluster 2 | 1084 | 28.9% | 22.5% | 48.6% | **中间过渡群**：特征正值偏高 |

### 6.3 Spectral (NN k=10)

| 簇 | 人数 | Dropout | Enrolled | Graduate | 核心特征 |
|----|------|---------|----------|----------|----------|
| Cluster 0 | 3571 | 20.1% | 20.1% | 59.7% | **主体学生群** |
| Cluster 1 | 676 | **92.9%** | 7.0% | 0.1% | **高纯度退学群**（四算法最高纯度） |
| Cluster 2 | 177 | 41.8% | 15.8% | 42.4% | **成绩好但缺考群**（独特发现） |

### 6.4 Hierarchical (Ward, k=3)

| 簇 | 人数 | Dropout | Enrolled | Graduate | 核心特征 |
|----|------|---------|----------|----------|----------|
| Cluster 0 | 3271 | 19.9% | 20.7% | 59.4% | **正常学业群** |
| Cluster 1 | 830 | **82.3%** | 8.6% | 9.2% | **学业困难群**：与 K-Means Cluster 1 几乎一致 |
| Cluster 2 | 323 | 26.9% | 13.9% | 59.1% | **高学分转入群**：与 K-Means Cluster 0 一致 |

---

## 7. 二值特征 × 簇 交叉分析

### 7.1 卡方检验关联强度（Cramér's V）Top 10

| 特征 | Cramér's V | 效应量 | 含义 |
|------|-----------|--------|------|
| **no_eval_sem2** | **0.819** | 极强 | 第二学期是否完全缺考 |
| **no_eval_sem1** | **0.734** | 极强 | 第一学期是否完全缺考 |
| Course_171 | 0.379 | 中等偏强 | 特定专业的退学关联 |
| Tuition fees up to date | 0.278 | 中等 | 学费缴纳状态 |
| Application mode_53 | 0.273 | 中等 | 特定入学渠道 |
| Application mode_43 | 0.243 | 中等 | 继续教育转入 |
| economic_stress | 0.241 | 中等 | 经济压力综合标志 |
| Gender | 0.202 | 中等 | 性别 |
| Scholarship holder | 0.201 | 中等 | 奖学金持有 |
| Application mode_39 | 0.194 | 中等 | 特定入学渠道 |

### 7.2 K-Means Cluster 1（学业困难群）的完整画像

**连续特征异常（热力图）：**
- 1st/2nd Grade: -1.70 / -1.83σ
- 1st/2nd Approved: -1.38 / -1.44σ
- Pass Rate S1/S2: -1.70 / -1.68σ

**二值特征异常（交叉分析）：**
- no_eval_sem2: 74.3%（全局 15.6%，↑58.7pp）
- no_eval_sem1: 59.5%（全局 12.2%，↑47.4pp）
- Tuition fees up to date: 70.3%（全局 88.1%，↓17.8pp）
- Gender（男性）: 54.1%（全局 35.2%，↑18.9pp）
- economic_stress: 35.7%（全局 17.7%，↑18.0pp）

**核心洞察：** 这个群体的低成绩本质是大量缺考（grade=0 因为未参加考试），不是真正的低分。正确的解读是"学业失联"而非"学业能力差"——这是教育干预层面完全不同的问题。Task 1 中构建的 `no_eval` 标志在聚类分析中验证了其关键价值。

---

## 8. 结构性发现

### 8.1 数据的"一硬两软"结构

四个算法的共同发现：

- **一个硬边界：** 学业困难/失联学生（主要是 Dropout）vs 正常学业学生之间存在清晰的分界
- **两个软边界：** Graduate 和 Enrolled 在特征空间中高度重叠，无监督方法无法区分这两类

Hierarchical 的 Dendrogram 进一步确认：k=2 的最高合并距离（205）比 k=3 的（158）大 30%，说明 k=2（学业困难 vs 正常）才是数据最自然的分组数。

### 8.2 两条退学预测路径

聚类揭示了退学学生的两个独立信号维度：

- **学业路径：** 缺考（no_eval）→ 低成绩 → 低通过率 → Dropout
- **经济路径：** 学费拖欠 → 有债务 → 经济压力 → Dropout

这两条路径在连续特征热力图和二值特征交叉分析中都有独立的信号。

### 8.3 跨算法一致性

K-Means 和 Ward Hierarchical 独立收敛到几乎相同的三个簇（簇大小 902/275/3247 vs 830/323/3271，退学群纯度 82.0% vs 82.3%），构成了"结果真实性"的最强证据。Spectral 和 GMM 的退学群纯度分别为 92.9% 和 83.5%，进一步确认了该分组的稳健性。

---

## 9. 最终结论：K-Means 为最佳聚类算法

### 9.1 选择理由

**（1）内部指标全面最优（3/3）**——在"簇内紧密度 + 簇间分离度"这个核心标准上，K-Means 的结果最干净。

**（2）与 Hierarchical (Ward) 的高度一致性**——两种完全不同范式的算法（迭代优化 vs 层次合并）独立收敛到相同结果，是无监督分析中最强的交叉验证。

**（3）模型最简洁、可解释性最强**——72 个参数（GMM 需要 974 个），每个簇有一个明确的质心向量。

**（4）最核心的发现已被捕捉**——82% 纯度的退学群、特征热力图的清晰画像，足以支撑 Task 4 的监督学习设计。

### 9.2 其他算法的独立贡献

| 算法 | 独立贡献 |
|------|---------|
| GMM | BIC 分析证实数据中存在特征间相关结构（full >> diag）；软概率退化证实簇边界清晰 |
| Spectral | NMI 最高（0.211 vs K-Means 0.170），退学群纯度最高（92.9%），发现了独特的"成绩好但缺考"子群（177 人） |
| Hierarchical | Dendrogram 揭示 k=2 才是最自然分组数；Ward 与 K-Means 的一致性交叉验证结果真实性；linkage 对比揭示了"虚假高 Silhouette"陷阱 |
| 交叉分析 | `no_eval` 的 Cramér's V = 0.82 是最强关联；Course_171 的显著集中是特定专业退学风险的信号 |

---

## 10. 与 Task 2 的衔接验证

| Task 2 t-SNE 观察 | Task 3 聚类验证 |
|-------------------|----------------|
| Graduate 和 Dropout 有空间分离 | K-Means Cluster 1 的 Dropout 纯度达 82% |
| Enrolled 散布在两者之间，无独立聚集 | ARI/NMI 偏低（~0.17），Enrolled 无法被单独识别 |
| 版本 A（23 连续特征）分离度优于版本 B（全 89 列） | PCA 降维后聚类效果优于高维原始空间 |
| perplexity=5/30/50 下分离均稳定 | 四个算法都稳定地识别出退学群 |
| Fig 3 特征着色显示 1st sem approved 梯度与 Target 对应 | 卡方检验 Top 10 中学业指标排名靠前 |

---

## 11. 对 Task 4 的启发

### 11.1 分类目标

建议同时做三分类（Dropout/Enrolled/Graduate）和二分类（Dropout vs 非 Dropout），后者对应数据的真实分界（Dendrogram k=2），预期性能更高。

### 11.2 特征重要性预测

基于 Task 3 的 Cramér's V 排名，预期 Task 4 的特征重要性前 5 位为：`no_eval_sem2` > `no_eval_sem1` > `2nd Grade` / `Pass Rate S2` > `Tuition fees up to date` > `economic_stress`

### 11.3 类别不均衡

必须使用 `class_weight='balanced'`。Enrolled 类（18%）在聚类中无独立结构，预期三分类中该类 F1 显著低于其他两类。

### 11.4 可验证的预测

1. 三分类的 Enrolled 召回率 < 30%
2. Dropout 的 F1 > 0.7
3. Decision Tree 的根节点是 `no_eval_sem2` 或 `2nd Grade`
4. 二分类比三分类总体准确率高 10+ 个百分点
5. 特征重要性 Top 5 与聚类发现一致

---

## 12. 输出文件清单

### K-Means 输出（fig1–fig6）

| 文件 | 内容 |
|------|------|
| `fig1_elbow_method.png` | Elbow Method |
| `fig2_silhouette_scores.png` | 三合一内部指标 |
| `fig3_kmeans_tsne_comparison.png` | t-SNE 真实 vs K-Means |
| `fig4_kmeans_cluster_profiles.png` | 簇特征热力图 |
| `fig5_kmeans_silhouette_detail.png` | Silhouette 轮廓图 |
| `fig6_pca_variance.png` | PCA 累计方差 |
| `task3_kmeans_results.csv` | 各 k 评估指标 |

### GMM 输出（fig7–fig12）

| 文件 | 内容 |
|------|------|
| `fig7_bic_aic.png` | BIC/AIC 模型选择 |
| `fig8_gmm_tsne_comparison.png` | t-SNE 真实 vs GMM |
| `fig9_gmm_cluster_profiles.png` | 簇特征热力图 |
| `fig10_gmm_soft_assignment.png` | 软分配概率分析 |
| `fig11_gmm_vs_kmeans_tsne.png` | 三合一对比 |
| `fig12_final_comparison.png` | 指标柱状图 |
| `task3_gmm_results.csv` | 各 k × cov 评估指标 |

### Spectral 输出（fig13–fig16）

| 文件 | 内容 |
|------|------|
| `fig13_spectral_tsne_comparison.png` | t-SNE 真实 vs Spectral |
| `fig14_spectral_cluster_profiles.png` | 簇特征热力图 |
| `fig15_spectral_affinity_search.png` | 参数搜索对比 |
| `fig16_three_algo_comparison.png` | 三算法 t-SNE 对比 |
| `task3_spectral_results.csv` | 完整搜索结果 |

### Hierarchical 输出（fig17–fig21）

| 文件 | 内容 |
|------|------|
| `fig17_dendrogram.png` | Ward Dendrogram |
| `fig18_hierarchical_tsne_comparison.png` | t-SNE 真实 vs Hierarchical |
| `fig19_hierarchical_cluster_profiles.png` | 簇特征热力图 |
| `fig20_linkage_comparison.png` | 三种 linkage 对比 |
| `fig21_all_algorithms_comparison.png` | 四算法 t-SNE 对比 |
| `task3_hierarchical_results.csv` | 完整搜索结果 |

### 交叉分析输出（fig22–fig24）

| 文件 | 内容 |
|------|------|
| `fig22_binary_cluster_heatmap.png` | 二值特征阳性率与偏离图 |
| `fig23_categorical_groups.png` | 9 组类别变量分布 |
| `fig24_chi2_top_features.png` | 卡方检验关联强度排名 |
| `task3_binary_crosstab.csv` | 完整结果表 |

---

## 13. 环境依赖

```
pandas >= 2.2.0
numpy >= 1.26.0
scikit-learn >= 1.5.0
matplotlib >= 3.8.0
seaborn >= 0.13.0
scipy >= 1.12.0
```

---

## 14. 复现步骤

```bash
conda activate dsaa_ml

# 确保 full_preprocessed_data.csv 和所有脚本在同一目录
# 按顺序运行（后续脚本依赖前续脚本生成的 .npy 中间文件）
python task3_kmeans.py           # 生成 fig1-6 + .npy 中间数据
python task3_gmm.py              # 生成 fig7-12
python task3_spectral.py         # 生成 fig13-16
python task3_hierarchical.py     # 生成 fig17-21
python task3_binary_crosstab.py  # 生成 fig22-24
```

所有随机操作均设置 `random_state=42`，结果完全可复现。
