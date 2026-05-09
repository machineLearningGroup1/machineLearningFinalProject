# Open-ended Exploration — Feature Importance & Feature Engineering Analysis

**数据集:** Predict Students' Dropout and Academic Success (UCI ID: 697)  
**脚本:** `open_ended_feature_importance.py` / `open_ended_ablation_v2.py`  
**输入:** `train_70.csv` / `val_10.csv` / `test_20.csv` / `full_preprocessed_data.csv`  
**输出:** 7 张可视化图表 + 2 份 CSV 数据文件

---

## 1. 研究动机：从 Mandatory Tasks 的未解问题出发

在完成 Task 1–5 的过程中，我们积累了多个层面的特征分析结果，但它们各自独立、从未被系统性地整合：

**Task 3（聚类）的发现：**
- K-Means 的簇特征热力图显示，连续特征（如 `1st/2nd sem grade`、`pass_rate`）的标准化均值偏差高达 ±1.7σ ~ ±3.4σ，是驱动簇结构的核心力量
- 二值特征交叉分析（Cramér's V）发现 `no_eval_sem2`（V = 0.819）和 `no_eval_sem1`（V = 0.734）是与簇标签关联最强的二值特征
- 四个算法（K-Means / GMM / Spectral / Hierarchical）的热力图呈现出高度一致的模式

**Task 4/5（监督学习）的发现：**
- 6 个模型（LR / SVM / Decision Tree / Random Forest / XGBoost / CatBoost）各自输出了特征重要性，但从未与聚类结果交叉对比
- 所有模型的 Enrolled F1 ≤ 0.52，验证了 Task 2/3 关于 Graduate/Enrolled 不可分的预言
- 特征工程的实际贡献从未被量化

**本次探索旨在回答三个核心问题：**

1. **跨模型一致性：** 三个树模型（RF / XGBoost / CatBoost）对特征重要性的排名是否一致？
2. **跨任务交叉验证：** 无监督聚类（Task 3）和监督学习（Task 4）对特征重要性的判断是否一致？如果不一致，原因是什么？
3. **特征工程价值量化：** Task 1 中设计的 8 个工程特征以及 89 个特征中的各个功能族群，各自对预测性能贡献多少？

---

## 2. 分析流程总览

```
Part A — 跨模型特征重要性分析 (open_ended_feature_importance.py)
    │
    ├─ Step 1  加载 train_70 / val_10 / test_20 数据
    ├─ Step 2  使用 Task 4/5 的最佳超参数重新训练 RF / XGBoost / CatBoost
    ├─ Step 3  提取三模型的特征重要性（RF: Gini; XGBoost: F-score; CatBoost: PredictionValuesChange）
    ├─ Step 4  图1 — 三模型 Top 15 特征柱状图（标注工程特征）
    ├─ Step 5  图2 — 四源排名热力图（RF / XGBoost / CatBoost / Cramér's V）
    ├─ Step 6  图3 — 全量 89 特征散点图 + 哑铃图（聚类 vs 监督排名对比）
    └─ 输出    feature_importance_table.csv（89×3 完整重要性矩阵）

Part B — 功能族群消融实验 (open_ended_ablation_v2.py)
    │
    ├─ Step 1  将 89 个特征按语义功能分为 8 个族群
    ├─ Step 2  基线模型（全 89 特征 RF）
    ├─ Step 3  逐组移除实验（8 组 × 1 次训练+评估）
    ├─ Step 4  充分性测试（"仅用学业特征" / "仅用入学背景"）
    ├─ Step 5  图4 — 族群消融瀑布图
    ├─ Step 6  图5 — 分类别影响拆解（Dropout / Enrolled / Graduate）
    ├─ Step 7  图6 — "仅用 X 能走多远" 充分性对比
    └─ 输出    ablation_results.csv
```

---

## 3. Part A：跨模型特征重要性分析

### 3.1 模型训练配置

三个模型均使用 Task 4/5 阶段通过网格搜索确定的最佳超参数，在 `train_70`（3,096 条）上训练，在 `val_10 + test_20`（1,328 条）上评估：

| 模型 | 关键超参数 | Eval weighted F1 |
|------|----------|-----------------|
| Random Forest | `n_estimators=300, max_features=sqrt, class_weight=balanced` | 0.7627 |
| XGBoost | `max_depth=4, lr=0.05, n_estimators=300, gamma=0.1, sample_weight=balanced` | 0.7531 |
| CatBoost | `depth=6, iterations=500, lr=0.05, auto_class_weights=Balanced` | 0.7598 |

### 3.2 图 1 — 三模型 Top 15 特征重要性对比

三个模型的 Top 10 特征如下（★ 表示 Task 1 工程特征）：

**Random Forest（Gini-based）：**

| 排名 | 特征 | 重要性 |
|------|------|--------|
| 1 | **pass_rate_sem2** ★ | 0.0880 |
| 2 | Curricular units 2nd sem (approved) | 0.0689 |
| 3 | Curricular units 2nd sem (grade) | 0.0623 |
| 4 | **pass_rate_sem1** ★ | 0.0606 |
| 5 | Curricular units 1st sem (grade) | 0.0484 |
| 6 | Admission grade | 0.0408 |
| 7 | Curricular units 1st sem (approved) | 0.0392 |
| 8 | Previous qualification (grade) | 0.0369 |
| 9 | Curricular units 2nd sem (evaluations) | 0.0353 |
| 10 | **grade_trend** ★ | 0.0316 |

**XGBoost（F-score / weight）：**

| 排名 | 特征 | 重要性 |
|------|------|--------|
| 1 | Admission grade | 0.0754 |
| 2 | Previous qualification (grade) | 0.0602 |
| 3 | **pass_rate_sem2** ★ | 0.0506 |
| 4 | **grade_trend** ★ | 0.0504 |
| 5 | Curricular units 2nd sem (grade) | 0.0476 |
| 6 | Curricular units 1st sem (grade) | 0.0454 |
| 7 | **pass_rate_sem1** ★ | 0.0433 |
| 8 | Curricular units 2nd sem (evaluations) | 0.0394 |
| 9 | GDP | 0.0372 |
| 10 | Unemployment rate | 0.0342 |

**CatBoost（PredictionValuesChange）：**

| 排名 | 特征 | 重要性 |
|------|------|--------|
| 1 | **pass_rate_sem2** ★ | 0.1385 |
| 2 | **pass_rate_sem1** ★ | 0.0694 |
| 3 | Admission grade | 0.0563 |
| 4 | Previous qualification (grade) | 0.0486 |
| 5 | Curricular units 2nd sem (grade) | 0.0420 |
| 6 | GDP | 0.0380 |
| 7 | Curricular units 1st sem (grade) | 0.0378 |
| 8 | **grade_trend** ★ | 0.0356 |
| 9 | Curricular units 2nd sem (evaluations) | 0.0351 |
| 10 | Tuition fees up to date | 0.0350 |

**图 1 关键发现：**

1. **`pass_rate_sem2` 是三模型中最稳定的顶级特征**：在 RF 和 CatBoost 中均排第 1，在 XGBoost 中排第 3。CatBoost 对它的依赖度尤其突出（重要性 0.139 是第 2 名的两倍）。
2. **工程特征在每个模型的 Top 10 中至少占 3 个席位**（`pass_rate_sem1/2` + `grade_trend`），证明 Task 1 特征工程的设计价值。
3. **XGBoost 更偏好入学背景特征**（Admission grade 排第 1），而 RF 和 CatBoost 更偏好学业过程特征（pass_rate 排第 1），反映了不同算法对特征利用方式的差异。

图中使用**深蓝色标注工程特征**、浅色标注原始特征，视觉上直接可见工程特征的贡献密度。

### 3.3 图 2 — 四源排名热力图

将三个监督模型的 Top 15 排名与 Task 3 Cramér's V 排名（二值特征）放入同一张热力图，观察跨方法的排名一致性与分歧。

**热力图结构：**
- 行：4 个来源（Random Forest / XGBoost / CatBoost / Cramér's V）
- 列：出现在任一来源 Top 15 中的所有特征（取并集）
- 颜色：排名越高（数字越小）颜色越深；"–" 表示该特征不在该来源的 Top 15 中

**图 2 关键发现：**

热力图揭示了一个显著的**"分裂模式"**——三个监督模型之间的排名高度一致（`pass_rate_sem2` / `Admission grade` / `S2 grade` 始终位居前列），但 Cramér's V 的排名与它们几乎不重叠。Cramér's V 的 Top 2（`no_eval_sem2` 排第 1、`no_eval_sem1` 排第 2）在三个监督模型中仅排 13–15 位甚至更靠后。

**但这并不矛盾。** Cramér's V 只衡量**二值特征**与簇标签的关联，无法覆盖连续特征。这个局限性促使我们在图 3 中设计了更完整的对比方法。

### 3.4 图 3 — 全量 89 特征：聚类区分力 vs 监督重要性

**方法论改进：** 图 2 中 Cramér's V 只覆盖了 66 个二值特征，遗漏了 Task 3 四张聚类热力图中那些标准化均值偏差极大（±1.7σ ~ ±3.4σ）的连续特征。图 3 将两类特征统一纳入：

- **连续特征（23 个）**：在 K-Means 聚类结果上计算各簇的标准化均值，取 **max|z-score 偏差|** 作为"聚类区分力"——即热力图中最深红/深蓝格子的绝对值
- **二值特征（66 个）**：使用 Cramér's V
- 两者归一化至 [0, 1] 后统一比较

**左面板（散点图）：** x 轴 = 聚类区分力，y 轴 = 三模型平均监督重要性

统计结果：**Pearson r = 0.328（p = 0.002），Spearman ρ = 0.413**——两种方法之间存在**显著的正相关**，但远非完美一致。

**右面板（哑铃图）：** 对监督排名前 20 的特征，并排显示其监督排名（●）和聚类排名（◆），连线越长表示两种方法的分歧越大。

#### 3.4.1 两种方法都认为重要的特征（9 个，两种排名均在 Top 15 内）

| 特征 | 监督排名 | 聚类排名 | 类型 |
|------|---------|---------|------|
| pass_rate_sem2 ★ | 1 | 13 | 工程 |
| pass_rate_sem1 ★ | 2 | 12 | 工程 |
| Curricular units 2nd sem (grade) | 4 | 10 | 原始 |
| Curricular units 1st sem (grade) | 6 | 11 | 原始 |
| Curricular units 2nd sem (approved) | 7 | 9 | 原始 |
| Curricular units 2nd sem (evaluations) | 9 | 14 | 原始 |
| Curricular units 1st sem (evaluations) | 13 | 8 | 原始 |
| Curricular units 1st sem (approved) | 14 | 7 | 原始 |
| no_eval_sem2 ★ | 15 | 2 | 工程 |

**共同点：** 这 9 个特征全部是**学业表现类**——两学期的成绩、通过率、考试参与情况。无论有没有标签，这些特征都能被识别为数据中最强的信号。

#### 3.4.2 聚类看重但监督学习不在意的特征

| 特征 | 聚类排名 | 监督排名 | 原因分析 |
|------|---------|---------|---------|
| Curricular units 1st sem (credited) | 1 | 29 | 在 K-Means Cluster 0 中 z = +3.40（热力图中最深红格），定义了"高学分转入群"（275 人），但该群 63.6% 是 Graduate，退学率不高，对预测无直接价值 |
| Curricular units 2nd sem (credited) | 3 | 33 | 同上，z = +3.26 |
| no_eval_sem1 ★ | 4 | 53 | 聚类中区分力极强（V = 0.734），但监督模型中其信息已被 `pass_rate` 和 `grade` 完全吸收——如果 pass_rate = 0，模型已知该生"什么都没通过"，无需额外的 no_eval 标志 |
| Curricular units 1st/2nd sem (enrolled) | 5, 6 | 21, 22 | 注册门数在聚类中驱动簇形状，但预测退学时不如通过率直接 |
| Course_171 | 15 | 62 | 特定专业在聚类中集中，但监督模型有更多全局信号 |

**核心洞察：** 聚类找的是"数据中最大的几何结构差异"——`credited` 列的巨大 z-score 偏差对 K-Means 来说是压倒性的分离信号（Voronoi 划分直接被它拉开），但这个结构并不直接对应退学风险。

#### 3.4.3 监督学习看重但聚类不在意的特征

| 特征 | 监督排名 | 聚类排名 | 原因分析 |
|------|---------|---------|---------|
| Admission grade | 3 | 56 | 入学成绩在各簇之间几乎无差异（热力图中 z ≈ 0），但对区分 Graduate vs Enrolled 的决策边界非常有效 |
| Previous qualification (grade) | 5 | 85 | 同上——入学前的学术背景在聚类中完全不可见（聚类由学业表现主导），但在监督学习中提供独立预测信号 |
| grade_trend ★ | 8 | 89（最后！）| 作为两学期成绩的差值，在聚类中完全无区分力（所有簇的 grade_trend 均 ≈ 0），但树模型的分裂决策中它被频繁选用 |
| GDP / Unemployment rate | 10, 11 | 74, 75 | 宏观经济变量对簇结构无贡献，但在监督中有非零预测力 |

**核心洞察：** 聚类被**学业表现的绝对值**主导——谁通过了课、谁成绩高。但监督学习还需要区分 **Graduate 和 Enrolled 这两个高度重叠的类**（Task 2 t-SNE 和 Task 3 聚类均已确认它们几何不可分），此时入学前的背景信息（`Admission grade`、`Prev qual grade`）就变得关键——它们提供了学业表现以外的**独立信号维度**。

#### 3.4.4 图 3 的理论意义

两种方法**互补而非冗余**：
- **聚类** 揭示数据的**几何结构**（"学业失联群" vs "正常群"的硬边界），对应 Task 3 发现的 k=2 自然分组
- **监督学习** 在此基础上额外利用**入学前背景信息**来处理聚类无法解决的问题——区分 Graduate 和 Enrolled

这解释了为什么 Task 3 的 ARI/NMI 较低（~0.17）：聚类只能捕捉数据的第一主结构（学业失联 vs 正常），而三分类预测还需要第二层信号（入学背景）。

---

## 4. Part B：功能族群消融实验

### 4.1 实验动机

Part A 的图 3 揭示了特征按功能角色分为不同族群（学业表现 / 入学背景 / 经济因素 / 宏观经济 / 人口统计 / 课程 one-hot 等），且各族群在聚类和监督中的角色截然不同。自然引出下一个问题：**每个功能族群对预测性能的实际贡献有多大？**

与 v1 消融实验（仅测试"工程特征 vs 原始特征"）不同，v2 按**语义功能**将 89 个特征分为 8 个族群，直接回应图 3 的分析发现。

### 4.2 特征族群定义

| 族群 | 特征数 | 包含的特征 | 动机 |
|------|--------|----------|------|
| **学业表现** | 16 | S1/S2 的 grade / approved / evaluations / enrolled / credited / without_evaluations + pass_rate_sem1/2 + no_eval_sem1/2 | 图 3 中聚类与监督共同认可的核心信号 |
| **入学背景** | 4 | Admission grade / Previous qualification (grade) / prev_qual_ordinal / Application order | 图 3 中"只有监督看重、聚类看不到"的特征 |
| **经济因素** | 4 | Tuition fees up to date / Debtor / Scholarship holder / economic_stress | Task 3 聚类揭示的"第二条退学路径" |
| **宏观经济** | 3 | GDP / Unemployment rate / Inflation rate | 外部环境变量 |
| **家庭背景** | 17 | 父母学历/职业分组（one-hot）+ family_edu_capital | 社会资本指标 |
| **人口统计** | 14 | Gender / age_group / Nacionality / Marital status / Displaced / Educational special needs | 人口学变量 |
| **课程 & 入学渠道** | 30 | Course_ 系列（16 列）+ Application mode_ 系列（13 列）+ Daytime attendance（1 列） | 类别 one-hot 编码 |
| **grade_trend** | 1 | grade_trend | v1 中发现的疑似噪声特征，单独检验 |

**覆盖验证：** 8 个族群合计 89 个特征，无遗漏，无重叠。

### 4.3 实验设计

**基础模型：** Random Forest（与 Task 4 最佳参数一致），在 `train_70` 上训练，在 `val_10 + test_20` 上评估。选择 RF 而非 XGBoost 的原因：RF 训练速度快、OOB 评估稳健、结果方差较低，适合需要多次重复训练的消融实验。

**实验类型：**

1. **逐组移除（Removal ablation）：** 每次移除一个族群，观察 weighted F1 的变化（Δ）。负值 = 移除后性能下降 = 该族群有正贡献。
2. **充分性测试（Sufficiency test）：** 仅保留某一个族群，观察它单独能达到多高的性能。

### 4.4 图 4 — 功能族群消融结果

| 移除的族群 | 特征数 | Weighted F1 | ΔwF1 | 性质 |
|-----------|--------|------------|------|------|
| 基线（全 89 特征） | 89 | 0.7627 | — | — |
| – 学业表现 | 16 | 0.6314 | **-0.1313** | 压倒性核心 |
| – 经济因素 | 4 | 0.7447 | **-0.0180** | 第二重要 |
| – 宏观经济 | 3 | 0.7552 | -0.0075 | 小但稳定 |
| – 家庭背景 | 17 | 0.7601 | -0.0026 | 微弱贡献 |
| – 人口统计 | 14 | 0.7627 | +0.0000 | 零贡献 |
| – 课程 & 入学渠道 | 30 | 0.7672 | **+0.0045** | 噪声（移除后提升） |
| – grade_trend | 1 | 0.7683 | **+0.0056** | 噪声（移除后提升） |
| – 入学背景 | 4 | 0.7691 | **+0.0064** | 冗余（移除后提升） |

**核心发现 1 — 极度不均衡的贡献结构：**

学业表现族群（16 个特征）的影响力（ΔwF1 = -0.1313）是第二名经济因素（-0.0180）的 **7.3 倍**。89 个特征中，只有 23 个（学业 + 经济 + 宏观）有正贡献，其余 66 个特征要么是零贡献、要么是噪声。

**核心发现 2 — 三类"负贡献"特征：**

- **grade_trend（+0.0056）：** 作为 `2nd sem grade - 1st sem grade` 的差值，在原始 grade 特征同时存在的情况下完全冗余，移除后模型减少过拟合。这是 Task 1 特征工程中唯一的"失败设计"——值得在报告中诚实讨论。
- **Course & Application mode one-hot（+0.0045）：** 30 个 one-hot 列引入了大量稀疏维度，增加了模型复杂度但未提供有效信号。这与 Task 3 中 PCA 降维的决策一致——PCA 24 维保留 90% 方差后，这些 one-hot 列的信息已被压缩掉。
- **入学背景（+0.0064）：** 虽然 `Admission grade` 在监督模型的特征重要性中排第 3（图 1），但消融结果显示移除它后 F1 反而上升。这说明**特征重要性 ≠ 特征不可替代性**——入学成绩好的学生学业表现通常也好，`Admission grade` 的信息已被学业表现特征间接覆盖，单独保留它反而引入冗余。

**核心发现 3 — 经济因素是真正的"第二信号"：**

4 个经济特征的独立贡献（ΔwF1 = -0.018）远超 17 个家庭背景特征（-0.003）。这与 Task 3 的发现完美对应——聚类揭示的"学业路径"（缺考 → 低成绩 → 退学）和"经济路径"（学费拖欠 → 债务 → 退学）两条退学路径在监督学习中都有独立、可量化的贡献。

### 4.5 图 5 — 分类别影响拆解

对五个关键族群，分别展示移除后对 Dropout / Enrolled / Graduate 三个类别 F1 的影响：

| 移除的族群 | ΔF1 Dropout | ΔF1 Enrolled | ΔF1 Graduate |
|-----------|------------|-------------|-------------|
| 学业表现 | -0.111 | **-0.231** | -0.108 |
| 经济因素 | -0.027 | **-0.040** | -0.004 |
| 宏观经济 | -0.009 | -0.030 | +0.002 |
| 入学背景 | +0.007 | +0.012 | +0.004 |
| grade_trend | +0.000 | +0.010 | +0.008 |

**图 5 关键发现：**

1. **Enrolled 类对学业特征的依赖度最高**：移除学业表现后，Enrolled F1 从 0.439 暴跌到 0.208（-0.231），跌幅是 Dropout（-0.111）的两倍。这说明 Enrolled 本就微弱的识别力几乎完全建立在学业表现特征之上——没有这些特征，模型对 Enrolled 几乎丧失分辨能力。

2. **经济因素对 Enrolled 的影响大于 Dropout**（-0.040 vs -0.027）：说明经济变量更多地帮助区分"还在读但有退学风险"的学生——这些学生学业表现可能尚可，但经济压力使他们处于不稳定状态。

3. **Graduate 类最"鲁棒"**：移除任何非学业族群后 Graduate F1 几乎不变（Δ ≤ 0.004），说明 Graduate 的识别主要依赖学业表现的高绝对值，不需要其他辅助信号。

### 4.6 图 6 — 充分性测试与 Martins et al. (2021) 的间接对比

| 配置 | 特征数 | Weighted F1 | 占全模型性能 |
|------|--------|------------|------------|
| 全模型（89 特征） | 89 | 0.7627 | 100% |
| **仅学业特征** | **16** | **0.7134** | **93.5%** |
| 仅入学背景 | 4 | 0.4638 | 60.8% |
| Martins et al. (2021) 最佳模型 | 25 | avg F1 = 0.65 | （间接参考）|

**与 Martins et al. (2021) 的对比说明：**

Martins et al. (2021) 使用的是同一数据来源（IPP）的**早期版本**（3,623 条 / 25 特征），与我们使用的 UCI 扩展版（4,424 条 / 36 → 89 特征）存在三点关键差异：

1. **标签定义不同：** 论文使用 Success / Relative Success / Failure（按毕业时间划分），我们使用 Graduate / Enrolled / Dropout。两套标签的语义不完全对应——"Enrolled"（在读）≠ "Relative Success"（延迟毕业），"Dropout"（退学）≠ "Failure"（超时或未毕业）。
2. **特征范围不同：** 论文**明确排除了入学后的学业表现数据**（原文 Section 3.1: "no information regarding academic performance after enrolment is used"），而我们的数据集包含了两学期的完整学业成绩。论文在 Section 5 中将"加入第一学年学业成绩"列为 future work。
3. **样本量与预处理不同：** 论文使用 SMOTE 进行过采样，我们使用 `class_weight='balanced'` 进行代价敏感学习。

因此，两项工作的 F1 数值**不可直接同口径对比**。图 6 中 Martins et al. 的 avg F1 = 0.65 仅作为一个**间接参考线**，用于定位我们的性能水平。

**图 6 关键发现：**

尽管两项工作不完全可比，但以下观察仍有意义：

1. **仅用 16 个学业特征（占总特征数的 18%）就能达到全模型 93.5% 的性能**（wF1 = 0.7134），这验证了 Martins et al. 在论文中提出的 future work 假设——加入学业成绩数据确实能显著提升预测能力。
2. **仅用 4 个入学背景特征的模型（wF1 = 0.4638）**与论文的 25 个入学时特征（avg F1 = 0.65）存在差距，这合理反映了我们的入学背景族群只有 4 个特征（论文有 25 个），信息量不足。
3. 剩余 73 个非学业特征合计只贡献了 6.5% 的边际提升，这对实际部署有直接意义——如果学校系统只需要收集两学期的课程成绩数据，就足以建立有效的退学预警系统。

---

## 5. 与 Mandatory Tasks 的衔接总结

### 5.1 与 Task 1（预处理）的衔接

| Task 1 决策 | 本次验证结果 |
|------------|------------|
| 构建 `pass_rate_sem1/2` | 三模型中排名最高的工程特征，RF 和 CatBoost 的 #1 |
| 构建 `no_eval_sem1/2` | 在聚类中极强（V = 0.82），在监督中被 pass_rate 吸收——设计正确但信息冗余 |
| 构建 `grade_trend` | 消融证实为噪声特征（移除后 +0.006），是唯一的"失败设计" |
| 构建 `economic_stress` | 消融证实经济族群独立贡献 -0.018，其中 economic_stress 是核心载体 |
| 构建 `family_edu_capital` | 贡献极弱（整个家庭族群仅 -0.003），投入产出比最低 |
| 构建 `prev_qual_ordinal` | 被 Admission grade 冗余覆盖，移除入学背景后 F1 反而上升 |

### 5.2 与 Task 2（t-SNE）的衔接

| Task 2 观察 | 本次解释 |
|------------|---------|
| 版本 A（23 连续特征）分离度优于版本 B（全 89 列） | 消融证实 66 个非学业特征中多数是噪声/冗余，它们在 t-SNE 距离计算中引入了干扰 |
| Graduate 和 Enrolled 完全重叠 | 监督模型的 Enrolled F1 ≤ 0.52，且消融显示 Enrolled 最依赖学业特征——学业特征无法区分这两类 |

### 5.3 与 Task 3（聚类）的衔接

| Task 3 发现 | 本次验证 |
|------------|---------|
| `no_eval_sem2` Cramér's V = 0.82 | 在聚类区分力中排第 2，但在监督中排第 15——信息被连续特征吸收 |
| 四张热力图中 1st/2nd Credited z-score 最高（±3.4σ） | 聚类区分力排第 1/3，但监督排名 29/33——定义了"高学分转入群"但不直接预测退学 |
| "两条退学路径"（学业 + 经济） | 消融验证：学业 Δ = -0.131，经济 Δ = -0.018，两条路径均有独立贡献 |
| K-Means 为最佳聚类算法 | 图 3 的聚类区分力正是基于 K-Means 聚类结果计算 |

### 5.4 与 Task 4/5（监督学习）的衔接

| Task 4/5 发现 | 本次深化 |
|-------------|---------|
| 6 个模型 Enrolled F1 ≤ 0.52 | 图 5 证实 Enrolled 最依赖学业特征且最脆弱 |
| 二分类 AUC ≈ 0.91，远优于三分类 | 消融解释：聚类只能捕捉一条硬边界，第二条边界需要入学背景信号 |
| XGBoost 三分类最优（0.77） | 图 1 揭示 XGBoost 更偏好入学背景特征，可能有助于捕捉 Graduate/Enrolled 边界 |

---

## 6. 方法论反思

### 6.1 特征重要性 ≠ 特征不可替代性

`Admission grade` 在三个模型的特征重要性中排第 3–6，但消融显示移除它后性能反而微升。这揭示了一个方法论陷阱：**Gini importance / F-score 衡量的是"模型使用该特征的频率"，而非"该特征是否不可替代"**。当多个特征携带相似信息时，模型会频繁使用它们中的每一个（重要性高），但任意去掉一个不会影响性能（不可替代性低）。

### 6.2 聚类区分力 vs 监督预测力的本质差异

聚类是**无目标函数**的——它找的是数据的**几何主结构**，被方差最大的维度主导。监督学习有**明确目标**——它会优先利用对分类边界最有用的特征，哪怕这些特征在全局方差中占比很小（如 `Admission grade`）。两种方法各有盲区：聚类看不到低方差但高预测力的特征，监督学习无法发现数据的自然分组结构。

### 6.3 grade_trend 的教训

`grade_trend = 2nd_grade - 1st_grade` 的设计逻辑是合理的（捕捉学业退步趋势），但在原始 grade 特征同时存在的情况下，树模型可以自行通过两次分裂实现相同的逻辑。手动构造差值特征不仅冗余，还可能因为仅在两学期均有成绩的 3,512 条（79.4%）上有意义而在其余 912 条上填充 0，引入了非均匀噪声。

### 6.4 与 Martins et al. (2021) 基准对比的方法论说明

本项目的参考论文 Martins et al. (2021) 使用了同一数据来源（葡萄牙 Portalegre 理工学院）的早期版本数据集。该论文的核心实验设计是**仅使用入学时可获得的信息**进行预测，不包含任何入学后的学业表现数据。其最佳模型（XGBoost + SMOTE）在三分类任务上达到 Average F1 = 0.65。

我们使用的 UCI 扩展版数据集（ID: 697）在论文数据的基础上**增加了两学期的学业成绩数据**（每学期 6 个指标：enrolled / evaluations / approved / grade / credited / without evaluations），这正是论文 Section 5 中明确提出的 future work 方向。此外，两项工作的标签定义也不相同（论文按毕业时间分为 Success / Relative Success / Failure，我们按最终状态分为 Graduate / Enrolled / Dropout）。

因此，在报告和 Presentation 中引用该论文时，正确的表述应为：

> "Martins et al. (2021) 在仅使用入学时特征的条件下达到了 avg F1 = 0.65。本项目使用了扩展数据集（包含两学期学业成绩），在不同的标签体系下达到了 weighted F1 = 0.76。虽然两者不完全可比，但我们的消融实验表明，仅凭 16 个学业表现特征就能达到 wF1 = 0.71，验证了论文提出的'加入学业成绩数据将提升预测性能'这一假设。"

**不应表述为**："我们超越了 Martins et al. 的基准 X 个百分点"，因为标签定义、特征范围和数据版本均不相同。

---

## 7. 输出文件清单

### Part A — 跨模型特征重要性分析

| 文件名 | 内容 |
|--------|------|
| `fig1_importance_comparison.png` | RF / XGBoost / CatBoost 三格并排 Top 15 特征柱状图（深蓝色 = 工程特征） |
| `fig2_top15_heatmap.png` | 四源排名热力图（RF / XGBoost / CatBoost / Cramér's V）|
| `fig3_revised_cluster_vs_supervised.png` | 左：全量 89 特征散点图（聚类区分力 vs 监督重要性）；右：Top 20 哑铃图（排名对比） |
| `feature_importance_table.csv` | 89 × 3 完整特征重要性矩阵（RF / XGBoost / CatBoost 归一化值） |

### Part B — 功能族群消融实验

| 文件名 | 内容 |
|--------|------|
| `fig4_ablation_by_function.png` | 8 个族群的 ΔwF1 瀑布图（红 = 性能下降，绿 = 移除后性能上升） |
| `fig5_perclass_ablation.png` | 三联图：Dropout / Enrolled / Graduate 各自受移除影响的程度 |
| `fig6_only_group_comparison.png` | "仅用学业特征" vs "仅用入学背景" vs "全模型" 对比 |
| `ablation_results.csv` | 全部实验结果表（含 per-class F1） |

---

## 8. 环境依赖

```
pandas >= 2.2.0
numpy >= 1.26.0
scikit-learn >= 1.5.0
matplotlib >= 3.8.0
scipy >= 1.12.0
xgboost >= 2.0.0
catboost >= 1.2.0
```

---

## 9. 复现步骤

```bash
conda activate dsaa_ml

# 确保预处理输出文件在 ../data_preprocessing_final/result/ 下
ls ../data_preprocessing_final/result/train_70.csv
ls ../data_preprocessing_final/result/val_10.csv
ls ../data_preprocessing_final/result/test_20.csv
ls ../data_preprocessing_final/result/full_preprocessed_data.csv

# Part A — 特征重要性分析
python open_ended_feature_importance.py

# Part B — 消融实验
python open_ended_ablation_v2.py
```

所有随机操作均设置 `random_state=42`，结果完全可复现。
