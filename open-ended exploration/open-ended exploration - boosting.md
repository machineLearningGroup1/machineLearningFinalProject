# 1. 研究动机与背景 (Motivation)

在 Mandatory Task 部分，我们已经训练并评估了四种基础机器学习模型：逻辑回归 (Logistic Regression)、支持向量机 (SVM)、随机森林 (Random Forest) 和决策树 (Decision Tree)。根据我们参考的相关文献（_Early prediction of student’s performance in higher education: A case study_），研究表明在处理高等教育学生辍学预测（特别是在处理类别不平衡的多分类任务时），Boosting 系列模型（如 Extreme Gradient Boosting 和 CatBoost）通常能够超越传统的机器学习方法，展现出更好的分类性能和泛化能力。

为了验证这一假设，并探索模型性能的上限（特别是对少数类 Dropout 和 Enrolled 学生的识别率）的可能性，我们在 Open-ended Exploration 环节引入了业界领先的两种 Boosting 框架：**XGBoost** 和 **CatBoost**，并基于测试集结果对这 6 种模型进行了严格的交叉对比与客观评估。
  
--------------------------------------------------------------------------------

# 2. Boosting 模型的构建与优化 (Implementation & Tuning)

针对 XGBoost 和 CatBoost，我们进行了专门的参数寻优与不平衡处理：

## **2.1 类别不平衡处理 (Class Imbalance Handling)**

- **XGBoost:** 在二分类中计算了严格的负正样本比例赋给 `scale_pos_weight`；在三分类中，通过 `compute_sample_weight` 计算了平衡的样本权重传入训练。
- **CatBoost:** 在初始化模型时，统一设置了 `auto_class_weights='Balanced'` 参数，使其在底层自动调整少数类梯度。

## **2.2 GridSearchCV 超参数调优** 
我们针对 F1-Score 指标进行了 5 折交叉验证：

- **XGBoost 最佳超参数:**
    - 二分类: `{'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 100, 'subsample': 1.0}`。
    - 三分类: `{'gamma': 0.1, 'learning_rate': 0.05, 'max_depth': 4, 'n_estimators': 300}`。
- - **CatBoost 最佳超参数:**
    - 二分类: `{'depth': 6, 'iterations': 200, 'l2_leaf_reg': 3, 'learning_rate': 0.05}`。
    - 三分类: `{'bootstrap_type': 'Bayesian', 'depth': 6, 'iterations': 500, 'learning_rate': 0.05}`。

--------------------------------------------------------------------------------

# 3. 核心评估：Boosting 模型真的更好吗？(Critical Evaluation)

基于测试集（Test Dataset）的详细评估结果，我们发现 Boosting 模型的优势是**具有场景局限性的**。

## 3.1 二分类任务 (Dropout vs Non-Dropout) 

在二分类任务中，Boosting 并没有展现出对传统模型的降维打击：

- **准确率 (Accuracy):** LR, SVM, Random Forest, XGBoost 和 CatBoost 的准确率惊人地一致，**全部为 0.86**。
- **辍学群体召回率 (Dropout Recall):** 我们的核心业务目标是“尽可能不漏掉任何一个潜在辍学生”。令人意外的是，**逻辑回归 (LR) 达到了全场最高的 0.82**，略高于 XGBoost 和 CatBoost 的 0.81。
- **结论:** 优秀的特征工程（详见 Data Preprocessing）使得数据的二分类边界相对清晰，简单的线性模型（LR）在此任务上展现了极高的性价比和鲁棒性，Boosting 模型在此表现出了边际收益递减。

## 3.2 三分类任务 (Dropout / Enrolled / Graduate) 

三分类任务（特别是难以界定的 Enrolled 类别）让模型间的差异显现：

- **最高准确率与辍学识别:** **XGBoost 展现了其强大的拟合能力，拿下了最高的整体准确率 (0.77)**，并实现了**最高的三分类 Dropout Recall (0.76)**，优于随机森林 (0.72) 和 LR (0.71)。
- **类别均衡度上的妥协:** 尽管 XGBoost 整体准确率最高，但它牺牲了对中间少数类 (Enrolled) 的识别，其 Enrolled F1 仅为 0.47。相比之下，**随机森林 (0.52) 和 CatBoost (0.51) 在应对 Enrolled 类别时表现更好**，获得了全场最高的 Macro F1 (0.71)。
- **结论:** 对于三分类，如果业务目标是最大化整体准确率和抓取辍学群体，XGBoost 是最优解；如果目标是保证所有三个群体都能被相对公平地识别，Random Forest 或 CatBoost 则是更均衡的选择。

--------------------------------------------------------------------------------


# 4. 模型评估与综合对比 (Performance Evaluation & Comparison)

基于测试集（Test Dataset）的综合评估结果，我们将 XGBoost 和 CatBoost 与 Mandatory 环节的基础模型进行了对比：

## 4.1 二分类任务表现 (Binary Classification)

- **总体准确率 (Accuracy):** XGBoost, CatBoost, LR, SVM, Random Forest 均达到了 0.86，高于决策树的 0.85。
- **Dropout 类别的 F1-Score:** CatBoost 与 LR、SVM、Random Forest 持平，达到 0.79；XGBoost 为 0.78。
- **Dropout 类别的召回率 (Recall):** XGBoost 和 CatBoost 均达到了 0.81，超越了 Random Forest (0.79) 和决策树 (0.74)，这意味着 Boosting 模型在“尽可能不漏掉任何一个可能退学的学生”这一核心目标上表现出色。

## 4.2 三分类任务表现 (Multi-class Classification)

三分类任务难度显著增加，主要体现在对中间类别 (Enrolled) 的识别上。

- **总体准确率 (Accuracy):** XGBoost 达到了 0.77（在所有模型中最高），Random Forest 为 0.76，CatBoost 为 0.75。
- **Dropout 识别能力:** XGBoost 和 CatBoost 对 Dropout 的 F1-Score 均为 0.77，略低于 Random Forest (0.77, recall偏低) 但高于 LR (0.76) 和 SVM (0.76)。
- **宏平均 F1-Score (Macro F1):** CatBoost 为 0.71，XGBoost 为 0.70，与 Random Forest (0.71) 表现相近，显著优于 Decision Tree (0.65)。

--------------------------------------------------------------------------------

# 5. 特征重要性分析 (Feature Importance Analysis)

Boosting 树模型天然支持输出特征重要性。通过对比两种模型的 Feature Importance，我们提炼出了对预测学生是否辍学最具影响力的关键指标：

- **CatBoost 视角:** 第二学期课程通过率 (`pass_rate_sem2`) 和第一学期课程通过率 (`pass_rate_sem1`) 是压倒性最重要的特征，其次是入学成绩 (`Admission grade`)。
- **XGBoost 视角:** 入学成绩 (`Admission grade`) 和前置学历分数 (`Previous qualification (grade)`) 最为重要，其次是成绩变化趋势 (`grade_trend`) 和第二学期课程通过率 (`pass_rate_sem2`)。

**结论:** 无论是 XGBoost 还是 CatBoost，预处理阶段我们自行构建的派生特征（如 `pass_rate_sem2`, `grade_trend`）均在预测中占据了核心主导地位，证明了特征工程在此项目中的巨大价值。

--------------------------------------------------------------------------------

# 6. 探索总结 (Conclusion)

本次 Open-ended Exploration 证明，虽然 Boosting 模型（XGBoost, CatBoost）在处理高难度、多边界的复杂分类任务（三分类）时能提供目前最高的预测上限（Accuracy=0.77），但它们并非不可战胜的。在清晰、处理得当的简单数据分布（二分类）下，轻量级的传统模型（如 Logistic Regression）不仅能达到相同的准确度，甚至能更好地满足特定的业务诉求（如 Recall）。