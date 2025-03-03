# Titanic 比赛概述

本文件简要介绍 Kaggle 的 Titanic 比赛以及本次实现的主要步骤。

## 比赛背景
- **目标**：预测泰坦尼克号乘客是否生存。
- **数据**：包含乘客的姓名、性别、年龄、客舱等级、票价等信息。
- **评价指标**：准确率或 ROC-AUC，具体依据提交结果而定。

## 数据预处理
- 填充缺失值（如 Age、Fare、Embarked）。
- 将类别数据转换为数值数据（如性别映射、独热编码）。
- 删除无用字段（如 Name、Ticket、Cabin、PassengerId）。

## 模型构建与训练
- **框架**：使用 PyTorch 构建神经网络。
- **模型结构**：
  - 三层全连接层，
  - 隐藏层采用 ReLU 激活函数，
  - 输出层使用 Sigmoid 函数预测生存概率。
- **损失函数**：二元交叉熵损失（BCELoss）。
- **优化器**：Adam 算法。

## 实现流程
1. 加载并预处理训练和测试数据。
2. 对训练数据进行数据增强（上采样少数类别）。
3. 使用 StandardScaler 对数据进行标准化，确保训练和测试数据特征一致。
4. 划分训练集和验证集，训练模型并实时评估。
5. 在测试集上进行预测，生成 submission.csv 文件提交至 Kaggle。

## 如何运行
1. 确保 `train.csv` 和 `test.csv` 位于相应目录下（例如与 `TitanicPyTorch_new.py` 同目录）。
2. 安装必要依赖：numpy、pandas、torch、scikit-learn。
3. 运行脚本：
   ```
   python TitanicPyTorch_new.py
   ```
4. 检查生成的 `submission.csv` 文件，上传至 Kaggle 提交结果。

## 相关链接
- [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic/overview)
- [PyTorch Documentation](https://pytorch.org/)

