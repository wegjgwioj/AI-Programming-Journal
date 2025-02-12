import numpy as np 
import pandas as pd 
import os

try:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
except ImportError:
    print("请安装 scikit-learn 库: pip install scikit-learn")
    raise
#  Validation Accuracy: 0.83
# 加载数据
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# 数据预处理
def preprocess_data(data):
    data = data.drop(['Ticket', 'Cabin', 'Name'], axis=1)  # 删除不需要的列
    data = data.assign(Age=data['Age'].fillna(data['Age'].median()))  # 填充缺失的年龄数据
    data = data.assign(Embarked=data['Embarked'].fillna(data['Embarked'].mode()[0]))  # 填充缺失的登船港口数据
    data = data.assign(Fare=data['Fare'].fillna(data['Fare'].median()))  # 填充缺失的票价数据
    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})  # 将性别转换为数值
    data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})  # 将登船港口转换为数值
    return data

train_data = preprocess_data(train_data)
test_data = preprocess_data(test_data)

# 将数据分为特征和目标
X = train_data.drop('Survived', axis=1)
y = train_data['Survived']

# 将训练数据分为训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练模型
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 验证模型
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f'Validation Accuracy: {accuracy:.2f}')

# 在测试数据上进行预测
passenger_ids = test_data['PassengerId']  # 保留 PassengerId 列
predictions = model.predict(test_data)

# 将预测结果保存到 CSV 文件，确保 Survived 列为整数
output = pd.DataFrame({'PassengerId': passenger_ids, 'Survived': predictions.astype(int)})  # 使用保留的 PassengerId 列
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")