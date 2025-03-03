import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
def load_and_preprocess_data():
    # 使用 os.path.join 构建文件路径
    data_dir = 'd:/GitHubPro/AI-Programming-Journal/kaggle/taitanic/data'
    train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    
    # 数据预处理：填充缺失值
    for df in [train_df, test_df]:
        df['Age'].fillna(df['Age'].median(), inplace=True)
        df['Fare'].fillna(df['Fare'].median(), inplace=True)
        df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    
    # 将性别映射为数值，并独热编码Embarked
    gender_map = {'male': 0, 'female': 1}
    for df in [train_df, test_df]:
        df['Sex'] = df['Sex'].map(gender_map)
        embarked_dummies = pd.get_dummies(df['Embarked'], prefix='Embarked')
        df.drop('Embarked', axis=1, inplace=True)
        df = df.join(embarked_dummies)
        # 将处理后的 df 写回列表以供后续使用
        if df is train_df:
            train_df = df
        else:
            test_df = df

    # 删除无用字段
    drop_cols = ['Name', 'Ticket', 'Cabin', 'PassengerId']
    train_df.drop(drop_cols, axis=1, inplace=True)
    test_df.drop(drop_cols, axis=1, inplace=True)
    
    # 如果 test_df 存在 Survived 列，则删除它
    if 'Survived' in test_df.columns:
        test_df.drop('Survived', axis=1, inplace=True)
    
    # 对齐 test_df 特征：保证 test_df 拥有与 train_df（去除 Survived）相同的列
    train_features = train_df.drop('Survived', axis=1).columns
    test_df = test_df.reindex(columns=train_features, fill_value=0)
    
    # 数据增强：上采样少数类别（假设Survived为0和1的二分类问题）
    df_majority = train_df[train_df.Survived==0]
    df_minority = train_df[train_df.Survived==1]
    df_minority_upsampled = resample(df_minority,
                                     replace=True,
                                     n_samples=len(df_majority),
                                     random_state=42)
    train_df = pd.concat([df_majority, df_minority_upsampled])
    
    # 分离特征和标签
    X = train_df.drop('Survived', axis=1).values
    y = train_df['Survived'].values.reshape(-1, 1)
    
    # 标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_test = scaler.transform(test_df.values)
    
    return X, y, X_test, scaler

class TitanicNet(nn.Module):
    def __init__(self, input_dim):
        super(TitanicNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

def train_model(X_train, y_train, X_val, y_val, input_dim, epochs=50, batch_size=32):
    model = TitanicNet(input_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        # 验证损失
        model.eval()
        with torch.no_grad():
            val_outputs = model(torch.FloatTensor(X_val))
            val_loss = criterion(val_outputs, torch.FloatTensor(y_val))
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
    return model

def main():
    X, y, X_test, scaler = load_and_preprocess_data()
    # 拆分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    input_dim = X_train.shape[1]
    
    model = train_model(X_train, y_train, X_val, y_val, input_dim)
    
    # 对测试数据预测
    model.eval()
    predictions = model(torch.FloatTensor(X_test)).detach().numpy()
    # 设定阈值0.5
    pred_labels = (predictions >= 0.5).astype(int)
    
    # 使用绝对路径读取 test.csv 获取 PassengerId
    data_dir = 'd:/GitHubPro/AI-Programming-Journal/kaggle/taitanic/data'
    test_csv_path = os.path.join(data_dir, 'test.csv')
    if 'PassengerId' in pd.read_csv(test_csv_path).columns:
        test_ids = pd.read_csv(test_csv_path)['PassengerId']
    else:
        test_ids = np.arange(len(pred_labels))
    submission = pd.DataFrame({'PassengerId': test_ids, 'Survived': pred_labels.flatten()})
    submission.to_csv('submission.csv', index=False)
    print("Submission file generated.")

if __name__ == '__main__':
    main()
