import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import os
'''
当前模型是一个简单的全连接网络，
可以尝试使用卷积神经网络(CNN)来进一步提升模型性能。
超参数调整
数据增强。。。
'''
# Load MNIST data from local CSV files
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Prepare the data
train_images = train_data.iloc[:, 1:].values.reshape(-1, 28, 28) / 255.0
train_labels = train_data.iloc[:, 0].values
test_images = test_data.values.reshape(-1, 28, 28) / 255.0

# Convert to PyTorch tensors
train_images = torch.tensor(train_images, dtype=torch.float32).unsqueeze(1)
train_labels = torch.tensor(train_labels, dtype=torch.long)
test_images = torch.tensor(test_images, dtype=torch.float32).unsqueeze(1)

train_dataset = torch.utils.data.TensorDataset(train_images, train_labels)
test_dataset = torch.utils.data.TensorDataset(test_images, torch.zeros(test_images.size(0), dtype=torch.long))

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = SimpleNN()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}')

model.eval()
predicted_labels = []
with torch.no_grad():
    for data, _ in test_loader:
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        predicted_labels.extend(predicted.numpy())

# Load sample submission file
sample_submission = pd.read_csv('sample_submission.csv')

# Create submission DataFrame
submission = pd.DataFrame({
    'ImageId': sample_submission['ImageId'],
    'Label': predicted_labels
})

# Save submission to CSV
submission.to_csv('submission.csv', index=False)

