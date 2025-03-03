#LSTM股票价格预测

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 数据准备
# 通过API Yahoo Finance获取历史股票数据
import yfinance as yf
data = yf.download('AAPL', start='2020-01-01', end='2021-01-01')
data.to_csv('AAPL.csv')
data = pd.read_csv('AAPL.csv')
data.head()


# 通过Tushare获取A股股票数据

# 通过本地CSV文件加载数据

# 数据预处理：
# 1. 归一化：使用MinMaxScaler或StandardScaler将数据缩放到[0,1]或标准化。
# 2. 构建时间窗口：用过去N天的数据预测未来M天的价格（例如用过去30天预测下1天收盘价）。
# 3. 划分数据集：按时间顺序分为训练集（70%~80%）、验证集（10%~15%）和测试集（10%~15%）。




'''
1. LSTM在股票预测中的适用性

时间序列特性：股票价格是典型的时间序列数据，具有时间依赖性和潜在模式。

LSTM的优势：LSTM擅长捕捉长期依赖关系，适合处理序列数据中的非线性关系，能有效避免传统RNN的梯度消失问题。

局限性：股票市场受多种因素（如政策、突发事件）影响，具有高噪声和非平稳性，预测需谨慎。
'''
'''
2. 实现步骤



数据来源：、Tushare）或本地CSV文件。

关键特征：

开盘价（Open）、收盘价（Close）、最高价（High）、最低价（Low）、成交量（Volume）等。

可添加技术指标（如移动平均线MA、RSI、MACD）。

数据预处理：

归一化：使用MinMaxScaler或StandardScaler将数据缩放到[0,1]或标准化。

构建时间窗口：用过去N天的数据预测未来M天的价格（例如用过去30天预测下1天收盘价）。

划分数据集：按时间顺序分为训练集（70%~80%）、验证集（10%~15%）和测试集（10%~15%）。
'''
'''

# 模型结构
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(time_steps, feature_dim)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))  # 预测未来1天的价格

model.compile(optimizer='adam', loss='mean_squared_error')

# 2.3 模型训练

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val),
    verbose=1
)
# 参数调整：可通过调整time_steps（时间窗口长度）、LSTM层数、神经元数量等优化模型。

#2.4 预测与评估

#预测：
predicted_prices = model.predict(X_test)
#反归一化：将预测结果还原到原始价格范围。

#评估指标：

#均方误差（MSE）、均方根误差（RMSE）、平均绝对误差（MAE）。

#可视化：绘制真实价格与预测价格的对比曲线。
'''
'''
3. 改进方向

特征工程：
结合新闻情感分析（如NLP处理财经新闻）。

添加宏观经济指标（如利率、GDP）。

模型优化：

使用双向LSTM（BiLSTM）或Attention机制。

结合CNN提取局部特征（Hybrid CNN-LSTM模型）。

集成方法：

将LSTM与Prophet、ARIMA等传统模型结合。

使用集成学习（如Stacking）。

4. 注意事项
过拟合风险：股票数据噪声大，需通过Dropout、早停法（Early Stopping）、正则化等手段避免过拟合。

动态更新：定期用新数据重新训练模型以适应市场变化。

实际应用限制：模型预测结果仅作参考，不可完全依赖（市场具有随机性和不可预测性）。
'''
'''
#5. 代码
# 数据加载与预处理
data = pd.read_csv('stock_data.csv')
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))

# 构建时间窗口
def create_dataset(data, time_steps=30):
    X, y = [], []
    for i in range(len(data)-time_steps):
        X.append(data[i:i+time_steps])
        y.append(data[i+time_steps])
    return np.array(X), np.array(y)

X, y = create_dataset(scaled_data)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# 训练模型与预测
model.fit(X_train, y_train, epochs=100, batch_size=32)
predicted = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predicted)
'''
'''
6. 结论
LSTM在股票价格预测中展现了处理时间序列数据的潜力，但需结合领域知识和持续优化。实际应用中建议将其作为辅助工具，结合基本面分析和风险管理策略。
'''
