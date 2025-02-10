def align_time_series(stock_data, sentiment_data):
    # 对齐股票数据和情感数据的时间序列
    combined_data = stock_data.merge(sentiment_data, on='date', how='inner')
    return combined_data

def train_and_evaluate(model, train_data, test_data, epochs=10, batch_size=32):
    # 训练和评估模型
    X_train, y_train = train_data.drop('target', axis=1), train_data['target']
    X_test, y_test = test_data.drop('target', axis=1), test_data['target']
    
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    predictions = model.predict(X_test)
    
    accuracy = (predictions == y_test).mean()
    print(f'Accuracy: {accuracy:.2f}')
    
    return predictions, accuracy