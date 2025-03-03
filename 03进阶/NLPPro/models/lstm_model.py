class LSTMModel:
    def __init__(self, input_shape, lstm_units=50, dropout_rate=0.2):
        self.input_shape = input_shape
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model = self.build_model()

    def build_model(self):
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout

        model = Sequential()
        model.add(LSTM(self.lstm_units, input_shape=self.input_shape, return_sequences=True))
        model.add(Dropout(self.dropout_rate))
        model.add(LSTM(self.lstm_units))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(1, activation='linear'))

        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def train(self, X_train, y_train, epochs=50, batch_size=32, validation_data=None):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=validation_data)

    def predict(self, X):
        return self.model.predict(X)