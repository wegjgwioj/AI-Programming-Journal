import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Attention, Concatenate

# Load dataset
df = pd.read_csv('IMDB Dataset.csv')

# Preprocess data
sentences = df['review'].values
labels = df['sentiment'].values
labels = np.array([1 if label == 'positive' else 0 for label in labels])

# Tokenize and pad sequences
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
padded_sequences = pad_sequences(sequences, maxlen=200)

# Split data into training and testing sets
train_size = int(len(sentences) * 0.8)
x_train, x_test = padded_sequences[:train_size], padded_sequences[train_size:]
y_train, y_test = labels[:train_size], labels[train_size:]

# Build model
input_layer = Input(shape=(200,))
embedding_layer = Embedding(input_dim=5000, output_dim=128, input_length=200)(input_layer)
lstm_layer = LSTM(128, return_sequences=True)(embedding_layer)
attention_layer = Attention()([lstm_layer, lstm_layer])
concat_layer = Concatenate()([lstm_layer, attention_layer])
dense_layer = Dense(64, activation='relu')(concat_layer)
output_layer = Dense(1, activation='sigmoid')(dense_layer)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))

# Evaluate model
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')