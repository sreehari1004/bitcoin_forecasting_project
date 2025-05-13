from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np


def reshape_for_brnn(X):
    return np.reshape(X, (X.shape[0], X.shape[1], 1))  # (samples, timesteps, features)


def build_brnn(input_shape):
    model = Sequential()
    model.add(Bidirectional(LSTM(50, activation='tanh', return_sequences=False), input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model


def train_brnn(X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
    X_train_brnn = reshape_for_brnn(X_train)
    X_val_brnn = reshape_for_brnn(X_val)

    model = build_brnn(input_shape=(X_train.shape[1], 1))
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train_brnn, y_train, validation_data=(X_val_brnn, y_val),
              epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[early_stop])
    return model


def predict_brnn(model, X_test):
    X_test_brnn = reshape_for_brnn(X_test)
    return model.predict(X_test_brnn).flatten()
