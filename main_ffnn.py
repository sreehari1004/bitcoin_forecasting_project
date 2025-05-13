# main_ffnn.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from preprocessing import load_and_preprocess
from evaluate import evaluate_model, plot_predictions
from models.ffnn_model import train_ffnn
from sklearn.model_selection import train_test_split
import numpy as np

# 1. Load and preprocess the data
file_path = 'data/bitcoin_5min_latest.csv'
X, y, df, scaler, timestamps = load_and_preprocess(file_path, n_lags=5)

# 2. Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 3. Train the FFNN model
model = train_ffnn(X_train, y_train)

# 4. Evaluate the model
rmse, predictions = evaluate_model(model, X_test, y_test, model_type='keras')
print(f"FFNN RMSE: {rmse:.4f}")

# 5. Plot predictions
plot_predictions(y_test, predictions, title="FFNN: Forecast vs Actual")

# Save the rmse for comparison
with open('Results/FFNN_rmse.txt', 'w') as file:
    file.write(str(rmse))




from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# 7. Calculate R² score
r2 = r2_score(y_test, predictions)
print(f"FFNN R² score: {r2:.4f}")

# 8. Calculate MAPE (Mean Absolute Percentage Error)
mape = np.mean(np.abs((y_test - predictions) / y_test)*100)
accuracy = 100 - mape
print(f"FFNN Accuracy: {accuracy:.4f}%")

# 9. Save the R² score and Accuracy into the result file
with open('Results/FFNN.txt', 'w') as file:
    file.write(f"RMSE: {rmse:.4f}\n")
    file.write(f"R²: {r2:.4f}\n")
    file.write(f"Accuracy: {accuracy:.4f}%\n")