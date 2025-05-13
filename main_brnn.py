import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from preprocessing import load_and_preprocess
from evaluate import plot_predictions
from models.brnn_model import train_brnn, predict_brnn  # Import BRNN model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# 1. Load and preprocess the data
file_path = 'data/bitcoin_5min_latest.csv'
X, y, df, scaler, timestamps = load_and_preprocess(file_path, n_lags=5)

# 2. Split into train, validation, and test sets
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, shuffle=False)

# 3. Train the BRNN model
model = train_brnn(X_train, y_train, X_val, y_val, epochs=100, batch_size=32)

# 4. Predict
predictions = predict_brnn(model, X_test)

# 5. Evaluate the model
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f"BRNN RMSE: {rmse:.4f}")

# 6. Plot predictions
plot_predictions(y_test, predictions, title="BRNN: Forecast vs Actual")

# Save the rmse for comparison
with open('Results/BRNN_rmse.txt', 'w') as file:
    file.write(str(rmse))


from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# 7. Calculate R² score
r2 = r2_score(y_test, predictions)
print(f"BRNN R² score: {r2:.4f}")

# 8. Calculate MAPE (Mean Absolute Percentage Error)
mape = np.mean(np.abs((y_test - predictions) / y_test)*100)
accuracy = 100 - mape
print(f"BRNN Accuracy: {accuracy:.4f}%")

# 9. Save the R² score and Accuracy into the result file
with open('Results/BRNN.txt', 'w') as file:
    file.write(f"RMSE: {rmse:.4f}\n")
    file.write(f"R²: {r2:.4f}\n")
    file.write(f"Accuracy: {accuracy:.4f}%\n")


import matplotlib.pyplot as plt
import os

# Ensure the output directory exists
os.makedirs("Results", exist_ok=True)

# Set ultra-large font sizes for maximum clarity
plt.rcParams.update({
    'font.size': 32,
    'axes.titlesize': 42,
    'axes.labelsize': 38,
    'xtick.labelsize': 30,
    'ytick.labelsize': 30,
    'legend.fontsize': 34,
    'figure.titlesize': 42
})

# Create a wide and tall figure
plt.figure(figsize=(22, 12))
plt.plot(y_test, label='Actual Price', color='navy', linewidth=4)
plt.plot(predictions, label='Predicted Price', color='red', linewidth=4)  # Changed to red

plt.title("BRNN: Forecast vs Actual Bitcoin Prices", pad=25)
plt.xlabel("Time Steps")
plt.ylabel("Scaled Price")
plt.legend(loc='upper left', frameon=False)
plt.grid(True, linestyle='--', linewidth=1.2)
plt.tight_layout()

# Save as high-resolution PDF
plt.savefig("Results/BRNN_Prediction_vs_Actual.pdf", format='pdf', dpi=600)
plt.close()
