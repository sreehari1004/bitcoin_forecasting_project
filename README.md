
#  Bitcoin Price Prediction using Deep Learning (LSTM & GRU)

##  Project Aim

This project aims to predict Bitcoin prices — a decentralized and highly volatile digital currency — using powerful deep learning models like **LSTM** and **GRU**. Unlike earlier studies that applied traditional machine learning models and found **Bidirectional RNN (BRNN)** to perform best (RMSE: 0.8475), our approach leverages deep sequence models that can better handle **long-term dependencies**, **randomness**, and **nonlinear patterns** in Bitcoin's price data.

---

##  Methodology

###  Data Collection

- **Source:** High-frequency Bitcoin price data from **2011 to 2021**
- **Original Size:** ~1 million data points
- **Used in Project:** 80,000 data points (trimmed due to compute limits)
- We prioritized **5-minute interval data** to capture fine-grained market movement.

---

###  Models Explored

| Model | Description |
|-------|-------------|
| SVR (Support Vector Regression) | Finds the optimal hyperplane for regression by maximizing margins. |
| GPR (Gated Recurrent Predictor) | A custom recurrent network with gating mechanisms. |
| RT (Regression Tree) | Decision tree-based regressor for continuous values. |
| kNN (K-Nearest Neighbors) | Predicts based on average of closest data points. |
| FFNN (Feedforward Neural Network) | Standard neural network without recurrence. |
| RBNN (Radial Basis Neural Network) | Uses radial basis functions for function approximation. |
| RBFNN (Radial Basis Function NN) | Maps inputs to higher dimensions for improved regression. |
| LSTM (Long Short-Term Memory) | Captures long-term time series dependencies with memory gates. |
| GRU (Gated Recurrent Unit) | Similar to LSTM but with fewer parameters and faster training. |

---

##  Results

| Model   | Forecasting Error |
|---------|-------------------|
| SVR     | 0.25306            |
| FFNN    | 0.01322            |
| kNN     | 0.18970            |
| RT      | 0.19051            |
| RBFNN   | 0.17168            |
| BRNN    | 0.02753            |
| **LSTM** | **0.01946**        |
| **GRU**  | **0.00856**        |

---

##  Key Findings

- **GRU** achieved the **lowest error** in forecasting Bitcoin prices: **0.00856**
- **SVR** had the highest error: **0.25306**
- Both **LSTM** and **GRU** outperformed traditional models by capturing deep temporal dependencies.
  
---

##  Repository Structure

