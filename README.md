# Stock Price Prediction using PyTorch & LSTMs

This repository contains an end-to-end pipeline for predicting stock prices (specifically configured for Microsoft - **MSFT**) using a Recurrent Neural Network with Long Short-Term Memory (LSTM) layers built in PyTorch.

---

## 📌 Project Overview

Stock market prediction is a classic time-series forecasting problem. This project implements:
1. **Data Retrieval:** Automated fetching of historical daily adjusted stock data via `yfinance`.
2. **Preprocessing & Scaling:** Normalization of closing prices using `StandardScaler` and creation of sliding temporal windows (`seq_length = 30`).
3. **Deep Learning Model:** A multi-layer LSTM neural network implemented in PyTorch (`nn.LSTM` + `nn.Linear`).
4. **Training & Optimization:** Optimized using Mean Squared Error loss (`MSELoss`) and the Adam optimizer over 200 epochs.
5. **Evaluation & Visualization:** Performance tracking using Root Mean Squared Error (RMSE) and custom Matplotlib subplots displaying actual vs. predicted prices alongside prediction errors.

---

## 🛠️ Requirements & Installation

You can install all required dependencies directly via `pip`:

```bash
pip install yfinance torch scikit-learn pandas numpy matplotlib
