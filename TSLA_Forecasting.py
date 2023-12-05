# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 21:31:07 2023

@author: Fayssal

Time Series project - Real-time forecast of Tesla stock
"""

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


tesla_stock = yf.Ticker("TSLA")

# Importing the hourly data of TSLA stock, including open, close, high, low, and volume 
tesla_hourly_data = tesla_stock.history(period="2y", interval="1h")

# Calculate the pct_change for each hour, to get normalized data
tesla_hourly_data['Price Variation'] = tesla_hourly_data['Close'].pct_change()
# For the first row, we will fill the missing value by the mean, as we do not want to have NaN
tesla_hourly_data['Price Variation'].fillna(tesla_hourly_data['Price Variation'].mean(), inplace=True)

# Short-term Volatility (per year) calculation
short_term_volatility = tesla_hourly_data['Price Variation'].std() * (252 * 8)**0.5
tesla_hourly_data['Short-term Volatility'] = short_term_volatility

# Long-term Volatility calculation
tesla_monthly_data = tesla_stock.history(period="15y", interval="1mo")
tesla_monthly_data['Price Variation'] = tesla_monthly_data['Close'].pct_change()
long_term_volatility = tesla_monthly_data['Price Variation'].std() * 12**0.5
tesla_hourly_data['Long-Term Volatility'] = long_term_volatility


# Describe the data
tesla_hourly_data.head()
tesla_hourly_data.info()
tesla_hourly_data.describe()

# Visualize the close of TSLA stock overtime
tesla_hourly_data['Close'].plot(figsize = (12,6))
plt.title("Hourly price overtime")
plt.xlabel("Date")
plt.ylabel('Price')
plt.show()

# Visualize distribution of price variations
tesla_hourly_data['Price Variation'].plot(kind='hist', bins=100, figsize=(12, 6))
plt.title('Distribution of Tesla Returns')
plt.xlabel('Returns')
plt.ylabel('Frequency')
plt.show()

# Visualize the density of the returns
sns.displot(tesla_hourly_data['Price Variation'], kde=True, bins=100)
plt.title('Density of Tesla returns')
plt.xlabel('Tesla Returns')
plt.ylabel('Observations')
plt.show()

""" The returns seem to follow a normal distribution. Check for normality with Jarque Bera Test : """

from scipy import stats
res = stats.jarque_bera(tesla_hourly_data['Price Variation'])

res.statistic
res.pvalue
""" We reject the null hypothesis, which means that our returns are not following a normal distribution """

# Correlation matrix
corr_matrix = tesla_hourly_data.corr()

plt.figure(figsize=(12,6))
sns.heatmap(corr_matrix, annot = True, fmt = ".2f", cmap = "coolwarm")

plt.title("Correlation Matrix")
plt.xlabel("Variables")
plt.ylabel("Variables")
plt.show()

""" We dont see any clear correlation between Price Variation and the other variables. """

# Test for Stationarity of the serie - Augmented Dickey-Fuller Test

from statsmodels.tsa.stattools import adfuller
adfuller(tesla_hourly_data['Price Variation']) 

""" p-value = 0.0, we reject the null hypothesis. Our serie is stationary, variance and mean are constant overtime,
we can proceed to the ACF and PACF"""

# ACF test - Autocorrelation Function
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

# Plot the ACF and PACF of the data
plot_acf(tesla_hourly_data["Price Variation"], lags=50)
plot_pacf(tesla_hourly_data["Price Variation"], lags=50)
plt.show()

""" The values of the autocorrelation are inside the confidence zone. It means that the observations are not significantly different from zero.
We can conclude that our serie is not auto-correlated, the returns are then hard to predict with the informations within the last 50 hourly prices.
A Auto-Regressive Integrated Moving Average is not adapted for this case, as we don't have any auto-correlation or pac within the serie
A better choice for our project is to use deep learning techniques. """

""" We will combine several models to predict TSLA stock price. We will then use a filter to improve the model precision thanks
to MACD indicator. Let's calculate MACD first """
# We use ema 26 and 12 for the calculation
ema_26 = tesla_hourly_data['Close'].ewm(span=26, adjust=False).mean()
ema_12 = tesla_hourly_data['Close'].ewm(span=12, adjust=False).mean()

tesla_hourly_data['MACD'] = ema_12 - ema_26
tesla_hourly_data['Signal Line'] = tesla_hourly_data['MACD'].ewm(span=9, adjust=False).mean()


""" LSTM Neural Network Model: """

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Normalize the features (X)
features = ['Close', 'Volume', 'MACD', "Signal Line"]
tesla_features = tesla_hourly_data[features]

scaler = MinMaxScaler(feature_range=(0,1))
tesla_features_normalized = scaler.fit_transform(tesla_features)

# Define the timeframe 
window_size = 10  # test for model performance if we take the last 10hours data
X = []
y = []

for i in range(window_size, len(tesla_features_normalized)):
    X.append(tesla_features_normalized[i-window_size:i, :])  # i-window_size:i represents the 3 last hours of trading
    y.append(tesla_features_normalized[i, 0])  

# Convert lists to numpy arrays
X, y = np.array(X), np.array(y)

# Reshape de X to get the shape required for LSTM : [samples, time steps, features] 
X = np.reshape(X, (X.shape[0], X.shape[1], len(features)))

""" We will use TimeSeriesSplit to train the model in n splits, it will prevent overfitting and bias for our results :"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import TimeSeriesSplit

# We define 5 splits
tscv = TimeSeriesSplit(n_splits=10)

# Create LSTM Model:
    
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

    
for train_index, test_index in tscv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X.shape[2]))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X.shape[2]))

    # Create and train LSTM Model
    lstm_model = create_lstm_model(X_train.shape[1:])
    lstm_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=1)
    
    # Evaluate the model
    loss = lstm_model.evaluate(X_test, y_test, verbose=0)
    print(f'Test Loss: {loss}')

predictions = lstm_model.predict(X_test)

# Metrics for analyzing LSTM performance
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, predictions)
print(f'Mean Absolute Error (MAE): {mae}')

from sklearn.metrics import mean_squared_error
import math

rmse = math.sqrt(mean_squared_error(y_test, predictions))
print(f'Root Mean Squared Error (RMSE): {rmse}')

from sklearn.metrics import r2_score

r2 = r2_score(y_test, predictions)
print(f'R-squared (R²): {r2}')

import matplotlib.pyplot as plt

# Plot real valus and predictions
plt.figure(figsize=(10,6))
plt.plot(y_test, label='Valeurs Réelles')
plt.plot(predictions, label='Predictions of the model')
plt.title('Comparison Prediction - Real values')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

num_features = tesla_features_normalized.shape[1]  # Get the number of features
# Get the last window_size hours of data to start the prediction process
last_window = tesla_features_normalized[-window_size:].reshape((1, window_size, num_features))

# Initialize the list to store future predictions
future_predictions = []

# Generate predictions for the next 10 hours
for _ in range(10):
    # Predict the next hour using the last window of data
    next_hour_prediction = lstm_model.predict(last_window)[0]
    
    # Append the prediction to the list of future predictions
    future_predictions.append(next_hour_prediction)
    
    # Create a new timestep with the prediction and zeros for other features
    # Here we assume that the prediction corresponds to the first feature
    new_timestep = np.zeros((1, 1, num_features))
    new_timestep[0, 0, 0] = next_hour_prediction  # Set the first feature to the predicted value
    
    # Update the last window to include the new timestep and remove the oldest one
    last_window = np.append(last_window[:, 1:, :], new_timestep, axis=1)

# For the training dataset, we need to inverse transform all features
predicted_prices = scaler.inverse_transform(np.hstack((predictions, np.zeros((predictions.shape[0], num_features - 1)))))

# Inverse transform the entire dataset to get the real values for the 'Close' feature
real_prices = scaler.inverse_transform(tesla_features_normalized)[:, 0]

# Plot the real values and the model's predictions
plt.figure(figsize=(15,7))
# Create an index for the x-axis
time_steps = np.arange(len(real_prices))
# Plot the known 'Close' values
plt.plot(time_steps, real_prices, label='Real Values', color='blue')
# Plot the model's predictions for 'Close'
plt.plot(time_steps[-len(predicted_prices):], predicted_prices[:, 0], label='Model Predictions', color='orange')

# If you want to include future predictions, you will need to inverse transform them as well
# Make sure to reshape future_predictions to match the scaler's expected number of features
future_pred_array = np.array(future_predictions).reshape(-1, 1)
future_pred_full = np.zeros((len(future_pred_array), num_features))
future_pred_full[:, 0] = future_pred_array[:, 0]
future_predictions_rescaled = scaler.inverse_transform(future_pred_full)[:, 0]

# Plot the future predictions
plt.plot(time_steps[-len(future_predictions):], future_predictions_rescaled, label='Future Predictions', linestyle='--', color='green')
plt.title('Model Predictions and Future Price Forecasting')
plt.xlabel('Time (Hourly Intervals)')
plt.ylabel('Price')
plt.legend()
plt.show()

# Number of hours to look back and forward
look_back_hours = 20
look_forward_hours = 10

# Prepare the real prices (last 20 hours)
real_prices_last_part = real_prices[-look_back_hours:]

# Prepare the future predictions (next 10 hours)
future_predictions_rescaled_last_part = future_predictions_rescaled[:look_forward_hours]

# Create a time index for plotting
time_index_past = np.arange(len(real_prices_last_part))
time_index_future = np.arange(len(real_prices_last_part), len(real_prices_last_part) + len(future_predictions_rescaled_last_part))

plt.figure(figsize=(15,7))
# Plot the past real prices
plt.plot(time_index_past, real_prices_last_part, label='Past Real Prices', color='blue')

# Plot the future predictions
plt.plot(time_index_future, future_predictions_rescaled_last_part, label='Future Predictions', color='red')
plt.title('Past Prices and Future Predictions')
plt.xlabel('Time (Hours)')
plt.ylabel('Price')
plt.legend()
plt.show()









