# Stock Price Prediction using LSTM Neural Networks

## Overview
This project explores the application of LSTM (Long Short-Term Memory) neural networks in predicting future stock prices, specifically focusing on Tesla (TSLA) hourly data. The project critically evaluates the effectiveness of LSTM models in stock price prediction, considering the inherent unpredictability of financial markets.

## Data Description
- **Source**: Tesla (TSLA) stock data from Yahoo Finance.
- **Features**: Hourly data including open, close, high, low, and volume.
- **Target**: Hourly price variations.
- **Period**: Last 2 years for hourly data; 15 years for monthly data.

## Data Preprocessing
- **Normalization**: Price variations and volatility are normalized.
- **Volatility Calculation**: Short-term and long-term volatility metrics are computed.
- **Statistical Tests**: Jarque-Bera test confirms non-normal distribution of returns; Augmented Dickey-Fuller test indicates stationarity.

## Visualization
- Hourly stock price trends and distribution of returns are visualized.
- Correlation matrix indicates a lack of clear correlation between price variation and other variables.

## Time Series Analysis
- ACF and PACF tests show no significant auto-correlation in the series, suggesting that traditional ARIMA models may not be effective.

## LSTM Model
- **Features**: Close price, volume, MACD, and signal line.
- **Normalization**: MinMaxScaler applied for feature scaling.
- **Window Size**: 10-hour window for time series data.
- **Architecture**: Sequential LSTM model with Dropout layers.
- **Training**: TimeSeriesSplit for robust training and validation.
- **Evaluation**: Test loss, Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R-squared metrics.

## Critique and Limitations
- **Market Unpredictability**: Despite using advanced LSTM models, predicting stock prices remains challenging due to market volatility and external factors.
- **Data Limitations**: Using historical data alone may not capture the full spectrum of market dynamics.
- **Model Limitations**: LSTM's reliance on past data may not effectively capture sudden market shifts.
- **Evaluation Results**: While the LSTM model shows potential in pattern recognition, its predictions for future stock prices are not highly reliable.
- **Further Research**: Incorporating additional data sources like news sentiment and macroeconomic indicators could enhance prediction accuracy.

## Conclusion
The project highlights the potential and limitations of using LSTM neural networks for stock price prediction. While LSTM can capture time-series patterns, the inherently unpredictable nature of stock markets poses significant challenges, and the model's predictions should be interpreted with caution.

## Future Scope
- Integration of alternative data sources for a more holistic view of market dynamics.
- Experimentation with different neural network architectures and hyperparameters.
- Continuous evaluation of model performance in the ever-changing financial market.
