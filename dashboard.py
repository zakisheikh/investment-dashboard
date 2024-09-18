import yfinance as yf
import pandas as pd
from sklearn.metrics import mean_squared_error

def get_market_data(tickers=['AAPL']):
    data = {}
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y", interval="1d")
        data[ticker] = hist[['Close']]
        
    # Example of calculating a simple MSE (use your own model as needed)
    # For the sake of this example, we'll just use random values
    predicted_prices = [150] * len(data['AAPL'])  # Dummy predicted prices
    actual_prices = data['AAPL']['Close'].values
    model_mse = mean_squared_error(actual_prices, predicted_prices)

    return data, model_mse
