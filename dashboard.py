import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def get_market_data(tickers=['AAPL', 'GOOGL', 'MSFT']):
    data = {}
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y", interval="1d")  # Extended period for ML

        # Prepare the data for machine learning
        hist['Return'] = hist['Close'].pct_change()  # Calculate daily returns
        hist['Lag1'] = hist['Close'].shift(1)  # Lagged closing price
        hist.dropna(inplace=True)  # Drop rows with NaN values

        # Define features (X) and labels (y)
        X = hist[['Lag1']]
        y = hist['Close']

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a simple linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predicting the test set
        predictions = model.predict(X_test)

        # Evaluate the model
        mse = mean_squared_error(y_test, predictions)

        # Store the model and evaluation results
        data[ticker] = {
            'Close': hist['Close'].tolist(),
            'Dates': hist.index.strftime('%Y-%m-%d').tolist(),
            'Model MSE': mse,
        }

    return data
