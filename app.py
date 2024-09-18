from flask import Flask, jsonify
import yfinance as yf
import pandas as pd

app = Flask(__name__)

@app.route('/stock/<ticker>', methods=['GET'])
def get_stock_data(ticker):
    stock_data = yf.Ticker(ticker)
    
    # Fetch 10 years of historical data
    history = stock_data.history(period="10y")

    # Check if history is empty
    if history.empty:
        return jsonify({"error": "No data found for the given ticker."}), 404

    # Calculate Moving Averages
    history['MA21'] = history['Close'].rolling(window=21).mean()
    history['MA50'] = history['Close'].rolling(window=50).mean()
    history['MA200'] = history['Close'].rolling(window=200).mean()

    # Calculate RSI
    delta = history['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    history['RSI'] = 100 - (100 / (1 + rs))

    # Convert index (Timestamp) to string and reset index
    history.reset_index(inplace=True)
    history['Date'] = history['Date'].dt.strftime('%m/%d/%y')

    # Prepare data for response
    response_data = history[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'MA21', 'MA50', 'MA200', 'RSI']].to_dict(orient='records')

    return jsonify(response_data)

@app.route('/stock/<ticker>/<time_frame>', methods=['GET'])
def get_stock_data_with_time_frame(ticker, time_frame):
    # Get historical data
    stock_data = yf.Ticker(ticker)
    history = stock_data.history(period="10y")

    # Check if history is empty
    if history.empty:
        return jsonify({"error": "No data found for the given ticker."}), 404

    # Resample data based on the time frame
    if time_frame in ['1m', '5m', '15m', '30m', '60m', '90m', '1h']:
        # Resample for intraday data
        history = history.resample('1T').mean()  # Resampling to minute-level data
    elif time_frame in ['1d', '5d', '1wk', '1mo', '3mo']:
        # Resample for daily or weekly data
        history = history.resample('1D').mean()  # Daily-level data
    else:
        return jsonify({"error": "Invalid time frame."}), 400

    # Calculate Moving Averages and RSI as before
    history['MA21'] = history['Close'].rolling(window=21).mean()
    history['MA50'] = history['Close'].rolling(window=50).mean()
    history['MA200'] = history['Close'].rolling(window=200).mean()

    delta = history['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    history['RSI'] = 100 - (100 / (1 + rs))

    # Format the date to MM/DD/YY
    history.reset_index(inplace=True)
    history['Date'] = history['Date'].dt.strftime('%m/%d/%y')

    # Prepare data for response
    response_data = history[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'MA21', 'MA50', 'MA200', 'RSI']].to_dict(orient='records')

    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)
