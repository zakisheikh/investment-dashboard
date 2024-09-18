from flask import Flask, jsonify
import yfinance as yf
import pandas as pd

app = Flask(__name__)

@app.route('/stock/<ticker>/<time_frame>', methods=['GET'])
def get_stock_data(ticker, time_frame):
    stock_data = yf.Ticker(ticker)

    # Define periods for different time frames
    periods = {
        '1m': '1m',     # 1 minute
        '2m': '2m',     # 2 minutes
        '5m': '5m',     # 5 minutes
        '15m': '15m',   # 15 minutes
        '30m': '30m',   # 30 minutes
        '60m': '60m',   # 60 minutes
        '90m': '90m',   # 90 minutes
        '1h': '1h',     # 1 hour
        '1d': '1d',     # 1 day
        '5d': '5d',     # 5 days
        '1wk': '1wk',   # 1 week
        '1mo': '1mo',   # 1 month
        '3mo': '3mo',   # 3 months
    }

    # Fetch historical data based on the time frame
    period = periods.get(time_frame)
    if not period:
        return jsonify({"error": "Invalid time frame."}), 400

    history = stock_data.history(period=period, interval=period)

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

    # Format the date to MM/DD/YY
    history.reset_index(inplace=True)
    history['Date'] = history['Date'].dt.strftime('%m/%d/%y')

    # Prepare data for response
    response_data = history[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'MA21', 'MA50', 'MA200', 'RSI']].to_dict(orient='records')

    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)
