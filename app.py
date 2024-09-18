from flask import Flask, jsonify
import yfinance as yf
import pandas as pd

app = Flask(__name__)

# Function to calculate RSI
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

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
    history['RSI'] = calculate_rsi(history)

    # Convert index (Timestamp) to string and reset index
    history.reset_index(inplace=True)
    history['Date'] = history['Date'].dt.strftime('%m/%d/%y')

    # Prepare data for response
    response_data = history[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'MA21', 'MA50', 'MA200', 'RSI']].to_dict(orient='records')

    return jsonify(response_data)

@app.route('/stock/<ticker>/<time_frame>', methods=['GET'])
def get_stock_data_with_time_frame(ticker, time_frame):
    stock_data = yf.Ticker(ticker)
    
    # Define valid intervals based on yfinance's capabilities
    valid_intervals = {
        '1m': ('1mo', '1m'),  # 1-month period, 1-minute interval
        '5m': ('1mo', '5m'),  # 1-month period, 5-minute interval
        '15m': ('1mo', '15m'),  # 1-month period, 15-minute interval
        '30m': ('1mo', '30m'),  # 1-month period, 30-minute interval
        '1h': ('1mo', '1h'),  # 1-month period, 1-hour interval
        '1d': ('10y', '1d'),  # 10-year period, 1-day interval
        '5d': ('1y', '5d'),  # 1-year period, 5-day interval
        '1wk': ('10y', '1wk'),  # 10-year period, 1-week interval
        '1mo': ('10y', '1mo')  # 10-year period, 1-month interval
    }

    if time_frame not in valid_intervals:
        return jsonify({"error": "Invalid time frame."}), 400

    period, interval = valid_intervals[time_frame]

    # Fetch data based on the time frame
    history = stock_data.history(period=period, interval=interval)

    # Check if history is empty
    if history.empty:
        return jsonify({"error": "No data found for the given ticker."}), 404

    # Calculate Moving Averages (only if there is enough data)
    history['MA21'] = history['Close'].rolling(window=21).mean()
    history['MA50'] = history['Close'].rolling(window=50).mean()
    history['MA200'] = history['Close'].rolling(window=200).mean()

    # Calculate RSI
    history['RSI'] = calculate_rsi(history)

    # Convert index (Timestamp) to string and reset index
    history.reset_index(inplace=True)
    history['Date'] = history['Date'].dt.strftime('%m/%d/%y')

    # Prepare data for response
    response_data = history[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'MA21', 'MA50', 'MA200', 'RSI']].to_dict(orient='records')

    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)
