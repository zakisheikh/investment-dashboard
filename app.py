from flask import Flask, jsonify
import yfinance as yf
import pandas as pd

app = Flask(__name__)

@app.route('/stock/<ticker>', methods=['GET'])
def get_stock_data(ticker):
    stock_data = yf.Ticker(ticker)
    history = stock_data.history(period="1y")  # Get 200 days of historical data

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
    history['Date'] = history['Date'].astype(str)

    # Prepare data for response
    response_data = history[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'MA21', 'MA50', 'MA200', 'RSI']].to_dict(orient='records')

    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)

