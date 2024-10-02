import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import talib
from scipy.signal import argrelextrema
import numpy as np

# Step 1: Download Historical Data with user choice for intraday or daily outlook
def download_data(ticker, outlook):
    """
    Download historical market data for the given ticker.
    The user can select either 'intraday' or 'daily' for different outlooks.
    """
    if outlook == 'intraday':
        period = '60d'   # 60 days of data for intraday trading
        interval = '5m'  # 5-minute intervals for day trading
    elif outlook == 'daily':
        period = '1y'    # 1 year of data for daily analysis
        interval = '1d'  # 1-day intervals for longer-term trading
    else:
        raise ValueError("Invalid outlook choice. Please choose 'intraday' or 'daily'.")
    
    stock = yf.Ticker(ticker)
    data = stock.history(period=period, interval=interval)
    return data

# Step 2: Calculate Technical Indicators
def calculate_indicators(data):
    # Calculate RSI (14-period)
    data['RSI'] = talib.RSI(data['Close'], timeperiod=14)
    
    # Calculate MACD
    data['MACD'], data['MACD_Signal'], _ = talib.MACD(data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    
    # Calculate Bollinger Bands (20-period)
    data['Upper_Band'], data['Middle_Band'], data['Lower_Band'] = talib.BBANDS(data['Close'], timeperiod=20, nbdevup=2, nbdevdn=2)
    
    # Calculate Moving Averages
    data['SMA50'] = talib.SMA(data['Close'], timeperiod=50)
    data['EMA20'] = talib.EMA(data['Close'], timeperiod=20)
    
    return data

# Step 3: Detect Reversals using a combination of signals
def detect_reversals(data, order=5):
    # Local minima and maxima detection
    data['minima'] = data['Close'][argrelextrema(data['Close'].values, np.less_equal, order=order)[0]]
    data['maxima'] = data['Close'][argrelextrema(data['Close'].values, np.greater_equal, order=order)[0]]
    
    # Define conditions for identifying potential reversals
    reversal_conditions = (
        ((data['RSI'] < 30) | (data['RSI'] > 70)) | 
        ((data['Close'] <= data['Lower_Band']) | (data['Close'] >= data['Upper_Band'])) |
        ((data['MACD'] < data['MACD_Signal']) | (data['MACD'] > data['MACD_Signal']))
    )
    
    data['Potential_Reversal'] = reversal_conditions
    return data

# Step 4: Basic Backtesting System for Day Trading Strategy
def backtest_strategy(data):
    initial_balance = 10000  # Initial capital in USD
    balance = initial_balance
    position = 0  # Track whether we're in a trade
    trade_log = []  # Keep track of trades
    num_trades = 0
    winning_trades = 0

    for i in range(1, len(data)):
        # Buy signal: RSI < 30 and price near the lower Bollinger Band
        if data['RSI'].iloc[i] < 30 and data['Close'].iloc[i] <= data['Lower_Band'].iloc[i] and position == 0:
            # Buy at current price
            buy_price = data['Close'].iloc[i]
            position = balance / buy_price  # Number of shares we can buy
            balance = 0  # Invest all money
            trade_log.append(f"Buy {position} shares at {buy_price} on {data.index[i]}")
        
        # Sell signal: RSI > 70 and price near the upper Bollinger Band
        elif data['RSI'].iloc[i] > 70 and data['Close'].iloc[i] >= data['Upper_Band'].iloc[i] and position > 0:
            # Sell at current price
            sell_price = data['Close'].iloc[i]
            balance = position * sell_price  # Liquidate position
            position = 0  # No more shares
            trade_log.append(f"Sell at {sell_price} on {data.index[i]}")
            
            # Track trade results
            profit = balance - initial_balance
            if profit > 0:
                winning_trades += 1
            num_trades += 1
    
    # Final results
    final_balance = balance if position == 0 else position * data['Close'].iloc[-1]
    total_return = (final_balance - initial_balance) / initial_balance * 100
    win_rate = winning_trades / num_trades if num_trades > 0 else 0

    print("\nTrade Summary:")
    for log in trade_log:
        print(log)
    
    print(f"\nInitial Balance: ${initial_balance}")
    print(f"Final Balance: ${final_balance:.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Number of Trades: {num_trades}")
    print(f"Winning Trades: {winning_trades}")
    print(f"Win Rate: {win_rate * 100:.2f}%")

# Step 5: Plot the data with reversal points and technical indicators
def plot_reversals(data, ticker):
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot the closing price
    ax.plot(data.index, data['Close'], label=f'{ticker} Close Price', color='blue')

    # Plot Bollinger Bands
    ax.plot(data.index, data['Upper_Band'], color='orange', label='Upper Bollinger Band')
    ax.plot(data.index, data['Lower_Band'], color='orange', label='Lower Bollinger Band')

    # Plot Moving Averages
    ax.plot(data.index, data['SMA50'], color='purple', label='50-day SMA')
    ax.plot(data.index, data['EMA20'], color='green', label='20-day EMA')

    # Plot reversal points
    ax.scatter(data.index, data['minima'], color='green', label='Minima (Potential Buy)', marker='v', s=100)
    ax.scatter(data.index, data['maxima'], color='red', label='Maxima (Potential Sell)', marker='^', s=100)

    # Plot potential reversal flags
    ax.scatter(data.index[data['Potential_Reversal']], data['Close'][data['Potential_Reversal']], 
               color='yellow', marker='*', s=200, label='Potential Reversal')

    ax.set_title(f'{ticker} Price with Technical Indicators and Reversals')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True)
    
    # Show the plot
    plt.show()

# Example usage with user input
ticker = input("Enter the ticker symbol for the stock or index (e.g., SPY, QQQ, NASDAQ): ").upper()

# Let the user choose between intraday or daily outlook
outlook = input("Choose your outlook (intraday/daily): ").lower()
if outlook not in ['intraday', 'daily']:
    print("Invalid input! Please choose either 'intraday' or 'daily'.")
else:
    data = download_data(ticker, outlook)
    data = calculate_indicators(data)
    data = detect_reversals(data)

    # Perform backtesting
    backtest_strategy(data)

    # Plot the reversals with indicators
    plot_reversals(data, ticker)
