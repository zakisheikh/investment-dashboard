import yfinance as yf
import pandas as pd
import talib
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# Step 1: Download Historical Data for multiple timeframes (Intraday and Daily)
def download_data(ticker):
    """
    Download both intraday and daily historical market data for a given ticker.
    """
    # Fetch intraday data (5-minute intervals, 60 days)
    intraday_data = yf.Ticker(ticker).history(period='60d', interval='5m')
    
    # Fetch daily data (1 year)
    daily_data = yf.Ticker(ticker).history(period='1y', interval='1d')
    
    return intraday_data, daily_data

# Step 2: Calculate Technical Indicators for both timeframes
def calculate_indicators(data):
    """
    Calculate key technical indicators such as RSI, MACD, Bollinger Bands, ADX, and Stochastic Oscillator.
    """
    # RSI
    data['RSI'] = talib.RSI(data['Close'], timeperiod=14)
    
    # MACD
    data['MACD'], data['MACD_Signal'], _ = talib.MACD(data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    
    # Bollinger Bands
    data['Upper_Band'], data['Middle_Band'], data['Lower_Band'] = talib.BBANDS(data['Close'], timeperiod=20, nbdevup=2, nbdevdn=2)
    
    # Stochastic Oscillator
    data['SlowK'], data['SlowD'] = talib.STOCH(data['High'], data['Low'], data['Close'], fastk_period=14, slowk_period=3, slowd_period=3)
    
    # ADX (Average Directional Index)
    data['ADX'] = talib.ADX(data['High'], data['Low'], data['Close'], timeperiod=14)
    
    # On-Balance Volume (OBV)
    data['OBV'] = talib.OBV(data['Close'], data['Volume'])
    
    # Moving Averages
    data['SMA50'] = talib.SMA(data['Close'], timeperiod=50)
    data['EMA20'] = talib.EMA(data['Close'], timeperiod=20)
    
    return data

# Step 3: Trend Detection and Confirmation using ADX
def detect_trend(data):
    """
    Detect if the market is trending using ADX. Returns 'trend' if ADX > 25, else 'range' (mean-reverting market).
    """
    latest_adx = data['ADX'].iloc[-1]
    if latest_adx > 25:
        return "trend"
    else:
        return "range"

# Step 4: Risk Management and Position Sizing
def calculate_position(balance, entry_price, stop_loss, risk_percentage=1):
    """
    Calculate the position size based on the account balance, risk per trade, and stop-loss level.
    """
    risk_amount = balance * (risk_percentage / 100)  # Risk per trade (e.g., 1% of balance)
    stop_loss_distance = abs(entry_price - stop_loss)
    
    if stop_loss_distance == 0:
        stop_loss_distance = entry_price * 0.01  # Avoid division by zero (1% default)
    
    position_size = risk_amount / stop_loss_distance
    return position_size

# Step 5: Backtesting the Strategy
def backtest_strategy(data, regime, initial_balance=10000, risk_percentage=1):
    """
    Perform a backtest of the strategy on the given data.
    """
    balance = initial_balance
    position = 0
    trade_log = []
    num_trades = 0
    winning_trades = 0

    for i in range(1, len(data)):
        latest_rsi = data['RSI'].iloc[i]
        latest_close = data['Close'].iloc[i]
        lower_band = data['Lower_Band'].iloc[i]
        upper_band = data['Upper_Band'].iloc[i]
        adx = data['ADX'].iloc[i]

        # Define stop-loss as 1% below/above the entry price for buys/sells
        stop_loss_buy = latest_close * 0.99
        stop_loss_sell = latest_close * 1.01

        # Buy condition for trend or range markets
        if regime == "trend" and latest_rsi < 30 and latest_close <= lower_band:
            position_size = calculate_position(balance, latest_close, stop_loss_buy, risk_percentage)
            balance -= position_size * latest_close
            position = position_size  # Buy the position
            trade_log.append(f"Buy {position_size} shares at {latest_close} on {data.index[i]}")
        elif regime == "range" and latest_rsi > 70 and latest_close >= upper_band:
            position_size = calculate_position(balance, latest_close, stop_loss_sell, risk_percentage)
            balance += position_size * latest_close
            position = 0  # Sell the position
            trade_log.append(f"Sell at {latest_close} on {data.index[i]}")

        # Track wins and losses
        if position == 0 and balance > initial_balance:
            winning_trades += 1
        num_trades += 1
    
    final_balance = balance + (position * data['Close'].iloc[-1]) if position > 0 else balance
    return final_balance, trade_log, winning_trades, num_trades

# Step 6: Visualize the Results (Plotly + Matplotlib)
def plot_reversals(data, ticker):
    """
    Visualize the price movements, indicators, and backtested trades.
    """
    fig = go.Figure()

    # Plot Closing Price
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name=f'{ticker} Close Price', line=dict(color='blue')))
    
    # Plot Bollinger Bands
    fig.add_trace(go.Scatter(x=data.index, y=data['Upper_Band'], mode='lines', name='Upper Bollinger Band', line=dict(color='orange', dash='dash')))
    fig.add_trace(go.Scatter(x=data.index, y=data['Lower_Band'], mode='lines', name='Lower Bollinger Band', line=dict(color='orange', dash='dash')))
    
    # Plot Moving Averages
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA50'], mode='lines', name='50-day SMA', line=dict(color='purple')))
    fig.add_trace(go.Scatter(x=data.index, y=data['EMA20'], mode='lines', name='20-day EMA', line=dict(color='green')))
    
    fig.update_layout(title=f'{ticker} Price with Technical Indicators', xaxis_title='Date', yaxis_title='Price', hovermode='x unified', showlegend=True)
    fig.show()

# Example usage
ticker = "SPY"  # You can replace this with any stock or index symbol
intraday_data, daily_data = download_data(ticker)

# Calculate indicators for both timeframes
intraday_data = calculate_indicators(intraday_data)
daily_data = calculate_indicators(daily_data)

# Detect the market regime using ADX
market_regime = detect_trend(daily_data)
print(f"Market Regime: {market_regime}")

# Backtest the strategy based on the market regime
final_balance, trades, wins, total_trades = backtest_strategy(daily_data, market_regime)

# Show the backtesting results
print(f"Final Balance: ${final_balance:.2f}")
print(f"Number of Trades: {total_trades}, Wins: {wins}")
for trade in trades:
    print(trade)

# Plot the results
plot_reversals(daily_data, ticker)
