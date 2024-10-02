import yfinance as yf
import pandas as pd
import talib
import numpy as np
import plotly.graph_objects as go
import math

# Step 1: Download Historical Data for multiple timeframes (Intraday and Daily)
def download_data(ticker):
    """
    Download both intraday and daily historical market data for a given ticker.
    """
    intraday_data = yf.Ticker(ticker).history(period='1mo', interval='5m')
    daily_data = yf.Ticker(ticker).history(period='1y', interval='1d')
    return intraday_data, daily_data

# Step 2: Calculate Technical Indicators
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

# Step 4: Risk Management and Position Sizing with Risk-Reward Ratio
def calculate_position(balance, entry_price, stop_loss, target_price, risk_percentage=1, min_rr_ratio=2):
    """
    Calculate the position size based on the account balance, risk per trade, stop-loss, and reward targets.
    Ensure minimum risk-reward ratio is met before placing the trade.
    """
    # Calculate the risk amount (e.g., risking 1% of the account balance)
    risk_amount = balance * (risk_percentage / 100)
    
    # Calculate the stop-loss and reward distance
    stop_loss_distance = abs(entry_price - stop_loss)
    reward_distance = abs(target_price - entry_price)
    
    # Ensure the minimum risk-reward ratio is met
    if reward_distance / stop_loss_distance < min_rr_ratio:
        return 0  # Do not place the trade if the risk-reward ratio is insufficient
    
    # Calculate position size, and ensure it doesn't exceed the balance
    position_size = min(risk_amount / stop_loss_distance, balance / entry_price)
    
    # Add a minimum position size threshold to avoid extremely small trades
    if position_size < 0.01:  # Assume we don't want to trade less than 0.01 shares
        position_size = 0
    
    return position_size

# Sharpe Ratio calculation: (mean return - risk-free rate) / standard deviation of returns
def calculate_sharpe_ratio(returns, risk_free_rate=0):
    if len(returns) == 0:
        return 0
    excess_returns = [r - risk_free_rate for r in returns]
    if np.std(excess_returns) == 0:
        return 0
    return np.mean(excess_returns) / np.std(excess_returns)

# Max Drawdown calculation: maximum peak-to-trough decline in balance
def calculate_max_drawdown(balance_history):
    peak = balance_history[0]
    max_drawdown = 0
    for balance in balance_history:
        peak = max(peak, balance)
        drawdown = (peak - balance) / peak
        max_drawdown = max(max_drawdown, drawdown)
    return max_drawdown

# Profit Factor: ratio of gross profits to gross losses
def calculate_profit_factor(gross_profit, gross_loss):
    if gross_loss == 0:
        return np.inf  # Avoid division by zero
    return gross_profit / abs(gross_loss)

# Step 5: Backtesting the Strategy and Calculating Performance Metrics
def backtest_strategy(data, regime, initial_balance=10000, risk_percentage=1, min_rr_ratio=2):
    """
    Perform a backtest of the strategy on the given data. 
    Calculate performance metrics like Sharpe ratio, max drawdown, and profit factor.
    """
    balance = initial_balance
    position = 0
    trade_log = []
    num_trades = 0
    wins = 0
    gross_profit = 0
    gross_loss = 0
    balance_history = [initial_balance]  # Track balance
