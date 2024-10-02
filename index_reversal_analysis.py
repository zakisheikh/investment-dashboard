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
    balance_history = [initial_balance]  # Track balance over time for max drawdown
    returns = []

    for i in range(1, len(data)):
        latest_rsi = data['RSI'].iloc[i]
        latest_close = data['Close'].iloc[i]
        lower_band = data['Lower_Band'].iloc[i]
        upper_band = data['Upper_Band'].iloc[i]
        stop_loss_buy = latest_close * 0.99
        stop_loss_sell = latest_close * 1.01
        target_price_buy = latest_close * 1.02  # 2% profit target for example
        target_price_sell = latest_close * 0.98  # 2% stop-loss target
        
        # Buy condition for range markets
        if regime == "range" and latest_rsi < 40 and latest_close <= lower_band:
            position_size = calculate_position(balance, latest_close, stop_loss_buy, target_price_buy, risk_percentage, min_rr_ratio)
            if position_size > 0:
                balance -= position_size * latest_close
                position = position_size  # Buy the position
                trade_log.append(f"Buy {position_size:.2f} shares at {latest_close:.2f} on {data.index[i]}")
        
        # Sell condition for range markets
        elif regime == "range" and latest_rsi > 60 and latest_close >= upper_band and position > 0:
            balance += position * latest_close
            profit = (position * latest_close) - (position * target_price_sell)  # Calculate profit/loss
            trade_log.append(f"Sell {position:.2f} shares at {latest_close:.2f} on {data.index[i]}")
            if profit > 0:
                gross_profit += profit
                wins += 1
            else:
                gross_loss += abs(profit)
            position = 0  # Reset position after sell

        # Track balance history and returns
        balance_history.append(balance)
        returns.append((balance - initial_balance) / initial_balance)

        num_trades += 1

    final_balance = balance + (position * data['Close'].iloc[-1]) if position > 0 else balance
    
    # Calculate performance metrics
    sharpe_ratio = calculate_sharpe_ratio(returns)
    max_drawdown = calculate_max_drawdown(balance_history)
    profit_factor = calculate_profit_factor(gross_profit, gross_loss)
    
    return final_balance, trade_log, wins, num_trades, sharpe_ratio, max_drawdown, profit_factor

# Step 6: Plot and Show Results
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
final_balance, trades, wins, total_trades, sharpe_ratio, max_drawdown, profit_factor = backtest_strategy(daily_data, market_regime)

# Show the backtesting results
print(f"Final Balance: ${final_balance:.2f}")
print(f"Number of Trades: {total_trades}, Wins: {wins}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}, Max Drawdown: {max_drawdown:.2f}, Profit Factor: {profit_factor:.2f}")
for trade in trades:
    print(trade)

# Plot the results
plot_reversals(daily_data, ticker)
