import yfinance as yf
import pandas as pd
import talib
import numpy as np

# Step 1: Download Historical Data for both intraday and daily timeframes
def download_data(ticker, interval='5m'):
    """
    Download both intraday and daily historical market data for a given ticker.
    Limit intraday data based on Yahoo Finance's API restrictions.
    """
    # Fetch intraday data (handle the limitation of 1m data)
    if interval == '1m':
        intraday_data = yf.Ticker(ticker).history(period='7d', interval=interval)  # Only 7 days for 1-minute intervals
    else:
        intraday_data = yf.Ticker(ticker).history(period='21d', interval=interval)  # 21 days for larger intervals
    
    # Fetch daily data (1 year)
    daily_data = yf.Ticker(ticker).history(period='1y', interval='1d')
    
    return intraday_data, daily_data

# Step 2: Calculate Technical Indicators for both timeframes
def calculate_indicators(data):
    """
    Calculate key technical indicators such as RSI, MACD, Bollinger Bands, and Moving Averages.
    """
    if len(data) == 0:
        print("No data available to calculate indicators.")
        return data

    # RSI
    data['RSI'] = talib.RSI(data['Close'], timeperiod=14)
    
    # MACD
    data['MACD'], data['MACD_Signal'], _ = talib.MACD(data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    
    # Bollinger Bands
    data['Upper_Band'], data['Middle_Band'], data['Lower_Band'] = talib.BBANDS(data['Close'], timeperiod=20, nbdevup=2, nbdevdn=2)
    
    # Moving Averages
    data['SMA50'] = talib.SMA(data['Close'], timeperiod=50)
    
    return data

# Step 3: User-Defined Parameters (Profit Target, Risk-Reward Ratio)
def get_user_parameters():
    """
    Get user-defined parameters like profit target and risk-reward ratio.
    """
    profit_target = float(input("Enter your profit target percentage (e.g., 2 for 2%): "))
    risk_reward_ratio = float(input("Enter your preferred risk-reward ratio (e.g., 2 for 2:1): "))
    return profit_target / 100, risk_reward_ratio

# Step 4: Backtest the Strategy
def backtest_strategy(intraday_data, daily_data, profit_target, risk_reward_ratio):
    """
    Perform a backtest of the strategy using both intraday and daily data.
    """
    balance = 10000  # Initial capital
    position = 0
    trade_log = []
    num_trades = 0

    # Check if intraday data is available
    if len(intraday_data) == 0:
        print("No intraday data available for backtesting.")
        return balance, trade_log

    # Use daily data to guide the overall trend and intraday data for trades
    for i in range(1, len(intraday_data)):
        latest_rsi_intraday = intraday_data['RSI'].iloc[i]
        latest_close_intraday = intraday_data['Close'].iloc[i]
        lower_band_intraday = intraday_data['Lower_Band'].iloc[i]
        upper_band_intraday = intraday_data['Upper_Band'].iloc[i]

        latest_rsi_daily = daily_data['RSI'].iloc[-1]  # Current daily RSI
        lower_band_daily = daily_data['Lower_Band'].iloc[-1]  # Current daily lower Bollinger Band
        upper_band_daily = daily_data['Upper_Band'].iloc[-1]  # Current daily upper Bollinger Band

        stop_loss = latest_close_intraday * (1 - (1 / risk_reward_ratio))  # Calculate stop-loss based on risk-reward ratio
        target_price = latest_close_intraday * (1 + profit_target)  # Calculate target price based on profit target

        # Use multi-timeframe confirmation for buy signals
        if latest_rsi_intraday < 40 and latest_close_intraday <= lower_band_intraday and latest_rsi_daily < 40 and position == 0:
            # Buy the position based on both intraday and daily confirmation
            position = balance / latest_close_intraday  # Number of shares we can buy
            balance = 0  # All money is invested
            trade_log.append(f"Buy {position:.2f} shares at {latest_close_intraday:.2f} on {intraday_data.index[i]}")
        
        # Sell condition using both intraday and daily confirmation
        elif latest_rsi_intraday > 60 and latest_close_intraday >= upper_band_intraday and latest_rsi_daily > 60 and position > 0:
            # Sell the position
            balance = position * latest_close_intraday  # Liquidate the position
            trade_log.append(f"Sell {position:.2f} shares at {latest_close_intraday:.2f} on {intraday_data.index[i]}")
            position = 0  # Reset position after selling

        num_trades += 1

    # Final balance after backtest
    final_balance = balance if position == 0 else position * intraday_data['Close'].iloc[-1]
    
    # Display the results of the backtest
    print(f"Final Balance: ${final_balance:.2f}")
    print(f"Number of Trades: {num_trades}")
    for log in trade_log:
        print(log)

    return final_balance, trade_log

# Step 5: Suggest Next Trade
def suggest_next_trade(intraday_data, daily_data):
    """
    Suggest the next trade (buy/sell) based on the latest indicators from both intraday and daily timeframes.
    """
    if len(intraday_data) == 0 or len(daily_data) == 0:
        return "No data available to suggest a trade."

    latest_rsi_intraday = intraday_data['RSI'].iloc[-1]
    latest_close_intraday = intraday_data['Close'].iloc[-1]
    lower_band_intraday = intraday_data['Lower_Band'].iloc[-1]
    upper_band_intraday = intraday_data['Upper_Band'].iloc[-1]

    latest_rsi_daily = daily_data['RSI'].iloc[-1]
    lower_band_daily = daily_data['Lower_Band'].iloc[-1]
    upper_band_daily = daily_data['Upper_Band'].iloc[-1]

    # Multi-timeframe confirmation for buy/sell decisions
    if latest_rsi_intraday < 40 and latest_close_intraday <= lower_band_intraday and latest_rsi_daily < 40:
        return f"Suggested Trade: Buy at {latest_close_intraday:.2f} (RSI: {latest_rsi_intraday:.2f} Intraday, {latest_rsi_daily:.2f} Daily)"
    elif latest_rsi_intraday > 60 and latest_close_intraday >= upper_band_intraday and latest_rsi_daily > 60:
        return f"Suggested Trade: Sell at {latest_close_intraday:.2f} (RSI: {latest_rsi_intraday:.2f} Intraday, {latest_rsi_daily:.2f} Daily)"
    else:
        return f"Hold (Intraday RSI: {latest_rsi_intraday:.2f}, Daily RSI: {latest_rsi_daily:.2f})"

# Main Program
if __name__ == "__main__":
    ticker = input("Enter the ticker symbol for the stock (e.g., SPY, QQQ, AAPL): ").upper()
    interval = input("Enter the interval for intraday data (e.g., 1m, 5m, 15m, 30m, 60m): ").lower()
    
    # Download historical data for both intraday and daily timeframes
    intraday_data, daily_data = download_data(ticker, interval)
    
    # Calculate indicators for both timeframes
    intraday_data = calculate_indicators(intraday_data)
    daily_data = calculate_indicators(daily_data)

    # Get user parameters (profit target and risk-reward ratio)
    profit_target, risk_reward_ratio = get_user_parameters()

    # Run backtest
    final_balance, trade_log = backtest_strategy(intraday_data, daily_data, profit_target, risk_reward_ratio)

    # Suggest the next trade based on current indicators
    next_trade = suggest_next_trade(intraday_data, daily_data)
    print(next_trade)
