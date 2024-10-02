import yfinance as yf
import pandas as pd
import ta
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Step 1: Download Historical Data for both intraday and daily timeframes
def download_data(ticker, interval='5m'):
    """
    Download both intraday and daily historical market data for a given ticker.
    Handle the limitations of Yahoo Finance's API for different intraday intervals.
    """
    if interval == '1m':
        # 1-minute interval allows only 7 days
        intraday_data = yf.Ticker(ticker).history(period='5d', interval=interval)
    elif interval in ['2m', '5m']:
        # 2-minute and 5-minute intervals allow 30 days
        intraday_data = yf.Ticker(ticker).history(period='1mo', interval=interval)
    elif interval in ['15m', '30m']:
        # 15-minute and 30-minute intervals allow 90 days
        intraday_data = yf.Ticker(ticker).history(period='3mo', interval=interval)
    elif interval in ['60m', '1h']:
        # 60-minute and 1-hour intervals allow 180 days
        intraday_data = yf.Ticker(ticker).history(period='6mo', interval=interval)
    else:
        st.error(f"Interval '{interval}' is not supported. Please choose a valid interval.")
        return None, None

    # Fetch daily data (1 year)
    daily_data = yf.Ticker(ticker).history(period='1y', interval='1d')
    
    return intraday_data, daily_data

# Step 2: Calculate Technical Indicators for both timeframes
def calculate_indicators(data):
    """
    Calculate key technical indicators such as RSI, MACD, Bollinger Bands, and Moving Averages using the 'ta' library.
    """
    if data is None or len(data) == 0:
        st.error("No data available to calculate indicators.")
        return data

    try:
        # RSI
        rsi = ta.momentum.RSIIndicator(close=data['Close'], window=14)
        data['RSI'] = rsi.rsi()
        
        # MACD
        macd = ta.trend.MACD(close=data['Close'], window_slow=26, window_fast=12, window_sign=9)
        data['MACD'] = macd.macd()
        data['MACD_Signal'] = macd.macd_signal()
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(close=data['Close'], window=20, window_dev=2)
        data['Upper_Band'] = bollinger.bollinger_hband()
        data['Middle_Band'] = bollinger.bollinger_mavg()
        data['Lower_Band'] = bollinger.bollinger_lband()
        
        # Moving Averages: SMA50 and SMA200
        sma50 = ta.trend.SMAIndicator(close=data['Close'], window=50)
        data['SMA50'] = sma50.sma_indicator()
        sma200 = ta.trend.SMAIndicator(close=data['Close'], window=200)
        data['SMA200'] = sma200.sma_indicator()
        
        return data
    except Exception as e:
        st.error(f"Error calculating indicators: {e}")
        return data

# Step 3: User-Defined Parameters (Profit Target, Risk-Reward Ratio)
def get_user_parameters():
    """
    Get user-defined parameters like profit target and risk-reward ratio.
    """
    profit_target = st.number_input("Enter your profit target percentage (e.g., 2 for 2%):", min_value=0.0, value=2.0)
    risk_reward_ratio = st.number_input("Enter your preferred risk-reward ratio (e.g., 2 for 2:1):", min_value=0.1, value=2.0)
    return profit_target / 100, risk_reward_ratio

# Step 4: Backtest the Strategy
def backtest_strategy(intraday_data, daily_data, profit_target, risk_reward_ratio):
    """
    Perform a backtest of the strategy using both intraday and daily data.
    Updated strategy with enhanced hold messaging and dynamic profit target as a minimum.
    Generates a chart showing buy/sell signals with arrows.
    """
    balance = 10000  # Initial capital
    position = 0
    trade_log = []
    buy_signals = []
    sell_signals = []
    num_trades = 0

    # Check if intraday data is available
    if intraday_data is None or len(intraday_data) == 0:
        st.error("No intraday data available for backtesting.")
        return balance, trade_log

    # Calculate indicators if not already done
    if 'RSI' not in intraday_data.columns:
        intraday_data = calculate_indicators(intraday_data)
    if 'RSI' not in daily_data.columns:
        daily_data = calculate_indicators(daily_data)

    # Use daily data to guide the overall trend and intraday data for trades
    for i in range(1, len(intraday_data)):
        latest_rsi_intraday = intraday_data['RSI'].iloc[i]
        latest_close_intraday = intraday_data['Close'].iloc[i]

        # Add Moving Average Crossover
        sma_short = intraday_data['SMA50'].iloc[i]  # Short-term moving average (50 periods)
        sma_long = intraday_data['SMA200'].iloc[i]  # Long-term moving average (200 periods)

        # Stop-loss and profit target (minimum target, allow trend to continue)
        stop_loss = latest_close_intraday * (1 - (1 / risk_reward_ratio))  # Calculate stop-loss based on risk-reward ratio
        target_price = latest_close_intraday * (1 + profit_target)  # Minimum target price, but trend can continue

        # Buy Condition: RSI < 30 (oversold) + Moving Average Crossover (SMA50 > SMA200)
        if latest_rsi_intraday < 30 and sma_short > sma_long and position == 0:
            # Buy the position based on both RSI and moving average confirmation
            position = balance / latest_close_intraday  # Number of shares we can buy
            balance = 0  # All money is invested
            trade_log.append(f"Buy {position:.2f} shares at ${latest_close_intraday:.2f} on {intraday_data.index[i]}")
            buy_signals.append((intraday_data.index[i], latest_close_intraday))  # Store buy signal for plotting
        
        # Sell Condition: RSI > 70 (overbought) + Moving Average Crossover (SMA50 < SMA200)
        elif latest_rsi_intraday > 70 and sma_short < sma_long and position > 0:
            # Sell the position
            balance = position * latest_close_intraday  # Liquidate the position
            trade_log.append(f"Sell {position:.2f} shares at ${latest_close_intraday:.2f} on {intraday_data.index[i]}")
            sell_signals.append((intraday_data.index[i], latest_close_intraday))  # Store sell signal for plotting
            position = 0  # Reset position after selling

        num_trades += 1

    # Final balance after backtest
    final_balance = balance if position == 0 else position * intraday_data['Close'].iloc[-1]
    
    # Display the results of the backtest
    st.success(f"Final Balance: ${final_balance:.2f}")
    st.info(f"Number of Trades: {num_trades}")
    for log in trade_log:
        st.write(log)

    # Enhanced hold messaging
    latest_rsi_intraday = intraday_data['RSI'].iloc[-1]
    latest_rsi_daily = daily_data['RSI'].iloc[-1]
    if 30 < latest_rsi_intraday < 50:
        st.warning(f"Holding: Possible Buy Signal Forming (RSI near 30) - Intraday RSI: {latest_rsi_intraday:.2f}, Daily RSI: {latest_rsi_daily:.2f}")
    elif 50 < latest_rsi_intraday < 70:
        st.warning(f"Holding: Possible Sell Signal Forming (RSI near 70) - Intraday RSI: {latest_rsi_intraday:.2f}, Daily RSI: {latest_rsi_daily:.2f}")
    else:
        st.info(f"Holding: Market in Neutral State - Intraday RSI: {latest_rsi_intraday:.2f}, Daily RSI: {latest_rsi_daily:.2f}")

    # Generate a chart with buy/sell signals
    plot_signals(intraday_data, buy_signals, sell_signals)
    return final_balance, trade_log

def plot_signals(data, buy_signals, sell_signals):
    """
    Plot the chart showing buy/sell signals using arrows.
    """
    plt.figure(figsize=(14, 8))
    plt.plot(data.index, data['Close'], label='Close Price', color='blue')

    # Plot buy signals with green arrows
    for signal in buy_signals:
        plt.annotate('Buy', xy=(signal[0], signal[1]), xytext=(signal[0], signal[1] + (data['Close'].max() * 0.02)),
                     arrowprops=dict(facecolor='green', shrink=0.05), color='green')

    # Plot sell signals with red arrows
    for signal in sell_signals:
        plt.annotate('Sell', xy=(signal[0], signal[1]), xytext=(signal[0], signal[1] - (data['Close'].max() * 0.02)),
                     arrowprops=dict(facecolor='red', shrink=0.05), color='red')

    plt.title('Price with Buy/Sell Signals')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)
    plt.clf()  # Clear the figure after plotting to prevent overlap

# Step 5: Suggest Next Trade
def suggest_next_trade(intraday_data, daily_data):
    """
    Suggest the next trade (buy/sell) based on the latest indicators from both intraday and daily timeframes.
    """
    if intraday_data is None or daily_data is None or len(intraday_data) == 0 or len(daily_data) == 0:
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
        return f"**Suggested Trade:** Buy at ${latest_close_intraday:.2f} (RSI: {latest_rsi_intraday:.2f} Intraday, {latest_rsi_daily:.2f} Daily)"
    elif latest_rsi_intraday > 60 and latest_close_intraday >= upper_band_intraday and latest_rsi_daily > 60:
        return f"**Suggested Trade:** Sell at ${latest_close_intraday:.2f} (RSI: {latest_rsi_intraday:.2f} Intraday, {latest_rsi_daily:.2f} Daily)"
    else:
        return f"**Hold** (Intraday RSI: {latest_rsi_intraday:.2f}, Daily RSI: {latest_rsi_daily:.2f})"

# Main Streamlit App
def main():
    st.title("📈 Day Trading Analysis and Prediction")

    st.sidebar.header("User Inputs")

    # User Inputs
    ticker = st.sidebar.text_input("Enter the ticker symbol for the stock (e.g., SPY, QQQ, AAPL):", value="AAPL").upper()
    interval = st.sidebar.selectbox("Select the interval for intraday data:", ["1m", "2m", "5m", "15m", "30m", "60m"])
    
    st.sidebar.markdown("---")
    
    # Get user-defined parameters
    profit_target, risk_reward_ratio = get_user_parameters()

    st.sidebar.markdown("---")
    st.sidebar.write("**Note:** Ensure your inputs are correct before running the backtest.")

    if st.button("Run Backtest"):
        with st.spinner("Downloading data..."):
            # Download historical data for both intraday and daily timeframes
            intraday_data, daily_data = download_data(ticker, interval)
        
        if intraday_data is None or daily_data is None:
            st.error("No valid data found for the specified ticker or interval.")
        else:
            with st.spinner("Calculating indicators and running backtest..."):
                # Calculate indicators for both timeframes
                intraday_data = calculate_indicators(intraday_data)
                daily_data = calculate_indicators(daily_data)

                # Run backtest
                final_balance, trade_log = backtest_strategy(intraday_data, daily_data, profit_target, risk_reward_ratio)

                # Suggest the next trade based on current indicators
                next_trade = suggest_next_trade(intraday_data, daily_data)
                st.markdown(next_trade)

                # Optionally, display the dataframes
                with st.expander("Show Intraday Data"):
                    st.dataframe(intraday_data.tail())

                with st.expander("Show Daily Data"):
                    st.dataframe(daily_data.tail())

if __name__ == "__main__":
    main()
