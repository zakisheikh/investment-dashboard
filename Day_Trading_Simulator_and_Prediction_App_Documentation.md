
# Day Trading Simulator and Prediction App

This document provides an overview of the Day Trading Simulator and Prediction App built using **Streamlit**, **yfinance**, and **ta-lib** for technical analysis. The goal of this app is to simulate a simple day trading strategy, allowing the user to backtest trades based on technical indicators such as RSI, MACD, Bollinger Bands, and Moving Averages.

## Key Features

### 1. Downloading Historical Data
The app allows the user to download both **intraday** and **daily** historical market data for a given stock ticker using the `yfinance` API. This functionality supports multiple intervals such as `1m`, `2m`, `5m`, `15m`, `30m`, and `60m` (1 minute, 5 minute, etc.). The app ensures that the correct date ranges are used based on the interval selected due to limitations in Yahoo Finance’s API.

### 2. Calculating Technical Indicators
Using the **`ta` library**, the app calculates key technical indicators, including:
- **RSI (Relative Strength Index):** Useful for identifying overbought or oversold conditions.
- **MACD (Moving Average Convergence Divergence):** Used for trend following and momentum.
- **Bollinger Bands:** Show volatility and potential price breakouts.
- **SMA (Simple Moving Averages):** Includes 50-period and 200-period moving averages to determine overall trend strength.

### 3. User-Defined Parameters
Users can define the following parameters for backtesting their trading strategy:
- **Profit Target:** A percentage that represents the profit target for each trade.
- **Risk-Reward Ratio (RRR):** The ratio of potential profit to the risk that the user is willing to take.

### 4. Strategy Backtesting
The app performs a backtest of the day trading strategy by analyzing both intraday and daily data. It evaluates the following:
- **Buy Signals:** Identified based on RSI values below 30 and a moving average crossover (SMA50 > SMA200).
- **Stop Loss:** Calculated based on 50% of the profit target, protecting against excessive losses.
- **Trailing Stop Loss:** Adjusts dynamically as the price moves in favor of the trade.
- **Profit Target:** Sells the position when the price meets or exceeds the profit target.

The app also generates a visual chart that shows the price data along with **buy** and **sell** signals.

### 5. Trade Suggestions
Once the backtest is completed, the app provides the user with a suggestion for the next trade based on the latest technical indicators from both intraday and daily timeframes.

### 6. Interactive Visuals and Logs
The app displays both the raw data (intraday and daily) and trade logs, showing each buy and sell action. It includes enhanced messages about holding conditions based on RSI values.

## How It Works
1. **Step 1:** The user selects a ticker symbol (e.g., AAPL, SPY) and a preferred intraday interval (e.g., 5-minute).
2. **Step 2:** The app downloads historical data using the `yfinance` API.
3. **Step 3:** The app calculates technical indicators such as RSI, MACD, Bollinger Bands, and moving averages.
4. **Step 4:** The user inputs their profit target and risk/reward ratio.
5. **Step 5:** The app backtests the strategy, applying buy and sell signals based on the user's inputs and displays the final balance and trade log.
6. **Step 6:** The app suggests the next trade based on the latest market data.

### Example of Trade Logic
- **Buy Signal:** RSI below 30 and SMA50 crossing above SMA200.
- **Stop Loss:** Automatically set at 50% of the profit target.
- **Trailing Stop Loss:** Adjusts dynamically as the trade becomes profitable.
- **Sell Signal:** Either the profit target is hit, or the price drops to the stop loss level.

## Note
No code snippets are shown in this document. Please refer to the repository for the full implementation.
