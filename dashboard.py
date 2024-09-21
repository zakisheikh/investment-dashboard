# dashboard.py
import pandas as pd
import yfinance as yf
import visualization
import numpy as np
import sys
from datetime import datetime, timedelta

def fetch_stock_data(symbol):
    # Calculate the date range for the last two years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*2)  # Two years of data
    
    stock_data = yf.download(symbol, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
    stock_data.reset_index(inplace=True)
    stock_data['Date'] = stock_data['Date'].dt.date  # Keep only the date part
    print(stock_data.head())  # Debugging line to check the fetched data
    return stock_data

def detect_cup_and_handle(stock_data):
    # Step 1: Calculate the moving average (50-day for smoothing out short-term trends)
    stock_data['SMA_50'] = stock_data['Close'].rolling(window=50).mean()
    
    # Step 2: Calculate the depth of the potential cup
    stock_data['peak'] = stock_data['Close'].rolling(window=30).max()  # Finding peaks (use 30-day window)
    stock_data['trough'] = stock_data['Close'].rolling(window=30).min()  # Finding troughs
    stock_data['depth'] = (stock_data['peak'] - stock_data['trough']) / stock_data['peak']  # Depth of the cup
    
    # Step 3: Identify potential cups (U-shaped bottoms, with a depth not exceeding 33%)
    potential_cups = stock_data[(stock_data['depth'] <= 0.33) & (stock_data['depth'] >= 0.15)].copy()
    
    # Step 4: Check duration of the cup (between 7 to 65 weeks ~ 35 to 325 trading days)
    potential_cups['cup_duration'] = potential_cups['Close'].rolling(window=50).apply(lambda x: len(x))  # Rough estimate of duration
    potential_cups = potential_cups[(potential_cups['cup_duration'] >= 35) & (potential_cups['cup_duration'] <= 325)]
    
    # Step 5: Check for handle formation
    handles = []
    for idx, row in potential_cups.iterrows():
        # Access the date directly from stock_data's index
        cup_date = stock_data.index[row.name]  # Get the actual date for the current row
        
        # Add the days to the date to get the start of the handle
        handle_start_date = cup_date + pd.DateOffset(days=10)  # Handles typically form shortly after the cup
        
        # Check if handle_start_date is in the DataFrame index
        if handle_start_date in stock_data.index:
            handle_data = stock_data.loc[handle_start_date:handle_start_date + pd.DateOffset(days=20)]  # 1-2 week handle range
            
            # Extract the cup bottom to calculate pullback
            cup_bottom = row['trough']  # Ensure this is defined properly
            
            max_close = row['Close']
            min_close = handle_data['Close'].min()
            pullback = (max_close - min_close) / max_close
            
            # Ensure pullback conditions are met
            if 0.08 <= pullback <= 0.12 and min_close > row['SMA_50']:  # Ensure handle is in upper half of cup
                handles.append((row.name, handle_data))

    return potential_cups, handles  # Return both the cups and handles detected

import sys

def main(symbol):
    # Fetch stock data
    stock_data = fetch_stock_data(symbol)
    
    if stock_data.empty:
        print(f"No data found for symbol: {symbol}")
        return
    
    print("Fetching data completed. Detecting patterns...")  # Debugging line
    cups, handles = detect_cup_and_handle(stock_data)
    
    if cups.empty:
        print(f"No Cup and Handle patterns detected for {symbol}.")
    else:
        print(f"Detected {len(cups)} Cup and Handle patterns for {symbol}.")
        # Debugging output of detected cups and handles
        print("Detected Cups:")
        print(cups)
        
        print("Detected Handles:")
        for handle in handles:
            print(f"Handle at: {handle[0]} with data: {handle[1]}")
        
        # Visualization part
        visualization.plot_stock_data_with_pattern(stock_data, cups, handles)
        visualization.summarize_cup_and_handle()
        print("Visualization and summary completed.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python dashboard.py <STOCK_SYMBOL>")
    else:
        stock_symbol = sys.argv[1].strip().upper()  # Read the stock symbol from command line
        main(stock_symbol)

