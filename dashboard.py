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
    # Ensure the index is datetime
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])  # Convert 'Date' to datetime if not already
    stock_data.set_index('Date', inplace=True)  # Set 'Date' as index if not already

    potential_cups = []  # Assuming this is a list to store potential cup data
    handles = []
    
    # Your logic to identify cups goes here...
    for idx, row in potential_cups.iterrows():
        # Access the date directly from the DataFrame's index
        cup_date = row.name  # This should be a timestamp now
        
        # Use cup_date to calculate the handle start date
        handle_start_date = cup_date + pd.DateOffset(days=10)  # Handles typically form shortly after the cup

        # Check if handle_start_date is in the DataFrame index
        if handle_start_date in stock_data.index:
            handle_data = stock_data.loc[handle_start_date:handle_start_date + pd.DateOffset(days=20)]  # Handle range
            
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

