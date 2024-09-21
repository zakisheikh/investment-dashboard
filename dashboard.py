# dashboard.py
import pandas as pd
import yfinance as yf
import visualization
import numpy as np

def fetch_stock_data(symbol):
    stock_data = yf.download(symbol, start="2020-01-01", end="2024-01-01")
    stock_data.reset_index(inplace=True)
    stock_data['Date'] = stock_data['Date'].dt.date  # Keep only the date part
    return stock_data

def detect_cup_and_handle(stock_data):
    cups = []
    handles = []
    
    # Parameters
    min_depth = 0.1  # Minimum cup depth as a percentage of the peak
    max_handle_depth = 0.05  # Maximum handle depth as a percentage of the peak
    min_handle_length = 3  # Minimum handle length in days

    # Identify potential cup bottoms and peaks
    for i in range(2, len(stock_data) - 2):
        if stock_data['Close'][i] < stock_data['Close'][i - 1] and stock_data['Close'][i] < stock_data['Close'][i + 1]:  # Local minimum
            cup_bottom = stock_data['Close'][i]
            cup_start = i - 2
            cup_end = i + 2
            
            if (cup_end < len(stock_data) and stock_data['Close'][cup_start] > cup_bottom * (1 + min_depth)):
                # We have a potential cup, now check for the handle
                handle_start = cup_end + 1
                while handle_start < len(stock_data):
                    if stock_data['Close'][handle_start] < cup_bottom * (1 + max_handle_depth):
                        handle_end = handle_start
                        while handle_end < len(stock_data) and stock_data['Close'][handle_end] < stock_data['Close'][handle_end - 1]:
                            handle_end += 1
                        if handle_end - handle_start >= min_handle_length:
                            # Valid handle found
                            cups.append({
                                'Start_Date': stock_data['Date'][cup_start],
                                'Bottom_Date': stock_data['Date'][i],
                                'End_Date': stock_data['Date'][cup_end],
                                'Handle_End_Date': stock_data['Date'][handle_end - 1],
                                'Bottom_Price': cup_bottom,
                                'Breakout_Date': stock_data['Date'][handle_end],
                                'Breakout_Price': stock_data['Close'][handle_end]
                            })
                            handles.append((handle_start, stock_data.iloc[handle_start:handle_end]))
                        break  # No need to check further for this cup
    
    cups_df = pd.DataFrame(cups)
    return cups_df, handles

def main(symbol):
    stock_data = fetch_stock_data(symbol)
    cups, handles = detect_cup_and_handle(stock_data)
    visualization.plot_stock_data_with_pattern(stock_data, cups, handles)
    visualization.summarize_cup_and_handle()

if __name__ == "__main__":
    main("NVDA")  # Replace "AAPL" with any stock symbol you want to analyze
