# pattern_recognition.py
import pandas as pd

def detect_cup_with_handle(stock_data):
    # Step 1: Calculate the moving average (50-day for smoothing out short-term trends)
    stock_data['SMA_50'] = stock_data['Close'].rolling(window=50).mean()
    
    # Step 2: Calculate the depth of the potential cup
    stock_data['peak'] = stock_data['Close'].rolling(window=30).max()  # Finding peaks (use 30-day window)
    stock_data['trough'] = stock_data['Close'].rolling(window=30).min()  # Finding troughs
    stock_data['depth'] = (stock_data['peak'] - stock_data['trough']) / stock_data['peak']  # Depth of the cup
    
    # Step 3: Identify potential cups (U-shaped bottoms, with a depth not exceeding 33%)
    potential_cups = stock_data[(stock_data['depth'] <= 0.33) & (stock_data['depth'] >= 0.15)]
    
    # Step 4: Check duration of the cup (between 7 to 65 weeks ~ 35 to 325 trading days)
    potential_cups['cup_duration'] = potential_cups['Close'].rolling(window=50).apply(lambda x: len(x))  # Rough estimate of duration
    potential_cups = potential_cups[(potential_cups['cup_duration'] >= 35) & (potential_cups['cup_duration'] <= 325)]
    
    # Step 5: Check for handle formation
    handles = []
    for idx, row in potential_cups.iterrows():
        # Handle should form on the right side of the cup and be a small pullback (8%-12%) in the upper half of the cup
        handle_start_idx = idx + 10  # Handles typically form shortly after the cup
        if handle_start_idx < len(stock_data):
            handle_data = stock_data.iloc[handle_start_idx:handle_start_idx + 20]  # 1-2 week handle range
            max_close = row['Close']
            min_close = handle_data['Close'].min()
            pullback = (max_close - min_close) / max_close
            
            if 0.08 <= pullback <= 0.12 and min_close > row['SMA_50']:  # Ensure handle is in upper half of cup
                handles.append((row.name, handle_data))
    
    return potential_cups, handles  # Return both the cups and handles detected
