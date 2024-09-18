import numpy as np
import pandas as pd

def detect_cup_handle(data, lookback=100):
    rolling_min = data['Close'].rolling(window=lookback).min()
    rolling_max = data['Close'].rolling(window=lookback).max()
    bottom_found = (data['Close'] <= rolling_min * 1.02)  # Cup condition
    handle_threshold = rolling_max * 0.95  # Handle condition
    handle_found = (data['Close'] < handle_threshold)
    cup_handle = bottom_found & handle_found
    return cup_handle

def detect_flat_base(data, lookback=50, threshold=0.03):
    max_price = data['Close'].rolling(window=lookback).max()
    min_price = data['Close'].rolling(window=lookback).min()
    base_range = max_price - min_price
    flat_base = (base_range / min_price) < threshold
    return flat_base

def detect_breakout(data, lookback=30):
    rolling_high = data['Close'].rolling(window=lookback).max()
    breakout = (data['Close'] > rolling_high.shift(1))
    return breakout

def calculate_composite_rating(stock_data, sp500_data):
    stock_return = stock_data['Close'].pct_change().cumsum()
    sp500_return = sp500_data['Close'].pct_change().cumsum()
    relative_strength = stock_return - sp500_return
    composite_rating = (relative_strength.mean() * 100).clip(0, 100)
    return composite_rating
