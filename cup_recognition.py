import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import mplfinance as mpf
from datetime import datetime, timedelta
import streamlit as st

# Suppress warnings (optional)
import warnings
warnings.filterwarnings('ignore')

# Step 1: Data Collection and Preparation

def fetch_stock_data(ticker, start_date, end_date):
    """
    Fetch historical stock data using yfinance.
    """
    data = yf.download(ticker, start=start_date, end=end_date)
    data.dropna(inplace=True)
    return data

def create_windows(data, window_size):
    """
    Create sliding windows of data.
    """
    windows = []
    for i in range(len(data) - window_size + 1):
        window = data[i:i+window_size].copy()
        windows.append(window)
    return windows

# Step 2: Data Labeling

def label_windows(windows):
    """
    Label windows as containing a cup and handle pattern or not.
    """
    labels = []
    for window in windows:
        label = detect_cup_and_handle_in_window(window)
        labels.append(label)
    return np.array(labels)

def detect_cup_and_handle_in_window(window):
    """
    Detects a cup and handle pattern in a window using heuristic rules.
    """
    close_prices = window['Close'].values
    volume = window['Volume'].values

    min_cup_length = 10
    max_cup_length = 60
    min_handle_length = 5
    max_handle_length = 20
    min_depth = 0.1
    max_depth = 0.5
    handle_max_retrace = 0.5

    cup_found = False
    for i in range(min_cup_length, min(max_cup_length, len(close_prices) - min_handle_length)):
        left_peak = close_prices[0]
        right_peak = close_prices[i]
        cup_bottom = np.min(close_prices[:i+1])
        bottom_index = np.argmin(close_prices[:i+1])

        if bottom_index == 0 or bottom_index == i:
            continue

        peak_average = (left_peak + right_peak) / 2
        depth = (peak_average - cup_bottom) / peak_average

        if depth < min_depth or depth > max_depth:
            continue

        for j in range(i + min_handle_length, min(i + max_handle_length, len(close_prices))):
            handle = close_prices[i+1:j+1]
            handle_max = np.max(handle)
            handle_min = np.min(handle)

            if handle_max > peak_average:
                continue

            handle_retrace = (handle_max - handle_min) / (peak_average - cup_bottom)
            if handle_retrace > handle_max_retrace:
                continue

            cup_found = True
            break

        if cup_found:
            break

    return 1 if cup_found else 0

# Step 3: Feature Engineering

def preprocess_windows(windows):
    """
    Preprocess windows by normalizing the data.
    """
    X = []
    for window in windows:
        prices = window['Adj Close'].values
        normalized_prices = (prices - prices.mean()) / prices.std()
        X.append(normalized_prices)
    X = np.array(X)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X

# Step 4: Model Building

def build_cnn_model(input_shape):
    """
    Build a CNN model for time series classification.
    """
    model = models.Sequential()
    model.add(layers.Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Conv1D(64, kernel_size=3, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Step 7: Prediction on New Data

def predict_on_new_data(model, data, window_size):
    """
    Use the trained model to predict patterns on new data.
    """
    windows = create_windows(data, window_size)
    X_new = preprocess_windows(windows)

    predictions_prob = model.predict(X_new, verbose=0)
    predictions = (predictions_prob > 0.5).astype("int32")
    return predictions, windows

# Step 5: Streamlit App Code

st.title("Cup and Handle Pattern Detection")
ticker = st.text_input("Enter the stock ticker symbol (e.g., AAPL, NVDA):", "AAPL")

start_date = '2010-01-01'
end_date = '2023-12-31'
window_size = 80

# Fetch data
data = fetch_stock_data(ticker, start_date, end_date)
st.write(f"🏗️  Fetching stock data for {ticker}... Hold on tight, weâ€™re diving into {len(data)} rows of financial history for {ticker}!")
# st.write(f"Fetched {len(data)} rows of data for {ticker}.")

# Create windows
windows = create_windows(data, window_size)
st.write(f"🔍 Scanning the data... We just crafted {len(windows)} windows of opportunity, each with a size of {window_size} days. Time to dig deep!")
# st.write(f"Created {len(windows)} windows of size {window_size}.")


# Label windows
labels = label_windows(windows)
positive_samples = labels.sum()
negative_samples = len(labels) - positive_samples
st.write(f"💡 Pattern detection at work... Weâ€™ve uncovered {positive_samples} potential cup and handle formations out of {len(labels)} windows. The hunt is on!")
# st.write(f"Labeled windows. Positive samples: {positive_samples}, Negative samples: {negative_samples}")

# Preprocess windows
X = preprocess_windows(windows)
y = labels

# Split data into training, validation, and test sets
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.15 / 0.85, random_state=42, stratify=y_temp)

input_shape = (X_train.shape[1], X_train.shape[2])
model = build_cnn_model(input_shape)

# Train model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val), verbose=0)

# Predict on new data (last year's data)
today = datetime.today()
one_year_ago = today - timedelta(days=365)
new_start_date = one_year_ago.strftime('%Y-%m-%d')
new_end_date = today.strftime('%Y-%m-%d')

new_data = fetch_stock_data(ticker, new_start_date, new_end_date)
predictions, new_windows = predict_on_new_data(model, new_data, window_size)

# Get indices where cup and handle patterns are detected
pattern_indices = np.where(predictions == 1)[0]

# Display last detected pattern
if len(pattern_indices) > 0:
    last_idx = pattern_indices[-1]
    window = new_windows[last_idx]
    dates = window.index

    # Format the dates without times
    start_date_formatted = dates[0].strftime('%Y-%m-%d')
    end_date_formatted = dates[-1].strftime('%Y-%m-%d')
    
    price_min = window['Low'].min()
    price_max = window['High'].max()
    
    # st.write(f"Last detected cup and handle pattern from {start_date_formatted} to {end_date_formatted}")
    # st.write(f"Price range: {price_min:.2f} to {price_max:.2f}")
    st.write(f"🎯 Jackpot! The last cup and handle pattern emerged between {start_date_formatted} and {end_date_formatted}.")
    st.write(f"📈 Ready for the numbers? Price danced from {price_min:.2f} to {price_max:.2f}.")





    
    candlestick_data = window[['Open', 'High', 'Low', 'Close']].copy()

    # Create a matplotlib figure to pass to mplfinance
    fig, ax = plt.subplots(figsize=(10, 6))

    # Use mplfinance to plot on the figure's axis
    mpf.plot(candlestick_data, type='candle', style='yahoo', ax=ax)

    # Display the plot using Streamlit's pyplot
    st.pyplot(fig)

else:
    st.write("No cup and handle pattern detected in the new data.")
