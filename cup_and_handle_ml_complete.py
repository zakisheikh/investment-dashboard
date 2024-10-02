# cup_and_handle_ml_complete.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import sys
import io
import mplfinance as mpf
from datetime import datetime, timedelta


# Suppress warnings (optional)
import warnings
warnings.filterwarnings('ignore')

# Step 1: Data Collection and Preparation

def fetch_stock_data(ticker, start_date, end_date):
    """
    Fetch historical stock data using yfinance.

    Parameters:
    - ticker (str): Stock ticker symbol.
    - start_date (str): Start date in 'YYYY-MM-DD' format.
    - end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
    - data (DataFrame): Historical stock data.
    """
    data = yf.download(ticker, start=start_date, end=end_date)
    data.dropna(inplace=True)
    return data

def create_windows(data, window_size):
    """
    Create sliding windows of data.

    Parameters:
    - data (DataFrame): Stock data.
    - window_size (int): Size of the window.

    Returns:
    - windows (list): List of DataFrame windows.
    """
    windows = []
    for i in range(len(data) - window_size + 1):
        window = data[i:i+window_size].copy()
        windows.append(window)
    return windows  # Return the list directly

# Step 2: Data Labeling

def label_windows(windows):
    """
    Label windows as containing a cup and handle pattern or not.

    Parameters:
    - windows (ndarray): Array of windows.

    Returns:
    - labels (ndarray): Array of labels (1 for pattern, 0 for no pattern).
    """
    labels = []
    for window in windows:
        label = detect_cup_and_handle_in_window(window)
        labels.append(label)
    return np.array(labels)

def detect_cup_and_handle_in_window(window):
    """
    Detects a cup and handle pattern in a window using heuristic rules.

    Parameters:
    - window (DataFrame): Window of stock data.

    Returns:
    - label (int): 1 if pattern is detected, 0 otherwise.
    """
    close_prices = window['Close'].values
    volume = window['Volume'].values

    # Parameters for cup and handle detection
    min_cup_length = 10
    max_cup_length = 60
    min_handle_length = 5
    max_handle_length = 20
    min_depth = 0.1  # Minimum depth of the cup (10%)
    max_depth = 0.5  # Maximum depth of the cup (50%)
    handle_max_retrace = 0.5  # Handle retracement should not exceed 50% of cup depth

    # Step 1: Identify the cup
    cup_found = False
    for i in range(min_cup_length, min(max_cup_length, len(close_prices) - min_handle_length)):
        left_peak = close_prices[0]
        right_peak = close_prices[i]
        cup_bottom = np.min(close_prices[:i+1])
        bottom_index = np.argmin(close_prices[:i+1])

        # Ensure the bottom is not at the edges
        if bottom_index == 0 or bottom_index == i:
            continue

        # Calculate cup depth
        peak_average = (left_peak + right_peak) / 2
        depth = (peak_average - cup_bottom) / peak_average

        if depth < min_depth or depth > max_depth:
            continue

        # Step 2: Identify the handle
        for j in range(i + min_handle_length, min(i + max_handle_length, len(close_prices))):
            handle = close_prices[i+1:j+1]
            handle_max = np.max(handle)
            handle_min = np.min(handle)

            # Handle should not exceed the peaks
            if handle_max > peak_average:
                continue

            # Handle retracement should not exceed 50% of cup depth
            handle_retrace = (handle_max - handle_min) / (peak_average - cup_bottom)
            if handle_retrace > handle_max_retrace:
                continue

            # If all conditions are met, pattern is found
            cup_found = True
            break

        if cup_found:
            break

    label = 1 if cup_found else 0
    return label

# Step 3: Feature Engineering

def preprocess_windows(windows):
    """
    Preprocess windows by normalizing the data.

    Parameters:
    - windows (ndarray): Array of windows.

    Returns:
    - X (ndarray): Preprocessed feature array.
    """
    X = []
    for window in windows:
        # Use adjusted close prices
        prices = window['Adj Close'].values
        # Normalize prices
        normalized_prices = (prices - prices.mean()) / prices.std()
        X.append(normalized_prices)
    X = np.array(X)
    # Reshape for CNN input (samples, time_steps, features)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X

# Step 4: Model Building

def build_cnn_model(input_shape):
    """
    Build a CNN model for time series classification.

    Parameters:
    - input_shape (tuple): Shape of the input data.

    Returns:
    - model (Model): Compiled Keras model.
    """
    model = models.Sequential()
    model.add(layers.Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Conv1D(64, kernel_size=3, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))  # Binary classification
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Step 5: Model Training

def train_model(model, X_train, y_train, X_val, y_val):
    """
    Train the CNN model.

    Parameters:
    - model (Model): Compiled Keras model.
    - X_train (ndarray): Training features.
    - y_train (ndarray): Training labels.
    - X_val (ndarray): Validation features.
    - y_val (ndarray): Validation labels.

    Returns:
    - history (History): Training history.
    """
    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_data=(X_val, y_val),
        verbose=0
    )
    return history

# Step 6: Model Evaluation

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on the test set.

    Parameters:
    - model (Model): Trained Keras model.
    - X_test (ndarray): Test features.
    - y_test (ndarray): Test labels.

    Returns:
    - None
    """

    # Suppress stdout temporarily
    original_stdout = sys.stdout
    sys.stdout = io.StringIO()

    # Perform prediction
    y_pred_prob = model.predict(X_test)

    # Restore stdout
    sys.stdout = original_stdout
    
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype("int32")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# Step 7: Prediction on New Data

def predict_on_new_data(model, data, window_size):
    """
    Use the trained model to predict patterns on new data.

    Parameters:
    - model (Model): Trained Keras model.
    - data (DataFrame): New stock data.
    - window_size (int): Size of the window.

    Returns:
    - predictions (ndarray): Array of predictions.
    """
    windows = create_windows(data, window_size)
    X_new = preprocess_windows(windows)

    # Suppress stdout temporarily
    original_stdout = sys.stdout
    sys.stdout = io.StringIO()
    
    predictions_prob = model.predict(X_new, verbose=0)

        # Restore stdout
    sys.stdout = original_stdout
    
    predictions = (predictions_prob > 0.5).astype("int32")
    return predictions, windows

# Step 8: Main Execution

if __name__ == '__main__':
    # Parameters
    ticker = input("Enter the stock ticker symbol (e.g., AAPL, NVDA): ")
    start_date = '2010-01-01'
    end_date = '2023-12-31'
    window_size = 60  # Adjust based on expected pattern length

    # Fetch data
    data = fetch_stock_data(ticker, start_date, end_date)
    print(f"Fetched {len(data)} rows of data for {ticker}.")

    # Create windows
    windows = create_windows(data, window_size)
    print(f"Created {len(windows)} windows of size {window_size}.")

    # Label windows
    labels = label_windows(windows)
    positive_samples = labels.sum()
    negative_samples = len(labels) - positive_samples
    print(f"Labeled windows. Positive samples: {positive_samples}, Negative samples: {negative_samples}")

    # Handle class imbalance if necessary
    # For example, you can undersample the majority class or use class weights
    # For simplicity, we'll proceed without addressing imbalance in this example

    # Preprocess windows
    X = preprocess_windows(windows)
    y = labels

    # Split data into training, validation, and test sets
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.15 / 0.85, random_state=42, stratify=y_temp)

    print(f"Training samples: {len(y_train)}, Validation samples: {len(y_val)}, Test samples: {len(y_test)}")

    # Build model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_cnn_model(input_shape)

    # Train model
    history = train_model(model, X_train, y_train, X_val, y_val)

    # Evaluate model
    evaluate_model(model, X_test, y_test)

    # Plot training history
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend()

    plt.show()

    # Save the model
    model.save('cup_and_handle_cnn_model.h5')
    print("Model saved as 'cup_and_handle_cnn_model.h5'.")

    # Predict on new data
    # For demonstration, we'll use recent data from the last year
    # Get today's date
    today = datetime.today()

    # Calculate the date one year ago from today
    one_year_ago = today - timedelta(days=365)

    # Format the dates as strings in 'YYYY-MM-DD' format
    new_start_date = one_year_ago.strftime('%Y-%m-%d')
    new_end_date = today.strftime('%Y-%m-%d')
    new_data = fetch_stock_data(ticker, new_start_date, new_end_date)
    predictions, new_windows = predict_on_new_data(model, new_data, window_size)

    # Find windows where pattern is predicted
    pattern_indices = np.where(predictions == 1)[0]
    print(f"Detected {len(pattern_indices)} potential cup and handle patterns in new data.")

    # Plot detected patterns
for i, idx in enumerate(pattern_indices, start=1):
    window = new_windows[idx]
    dates = window.index
    date_range = f"{dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}"
    
    # Calculate the price range
    price_min = window['Low'].min()
    price_max = window['High'].max()
    price_range = f"Price range: {price_min:.2f} to {price_max:.2f}"
    
    # Prepare the data for candlestick chart
    candlestick_data = window[['Open', 'High', 'Low', 'Close']].copy()
    candlestick_data.index = dates

    # Define the filename for the last pattern's plot
    if i == len(pattern_indices):
        plot_filename = 'latest_cup_handle_pattern.png'
    else:
        plot_filename = f'cup_handle_pattern_{i}.png'  # e.g., pattern_1.png, pattern_2.png, etc.
    
    # Plot the candlestick chart
    mpf.plot(
        candlestick_data, 
        type='candle', 
        title=f"Cup and Handle Pattern {i}/{len(pattern_indices)}\n{date_range}\n{price_range}", 
        style='yahoo'
    )
    
    # Conditional prompt
    if i < len(pattern_indices):
        input("Press Enter after closing the plot to view the next pattern...")
    else:
        print("This was the last detected cup and handle pattern.")
        input("Press Enter to exit the program.")


# After the plotting loop
print("\nAll detected cup and handle patterns have been reviewed.")
