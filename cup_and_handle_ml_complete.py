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

def get_analysis_type():
    while True:
        analysis_type = input('Select analysis type ("short-term" or "long-term"): ').strip().lower()
        if analysis_type in ['short-term', 'long-term']:
            return analysis_type
        else:
            print('Invalid input. Please enter "short-term" or "long-term".')

def get_ticker_symbol():
    while True:
        ticker = input('Enter the ticker symbol (e.g., "AAPL", "MSFT"): ').strip().upper()
        if ticker:
            return ticker
        else:
            print('Invalid input. Please enter a valid ticker symbol.')

# Step 8: Main Execution

if __name__ == '__main__':
    # Get analysis type from user
    analysis_type = get_analysis_type()

    # Get ticker symbol from user
    ticker = get_ticker_symbol()

    # Set parameters based on analysis type
    if analysis_type == 'short-term':
        interval = '1d'  # Daily data
        window_size = 60
        min_cup_length = 10
        max_cup_length = 60
        min_handle_length = 5
        max_handle_length = 20
        min_depth = 0.1
        max_depth = 0.5
        handle_max_retrace = 0.5

        # Calculate dynamic dates
        end_date = datetime.today()
        start_date = end_date - timedelta(days=365)  # Fetch past 1 year of data
    elif analysis_type == 'long-term':
        interval = '1wk'  # Weekly data
        window_size = 65
        min_cup_length = 7
        max_cup_length = 65
        min_handle_length = 1
        max_handle_length = 4
        min_depth = 0.12
        max_depth = 0.33
        handle_max_retrace = 0.12

        # Calculate dynamic dates
        end_date = datetime.today()
        start_date = end_date - timedelta(weeks=130)  # Fetch past 2.5 years of data
    else:
        raise ValueError('Invalid analysis type selected.')

    # Convert dates to strings in 'YYYY-MM-DD' format
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    # Fetch data
    data = fetch_stock_data(ticker, start_date_str, end_date_str, interval)
    if data.empty:
        print(f"No data fetched for ticker '{ticker}'. Please check the ticker symbol and try again.")
        sys.exit(1)
    print(f"Fetched {len(data)} rows of data for {ticker} from {start_date_str} to {end_date_str}.")

    # Label windows
    labels = label_windows(windows, min_cup_length, max_cup_length, min_handle_length,
                           max_handle_length, min_depth, max_depth, handle_max_retrace)
    positive_samples = labels.sum()
    negative_samples = len(labels) - positive_samples
    print(f"Labeled windows. Positive samples: {positive_samples}, Negative samples: {negative_samples}")

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
    model_filename = f'cup_and_handle_cnn_model_{ticker}.keras'
    model.save(model_filename)
    print(f"Model saved as '{model_filename}'.")

    # Predict on new data
    new_start_date = '2023-09-30'
    new_end_date = '2023-12-31'
    new_data = fetch_stock_data(ticker, new_start_date, new_end_date, interval)
    if new_data.empty:
        print(f"No new data fetched for ticker '{ticker}'. Please check the ticker symbol and try again.")
        sys.exit(1)
    predictions, new_windows = predict_on_new_data(model, new_data, window_size, min_cup_length,
                                                   max_cup_length, min_handle_length, max_handle_length,
                                                   min_depth, max_depth, handle_max_retrace)

    # Find windows where pattern is predicted
    pattern_indices = np.where(predictions == 1)[0]
    print(f"Detected {len(pattern_indices)} potential cup and handle patterns in new data.")

    # Initialize list to store pattern details
    pattern_details = []

    for idx in pattern_indices:
        window = new_windows[idx]
        dates = window.index
        start_date = dates[0].strftime('%Y-%m-%d')
        end_date = dates[-1].strftime('%Y-%m-%d')
        prices = window['Adj Close']
        min_price = prices.min()
        max_price = prices.max()

        print(f"\nPattern detected in {ticker} from {start_date} to {end_date} (Index {idx})")
        print(f"Price range: ${min_price:.2f} - ${max_price:.2f}")

        # Append details to list
        pattern_details.append({
            'Ticker': ticker,
            'Start Date': start_date,
            'End Date': end_date,
            'Min Price': min_price,
            'Max Price': max_price
        })

        # Plot the detected pattern
        plt.figure(figsize=(10, 5))
        plt.plot(dates, prices)
        plt.title(f"Cup and Handle Pattern Detected in {ticker}\n{start_date} to {end_date}")
        plt.xlabel('Date')
        plt.ylabel('Adjusted Close Price')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        # Optionally, save the plot
        # plot_filename = f"{ticker}_pattern_{idx}_{start_date}_to_{end_date}.png"
        # plt.savefig(plot_filename)
        # print(f"Plot saved as {plot_filename}")

    # Create a summary DataFrame
    if pattern_details:
        pattern_df = pd.DataFrame(pattern_details)
        print("\nSummary of Detected Patterns:")
        print(pattern_df)
    else:
        print("No patterns detected in the new data.")

