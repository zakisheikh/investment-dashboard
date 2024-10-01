import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import models, layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime, timedelta
import sys

# Function to get analysis type
def get_analysis_type():
    print("Prompting for analysis type...")
    while True:
        analysis_type = input('Select analysis type ("short-term" or "long-term"): ').strip().lower()
        if analysis_type in ['short-term', 'long-term']:
            print(f"Analysis type selected: {analysis_type}")
            return analysis_type
        else:
            print('Invalid input. Please enter "short-term" or "long-term".')

# Function to get ticker symbol
def get_ticker_symbol():
    print("Prompting for ticker symbol...")
    while True:
        ticker = input('Enter the ticker symbol (e.g., "AAPL", "MSFT"): ').strip().upper()
        if ticker:
            print(f"Ticker symbol entered: {ticker}")
            return ticker
        else:
            print('Invalid input. Please enter a valid ticker symbol.')

# Function to fetch stock data
def fetch_stock_data(ticker, start_date, end_date, interval):
    print(f"Fetching data for {ticker} from {start_date} to {end_date} with interval '{interval}'...")
    data = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)
    if data.empty:
        print(f"Warning: No data fetched for ticker '{ticker}'.")
    data.dropna(inplace=True)
    return data

# Function to create windows
def create_windows(data, window_size):
    print(f"Creating windows of size {window_size}...")
    windows = []
    for i in range(len(data) - window_size + 1):
        window = data.iloc[i:i+window_size]
        windows.append(window)
    print(f"Created {len(windows)} windows.")
    return windows

# Function to preprocess windows
def preprocess_windows(windows):
    print("Preprocessing windows...")
    X = []
    for window in windows:
        # Normalize the 'Adj Close' prices
        prices = window['Adj Close'].values
        prices_scaled = (prices - np.min(prices)) / (np.max(prices) - np.min(prices))
        X.append(prices_scaled)
    X = np.array(X)
    X = X.reshape(X.shape[0], X.shape[1], 1)  # Adding channel dimension for CNN
    print(f"Preprocessed windows into shape: {X.shape}")
    return X

# Function to detect cup and handle in a window
def detect_cup_and_handle_in_window(window, min_cup_length, max_cup_length, min_handle_length,
                                    max_handle_length, min_depth, max_depth, handle_max_retrace):
    # Implement actual cup and handle detection logic here
    # This is a simplistic example and should be replaced with robust logic
    prices = window['Adj Close'].values
    try:
        min_index = np.argmin(prices)
        left = prices[:min_index]
        right = prices[min_index:]
        if len(left) < min_cup_length or len(right) < min_handle_length:
            return 0
        left_avg = np.mean(left[-min_cup_length:])
        right_avg = np.mean(right[:min_handle_length])
        depth = (left_avg - right_avg) / left_avg
        if min_depth <= depth <= max_depth:
            # Further checks for handle retracement can be added here
            return 1
        else:
            return 0
    except Exception as e:
        return 0

# Function to label windows
def label_windows(windows, min_cup_length, max_cup_length, min_handle_length,
                  max_handle_length, min_depth, max_depth, handle_max_retrace):
    print("Labeling windows...")
    labels = []
    for idx, window in enumerate(windows):
        label = detect_cup_and_handle_in_window(window, min_cup_length, max_cup_length,
                                                min_handle_length, max_handle_length,
                                                min_depth, max_depth, handle_max_retrace)
        labels.append(label)
    labels = np.array(labels)
    print(f"Labeled windows. Positive samples: {labels.sum()}, Negative samples: {len(labels) - labels.sum()}")
    return labels

# Function to build the CNN model
def build_cnn_model(input_shape):
    print("Building CNN model...")
    model = models.Sequential()
    model.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu',
                            kernel_regularizer=regularizers.l2(0.001),
                            input_shape=input_shape))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Dropout(0.5))  # Dropout layer to prevent overfitting
    model.add(layers.Conv1D(filters=128, kernel_size=3, activation='relu',
                            kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Dropout(0.5))  # Dropout layer
    model.add(layers.Flatten())
    model.add(layers.Dense(100, activation='relu',
                           kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    print("CNN model built successfully.")
    return model

# Function to train the model
def train_model(model, X_train, y_train, X_val, y_val, epochs=20, batch_size=32):
    print("Training the model...")
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(X_val, y_val),
                        callbacks=[early_stop],
                        verbose=1)
    print("Model training completed.")
    return history

# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    print("Evaluating the model on the test set...")
    predictions_prob = model.predict(X_test)
    predictions = (predictions_prob > 0.5).astype("int32")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

# Function to predict on new data
def predict_on_new_data(model, data, window_size, min_cup_length, max_cup_length,
                        min_handle_length, max_handle_length, min_depth, max_depth,
                        handle_max_retrace):
    print("Predicting on new data...")
    windows = create_windows(data, window_size)
    X_new = preprocess_windows(windows)
    predictions_prob = model.predict(X_new, verbose=0)
    predictions = (predictions_prob > 0.5).astype("int32")
    return predictions, windows

# Main Execution Block
if __name__ == '__main__':
    try:
        print("Starting Cup and Handle Pattern Detection Program...")
        
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
            required_days = window_size + max_cup_length + 100  # Additional days for training
            start_date = end_date - timedelta(days=required_days)
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
            required_weeks = window_size + max_cup_length + 52  # Additional weeks for training
            start_date = end_date - timedelta(weeks=required_weeks)
        else:
            raise ValueError('Invalid analysis type selected.')
        
        # Convert dates to strings
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        # Fetch data
        data = fetch_stock_data(ticker, start_date_str, end_date_str, interval)
        if data.empty or len(data) < window_size:
            print(f"Not enough data fetched for ticker '{ticker}'. Please check the ticker symbol or date range and try again.")
            sys.exit(1)
        print(f"Fetched {len(data)} rows of data for {ticker} from {start_date_str} to {end_date_str}.")
        
        # Create windows
        windows = create_windows(data, window_size)
        
        # Label windows
        labels = label_windows(windows, min_cup_length, max_cup_length, min_handle_length,
                               max_handle_length, min_depth, max_depth, handle_max_retrace)
        
        # Preprocess windows
        X = preprocess_windows(windows)
        y = labels
        
        # Split data into training, validation, and test sets
        print("Splitting data into training, validation, and test sets...")
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
        print("Plotting training history...")
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
        # Calculate dynamic dates for new data
        if analysis_type == 'short-term':
            new_end_date = datetime.today()
            new_start_date = new_end_date - timedelta(days=window_size)
        elif analysis_type == 'long-term':
            new_end_date = datetime.today()
            new_start_date = new_end_date - timedelta(weeks=window_size)
        else:
            raise ValueError('Invalid analysis type selected.')
        
        # Convert dates to strings
        new_start_date_str = new_start_date.strftime('%Y-%m-%d')
        new_end_date_str = new_end_date.strftime('%Y-%m-%d')
        
        # Fetch new data
        new_data = fetch_stock_data(ticker, new_start_date_str, new_end_date_str, interval)
        if new_data.empty:
            print(f"No new data fetched for ticker '{ticker}'. Please check the ticker symbol and try again.")
            sys.exit(1)
        print(f"Fetched {len(new_data)} rows of new data for {ticker} from {new_start_date_str} to {new_end_date_str}.")
        
        # Predict on new data
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
        
        # Create a summary DataFrame and save to CSV
        if pattern_details:
            pattern_df = pd.DataFrame(pattern_details)
            summary_filename = f'detected_patterns_{ticker}.csv'
            pattern_df.to_csv(summary_filename, index=False)
            print(f"\nSummary of detected patterns saved to '{summary_filename}'.")
            print(pattern_df)
        else:
            print("No patterns detected in the new data.")
    
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)
