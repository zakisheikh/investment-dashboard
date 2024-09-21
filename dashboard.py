import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

def fetch_stock_data(stock_symbol):
    """Fetch stock data from Yahoo Finance."""
    try:
        # Fetch data for a larger date range to find the actual available history
        stock_data_full = yf.download(stock_symbol, start="2020-01-01", end=datetime.now())
        
        # Get the actual start date of the fetched data
        actual_start_date = stock_data_full.index.min()
        current_date = datetime.now()

        # Set the start date to either two years ago or the actual start date
        if actual_start_date < (current_date - timedelta(days=730)):
            start_date = actual_start_date
        else:
            start_date = current_date - timedelta(days=730)
        
        # Fetch data for the determined date range
        stock_data = yf.download(stock_symbol, start=start_date, end=current_date)
        stock_data.reset_index(inplace=True)
        return stock_data
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()

def detect_cup_and_handle(stock_data):
    """Detect cup and handle patterns in stock data."""
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    stock_data.set_index('Date', inplace=True)

    potential_cups = []
    handles = []

    # Iterate through the stock data to find potential cups
    for idx in range(1, len(stock_data) - 1):
        if (stock_data['Low'].iloc[idx] < stock_data['Low'].iloc[idx - 1] and
            stock_data['High'].iloc[idx] > stock_data['High'].iloc[idx - 1]):
            potential_cups.append(stock_data.iloc[idx])

    potential_cups_df = pd.DataFrame(potential_cups)

    # Iterate over potential cups to find handles
    for idx, row in potential_cups_df.iterrows():
        cup_date = row.name
        handle_start_date = cup_date + pd.DateOffset(days=10)

        if handle_start_date in stock_data.index:
            handle_data = stock_data.loc[handle_start_date:handle_start_date + pd.DateOffset(days=20)]
            cup_bottom = row['Low']
            max_close = row['Close']
            min_close = handle_data['Close'].min()
            pullback = (max_close - min_close) / max_close

            if 0.08 <= pullback <= 0.12:
                handles.append((row.name, handle_data, row['High']))  # Include the left side peak

    return potential_cups_df, handles

def calculate_buy_points(handles):
    """Calculate buy points based on the left side peak."""
    buy_points = []
    for handle in handles:
        left_side_peak = handle[2]  # Left side peak from the handle tuple
        buy_point = left_side_peak * 1.1  # 10% above the left side peak
        buy_points.append((handle[0], buy_point))  # Include date with buy point
    return buy_points

def main(stock_symbol):
    """Main function to execute the cup and handle detection."""
    print(f"Fetching data for {stock_symbol}...")
    stock_data = fetch_stock_data(stock_symbol)
    
    if stock_data.empty:
        print("No data fetched. Exiting...")
        return

    print("Fetching data completed. Detecting patterns...")
    cups, handles = detect_cup_and_handle(stock_data)
    
    print(f"Detected {len(cups)} potential cups.")
    for cup in cups.iterrows():
        print(f"Cup detected on {cup[0]}: Low = {cup[1]['Low']}, Close = {cup[1]['Close']}")

    print(f"Detected {len(handles)} potential handles.")
    buy_points = calculate_buy_points(handles)
    for handle in handles:
        print(f"Handle detected on {handle[0]} with data:\n{handle[1]}")

    print(f"\nCalculated Buy Points:")
    for date, buy_point in buy_points:
        print(f"Buy Point for handle detected on {date}: {buy_point:.2f}")

if __name__ == "__main__":
    stock_symbol = "AMD"  # Example stock symbol
    main(stock_symbol)
