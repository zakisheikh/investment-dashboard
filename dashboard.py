import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

def fetch_stock_data(stock_symbol):
    """Fetch stock data from Yahoo Finance."""
    try:
        stock_data_full = yf.download(stock_symbol, start="2020-01-01", end=datetime.now())
        actual_start_date = stock_data_full.index.min()
        current_date = datetime.now()

        if actual_start_date < (current_date - timedelta(days=730)):
            start_date = actual_start_date
        else:
            start_date = current_date - timedelta(days=730)

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

    # Iterate over potential cups to find handles and calculate buy points
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
                handles.append((row.name, handle_data))

                # Calculate buy point as the peak of the right side of the cup
                right_side_peak = handle_data['High'].max()  # Highest point during the handle
                buy_point = right_side_peak
                buy_zone_low = buy_point  # Buy zone starts at buy point
                buy_zone_high = buy_point * 1.05  # 5% above buy point

                # Store cup with buy point and zone information
                potential_cups_df.at[row.name, 'Buy Point'] = buy_point
                potential_cups_df.at[row.name, 'Buy Zone Low'] = buy_zone_low
                potential_cups_df.at[row.name, 'Buy Zone High'] = buy_zone_high

    return potential_cups_df, handles

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
        print(f"Cup detected on {cup[0]}: Low = {cup[1]['Low']}, Close = {cup[1]['Close']}, "
              f"Buy Point = {cup[1]['Buy Point']}, Buy Zone = ({cup[1]['Buy Zone Low']}, {cup[1]['Buy Zone High']})")

    print(f"Detected {len(handles)} potential handles.")
    for handle in handles:
        print(f"Handle detected on {handle[0]} with data:\n{handle[1]}")

if __name__ == "__main__":
    stock_symbol = "AMD"  # Example stock symbol
    main(stock_symbol)
