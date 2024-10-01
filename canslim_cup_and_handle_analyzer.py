# canslim_cup_and_handle_analyzer.py

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import os
import requests

# Suppress warnings (optional)
import warnings
warnings.filterwarnings('ignore')

# Step 1: Fetch Historical Stock Data
def fetch_stock_data(ticker, period='1y', interval='1d'):
    """
    Fetch historical stock data using yfinance.

    Parameters:
    - ticker (str): Stock ticker symbol.
    - period (str): Data period (e.g., '1y', '2y').
    - interval (str): Data interval (e.g., '1d', '1wk').

    Returns:
    - data (DataFrame): Historical stock data.
    """
    data = yf.download(ticker, period=period, interval=interval)
    data.dropna(inplace=True)
    data['Ticker'] = ticker  # Add ticker symbol for reference
    return data

# Step 2: CANSLIM Analysis Functions

# A. Get Earnings Growth
def get_earnings_growth(ticker):
    """
    Fetch and calculate current and annual earnings growth.

    Parameters:
    - ticker (str): Stock ticker symbol.

    Returns:
    - earnings_data (dict): Dictionary containing earnings growth percentages.
    """
    stock = yf.Ticker(ticker)
    # Get quarterly earnings (Income Statement)
    quarterly_earnings = stock.quarterly_earnings
    annual_earnings = stock.earnings

    # Check if data is available
    if quarterly_earnings.empty or annual_earnings.empty:
        return None

    # Calculate current earnings growth (most recent quarter vs same quarter last year)
    try:
        recent_quarter = quarterly_earnings.iloc[-1]['Earnings']
        same_quarter_last_year = quarterly_earnings.iloc[-5]['Earnings']
        current_growth_percentage = ((recent_quarter - same_quarter_last_year) / abs(same_quarter_last_year)) * 100
    except IndexError:
        current_growth_percentage = None

    # Calculate annual earnings growth (most recent year vs previous year)
    try:
        recent_year = annual_earnings.iloc[-1]['Earnings']
        previous_year = annual_earnings.iloc[-2]['Earnings']
        annual_growth_percentage = ((recent_year - previous_year) / abs(previous_year)) * 100
    except IndexError:
        annual_growth_percentage = None

    earnings_data = {
        'current_earnings_growth': current_growth_percentage,
        'annual_earnings_growth': annual_growth_percentage
    }

    return earnings_data

# B. Check for New Products/Services (Simplified)
def check_new_products(ticker):
    """
    Placeholder function to simulate checking for new products or services.

    Parameters:
    - ticker (str): Stock ticker symbol.

    Returns:
    - has_new_products (bool): True if new products/services are found.
    """
    # For demonstration purposes, we'll return True
    has_new_products = True
    return has_new_products

# C. Get Institutional Ownership (Simplified)
def get_institutional_ownership(ticker):
    """
    Placeholder function to simulate fetching institutional ownership data.

    Parameters:
    - ticker (str): Stock ticker symbol.

    Returns:
    - institutional_ownership (float): Percentage of institutional ownership.
    """
    # For demonstration purposes, we'll assume 60% ownership
    institutional_ownership = 60.0
    return institutional_ownership

# D. Check if Market Leader (Simplified)
def is_market_leader(ticker):
    """
    Placeholder function to simulate checking if the stock is a market leader.

    Parameters:
    - ticker (str): Stock ticker symbol.

    Returns:
    - is_leader (bool): True if the stock is a market leader.
    """
    # For demonstration purposes, we'll return True
    is_leader = True
    return is_leader

# E. Get Market Direction (Simplified)
def get_market_direction():
    """
    Placeholder function to determine the overall market trend.

    Returns:
    - market_trend (str): 'Uptrend', 'Downtrend', or 'Sideways'
    """
    # For demonstration purposes, we'll set the market direction to 'Uptrend'
    market_trend = 'Uptrend'
    return market_trend

# Step 3: Cup and Handle Pattern Detection (Latest Pattern)
def detect_recent_cup_and_handle(data):
    """
    Detect the most recent (and possibly forming) Cup and Handle pattern in stock data.

    Parameters:
    - data (DataFrame): Historical stock data.

    Returns:
    - pattern (dict): Details of the most recent pattern detected.
    """
    close = data['Close'].values
    volume = data['Volume'].values
    pattern = None  # Initialize pattern as None

    # Calculate Moving Averages
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean()

    # Parameters (Adjust as needed)
    min_cup_length = 15
    max_cup_length = 60
    max_handle_length = 20
    tolerance = 0.05  # 5% tolerance for peak alignment
    min_depth = 0.15  # Minimum 15% depth for the cup

    # Start from the most recent data point and move backwards
    for i in range(len(close) - min_cup_length - max_handle_length - 1, 0, -1):
        # Check if stock is in uptrend (MA50 above MA200)
        if data['MA50'].iloc[i] < data['MA200'].iloc[i]:
            continue  # Skip if not in an uptrend

        for j in range(i + min_cup_length, min(i + max_cup_length, len(close) - max_handle_length - 1)):
            left_peak = close[i]
            right_peak = close[j]
            bottom = np.min(close[i:j+1])
            bottom_index = np.argmin(close[i:j+1]) + i

            # Check if peaks are within tolerance
            peak_diff = abs(left_peak - right_peak) / left_peak
            if peak_diff > tolerance:
                continue

            # Check cup depth
            depth = 1 - (bottom / ((left_peak + right_peak) / 2))
            if depth < min_depth:
                continue

            # Ensure U-shape (bottom not at the edges)
            if bottom_index == i or bottom_index == j:
                continue

            # Handle detection
            handle_start = j + 1
            handle_end = handle_start + max_handle_length
            handle_end = min(handle_end, len(close) - 2)  # Adjust for data range

            handle = close[handle_start:handle_end+1]
            handle_volume = volume[handle_start:handle_end+1]
            handle_max = np.max(handle)
            handle_min = np.min(handle)

            # Handle may still be forming, so we relax some conditions
            # Handle should not exceed the peaks
            if handle_max > right_peak * (1 + tolerance):
                continue

            # Handle retracement should not exceed 50% of cup depth
            handle_retracement = (right_peak - handle_min) / (right_peak - bottom)
            if handle_retracement > 0.5:
                continue

            # Volume decrease during handle formation (optional for forming patterns)
            avg_cup_volume = np.mean(volume[i:j+1])
            avg_handle_volume = np.mean(handle_volume)
            if avg_handle_volume > avg_cup_volume:
                continue  # Skip if volume doesn't decrease during handle

            # Breakout may not have occurred yet
            breakout_index = handle_end + 1
            breakout_occurred = False
            if breakout_index < len(close):
                breakout_volume = volume[breakout_index]
                avg_volume = np.mean(volume[max(breakout_index - 10, 0):breakout_index])
                if breakout_volume >= avg_volume * 1.5:
                    breakout_occurred = True
            else:
                breakout_volume = None
                avg_volume = None

            # Store the most recent pattern and exit loops
            pattern = {
                'ticker': data['Ticker'],
                'cup_start': i,
                'cup_end': j,
                'handle_end': handle_end,
                'breakout_index': breakout_index if breakout_index < len(close) else None,
                'cup_start_date': data.index[i].date(),
                'cup_end_date': data.index[j].date(),
                'handle_end_date': data.index[handle_end].date(),
                'breakout_date': data.index[breakout_index].date() if breakout_index < len(close) else None,
                'left_peak': left_peak,
                'right_peak': right_peak,
                'bottom': bottom,
                'handle_min': handle_min,
                'breakout_volume': breakout_volume,
                'avg_volume': avg_volume,
                'breakout_occurred': breakout_occurred
            }
            return pattern  # Return the most recent pattern detected

    return pattern  # Return None if no pattern found

# Step 4: Visualization
def plot_recent_cup_and_handle(data, pattern, output_dir='plots'):
    """
    Plot the most recent Cup and Handle pattern and save the plot to a file.

    Parameters:
    - data (DataFrame): Historical stock data.
    - pattern (dict): Details of the detected pattern.
    - output_dir (str): Directory to save plot image.
    """
    if pattern is None:
        print("No recent Cup and Handle pattern to plot.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    start = pattern['cup_start']
    end = pattern['handle_end']
    breakout = pattern['breakout_index']
    ticker = pattern['ticker']

    plt.figure(figsize=(14, 7))
    plt.plot(data.index[start:end+2], data['Close'][start:end+2], label='Close Price', color='blue')
    plt.axvline(x=data.index[pattern['cup_start']], color='green', linestyle='--', label='Cup Start')
    plt.axvline(x=data.index[pattern['cup_end']], color='orange', linestyle='--', label='Cup End')
    plt.axvline(x=data.index[pattern['handle_end']], color='red', linestyle='--', label='Handle End')

    if pattern['breakout_occurred']:
        plt.axvline(x=data.index[pattern['breakout_index']], color='purple', linestyle='--', label='Breakout Day')
        plt.title(f"{ticker}: Recent Cup and Handle Pattern (Breakout Occurred)")
    else:
        plt.title(f"{ticker}: Recent Cup and Handle Pattern (Forming)")

    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()

    filename = f"{ticker}_Recent_CupAndHandle.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    plt.close()
    print(f"Pattern plot saved: {filepath}")

# Step 5: Generate Report
def generate_recent_pattern_report(pattern, canslim_criteria, output_file='RecentCupAndHandleReport.txt'):
    """
    Generate a report of the most recent detected pattern and CANSLIM analysis.

    Parameters:
    - pattern (dict): Details of the detected pattern.
    - canslim_criteria (dict): Results of the CANSLIM analysis.
    - output_file (str): Filename for the report.
    """
    with open(output_file, 'w') as report:
        if pattern is None:
            report.write("No recent Cup and Handle pattern detected.\n\n")
        else:
            report.write(f"Recent Cup and Handle Pattern in {pattern['ticker']}:\n")
            report.write(f" - Cup Start Date: {pattern['cup_start_date']}\n")
            report.write(f" - Cup End Date: {pattern['cup_end_date']}\n")
            report.write(f" - Handle End Date: {pattern['handle_end_date']}\n")

            if pattern['breakout_occurred']:
                report.write(f" - Breakout Date: {pattern['breakout_date']}\n")
                report.write(f" - Breakout Occurred: Yes\n")
                report.write(f" - Breakout Volume: {pattern['breakout_volume']}\n")
                report.write(f" - Average Volume Before Breakout: {pattern['avg_volume']:.2f}\n")
            else:
                report.write(f" - Breakout Occurred: No (Pattern may be forming)\n")

            report.write(f" - Left Peak Price: ${pattern['left_peak']:.2f}\n")
            report.write(f" - Right Peak Price: ${pattern['right_peak']:.2f}\n")
            report.write(f" - Cup Bottom Price: ${pattern['bottom']:.2f}\n")
            report.write(f" - Handle Minimum Price: ${pattern['handle_min']:.2f}\n\n")

        # CANSLIM Analysis
        report.write("CANSLIM Analysis:\n")
        for criterion, result in canslim_criteria.items():
            meets = "Yes" if result['meets'] else "No"
            report.write(f" - {criterion}: {result['value']} (Meets Criterion: {meets})\n")

        # Overall Assessment
        all_criteria_met = all(value['meets'] for value in canslim_criteria.values())
        if all_criteria_met:
            report.write("\nOverall Assessment: The stock meets all CANSLIM criteria.")
        else:
            report.write("\nOverall Assessment: The stock does not meet all CANSLIM criteria.")

    print(f"Report generated: {output_file}")

# Step 6: Main Execution
if __name__ == '__main__':
    ticker = 'AAPL'  # Replace with desired ticker symbol
    print(f"Processing {ticker} for CANSLIM analysis and recent Cup and Handle pattern...")

    # Fetch data
    data = fetch_stock_data(ticker, period='1y')
    if data.empty:
        print(f"No data retrieved for {ticker}.")
    else:
        # Perform CANSLIM analysis
        earnings_data = get_earnings_growth(ticker)
        new_products = check_new_products(ticker)
        institutional_ownership = get_institutional_ownership(ticker)
        market_leader = is_market_leader(ticker)
        market_direction = get_market_direction()

        # Detect recent Cup and Handle pattern
        pattern = detect_recent_cup_and_handle(data)

        # Compile CANSLIM criteria
        canslim_criteria = {
            'Current Earnings Growth': {
                'value': f"{earnings_data['current_earnings_growth']:.2f}%" if earnings_data and earnings_data['current_earnings_growth'] else "Data Not Available",
                'meets': earnings_data and earnings_data['current_earnings_growth'] and earnings_data['current_earnings_growth'] > 20
            },
            'Annual Earnings Growth': {
                'value': f"{earnings_data['annual_earnings_growth']:.2f}%" if earnings_data and earnings_data['annual_earnings_growth'] else "Data Not Available",
                'meets': earnings_data and earnings_data['annual_earnings_growth'] and earnings_data['annual_earnings_growth'] > 25
            },
            'New Products/Services': {
                'value': "Yes" if new_products else "No",
                'meets': new_products
            },
            'Supply and Demand (Volume Analysis)': {
                'value': "Pattern Detected" if pattern else "No Pattern",
                'meets': pattern is not None
            },
            'Leader or Laggard': {
                'value': "Leader" if market_leader else "Laggard",
                'meets': market_leader
            },
            'Institutional Sponsorship': {
                'value': f"{institutional_ownership:.2f}%",
                'meets': institutional_ownership and institutional_ownership > 50
            },
            'Market Direction': {
                'value': market_direction,
                'meets': market_direction == 'Uptrend'
            }
        }

        # Output CANSLIM analysis results
        print("\nCANSLIM Analysis Results:")
        for criterion, result in canslim_criteria.items():
            meets = "✅" if result['meets'] else "❌"
            print(f"{criterion}: {result['value']} {meets}")

        # Overall Assessment
        all_criteria_met = all(value['meets'] for value in canslim_criteria.values())
        if all_criteria_met:
            print(f"\nOverall Assessment: {ticker} meets all the CANSLIM criteria.")
        else:
            print(f"\nOverall Assessment: {ticker} does not meet all the CANSLIM criteria.")

        # Plot and report
        plot_recent_cup_and_handle(data, pattern)
        generate_recent_pattern_report(pattern, canslim_criteria, output_file=f"{ticker}_CANSLIM_Report.txt")
