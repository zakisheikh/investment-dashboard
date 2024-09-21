# visualization.py
import matplotlib.pyplot as plt

def plot_stock_data_with_pattern(stock_data, cups, handles):
    plt.figure(figsize=(12, 8))
    plt.plot(stock_data.index, stock_data['Close'], label='Close Price', color='blue', alpha=0.5)
    
    # Highlight detected cups
    for idx, cup_data in cups.iterrows():
        cup_peak_index = stock_data.index.get_loc(cup_data.name)  # Get integer location of the cup peak
        plt.axvline(x=cup_data.name, color='orange', linestyle='--', label='Cup Start' if idx == cups.index[0] else "")
        plt.plot(stock_data.index[cup_peak_index], cup_data['Close'], 'bo', markersize=8, label='Cup Peak' if idx == cups.index[0] else "")
    
    # Highlight handles if any are found
    if handles:
        for (start_idx, handle_data) in handles:
            plt.axvline(x=start_idx, color='green', linestyle='--', label='Handle Start' if start_idx == handles[0][0] else "")
            plt.plot(handle_data.index, handle_data['Close'], color='red', linestyle='-', linewidth=2, label='Handle Pattern' if start_idx == handles[0][0] else "")
            plt.plot(handle_data.index[0], handle_data['Close'].iloc[0], 'ro', markersize=8)  # Mark the start of the handle

    plt.title("Stock Price with Detected Cup-with-Handle Pattern", fontsize=16)
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Price", fontsize=14)
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
