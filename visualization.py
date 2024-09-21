# visualization.py
import matplotlib.pyplot as plt

def plot_stock_data_with_pattern(stock_data, cups, handles):
    plt.figure(figsize=(10, 6))
    plt.plot(stock_data['Close'], label='Close Price')
    
    # Highlight detected cups
    for idx, cup_data in cups.iterrows():
        plt.plot(idx, cup_data['Close'], 'bo', label='Cup Pattern' if idx == cups.index[0] else "")
    
    # Highlight handles if any are found
    if handles:
        for (start_idx, handle_data) in handles:
            plt.plot(handle_data.index, handle_data['Close'], 'ro', label='Handle Pattern' if start_idx == handles[0][0] else "")
    else:
        print("No handles found for the detected cup patterns.")
    
    plt.title("Stock Price with Detected Cup-with-Handle Pattern")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()
