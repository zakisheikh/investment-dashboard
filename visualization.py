import matplotlib.pyplot as plt

def plot_stock_data_with_pattern(stock_data, cups, handles):
    plt.figure(figsize=(14, 7))
    
    # Plot the close price line
    plt.plot(stock_data.index, stock_data['Close'], label='Close Price', color='blue', alpha=0.7, linewidth=2)

    # Highlight detected cups
    for idx, cup_data in cups.iterrows():
        plt.axvline(x=cup_data.name, color='orange', linestyle='--', alpha=0.5, label='Cup Start' if idx == cups.index[0] else "")
        plt.plot(cup_data.name, cup_data['Close'], 'o', markersize=10, color='gold', label='Cup Peak' if idx == cups.index[0] else "")
        plt.annotate('Cup Peak', (cup_data.name, cup_data['Close']), textcoords="offset points", xytext=(0,10), ha='center')

    # Highlight handles
    if handles:
        for start_idx, handle_data in handles:
            plt.axvline(x=start_idx, color='green', linestyle='--', alpha=0.5, label='Handle Start' if start_idx == handles[0][0] else "")
            plt.plot(handle_data.index, handle_data['Close'], color='red', linewidth=2, linestyle='-', label='Handle Pattern' if start_idx == handles[0][0] else "")
            plt.plot(handle_data.index[0], handle_data['Close'].iloc[0], 'o', markersize=10, color='purple')  # Mark start of the handle
            plt.annotate('Handle Start', (handle_data.index[0], handle_data['Close'].iloc[0]), textcoords="offset points", xytext=(0,10), ha='center')

    # Title and labels
    plt.title("Stock Price with Detected Cup-and-Handle Pattern", fontsize=18)
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Price", fontsize=14)
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(visible=True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()
