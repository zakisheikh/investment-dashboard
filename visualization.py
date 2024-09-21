# visualization.py
import matplotlib.pyplot as plt
import pandas as pd

def plot_stock_data_with_pattern(stock_data, cups, handles):
    plt.figure(figsize=(14, 8))
    plt.plot(stock_data['Date'], stock_data['Close'], label='Close Price', color='blue', alpha=0.6)

    # Highlight detected cups
    for idx, cup_data in cups.iterrows():
        plt.axvspan(cup_data['Start_Date'], cup_data['Bottom_Date'], color='lightgray', alpha=0.5, label='Cup Decline' if idx == cups.index[0] else "")
        plt.plot(cup_data['Bottom_Date'], cup_data['Bottom_Price'], 'bo', markersize=8, label='Cup Bottom' if idx == cups.index[0] else "")
        plt.axvspan(cup_data['Bottom_Date'], cup_data['End_Date'], color='lightblue', alpha=0.5, label='Cup Rise' if idx == cups.index[0] else "")
        plt.axvspan(cup_data['End_Date'], cup_data['Handle_End_Date'], color='lightgreen', alpha=0.5, label='Handle Formation' if idx == cups.index[0] else "")
        plt.plot(cup_data['Breakout_Date'], cup_data['Breakout_Price'], 'ro', markersize=8, label='Breakout' if idx == cups.index[0] else "")

    # Highlight handles if any are found
    if handles:
        for (start_idx, handle_data) in handles:
            plt.plot(handle_data['Date'], handle_data['Close'], color='red', linestyle='-', linewidth=2, label='Handle Pattern' if start_idx == handles[0][0] else "")
            plt.plot(handle_data['Date'].iloc[0], handle_data['Close'].iloc[0], 'ro', markersize=8)  # Mark the start of the handle

    plt.title("Stock Price with Detected Cup-with-Handle Pattern", fontsize=16)
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Price", fontsize=14)
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

def summarize_cup_and_handle():
    summary = """
    The Cup and Handle pattern is a bullish continuation signal formed by six stages:
    1. **Decline**: A drop from the peak, indicating selling pressure.
    2. **Bottom**: A reversal forming the bottom of the cup, attracting bargain hunters.
    3. **Rise**: Price rallies back to the peak, building bullish sentiment.
    4. **Handle Formation**: A slight pullback to create the handle, where volume declines.
    5. **Breakout**: Price breaks above the rim of the cup, signaling strong buying interest.
    6. **Price Target Achievement**: Height of the cup projected upward for price target.

    Understanding this pattern can help traders identify bullish opportunities effectively.
    """
    print(summary)
