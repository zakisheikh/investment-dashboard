# dashboard.py
import sys
import data_fetch
import pattern_recognition
import visualization

def main(symbol):
    # Get data for last two years dynamically
    stock_data = data_fetch.get_stock_data(symbol)
    
    # Detect Cup-with-Handle pattern
    cups, handles = pattern_recognition.detect_cup_with_handle(stock_data)
    
    # Visualize stock price and pattern
    visualization.plot_stock_data_with_pattern(stock_data, cups, handles)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python dashboard.py <stock_symbol>")
    else:
        symbol = sys.argv[1]
        main(symbol)
