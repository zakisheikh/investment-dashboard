# data_fetch.py
import yfinance as yf
from datetime import datetime, timedelta

def get_stock_data(symbol, years_back=2):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years_back * 365)
    
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    return stock_data
