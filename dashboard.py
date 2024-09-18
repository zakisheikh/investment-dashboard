import yfinance as yf
from pattern_recognition import detect_cup_handle, detect_flat_base, detect_breakout, calculate_composite_rating
import plotly.graph_objs as go
from dash import html, dcc

def fetch_stock_data(ticker, start, end):
    return yf.download(ticker, start=start, end=end)

def create_dashboard():
    stock_data = fetch_stock_data('AAPL', '2020-01-01', '2023-01-01')
    sp500_data = fetch_stock_data('^GSPC', '2020-01-01', '2023-01-01')

    # Detect patterns
    stock_data['CupHandle'] = detect_cup_handle(stock_data)
    stock_data['FlatBase'] = detect_flat_base(stock_data)
    stock_data['Breakout'] = detect_breakout(stock_data)

    # Composite rating
    composite_rating = calculate_composite_rating(stock_data, sp500_data)
    
    # Graph to display stock prices and detected patterns
    fig = go.Figure([go.Scatter(x=stock_data.index, y=stock_data['Close'], name="Stock Price")])

    # Highlight patterns (CupHandle, FlatBase)
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], 
                             mode='markers', marker=dict(color='blue', size=10), 
                             name='CupHandle', visible=stock_data['CupHandle']))

    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], 
                             mode='markers', marker=dict(color='green', size=10), 
                             name='FlatBase', visible=stock_data['FlatBase']))

    # Layout for the Dash application
    layout = html.Div([
        html.H1("CANSLIM Stock Pattern Recognition"),
        dcc.Graph(figure=fig),
        html.P(f"Composite Rating: {composite_rating:.2f}%")
    ])
    
    return layout
