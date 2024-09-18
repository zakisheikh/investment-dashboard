import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go

st.title("Stock Dashboard")

# Add your company logo
st.image("UHURULOGO.jpg", width=200)  # Adjust the width as needed

# Input for stock ticker
ticker = st.text_input("Enter Stock Ticker:", "AAPL")

if st.button("Get Stock Data"):
    try:
        # Fetch stock data from the Flask API
        response = requests.get(f'http://127.0.0.1:5000/stock/{ticker}')
        
        # Check if the response is successful
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data)

            # Format price data to two decimal places
            df['Close'] = df['Close'].round(2)
            df['Open'] = df['Open'].round(2)
            df['High'] = df['High'].round(2)
            df['Low'] = df['Low'].round(2)

            # Format the 'Date' column to MM/DD/YY with UTC handling
            df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.strftime('%m/%d/%y')

            # Set the 'Date' as the index for plotting
            df.set_index('Date', inplace=True)

            # Create the Plotly figure
            fig = go.Figure()

            # Candlestick chart for price action
            fig.add_trace(go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Candlestick',
                increasing_line_color='green',
                decreasing_line_color='red'
            ))

            # Create a separate figure for volume
            volume_fig = go.Figure()

            # Volume bar chart
            volume_fig.add_trace(go.Bar(
                x=df.index,
                y=df['Volume'],
                name='Volume',
                marker_color='lightgray'
            ))

            # Add moving averages for the price
            if 'MA21' in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df['MA21'], mode='lines', name='MA21', line=dict(color='orange', dash='dash')))
            if 'MA50' in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], mode='lines', name='MA50', line=dict(color='blue', dash='dash')))
            if 'MA200' in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df['MA200'], mode='lines', name='MA200', line=dict(color='red', dash='dash')))

            # Update layout for the main price chart
            fig.update_layout(
                title=f"{ticker} Stock Price",
                xaxis_title="Date",
                yaxis_title="Price",
                template='plotly_white',
                height=600,
                xaxis_rangeslider_visible=False
            )

            # Show the candlestick plot in Streamlit
            st.plotly_chart(fig)

            # Update layout for the volume chart
            volume_fig.update_layout(
                title=f"{ticker} Volume",
                xaxis_title="Date",
                yaxis_title="Volume",
                template='plotly_white',
                height=300
            )

            # Show the volume plot in Streamlit
            st.plotly_chart(volume_fig)

        elif response.status_code == 404:
            st.error("Error fetching data: " + response.json()['error'])
        else:
            st.error("An unexpected error occurred.")

    except Exception as e:
        st.error(f"An error occurred: {e}")
