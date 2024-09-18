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

            # Price action
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=df.index, y=df['MA21'], mode='lines', name='MA21', line=dict(color='orange', dash='dash')))
            fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], mode='lines', name='MA50', line=dict(color='green', dash='dash')))
            fig.add_trace(go.Scatter(x=df.index, y=df['MA200'], mode='lines', name='MA200', line=dict(color='red', dash='dash')))

            # RSI as a separate axis
            fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI', line=dict(color='purple'), yaxis='y2'))

            # Volume as a bar chart
            fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color='lightgray', yaxis='y3'))

            # Update layout for multiple y-axes
            fig.update_layout(
                title=f"{ticker} Stock Data",
                xaxis_title="Date",
                yaxis_title="Price",
                yaxis2=dict(title='RSI', overlaying='y', side='right', showgrid=False),
                yaxis3=dict(title='Volume', overlaying='y', side='right', position=0.95, showgrid=False),
                legend=dict(x=0.01, y=0.99),
                template='plotly_white',
                height=600
            )

            # Show the plot in Streamlit
            st.plotly_chart(fig)

        elif response.status_code == 404:
            st.error("Error fetching data: " + response.json()['error'])
        else:
            st.error("An unexpected error occurred.")

    except Exception as e:
        st.error(f"An error occurred: {e}")
