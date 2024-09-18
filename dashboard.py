import pandas as pd
import requests
import streamlit as st
import plotly.graph_objs as go
from datetime import datetime, timedelta

# Function to calculate default start date based on time frame
def calculate_default_start_date(time_frame):
    today = datetime.utcnow()
    if time_frame == 'd':
        return today - timedelta(days=180)  # 6 months for daily
    elif time_frame == 'w':
        return today - timedelta(days=365)  # 1 year for weekly
    else:
        return None  # No default for other time frames

# Main Streamlit application
st.title("Stock Data Dashboard")

# User input for ticker symbol and time frame
ticker = st.text_input("Enter stock ticker (e.g., AAPL):", "AAPL")
time_frame = st.selectbox("Select time frame:", ["d", "w", "1w", "1m"])

# Set default start date based on selected time frame
default_start_date = calculate_default_start_date(time_frame)

if st.button("Get Stock Data"):
    try:
        # Format the default start date for the API request
        start_date_str = default_start_date.strftime('%Y-%m-%d') if default_start_date else None
        url = f'http://127.0.0.1:5000/stock/{ticker}/{time_frame}'
        
        # Only include the start date if it's available
        if start_date_str:
            url += f'?start_date={start_date_str}'

        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data)

            # Format price data to two decimal places
            df['Close'] = df['Close'].round(2)
            df['Open'] = df['Open'].round(2)
            df['High'] = df['High'].round(2)
            df['Low'] = df['Low'].round(2)

            # Update this line to use the correct format
            df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y', utc=True)
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
                decreasing_line_color='red',
                hovertext=(
                    "Date: " + df.index.astype(str) + "<br>" +
                    "Open: " + df['Open'].astype(str) + "<br>" +
                    "High: " + df['High'].astype(str) + "<br>" +
                    "Low: " + df['Low'].astype(str) + "<br>" +
                    "Close: " + df['Close'].astype(str) + "<br>" +
                    "Volume: " + df['Volume'].astype(str)
                ),
                hoverinfo='text'  # Show only custom text
            ))

            # Determine color for volume bars
            volume_color = ['green' if close >= open else 'red' for close, open in zip(df['Close'], df['Open'])]

            # Create a separate figure for volume
            volume_fig = go.Figure()

            # Volume bar chart with color based on price movement
            volume_fig.add_trace(go.Bar(
                x=df.index,
                y=df['Volume'],
                name='Volume',
                marker_color=volume_color
            ))

            # Add moving averages for the price if they exist
            for ma in ['MA21', 'MA50', 'MA200']:
                if ma in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df[ma],
                        mode='lines',
                        name=ma,
                        line=dict(dash='dash')
                    ))

            # Get the last price for display
            last_price = df['Close'].iloc[-1]

            # Update layout for the main price chart
            fig.update_layout(
                title=f"{ticker} Stock Price - Last Price: ${last_price:.2f}",
                xaxis_title="Date",
                yaxis_title="Price",
                template='plotly_white',
                height=600,
                xaxis_rangeslider_visible=False
            )

            # Show the candlestick plot in Streamlit
            st.plotly_chart(fig)

            # Update layout for the volume chart without title
            volume_fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Volume",
                template='plotly_white',
                height=300,
                showlegend=False  # Hide legend for volume
            )

            # Show the volume plot in Streamlit right below the candlestick
            st.plotly_chart(volume_fig)

            # Key information display
            st.markdown("""
            **Key:**
            - **Green Volume Bars**: Closing price was higher than the opening price.
            - **Red Volume Bars**: Closing price was lower than the opening price.
            """)

        elif response.status_code == 404:
            st.error("Error fetching data: " + response.json().get('error', 'Unknown error'))
        else:
            st.error("An unexpected error occurred.")

    except Exception as e:
        st.error(f"An error occurred: {e}")
