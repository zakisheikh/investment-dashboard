import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go

st.title("Stock Dashboard")

# Add your company logo
st.image("UHURULOGO.jpg", width=200)  # Adjust the width as needed

# Input for stock ticker
ticker = st.text_input("Enter Stock Ticker:", "AAPL")

# Select time frame
time_frame = st.selectbox("Select Time Frame:", ["1m", "5m", "1d", "5d", "1w", "1mo"])

if st.button("Get Stock Data"):
    try:
        # Determine the start date based on the selected time frame
        url = f'http://127.0.0.1:5000/stock/{ticker}/{time_frame}'

        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data)

            # Format price data to two decimal places
            df['Close'] = df['Close'].round(2)
            df['Open'] = df['Open'].round(2)
            df['High'] = df['High'].round(2)
            df['Low'] = df['Low'].round(2)

            # Convert Date column and set as index
            df['Date'] = pd.to_datetime(df['Date'], utc=True)
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

            # Add moving averages if they exist
            for col in df.columns:
                if col.startswith('MA'):
                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df[col],
                        mode='lines',
                        name=col,
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
