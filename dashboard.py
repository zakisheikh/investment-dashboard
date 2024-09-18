import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

# Add your company logo
st.image("UHURULOGO.jpg", width=200)  # Adjust the width as needed

st.title("Stock Analysis Dashboard")

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

            # Convert the 'Date' column to datetime format
            df['Date'] = pd.to_datetime(df['Date'])

            # Format price data to two decimal places
            df['Close'] = df['Close'].round(2)
            df['Open'] = df['Open'].round(2)
            df['High'] = df['High'].round(2)
            df['Low'] = df['Low'].round(2)

            # Format the 'Date' column to MM/DD/YY for display
            df['Display_Date'] = df['Date'].dt.strftime('%m/%d/%y')

            # Set the 'Date' as the index for plotting
            df.set_index('Display_Date', inplace=True)

            # Display stock data
            st.subheader(f"{ticker} Stock Data")
            st.write(df)

            # Create an interactive chart
            fig, ax1 = plt.subplots(figsize=(10, 6))

            # Plot price action
            ax1.plot(df.index, df['Close'], label='Close', color='blue')
            ax1.plot(df.index, df['MA21'], label='MA21', color='orange', linestyle='--')
            ax1.plot(df.index, df['MA50'], label='MA50', color='green', linestyle='--')
            ax1.plot(df.index, df['MA200'], label='MA200', color='red', linestyle='--')

            # Create a second y-axis for RSI
            ax2 = ax1.twinx()
            ax2.plot(df.index, df['RSI'], label='RSI', color='purple', linestyle=':')

            # Create a third y-axis for Volume
            ax3 = ax1.twinx()
            ax3.spines['right'].set_position(('outward', 60))  # Offset the third axis
            ax3.bar(df.index, df['Volume'], label='Volume', color='lightgray', alpha=0.5)

            # Add legends
            ax1.legend(loc='upper left')
            ax2.legend(loc='upper right')
            ax3.legend(loc='center right')

            # Set labels
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Price')
            ax2.set_ylabel('RSI')
            ax3.set_ylabel('Volume')

            # Show the plot in Streamlit
            st.pyplot(fig)

        elif response.status_code == 404:
            st.error("Error fetching data: " + response.json()['error'])
        else:
            st.error("An unexpected error occurred.")

    except Exception as e:
        st.error(f"An error occurred: {e}")
