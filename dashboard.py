import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

st.title("Stock Dashboard")

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

            # Display stock data
            st.subheader(f"{ticker} Stock Data")
            st.write(df)

            # Plot Closing Prices
            st.line_chart(df.set_index('Date')['Close'])

            # Moving Averages
            st.subheader("Moving Averages")
            st.line_chart(df.set_index('Date')[['Close', 'MA21', 'MA50', 'MA200']])

            # RSI
            st.subheader("RSI")
            st.line_chart(df.set_index('Date')['RSI'])

        elif response.status_code == 404:
            st.error("Error fetching data: " + response.json()['error'])
        else:
            st.error("An unexpected error occurred.")

    except Exception as e:
        st.error(f"An error occurred: {e}")

