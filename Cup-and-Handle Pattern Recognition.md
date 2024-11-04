# Cup and Handle Pattern Detection

### Overview
This application detects the **Cup and Handle** pattern in historical stock data, providing traders with insights into potential trading opportunities. Utilizing deep learning and heuristic rules, this tool identifies windows of stock data that may exhibit this pattern, often seen as a bullish signal in technical analysis.

---

### 1. Data Collection
The application begins by collecting historical stock data for a specified ticker using **Yahoo Finance**. This data includes:
- Open, High, Low, Close prices
- Volume
- Adjusted Close prices

The data spans from January 1, 2010, to December 31, 2023, by default, and is essential for identifying historical patterns.

### 2. Data Windowing
To identify patterns, we split the stock data into sliding windows. Each window represents a subset of historical data over a fixed period, enabling the model to analyze small segments of the stock's price movements and detect specific patterns.

**Default Window Size:** 60 days

### 3. Pattern Detection
The *Cup and Handle* pattern is detected through a series of heuristic rules applied within each window. These rules check for:
- **Cup Shape**: Defined by a dip in price followed by a recovery, mimicking the shape of a "U".
- **Handle Formation**: Following the cup, a smaller downward trend (the handle) should appear, signaling potential breakout points.
  
Key attributes of this detection:
- **Depth**: Ratio of the dip to the original peak average, ensuring a consistent "cup" shape.
- **Handle Retrace**: Checks the retracement level in the handle, ensuring it doesn’t exceed 50% of the cup's depth.

### 4. Feature Engineering
Each window of stock data is normalized to standardize prices and improve the neural network's ability to learn patterns consistently across various stock tickers and price ranges.

### 5. Model Building
A Convolutional Neural Network (CNN) is built to classify the presence of a *Cup and Handle* pattern in each window:
- **Architecture**: Two convolutional layers with max pooling, followed by dense layers.
- **Output**: A binary classification (pattern vs. no pattern) with accuracy as the evaluation metric.
- **Training**: The model is trained on labeled windows of data to learn and recognize the *Cup and Handle* pattern.

### 6. Streamlit User Interface
The app’s interface provides an interactive experience:
- **Ticker Input**: Users enter a stock ticker symbol (e.g., AAPL, NVDA).
- **Window Creation**: The app scans the historical data, creating multiple windows.
- **Pattern Detection Feedback**: The app displays a count of detected patterns, providing users with immediate insights.

### 7. Predictions
The model predicts the presence of patterns on new data (typically the past year of stock data). Each prediction labels a window as either containing a *Cup and Handle* pattern or not.

---

### 8. Visualization
When a pattern is detected, the app generates a candlestick chart showing the stock price action within the identified window:
- **Date Range**: Start and end dates of the pattern.
- **Price Range**: Minimum and maximum prices within the detected pattern.
- **Candlestick Chart**: Plotted using `mplfinance`, providing a visual confirmation of the *Cup and Handle* pattern.

### Example Output
If a pattern is detected, the app will display:
- **Pattern Dates**: Start and end dates for the most recent pattern.
- **Price Range**: High and low prices within the pattern period.
- **Candlestick Chart**: A visual representation of the detected pattern for further analysis.

If no pattern is detected, a message will inform the user accordingly.

---
