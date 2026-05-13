import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Stock Market Forecasting",
    page_icon="📈",
    layout="wide"
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------

st.markdown("""
<style>

/* ONLY TOP METRIC CARDS */

div[data-testid="metric-container"] {

    background: linear-gradient(to right, #11998e, #38ef7d);

    border: none;

    padding: 20px;

    border-radius: 15px;

    box-shadow: 0px 4px 15px rgba(0,0,0,0.3);

}

/* Metric Label */

div[data-testid="metric-container"] label {

    color: white !important;

    font-size: 18px !important;

    font-weight: bold;
}

/* Metric Value */

div[data-testid="metric-container"] > div {

    color: white !important;

    font-size: 28px !important;

    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)
# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.title("📈 AI Stock Market Forecasting Dashboard")

st.markdown("""
### Deep Learning Based Forecasting using LSTM Network
""")

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.header("⚙️ Dashboard Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload Stock CSV File",
    type=['csv']
)

# ---------------------------------------------------
# MODEL ARCHITECTURE
# ---------------------------------------------------

model = Sequential()

model.add(LSTM(100, return_sequences=True, input_shape=(60,5)))
model.add(Dropout(0.3))

model.add(LSTM(100, return_sequences=True))
model.add(Dropout(0.3))

model.add(LSTM(50))
model.add(Dropout(0.3))

model.add(Dense(1))

# LOAD WEIGHTS

model.load_weights("stock_weights.weights.h5")

# ---------------------------------------------------
# PROCESS FILE
# ---------------------------------------------------

if uploaded_file is not None:

    # ---------------------------------------------------
    # READ DATA
    # ---------------------------------------------------

    df = pd.read_csv(uploaded_file)

    # ---------------------------------------------------
    # CLEAN COLUMN NAMES
    # ---------------------------------------------------

    df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']

    # ---------------------------------------------------
    # CLEAN NUMERIC COLUMNS
    # ---------------------------------------------------

    cols = ['Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']

    for col in cols:
        df[col] = df[col].astype(str).str.replace(',', '')
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # ---------------------------------------------------
    # DATE CONVERSION
    # ---------------------------------------------------

    df['Date'] = pd.to_datetime(df['Date'])

    df = df.sort_values('Date')

    df.set_index('Date', inplace=True)

    # ---------------------------------------------------
    # HANDLE MISSING VALUES
    # ---------------------------------------------------

    df.fillna(method='ffill', inplace=True)

    # ---------------------------------------------------
    # FEATURE ENGINEERING
    # ---------------------------------------------------

    df['MA50'] = df['Close'].rolling(50).mean()

    df['MA100'] = df['Close'].rolling(100).mean()

    df['Return'] = df['Close'].pct_change()

    df['Volatility'] = df['Return'].rolling(10).std()

    df.dropna(inplace=True)

    # ---------------------------------------------------
    # DASHBOARD METRICS
    # ---------------------------------------------------

    latest_close = df['Close'].iloc[-1]

    highest_price = df['Close'].max()

    lowest_price = df['Close'].min()

    avg_volume = int(df['Volume'].mean())

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("📌 Latest Price", f"{latest_close:.2f}")

    col2.metric("📈 Highest Price", f"{highest_price:.2f}")

    col3.metric("📉 Lowest Price", f"{lowest_price:.2f}")

    col4.metric("📊 Avg Volume", f"{avg_volume:,}")

    # ---------------------------------------------------
    # DATA PREVIEW
    # ---------------------------------------------------

    st.subheader("📂 Dataset Preview")

    st.dataframe(df.head())

    # ---------------------------------------------------
    # STOCK TREND GRAPH
    # ---------------------------------------------------

    st.subheader("📈 Closing Price Trend")

    fig1, ax1 = plt.subplots(figsize=(14,6))

    ax1.plot(df['Close'])

    ax1.set_xlabel("Date")

    ax1.set_ylabel("Price")

    ax1.set_title("Closing Price Trend")

    st.pyplot(fig1)

    # ---------------------------------------------------
    # MOVING AVERAGE GRAPH
    # ---------------------------------------------------

    st.subheader("📊 Moving Average Analysis")

    fig2, ax2 = plt.subplots(figsize=(14,6))

    ax2.plot(df['Close'], label='Close')

    ax2.plot(df['MA50'], label='MA50')

    ax2.plot(df['MA100'], label='MA100')

    ax2.legend()

    st.pyplot(fig2)

    # ---------------------------------------------------
    # FEATURE SELECTION
    # ---------------------------------------------------

    data = df[['Close', 'Volume', 'MA50', 'MA100', 'Volatility']]

    # ---------------------------------------------------
    # SCALING
    # ---------------------------------------------------

    scaler = MinMaxScaler()

    scaled_data = scaler.fit_transform(data)

    # ---------------------------------------------------
    # CREATE SEQUENCES
    # ---------------------------------------------------

    def create_dataset(data, time_step=60):

        X, y = [], []

        for i in range(time_step, len(data)):

            X.append(data[i-time_step:i])

            y.append(data[i,0])

        return np.array(X), np.array(y)

    X, y = create_dataset(scaled_data)

    # ---------------------------------------------------
    # SPLIT DATA
    # ---------------------------------------------------

    split = int(len(X) * 0.8)

    X_test = X[split:]

    y_test = y[split:]

    # ---------------------------------------------------
    # PREDICTION
    # ---------------------------------------------------

    pred = model.predict(X_test)

    # ---------------------------------------------------
    # INVERSE SCALING
    # ---------------------------------------------------

    close_scaler = MinMaxScaler()

    close_scaler.fit(df[['Close']])

    pred = close_scaler.inverse_transform(pred)

    actual = close_scaler.inverse_transform(y_test.reshape(-1,1))

    # ---------------------------------------------------
    # PREDICTION GRAPH
    # ---------------------------------------------------

    st.subheader("🤖 Actual vs Predicted Price")

    fig3, ax3 = plt.subplots(figsize=(14,6))

    ax3.plot(actual, label='Actual')

    ax3.plot(pred, label='Predicted')

    ax3.set_xlabel("Time")

    ax3.set_ylabel("Price")

    ax3.legend()

    st.pyplot(fig3)

    # ---------------------------------------------------
    # FUTURE PREDICTION
    # ---------------------------------------------------

    st.subheader("🔮 Next Day Forecast")

    last_60 = scaled_data[-60:]

    X_input = last_60.reshape(1, 60, scaled_data.shape[1])

    future_pred = model.predict(X_input)

    future_price = close_scaler.inverse_transform(future_pred)

    st.success(
        f"Predicted Next Closing Price: ₹ {future_price[0][0]:.2f}"
    )

    # ---------------------------------------------------
    # PROJECT SUMMARY
    # ---------------------------------------------------

    st.subheader("📌 Project Summary")

    st.info("""
    This AI system uses LSTM Deep Learning architecture 
    to analyze historical stock market trends and forecast 
    future stock prices using time-series forecasting techniques.
    """)