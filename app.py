import streamlit as st

# Check for yfinance
try:
    import yfinance as yf
except ImportError:
    st.error("The 'yfinance' library is not installed. Please run 'pip install yfinance' and restart the app.")
    st.stop()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import ta
import os

# Check for TensorFlow and catch DLL load errors
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense
    tf_available = True
except (ImportError, OSError):
    tf_available = False

st.set_page_config(layout="wide")
st.title("📈 Stock Price Forecasting with LSTM")

# --- User Inputs ---
stock = st.text_input("Enter Stock Ticker (e.g., NVDA, AAPL):", "NVDA")
forecast_days = st.slider("Forecast Days into the Future:", 1, 30, 7)
use_uploaded = st.checkbox("Upload Your Own CSV File")
indicator_option = st.selectbox("Select Technical Indicator", ["SMA_14", "RSI_14", "MACD"])

# --- Live Price Function ---
@st.cache_data(ttl=300)
def get_live_price(ticker):
    data = yf.download(ticker, period="1d", interval="1m")
    latest = data.iloc[-1]
    return {
        "price": latest["Close"],
        "high": latest["High"],
        "low": latest["Low"],
        "volume": latest["Volume"],
        "time": latest.name
    }

# --- Add Indicators Function ---
def add_technical_indicators(df):
    close_series = pd.Series(df["Close"].values.flatten(), index=df.index)
    df["SMA_14"] = ta.trend.SMAIndicator(close_series, window=14).sma_indicator()
    df["RSI_14"] = ta.momentum.RSIIndicator(close_series, window=14).rsi()
    df["MACD"] = ta.trend.MACD(close_series).macd_diff()
    return df.fillna(0)

# --- Data Loading ---
if use_uploaded:
    file = st.file_uploader("Upload CSV with 'Close' column", type=["csv"])
    if file:
        df = pd.read_csv(file)
        df = df[["Close"]]
else:
    df = yf.download(stock, start="2018-01-01", end=datetime.today().strftime("%Y-%m-%d"))
    df = df[["Close"]]

# --- Prepare Data ---
df = add_technical_indicators(df)

# --- Live Price Display ---
if not use_uploaded:
    live = get_live_price(stock)
    price = live.get("price")
    time = live.get("time")
    try:
        price_display = f"${price:.2f}"
    except Exception:
        price_display = str(price)
    st.metric("Live Price", price_display, help=f"Updated: {time}")

# --- Historical Price Chart ---
st.subheader("Historical Prices")
st.line_chart(df['Close'])

# --- Indicator Plot ---
st.subheader(f"{indicator_option} Indicator")
fig, ax = plt.subplots()
ax.plot(df[indicator_option], label=indicator_option)
ax.legend()
st.pyplot(fig)

# --- RSI Signal ---
latest_rsi = df['RSI_14'].dropna().iloc[-1]
def get_signal(rsi):
    return "BUY" if rsi < 30 else ("SELL" if rsi > 70 else "HOLD")
signal = get_signal(latest_rsi)
st.subheader("RSI-Based Trading Signal")
st.write(f"RSI (14): {latest_rsi:.2f} — {signal}")


# --- Forecast Section ---
seq_len = 60

if not tf_available:
    st.info("TensorFlow not available; using simple moving average fallback for forecasting.")
    values = df["Close"].values
    if len(values) >= seq_len:
        sma = values[-seq_len:].mean()
    else:
        sma = values.mean()
    forecasted = np.array([[sma]] * forecast_days)

    st.subheader("SMA Fallback Forecast")
    st.line_chart(pd.DataFrame(forecasted, columns=["Forecasted_Close"]))
    st.download_button(
        "Download Forecast CSV",
        pd.DataFrame(forecasted, columns=["Forecasted_Close"]).to_csv(index=False),
        file_name=f"{stock}_forecast.csv"
    )
else:
    # Scale and sequence data
    features = ["Close", "SMA_14", "RSI_14", "MACD"]
    scaler_feat = MinMaxScaler().fit(df[features])
    scaler_close = MinMaxScaler().fit(df[["Close"]])
    data_feat = scaler_feat.transform(df[features])

    X, y = [], []
    for i in range(seq_len, len(data_feat)):
        X.append(data_feat[i-seq_len:i])
        y.append(data_feat[i, 0])
    X = np.array(X).reshape(-1, seq_len, len(features))
    y = np.array(y)

    # Load or train model
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", f"{stock}_lstm.h5")
    if os.path.exists(model_path):
        model = load_model(model_path)
        st.write("Loaded pre-trained model.")
    else:
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(seq_len, len(features))),
            LSTM(50),
            Dense(1)
        ])
        model.compile(loss="mean_squared_error", optimizer="adam")
        model.fit(X, y, epochs=10, batch_size=32, verbose=0)
        model.save(model_path)
        st.write("Model trained and saved.")

    # Forecast future
    last_seq = data_feat[-seq_len:]
    preds = []
    seq = last_seq.copy()
    for _ in range(forecast_days):
        p = model.predict(seq.reshape(1, seq_len, len(features)))[0][0]
        preds.append(p)
        seq = np.vstack([seq[1:], [p] + list(seq[-1, 1:])])

    forecasted = scaler_close.inverse_transform(np.array(preds).reshape(-1, 1))

    # Display forecast
    st.subheader("LSTM Forecast")
    st.line_chart(pd.DataFrame(forecasted, columns=["Forecasted_Close"]))
    st.download_button(
        "Download Forecast CSV",
        pd.DataFrame(forecasted, columns=["Forecasted_Close"]).to_csv(index=False),
        file_name=f"{stock}_forecast.csv"
    )
