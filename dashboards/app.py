import streamlit as st
import pandas as pd
import duckdb
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

st.set_page_config(page_title="📊 Stock Market Dashboard", layout="wide")

# Load data
try:
    con = duckdb.connect("data/market_data.duckdb")
    df = con.execute("SELECT * FROM stock_prices").fetchdf()
    con.close()
except Exception as e:
    st.error(f"❌ Failed to load data: {e}")
    st.stop()

df['date'] = pd.to_datetime(df['date'])

# Sidebar filters
st.sidebar.header("📊 Filter Options")
sectors = sorted(df['sector'].dropna().unique())
selected_sector = st.sidebar.selectbox("Select Sector", ["All"] + sectors)
if selected_sector != "All":
    df = df[df['sector'] == selected_sector]

tickers = sorted(df['ticker'].unique())
selected_tickers = st.sidebar.multiselect("Select Tickers", tickers, default=[tickers[0]])

min_date = df["date"].min().date()
max_date = df["date"].max().date()
date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)

ema_toggle = st.sidebar.checkbox("📈 Show EMA (20)", value=True)

df_filtered = df[df["ticker"].isin(selected_tickers)].copy()
df_filtered = df_filtered[(df_filtered['date'].dt.date >= date_range[0]) & (df_filtered['date'].dt.date <= date_range[1])]

if df_filtered.empty:
    st.warning("⚠️ No data available for selected filters.")
    st.stop()

def compute_rsi(data, window=14):
    delta = data["close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain, index=data.index).rolling(window=window).mean()
    avg_loss = pd.Series(loss, index=data.index).rolling(window=window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_bollinger(data, window=20, num_std=2):
    sma = data['close'].rolling(window=window).mean()
    std = data['close'].rolling(window=window).std()
    upper_band = sma + (num_std * std)
    lower_band = sma - (num_std * std)
    return sma, upper_band, lower_band

st.title("📈 Stock Dashboard")

for ticker in selected_tickers:
    st.subheader(f"🔹 {ticker} Analysis")
    df_ticker = df_filtered[df_filtered['ticker'] == ticker].copy()
    df_ticker["RSI"] = compute_rsi(df_ticker)
    df_ticker["SMA"], df_ticker["BB_upper"], df_ticker["BB_lower"] = compute_bollinger(df_ticker)
    if ema_toggle:
        df_ticker["EMA"] = df_ticker["close"].ewm(span=20).mean()

    st.markdown("🔦 Candlestick Chart")
    fig_candle = go.Figure(data=[
        go.Candlestick(x=df_ticker['date'],
                       open=df_ticker['open'],
                       high=df_ticker['high'],
                       low=df_ticker['low'],
                       close=df_ticker['close'],
                       name='OHLC')
    ])
    if ema_toggle:
        fig_candle.add_trace(go.Scatter(x=df_ticker["date"], y=df_ticker["EMA"], mode="lines", name="EMA20"))

    fig_candle.update_layout(
        height=500,
        margin=dict(t=10, b=10),
        xaxis_title="Date",
        yaxis_title="Price",
        showlegend=True
    )
    st.plotly_chart(fig_candle, use_container_width=True)

    fig_rsi = px.line(df_ticker, x="date", y="RSI", title="RSI Over Time")
    fig_rsi.add_hline(y=70, line_dash="dot", line_color="red")
    fig_rsi.add_hline(y=30, line_dash="dot", line_color="green")
    st.plotly_chart(fig_rsi, use_container_width=True)

    fig_vol = px.bar(df_ticker, x="date", y="volume", title="Trading Volume Over Time")
    st.plotly_chart(fig_vol, use_container_width=True)

# --- LSTM Forecast ---
if "AAPL" in selected_tickers:
    st.subheader("🔮 AAPL Price Forecast")
    df_aapl = df[df['ticker'] == 'AAPL'].sort_values("date")
    scaler = MinMaxScaler()
    scaled_close = scaler.fit_transform(df_aapl['close'].values.reshape(-1, 1))

    X, y = [], []
    lookback = 30
    for i in range(lookback, len(scaled_close) - 5):
        X.append(scaled_close[i - lookback:i])
        y.append(scaled_close[i:i+5])
    X, y = np.array(X), np.array(y)

    model = Sequential([
        LSTM(50, return_sequences=False, input_shape=(lookback, 1)),
        Dropout(0.2),
        Dense(5)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)

    last_input = scaled_close[-lookback:]
    last_input = last_input.reshape(1, lookback, 1)
    pred_scaled = model.predict(last_input)[0]
    pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()

    future_dates = pd.date_range(start=df_aapl['date'].max() + pd.Timedelta(days=1), periods=5)
    fig_forecast = px.line(x=future_dates, y=pred, labels={'x': 'Date', 'y': 'Predicted Close'}, title="5-Day Price Forecast for AAPL")
    st.plotly_chart(fig_forecast, use_container_width=True)

# CSV Export
st.download_button("Download Filtered Data as CSV", data=df_filtered.to_csv(index=False), file_name="filtered_stock_data.csv")
