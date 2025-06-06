import streamlit as st
import pandas as pd
import duckdb
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Set Streamlit layout
st.set_page_config(page_title="Stock Market Dashboard", layout="wide")

# --- Load Data from DuckDB ---
try:
    con = duckdb.connect("data/market_data.duckdb")
    df = con.execute("SELECT * FROM stock_prices").fetchdf()
    con.close()
    st.write("\u2705 Loaded columns:", df.columns.tolist())
except Exception as e:
    st.error(f"\u274c Failed to load data: {e}")
    st.stop()

# Parse date
df['date'] = pd.to_datetime(df['date'])

# --- Sidebar Filters ---
st.sidebar.header("\ud83d\udcca Filter Options")

sectors = sorted(df['sector'].dropna().unique())
selected_sector = st.sidebar.selectbox("Select Sector", ["All"] + sectors)
if selected_sector != "All":
    df = df[df['sector'] == selected_sector]

tickers = sorted(df['ticker'].unique())
selected_tickers = st.sidebar.multiselect("Select Tickers", tickers, default=[tickers[0]])

min_date = df["date"].min().date()
max_date = df["date"].max().date()
date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)

indicator = st.sidebar.radio("Select Moving Average Type", ["SMA", "EMA"])

# --- Filter Data ---
df_filtered = df[df["ticker"].isin(selected_tickers)].copy()
df_filtered = df_filtered[(df_filtered['date'].dt.date >= date_range[0]) & (df_filtered['date'].dt.date <= date_range[1])]

if df_filtered.empty:
    st.warning("\u26a0\ufe0f No data available for selected filters.")
    st.stop()

# --- Indicator Calculations ---
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

st.title("\ud83d\udcc8 Stock Dashboard")

for ticker in selected_tickers:
    st.subheader(f"### \ud83d\udd39 {ticker} Analysis")
    df_ticker = df_filtered[df_filtered['ticker'] == ticker].copy()

    if indicator == "SMA":
        df_ticker["MA"] = df_ticker["close"].rolling(window=20).mean()
    else:
        df_ticker["MA"] = df_ticker["close"].ewm(span=20, adjust=False).mean()

    df_ticker["RSI"] = compute_rsi(df_ticker)
    df_ticker["SMA"], df_ticker["BB_upper"], df_ticker["BB_lower"] = compute_bollinger(df_ticker)

    # --- Candlestick Chart ---
    fig_candle = go.Figure(data=[
        go.Candlestick(
            x=df_ticker['date'],
            open=df_ticker['open'],
            high=df_ticker['high'],
            low=df_ticker['low'],
            close=df_ticker['close'],
            name='Candlestick'
        ),
        go.Scatter(x=df_ticker["date"], y=df_ticker["MA"], line=dict(color='orange'), name=indicator)
    ])
    fig_candle.update_layout(title="Candlestick with MA", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig_candle, use_container_width=True)

    # --- RSI ---
    fig_rsi = px.line(df_ticker, x="date", y="RSI", title="RSI Over Time")
    fig_rsi.add_hline(y=70, line_dash="dot", line_color="red")
    fig_rsi.add_hline(y=30, line_dash="dot", line_color="green")
    st.plotly_chart(fig_rsi, use_container_width=True)

    # --- Volume ---
    fig_vol = px.bar(df_ticker, x="date", y="volume", title="Trading Volume Over Time")
    st.plotly_chart(fig_vol, use_container_width=True)

    # --- Forecast using Ridge ---
    if ticker == "AAPL":
        st.subheader("\ud83d\udd2e AAPL Price Forecast (5 Days)")
        df_aapl = df[df['ticker'] == 'AAPL'].sort_values("date")

        recent_data = df_aapl['close'].rolling(window=7).mean().dropna()
        X = recent_data[:-1].values.reshape(-1, 1)
        y = df_aapl['close'][7:len(recent_data)+1].values

        model = Ridge()
        model.fit(X, y)
        next_input = recent_data.iloc[-1]

        preds = []
        for i in range(5):
            pred = model.predict(np.array([[next_input]]))[0]
            preds.append(pred)
            next_input = (next_input * 6 + pred) / 7  # weighted average for rolling mean

        future_dates = pd.date_range(start=df_aapl['date'].max() + pd.Timedelta(days=1), periods=5)
        fig_forecast = px.line(x=future_dates, y=preds, labels={'x': 'Date', 'y': 'Predicted Close'}, title="5-Day Price Forecast for AAPL")
        st.plotly_chart(fig_forecast, use_container_width=True)

# --- CSV Download ---
st.download_button("Download Filtered Data as CSV", data=df_filtered.to_csv(index=False), file_name="filtered_stock_data.csv")
