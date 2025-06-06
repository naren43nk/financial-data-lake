# dashboards/app.py

import streamlit as st
import pandas as pd
import duckdb
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="📈 Stock Analysis Dashboard", layout="wide")

# Load data
try:
    con = duckdb.connect("data/market_data.duckdb")
    df = con.execute("SELECT * FROM stock_prices").fetchdf()
    con.close()
except Exception as e:
    st.error(f"❌ Failed to load data: {e}")
    st.stop()

df["date"] = pd.to_datetime(df["date"])

# Sidebar
st.sidebar.title("🔧 Filters")
sectors = sorted(df['sector'].dropna().unique())
selected_sector = st.sidebar.selectbox("Select Sector", ["All"] + sectors)
if selected_sector != "All":
    df = df[df["sector"] == selected_sector]

tickers = sorted(df["ticker"].unique())
selected_tickers = st.sidebar.multiselect("Select Tickers", tickers, default=[tickers[0]])

ma_type = st.sidebar.radio("Moving Average Type", ["EMA", "SMA"])
ma_window = st.sidebar.slider("Moving Average Window", 5, 50, 20)

min_date, max_date = df["date"].min().date(), df["date"].max().date()
date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date])

df_filtered = df[df["ticker"].isin(selected_tickers)]
df_filtered = df_filtered[
    (df_filtered["date"].dt.date >= date_range[0]) &
    (df_filtered["date"].dt.date <= date_range[1])
]

if df_filtered.empty:
    st.warning("⚠️ No data found for selected filters.")
    st.stop()

# Indicators
def moving_avg(series, window, method="EMA"):
    if method == "EMA":
        return series.ewm(span=window, adjust=False).mean()
    else:
        return series.rolling(window=window).mean()

def compute_rsi(data, window=14):
    delta = data["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# 📊 Multi-Ticker Comparison
st.title("📈 Multi-Ticker Close Price Comparison")
fig_compare = px.line(df_filtered, x="date", y="close", color="ticker", title="Close Price Comparison")
st.plotly_chart(fig_compare, use_container_width=True)

# 🔍 Per-Ticker Analysis
for ticker in selected_tickers:
    st.markdown(f"---\n### 🔍 {ticker} Analysis")
    df_ticker = df_filtered[df_filtered["ticker"] == ticker].copy()
    df_ticker["MA"] = moving_avg(df_ticker["close"], ma_window, ma_type)
    df_ticker["RSI"] = compute_rsi(df_ticker)

    # Candlestick + MA
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df_ticker["date"],
        open=df_ticker["open"],
        high=df_ticker["high"],
        low=df_ticker["low"],
        close=df_ticker["close"],
        name="Candlestick"
    ))
    fig.add_trace(go.Scatter(
        x=df_ticker["date"],
        y=df_ticker["MA"],
        mode='lines',
        name=f"{ma_type} {ma_window}",
        line=dict(color="orange")
    ))
    fig.update_layout(title=f"{ticker} Candlestick with {ma_type}", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig, use_container_width=True)

    # RSI Plot
    st.markdown("#### RSI Indicator")
    fig_rsi = px.line(df_ticker, x="date", y="RSI", title="RSI Over Time")
    fig_rsi.add_hline(y=70, line_dash="dot", line_color="red")
    fig_rsi.add_hline(y=30, line_dash="dot", line_color="green")
    st.plotly_chart(fig_rsi, use_container_width=True)

# 🔮 Forecast (Linear Regression – AAPL only)
if "AAPL" in selected_tickers:
    st.markdown("### 🔮 AAPL Forecast (Linear Regression – Next 5 Days)")
    df_aapl = df[df["ticker"] == "AAPL"].sort_values("date")
    close_prices = df_aapl["close"].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close_prices)

    X, y = [], []
    lookback = 30
    for i in range(lookback, len(scaled) - 5):
        X.append(scaled[i - lookback:i].flatten())
        y.append(scaled[i:i + 5].flatten())
    X, y = np.array(X), np.array(y)

    model = LinearRegression()
    model.fit(X, y)

    last_input = scaled[-lookback:].flatten().reshape(1, -1)
    pred_scaled = model.predict(last_input).flatten()
    pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    future_dates = pd.date_range(start=df_aapl["date"].max() + pd.Timedelta(days=1), periods=5)

    fig_pred = px.line(x=future_dates, y=pred, labels={"x": "Date", "y": "Predicted Close"}, title="5-Day Closing Price Prediction")
    st.plotly_chart(fig_pred, use_container_width=True)

# 🧪 Backtesting Section
st.markdown("### 🧪 Strategy Backtesting: Buy & Hold vs MA Crossover")
bt_ticker = st.selectbox("Ticker for Backtest", tickers, index=0)
df_bt = df[df["ticker"] == bt_ticker].sort_values("date").copy()
df_bt["MA"] = moving_avg(df_bt["close"], ma_window, ma_type)
df_bt.dropna(inplace=True)

df_bt["signal"] = np.where(df_bt["close"] > df_bt["MA"], 1, 0)
df_bt["returns"] = df_bt["close"].pct_change()
df_bt["strategy_returns"] = df_bt["returns"] * df_bt["signal"]
df_bt["cum_buy_hold"] = (1 + df_bt["returns"]).cumprod()
df_bt["cum_strategy"] = (1 + df_bt["strategy_returns"]).cumprod()

fig_bt = go.Figure()
fig_bt.add_trace(go.Scatter(x=df_bt["date"], y=df_bt["cum_buy_hold"], mode="lines", name="Buy & Hold"))
fig_bt.add_trace(go.Scatter(x=df_bt["date"], y=df_bt["cum_strategy"], mode="lines", name=f"{ma_type} Crossover"))
fig_bt.update_layout(title=f"{bt_ticker} – Strategy Backtest", xaxis_title="Date", yaxis_title="Cumulative Returns")
st.plotly_chart(fig_bt, use_container_width=True)

# 📥 Download
st.download_button("📥 Download Filtered Data", data=df_filtered.to_csv(index=False), file_name="filtered_data.csv")
