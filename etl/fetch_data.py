# etl/fetch_data.py

import yfinance as yf
import pandas as pd
import duckdb
import os

# Define tickers and sector mapping
tickers = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "GOOGL": "Communication Services"
}

os.makedirs("data", exist_ok=True)

# Fetch stock data for a single ticker
def fetch_stock_data(ticker):
    df = yf.download(ticker, start="2020-01-01", end="2024-12-31", auto_adjust=False)
    if df.empty:
        print(f"⚠️ No data found for {ticker}")
        return pd.DataFrame()
    
    # If columns are multi-index, flatten them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.reset_index(inplace=True)
    df.rename(columns={
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume"
    }, inplace=True)
    
    df["ticker"] = ticker
    df["sector"] = tickers[ticker]
    return df[["date", "open", "high", "low", "close", "adj_close", "volume", "ticker", "sector"]]

# Collect and combine data
all_data = []
for ticker in tickers:
    df = fetch_stock_data(ticker)
    if not df.empty:
        all_data.append(df)

# Save to DuckDB
if all_data:
    final_df = pd.concat(all_data, ignore_index=True)
    print("✅ Final Columns:", final_df.columns.tolist())

    con = duckdb.connect("data/market_data.duckdb")
    con.register("temp_df", final_df)
    con.execute("DROP TABLE IF EXISTS stock_prices")
    con.execute("CREATE TABLE stock_prices AS SELECT * FROM temp_df")
    schema = con.execute("DESCRIBE stock_prices").fetchdf()
    print("📋 Table Schema:\n", schema)
    con.unregister("temp_df")
    con.close()
    print("✅ Data saved to DuckDB.")
else:
    print("❌ No data collected.")
