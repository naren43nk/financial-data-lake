import yfinance as yf
import pandas as pd
import duckdb
import os
from datetime import datetime

tickers = ['AAPL', 'MSFT', 'GOOGL']
start_date = "2020-01-01"
end_date = datetime.today().strftime('%Y-%m-%d'
)
sector_map = {
    'AAPL': 'Technology',
    'MSFT': 'Technology',
    'GOOGL': 'Communication Services'
}

db_path = "data/market_data.duckdb"
if os.path.exists(db_path):
    os.remove(db_path)

def fetch_stock_data(ticker):
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date, auto_adjust=False)

    if df.empty:
        print(f"❌ No data for {ticker}")
        return None

    df.reset_index(inplace=True)

    df.columns = [col.lower().replace(" ", "_") for col in df.columns]

    required = ['date', 'open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required):
        print(f"❌ Missing required cols for {ticker}")
        return None

    if 'adj_close' not in df.columns:
        # yfinance sometimes omits it, so create fallback
        df['adj_close'] = df['close']

    df['ticker'] = ticker
    df['sector'] = sector_map[ticker]

    return df[['date', 'open', 'high', 'low', 'close', 'adj_close', 'volume', 'ticker', 'sector']]

frames = [fetch_stock_data(t) for t in tickers]
valid = [df for df in frames if df is not None]

if not valid:
    raise ValueError("❌ No valid data fetched.")

final_df = pd.concat(valid, ignore_index=True)
print(f"✅ Final shape: {final_df.shape}")
print(f"✅ Columns: {final_df.columns.tolist()}")

con = duckdb.connect(db_path)
con.execute("CREATE TABLE stock_prices AS SELECT * FROM final_df")
con.close()

print("✅ ETL complete and saved to DuckDB.")
