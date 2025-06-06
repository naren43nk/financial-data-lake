import schedule
import time
from fetch_data import fetch_stock_data, store_to_duckdb
from datetime import datetime
import pandas as pd

def job():
    tickers = ["AAPL", "MSFT", "GOOGL"]
    start = "2020-01-01"
    end = datetime.today().strftime('%Y-%m-%d')

    all_data = pd.DataFrame()
    for ticker in tickers:
        print(f"[{datetime.now()}] Fetching data for {ticker}...")
        df = fetch_stock_data(ticker, start, end)
        all_data = pd.concat([all_data, df], ignore_index=True)

    print(f"[{datetime.now()}] Storing to DuckDB...")
    store_to_duckdb(all_data)
    print("✅ Daily refresh complete.")

# 🔁 Every day at 6 PM
schedule.every(1).minutes.do(job)

print("🕒 Scheduler started. Waiting for the next run...")

# 🟡 Keep the script alive
while True:
    schedule.run_pending()
    time.sleep(60)
