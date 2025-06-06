# 📈 Financial Market Dashboard

An interactive **stock analysis dashboard** built using **Streamlit**, **Plotly**, **yFinance**, and **DuckDB**.  
It enables users to explore historical stock trends, compare tickers, run forecasting, and simulate trading strategies in a clean, interactive UI.

---

## 🚀 Features

- 📊 **Multi-Ticker Comparison** – Visualize multiple stock close prices together
- 🔁 **SMA/EMA Toggle** – Choose between **Simple** or **Exponential Moving Averages**
- 🔍 **Technical Indicators** – Candlestick chart, RSI, and Moving Average overlays
- 🔮 **Forecasting** – Predict next 5-day **AAPL closing prices** using lightweight **Linear Regression**
- 🧪 **Backtesting** – Simulate and compare **Buy & Hold** vs **MA Crossover Strategy**
- 📥 **CSV Download** – Export filtered results for offline use

---

## 🧠 Tech Stack

- **Frontend:** [Streamlit](https://streamlit.io/)
- **Data Source:** [yFinance](https://pypi.org/project/yfinance/)
- **Database Engine:** [DuckDB](https://duckdb.org/)
- **ML Forecasting:** `LinearRegression` from `scikit-learn`
- **Plotting:** [Plotly](https://plotly.com/)

---

## 📁 Project Structure

financial-data-lake/
│
├── dashboards/
│ └── app.py # 🚀 Streamlit dashboard (entry point)
├── etl/
│ └── fetch_data.py # 🔄 Script to fetch and store stock data
├── data/ # 📦 Contains market_data.duckdb (auto-created)
├── requirements.txt # 📌 Python dependencies
├── setup.sh # ⚙️ Optional startup script for Streamlit Cloud
└── README.md # 📘 Project documentation

💹 Default Tickers Included
Ticker	Company	Sector
AAPL	Apple Inc.	Technology
MSFT	Microsoft Corp	Technology
GOOGL	Alphabet Inc.	Communication Services