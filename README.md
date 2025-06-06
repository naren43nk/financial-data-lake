# 📊 Financial Market Data Lake

A powerful dashboard built using Streamlit to visualize and forecast stock data with:
- Candlestick + EMA/SMA analysis
- RSI and Bollinger Bands
- Trading volume and LSTM forecast
- CSV export
---

## 🚀 Features

- 🔄 Daily ETL pipeline using `yfinance`
- 🗃️ Local data lake powered by DuckDB
- 📅 Optional automated refresh using `schedule` or Airflow
- 📈 Interactive dashboards via Streamlit (optional)
- ✅ Clean structure for production-ready data engineering

---

## 🧰 Tech Stack

- **Python**
- **yFinance** – for financial data extraction
- **DuckDB** – for local OLAP-style storage
- **Pandas** – for data transformation
- **Airflow** *(optional)* – for orchestration
- **Streamlit** *(optional)* – for dashboarding

---

## 📁 Folder Structure

financial-data-lake/
│
├── data/ # Raw and processed data
├── etl/ # ETL scripts
│ ├── fetch_data.py
│ └── utils.py
├── dags/ # Airflow DAGs
├── notebooks/ # Jupyter notebooks for EDA
├── dashboards/ # Streamlit dashboard
├── requirements.txt # Python dependencies
├── .gitignore
└── README.md


---

## 🏁 How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run ETL manually
python etl/fetch_data.py

# Optional: Run daily with schedule or Airflow



👨‍💻 Author
Narendran Mohan
LinkedIn | GitHub

