# India Market Dashboard

Shows every sector, the stocks inside it, returns (1D/1W/1M/1Y), technicals
(RSI, MACD, moving averages), fundamentals (P/E, P/B, ROE, debt) and news.

## Setup (one time)
1. Install Python 3.10+ from python.org (tick "Add to PATH").
2. Put app.py and requirements.txt in one folder.
3. Open Terminal / Command Prompt in that folder and run:
   pip install -r requirements.txt

## Run
   streamlit run app.py

A browser page opens. Press "Refresh data" in the sidebar to update.

## Optional AI summaries
Get an API key at console.anthropic.com and paste it in the sidebar.
Without a key everything else still works.

## Notes
- Prices come from Yahoo Finance and may be delayed by a few minutes.
- Sector stock lists come from NSE's official sector index files.
- Not investment advice.
