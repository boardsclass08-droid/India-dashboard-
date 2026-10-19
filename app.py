"""
India Market Dashboard
Sectors -> stocks -> returns, fundamentals, technicals, news, AI summary.
Run:  streamlit run app.py
"""
import io
import xml.etree.ElementTree as ET
from datetime import datetime
from urllib.parse import quote

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="India Market Dashboard", layout="wide")

# ---------------------------------------------------------------- sectors
# NSE publishes constituents of each sector index as a CSV.
BASE = "https://www.niftyindices.com/IndexConstituent/"
SECTOR_FILES = {
    "Banking": "ind_niftybanklist.csv",
    "Private Bank": "ind_nifty_privatebanklist.csv",
    "PSU Bank": "ind_niftyPSUbanklist.csv",
    "Financial Services": "ind_niftyfinancelist.csv",
    "IT": "ind_niftyitlist.csv",
    "Auto": "ind_niftyautolist.csv",
    "FMCG": "ind_niftyfmcglist.csv",
    "Pharma": "ind_niftypharmalist.csv",
    "Healthcare": "ind_niftyhealthcarelist.csv",
    "Metal": "ind_niftymetallist.csv",
    "Energy": "ind_niftyenergylist.csv",
    "Oil & Gas": "ind_niftyoilgaslist.csv",
    "Realty": "ind_niftyrealtylist.csv",
    "Media": "ind_niftymedialist.csv",
    "Consumer Durables": "ind_niftyconsumerdurableslist.csv",
    "Infrastructure": "ind_niftyinfralist.csv",
}
# Backup if the NSE download fails (small sample, not complete)
FALLBACK = {
    "Banking": ["HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK", "KOTAKBANK"],
    "IT": ["TCS", "INFY", "HCLTECH", "WIPRO", "TECHM"],
    "Auto": ["MARUTI", "M&M", "TATAMOTORS", "BAJAJ-AUTO", "EICHERMOT"],
    "FMCG": ["HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "DABUR"],
    "Pharma": ["SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "LUPIN"],
    "Metal": ["TATASTEEL", "JSWSTEEL", "HINDALCO", "VEDL", "COALINDIA"],
    "Energy": ["RELIANCE", "ONGC", "NTPC", "POWERGRID", "BPCL"],
    "Realty": ["DLF", "GODREJPROP", "OBEROIRLTY", "PRESTIGE", "LODHA"],
}
SECTOR_INDEX = {  # Yahoo tickers for sector index performance
    "Banking": "^NSEBANK", "IT": "^CNXIT", "Auto": "^CNXAUTO",
    "FMCG": "^CNXFMCG", "Pharma": "^CNXPHARMA", "Metal": "^CNXMETAL",
    "Energy": "^CNXENERGY", "Realty": "^CNXREALTY", "Media": "^CNXMEDIA",
    "PSU Bank": "^CNXPSUBANK", "Infrastructure": "^CNXINFRA",
    "Financial Services": "^CNXFIN",
}
HEADERS = {"User-Agent": "Mozilla/5.0"}


@st.cache_data(ttl=86400)
def get_constituents(sector: str):
    try:
        r = requests.get(BASE + SECTOR_FILES[sector], headers=HEADERS, timeout=15)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        return df["Symbol"].dropna().tolist(), "NSE official list"
    except Exception:
        return FALLBACK.get(sector, []), "built-in sample list (NSE download failed)"


def yf_sym(s: str) -> str:
    return s.replace("&", "%26") if False else f"{s}.NS"


# ---------------------------------------------------------------- data
@st.cache_data(ttl=300)
def get_prices(symbols, period="1y"):
    tickers = [yf_sym(s) for s in symbols]
    df = yf.download(tickers, period=period, auto_adjust=True,
                     progress=False, group_by="ticker", threads=True)
    return df


def rsi(series, n=14):
    d = series.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def pct(a, b):
    return (a / b - 1) * 100 if b else float("nan")


def summarize(sym, df):
    try:
        c = df[yf_sym(sym)]["Close"].dropna()
        if len(c) < 30:
            return None
        last = c.iloc[-1]
        sma50 = c.rolling(50).mean().iloc[-1]
        sma200 = c.rolling(200).mean().iloc[-1] if len(c) >= 200 else float("nan")
        r = rsi(c).iloc[-1]
        trend = "Bullish" if last > sma50 and (pd.isna(sma200) or last > sma200) \
            else "Bearish" if last < sma50 and (pd.isna(sma200) or last < sma200) else "Mixed"
        return {
            "Symbol": sym, "Price": round(last, 2),
            "1D %": round(pct(last, c.iloc[-2]), 2),
            "1W %": round(pct(last, c.iloc[-6]), 2) if len(c) > 6 else None,
            "1M %": round(pct(last, c.iloc[-22]), 2) if len(c) > 22 else None,
            "1Y %": round(pct(last, c.iloc[0]), 2),
            "RSI": round(r, 1), "Trend": trend,
        }
    except Exception:
        return None


@st.cache_data(ttl=3600)
def get_info(sym):
    try:
        return yf.Ticker(yf_sym(sym)).info
    except Exception:
        return {}


@st.cache_data(ttl=600)
def stock_news(sym):
    try:
        return yf.Ticker(yf_sym(sym)).news or []
    except Exception:
        return []


@st.cache_data(ttl=600)
def google_news(query, n=8):
    url = f"https://news.google.com/rss/search?q={quote(query)}&hl=en-IN&gl=IN&ceid=IN:en"
    try:
        root = ET.fromstring(requests.get(url, headers=HEADERS, timeout=15).content)
        out = []
        for it in root.iter("item"):
            out.append({"title": it.findtext("title"), "link": it.findtext("link"),
                        "date": it.findtext("pubDate")})
            if len(out) >= n:
                break
        return out
    except Exception:
        return []


def ai_explain(prompt):
    key = st.session_state.get("api_key")
    if not key:
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        msg = client.messages.create(
            model="claude-sonnet-5-5", max_tokens=700,
            messages=[{"role": "user", "content": prompt}])
        return msg.content[0].text
    except Exception as e:
        return f"AI error: {e}"


# ---------------------------------------------------------------- UI
st.title("India Market Dashboard")
st.caption(f"Data: Yahoo Finance + NSE lists + Google News. Refreshed {datetime.now():%d %b %Y %H:%M}. "
           "Prices can be delayed. Not investment advice.")

with st.sidebar:
    st.header("Settings")
    st.session_state["api_key"] = st.text_input(
        "Anthropic API key (optional, for AI summaries)", type="password")
    if st.button("Refresh data"):
        st.cache_data.clear()
        st.rerun()

tab1, tab2, tab3, tab4 = st.tabs(["Sectors", "Stocks in sector", "Stock detail", "Market news"])

# ---- Tab 1: sector heatmap
with tab1:
    st.subheader("Sector performance (index returns)")
    rows = []
    idx = list(SECTOR_INDEX.items())
    data = yf.download([t for _, t in idx], period="1y", auto_adjust=True,
                       progress=False, group_by="ticker")
    for name, t in idx:
        try:
            c = data[t]["Close"].dropna()
            rows.append({"Sector": name, "1D %": round(pct(c.iloc[-1], c.iloc[-2]), 2),
                         "1W %": round(pct(c.iloc[-1], c.iloc[-6]), 2),
                         "1M %": round(pct(c.iloc[-1], c.iloc[-22]), 2),
                         "1Y %": round(pct(c.iloc[-1], c.iloc[0]), 2)})
        except Exception:
            pass
    if rows:
        sdf = pd.DataFrame(rows).sort_values("1D %", ascending=False)
        st.dataframe(sdf.style.map(lambda v: "color:green" if v > 0 else "color:red",
                                   subset=["1D %", "1W %", "1M %", "1Y %"]),
                     use_container_width=True, hide_index=True)
        st.bar_chart(sdf.set_index("Sector")["1D %"])
    else:
        st.warning("Could not load sector indices. Check your internet connection.")

# ---- Tab 2: stocks in sector
with tab2:
    sector = st.selectbox("Sector", list(SECTOR_FILES.keys()))
    syms, src = get_constituents(sector)
    st.caption(f"{len(syms)} stocks. Source: {src}")
    if syms:
        with st.spinner("Loading prices..."):
            px = get_prices(syms)
        table = [s for s in (summarize(x, px) for x in syms) if s]
        if table:
            tdf = pd.DataFrame(table)
            st.dataframe(
                tdf.style.map(lambda v: "color:green" if v > 0 else "color:red",
                              subset=["1D %", "1W %", "1M %", "1Y %"]),
                use_container_width=True, hide_index=True)
            st.session_state["sector_syms"] = tdf["Symbol"].tolist()
            st.caption("RSI above 70 = overbought, below 30 = oversold. "
                       "Trend compares price with 50 and 200 day averages.")

# ---- Tab 3: stock detail
with tab3:
    options = st.session_state.get("sector_syms", FALLBACK["IT"])
    sym = st.selectbox("Stock", options)
    custom = st.text_input("Or type any NSE symbol (e.g. TATAPOWER)")
    if custom:
        sym = custom.strip().upper()

    hist = yf.Ticker(yf_sym(sym)).history(period="1y", auto_adjust=True)
    info = get_info(sym)
    if hist.empty:
        st.error("No data found for this symbol.")
    else:
        hist["SMA50"] = hist["Close"].rolling(50).mean()
        hist["SMA200"] = hist["Close"].rolling(200).mean()
        fig = go.Figure([
            go.Candlestick(x=hist.index, open=hist["Open"], high=hist["High"],
                           low=hist["Low"], close=hist["Close"], name="Price"),
            go.Scatter(x=hist.index, y=hist["SMA50"], name="SMA 50"),
            go.Scatter(x=hist.index, y=hist["SMA200"], name="SMA 200"),
        ])
        fig.update_layout(height=450, xaxis_rangeslider_visible=False,
                          margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig, use_container_width=True)

        ema12 = hist["Close"].ewm(span=12).mean()
        ema26 = hist["Close"].ewm(span=26).mean()
        macd = ema12 - ema26
        macd_sig = macd.ewm(span=9).mean()
        tech = {
            "RSI (14)": round(rsi(hist["Close"]).iloc[-1], 1),
            "MACD": "Bullish" if macd.iloc[-1] > macd_sig.iloc[-1] else "Bearish",
            "Price vs SMA50": "Above" if hist["Close"].iloc[-1] > hist["SMA50"].iloc[-1] else "Below",
            "52w High": round(hist["High"].max(), 2),
            "52w Low": round(hist["Low"].min(), 2),
        }
        fund = {
            "Name": info.get("longName"), "Industry": info.get("industry"),
            "Market Cap (Cr)": round(info["marketCap"] / 1e7) if info.get("marketCap") else None,
            "P/E (trailing)": info.get("trailingPE"), "Forward P/E": info.get("forwardPE"),
            "P/B": info.get("priceToBook"),
            "ROE %": round(info["returnOnEquity"] * 100, 1) if info.get("returnOnEquity") else None,
            "Debt/Equity": info.get("debtToEquity"),
            "Profit margin %": round(info["profitMargins"] * 100, 1) if info.get("profitMargins") else None,
            "Revenue growth %": round(info["revenueGrowth"] * 100, 1) if info.get("revenueGrowth") else None,
            "Dividend yield %": info.get("dividendYield"),
        }
        c1, c2 = st.columns(2)
        c1.subheader("Technicals")
        c1.table(pd.DataFrame(tech.items(), columns=["Metric", "Value"]).astype(str))
        c2.subheader("Fundamentals")
        c2.table(pd.DataFrame(fund.items(), columns=["Metric", "Value"]).astype(str))

        st.subheader("News")
        items = google_news(f"{info.get('longName') or sym} share NSE", 6)
        for n in items:
            st.markdown(f"- [{n['title']}]({n['link']})")

        if st.button("AI summary of this stock"):
            headlines = "\n".join(n["title"] for n in items)
            out = ai_explain(
                f"You are an Indian equity analyst. Stock: {sym}.\n"
                f"Technicals: {tech}\nFundamentals: {fund}\nHeadlines:\n{headlines}\n"
                "Give a plain-English summary: strengths, risks, and what the news "
                "means. No buy/sell advice. Under 200 words.")
            st.write(out or "Add your Anthropic API key in the sidebar to enable AI summaries.")

# ---- Tab 4: market-wide news
with tab4:
    st.subheader("News moving the Indian market")
    topics = {
        "RBI & interest rates": "RBI repo rate India",
        "FII / DII flows": "FII DII flows India market",
        "Crude oil & rupee": "crude oil rupee India market",
        "Budget & policy": "India government policy stock market",
        "Global cues": "Wall Street Fed Asia markets India Nifty",
        "Nifty & Sensex": "Nifty Sensex today",
    }
    all_heads = []
    for label, q in topics.items():
        with st.expander(label, expanded=(label == "Nifty & Sensex")):
            for n in google_news(q, 6):
                all_heads.append(n["title"])
                st.markdown(f"- [{n['title']}]({n['link']})")
    if st.button("AI: what matters for Indian stocks today?"):
        out = ai_explain(
            "These are today's headlines affecting the Indian stock market:\n"
            + "\n".join(all_heads[:30])
            + "\nSummarize in simple English which sectors may be affected and why, "
              "in under 200 words. No buy/sell advice.")
        st.write(out or "Add your Anthropic API key in the sidebar to enable AI summaries.")
