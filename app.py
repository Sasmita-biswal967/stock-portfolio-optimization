import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from pypfopt import EfficientFrontier, risk_models, expected_returns, plotting
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

# PAGE CONFIG 
st.set_page_config(layout="wide")
plt.style.use("seaborn-v0_8-darkgrid")

st.title("Stock Portfolio Optimization")
st.markdown("""
**Stock Portfolio Optimizer**
- 10 years + real-time data  
- Lightweight ML prediction  
- Efficient Frontier optimization  
""")
st.divider()

# INPUT SECTION
st.header("Inputs")

col1, col2 = st.columns(2)

with col1:
    tickers_input = st.text_input(
        "US Stock Tickers (comma separated)",
        placeholder="AAPL, MSFT, NVDA, AMZN"
    )

with col2:
    capital = st.number_input(
        "Investment Amount ($)",
        min_value=1000.0,
        value=100000.0,
        step=1000.0
    )

run = st.button("Run Portfolio Optimization")

@st.cache_data(ttl=3600)
def load_stock_data(ticker, start, end):
    return yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False
    )

# FEATURE ENGINEERING
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def prepare_features(df, ticker):
    df = df[[ticker]].copy()
    df["LogReturn"] = np.log(df[ticker] / df[ticker].shift(1))
    df["Lag1"] = df["LogReturn"].shift(1)
    df["MA5"] = df[ticker].rolling(5).mean()
    df["Vol10"] = df["LogReturn"].rolling(10).std()
    df["RSI"] = calculate_rsi(df[ticker])
    df.dropna(inplace=True)

    X = df[["Lag1", "MA5", "Vol10", "RSI"]]
    y = df["LogReturn"]
    return X, y

# MAIN LOGIC 
if run and tickers_input:

    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    if len(tickers) > 6:
        st.warning("Please select up to 6 stocks for faster execution.")
        st.stop()

    end_date = datetime.today()
    start_date = end_date - timedelta(days=365 * 10)

    with st.spinner("Loading stock data..."):
        prices = pd.DataFrame()
        for t in tickers:
            data = load_stock_data(t, start_date, end_date)
            if not data.empty:
                prices[t] = data["Close"]

    prices.ffill(inplace=True)
    prices.bfill(inplace=True)

    if prices.empty or len(prices) < 100:
        st.error("Not enough data.")
        st.stop()

    # ML PREDICTION 
    st.subheader("ML Return Prediction (Fast XGBoost)")

    predicted_returns = {}
    metrics = []

    for t in tickers:
        X, y = prepare_features(prices, t)

        if len(X) < 50:
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        model = XGBRegressor(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            random_state=42
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)

        metrics.append({
            "Ticker": t,
            "MAE": round(mae, 6)
        })

        predicted_returns[t] = model.predict(X.iloc[-1:].values)[0] * 252

    st.dataframe(pd.DataFrame(metrics), use_container_width=True)

    #  PORTFOLIO OPTIMIZATION
    st.subheader("Portfolio Optimization")

    mu = expected_returns.mean_historical_return(prices)
    S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()

    ef = EfficientFrontier(mu, S)
    ef.max_sharpe(risk_free_rate=0.04)
    weights = ef.clean_weights()

    st.write("**Optimal Weights**")
    st.json(weights)

    ret, vol, sharpe = ef.portfolio_performance(risk_free_rate=0.04)

    col1, col2, col3 = st.columns(3)
    col1.metric("Expected Return", f"{ret:.2%}")
    col2.metric("Volatility", f"{vol:.2%}")
    col3.metric("Sharpe Ratio", f"{sharpe:.2f}")

    #  DISCRETE ALLOCATION 
    st.subheader("Capital Allocation")

    latest_prices = get_latest_prices(prices)
    da = DiscreteAllocation(weights, latest_prices, capital)
    allocation, leftover = da.greedy_portfolio()

    alloc_df = pd.DataFrame([
        {
            "Ticker": k,
            "Shares": v,
            "Price": latest_prices[k],
            "Total Value": v * latest_prices[k]
        }
        for k, v in allocation.items()
    ])

    st.dataframe(alloc_df, use_container_width=True)
    st.write(f"**Remaining Cash:** ${leftover:,.2f}")

    # VISUALS
    st.subheader("Visuals")

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(weights.values(), labels=weights.keys(), autopct="%1.1f%%")
    ax.set_title("Portfolio Allocation")
    st.pyplot(fig)

    axs[1, 1].axis("off")

    st.pyplot(fig)

