import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from pypfopt import EfficientFrontier, risk_models, expected_returns, plotting
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# PAGE CONFIG
st.set_page_config(layout="wide")
plt.style.use("seaborn-v0_8-darkgrid")

st.title("Smart Stock Portfolio Optimization System")
st.markdown("""
**Machine Learning + Modern Portfolio Theory**
- Uses last **10 years + real-time data**
- Predicts returns using **XGBoost**
- Optimizes portfolio using **Efficient Frontier**
""")
st.divider()

# INPUT SECTION
st.header("Portfolio Inputs")

col1, col2 = st.columns(2)

with col1:
    tickers_input = st.text_input(
        "Stock Tickers (comma separated)",
        placeholder="TCS.NS, INFY.NS, RELIANCE.NS"
    )

with col2:
    capital = st.number_input(
        "Total Capital",
        min_value=1000.0,
        value=100000.0,
        step=1000.0
    )

# FETCH DATA
if tickers_input and capital:

    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    end_date = datetime.today()
    start_date = end_date - timedelta(days=365 * 15)

    with st.spinner("Downloading 15 years of stock data..."):
        adj_close_df = pd.DataFrame()

        for ticker in tickers:
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                auto_adjust=True,
                progress=False
            )
            if not data.empty:
                adj_close_df[ticker] = data["Close"]

    if adj_close_df.empty or len(adj_close_df) < 100:
        st.error("Not enough data. Try different stocks.")
        st.stop()

    adj_close_df.ffill(inplace=True)
    adj_close_df.bfill(inplace=True)

    # UTILITY FUNCTIONS 
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
        df["Lag5"] = df["LogReturn"].shift(5)
        df["MA5"] = df[ticker].rolling(5).mean()
        df["MA10"] = df[ticker].rolling(10).mean()
        df["Vol10"] = df["LogReturn"].rolling(10).std()
        df["RSI"] = calculate_rsi(df[ticker])
        df.dropna(inplace=True)

        X = df[["Lag1", "Lag5", "MA5", "MA10", "Vol10", "RSI"]]
        y = df["LogReturn"]
        return X, y

    # ML PREDICTION
    st.subheader("Machine Learning Return Prediction (XGBoost)")

    predicted_returns = {}
    valid_tickers = []
    metrics = []

    for ticker in tickers:
        try:
            X, y = prepare_features(adj_close_df, ticker)

            if len(X) < 50:
                continue

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, shuffle=False
            )

            param_grid = {
                "n_estimators": [50, 100],
                "learning_rate": [0.05, 0.1],
                "max_depth": [3, 5]
            }

            model = XGBRegressor(random_state=42)
            grid = GridSearchCV(
                model,
                param_grid,
                cv=3,
                scoring="neg_mean_squared_error",
                n_jobs=-1
            )
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_

            y_pred = best_model.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            metrics.append({
                "Ticker": ticker,
                "MAE": round(mae, 6),
                "RMSE": round(rmse, 6),
                "R² Score": round(r2, 4)
            })

            latest_return = best_model.predict(X.iloc[-1:].values)[0]
            predicted_returns[ticker] = latest_return * 252
            valid_tickers.append(ticker)

        except Exception as e:
            st.warning(f"{ticker} skipped")

    if not valid_tickers:
        st.error("No valid stocks for ML prediction.")
        st.stop()

    st.dataframe(pd.DataFrame(metrics), use_container_width=True)

    # PORTFOLIO OPTIMIZATION 
    st.subheader("Portfolio Optimization (Efficient Frontier)")

    mu = expected_returns.mean_historical_return(adj_close_df[valid_tickers])
    S = risk_models.CovarianceShrinkage(
        adj_close_df[valid_tickers]
    ).ledoit_wolf()

    ef = EfficientFrontier(mu, S)
    ef.max_sharpe(risk_free_rate=0.06)
    weights = ef.clean_weights()

    st.write("**Optimal Portfolio Weights**")
    st.json(weights)

    ret, vol, sharpe = ef.portfolio_performance(risk_free_rate=0.06)

    col1, col2, col3 = st.columns(3)
    col1.metric("Expected Return", f"{ret:.2%}")
    col2.metric("Volatility", f"{vol:.2%}")
    col3.metric("Sharpe Ratio", f"{sharpe:.2f}")

    # DISCRETE ALLOCATION 
    st.subheader("Capital Allocation")

    latest_prices = get_latest_prices(adj_close_df[valid_tickers])
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
    st.write(f"**Remaining Cash:** ₹{leftover:,.2f}")

    # ================= VISUALIZATION =================
    st.subheader("Visual Analysis")

    fig, axs = plt.subplots(2, 2, figsize=(16, 11))

    # Pie Chart
    axs[0, 0].pie(
        [weights[t] for t in valid_tickers],
        labels=valid_tickers,
        autopct="%1.1f%%"
    )
    axs[0, 0].set_title("Portfolio Allocation")

    # Efficient Frontier
    plotting.plot_efficient_frontier(ef, ax=axs[0, 1], show_assets=False)
    axs[0, 1].set_title("Efficient Frontier")

    # Portfolio Performance
    log_returns = np.log(adj_close_df[valid_tickers] /
                         adj_close_df[valid_tickers].shift(1)).dropna()

    portfolio_value = (log_returns.mul(
        [weights[t] for t in valid_tickers], axis=1
    ).sum(axis=1)).cumsum().apply(np.exp)

    axs[1, 0].plot(portfolio_value)
    axs[1, 0].set_title("Historical Portfolio Growth")

    axs[1, 1].axis("off")

    st.pyplot(fig)

