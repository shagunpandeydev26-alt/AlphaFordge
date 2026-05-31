import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime
from stable_baselines3 import PPO
from pathlib import Path
from envs import SingleStockTradingEnv

# --------------------------
# Model Management Section
# --------------------------
MODELS_DIR = Path(__file__).parent.parent / "models"  # src/ and models/ are siblings
env = SingleStockTradingEnv()

def load_model(ticker):
    """Load model from models/{ticker}.zip"""
    model_path = MODELS_DIR / f"{ticker}.zip"
     
    if not model_path.exists():
        st.error(f"No model found for {ticker}!")
        return None

    try:
        model = PPO.load(model_path, env=env)
        st.success(f"Loaded model for {ticker}")
        return  model
        
    except Exception as e:
        st.error(f"Error loading model for {ticker}: {str(e)}")
        return None

# --------------------------
# Streamlit UI
# --------------------------
st.title("Single-Stock RL Trading Agent")

# Get tickers
TICKERS = [f.stem for f in MODELS_DIR.glob("*.zip")]

# Main interface
col1, col2 = st.columns(2)
with col1:
    # Ticker selection
    selected_ticker = st.selectbox("Select stock ticker", TICKERS)

with col2:
    # Portfolio Value Input
    portfolio_value = st.number_input("Portfolio Value ($)", 
                                    min_value=0.0, 
                                    value=100000.0,
                                    step=1000.0)

# Date selection
prediction_date = st.date_input("Prediction date", 
                              value=datetime.today(), 
                              min_value=datetime(2023, 1, 1), 
                              max_value=datetime.today())

# Load model when ticker is selected
if selected_ticker:
    if 'loaded_model' not in st.session_state or st.session_state.current_ticker != selected_ticker:
        with st.spinner(f"Loading {selected_ticker} model..."):
            model = load_model(selected_ticker)
            if model:
                st.session_state.loaded_model = model
                st.session_state.current_ticker = selected_ticker

# Historical Data
st.header(f"{selected_ticker} Historical Trends")
hist_start = st.date_input("History start date", 
                         value=prediction_date - pd.DateOffset(months=6),
                         max_value=prediction_date)

if selected_ticker:
    data = yf.download(selected_ticker, start=hist_start, end=prediction_date)['Adj Close']
    st.line_chart(data)

# Prediction Section
st.header("Trading Recommendation")
if st.button("Generate Prediction"):
    if 'loaded_model' not in st.session_state:
        st.error("Model failed to load!")
    else:
        # prediction logic here, defined variables :
        # - st.session_state.loaded_model
        # - selected_ticker
        # - prediction_date
        # - portfolio_value
        
        # Example prediction output
        st.success(f"Prediction for {selected_ticker} on {prediction_date}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Recommended Action", "BUY")
        with col2:
            st.metric("Expected Return", "+2.8%")
        with col3:
            st.metric("Confidence Level", "82%")

        st.subheader("Position Sizing")
        st.write(f"Suggested investment: ${portfolio_value * 0.15:,.2f} (15% of portfolio)")
        st.write("Risk management: Stop-loss at 5% below entry")