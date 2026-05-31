import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import sys

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from stable_baselines3 import PPO
from src.envs import SingleStockTradingEnv
from src.data.data_loader import DataLoader
from src.agents.PPOAgent import TradingPPOAgent
from src.inference.inference import TradingInferenceEngine
from src.utils.metrics import PerformanceMetrics

# --------------------------
# Model Management Section
# --------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent 
MODELS_DIR = BASE_DIR / "models"  # src/ and models/ are siblings

class ModelManager:
    """Handles model loading and management for the UI"""
    
    def __init__(self):
        self.data_loader = DataLoader()
        self.inference_engine = TradingInferenceEngine()
    
    def get_available_tickers(self):
        """Get list of available model tickers"""
        return [f.stem for f in MODELS_DIR.glob("*.zip")]
    
    def load_model(self, ticker, env):
        """Load model from models/{ticker}.zip"""
        model_path = MODELS_DIR / f"{ticker}.zip"

        if not model_path.exists():
            st.error(f"No model found for {ticker}!")
            return None

        try:
            model = PPO.load(model_path, env=env)
            print(f"Loaded model for {ticker}")
            st.success(f"Loaded model for {ticker}")
            return model

        except Exception as e:
            st.error(f"Error loading model for {ticker}: {str(e)}")
            return None

    def predict_action(self, model, env, portfolio_value, num_stock_shares):
        """Predict next action using the model"""
        return self.inference_engine.predict_action(
            model, env, portfolio_value, num_stock_shares
        )

# Initialize model manager
model_manager = ModelManager()

# --------------------------
# Streamlit UI
# --------------------------
st.title("Single-Stock RL Trading Agent")

# Get tickers
TICKERS = model_manager.get_available_tickers()

# Main interface
col1, col2, col3 = st.columns(3)
with col1:
    selected_ticker = st.selectbox("Select stock ticker", TICKERS)

with col2:
    num_stock_shares = st.number_input("Number of shares", 
                                       min_value=0, 
                                       value=0,
                                       step=1)

with col3:
    portfolio_value = st.number_input("Portfolio Value ($)", 
                                      min_value=0.0, 
                                      value=10000.0,
                                      step=100.0)

# Date
prediction_date = datetime.today()

# Update environment when ticker changes
if selected_ticker:
    if 'current_ticker' not in st.session_state or st.session_state.current_ticker != selected_ticker:
        print(f"Updating environment for {selected_ticker}")
        st.session_state.env = SingleStockTradingEnv(
            ticker=selected_ticker, 
            start_date="2019-01-01", 
            end_date=prediction_date.strftime("%Y-%m-%d")
        )
        st.session_state.current_ticker = selected_ticker

    # Load model when ticker changes
    with st.spinner(f"Loading {selected_ticker} model..."):
        model = model_manager.load_model(selected_ticker, st.session_state.env)
        if model:
            st.session_state.loaded_model = model

# Historical Data
st.header(f"{selected_ticker} Historical Trends")
hist_start = st.date_input("History start date", 
                         value=prediction_date - pd.DateOffset(months=24),
                         max_value=prediction_date - pd.DateOffset(months=6))

if selected_ticker:
    data = st.session_state.env.df['close']
    st.line_chart(data)

# Prediction Section
st.header("Trading Recommendation")
if st.button("Generate Prediction"):
    if 'loaded_model' not in st.session_state:
        st.error("Model failed to load!")
    else:
        st.success(f"Prediction for {selected_ticker} on {prediction_date.strftime('%Y-%m-%d')}")
        action = model_manager.predict_action(
            st.session_state.loaded_model, 
            st.session_state.env, 
            portfolio_value, 
            num_stock_shares
        )

        recommendation = "BUY" if action > 0 else ("SELL" if action < 0 else "HOLD")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Recommended Action", recommendation)
        with col2:
            st.metric("Recommended Quantity", action if action >= 0 else -action)
