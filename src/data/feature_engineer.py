# Contains functions to generate technical indicators and other features for RL training.

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import ta


class FeatureEngineer:
    """
    Feature engineering class for generating technical indicators and other features
    used in RL trading model training.
    """
    
    def __init__(self):
        self.feature_columns = []
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add comprehensive technical indicators to the dataframe.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added technical indicators
        """
        df = df.copy()
        
        # Price-based indicators
        df = self._add_price_indicators(df)
        
        # Volume-based indicators  
        df = self._add_volume_indicators(df)
        
        # Momentum indicators
        df = self._add_momentum_indicators(df)
        
        # Volatility indicators
        df = self._add_volatility_indicators(df)
        
        # Trend indicators
        df = self._add_trend_indicators(df)
        
        return df
    
    def _add_price_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based technical indicators."""
        # Moving averages
        for period in [5, 10, 20, 50, 200]:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
        
        # Price ratios
        df['price_to_sma_20'] = df['close'] / df['sma_20']
        df['sma_5_to_sma_20'] = df['sma_5'] / df['sma_20']
        
        # High-Low indicators
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_to_high'] = df['close'] / df['high']
        df['close_to_low'] = df['close'] / df['low']
        
        return df
    
    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators."""
        # Volume moving averages
        df['volume_sma_10'] = df['volume'].rolling(window=10).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_10']
        
        # Volume-price indicators
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        
        # On-Balance Volume
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
        
        return df
    
    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators."""
        # RSI
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_histogram'] = macd.macd_diff()
        
        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # Williams %R
        df['williams_r'] = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close']).williams_r()
        
        return df
    
    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicators."""
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['close'])
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Average True Range
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
        
        # Historical volatility
        df['volatility_10'] = df['close'].pct_change().rolling(window=10).std() * np.sqrt(252)
        df['volatility_30'] = df['close'].pct_change().rolling(window=30).std() * np.sqrt(252)
        
        return df
    
    def _add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend indicators."""
        # ADX (Average Directional Index)
        adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
        df['adx'] = adx.adx()
        df['di_plus'] = adx.adx_pos()
        df['di_minus'] = adx.adx_neg()
        
        # Aroon
        aroon = ta.trend.AroonIndicator(df['high'], df['low'])
        df['aroon_up'] = aroon.aroon_up()
        df['aroon_down'] = aroon.aroon_down()
        df['aroon_indicator'] = aroon.aroon_indicator()
        
        # Parabolic SAR
        df['sar'] = ta.trend.PSARIndicator(df['high'], df['low'], df['close']).psar()
        
        return df
    
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        df = df.copy()
        
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        
        # Time features
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['is_month_start'] = df.index.is_month_start.astype(int)
        df['is_month_end'] = df.index.is_month_end.astype(int)
        df['is_quarter_start'] = df.index.is_quarter_start.astype(int)
        df['is_quarter_end'] = df.index.is_quarter_end.astype(int)
        
        return df
    
    def add_lag_features(self, df: pd.DataFrame, columns: List[str], 
                        lags: List[int] = [1, 2, 3, 5]) -> pd.DataFrame:
        """Add lagged features for specified columns."""
        df = df.copy()
        
        for col in columns:
            if col in df.columns:
                for lag in lags:
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        return df
    
    def add_rolling_features(self, df: pd.DataFrame, columns: List[str],
                           windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """Add rolling statistics for specified columns."""
        df = df.copy()
        
        for col in columns:
            if col in df.columns:
                for window in windows:
                    df[f'{col}_mean_{window}'] = df[col].rolling(window=window).mean()
                    df[f'{col}_std_{window}'] = df[col].rolling(window=window).std()
                    df[f'{col}_min_{window}'] = df[col].rolling(window=window).min()
                    df[f'{col}_max_{window}'] = df[col].rolling(window=window).max()
        
        return df
    
    def add_market_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market regime detection features."""
        df = df.copy()
        
        # Trend strength
        df['trend_strength'] = abs(df['close'].pct_change(20))
        
        # Market regime based on volatility
        vol_20 = df['close'].pct_change().rolling(20).std()
        df['high_vol_regime'] = (vol_20 > vol_20.rolling(100).quantile(0.75)).astype(int)
        df['low_vol_regime'] = (vol_20 < vol_20.rolling(100).quantile(0.25)).astype(int)
        
        # Bull/Bear market indicators
        df['bull_market'] = (df['close'] > df['sma_200']).astype(int)
        df['bear_market'] = (df['close'] < df['sma_200']).astype(int)
        
        return df
    
    def engineer_features(self, df: pd.DataFrame, 
                         include_technical: bool = True,
                         include_time: bool = True,
                         include_lags: bool = True,
                         include_rolling: bool = True,
                         include_regime: bool = True) -> pd.DataFrame:
        """
        Complete feature engineering pipeline.
        
        Args:
            df: Input DataFrame with OHLCV data
            include_technical: Whether to include technical indicators
            include_time: Whether to include time features
            include_lags: Whether to include lag features
            include_rolling: Whether to include rolling features
            include_regime: Whether to include market regime features
            
        Returns:
            DataFrame with engineered features
        """
        engineered_df = df.copy()
        
        if include_technical:
            engineered_df = self.add_technical_indicators(engineered_df)
        
        if include_time:
            engineered_df = self.add_time_features(engineered_df)
        
        if include_lags:
            price_cols = ['close', 'volume', 'rsi'] if include_technical else ['close', 'volume']
            engineered_df = self.add_lag_features(engineered_df, price_cols)
        
        if include_rolling:
            price_cols = ['close', 'volume']
            engineered_df = self.add_rolling_features(engineered_df, price_cols)
        
        if include_regime and include_technical:
            engineered_df = self.add_market_regime_features(engineered_df)
        
        # Remove NaN values
        engineered_df = engineered_df.dropna()
        
        # Store feature columns (exclude original OHLCV)
        original_cols = ['open', 'high', 'low', 'close', 'volume']
        self.feature_columns = [col for col in engineered_df.columns if col not in original_cols]
        
        return engineered_df
    
    def get_feature_importance(self, df: pd.DataFrame, target: str = 'returns') -> pd.DataFrame:
        """
        Calculate feature importance using correlation with target.
        
        Args:
            df: DataFrame with features
            target: Target column name
            
        Returns:
            DataFrame with feature importance scores
        """
        if target not in df.columns:
            df[target] = df['close'].pct_change()
        
        # Calculate correlations
        correlations = df[self.feature_columns].corrwith(df[target]).abs()
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': correlations.index,
            'importance': correlations.values
        }).sort_values('importance', ascending=False)
        
        return importance_df
