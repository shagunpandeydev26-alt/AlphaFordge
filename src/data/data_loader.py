"""
Data loading and preprocessing utilities for RL trading
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from datetime import datetime, timedelta
import yfinance as yf
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.config import INDICATORS
from finrl.meta.preprocessor.preprocessors import FeatureEngineer


class StockDataLoader:
    """Handles loading and preprocessing of stock market data"""
    
    def __init__(self, use_finrl: bool = True):
        """
        Initialize data loader
        
        Args:
            use_finrl: Whether to use FinRL's YahooDownloader or yfinance directly
        """
        self.use_finrl = use_finrl
        self.feature_engineer = FeatureEngineer(
            use_technical_indicator=True,
            tech_indicator_list=INDICATORS,
            use_turbulence=True
        )
    
    def load_single_stock_data(self, 
                             ticker: str, 
                             start_date: str, 
                             end_date: str,
                             include_benchmark: bool = True,
                             benchmark_ticker: str = "^GSPC") -> pd.DataFrame:
        """
        Load data for a single stock with optional benchmark
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            include_benchmark: Whether to include benchmark data
            benchmark_ticker: Benchmark ticker (default S&P 500)
            
        Returns:
            DataFrame with stock data and technical indicators
        """
        
        try:
            if self.use_finrl:
                # Use FinRL's downloader
                df_stock = YahooDownloader(
                    start_date=start_date, 
                    end_date=end_date, 
                    ticker_list=[ticker]
                ).fetch_data()
                
                if include_benchmark:
                    df_benchmark = YahooDownloader(
                        start_date=start_date, 
                        end_date=end_date, 
                        ticker_list=[benchmark_ticker]
                    ).fetch_data()
                    
                    # Merge with benchmark
                    df = pd.merge(df_stock, df_benchmark[['date', 'close']], 
                                on='date', suffixes=('', '_benchmark'))
                else:
                    df = df_stock
                    
            else:
                # Use yfinance directly
                df_stock = self._download_with_yfinance(ticker, start_date, end_date)
                
                if include_benchmark:
                    df_benchmark = self._download_with_yfinance(benchmark_ticker, start_date, end_date)
                    df = pd.merge(df_stock, df_benchmark[['date', 'close']], 
                                on='date', suffixes=('', '_benchmark'))
                else:
                    df = df_stock
            
            # Add technical indicators
            df = self.feature_engineer.preprocess_data(df)
            
            return df
            
        except Exception as e:
            print(f"Error loading data for {ticker}: {e}")
            # Fallback to yfinance if FinRL fails
            if self.use_finrl:
                print("Falling back to yfinance...")
                return self._load_with_yfinance_fallback(ticker, start_date, end_date, 
                                                       include_benchmark, benchmark_ticker)
            raise
    
    def load_multiple_stocks_data(self, 
                                tickers: List[str], 
                                start_date: str, 
                                end_date: str,
                                include_benchmark: bool = True,
                                benchmark_ticker: str = "^GSPC") -> pd.DataFrame:
        """
        Load data for multiple stocks
        
        Args:
            tickers: List of stock ticker symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            include_benchmark: Whether to include benchmark data
            benchmark_ticker: Benchmark ticker (default S&P 500)
            
        Returns:
            DataFrame with multiple stocks data and technical indicators
        """
        
        try:
            if self.use_finrl:
                # Use FinRL's downloader
                df_stocks = YahooDownloader(
                    start_date=start_date, 
                    end_date=end_date, 
                    ticker_list=tickers
                ).fetch_data()
                
                if include_benchmark:
                    df_benchmark = YahooDownloader(
                        start_date=start_date, 
                        end_date=end_date, 
                        ticker_list=[benchmark_ticker]
                    ).fetch_data()
                    
                    # Merge with benchmark for each stock
                    dfs = []
                    for ticker in tickers:
                        df_ticker = df_stocks[df_stocks['tic'] == ticker].copy()
                        df_merged = pd.merge(df_ticker, df_benchmark[['date', 'close']], 
                                           on='date', suffixes=('', '_benchmark'))
                        dfs.append(df_merged)
                    
                    df = pd.concat(dfs, ignore_index=True)
                else:
                    df = df_stocks
                    
            else:
                # Use yfinance directly
                dfs = []
                for ticker in tickers:
                    df_ticker = self._download_with_yfinance(ticker, start_date, end_date)
                    dfs.append(df_ticker)
                
                df_stocks = pd.concat(dfs, ignore_index=True)
                
                if include_benchmark:
                    df_benchmark = self._download_with_yfinance(benchmark_ticker, start_date, end_date)
                    
                    dfs_merged = []
                    for ticker in tickers:
                        df_ticker = df_stocks[df_stocks['tic'] == ticker].copy()
                        df_merged = pd.merge(df_ticker, df_benchmark[['date', 'close']], 
                                           on='date', suffixes=('', '_benchmark'))
                        dfs_merged.append(df_merged)
                    
                    df = pd.concat(dfs_merged, ignore_index=True)
                else:
                    df = df_stocks
            
            # Add technical indicators
            df = self.feature_engineer.preprocess_data(df)
            
            return df
            
        except Exception as e:
            print(f"Error loading data for {tickers}: {e}")
            raise
    
    def _download_with_yfinance(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Download data using yfinance directly"""
        
        df = yf.download(ticker, start=start_date, end=end_date)
        df.reset_index(inplace=True)
        
        # Normalize column names
        df.columns = [col.lower() for col in df.columns]
        df['tic'] = ticker
        
        # Ensure required columns exist
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'tic']
        for col in required_cols:
            if col not in df.columns:
                if col == 'date':
                    df['date'] = df.index
                else:
                    raise ValueError(f"Required column '{col}' not found in data")
        
        return df[required_cols]
    
    def _load_with_yfinance_fallback(self, 
                                   ticker: str, 
                                   start_date: str, 
                                   end_date: str,
                                   include_benchmark: bool = True,
                                   benchmark_ticker: str = "^GSPC") -> pd.DataFrame:
        """Fallback data loading using yfinance"""
        
        df_stock = self._download_with_yfinance(ticker, start_date, end_date)
        
        if include_benchmark:
            df_benchmark = self._download_with_yfinance(benchmark_ticker, start_date, end_date)
            df = pd.merge(df_stock, df_benchmark[['date', 'close']], 
                        on='date', suffixes=('', '_benchmark'))
        else:
            df = df_stock
        
        # Add technical indicators
        df = self.feature_engineer.preprocess_data(df)
        
        return df
    
    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate loaded data for completeness and quality
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        
        issues = []
        
        # Check required columns
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'tic']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")
        
        # Check for missing values
        if df.isnull().any().any():
            null_cols = df.columns[df.isnull().any()].tolist()
            issues.append(f"Columns with missing values: {null_cols}")
        
        # Check date range
        if len(df) == 0:
            issues.append("No data found")
        else:
            date_range = df['date'].max() - df['date'].min()
            if date_range.days < 30:
                issues.append(f"Insufficient data range: {date_range.days} days")
        
        # Check for price anomalies
        if 'close' in df.columns:
            if (df['close'] <= 0).any():
                issues.append("Found non-positive prices")
            
            # Check for extreme price movements (>50% in one day)
            daily_returns = df['close'].pct_change().abs()
            if (daily_returns > 0.5).any():
                issues.append("Found extreme daily price movements (>50%)")
        
        return len(issues) == 0, issues
    
    def get_data_summary(self, df: pd.DataFrame) -> dict:
        """
        Get summary statistics for loaded data
        
        Args:
            df: DataFrame to summarize
            
        Returns:
            Dictionary with summary statistics
        """
        
        summary = {
            'num_records': len(df),
            'date_range': {
                'start': df['date'].min(),
                'end': df['date'].max(),
                'days': (df['date'].max() - df['date'].min()).days
            },
            'tickers': df['tic'].unique().tolist() if 'tic' in df.columns else [],
            'columns': df.columns.tolist()
        }
        
        if 'close' in df.columns:
            summary['price_stats'] = {
                'min': df['close'].min(),
                'max': df['close'].max(),
                'mean': df['close'].mean(),
                'std': df['close'].std()
            }
        
        return summary


# Convenience functions
def load_stock_data(ticker: str, 
                   start_date: str, 
                   end_date: str,
                   include_benchmark: bool = True,
                   use_finrl: bool = True) -> pd.DataFrame:
    """
    Convenience function to load single stock data
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        include_benchmark: Whether to include benchmark data
        use_finrl: Whether to use FinRL's downloader
        
    Returns:
        DataFrame with stock data and technical indicators
    """
    
    loader = StockDataLoader(use_finrl=use_finrl)
    return loader.load_single_stock_data(ticker, start_date, end_date, include_benchmark)


def load_multiple_stocks_data(tickers: List[str], 
                            start_date: str, 
                            end_date: str,
                            include_benchmark: bool = True,
                            use_finrl: bool = True) -> pd.DataFrame:
    """
    Convenience function to load multiple stocks data
    
    Args:
        tickers: List of stock ticker symbols
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        include_benchmark: Whether to include benchmark data
        use_finrl: Whether to use FinRL's downloader
        
    Returns:
        DataFrame with multiple stocks data and technical indicators
    """
    
    loader = StockDataLoader(use_finrl=use_finrl)
    return loader.load_multiple_stocks_data(tickers, start_date, end_date, include_benchmark)


# Alias for backward compatibility
DataLoader = StockDataLoader