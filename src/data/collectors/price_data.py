"""
Module for collecting historical price data from various sources.
"""

import os
import time
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
from src.utils import ensure_dir_exists, INDICES

# Setup logging
logger = logging.getLogger('stock_screener.data.price')

class PriceDataCollector:
    """
    Class for collecting and managing historical price data.
    """
    
    def __init__(self, data_dir='data'):
        """
        Initialize the price data collector.
        
        Args:
            data_dir (str, optional): Base directory for data storage. Defaults to 'data'.
        """
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, 'raw', 'prices')
        self.processed_dir = os.path.join(data_dir, 'processed', 'prices')
        
        # Ensure directories exist
        ensure_dir_exists(self.raw_dir)
        ensure_dir_exists(self.processed_dir)
    
    def get_index_tickers(self, index_name):
        """
        Get the list of tickers for a given index.
        
        Args:
            index_name (str): Name of the index (key in INDICES dict)
            
        Returns:
            list: List of ticker symbols
        """
        if index_name not in INDICES:
            logger.error(f"Index {index_name} not found in supported indices")
            return []
        
        index_symbol = INDICES[index_name]
        
        # For now, this is a simplified approach that works for some indices
        # In a real implementation, we would use a more robust method to get constituents
        try:
            if index_name == 'SP500':
                # S&P 500 tickers
                return self._get_sp500_tickers()
            elif index_name == 'NASDAQ':
                # NASDAQ 100 tickers
                return self._get_nasdaq100_tickers()
            else:
                # For other indices, we would need to implement custom methods
                # or use external data providers
                logger.warning(f"Getting constituents for {index_name} is not fully implemented")
                return []
        except Exception as e:
            logger.error(f"Failed to get tickers for {index_name}: {e}")
            return []
    
    def _get_sp500_tickers(self):
        """
        Get the list of S&P 500 tickers.
        
        Returns:
            list: List of S&P 500 ticker symbols
        """
        try:
            # Using Wikipedia table as a source
            table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
            df = table[0]
            return df['Symbol'].str.replace('.', '-').tolist()
        except Exception as e:
            logger.error(f"Failed to get S&P 500 tickers: {e}")
            return []
    
    def _get_nasdaq100_tickers(self):
        """
        Get the list of NASDAQ 100 tickers.
        
        Returns:
            list: List of NASDAQ 100 ticker symbols
        """
        try:
            # Using Wikipedia table as a source
            table = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')
            df = table[4]  # The table with the constituents
            return df['Ticker'].tolist()
        except Exception as e:
            logger.error(f"Failed to get NASDAQ 100 tickers: {e}")
            return []
    
    def download_price_data(self, tickers, start_date=None, end_date=None, interval='1d'):
        """
        Download historical price data for a list of tickers.
        
        Args:
            tickers (list): List of ticker symbols
            start_date (str, optional): Start date in YYYY-MM-DD format. Defaults to 5 years ago.
            end_date (str, optional): End date in YYYY-MM-DD format. Defaults to today.
            interval (str, optional): Data interval. Defaults to '1d'.
            
        Returns:
            dict: Dictionary with ticker symbols as keys and DataFrames as values
        """
        # Set default dates if not provided
        if end_date is None:
            end_date = datetime.today().strftime('%Y-%m-%d')
        if start_date is None:
            start_date = (datetime.today() - timedelta(days=5*365)).strftime('%Y-%m-%d')
        
        logger.info(f"Downloading price data for {len(tickers)} tickers from {start_date} to {end_date}")
        
        # Download data for each ticker
        price_data = {}
        for i, ticker in enumerate(tickers):
            try:
                # Log progress
                if i % 10 == 0:
                    logger.info(f"Downloading data for ticker {i+1}/{len(tickers)}")
                
                # Download data using yfinance
                data = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)
                
                if len(data) > 0:
                    price_data[ticker] = data
                else:
                    logger.warning(f"No data returned for {ticker}")
                
                # Add a small delay to avoid hitting API rate limits
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Failed to download data for {ticker}: {e}")
        
        logger.info(f"Successfully downloaded data for {len(price_data)} tickers")
        return price_data
    
    def save_price_data(self, price_data, filename='price_data.pkl'):
        """
        Save price data to disk.
        
        Args:
            price_data (dict): Dictionary with ticker symbols as keys and DataFrames as values
            filename (str, optional): Filename to save data. Defaults to 'price_data.pkl'.
        """
        filepath = os.path.join(self.raw_dir, filename)
        try:
            pd.to_pickle(price_data, filepath)
            logger.info(f"Price data saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save price data: {e}")
    
    def load_price_data(self, filename='price_data.pkl'):
        """
        Load price data from disk.
        
        Args:
            filename (str, optional): Filename to load data from. Defaults to 'price_data.pkl'.
            
        Returns:
            dict: Dictionary with ticker symbols as keys and DataFrames as values
        """
        filepath = os.path.join(self.raw_dir, filename)
        try:
            price_data = pd.read_pickle(filepath)
            logger.info(f"Price data loaded from {filepath}")
            return price_data
        except Exception as e:
            logger.error(f"Failed to load price data: {e}")
            return {}
    
    def process_price_data(self, price_data):
        """
        Process raw price data to calculate returns and other derived metrics.
        
        Args:
            price_data (dict): Dictionary with ticker symbols as keys and DataFrames as values
            
        Returns:
            dict: Dictionary with processed data
        """
        processed_data = {}
        
        for ticker, data in price_data.items():
            try:
                # Make sure we have the expected columns
                if 'Adj Close' not in data.columns:
                    logger.warning(f"Missing 'Adj Close' column for {ticker}")
                    continue
                
                # Extract adjusted close prices
                prices = data['Adj Close']
                
                # Calculate returns
                daily_returns = prices.pct_change().dropna()
                
                # Calculate rolling returns
                weekly_returns = prices.resample('W').last().pct_change().dropna()
                monthly_returns = prices.resample('M').last().pct_change().dropna()
                
                # Calculate rolling volatility (21-day window)
                volatility = daily_returns.rolling(window=21).std() * np.sqrt(252)  # Annualized
                
                # Store processed data
                processed_data[ticker] = {
                    'prices': prices,
                    'daily_returns': daily_returns,
                    'weekly_returns': weekly_returns,
                    'monthly_returns': monthly_returns,
                    'volatility': volatility
                }
            except Exception as e:
                logger.error(f"Failed to process data for {ticker}: {e}")
        
        logger.info(f"Processed price data for {len(processed_data)} tickers")
        return processed_data
    
    def save_processed_data(self, processed_data, filename='processed_price_data.pkl'):
        """
        Save processed price data to disk.
        
        Args:
            processed_data (dict): Dictionary with processed price data
            filename (str, optional): Filename to save data. Defaults to 'processed_price_data.pkl'.
        """
        filepath = os.path.join(self.processed_dir, filename)
        try:
            pd.to_pickle(processed_data, filepath)
            logger.info(f"Processed price data saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save processed price data: {e}")
    
    def load_processed_data(self, filename='processed_price_data.pkl'):
        """
        Load processed price data from disk.
        
        Args:
            filename (str, optional): Filename to load data from. Defaults to 'processed_price_data.pkl'.
            
        Returns:
            dict: Dictionary with processed price data
        """
        filepath = os.path.join(self.processed_dir, filename)
        try:
            processed_data = pd.read_pickle(filepath)
            logger.info(f"Processed price data loaded from {filepath}")
            return processed_data
        except Exception as e:
            logger.error(f"Failed to load processed price data: {e}")
            return {}
    
    def get_latest_data(self, tickers, lookback_days=365*2):
        """
        Get the latest price data for a list of tickers with specified lookback period.
        
        Args:
            tickers (list): List of ticker symbols
            lookback_days (int, optional): Number of days to look back. Defaults to 2 years.
            
        Returns:
            dict: Dictionary with processed data for the specified tickers
        """
        start_date = (datetime.today() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
        price_data = self.download_price_data(tickers, start_date=start_date)
        processed_data = self.process_price_data(price_data)
        return processed_data
