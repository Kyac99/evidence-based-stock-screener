"""
Module for collecting historical price data from various sources.
Currently supports Yahoo Finance via yfinance library and Alpha Vantage API.
"""

import os
import logging
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import time
import requests
from pathlib import Path
import configparser
import json

# Setup logging
logger = logging.getLogger('stock_screener.data.collectors.price_data')

class PriceDataCollector:
    """
    Class for collecting and processing historical price data.
    """
    
    def __init__(self, config_path=None):
        """
        Initialize the price data collector.
        
        Args:
            config_path (str, optional): Path to the config file. Defaults to None.
        """
        self.config = self._load_config(config_path)
        self.cache_dir = self._get_cache_dir()
        self.api_keys = self._get_api_keys()
        
    def _load_config(self, config_path):
        """
        Load configuration from the config file.
        
        Args:
            config_path (str, optional): Path to the config file.
            
        Returns:
            configparser.ConfigParser: Config parser object
        """
        if config_path is None:
            # Default config path
            config_path = os.path.join('config', 'config.ini')
        
        config = configparser.ConfigParser()
        
        if os.path.exists(config_path):
            config.read(config_path)
            logger.info(f"Loaded configuration from {config_path}")
        else:
            logger.warning(f"Config file not found at {config_path}, using default configuration")
        
        return config
    
    def _get_cache_dir(self):
        """
        Get the cache directory from config or use default.
        
        Returns:
            str: Path to the cache directory
        """
        cache_dir = 'data/cache'
        
        if self.config.has_section('DATA') and self.config.has_option('DATA', 'CACHE_DIR'):
            cache_dir = self.config.get('DATA', 'CACHE_DIR')
        
        # Create the directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        return cache_dir
    
    def _get_api_keys(self):
        """
        Get API keys from the config file.
        
        Returns:
            dict: Dictionary with API keys
        """
        api_keys = {}
        
        if self.config.has_section('API_KEYS'):
            for key, value in self.config['API_KEYS'].items():
                api_keys[key.upper()] = value
        
        return api_keys
    
    def collect_index_data(self, index_symbol, start_date=None, end_date=None, source='yahoo'):
        """
        Collect historical price data for an index.
        
        Args:
            index_symbol (str): Symbol of the index (e.g., "^GSPC" for S&P 500)
            start_date (str, optional): Start date in format 'YYYY-MM-DD'. Defaults to 1 year ago.
            end_date (str, optional): End date in format 'YYYY-MM-DD'. Defaults to today.
            source (str, optional): Data source, either 'yahoo' or 'alpha_vantage'. Defaults to 'yahoo'.
            
        Returns:
            pd.DataFrame: DataFrame with historical price data
        """
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        logger.info(f"Collecting index data for {index_symbol} from {start_date} to {end_date} using {source}")
        
        if source.lower() == 'yahoo':
            return self._collect_yahoo_index_data(index_symbol, start_date, end_date)
        elif source.lower() == 'alpha_vantage':
            return self._collect_alpha_vantage_index_data(index_symbol, start_date, end_date)
        else:
            logger.error(f"Unsupported data source: {source}")
            return None
    
    def _collect_yahoo_index_data(self, index_symbol, start_date, end_date):
        """
        Collect index data using Yahoo Finance.
        
        Args:
            index_symbol (str): Symbol of the index
            start_date (str): Start date in format 'YYYY-MM-DD'
            end_date (str): End date in format 'YYYY-MM-DD'
            
        Returns:
            pd.DataFrame: DataFrame with historical price data
        """
        try:
            index_data = yf.download(
                index_symbol,
                start=start_date,
                end=end_date,
                progress=False
            )
            
            # Keep only the Adj Close column
            if 'Adj Close' in index_data.columns:
                index_data = index_data['Adj Close']
            
            logger.info(f"Successfully collected {len(index_data)} data points for {index_symbol}")
            return index_data
            
        except Exception as e:
            logger.error(f"Failed to collect Yahoo Finance data for {index_symbol}: {e}")
            return None
    
    def _collect_alpha_vantage_index_data(self, index_symbol, start_date, end_date):
        """
        Collect index data using Alpha Vantage API.
        
        Args:
            index_symbol (str): Symbol of the index
            start_date (str): Start date in format 'YYYY-MM-DD'
            end_date (str): End date in format 'YYYY-MM-DD'
            
        Returns:
            pd.DataFrame: DataFrame with historical price data
        """
        api_key = self.api_keys.get('ALPHA_VANTAGE')
        
        if api_key is None or api_key.lower() == 'none' or api_key == 'votre_clé_alphavantage_ici':
            logger.error("Alpha Vantage API key not found in config")
            return None
        
        try:
            # Alpha Vantage API endpoint for daily time series
            base_url = 'https://www.alphavantage.co/query'
            
            # Parameters for the API request
            params = {
                'function': 'TIME_SERIES_DAILY_ADJUSTED',
                'symbol': index_symbol,
                'outputsize': 'full',
                'apikey': api_key
            }
            
            # Make the API request
            response = requests.get(base_url, params=params)
            data = response.json()
            
            # Parse the response
            if 'Time Series (Daily)' not in data:
                logger.error(f"Invalid response from Alpha Vantage for {index_symbol}: {data}")
                return None
            
            time_series = data['Time Series (Daily)']
            
            # Convert to DataFrame
            df = pd.DataFrame(time_series).T
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            # Extract adjusted close
            df = df['5. adjusted close'].astype(float)
            
            # Filter by date range
            df = df[(df.index >= start_date) & (df.index <= end_date)]
            
            logger.info(f"Successfully collected {len(df)} data points for {index_symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to collect Alpha Vantage data for {index_symbol}: {e}")
            return None
    
    def collect_stock_data(self, tickers, start_date=None, end_date=None, source='yahoo'):
        """
        Collect historical price data for a list of stock tickers.
        
        Args:
            tickers (list): List of stock tickers
            start_date (str, optional): Start date in format 'YYYY-MM-DD'. Defaults to 1 year ago.
            end_date (str, optional): End date in format 'YYYY-MM-DD'. Defaults to today.
            source (str, optional): Data source, either 'yahoo' or 'alpha_vantage'. Defaults to 'yahoo'.
            
        Returns:
            dict: Dictionary with ticker as key and DataFrame with historical price data as value
        """
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        logger.info(f"Collecting stock data for {len(tickers)} tickers from {start_date} to {end_date} using {source}")
        
        stock_data = {}
        
        if source.lower() == 'yahoo':
            for ticker in tickers:
                try:
                    # Use the cache if available
                    cache_file = self._get_cache_file_path(ticker, start_date, end_date, source)
                    
                    if os.path.exists(cache_file) and not self._is_cache_expired(cache_file):
                        # Load data from cache
                        data = self._load_from_cache(cache_file)
                        logger.info(f"Loaded cached data for {ticker}")
                    else:
                        # Download data
                        data = self._collect_yahoo_stock_data(ticker, start_date, end_date)
                        
                        if data is not None:
                            # Save to cache
                            self._save_to_cache(data, cache_file)
                    
                    # Store data if valid
                    if data is not None and not data.empty:
                        stock_data[ticker] = {
                            'prices': data['Adj Close'],
                            'volume': data['Volume'] if 'Volume' in data.columns else None,
                            'returns': data['Adj Close'].pct_change().dropna()
                        }
                    
                except Exception as e:
                    logger.error(f"Failed to collect data for {ticker}: {e}")
            
        elif source.lower() == 'alpha_vantage':
            for ticker in tickers:
                try:
                    # Use the cache if available
                    cache_file = self._get_cache_file_path(ticker, start_date, end_date, source)
                    
                    if os.path.exists(cache_file) and not self._is_cache_expired(cache_file):
                        # Load data from cache
                        data = self._load_from_cache(cache_file)
                        logger.info(f"Loaded cached data for {ticker}")
                    else:
                        # Download data
                        data = self._collect_alpha_vantage_stock_data(ticker, start_date, end_date)
                        
                        if data is not None:
                            # Save to cache
                            self._save_to_cache(data, cache_file)
                            
                            # Prevent hitting rate limits
                            time.sleep(0.5)  # Wait half a second between requests
                    
                    # Store data if valid
                    if data is not None and not data.empty:
                        stock_data[ticker] = {
                            'prices': data['5. adjusted close'] if '5. adjusted close' in data.columns else data,
                            'volume': data['6. volume'] if '6. volume' in data.columns else None,
                            'returns': data['5. adjusted close'].pct_change().dropna() if '5. adjusted close' in data.columns else data.pct_change().dropna()
                        }
                    
                except Exception as e:
                    logger.error(f"Failed to collect data for {ticker}: {e}")
        else:
            logger.error(f"Unsupported data source: {source}")
        
        logger.info(f"Successfully collected data for {len(stock_data)} out of {len(tickers)} tickers")
        return stock_data
    
    def _collect_yahoo_stock_data(self, ticker, start_date, end_date):
        """
        Collect stock data using Yahoo Finance.
        
        Args:
            ticker (str): Stock ticker
            start_date (str): Start date in format 'YYYY-MM-DD'
            end_date (str): End date in format 'YYYY-MM-DD'
            
        Returns:
            pd.DataFrame: DataFrame with historical price data
        """
        try:
            stock_data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False
            )
            
            logger.info(f"Successfully collected {len(stock_data)} data points for {ticker}")
            return stock_data
            
        except Exception as e:
            logger.error(f"Failed to collect Yahoo Finance data for {ticker}: {e}")
            return None
    
    def _collect_alpha_vantage_stock_data(self, ticker, start_date, end_date):
        """
        Collect stock data using Alpha Vantage API.
        
        Args:
            ticker (str): Stock ticker
            start_date (str): Start date in format 'YYYY-MM-DD'
            end_date (str): End date in format 'YYYY-MM-DD'
            
        Returns:
            pd.DataFrame: DataFrame with historical price data
        """
        api_key = self.api_keys.get('ALPHA_VANTAGE')
        
        if api_key is None or api_key.lower() == 'none' or api_key == 'votre_clé_alphavantage_ici':
            logger.error("Alpha Vantage API key not found in config")
            return None
        
        try:
            # Alpha Vantage API endpoint for daily time series
            base_url = 'https://www.alphavantage.co/query'
            
            # Parameters for the API request
            params = {
                'function': 'TIME_SERIES_DAILY_ADJUSTED',
                'symbol': ticker,
                'outputsize': 'full',
                'apikey': api_key
            }
            
            # Make the API request
            response = requests.get(base_url, params=params)
            data = response.json()
            
            # Parse the response
            if 'Time Series (Daily)' not in data:
                logger.error(f"Invalid response from Alpha Vantage for {ticker}: {data}")
                return None
            
            time_series = data['Time Series (Daily)']
            
            # Convert to DataFrame
            df = pd.DataFrame(time_series).T
            
            # Convert index to datetime
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            # Convert columns to float
            for col in df.columns:
                df[col] = df[col].astype(float)
            
            # Filter by date range
            df = df[(df.index >= start_date) & (df.index <= end_date)]
            
            logger.info(f"Successfully collected {len(df)} data points for {ticker}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to collect Alpha Vantage data for {ticker}: {e}")
            return None
    
    def _get_cache_file_path(self, ticker, start_date, end_date, source):
        """
        Get the path to the cache file for a ticker.
        
        Args:
            ticker (str): Stock ticker
            start_date (str): Start date in format 'YYYY-MM-DD'
            end_date (str): End date in format 'YYYY-MM-DD'
            source (str): Data source
            
        Returns:
            str: Path to the cache file
        """
        cache_subdir = os.path.join(self.cache_dir, source.lower())
        os.makedirs(cache_subdir, exist_ok=True)
        
        file_name = f"{ticker}_{start_date}_{end_date}.pkl"
        return os.path.join(cache_subdir, file_name)
    
    def _save_to_cache(self, data, cache_file):
        """
        Save data to the cache.
        
        Args:
            data (pd.DataFrame): Data to save
            cache_file (str): Path to the cache file
        """
        try:
            data.to_pickle(cache_file)
            logger.info(f"Saved data to cache: {cache_file}")
        except Exception as e:
            logger.error(f"Failed to save data to cache: {e}")
    
    def _load_from_cache(self, cache_file):
        """
        Load data from the cache.
        
        Args:
            cache_file (str): Path to the cache file
            
        Returns:
            pd.DataFrame: Data from the cache
        """
        try:
            data = pd.read_pickle(cache_file)
            return data
        except Exception as e:
            logger.error(f"Failed to load data from cache: {e}")
            return None
    
    def _is_cache_expired(self, cache_file):
        """
        Check if the cache is expired.
        
        Args:
            cache_file (str): Path to the cache file
            
        Returns:
            bool: True if the cache is expired, False otherwise
        """
        expiry_days = 7  # Default expiry days
        
        if self.config.has_section('DATA') and self.config.has_option('DATA', 'EXPIRY_DAYS'):
            try:
                expiry_days = self.config.getint('DATA', 'EXPIRY_DAYS')
            except:
                pass
        
        # Check if cache is in offline mode (never expire)
        if self.config.has_section('DATA') and self.config.has_option('DATA', 'OFFLINE_MODE'):
            try:
                offline_mode = self.config.getboolean('DATA', 'OFFLINE_MODE')
                if offline_mode:
                    return False
            except:
                pass
        
        # Get file modification time
        file_mtime = os.path.getmtime(cache_file)
        file_age = datetime.now() - datetime.fromtimestamp(file_mtime)
        
        # Check if the file is older than expiry_days
        return file_age.days > expiry_days

    def get_index_components(self, index_name, source='wikipedia'):
        """
        Get the list of components for a given index.
        
        Args:
            index_name (str): Name of the index (e.g., "SP500", "NASDAQ100")
            source (str, optional): Source for the components. Can be 'wikipedia', 'csv', or 'api'.
                                   Defaults to 'wikipedia'.
            
        Returns:
            list: List of component tickers
        """
        logger.info(f"Getting components for {index_name} from {source}")
        
        if source.lower() == 'wikipedia':
            return self._get_components_from_wikipedia(index_name)
        elif source.lower() == 'csv':
            return self._get_components_from_csv(index_name)
        elif source.lower() == 'api':
            return self._get_components_from_api(index_name)
        else:
            logger.error(f"Unsupported source for components: {source}")
            return []
    
    def _get_components_from_wikipedia(self, index_name):
        """
        Get index components from Wikipedia.
        
        Args:
            index_name (str): Name of the index
            
        Returns:
            list: List of component tickers
        """
        try:
            if index_name.upper() == 'SP500':
                # S&P 500 components
                tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
                df = tables[0]
                tickers = df['Symbol'].tolist()
                
                # Clean tickers
                tickers = [ticker.replace('.', '-') for ticker in tickers]
                
                return tickers
            elif index_name.upper() == 'NASDAQ100':
                # NASDAQ 100 components
                tables = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')
                df = tables[4]  # The table with the components
                tickers = df['Ticker'].tolist()
                
                return tickers
            elif index_name.upper() == 'EUROSTOXX50':
                # EURO STOXX 50 components
                tables = pd.read_html('https://en.wikipedia.org/wiki/EURO_STOXX_50')
                df = tables[2]  # The table with the components
                tickers = df['Ticker'].tolist()
                
                return tickers
            elif index_name.upper() == 'CAC40':
                # CAC 40 components
                tables = pd.read_html('https://en.wikipedia.org/wiki/CAC_40')
                df = tables[1]  # The table with the components
                tickers = df['Ticker'].tolist()
                
                return tickers
            else:
                logger.error(f"No Wikipedia source defined for index {index_name}")
                return []
                
        except Exception as e:
            logger.error(f"Failed to get components from Wikipedia for {index_name}: {e}")
            return []
    
    def _get_components_from_csv(self, index_name):
        """
        Get index components from a CSV file.
        
        Args:
            index_name (str): Name of the index
            
        Returns:
            list: List of component tickers
        """
        # Get the path to the CSV file from the settings
        settings_path = os.path.join('config', 'screener_settings.json')
        
        try:
            with open(settings_path, 'r') as file:
                settings = json.load(file)
            
            if 'indices' not in settings:
                logger.error("No indices defined in settings")
                return []
            
            # Find the index in settings
            index_settings = None
            for idx, config in settings['indices'].items():
                if idx.upper() == index_name.upper():
                    index_settings = config
                    break
            
            if index_settings is None:
                logger.error(f"Index {index_name} not found in settings")
                return []
            
            # Check if the index has a components file defined
            if index_settings.get('components_source') != 'csv' or 'components_file' not in index_settings:
                logger.error(f"No CSV components file defined for index {index_name}")
                return []
            
            components_file = index_settings['components_file']
            
            # Read the CSV file
            if os.path.exists(components_file):
                df = pd.read_csv(components_file)
                
                if 'ticker' in df.columns:
                    tickers = df['ticker'].tolist()
                    return tickers
                else:
                    logger.error(f"CSV file does not contain a 'ticker' column: {components_file}")
                    return []
            else:
                logger.error(f"Components file not found: {components_file}")
                return []
                
        except Exception as e:
            logger.error(f"Failed to get components from CSV for {index_name}: {e}")
            return []
    
    def _get_components_from_api(self, index_name):
        """
        Get index components from an API.
        
        Args:
            index_name (str): Name of the index
            
        Returns:
            list: List of component tickers
        """
        # This is a placeholder for API implementation
        # Currently not implemented
        logger.error(f"Getting components from API not implemented for {index_name}")
        return []
