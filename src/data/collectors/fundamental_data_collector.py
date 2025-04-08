"""
Module for collecting fundamental financial data for stocks.
Currently supports Alpha Vantage API for basic financials.
"""

import os
import logging
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime
import configparser
import json

# Setup logging
logger = logging.getLogger('stock_screener.data.collectors.fundamental_data')

class FundamentalDataCollector:
    """
    Class for collecting and processing fundamental financial data.
    """
    
    def __init__(self, config_path=None):
        """
        Initialize the fundamental data collector.
        
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
        os.makedirs(os.path.join(cache_dir, 'fundamentals'), exist_ok=True)
        
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
    
    def collect_company_overview(self, tickers):
        """
        Collect company overview data for a list of tickers.
        
        Args:
            tickers (list): List of stock tickers
            
        Returns:
            pd.DataFrame: DataFrame with company overview data
        """
        logger.info(f"Collecting company overview data for {len(tickers)} tickers")
        
        # Check if Alpha Vantage API key is available
        api_key = self.api_keys.get('ALPHA_VANTAGE')
        
        if api_key is None or api_key.lower() == 'none' or api_key == 'votre_clé_alphavantage_ici':
            logger.error("Alpha Vantage API key not found in config")
            return None
        
        results = []
        
        for ticker in tickers:
            try:
                # Use the cache if available
                cache_file = os.path.join(self.cache_dir, 'fundamentals', f"{ticker}_overview.json")
                
                if os.path.exists(cache_file) and not self._is_cache_expired(cache_file):
                    # Load from cache
                    overview = self._load_from_cache(cache_file)
                    logger.info(f"Loaded overview data from cache for {ticker}")
                else:
                    # Alpha Vantage API endpoint for company overview
                    base_url = 'https://www.alphavantage.co/query'
                    
                    # Parameters for the API request
                    params = {
                        'function': 'OVERVIEW',
                        'symbol': ticker,
                        'apikey': api_key
                    }
                    
                    # Make the API request
                    response = requests.get(base_url, params=params)
                    overview = response.json()
                    
                    # Check if the response contains expected data
                    if 'Symbol' not in overview:
                        logger.warning(f"Invalid response from Alpha Vantage for {ticker}: {overview}")
                        continue
                    
                    # Save to cache
                    self._save_to_cache(overview, cache_file)
                    
                    # Respect API rate limits
                    time.sleep(0.5)  # Wait half a second between requests
                
                # Add to results
                results.append(overview)
                
            except Exception as e:
                logger.error(f"Failed to collect overview data for {ticker}: {e}")
        
        if not results:
            logger.error("No overview data collected")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Convert numeric columns to float
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='ignore')
            except:
                pass
        
        logger.info(f"Successfully collected overview data for {len(df)} tickers")
        return df
    
    def collect_income_statement(self, tickers):
        """
        Collect income statement data for a list of tickers.
        
        Args:
            tickers (list): List of stock tickers
            
        Returns:
            dict: Dictionary with ticker as key and DataFrame with income statement data as value
        """
        logger.info(f"Collecting income statement data for {len(tickers)} tickers")
        
        # Check if Alpha Vantage API key is available
        api_key = self.api_keys.get('ALPHA_VANTAGE')
        
        if api_key is None or api_key.lower() == 'none' or api_key == 'votre_clé_alphavantage_ici':
            logger.error("Alpha Vantage API key not found in config")
            return None
        
        income_statements = {}
        
        for ticker in tickers:
            try:
                # Use the cache if available
                cache_file = os.path.join(self.cache_dir, 'fundamentals', f"{ticker}_income.json")
                
                if os.path.exists(cache_file) and not self._is_cache_expired(cache_file):
                    # Load from cache
                    income_data = self._load_from_cache(cache_file)
                    logger.info(f"Loaded income statement data from cache for {ticker}")
                else:
                    # Alpha Vantage API endpoint for income statement
                    base_url = 'https://www.alphavantage.co/query'
                    
                    # Parameters for the API request
                    params = {
                        'function': 'INCOME_STATEMENT',
                        'symbol': ticker,
                        'apikey': api_key
                    }
                    
                    # Make the API request
                    response = requests.get(base_url, params=params)
                    income_data = response.json()
                    
                    # Check if the response contains expected data
                    if 'annualReports' not in income_data and 'quarterlyReports' not in income_data:
                        logger.warning(f"Invalid response from Alpha Vantage for {ticker}: {income_data}")
                        continue
                    
                    # Save to cache
                    self._save_to_cache(income_data, cache_file)
                    
                    # Respect API rate limits
                    time.sleep(0.5)  # Wait half a second between requests
                
                # Process the data
                annual_reports = income_data.get('annualReports', [])
                quarterly_reports = income_data.get('quarterlyReports', [])
                
                # Convert to DataFrame
                annual_df = pd.DataFrame(annual_reports)
                quarterly_df = pd.DataFrame(quarterly_reports)
                
                # Convert numeric columns to float
                for df in [annual_df, quarterly_df]:
                    if not df.empty:
                        for col in df.columns:
                            if col != 'fiscalDateEnding':
                                try:
                                    df[col] = pd.to_numeric(df[col], errors='ignore')
                                except:
                                    pass
                        
                        # Set fiscalDateEnding as index
                        df['fiscalDateEnding'] = pd.to_datetime(df['fiscalDateEnding'])
                        df.set_index('fiscalDateEnding', inplace=True)
                        df.sort_index(inplace=True)
                
                # Store the processed data
                income_statements[ticker] = {
                    'annual': annual_df,
                    'quarterly': quarterly_df
                }
                
            except Exception as e:
                logger.error(f"Failed to collect income statement data for {ticker}: {e}")
        
        logger.info(f"Successfully collected income statement data for {len(income_statements)} tickers")
        return income_statements
    
    def collect_balance_sheet(self, tickers):
        """
        Collect balance sheet data for a list of tickers.
        
        Args:
            tickers (list): List of stock tickers
            
        Returns:
            dict: Dictionary with ticker as key and DataFrame with balance sheet data as value
        """
        logger.info(f"Collecting balance sheet data for {len(tickers)} tickers")
        
        # Check if Alpha Vantage API key is available
        api_key = self.api_keys.get('ALPHA_VANTAGE')
        
        if api_key is None or api_key.lower() == 'none' or api_key == 'votre_clé_alphavantage_ici':
            logger.error("Alpha Vantage API key not found in config")
            return None
        
        balance_sheets = {}
        
        for ticker in tickers:
            try:
                # Use the cache if available
                cache_file = os.path.join(self.cache_dir, 'fundamentals', f"{ticker}_balance.json")
                
                if os.path.exists(cache_file) and not self._is_cache_expired(cache_file):
                    # Load from cache
                    balance_data = self._load_from_cache(cache_file)
                    logger.info(f"Loaded balance sheet data from cache for {ticker}")
                else:
                    # Alpha Vantage API endpoint for balance sheet
                    base_url = 'https://www.alphavantage.co/query'
                    
                    # Parameters for the API request
                    params = {
                        'function': 'BALANCE_SHEET',
                        'symbol': ticker,
                        'apikey': api_key
                    }
                    
                    # Make the API request
                    response = requests.get(base_url, params=params)
                    balance_data = response.json()
                    
                    # Check if the response contains expected data
                    if 'annualReports' not in balance_data and 'quarterlyReports' not in balance_data:
                        logger.warning(f"Invalid response from Alpha Vantage for {ticker}: {balance_data}")
                        continue
                    
                    # Save to cache
                    self._save_to_cache(balance_data, cache_file)
                    
                    # Respect API rate limits
                    time.sleep(0.5)  # Wait half a second between requests
                
                # Process the data
                annual_reports = balance_data.get('annualReports', [])
                quarterly_reports = balance_data.get('quarterlyReports', [])
                
                # Convert to DataFrame
                annual_df = pd.DataFrame(annual_reports)
                quarterly_df = pd.DataFrame(quarterly_reports)
                
                # Convert numeric columns to float
                for df in [annual_df, quarterly_df]:
                    if not df.empty:
                        for col in df.columns:
                            if col != 'fiscalDateEnding':
                                try:
                                    df[col] = pd.to_numeric(df[col], errors='ignore')
                                except:
                                    pass
                        
                        # Set fiscalDateEnding as index
                        df['fiscalDateEnding'] = pd.to_datetime(df['fiscalDateEnding'])
                        df.set_index('fiscalDateEnding', inplace=True)
                        df.sort_index(inplace=True)
                
                # Store the processed data
                balance_sheets[ticker] = {
                    'annual': annual_df,
                    'quarterly': quarterly_df
                }
                
            except Exception as e:
                logger.error(f"Failed to collect balance sheet data for {ticker}: {e}")
        
        logger.info(f"Successfully collected balance sheet data for {len(balance_sheets)} tickers")
        return balance_sheets
    
    def collect_cash_flow(self, tickers):
        """
        Collect cash flow data for a list of tickers.
        
        Args:
            tickers (list): List of stock tickers
            
        Returns:
            dict: Dictionary with ticker as key and DataFrame with cash flow data as value
        """
        logger.info(f"Collecting cash flow data for {len(tickers)} tickers")
        
        # Check if Alpha Vantage API key is available
        api_key = self.api_keys.get('ALPHA_VANTAGE')
        
        if api_key is None or api_key.lower() == 'none' or api_key == 'votre_clé_alphavantage_ici':
            logger.error("Alpha Vantage API key not found in config")
            return None
        
        cash_flows = {}
        
        for ticker in tickers:
            try:
                # Use the cache if available
                cache_file = os.path.join(self.cache_dir, 'fundamentals', f"{ticker}_cashflow.json")
                
                if os.path.exists(cache_file) and not self._is_cache_expired(cache_file):
                    # Load from cache
                    cash_flow_data = self._load_from_cache(cache_file)
                    logger.info(f"Loaded cash flow data from cache for {ticker}")
                else:
                    # Alpha Vantage API endpoint for cash flow
                    base_url = 'https://www.alphavantage.co/query'
                    
                    # Parameters for the API request
                    params = {
                        'function': 'CASH_FLOW',
                        'symbol': ticker,
                        'apikey': api_key
                    }
                    
                    # Make the API request
                    response = requests.get(base_url, params=params)
                    cash_flow_data = response.json()
                    
                    # Check if the response contains expected data
                    if 'annualReports' not in cash_flow_data and 'quarterlyReports' not in cash_flow_data:
                        logger.warning(f"Invalid response from Alpha Vantage for {ticker}: {cash_flow_data}")
                        continue
                    
                    # Save to cache
                    self._save_to_cache(cash_flow_data, cache_file)
                    
                    # Respect API rate limits
                    time.sleep(0.5)  # Wait half a second between requests
                
                # Process the data
                annual_reports = cash_flow_data.get('annualReports', [])
                quarterly_reports = cash_flow_data.get('quarterlyReports', [])
                
                # Convert to DataFrame
                annual_df = pd.DataFrame(annual_reports)
                quarterly_df = pd.DataFrame(quarterly_reports)
                
                # Convert numeric columns to float
                for df in [annual_df, quarterly_df]:
                    if not df.empty:
                        for col in df.columns:
                            if col != 'fiscalDateEnding':
                                try:
                                    df[col] = pd.to_numeric(df[col], errors='ignore')
                                except:
                                    pass
                        
                        # Set fiscalDateEnding as index
                        df['fiscalDateEnding'] = pd.to_datetime(df['fiscalDateEnding'])
                        df.set_index('fiscalDateEnding', inplace=True)
                        df.sort_index(inplace=True)
                
                # Store the processed data
                cash_flows[ticker] = {
                    'annual': annual_df,
                    'quarterly': quarterly_df
                }
                
            except Exception as e:
                logger.error(f"Failed to collect cash flow data for {ticker}: {e}")
        
        logger.info(f"Successfully collected cash flow data for {len(cash_flows)} tickers")
        return cash_flows
    
    def _save_to_cache(self, data, cache_file):
        """
        Save data to the cache.
        
        Args:
            data: Data to save
            cache_file (str): Path to the cache file
        """
        try:
            with open(cache_file, 'w') as file:
                json.dump(data, file)
            logger.info(f"Saved data to cache: {cache_file}")
        except Exception as e:
            logger.error(f"Failed to save data to cache: {e}")
    
    def _load_from_cache(self, cache_file):
        """
        Load data from the cache.
        
        Args:
            cache_file (str): Path to the cache file
            
        Returns:
            Data from the cache
        """
        try:
            with open(cache_file, 'r') as file:
                data = json.load(file)
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
    
    def collect_earnings_estimates(self, tickers):
        """
        Collect earnings estimates for a list of tickers.
        Note: This is a simplified implementation since Alpha Vantage doesn't provide analyst estimates.
        
        Args:
            tickers (list): List of stock tickers
            
        Returns:
            pd.DataFrame: DataFrame with earnings estimates
        """
        logger.info(f"Collecting earnings estimates for {len(tickers)} tickers")
        
        # This is a placeholder for earnings estimates
        # In a real implementation, you would use a service like Refinitiv, FactSet, or Bloomberg
        # Here we'll use the company overview data to get some basic info
        
        company_data = self.collect_company_overview(tickers)
        
        if company_data is None or company_data.empty:
            logger.error("Failed to collect company data for estimates")
            return None
        
        # Extract relevant columns for estimates
        estimate_columns = ['Symbol', 'Name', 'Sector', 'Industry', 'EPS', 'EPSGrowth', 'PERatio', 'PEGRatio', 'AnalystTargetPrice']
        
        # Check which columns are actually available
        available_columns = [col for col in estimate_columns if col in company_data.columns]
        
        if not available_columns:
            logger.error("No estimate-related columns found in company data")
            return None
        
        estimates_df = company_data[available_columns].copy()
        
        # Rename columns for consistency
        column_mapping = {
            'Symbol': 'ticker',
            'EPS': 'current_eps',
            'EPSGrowth': 'eps_growth',
            'PERatio': 'pe_ratio',
            'PEGRatio': 'peg_ratio',
            'AnalystTargetPrice': 'target_price'
        }
        
        # Rename only columns that exist
        rename_cols = {old: new for old, new in column_mapping.items() if old in estimates_df.columns}
        estimates_df.rename(columns=rename_cols, inplace=True)
        
        # Set ticker as index if it exists
        if 'ticker' in estimates_df.columns:
            estimates_df.set_index('ticker', inplace=True)
        
        logger.info(f"Created simplified earnings estimates for {len(estimates_df)} tickers")
        return estimates_df
    
    def collect_all_fundamentals(self, tickers):
        """
        Collect all fundamental data for a list of tickers.
        
        Args:
            tickers (list): List of stock tickers
            
        Returns:
            dict: Dictionary with different types of fundamental data
        """
        logger.info(f"Collecting all fundamental data for {len(tickers)} tickers")
        
        # Limit the number of tickers to prevent API rate limits
        max_tickers = 5
        if len(tickers) > max_tickers:
            logger.warning(f"Limiting to {max_tickers} tickers to prevent API rate limits")
            tickers = tickers[:max_tickers]
        
        # Collect company overview
        company_overview = self.collect_company_overview(tickers)
        
        # Collect income statement
        income_statements = self.collect_income_statement(tickers)
        
        # Collect balance sheet
        balance_sheets = self.collect_balance_sheet(tickers)
        
        # Collect cash flow
        cash_flows = self.collect_cash_flow(tickers)
        
        # Collect earnings estimates
        earnings_estimates = self.collect_earnings_estimates(tickers)
        
        # Return all data
        fundamental_data = {
            'company_overview': company_overview,
            'income_statements': income_statements,
            'balance_sheets': balance_sheets,
            'cash_flows': cash_flows,
            'earnings_estimates': earnings_estimates
        }
        
        logger.info("Successfully collected all fundamental data")
        return fundamental_data
