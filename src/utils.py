"""
Utilities and helper functions for the stock screener.
"""

import os
import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Setup logging
logger = logging.getLogger('stock_screener')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

# Constants
INDICES = {
    'SP500': '^GSPC',
    'NASDAQ': '^IXIC',
    'EUROSTOXX50': '^STOXX50E',
    'MSCI_WORLD_TECH': 'XLK',  # Using Technology Select Sector SPDR Fund as proxy
    'RUSSELL2000': '^RUT',  # Small cap US
    'EUROSTOXX_SMALL': 'SMEZ.PA',  # iShares EURO STOXX Small ETF as proxy
    'EUROSTOXX_MID': 'DJMC.PA',  # SPDR EURO STOXX Mid ETF as proxy
}

def ensure_dir_exists(directory):
    """
    Ensure the specified directory exists, creating it if necessary.
    
    Args:
        directory (str): Directory path to check/create
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")

def get_trading_days(start_date, end_date=None, calendar='NYSE'):
    """
    Get a list of trading days between start_date and end_date.
    
    Args:
        start_date (str or datetime): Start date
        end_date (str or datetime, optional): End date. Defaults to today.
        calendar (str, optional): Trading calendar to use. Defaults to 'NYSE'.
    
    Returns:
        pd.DatetimeIndex: Trading days
    """
    if end_date is None:
        end_date = datetime.today()
    
    # Convert string dates to datetime if necessary
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
    
    # Use pandas market_calendar if available, otherwise approximate
    try:
        import pandas_market_calendars as mcal
        calendar = mcal.get_calendar(calendar)
        trading_days = calendar.valid_days(start_date=start_date, end_date=end_date)
        return trading_days
    except ImportError:
        # Approximate trading days (excluding weekends)
        all_days = pd.date_range(start=start_date, end=end_date)
        trading_days = all_days[all_days.dayofweek < 5]
        return trading_days

def calculate_returns(prices, period='daily'):
    """
    Calculate returns from price data.
    
    Args:
        prices (pd.Series or pd.DataFrame): Price data
        period (str, optional): Return period. Options: 'daily', 'weekly', 'monthly'. Defaults to 'daily'.
    
    Returns:
        pd.Series or pd.DataFrame: Returns data
    """
    if period == 'daily':
        return prices.pct_change().dropna()
    elif period == 'weekly':
        return prices.resample('W').last().pct_change().dropna()
    elif period == 'monthly':
        return prices.resample('M').last().pct_change().dropna()
    else:
        raise ValueError(f"Invalid period: {period}. Choose from 'daily', 'weekly', 'monthly'")

def calculate_rolling_volatility(returns, window=21):
    """
    Calculate rolling volatility from returns data.
    
    Args:
        returns (pd.Series or pd.DataFrame): Returns data
        window (int, optional): Rolling window size. Defaults to 21 (approx. 1 month).
    
    Returns:
        pd.Series or pd.DataFrame: Rolling volatility
    """
    return returns.rolling(window=window).std() * np.sqrt(252)  # Annualized

def normalize_score(series, min_percentile=0.01, max_percentile=0.99):
    """
    Normalize a series to a 0-100 score, handling outliers with percentile caps.
    
    Args:
        series (pd.Series): Data to normalize
        min_percentile (float, optional): Min percentile to cap. Defaults to 0.01.
        max_percentile (float, optional): Max percentile to cap. Defaults to 0.99.
    
    Returns:
        pd.Series: Normalized scores (0-100)
    """
    # Handle missing values
    series = series.copy()
    
    # Get percentile values for capping
    min_val = series.quantile(min_percentile)
    max_val = series.quantile(max_percentile)
    
    # Cap values at percentiles to handle outliers
    series = series.clip(lower=min_val, upper=max_val)
    
    # Min-max normalization to 0-100 scale
    min_series = series.min()
    max_series = series.max()
    
    if max_series == min_series:
        # If all values are the same, return middle score
        return pd.Series(50, index=series.index)
    
    return ((series - min_series) / (max_series - min_series)) * 100

def winsorize(series, limits=(0.01, 0.01)):
    """
    Winsorize a series (caps extreme values at specified percentiles).
    
    Args:
        series (pd.Series): Data to winsorize
        limits (tuple, optional): (lower percentile, upper percentile). Defaults to (0.01, 0.01).
    
    Returns:
        pd.Series: Winsorized data
    """
    lower_limit = series.quantile(limits[0])
    upper_limit = series.quantile(1 - limits[1])
    return series.clip(lower=lower_limit, upper=upper_limit)

def get_index_constituents(index_symbol):
    """
    Get the constituents of an index.
    
    Args:
        index_symbol (str): Index symbol from the INDICES dictionary
    
    Returns:
        list: List of ticker symbols for index constituents
    """
    # TODO: Implement this function using either:
    # 1. A data provider API
    # 2. Web scraping from a reliable source
    # 3. Local files with index compositions
    
    # For now, return a placeholder message
    logger.warning(f"Getting index constituents for {index_symbol} is not fully implemented")
    return []

def get_sector_industry_data(tickers):
    """
    Get sector and industry classification for a list of tickers.
    
    Args:
        tickers (list): List of ticker symbols
    
    Returns:
        pd.DataFrame: DataFrame with ticker, sector, and industry
    """
    # TODO: Implement this function using a data provider API or web scraping
    
    # For now, return a placeholder DataFrame
    logger.warning("Getting sector/industry data is not fully implemented")
    return pd.DataFrame({
        'ticker': tickers,
        'sector': ['Unknown'] * len(tickers),
        'industry': ['Unknown'] * len(tickers)
    })
