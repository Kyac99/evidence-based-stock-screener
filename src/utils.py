"""
Utility functions used across the stock screener.
"""

import os
import logging
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
import json
import configparser

# Setup logging
logger = logging.getLogger('stock_screener.utils')

def setup_logging(log_level=logging.INFO):
    """
    Setup logging configuration.
    
    Args:
        log_level (int, optional): Logging level. Defaults to logging.INFO.
    """
    # Get log config from config.ini if available
    config_path = os.path.join('config', 'config.ini')
    log_file = 'logs/screener.log'
    rotate = True
    max_size = 5242880  # 5 MB
    backup_count = 3
    
    if os.path.exists(config_path):
        config = configparser.ConfigParser()
        config.read(config_path)
        
        if config.has_section('LOGGING'):
            if config.has_option('LOGGING', 'LEVEL'):
                level_str = config.get('LOGGING', 'LEVEL').upper()
                if level_str == 'DEBUG':
                    log_level = logging.DEBUG
                elif level_str == 'INFO':
                    log_level = logging.INFO
                elif level_str == 'WARNING':
                    log_level = logging.WARNING
                elif level_str == 'ERROR':
                    log_level = logging.ERROR
            
            if config.has_option('LOGGING', 'FILE'):
                log_file = config.get('LOGGING', 'FILE')
            
            if config.has_option('LOGGING', 'ROTATE'):
                rotate = config.getboolean('LOGGING', 'ROTATE')
            
            if config.has_option('LOGGING', 'MAX_SIZE'):
                max_size = config.getint('LOGGING', 'MAX_SIZE')
            
            if config.has_option('LOGGING', 'BACKUP_COUNT'):
                backup_count = config.getint('LOGGING', 'BACKUP_COUNT')
    
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure logging
    if rotate:
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            log_file, maxBytes=max_size, backupCount=backup_count
        )
    else:
        file_handler = logging.FileHandler(log_file)
    
    console_handler = logging.StreamHandler()
    
    # Format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    logger.info("Logging setup complete")

def load_settings(settings_path=None):
    """
    Load settings from the settings file.
    
    Args:
        settings_path (str, optional): Path to the settings file. Defaults to None.
        
    Returns:
        dict: Dictionary with settings
    """
    if settings_path is None:
        settings_path = os.path.join('config', 'screener_settings.json')
    
    try:
        with open(settings_path, 'r') as file:
            settings = json.load(file)
        logger.info(f"Loaded settings from {settings_path}")
        return settings
    except Exception as e:
        logger.error(f"Failed to load settings from {settings_path}: {e}")
        return {}

def winsorize(series, limits=(0.05, 0.05)):
    """
    Winsorize a series to handle outliers.
    
    Args:
        series (pd.Series): Series to winsorize
        limits (tuple, optional): Limits for winsorization, (lower, upper).
                                  Defaults to (0.05, 0.05).
        
    Returns:
        pd.Series: Winsorized series
    """
    try:
        # Drop NaN values before winsorizing
        clean_series = series.dropna()
        
        if len(clean_series) <= 2:
            logger.warning("Too few values to winsorize")
            return series
        
        # Calculate thresholds
        lower_limit = limits[0]
        upper_limit = limits[1]
        
        lower_threshold = clean_series.quantile(lower_limit)
        upper_threshold = clean_series.quantile(1 - upper_limit)
        
        # Winsorize
        winsorized = series.copy()
        winsorized = winsorized.clip(lower=lower_threshold, upper=upper_threshold)
        
        return winsorized
    except Exception as e:
        logger.error(f"Failed to winsorize series: {e}")
        return series

def normalize_score(series, min_val=None, max_val=None):
    """
    Normalize a series to a 0-100 scale.
    
    Args:
        series (pd.Series): Series to normalize
        min_val (float, optional): Minimum value for normalization. Defaults to None.
        max_val (float, optional): Maximum value for normalization. Defaults to None.
        
    Returns:
        pd.Series: Normalized series
    """
    try:
        # Drop NaN values before normalizing
        clean_series = series.dropna()
        
        if len(clean_series) <= 1:
            logger.warning("Too few values to normalize")
            return pd.Series(index=series.index, data=50)  # Return neutral score
        
        # Get min and max values if not provided
        if min_val is None:
            min_val = clean_series.min()
        
        if max_val is None:
            max_val = clean_series.max()
        
        # Check if all values are the same
        if min_val == max_val:
            logger.warning("All values are the same, returning 50")
            return pd.Series(index=series.index, data=50)  # Return neutral score
        
        # Normalize to 0-100 scale
        normalized = (series - min_val) / (max_val - min_val) * 100
        
        # Clip values to 0-100 range
        normalized = normalized.clip(0, 100)
        
        return normalized
    except Exception as e:
        logger.error(f"Failed to normalize series: {e}")
        return pd.Series(index=series.index, data=50)  # Return neutral score

def calculate_z_score(series):
    """
    Calculate z-scores for a series.
    
    Args:
        series (pd.Series): Series to calculate z-scores for
        
    Returns:
        pd.Series: Series with z-scores
    """
    try:
        # Drop NaN values before calculating z-scores
        clean_series = series.dropna()
        
        if len(clean_series) <= 1:
            logger.warning("Too few values to calculate z-scores")
            return pd.Series(index=series.index, data=0)  # Return neutral score
        
        # Calculate z-scores
        z_scores = pd.Series(index=series.index)
        mean = clean_series.mean()
        std = clean_series.std()
        
        if std == 0:
            logger.warning("Standard deviation is zero, returning zeros")
            return pd.Series(index=series.index, data=0)  # Return neutral score
        
        z_scores = (series - mean) / std
        
        return z_scores
    except Exception as e:
        logger.error(f"Failed to calculate z-scores: {e}")
        return pd.Series(index=series.index, data=0)  # Return neutral score

def remove_outliers(series, z_threshold=3.0):
    """
    Remove outliers from a series based on z-scores.
    
    Args:
        series (pd.Series): Series to remove outliers from
        z_threshold (float, optional): Z-score threshold for outliers. Defaults to 3.0.
        
    Returns:
        pd.Series: Series with outliers removed
    """
    try:
        # Calculate z-scores
        z_scores = calculate_z_score(series)
        
        # Remove outliers
        clean_series = series[abs(z_scores) <= z_threshold]
        
        return clean_series
    except Exception as e:
        logger.error(f"Failed to remove outliers: {e}")
        return series

def calculate_percentile_rank(series):
    """
    Calculate percentile ranks for a series.
    
    Args:
        series (pd.Series): Series to calculate percentile ranks for
        
    Returns:
        pd.Series: Series with percentile ranks (0-100)
    """
    try:
        # Drop NaN values before calculating percentile ranks
        clean_series = series.dropna()
        
        if len(clean_series) <= 1:
            logger.warning("Too few values to calculate percentile ranks")
            return pd.Series(index=series.index, data=50)  # Return neutral score
        
        # Calculate percentile ranks
        ranks = clean_series.rank(pct=True) * 100
        
        # Fill NaN values with median
        if len(ranks) < len(series):
            median_rank = ranks.median()
            ranks = pd.Series(index=series.index)
            ranks[clean_series.index] = clean_series.rank(pct=True) * 100
            ranks.fillna(median_rank, inplace=True)
        
        return ranks
    except Exception as e:
        logger.error(f"Failed to calculate percentile ranks: {e}")
        return pd.Series(index=series.index, data=50)  # Return neutral score

def calculate_rolling_stats(series, window=30):
    """
    Calculate rolling statistics for a series.
    
    Args:
        series (pd.Series): Series to calculate rolling statistics for
        window (int, optional): Rolling window size. Defaults to 30.
        
    Returns:
        dict: Dictionary with rolling statistics
    """
    try:
        # Calculate rolling statistics
        rolling_mean = series.rolling(window=window).mean()
        rolling_std = series.rolling(window=window).std()
        rolling_min = series.rolling(window=window).min()
        rolling_max = series.rolling(window=window).max()
        
        # Create dictionary with results
        stats = {
            'rolling_mean': rolling_mean,
            'rolling_std': rolling_std,
            'rolling_min': rolling_min,
            'rolling_max': rolling_max
        }
        
        return stats
    except Exception as e:
        logger.error(f"Failed to calculate rolling statistics: {e}")
        return {}

def calculate_returns(prices, periods=[1, 5, 10, 21, 63, 126, 252]):
    """
    Calculate returns for different periods.
    
    Args:
        prices (pd.Series): Series with prices
        periods (list, optional): List of periods to calculate returns for.
                                  Defaults to [1, 5, 10, 21, 63, 126, 252].
        
    Returns:
        pd.DataFrame: DataFrame with returns for different periods
    """
    try:
        returns = pd.DataFrame(index=prices.index)
        
        for period in periods:
            returns[f'return_{period}d'] = prices.pct_change(period)
        
        return returns
    except Exception as e:
        logger.error(f"Failed to calculate returns: {e}")
        return pd.DataFrame()

def calculate_volatility(returns, window=30):
    """
    Calculate volatility for a returns series.
    
    Args:
        returns (pd.Series): Series with returns
        window (int, optional): Window size for volatility calculation. Defaults to 30.
        
    Returns:
        pd.Series: Series with volatility
    """
    try:
        # Calculate volatility (standard deviation of returns)
        volatility = returns.rolling(window=window).std() * np.sqrt(252)  # Annualize
        
        return volatility
    except Exception as e:
        logger.error(f"Failed to calculate volatility: {e}")
        return pd.Series()

def calculate_sharpe_ratio(returns, risk_free_rate=0):
    """
    Calculate Sharpe ratio for a returns series.
    
    Args:
        returns (pd.Series): Series with returns
        risk_free_rate (float, optional): Risk-free rate. Defaults to 0.
        
    Returns:
        float: Sharpe ratio
    """
    try:
        # Calculate Sharpe ratio
        excess_returns = returns - risk_free_rate
        sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(252)  # Annualize
        
        return sharpe_ratio
    except Exception as e:
        logger.error(f"Failed to calculate Sharpe ratio: {e}")
        return np.nan

def calculate_drawdowns(prices):
    """
    Calculate drawdowns for a price series.
    
    Args:
        prices (pd.Series): Series with prices
        
    Returns:
        pd.DataFrame: DataFrame with drawdowns
    """
    try:
        # Calculate running maximum
        running_max = prices.cummax()
        
        # Calculate drawdowns
        drawdowns = (prices / running_max - 1) * 100  # Convert to percentage
        
        # Calculate maximum drawdown
        max_drawdown = drawdowns.min()
        
        return {
            'drawdowns': drawdowns,
            'max_drawdown': max_drawdown
        }
    except Exception as e:
        logger.error(f"Failed to calculate drawdowns: {e}")
        return {
            'drawdowns': pd.Series(),
            'max_drawdown': np.nan
        }

def convert_to_dataframe(data, ticker_col='ticker'):
    """
    Convert various data formats to a DataFrame.
    
    Args:
        data: Data to convert (list, dict, Series, or DataFrame)
        ticker_col (str, optional): Column name for tickers if a dict is provided.
                                    Defaults to 'ticker'.
        
    Returns:
        pd.DataFrame: Converted DataFrame
    """
    try:
        if isinstance(data, pd.DataFrame):
            return data
        elif isinstance(data, pd.Series):
            return pd.DataFrame(data)
        elif isinstance(data, dict):
            if all(isinstance(v, (int, float, str, bool)) for v in data.values()):
                # Dict of values
                return pd.DataFrame([data])
            else:
                # Dict of Series or other objects
                df = pd.DataFrame(data)
                if ticker_col not in df.columns and df.index.name != ticker_col:
                    df.index.name = ticker_col
                return df
        elif isinstance(data, list):
            if all(isinstance(item, (int, float, str, bool)) for item in data):
                # List of values
                return pd.DataFrame({0: data})
            else:
                # List of dicts or other objects
                return pd.DataFrame(data)
        else:
            logger.error(f"Unsupported data type: {type(data)}")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Failed to convert to DataFrame: {e}")
        return pd.DataFrame()

def export_to_excel(data, file_path, sheet_name='Sheet1'):
    """
    Export data to an Excel file.
    
    Args:
        data (pd.DataFrame): Data to export
        file_path (str): Path to the Excel file
        sheet_name (str, optional): Sheet name. Defaults to 'Sheet1'.
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Convert to DataFrame if necessary
        df = convert_to_dataframe(data)
        
        # Export to Excel
        df.to_excel(file_path, sheet_name=sheet_name)
        
        logger.info(f"Exported data to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to export to Excel: {e}")
        return False

def export_to_csv(data, file_path):
    """
    Export data to a CSV file.
    
    Args:
        data (pd.DataFrame): Data to export
        file_path (str): Path to the CSV file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Convert to DataFrame if necessary
        df = convert_to_dataframe(data)
        
        # Export to CSV
        df.to_csv(file_path)
        
        logger.info(f"Exported data to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to export to CSV: {e}")
        return False

def filter_by_market_cap(data, min_market_cap=None, max_market_cap=None, market_cap_col='MarketCapitalization'):
    """
    Filter data by market capitalization.
    
    Args:
        data (pd.DataFrame): Data to filter
        min_market_cap (float, optional): Minimum market cap. Defaults to None.
        max_market_cap (float, optional): Maximum market cap. Defaults to None.
        market_cap_col (str, optional): Column with market cap data. Defaults to 'MarketCapitalization'.
        
    Returns:
        pd.DataFrame: Filtered data
    """
    try:
        # Convert to DataFrame if necessary
        df = convert_to_dataframe(data)
        
        if market_cap_col not in df.columns:
            logger.warning(f"Market cap column '{market_cap_col}' not found in data")
            return df
        
        # Convert market cap to numeric
        df[market_cap_col] = pd.to_numeric(df[market_cap_col], errors='coerce')
        
        # Apply filters
        if min_market_cap is not None:
            df = df[df[market_cap_col] >= min_market_cap]
        
        if max_market_cap is not None:
            df = df[df[market_cap_col] <= max_market_cap]
        
        return df
    except Exception as e:
        logger.error(f"Failed to filter by market cap: {e}")
        return data

def filter_by_sector(data, include_sectors=None, exclude_sectors=None, sector_col='Sector'):
    """
    Filter data by sector.
    
    Args:
        data (pd.DataFrame): Data to filter
        include_sectors (list, optional): Sectors to include. Defaults to None.
        exclude_sectors (list, optional): Sectors to exclude. Defaults to None.
        sector_col (str, optional): Column with sector data. Defaults to 'Sector'.
        
    Returns:
        pd.DataFrame: Filtered data
    """
    try:
        # Convert to DataFrame if necessary
        df = convert_to_dataframe(data)
        
        if sector_col not in df.columns:
            logger.warning(f"Sector column '{sector_col}' not found in data")
            return df
        
        # Apply filters
        if include_sectors is not None:
            df = df[df[sector_col].isin(include_sectors)]
        
        if exclude_sectors is not None:
            df = df[~df[sector_col].isin(exclude_sectors)]
        
        return df
    except Exception as e:
        logger.error(f"Failed to filter by sector: {e}")
        return data

def filter_by_value(data, column, min_value=None, max_value=None):
    """
    Filter data by value in a column.
    
    Args:
        data (pd.DataFrame): Data to filter
        column (str): Column to filter on
        min_value (float, optional): Minimum value. Defaults to None.
        max_value (float, optional): Maximum value. Defaults to None.
        
    Returns:
        pd.DataFrame: Filtered data
    """
    try:
        # Convert to DataFrame if necessary
        df = convert_to_dataframe(data)
        
        if column not in df.columns:
            logger.warning(f"Column '{column}' not found in data")
            return df
        
        # Convert to numeric
        df[column] = pd.to_numeric(df[column], errors='coerce')
        
        # Apply filters
        if min_value is not None:
            df = df[df[column] >= min_value]
        
        if max_value is not None:
            df = df[df[column] <= max_value]
        
        return df
    except Exception as e:
        logger.error(f"Failed to filter by value: {e}")
        return data
