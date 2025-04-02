"""
Module for collecting fundamental financial data from various sources.
"""

import os
import time
import logging
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from src.utils import ensure_dir_exists

# Setup logging
logger = logging.getLogger('stock_screener.data.fundamental')

class FundamentalDataCollector:
    """
    Class for collecting and managing fundamental financial data.
    """
    
    def __init__(self, data_dir='data'):
        """
        Initialize the fundamental data collector.
        
        Args:
            data_dir (str, optional): Base directory for data storage. Defaults to 'data'.
        """
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, 'raw', 'fundamentals')
        self.processed_dir = os.path.join(data_dir, 'processed', 'fundamentals')
        
        # Ensure directories exist
        ensure_dir_exists(self.raw_dir)
        ensure_dir_exists(self.processed_dir)
    
    def get_financial_ratios(self, tickers):
        """
        Get key financial ratios for a list of tickers.
        
        Args:
            tickers (list): List of ticker symbols
            
        Returns:
            pd.DataFrame: DataFrame with financial ratios for each ticker
        """
        logger.info(f"Getting financial ratios for {len(tickers)} tickers")
        
        # Initialize an empty DataFrame to store results
        all_ratios = []
        
        for i, ticker in enumerate(tickers):
            try:
                # Log progress
                if i % 10 == 0:
                    logger.info(f"Getting ratios for ticker {i+1}/{len(tickers)}")
                
                # Get ticker info using yfinance
                ticker_obj = yf.Ticker(ticker)
                
                # Get key stats from info
                info = ticker_obj.info
                
                # Get financial ratios from info dictionary
                ratios = {
                    'ticker': ticker,
                    'sector': info.get('sector', 'Unknown'),
                    'industry': info.get('industry', 'Unknown'),
                    'market_cap': info.get('marketCap', None),
                    'pe_ratio': info.get('trailingPE', None),
                    'forward_pe': info.get('forwardPE', None),
                    'peg_ratio': info.get('pegRatio', None),
                    'price_to_book': info.get('priceToBook', None),
                    'price_to_sales': info.get('priceToSalesTrailing12Months', None),
                    'dividend_yield': info.get('dividendYield', None) * 100 if info.get('dividendYield') is not None else None,
                    'roe': info.get('returnOnEquity', None) * 100 if info.get('returnOnEquity') is not None else None,
                    'roa': info.get('returnOnAssets', None) * 100 if info.get('returnOnAssets') is not None else None,
                    'profit_margin': info.get('profitMargins', None) * 100 if info.get('profitMargins') is not None else None,
                    'operating_margin': info.get('operatingMargins', None) * 100 if info.get('operatingMargins') is not None else None,
                    'current_ratio': info.get('currentRatio', None),
                    'quick_ratio': info.get('quickRatio', None),
                    'debt_to_equity': info.get('debtToEquity', None),
                    'beta': info.get('beta', None),
                    'timestamp': datetime.now()
                }
                
                # Add to the results
                all_ratios.append(ratios)
                
                # Add a small delay to avoid hitting API rate limits
                time.sleep(0.2)
                
            except Exception as e:
                logger.error(f"Failed to get ratios for {ticker}: {e}")
                # Add ticker with NaN values to maintain the complete list
                all_ratios.append({'ticker': ticker, 'timestamp': datetime.now()})
        
        # Convert to DataFrame
        ratios_df = pd.DataFrame(all_ratios)
        
        logger.info(f"Successfully got ratios for {len(ratios_df)} tickers")
        return ratios_df
    
    def get_analyst_estimates(self, tickers):
        """
        Get analyst estimates and revisions for a list of tickers.
        
        Args:
            tickers (list): List of ticker symbols
            
        Returns:
            pd.DataFrame: DataFrame with analyst estimates and revisions
        """
        logger.info(f"Getting analyst estimates for {len(tickers)} tickers")
        
        # Initialize an empty list to store results
        all_estimates = []
        
        for i, ticker in enumerate(tickers):
            try:
                # Log progress
                if i % 10 == 0:
                    logger.info(f"Getting estimates for ticker {i+1}/{len(tickers)}")
                
                # Get ticker object from yfinance
                ticker_obj = yf.Ticker(ticker)
                
                # Get analyst recommendations
                recommendations = ticker_obj.recommendations
                
                # Get earnings and revenue estimates
                earnings_trend = None
                revenue_trend = None
                
                try:
                    # Get earnings estimates from Yahoo Finance (if available)
                    earnings = ticker_obj.earnings_trend
                    
                    if earnings is not None and not earnings.empty:
                        earnings_trend = {
                            'current_quarter_estimate': earnings.iloc[0, 0] if 0 in earnings.columns else None,
                            'next_quarter_estimate': earnings.iloc[1, 0] if 0 in earnings.columns else None,
                            'current_year_estimate': earnings.iloc[2, 0] if 0 in earnings.columns else None,
                            'next_year_estimate': earnings.iloc[3, 0] if 0 in earnings.columns else None,
                            'current_quarter_growth': earnings.iloc[0, 5] if 5 in earnings.columns else None,
                            'next_quarter_growth': earnings.iloc[1, 5] if 5 in earnings.columns else None,
                            'current_year_growth': earnings.iloc[2, 5] if 5 in earnings.columns else None,
                            'next_year_growth': earnings.iloc[3, 5] if 5 in earnings.columns else None
                        }
                        
                    # Get revenue estimates (not directly available in yfinance)
                    # We would need to implement additional scraping for this
                except Exception as e:
                    logger.warning(f"Failed to get earnings trend for {ticker}: {e}")
                
                # Compute recommendation stats
                if recommendations is not None and not recommendations.empty:
                    latest_date = recommendations.index.max()
                    last_month_date = latest_date - timedelta(days=30)
                    
                    # Get latest recommendations
                    latest_recommendations = recommendations.loc[recommendations.index >= last_month_date]
                    
                    # Calculate average recommendation (1=Strong Buy, 5=Strong Sell)
                    recommendation_mapping = {
                        'strong buy': 1,
                        'buy': 2,
                        'hold': 3,
                        'underperform': 4,
                        'sell': 5
                    }
                    
                    if 'To Grade' in latest_recommendations.columns:
                        rec_values = latest_recommendations['To Grade'].str.lower().map(recommendation_mapping)
                        avg_recommendation = rec_values.mean() if not rec_values.empty else None
                    else:
                        avg_recommendation = None
                        
                    # Count recommendations by type
                    if 'To Grade' in latest_recommendations.columns:
                        rec_counts = latest_recommendations['To Grade'].str.lower().value_counts()
                        strong_buy_count = rec_counts.get('strong buy', 0)
                        buy_count = rec_counts.get('buy', 0)
                        hold_count = rec_counts.get('hold', 0)
                        sell_count = rec_counts.get('sell', 0) + rec_counts.get('underperform', 0)
                    else:
                        strong_buy_count = buy_count = hold_count = sell_count = 0
                else:
                    avg_recommendation = None
                    strong_buy_count = buy_count = hold_count = sell_count = 0
                
                # Create a dictionary with the estimates
                estimates = {
                    'ticker': ticker,
                    'avg_recommendation': avg_recommendation,
                    'strong_buy_count': strong_buy_count,
                    'buy_count': buy_count,
                    'hold_count': hold_count,
                    'sell_count': sell_count,
                    'timestamp': datetime.now()
                }
                
                # Add earnings trend if available
                if earnings_trend is not None:
                    estimates.update(earnings_trend)
                
                # Add to the results
                all_estimates.append(estimates)
                
                # Add a small delay to avoid hitting API rate limits
                time.sleep(0.2)
                
            except Exception as e:
                logger.error(f"Failed to get estimates for {ticker}: {e}")
                # Add ticker with NaN values to maintain the complete list
                all_estimates.append({'ticker': ticker, 'timestamp': datetime.now()})
        
        # Convert to DataFrame
        estimates_df = pd.DataFrame(all_estimates)
        
        logger.info(f"Successfully got estimates for {len(estimates_df)} tickers")
        return estimates_df
    
    def get_historical_financials(self, tickers):
        """
        Get historical financial statements for a list of tickers.
        
        Args:
            tickers (list): List of ticker symbols
            
        Returns:
            dict: Dictionary with income statements, balance sheets, and cash flows
        """
        logger.info(f"Getting historical financials for {len(tickers)} tickers")
        
        # Initialize dictionaries to store results
        income_statements = {}
        balance_sheets = {}
        cash_flows = {}
        
        for i, ticker in enumerate(tickers):
            try:
                # Log progress
                if i % 10 == 0:
                    logger.info(f"Getting financials for ticker {i+1}/{len(tickers)}")
                
                # Get ticker object from yfinance
                ticker_obj = yf.Ticker(ticker)
                
                # Get financial statements
                income_statements[ticker] = ticker_obj.income_stmt
                balance_sheets[ticker] = ticker_obj.balance_sheet
                cash_flows[ticker] = ticker_obj.cashflow
                
                # Add a small delay to avoid hitting API rate limits
                time.sleep(0.2)
                
            except Exception as e:
                logger.error(f"Failed to get financials for {ticker}: {e}")
        
        # Combine into a dictionary
        financials = {
            'income_statements': income_statements,
            'balance_sheets': balance_sheets,
            'cash_flows': cash_flows
        }
        
        logger.info(f"Successfully got financials for {len(income_statements)} tickers")
        return financials
    
    def save_fundamental_data(self, data, filename, data_type='ratios'):
        """
        Save fundamental data to disk.
        
        Args:
            data: Data to save (DataFrame or dictionary)
            filename (str): Filename to save data
            data_type (str, optional): Type of data ('ratios', 'estimates', 'financials'). Defaults to 'ratios'.
        """
        filepath = os.path.join(self.raw_dir, filename)
        try:
            if data_type == 'financials':
                # For financials, save the dictionary with pickle
                pd.to_pickle(data, filepath)
            else:
                # For DataFrames (ratios, estimates), save as parquet
                data.to_parquet(filepath)
            
            logger.info(f"{data_type.capitalize()} data saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save {data_type} data: {e}")
    
    def load_fundamental_data(self, filename, data_type='ratios'):
        """
        Load fundamental data from disk.
        
        Args:
            filename (str): Filename to load data from
            data_type (str, optional): Type of data ('ratios', 'estimates', 'financials'). Defaults to 'ratios'.
            
        Returns:
            DataFrame or dict: Loaded data
        """
        filepath = os.path.join(self.raw_dir, filename)
        try:
            if data_type == 'financials':
                # For financials, load the dictionary with pickle
                data = pd.read_pickle(filepath)
            else:
                # For DataFrames (ratios, estimates), load from parquet
                data = pd.read_parquet(filepath)
            
            logger.info(f"{data_type.capitalize()} data loaded from {filepath}")
            return data
        except Exception as e:
            logger.error(f"Failed to load {data_type} data: {e}")
            return None
    
    def process_financial_ratios(self, ratios_df):
        """
        Process raw financial ratios data to calculate derived metrics.
        
        Args:
            ratios_df (pd.DataFrame): DataFrame with financial ratios
            
        Returns:
            pd.DataFrame: DataFrame with processed ratios
        """
        # Make a copy to avoid modifying the original
        processed_ratios = ratios_df.copy()
        
        # Calculate additional metrics
        
        # Earnings Yield (inverse of P/E)
        processed_ratios['earnings_yield'] = (1 / processed_ratios['pe_ratio']) * 100 if 'pe_ratio' in processed_ratios.columns else None
        
        # Free Cash Flow Yield (would require additional data)
        
        # Replace infinite values with NaN
        processed_ratios = processed_ratios.replace([np.inf, -np.inf], np.nan)
        
        # Group by sector to calculate relative metrics
        if 'sector' in processed_ratios.columns:
            sector_groups = processed_ratios.groupby('sector')
            
            # For each numerical column, calculate the percentile within the sector
            numerical_cols = processed_ratios.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                # Skip non-ratio columns
                if col in ['market_cap', 'timestamp']:
                    continue
                
                # Calculate sector-relative percentile rank
                relative_col_name = f'relative_{col}'
                processed_ratios[relative_col_name] = processed_ratios.groupby('sector')[col].transform(
                    lambda x: x.rank(pct=True) * 100 if len(x.dropna()) > 1 else 50
                )
        
        logger.info("Financial ratios processed successfully")
        return processed_ratios
    
    def process_analyst_estimates(self, estimates_df):
        """
        Process raw analyst estimates data to calculate derived metrics.
        
        Args:
            estimates_df (pd.DataFrame): DataFrame with analyst estimates
            
        Returns:
            pd.DataFrame: DataFrame with processed estimates
        """
        # Make a copy to avoid modifying the original
        processed_estimates = estimates_df.copy()
        
        # Calculate recommendation score (1-5 scale, 1 being strong buy)
        if 'avg_recommendation' in processed_estimates.columns:
            # Invert the scale to make higher values better (for consistency with other metrics)
            processed_estimates['recommendation_score'] = 6 - processed_estimates['avg_recommendation']
            
            # Normalize to 0-100 scale
            processed_estimates['recommendation_score'] = (processed_estimates['recommendation_score'] - 1) * 25
        
        # Calculate percentage of buys among all recommendations
        buy_cols = ['strong_buy_count', 'buy_count', 'hold_count', 'sell_count']
        if all(col in processed_estimates.columns for col in buy_cols):
            processed_estimates['total_ratings'] = (
                processed_estimates['strong_buy_count'] + 
                processed_estimates['buy_count'] + 
                processed_estimates['hold_count'] + 
                processed_estimates['sell_count']
            )
            
            # Calculate percentage of buys
            processed_estimates['buy_percentage'] = 100 * (
                processed_estimates['strong_buy_count'] + processed_estimates['buy_count']
            ) / processed_estimates['total_ratings']
            
            # Replace NaN with 0
            processed_estimates['buy_percentage'] = processed_estimates['buy_percentage'].fillna(0)
        
        # Growth revisions (if data available)
        growth_cols = [
            'current_quarter_growth', 'next_quarter_growth', 
            'current_year_growth', 'next_year_growth'
        ]
        
        if all(col in processed_estimates.columns for col in growth_cols):
            # Calculate average growth
            processed_estimates['avg_growth'] = processed_estimates[growth_cols].mean(axis=1)
        
        logger.info("Analyst estimates processed successfully")
        return processed_estimates
    
    def process_financials(self, financials):
        """
        Process raw financial statements to calculate derived metrics.
        
        Args:
            financials (dict): Dictionary with income statements, balance sheets, and cash flows
            
        Returns:
            dict: Dictionary with processed financial metrics
        """
        # Initialize dictionary for results
        processed_metrics = {}
        
        for ticker, income_stmt in financials['income_statements'].items():
            try:
                balance_sheet = financials['balance_sheets'].get(ticker)
                cash_flow = financials['cash_flows'].get(ticker)
                
                if income_stmt is None or income_stmt.empty:
                    logger.warning(f"No income statement data for {ticker}")
                    continue
                
                if balance_sheet is None or balance_sheet.empty:
                    logger.warning(f"No balance sheet data for {ticker}")
                    continue
                
                if cash_flow is None or cash_flow.empty:
                    logger.warning(f"No cash flow data for {ticker}")
                    continue
                
                # Get the last few years of data
                last_few_years = min(len(income_stmt.columns), 5)
                
                # Calculate key financial metrics
                metrics = pd.DataFrame(index=income_stmt.columns[:last_few_years])
                
                # Revenue and Net Income
                if 'Total Revenue' in income_stmt.index:
                    metrics['Revenue'] = income_stmt.loc['Total Revenue']
                
                if 'Net Income' in income_stmt.index:
                    metrics['NetIncome'] = income_stmt.loc['Net Income']
                
                # Profit Margin
                if all(x in metrics.columns for x in ['Revenue', 'NetIncome']):
                    metrics['ProfitMargin'] = metrics['NetIncome'] / metrics['Revenue'] * 100
                
                # ROE
                if 'NetIncome' in metrics.columns and 'Total Stockholder Equity' in balance_sheet.index:
                    metrics['ROE'] = metrics['NetIncome'] / balance_sheet.loc['Total Stockholder Equity'] * 100
                
                # ROA
                if 'NetIncome' in metrics.columns and 'Total Assets' in balance_sheet.index:
                    metrics['ROA'] = metrics['NetIncome'] / balance_sheet.loc['Total Assets'] * 100
                
                # Debt to Equity
                if all(x in balance_sheet.index for x in ['Total Debt', 'Total Stockholder Equity']):
                    metrics['DebtToEquity'] = balance_sheet.loc['Total Debt'] / balance_sheet.loc['Total Stockholder Equity']
                
                # Free Cash Flow
                if all(x in cash_flow.index for x in ['Free Cash Flow']):
                    metrics['FreeCashFlow'] = cash_flow.loc['Free Cash Flow']
                
                # Store metrics for this ticker
                processed_metrics[ticker] = metrics
                
            except Exception as e:
                logger.error(f"Failed to process financials for {ticker}: {e}")
        
        logger.info(f"Processed financials for {len(processed_metrics)} tickers")
        return processed_metrics
    
    def save_processed_fundamental_data(self, data, filename, data_type='ratios'):
        """
        Save processed fundamental data to disk.
        
        Args:
            data: Processed data to save
            filename (str): Filename to save data
            data_type (str, optional): Type of data ('ratios', 'estimates', 'metrics'). Defaults to 'ratios'.
        """
        filepath = os.path.join(self.processed_dir, filename)
        try:
            if data_type == 'metrics':
                # For financial metrics, save the dictionary with pickle
                pd.to_pickle(data, filepath)
            else:
                # For DataFrames (ratios, estimates), save as parquet
                data.to_parquet(filepath)
            
            logger.info(f"Processed {data_type} data saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save processed {data_type} data: {e}")
    
    def load_processed_fundamental_data(self, filename, data_type='ratios'):
        """
        Load processed fundamental data from disk.
        
        Args:
            filename (str): Filename to load data from
            data_type (str, optional): Type of data ('ratios', 'estimates', 'metrics'). Defaults to 'ratios'.
            
        Returns:
            DataFrame or dict: Loaded processed data
        """
        filepath = os.path.join(self.processed_dir, filename)
        try:
            if data_type == 'metrics':
                # For financial metrics, load the dictionary with pickle
                data = pd.read_pickle(filepath)
            else:
                # For DataFrames (ratios, estimates), load from parquet
                data = pd.read_parquet(filepath)
            
            logger.info(f"Processed {data_type} data loaded from {filepath}")
            return data
        except Exception as e:
            logger.error(f"Failed to load processed {data_type} data: {e}")
            return None
