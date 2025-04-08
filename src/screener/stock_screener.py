"""
Main stock screener module.
Combines data collection and factor calculation to screen for investment opportunities.
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# Import internal modules
from src.data.collectors.price_data_collector import PriceDataCollector
from src.data.collectors.fundamental_data_collector import FundamentalDataCollector
from src.factors.momentum import MomentumFactor
from src.factors.quality import QualityFactor
from src.utils import (
    setup_logging, load_settings, normalize_score, winsorize, 
    filter_by_market_cap, filter_by_sector, filter_by_value,
    export_to_excel, export_to_csv
)

# Setup logging
logger = logging.getLogger('stock_screener')

class StockScreener:
    """
    Main class for the stock screener.
    """
    
    def __init__(self, config_path=None, settings_path=None):
        """
        Initialize the stock screener.
        
        Args:
            config_path (str, optional): Path to the config file. Defaults to None.
            settings_path (str, optional): Path to the settings file. Defaults to None.
        """
        # Setup logging
        setup_logging()
        
        # Load settings
        self.settings = load_settings(settings_path)
        
        # Initialize data collectors
        self.price_collector = PriceDataCollector(config_path)
        self.fundamental_collector = FundamentalDataCollector(config_path)
        
        # Initialize factor calculators
        self.momentum_factor = MomentumFactor()
        self.quality_factor = QualityFactor()
        
        # Initialize data containers
        self.index_components = {}
        self.price_data = {}
        self.fundamental_data = {}
        self.momentum_scores = None
        self.quality_scores = None
        self.combined_scores = None
        
        logger.info("Stock screener initialized")
    
    def load_index_data(self, index_name):
        """
        Load data for a specific index.
        
        Args:
            index_name (str): Name of the index, e.g., 'SP500', 'NASDAQ'
            
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info(f"Loading data for index: {index_name}")
        
        try:
            # Check if the index is defined in settings
            if 'indices' not in self.settings or index_name not in self.settings['indices']:
                logger.error(f"Index {index_name} not found in settings")
                return False
            
            index_settings = self.settings['indices'][index_name]
            
            # Get index components
            components_source = index_settings.get('components_source', 'wikipedia')
            
            logger.info(f"Getting components for {index_name} from {components_source}")
            components = self.price_collector.get_index_components(index_name, source=components_source)
            
            if not components:
                logger.error(f"Failed to get components for index {index_name}")
                return False
            
            self.index_components[index_name] = components
            
            # Get price data
            logger.info(f"Getting price data for {len(components)} stocks")
            
            # Determine date range
            end_date = datetime.now().strftime('%Y-%m-%d')
            days = self.settings.get('data', {}).get('price_history_days', 365)
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            # Get price data
            price_data = self.price_collector.collect_stock_data(
                components, 
                start_date=start_date,
                end_date=end_date,
                source=index_settings.get('source', 'yahoo')
            )
            
            if not price_data:
                logger.error(f"Failed to get price data for index {index_name}")
                return False
            
            self.price_data = price_data
            
            # Get fundamental data (for a subset to avoid API limits)
            max_stocks = 50  # Limit to avoid API rate limits
            subset_tickers = list(price_data.keys())[:max_stocks]
            
            logger.info(f"Getting fundamental data for {len(subset_tickers)} stocks")
            fundamental_data = self.fundamental_collector.collect_all_fundamentals(subset_tickers)
            
            if not fundamental_data:
                logger.warning(f"Failed to get fundamental data for index {index_name}")
                # Continue without fundamental data
            else:
                self.fundamental_data = fundamental_data
            
            logger.info(f"Successfully loaded data for index {index_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load index data: {e}")
            return False
    
    def calculate_scores(self):
        """
        Calculate quality and momentum scores for the loaded data.
        
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info("Calculating factor scores")
        
        try:
            # Check if data is loaded
            if not self.price_data:
                logger.error("No price data loaded")
                return False
            
            # Calculate momentum scores
            logger.info("Calculating momentum scores")
            self.momentum_factor.set_data(self.price_data, self.fundamental_data)
            
            # Get momentum weights from settings
            momentum_settings = self.settings.get('factors', {}).get('momentum', {})
            technical_weight = momentum_settings.get('price_momentum_weight', 0.7)
            fundamental_weight = momentum_settings.get('earnings_revision_weight', 0.3)
            
            # Calculate momentum scores
            self.momentum_scores = self.momentum_factor.calculate_combined_momentum(
                technical_weight=technical_weight,
                fundamental_weight=fundamental_weight
            )
            
            if self.momentum_scores is None or self.momentum_scores.empty:
                logger.error("Failed to calculate momentum scores")
                return False
            
            # Calculate quality scores
            logger.info("Calculating quality scores")
            self.quality_factor.set_data(self.fundamental_data)
            self.quality_scores = self.quality_factor.calculate_combined_quality()
            
            if self.quality_scores is None or self.quality_scores.empty:
                logger.warning("Failed to calculate quality scores, proceeding with momentum only")
                # Continue without quality scores
            
            # Combine scores
            logger.info("Combining scores")
            self._combine_scores()
            
            logger.info("Successfully calculated factor scores")
            return True
            
        except Exception as e:
            logger.error(f"Failed to calculate scores: {e}")
            return False
    
    def _combine_scores(self):
        """
        Combine momentum and quality scores into a final score.
        """
        try:
            # Get weights from settings
            scoring_settings = self.settings.get('scoring', {})
            momentum_weight = scoring_settings.get('momentum_weight', 0.5)
            quality_weight = scoring_settings.get('quality_weight', 0.5)
            
            # Normalize weights
            total_weight = momentum_weight + quality_weight
            momentum_weight = momentum_weight / total_weight
            quality_weight = quality_weight / total_weight
            
            # Start with momentum scores
            combined = self.momentum_scores.copy()
            
            # If quality scores are available, combine them
            if self.quality_scores is not None and not self.quality_scores.empty:
                # Merge with quality scores
                combined = combined.join(self.quality_scores, how='left')
                
                # Calculate combined score
                combined['combined_score'] = (
                    combined['momentum_score'] * momentum_weight +
                    combined['quality_score'].fillna(50) * quality_weight  # Use neutral score for missing values
                )
            else:
                # If no quality scores, use only momentum
                combined['combined_score'] = combined['momentum_score']
            
            # Store combined scores
            self.combined_scores = combined
            
            logger.info(f"Combined scores calculated for {len(combined)} stocks")
            return True
            
        except Exception as e:
            logger.error(f"Failed to combine scores: {e}")
            return False
    
    def get_top_stocks(self, n=20, sort_by='combined_score'):
        """
        Get the top N stocks based on the specified score.
        
        Args:
            n (int, optional): Number of stocks to return. Defaults to 20.
            sort_by (str, optional): Score to sort by. Can be 'combined_score', 
                                    'momentum_score', or 'quality_score'.
                                    Defaults to 'combined_score'.
            
        Returns:
            pd.DataFrame: DataFrame with the top N stocks
        """
        logger.info(f"Getting top {n} stocks sorted by {sort_by}")
        
        try:
            # Check if scores are calculated
            if self.combined_scores is None or self.combined_scores.empty:
                logger.error("No scores calculated")
                return None
            
            # Check if the sort column exists
            if sort_by not in self.combined_scores.columns:
                logger.error(f"Sort column {sort_by} not found")
                return None
            
            # Apply additional filters from settings
            filtered_scores = self._apply_filters(self.combined_scores)
            
            # Sort by the specified score
            sorted_scores = filtered_scores.sort_values(by=sort_by, ascending=False)
            
            # Get top N
            top_stocks = sorted_scores.head(n)
            
            logger.info(f"Retrieved top {len(top_stocks)} stocks")
            return top_stocks
            
        except Exception as e:
            logger.error(f"Failed to get top stocks: {e}")
            return None
    
    def _apply_filters(self, scores):
        """
        Apply filters from settings to the scores.
        
        Args:
            scores (pd.DataFrame): Scores to filter
            
        Returns:
            pd.DataFrame: Filtered scores
        """
        try:
            filtered_scores = scores.copy()
            
            # Get filters from settings
            filter_settings = self.settings.get('filters', {})
            
            # Apply filters if fundamental data is available
            if self.fundamental_data and 'company_overview' in self.fundamental_data:
                company_data = self.fundamental_data['company_overview']
                
                # Ensure company_data uses the same index as scores
                if company_data is not None and not company_data.empty:
                    # Merge company data with scores
                    if 'Symbol' in company_data.columns:
                        company_data.set_index('Symbol', inplace=True)
                    
                    merged_data = filtered_scores.join(company_data, how='left')
                    
                    # Apply market cap filter
                    min_market_cap = filter_settings.get('min_market_cap')
                    if min_market_cap is not None:
                        merged_data = filter_by_market_cap(merged_data, min_market_cap=min_market_cap)
                    
                    # Apply sector filter
                    exclude_sectors = filter_settings.get('exclude_sectors')
                    if exclude_sectors is not None:
                        merged_data = filter_by_sector(merged_data, exclude_sectors=exclude_sectors)
                    
                    # Apply PE ratio filter
                    max_pe_ratio = filter_settings.get('max_pe_ratio')
                    if max_pe_ratio is not None and 'PERatio' in merged_data.columns:
                        merged_data = filter_by_value(merged_data, 'PERatio', max_value=max_pe_ratio)
                    
                    # Apply minimum return filter
                    min_return_1y = filter_settings.get('min_return_1y')
                    if min_return_1y is not None and 'momentum_12m' in merged_data.columns:
                        merged_data = filter_by_value(merged_data, 'momentum_12m', min_value=min_return_1y)
                    
                    # Apply volatility filter
                    max_volatility = filter_settings.get('max_volatility')
                    if max_volatility is not None and 'Volatility' in merged_data.columns:
                        merged_data = filter_by_value(merged_data, 'Volatility', max_value=max_volatility)
                    
                    # Keep only score columns
                    score_columns = [col for col in filtered_scores.columns]
                    filtered_scores = merged_data[score_columns]
            
            logger.info(f"Applied filters, {len(filtered_scores)} stocks remaining")
            return filtered_scores
            
        except Exception as e:
            logger.error(f"Failed to apply filters: {e}")
            return scores
    
    def get_stock_details(self, ticker):
        """
        Get detailed information for a specific stock.
        
        Args:
            ticker (str): Stock ticker
            
        Returns:
            dict: Dictionary with stock details
        """
        logger.info(f"Getting details for stock: {ticker}")
        
        try:
            details = {}
            
            # Get price data
            if ticker in self.price_data:
                details['price_data'] = self.price_data[ticker]
            
            # Get fundamental data
            if self.fundamental_data and 'company_overview' in self.fundamental_data:
                company_data = self.fundamental_data['company_overview']
                
                if company_data is not None and not company_data.empty:
                    if 'Symbol' in company_data.columns:
                        company_data.set_index('Symbol', inplace=True)
                    
                    if ticker in company_data.index:
                        details['company_data'] = company_data.loc[ticker].to_dict()
            
            # Get scores
            if self.combined_scores is not None and ticker in self.combined_scores.index:
                details['scores'] = self.combined_scores.loc[ticker].to_dict()
            
            # Get financial statements
            financial_statements = {}
            
            for statement_type in ['income_statements', 'balance_sheets', 'cash_flows']:
                if statement_type in self.fundamental_data and ticker in self.fundamental_data[statement_type]:
                    financial_statements[statement_type] = self.fundamental_data[statement_type][ticker]
            
            details['financial_statements'] = financial_statements
            
            logger.info(f"Retrieved details for {ticker}")
            return details
            
        except Exception as e:
            logger.error(f"Failed to get stock details: {e}")
            return {}
    
    def export_results(self, file_path):
        """
        Export screening results to a file.
        
        Args:
            file_path (str): Path to the output file
            
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info(f"Exporting results to {file_path}")
        
        try:
            # Check if results are available
            if self.combined_scores is None or self.combined_scores.empty:
                logger.error("No results to export")
                return False
            
            # Determine file format based on extension
            if file_path.endswith('.xlsx'):
                return export_to_excel(self.combined_scores, file_path)
            elif file_path.endswith('.csv'):
                return export_to_csv(self.combined_scores, file_path)
            else:
                logger.error(f"Unsupported file format: {file_path}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to export results: {e}")
            return False
    
    def get_index_performance(self, index_name, start_date=None, end_date=None):
        """
        Get performance data for an index.
        
        Args:
            index_name (str): Name of the index
            start_date (str, optional): Start date in format 'YYYY-MM-DD'. Defaults to 1 year ago.
            end_date (str, optional): End date in format 'YYYY-MM-DD'. Defaults to today.
            
        Returns:
            pd.Series: Series with index performance data
        """
        logger.info(f"Getting performance data for index: {index_name}")
        
        try:
            # Check if the index is defined in settings
            if 'indices' not in self.settings or index_name not in self.settings['indices']:
                logger.error(f"Index {index_name} not found in settings")
                return None
            
            index_settings = self.settings['indices'][index_name]
            index_symbol = index_settings.get('symbol')
            
            if not index_symbol:
                logger.error(f"No symbol defined for index {index_name}")
                return None
            
            # Set date range
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            
            # Get index data
            index_data = self.price_collector.collect_index_data(
                index_symbol, 
                start_date=start_date,
                end_date=end_date,
                source=index_settings.get('source', 'yahoo')
            )
            
            if index_data is None or index_data.empty:
                logger.error(f"Failed to get performance data for index {index_name}")
                return None
            
            logger.info(f"Retrieved performance data for index {index_name}")
            return index_data
            
        except Exception as e:
            logger.error(f"Failed to get index performance: {e}")
            return None
    
    def compare_performance(self, tickers, index_name=None, start_date=None, end_date=None):
        """
        Compare performance of selected stocks with an index.
        
        Args:
            tickers (list): List of stock tickers to compare
            index_name (str, optional): Name of the index to compare with. Defaults to None.
            start_date (str, optional): Start date in format 'YYYY-MM-DD'. Defaults to 1 year ago.
            end_date (str, optional): End date in format 'YYYY-MM-DD'. Defaults to today.
            
        Returns:
            pd.DataFrame: DataFrame with performance comparison
        """
        logger.info(f"Comparing performance of {len(tickers)} stocks with index {index_name}")
        
        try:
            # Set date range
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            
            # Initialize performance DataFrame
            performance = pd.DataFrame()
            
            # Get stock data
            for ticker in tickers:
                if ticker in self.price_data:
                    # Get price data for the stock
                    price_data = self.price_data[ticker]['prices']
                    
                    # Normalize to starting value
                    first_value = price_data.iloc[0]
                    normalized = price_data / first_value * 100
                    
                    # Add to performance DataFrame
                    performance[ticker] = normalized
            
            # Get index data if requested
            if index_name is not None:
                index_data = self.get_index_performance(
                    index_name,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if index_data is not None and not index_data.empty:
                    # Normalize to starting value
                    first_value = index_data.iloc[0]
                    normalized = index_data / first_value * 100
                    
                    # Add to performance DataFrame
                    performance[index_name] = normalized
            
            logger.info(f"Compared performance of {len(tickers)} stocks with index {index_name}")
            return performance
            
        except Exception as e:
            logger.error(f"Failed to compare performance: {e}")
            return None

    def load_custom_data(self, ticker_list=None, start_date=None, end_date=None):
        """
        Load data for a custom list of tickers.
        
        Args:
            ticker_list (list, optional): List of stock tickers. Defaults to None.
            start_date (str, optional): Start date in format 'YYYY-MM-DD'. Defaults to 1 year ago.
            end_date (str, optional): End date in format 'YYYY-MM-DD'. Defaults to today.
            
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info(f"Loading data for {len(ticker_list) if ticker_list else 0} custom tickers")
        
        try:
            if not ticker_list:
                logger.error("No tickers provided")
                return False
            
            # Determine date range
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            if start_date is None:
                days = self.settings.get('data', {}).get('price_history_days', 365)
                start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            # Get price data
            price_data = self.price_collector.collect_stock_data(
                ticker_list, 
                start_date=start_date,
                end_date=end_date,
                source='yahoo'  # Default to Yahoo for custom lists
            )
            
            if not price_data:
                logger.error("Failed to get price data for custom tickers")
                return False
            
            self.price_data = price_data
            
            # Get fundamental data (for a subset to avoid API limits)
            max_stocks = 50  # Limit to avoid API rate limits
            subset_tickers = list(price_data.keys())[:max_stocks]
            
            logger.info(f"Getting fundamental data for {len(subset_tickers)} custom tickers")
            fundamental_data = self.fundamental_collector.collect_all_fundamentals(subset_tickers)
            
            if not fundamental_data:
                logger.warning("Failed to get fundamental data for custom tickers")
                # Continue without fundamental data
            else:
                self.fundamental_data = fundamental_data
            
            logger.info(f"Successfully loaded data for {len(price_data)} custom tickers")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load custom data: {e}")
            return False
