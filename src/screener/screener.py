"""
Main module for the stock screener that integrates Momentum and Quality factors.
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime

from src.data.collectors.price_data import PriceDataCollector
from src.data.collectors.fundamental_data import FundamentalDataCollector
from src.factors.momentum import MomentumFactor
from src.factors.quality import QualityFactor
from src.utils import ensure_dir_exists, INDICES

# Setup logging
logger = logging.getLogger('stock_screener.screener')

class StockScreener:
    """
    Main class for screening stocks based on multiple factors.
    """
    
    def __init__(self, data_dir='data'):
        """
        Initialize the stock screener.
        
        Args:
            data_dir (str, optional): Base directory for data storage. Defaults to 'data'.
        """
        self.data_dir = data_dir
        self.reports_dir = os.path.join('reports')
        
        # Ensure directories exist
        ensure_dir_exists(self.reports_dir)
        
        # Initialize data collectors
        self.price_collector = PriceDataCollector(data_dir)
        self.fundamental_collector = FundamentalDataCollector(data_dir)
        
        # Initialize factor calculators
        self.momentum_factor = MomentumFactor()
        self.quality_factor = QualityFactor()
        
        # Data containers
        self.price_data = None
        self.fundamental_data = None
        self.momentum_scores = None
        self.quality_scores = None
        self.combined_scores = None
        self.current_index = None
    
    def load_index_data(self, index_name, refresh_data=False, lookback_days=365*2):
        """
        Load data for a specific index.
        
        Args:
            index_name (str): Name of the index to load
            refresh_data (bool, optional): Whether to refresh the data. Defaults to False.
            lookback_days (int, optional): Days to look back for price data. Defaults to 2 years.
            
        Returns:
            bool: True if successful, False otherwise
        """
        if index_name not in INDICES:
            logger.error(f"Index {index_name} not found in supported indices")
            return False
        
        logger.info(f"Loading data for index: {index_name}")
        self.current_index = index_name
        
        # Get tickers for the index
        tickers = self.price_collector.get_index_tickers(index_name)
        
        if not tickers:
            logger.error(f"No tickers found for index {index_name}")
            return False
        
        logger.info(f"Found {len(tickers)} tickers for {index_name}")
        
        # Define filenames for data
        price_data_file = f"{index_name.lower()}_price_data.pkl"
        processed_price_file = f"{index_name.lower()}_processed_price_data.pkl"
        ratios_file = f"{index_name.lower()}_ratios.parquet"
        estimates_file = f"{index_name.lower()}_estimates.parquet"
        financials_file = f"{index_name.lower()}_financials.pkl"
        
        # Check if we need to refresh price data
        if refresh_data or not os.path.exists(os.path.join(self.data_dir, 'processed', 'prices', processed_price_file)):
            logger.info("Collecting fresh price data")
            
            # Get the latest price data
            price_data = self.price_collector.get_latest_data(tickers, lookback_days)
            
            # Save the raw and processed data
            self.price_collector.save_price_data(price_data, price_data_file)
            self.price_collector.save_processed_data(price_data, processed_price_file)
        else:
            # Load existing processed price data
            logger.info("Loading existing price data")
            price_data = self.price_collector.load_processed_data(processed_price_file)
        
        # Store price data
        self.price_data = price_data
        
        # Check if we need to refresh fundamental data
        if refresh_data or not os.path.exists(os.path.join(self.data_dir, 'raw', 'fundamentals', ratios_file)):
            logger.info("Collecting fresh fundamental data")
            
            # Get financial ratios
            ratios_df = self.fundamental_collector.get_financial_ratios(tickers)
            
            # Get analyst estimates
            estimates_df = self.fundamental_collector.get_analyst_estimates(tickers)
            
            # Get historical financials
            financials = self.fundamental_collector.get_historical_financials(tickers)
            
            # Save the raw data
            self.fundamental_collector.save_fundamental_data(ratios_df, ratios_file, 'ratios')
            self.fundamental_collector.save_fundamental_data(estimates_df, estimates_file, 'estimates')
            self.fundamental_collector.save_fundamental_data(financials, financials_file, 'financials')
            
            # Process the data
            processed_ratios = self.fundamental_collector.process_financial_ratios(ratios_df)
            processed_estimates = self.fundamental_collector.process_analyst_estimates(estimates_df)
            processed_metrics = self.fundamental_collector.process_financials(financials)
            
            # Save the processed data
            self.fundamental_collector.save_processed_fundamental_data(processed_ratios, f"processed_{ratios_file}", 'ratios')
            self.fundamental_collector.save_processed_fundamental_data(processed_estimates, f"processed_{estimates_file}", 'estimates')
            self.fundamental_collector.save_processed_fundamental_data(processed_metrics, f"processed_{financials_file}", 'metrics')
            
            # Store fundamental data
            self.fundamental_data = {
                'ratios': processed_ratios,
                'estimates': processed_estimates,
                'financials': processed_metrics
            }
        else:
            # Load existing processed fundamental data
            logger.info("Loading existing fundamental data")
            
            processed_ratios = self.fundamental_collector.load_processed_fundamental_data(f"processed_{ratios_file}", 'ratios')
            processed_estimates = self.fundamental_collector.load_processed_fundamental_data(f"processed_{estimates_file}", 'estimates')
            processed_metrics = self.fundamental_collector.load_processed_fundamental_data(f"processed_{financials_file}", 'metrics')
            
            # Store fundamental data
            self.fundamental_data = {
                'ratios': processed_ratios,
                'estimates': processed_estimates,
                'financials': processed_metrics
            }
        
        logger.info(f"Successfully loaded data for {index_name}")
        return True
    
    def calculate_scores(self, momentum_weight=0.5, quality_weight=0.5):
        """
        Calculate Momentum and Quality scores for the current data.
        
        Args:
            momentum_weight (float, optional): Weight for Momentum factor. Defaults to 0.5.
            quality_weight (float, optional): Weight for Quality factor. Defaults to 0.5.
            
        Returns:
            pd.DataFrame: DataFrame with combined scores
        """
        if self.price_data is None:
            logger.error("No price data loaded, please load data first")
            return None
        
        logger.info("Calculating Momentum and Quality scores")
        
        # Calculate Momentum scores
        self.momentum_factor.set_data(self.price_data, self.fundamental_data)
        self.momentum_scores = self.momentum_factor.calculate_combined_momentum()
        
        # Calculate Quality scores
        if self.fundamental_data is not None:
            self.quality_factor.set_data(self.fundamental_data)
            self.quality_scores = self.quality_factor.calculate_combined_quality()
        else:
            logger.warning("No fundamental data available, Quality scores will be neutral")
            # Create neutral quality scores for all tickers
            tickers = list(self.price_data.keys())
            self.quality_scores = pd.DataFrame(50, index=tickers, columns=['quality_score'])
        
        # Combine Momentum and Quality scores
        if self.momentum_scores is None or self.quality_scores is None:
            logger.error("Failed to calculate either Momentum or Quality scores")
            return None
        
        # Normalize weights to sum to 1
        total_weight = momentum_weight + quality_weight
        momentum_weight = momentum_weight / total_weight
        quality_weight = quality_weight / total_weight
        
        # Get the common tickers
        momentum_tickers = set(self.momentum_scores.index)
        quality_tickers = set(self.quality_scores.index)
        common_tickers = momentum_tickers.intersection(quality_tickers)
        
        if not common_tickers:
            logger.error("No common tickers between Momentum and Quality scores")
            return None
        
        # Create a DataFrame with the common tickers
        combined_scores = pd.DataFrame(index=list(common_tickers))
        
        # Add individual factor scores
        combined_scores['momentum_score'] = self.momentum_scores.loc[common_tickers, 'momentum_score']
        combined_scores['quality_score'] = self.quality_scores.loc[common_tickers, 'quality_score']
        
        # Calculate combined score
        combined_scores['combined_score'] = (
            combined_scores['momentum_score'] * momentum_weight +
            combined_scores['quality_score'] * quality_weight
        )
        
        # Sort by combined score (descending)
        combined_scores = combined_scores.sort_values('combined_score', ascending=False)
        
        # Store combined scores
        self.combined_scores = combined_scores
        
        logger.info(f"Calculated scores for {len(combined_scores)} tickers")
        return combined_scores
    
    def get_top_stocks(self, n=20, factor='combined'):
        """
        Get the top N stocks based on the specified factor.
        
        Args:
            n (int, optional): Number of stocks to return. Defaults to 20.
            factor (str, optional): Factor to use ('combined', 'momentum', 'quality'). Defaults to 'combined'.
            
        Returns:
            pd.DataFrame: DataFrame with the top N stocks
        """
        if self.combined_scores is None:
            logger.error("No scores calculated, please calculate scores first")
            return None
        
        # Select the appropriate score column
        if factor == 'combined':
            score_col = 'combined_score'
        elif factor == 'momentum':
            score_col = 'momentum_score'
        elif factor == 'quality':
            score_col = 'quality_score'
        else:
            logger.error(f"Invalid factor: {factor}")
            return None
        
        # Sort by the selected score (descending)
        sorted_scores = self.combined_scores.sort_values(score_col, ascending=False)
        
        # Get the top N stocks
        top_stocks = sorted_scores.head(n)
        
        # Add sector/industry information if available
        if self.fundamental_data is not None and 'ratios' in self.fundamental_data:
            ratios_df = self.fundamental_data['ratios']
            
            if 'sector' in ratios_df.columns and 'industry' in ratios_df.columns:
                # Create a mapping from ticker to sector/industry
                sector_map = ratios_df.set_index('ticker')['sector'].to_dict() if 'ticker' in ratios_df.columns else {}
                industry_map = ratios_df.set_index('ticker')['industry'].to_dict() if 'ticker' in ratios_df.columns else {}
                
                # Add sector/industry columns
                top_stocks['sector'] = top_stocks.index.map(lambda x: sector_map.get(x, 'Unknown'))
                top_stocks['industry'] = top_stocks.index.map(lambda x: industry_map.get(x, 'Unknown'))
        
        logger.info(f"Selected top {n} stocks based on {factor} factor")
        return top_stocks
    
    def filter_stocks(self, min_momentum=None, min_quality=None, sectors=None, industries=None):
        """
        Filter stocks based on various criteria.
        
        Args:
            min_momentum (float, optional): Minimum Momentum score. Defaults to None.
            min_quality (float, optional): Minimum Quality score. Defaults to None.
            sectors (list, optional): List of sectors to include. Defaults to None.
            industries (list, optional): List of industries to include. Defaults to None.
            
        Returns:
            pd.DataFrame: DataFrame with filtered stocks
        """
        if self.combined_scores is None:
            logger.error("No scores calculated, please calculate scores first")
            return None
        
        # Start with all stocks
        filtered_scores = self.combined_scores.copy()
        
        # Filter by Momentum score
        if min_momentum is not None:
            filtered_scores = filtered_scores[filtered_scores['momentum_score'] >= min_momentum]
        
        # Filter by Quality score
        if min_quality is not None:
            filtered_scores = filtered_scores[filtered_scores['quality_score'] >= min_quality]
        
        # Filter by sector and industry if fundamental data is available
        if (sectors or industries) and self.fundamental_data is not None and 'ratios' in self.fundamental_data:
            ratios_df = self.fundamental_data['ratios']
            
            if 'sector' in ratios_df.columns and 'industry' in ratios_df.columns:
                # Create a copy for filtering
                filtered_ratios = ratios_df.copy()
                
                # Filter by sector
                if sectors:
                    filtered_ratios = filtered_ratios[filtered_ratios['sector'].isin(sectors)]
                
                # Filter by industry
                if industries:
                    filtered_ratios = filtered_ratios[filtered_ratios['industry'].isin(industries)]
                
                # Get the filtered tickers
                filtered_tickers = set(filtered_ratios['ticker'])
                
                # Filter scores by the filtered tickers
                filtered_scores = filtered_scores[filtered_scores.index.isin(filtered_tickers)]
        
        # Sort by combined score (descending)
        filtered_scores = filtered_scores.sort_values('combined_score', ascending=False)
        
        logger.info(f"Filtered {len(filtered_scores)} stocks based on criteria")
        return filtered_scores
    
    def export_results(self, filename=None, format='xlsx', include_details=True):
        """
        Export screening results to a file.
        
        Args:
            filename (str, optional): Filename to export to. Defaults to None.
            format (str, optional): Export format ('xlsx', 'csv', 'html'). Defaults to 'xlsx'.
            include_details (bool, optional): Whether to include detailed metrics. Defaults to True.
            
        Returns:
            str: Path to the exported file
        """
        if self.combined_scores is None:
            logger.error("No scores calculated, please calculate scores first")
            return None
        
        # Create a copy of the scores for export
        export_df = self.combined_scores.copy()
        
        # Add ticker column (from index)
        export_df = export_df.reset_index().rename(columns={'index': 'ticker'})
        
        # Add sector/industry information if available
        if self.fundamental_data is not None and 'ratios' in self.fundamental_data:
            ratios_df = self.fundamental_data['ratios']
            
            if 'sector' in ratios_df.columns and 'industry' in ratios_df.columns:
                # Create a mapping from ticker to sector/industry
                sector_map = ratios_df.set_index('ticker')['sector'].to_dict() if 'ticker' in ratios_df.columns else {}
                industry_map = ratios_df.set_index('ticker')['industry'].to_dict() if 'ticker' in ratios_df.columns else {}
                
                # Add sector/industry columns
                export_df['sector'] = export_df['ticker'].map(lambda x: sector_map.get(x, 'Unknown'))
                export_df['industry'] = export_df['ticker'].map(lambda x: industry_map.get(x, 'Unknown'))
        
        # Add more detailed metrics if requested
        if include_details and self.price_data is not None and self.fundamental_data is not None:
            # Add price momentum metrics
            momentum_details = self.momentum_factor.calculate_price_momentum()
            if momentum_details is not None:
                momentum_details = momentum_details.reset_index()
                export_df = pd.merge(export_df, momentum_details, on='ticker', how='left')
            
            # Add profitability metrics
            profitability = self.quality_factor.calculate_profitability_metrics()
            if profitability is not None:
                profitability = profitability.reset_index()
                export_df = pd.merge(export_df, profitability, on='ticker', how='left')
            
            # Add financial health metrics
            financial_health = self.quality_factor.calculate_financial_health_metrics()
            if financial_health is not None:
                financial_health = financial_health.reset_index()
                export_df = pd.merge(export_df, financial_health, on='ticker', how='left')
        
        # Generate a filename if not provided
        if filename is None:
            current_date = datetime.now().strftime('%Y%m%d')
            index_name = self.current_index or 'stocks'
            filename = f"{index_name.lower()}_screening_results_{current_date}.{format}"
        
        # Ensure the extension matches the format
        if not filename.endswith(f'.{format}'):
            filename = f"{filename}.{format}"
        
        # Full path to the export file
        export_path = os.path.join(self.reports_dir, filename)
        
        # Export the data
        try:
            if format == 'xlsx':
                export_df.to_excel(export_path, index=False)
            elif format == 'csv':
                export_df.to_csv(export_path, index=False)
            elif format == 'html':
                export_df.to_html(export_path, index=False)
            else:
                logger.error(f"Unsupported export format: {format}")
                return None
            
            logger.info(f"Results exported to {export_path}")
            return export_path
        except Exception as e:
            logger.error(f"Failed to export results: {e}")
            return None
    
    def print_summary(self, n=20):
        """
        Print a summary of the screening results.
        
        Args:
            n (int, optional): Number of top stocks to include. Defaults to 20.
        """
        if self.combined_scores is None:
            logger.error("No scores calculated, please calculate scores first")
            return
        
        # Get the top N stocks
        top_stocks = self.get_top_stocks(n)
        
        # Print the summary
        print(f"\n{'=' * 60}")
        print(f"Stock Screening Results - {self.current_index}")
        print(f"{'=' * 60}")
        
        print(f"\nTop {n} Stocks by Combined Score:\n")
        
        # Format the DataFrame for display
        display_df = top_stocks.copy()
        display_df = display_df.reset_index().rename(columns={'index': 'Ticker'})
        
        # Round scores to 2 decimal places
        for col in ['momentum_score', 'quality_score', 'combined_score']:
            if col in display_df.columns:
                display_df[col] = display_df[col].round(2)
        
        # Rename columns for display
        rename_map = {
            'momentum_score': 'Momentum',
            'quality_score': 'Quality',
            'combined_score': 'Combined',
            'sector': 'Sector',
            'industry': 'Industry'
        }
        
        display_df = display_df.rename(columns=rename_map)
        
        # Select columns to display
        display_cols = ['Ticker', 'Combined', 'Momentum', 'Quality']
        if 'Sector' in display_df.columns:
            display_cols.append('Sector')
        if 'Industry' in display_df.columns:
            display_cols.append('Industry')
        
        display_df = display_df[display_cols]
        
        # Print the DataFrame
        print(display_df.to_string(index=False))
        
        print(f"\n{'=' * 60}")
        print("Note: Scores are normalized to a 0-100 scale, with higher values being better.")
        print(f"{'=' * 60}\n")