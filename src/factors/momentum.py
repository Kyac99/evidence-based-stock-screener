"""
Module for calculating Momentum factor metrics.

The Momentum factor combines:
1. Technical momentum (price trend analysis)
2. Fundamental momentum (analyst revisions and earnings growth)
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.utils import normalize_score, winsorize

# Setup logging
logger = logging.getLogger('stock_screener.factors.momentum')

class MomentumFactor:
    """
    Class for calculating Momentum factor metrics.
    """
    
    def __init__(self):
        """
        Initialize the Momentum factor calculator.
        """
        self.price_data = None
        self.fundamental_data = None
    
    def set_data(self, price_data, fundamental_data=None):
        """
        Set the data for momentum calculations.
        
        Args:
            price_data (dict): Dictionary with processed price data
            fundamental_data (dict, optional): Dictionary with fundamental data. Defaults to None.
        """
        self.price_data = price_data
        self.fundamental_data = fundamental_data
    
    def calculate_price_momentum(self, periods=[1, 3, 6, 12]):
        """
        Calculate price momentum metrics for different time periods.
        
        Args:
            periods (list, optional): List of periods in months to calculate momentum.
                                      Defaults to [1, 3, 6, 12].
            
        Returns:
            pd.DataFrame: DataFrame with price momentum metrics
        """
        if self.price_data is None:
            logger.error("Price data not set, cannot calculate price momentum")
            return None
        
        logger.info("Calculating price momentum metrics")
        
        # Convert periods from months to days (approximately)
        days_per_month = 21  # Trading days per month (approximate)
        period_days = [p * days_per_month for p in periods]
        
        # Initialize DataFrame to store results
        results = []
        
        for ticker, data in self.price_data.items():
            try:
                # Get prices
                prices = data.get('prices')
                
                if prices is None or len(prices) == 0:
                    logger.warning(f"No price data for {ticker}, skipping")
                    continue
                
                # Get the latest date
                latest_date = prices.index[-1]
                
                # Calculate momentum for each period
                momentum_values = {}
                momentum_values['ticker'] = ticker
                
                for period, days in zip(periods, period_days):
                    # Calculate price change over the period
                    start_date = latest_date - timedelta(days=days)
                    
                    # Find the closest date if the exact start date is not available
                    close_dates = prices.index[prices.index <= start_date]
                    if len(close_dates) == 0:
                        # If no date before the start date, skip this period
                        momentum_values[f'momentum_{period}m'] = np.nan
                        continue
                    
                    start_date = close_dates[-1]
                    start_price = prices.loc[start_date]
                    end_price = prices.iloc[-1]
                    
                    # Momentum: (end_price / start_price) - 1
                    momentum = (end_price / start_price) - 1
                    momentum_values[f'momentum_{period}m'] = momentum * 100  # Convert to percentage
                
                # Add to results
                results.append(momentum_values)
                
            except Exception as e:
                logger.error(f"Failed to calculate price momentum for {ticker}: {e}")
        
        # Convert to DataFrame
        momentum_df = pd.DataFrame(results)
        
        # Set ticker as index
        if not momentum_df.empty and 'ticker' in momentum_df.columns:
            momentum_df = momentum_df.set_index('ticker')
        
        logger.info(f"Calculated price momentum for {len(momentum_df)} tickers")
        return momentum_df
    
    def calculate_analyst_momentum(self):
        """
        Calculate momentum based on analyst estimates and revisions.
        
        Returns:
            pd.DataFrame: DataFrame with analyst momentum metrics
        """
        if self.fundamental_data is None or not isinstance(self.fundamental_data, dict):
            logger.error("Fundamental data not set correctly, cannot calculate analyst momentum")
            return None
        
        logger.info("Calculating analyst momentum metrics")
        
        # Get analyst estimates from fundamental data
        estimates_df = self.fundamental_data.get('estimates')
        
        if estimates_df is None or len(estimates_df) == 0:
            logger.error("No analyst estimates data available")
            return None
        
        # Ensure 'ticker' is a column and not the index
        if 'ticker' not in estimates_df.columns and estimates_df.index.name == 'ticker':
            estimates_df = estimates_df.reset_index()
        
        # Calculate analyst momentum metrics
        analyst_momentum = []
        
        for _, row in estimates_df.iterrows():
            ticker = row['ticker']
            
            try:
                # Check for required columns
                required_columns = ['current_quarter_estimate', 'current_year_estimate', 
                                    'next_quarter_estimate', 'next_year_estimate',
                                    'buy_percentage', 'recommendation_score']
                
                missing_columns = [col for col in required_columns if col not in row.index]
                if missing_columns:
                    logger.warning(f"Missing columns for {ticker}: {missing_columns}")
                    continue
                
                # Calculate analyst consensus score (if available)
                if 'recommendation_score' in row:
                    analyst_consensus = row['recommendation_score']
                else:
                    analyst_consensus = None
                
                # Calculate buy percentage (if available)
                if 'buy_percentage' in row:
                    buy_percentage = row['buy_percentage']
                else:
                    buy_percentage = None
                
                # Calculate earnings growth (if available)
                if all(col in row.index for col in ['current_year_growth', 'next_year_growth']):
                    earnings_growth = (row['current_year_growth'] + row['next_year_growth']) / 2
                else:
                    earnings_growth = None
                
                # Create row for analyst momentum
                momentum_values = {
                    'ticker': ticker,
                    'analyst_consensus': analyst_consensus,
                    'buy_percentage': buy_percentage,
                    'earnings_growth': earnings_growth
                }
                
                # Add to results
                analyst_momentum.append(momentum_values)
                
            except Exception as e:
                logger.error(f"Failed to calculate analyst momentum for {ticker}: {e}")
        
        # Convert to DataFrame
        analyst_momentum_df = pd.DataFrame(analyst_momentum)
        
        # Set ticker as index
        if not analyst_momentum_df.empty and 'ticker' in analyst_momentum_df.columns:
            analyst_momentum_df = analyst_momentum_df.set_index('ticker')
        
        logger.info(f"Calculated analyst momentum for {len(analyst_momentum_df)} tickers")
        return analyst_momentum_df
    
    def calculate_combined_momentum(self, technical_weight=0.6, fundamental_weight=0.4):
        """
        Calculate combined momentum score using both price and analyst momentum.
        
        Args:
            technical_weight (float, optional): Weight for technical momentum. Defaults to 0.6.
            fundamental_weight (float, optional): Weight for fundamental momentum. Defaults to 0.4.
            
        Returns:
            pd.DataFrame: DataFrame with combined momentum scores
        """
        # Calculate price momentum
        price_momentum = self.calculate_price_momentum()
        
        # Calculate analyst momentum if fundamental data is available
        if self.fundamental_data is not None:
            analyst_momentum = self.calculate_analyst_momentum()
        else:
            analyst_momentum = None
            fundamental_weight = 0
            technical_weight = 1
        
        # If either calculation failed, adjust weights
        if price_momentum is None:
            logger.error("Failed to calculate price momentum")
            return None
        
        if analyst_momentum is None and fundamental_weight > 0:
            logger.warning("No analyst momentum data available, using only price momentum")
            fundamental_weight = 0
            technical_weight = 1
        
        # Normalize weights to sum to 1
        total_weight = technical_weight + fundamental_weight
        technical_weight = technical_weight / total_weight
        fundamental_weight = fundamental_weight / total_weight
        
        logger.info(f"Using weights: Technical={technical_weight:.2f}, Fundamental={fundamental_weight:.2f}")
        
        # Calculate technical momentum score
        momentum_columns = [col for col in price_momentum.columns if col.startswith('momentum_')]
        
        # Define weights for different time periods
        period_weights = {
            'momentum_1m': 0.1,   # 1 month: 10%
            'momentum_3m': 0.2,   # 3 months: 20%
            'momentum_6m': 0.3,   # 6 months: 30%
            'momentum_12m': 0.4   # 12 months: 40%
        }
        
        # Calculate technical momentum score
        technical_scores = pd.DataFrame(index=price_momentum.index)
        
        for col in momentum_columns:
            if col in period_weights:
                # Winsorize to handle outliers
                price_momentum[col] = winsorize(price_momentum[col])
                
                # Normalize to 0-100 scale
                technical_scores[col] = normalize_score(price_momentum[col])
        
        # Weighted average of technical scores
        technical_scores['technical_momentum_score'] = 0
        for col in momentum_columns:
            if col in period_weights and col in technical_scores.columns:
                technical_scores['technical_momentum_score'] += technical_scores[col] * period_weights[col]
        
        # Calculate fundamental momentum score (if available)
        if analyst_momentum is not None:
            fundamental_scores = pd.DataFrame(index=analyst_momentum.index)
            
            # Normalize metrics to 0-100 scale
            if 'analyst_consensus' in analyst_momentum.columns:
                fundamental_scores['analyst_consensus_score'] = normalize_score(analyst_momentum['analyst_consensus'])
            else:
                fundamental_scores['analyst_consensus_score'] = 50  # Neutral score
            
            if 'buy_percentage' in analyst_momentum.columns:
                fundamental_scores['buy_percentage_score'] = normalize_score(analyst_momentum['buy_percentage'])
            else:
                fundamental_scores['buy_percentage_score'] = 50  # Neutral score
            
            if 'earnings_growth' in analyst_momentum.columns:
                # Winsorize to handle outliers
                earnings_growth = winsorize(analyst_momentum['earnings_growth'])
                fundamental_scores['earnings_growth_score'] = normalize_score(earnings_growth)
            else:
                fundamental_scores['earnings_growth_score'] = 50  # Neutral score
            
            # Weighted average of fundamental scores
            fundamental_scores['fundamental_momentum_score'] = (
                fundamental_scores['analyst_consensus_score'] * 0.4 +
                fundamental_scores['buy_percentage_score'] * 0.3 +
                fundamental_scores['earnings_growth_score'] * 0.3
            )
        
        # Combine technical and fundamental scores
        momentum_scores = technical_scores[['technical_momentum_score']].copy()
        
        if analyst_momentum is not None:
            # Merge technical and fundamental scores
            momentum_scores = momentum_scores.join(
                fundamental_scores[['fundamental_momentum_score']], how='left'
            )
            
            # Fill missing values with neutral score
            momentum_scores['fundamental_momentum_score'] = momentum_scores['fundamental_momentum_score'].fillna(50)
            
            # Calculate combined score
            momentum_scores['momentum_score'] = (
                momentum_scores['technical_momentum_score'] * technical_weight +
                momentum_scores['fundamental_momentum_score'] * fundamental_weight
            )
        else:
            # If no fundamental data, technical score is the final score
            momentum_scores['momentum_score'] = momentum_scores['technical_momentum_score']
        
        logger.info(f"Calculated combined momentum scores for {len(momentum_scores)} tickers")
        return momentum_scores