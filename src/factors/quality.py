"""
Module for calculating Quality factor metrics.

The Quality factor focuses on companies with:
1. Strong profitability
2. Healthy balance sheets
3. Consistent earnings
4. Good margin stability
5. High return on capital
"""

import logging
import pandas as pd
import numpy as np
from src.utils import normalize_score, winsorize

# Setup logging
logger = logging.getLogger('stock_screener.factors.quality')

class QualityFactor:
    """
    Class for calculating Quality factor metrics.
    """
    
    def __init__(self):
        """
        Initialize the Quality factor calculator.
        """
        self.fundamental_data = None
    
    def set_data(self, fundamental_data):
        """
        Set the data for quality calculations.
        
        Args:
            fundamental_data (dict): Dictionary with fundamental data
        """
        self.fundamental_data = fundamental_data
    
    def calculate_profitability_metrics(self):
        """
        Calculate profitability metrics from fundamental data.
        
        Returns:
            pd.DataFrame: DataFrame with profitability metrics
        """
        if self.fundamental_data is None or not isinstance(self.fundamental_data, dict):
            logger.error("Fundamental data not set correctly, cannot calculate profitability metrics")
            return None
        
        logger.info("Calculating profitability metrics")
        
        # Get financial ratios from fundamental data
        ratios_df = self.fundamental_data.get('ratios')
        
        if ratios_df is None or len(ratios_df) == 0:
            logger.error("No financial ratios data available")
            return None
        
        # Ensure 'ticker' is a column and not the index
        if 'ticker' not in ratios_df.columns and ratios_df.index.name == 'ticker':
            ratios_df = ratios_df.reset_index()
        
        # Select profitability metrics
        profitability_columns = [
            'ticker', 'roe', 'roa', 'profit_margin', 
            'operating_margin', 'relative_roe', 'relative_roa', 
            'relative_profit_margin', 'relative_operating_margin'
        ]
        
        # Filter columns that exist in the data
        available_columns = [col for col in profitability_columns if col in ratios_df.columns]
        
        # Create profitability DataFrame
        profitability_df = ratios_df[available_columns].copy()
        
        # Set ticker as index
        if 'ticker' in profitability_df.columns:
            profitability_df = profitability_df.set_index('ticker')
        
        logger.info(f"Calculated profitability metrics for {len(profitability_df)} tickers")
        return profitability_df
    
    def calculate_financial_health_metrics(self):
        """
        Calculate financial health metrics from fundamental data.
        
        Returns:
            pd.DataFrame: DataFrame with financial health metrics
        """
        if self.fundamental_data is None or not isinstance(self.fundamental_data, dict):
            logger.error("Fundamental data not set correctly, cannot calculate financial health metrics")
            return None
        
        logger.info("Calculating financial health metrics")
        
        # Get financial ratios from fundamental data
        ratios_df = self.fundamental_data.get('ratios')
        
        if ratios_df is None or len(ratios_df) == 0:
            logger.error("No financial ratios data available")
            return None
        
        # Ensure 'ticker' is a column and not the index
        if 'ticker' not in ratios_df.columns and ratios_df.index.name == 'ticker':
            ratios_df = ratios_df.reset_index()
        
        # Select financial health metrics
        health_columns = [
            'ticker', 'current_ratio', 'quick_ratio', 'debt_to_equity',
            'relative_current_ratio', 'relative_quick_ratio', 'relative_debt_to_equity'
        ]
        
        # Filter columns that exist in the data
        available_columns = [col for col in health_columns if col in ratios_df.columns]
        
        # Create financial health DataFrame
        health_df = ratios_df[available_columns].copy()
        
        # Set ticker as index
        if 'ticker' in health_df.columns:
            health_df = health_df.set_index('ticker')
        
        logger.info(f"Calculated financial health metrics for {len(health_df)} tickers")
        return health_df
    
    def calculate_earnings_stability(self):
        """
        Calculate earnings stability metrics from historical financial data.
        
        Returns:
            pd.DataFrame: DataFrame with earnings stability metrics
        """
        if self.fundamental_data is None or not isinstance(self.fundamental_data, dict):
            logger.error("Fundamental data not set correctly, cannot calculate earnings stability")
            return None
        
        logger.info("Calculating earnings stability metrics")
        
        # Get historical financials from fundamental data
        financials = self.fundamental_data.get('financials')
        
        if financials is None or not financials:
            logger.error("No historical financial data available")
            return None
        
        # Calculate earnings stability metrics
        stability_metrics = []
        
        for ticker, metrics in financials.items():
            try:
                # Check if metrics DataFrame exists and has required data
                if 'NetIncome' not in metrics.columns:
                    logger.warning(f"Missing net income data for {ticker}")
                    continue
                
                # Get net income data
                net_income = metrics['NetIncome']
                
                # Calculate coefficient of variation (standard deviation / mean)
                # Lower coefficient means more stable earnings
                if len(net_income) > 1 and net_income.mean() != 0:
                    cv = net_income.std() / abs(net_income.mean())
                    
                    # Invert to make higher values better (for consistency with other metrics)
                    # Cap at 1 to handle very low CV values
                    earnings_stability = min(1 / max(cv, 0.01), 100)
                else:
                    earnings_stability = None
                
                # Calculate year-over-year growth
                if len(net_income) > 1:
                    yoy_growth = []
                    
                    for i in range(1, len(net_income)):
                        if net_income.iloc[i-1] > 0:
                            growth = (net_income.iloc[i] - net_income.iloc[i-1]) / net_income.iloc[i-1]
                            yoy_growth.append(growth)
                    
                    # Calculate average growth and growth stability
                    if yoy_growth:
                        avg_growth = np.mean(yoy_growth)
                        growth_stability = np.std(yoy_growth) if len(yoy_growth) > 1 else 0
                    else:
                        avg_growth = None
                        growth_stability = None
                else:
                    avg_growth = None
                    growth_stability = None
                
                # Add to results
                stability_metrics.append({
                    'ticker': ticker,
                    'earnings_stability': earnings_stability,
                    'avg_growth': avg_growth,
                    'growth_stability': growth_stability
                })
                
            except Exception as e:
                logger.error(f"Failed to calculate earnings stability for {ticker}: {e}")
        
        # Convert to DataFrame
        stability_df = pd.DataFrame(stability_metrics)
        
        # Set ticker as index
        if not stability_df.empty and 'ticker' in stability_df.columns:
            stability_df = stability_df.set_index('ticker')
        
        logger.info(f"Calculated earnings stability metrics for {len(stability_df)} tickers")
        return stability_df
    
    def calculate_margin_stability(self):
        """
        Calculate margin stability metrics from historical financial data.
        
        Returns:
            pd.DataFrame: DataFrame with margin stability metrics
        """
        if self.fundamental_data is None or not isinstance(self.fundamental_data, dict):
            logger.error("Fundamental data not set correctly, cannot calculate margin stability")
            return None
        
        logger.info("Calculating margin stability metrics")
        
        # Get historical financials from fundamental data
        financials = self.fundamental_data.get('financials')
        
        if financials is None or not financials:
            logger.error("No historical financial data available")
            return None
        
        # Calculate margin stability metrics
        margin_metrics = []
        
        for ticker, metrics in financials.items():
            try:
                # Check if metrics DataFrame exists and has required columns
                required_columns = ['ProfitMargin']
                
                if not all(col in metrics.columns for col in required_columns):
                    logger.warning(f"Missing profit margin data for {ticker}")
                    continue
                
                # Get margin data
                profit_margin = metrics['ProfitMargin']
                
                # Calculate coefficient of variation for profit margin
                if len(profit_margin) > 1 and profit_margin.mean() != 0:
                    cv = profit_margin.std() / abs(profit_margin.mean())
                    
                    # Invert to make higher values better (for consistency with other metrics)
                    # Cap at 1 to handle very low CV values
                    margin_stability = min(1 / max(cv, 0.01), 100)
                else:
                    margin_stability = None
                
                # Add to results
                margin_metrics.append({
                    'ticker': ticker,
                    'margin_stability': margin_stability,
                    'avg_profit_margin': profit_margin.mean() if len(profit_margin) > 0 else None
                })
                
            except Exception as e:
                logger.error(f"Failed to calculate margin stability for {ticker}: {e}")
        
        # Convert to DataFrame
        margin_df = pd.DataFrame(margin_metrics)
        
        # Set ticker as index
        if not margin_df.empty and 'ticker' in margin_df.columns:
            margin_df = margin_df.set_index('ticker')
        
        logger.info(f"Calculated margin stability metrics for {len(margin_df)} tickers")
        return margin_df
    
    def calculate_return_on_capital(self):
        """
        Calculate return on capital metrics from fundamental data.
        
        Returns:
            pd.DataFrame: DataFrame with return on capital metrics
        """
        if self.fundamental_data is None or not isinstance(self.fundamental_data, dict):
            logger.error("Fundamental data not set correctly, cannot calculate return on capital")
            return None
        
        logger.info("Calculating return on capital metrics")
        
        # Get financial ratios from fundamental data
        ratios_df = self.fundamental_data.get('ratios')
        
        if ratios_df is None or len(ratios_df) == 0:
            logger.error("No financial ratios data available")
            return None
        
        # Get historical financials from fundamental data
        financials = self.fundamental_data.get('financials')
        
        # Ensure 'ticker' is a column and not the index in ratios_df
        if 'ticker' not in ratios_df.columns and ratios_df.index.name == 'ticker':
            ratios_df = ratios_df.reset_index()
        
        # Calculate return on capital metrics
        roc_metrics = []
        
        for _, row in ratios_df.iterrows():
            ticker = row['ticker']
            
            try:
                # Get ROE and ROA from ratios (if available)
                roe = row.get('roe')
                roa = row.get('roa')
                
                # Calculate ROCE from financials (if available)
                roce = None
                
                if financials is not None and ticker in financials:
                    metrics = financials[ticker]
                    
                    if 'NetIncome' in metrics.columns and 'Total Assets' in metrics.index and 'Total Current Liabilities' in metrics.index:
                        # Calculate average values for the last few years
                        net_income = metrics['NetIncome'].mean()
                        total_assets = metrics.loc['Total Assets'].mean()
                        current_liabilities = metrics.loc['Total Current Liabilities'].mean()
                        
                        # Calculate ROCE: Net Income / (Total Assets - Current Liabilities)
                        if total_assets - current_liabilities > 0:
                            roce = net_income / (total_assets - current_liabilities) * 100
                
                # Add to results
                roc_metrics.append({
                    'ticker': ticker,
                    'roe': roe,
                    'roa': roa,
                    'roce': roce
                })
                
            except Exception as e:
                logger.error(f"Failed to calculate return on capital for {ticker}: {e}")
        
        # Convert to DataFrame
        roc_df = pd.DataFrame(roc_metrics)
        
        # Set ticker as index
        if not roc_df.empty and 'ticker' in roc_df.columns:
            roc_df = roc_df.set_index('ticker')
        
        logger.info(f"Calculated return on capital metrics for {len(roc_df)} tickers")
        return roc_df
    
    def calculate_combined_quality(self):
        """
        Calculate combined quality score using all quality metrics.
        
        Returns:
            pd.DataFrame: DataFrame with combined quality scores
        """
        # Calculate individual quality components
        profitability = self.calculate_profitability_metrics()
        financial_health = self.calculate_financial_health_metrics()
        earnings_stability = self.calculate_earnings_stability()
        margin_stability = self.calculate_margin_stability()
        return_on_capital = self.calculate_return_on_capital()
        
        # If any calculation failed, we'll still proceed with what we have
        failed_components = []
        
        if profitability is None:
            failed_components.append("profitability")
        if financial_health is None:
            failed_components.append("financial health")
        if earnings_stability is None:
            failed_components.append("earnings stability")
        if margin_stability is None:
            failed_components.append("margin stability")
        if return_on_capital is None:
            failed_components.append("return on capital")
        
        if failed_components:
            logger.warning(f"Missing data for components: {', '.join(failed_components)}")
        
        if profitability is None and financial_health is None and return_on_capital is None:
            logger.error("Failed to calculate key quality components")
            return None
        
        # Collect all available tickers from the components
        tickers = set()
        
        if profitability is not None:
            tickers.update(profitability.index)
        if financial_health is not None:
            tickers.update(financial_health.index)
        if earnings_stability is not None:
            tickers.update(earnings_stability.index)
        if margin_stability is not None:
            tickers.update(margin_stability.index)
        if return_on_capital is not None:
            tickers.update(return_on_capital.index)
        
        # Create a DataFrame with all tickers
        quality_scores = pd.DataFrame(index=list(tickers))
        
        # Calculate profitability score
        if profitability is not None:
            # Select metrics to use
            profit_metrics = []
            if 'profit_margin' in profitability.columns:
                profit_metrics.append('profit_margin')
            if 'operating_margin' in profitability.columns:
                profit_metrics.append('operating_margin')
            if 'roe' in profitability.columns:
                profit_metrics.append('roe')
            if 'roa' in profitability.columns:
                profit_metrics.append('roa')
            
            # Use relative metrics if available, otherwise use absolute
            relative_metrics = [f'relative_{metric}' for metric in profit_metrics if f'relative_{metric}' in profitability.columns]
            
            if relative_metrics:
                profit_metrics = relative_metrics
            
            # Calculate profitability score
            if profit_metrics:
                # Winsorize to handle outliers
                for metric in profit_metrics:
                    profitability[f'{metric}_winsorized'] = winsorize(profitability[metric])
                    profitability[f'{metric}_score'] = normalize_score(profitability[f'{metric}_winsorized'])
                
                # Calculate average score
                score_columns = [f'{metric}_score' for metric in profit_metrics]
                profitability['profitability_score'] = profitability[score_columns].mean(axis=1)
                
                # Add to quality scores
                quality_scores = quality_scores.join(profitability[['profitability_score']], how='left')
        
        # Calculate financial health score
        if financial_health is not None:
            # Select metrics to use
            health_metrics = []
            if 'current_ratio' in financial_health.columns:
                health_metrics.append('current_ratio')
            if 'quick_ratio' in financial_health.columns:
                health_metrics.append('quick_ratio')
            if 'debt_to_equity' in financial_health.columns:
                health_metrics.append('debt_to_equity')
                # For debt_to_equity, lower is better, so invert the score
                financial_health['debt_to_equity'] = 1 / financial_health['debt_to_equity'].replace([0, np.inf, -np.inf], np.nan)
            
            # Use relative metrics if available, otherwise use absolute
            relative_metrics = [f'relative_{metric}' for metric in health_metrics if f'relative_{metric}' in financial_health.columns]
            
            if relative_metrics:
                health_metrics = relative_metrics
            
            # Calculate financial health score
            if health_metrics:
                # Winsorize to handle outliers
                for metric in health_metrics:
                    financial_health[f'{metric}_winsorized'] = winsorize(financial_health[metric])
                    financial_health[f'{metric}_score'] = normalize_score(financial_health[f'{metric}_winsorized'])
                
                # Calculate average score
                score_columns = [f'{metric}_score' for metric in health_metrics]
                financial_health['financial_health_score'] = financial_health[score_columns].mean(axis=1)
                
                # Add to quality scores
                quality_scores = quality_scores.join(financial_health[['financial_health_score']], how='left')
        
        # Calculate earnings stability score
        if earnings_stability is not None:
            # Winsorize to handle outliers
            if 'earnings_stability' in earnings_stability.columns:
                earnings_stability['earnings_stability_winsorized'] = winsorize(earnings_stability['earnings_stability'])
                earnings_stability['earnings_stability_score'] = normalize_score(earnings_stability['earnings_stability_winsorized'])
            
            if 'growth_stability' in earnings_stability.columns:
                # For growth_stability, lower is better, so invert the score
                earnings_stability['growth_stability'] = 1 / earnings_stability['growth_stability'].replace([0, np.inf, -np.inf], np.nan)
                earnings_stability['growth_stability_winsorized'] = winsorize(earnings_stability['growth_stability'])
                earnings_stability['growth_stability_score'] = normalize_score(earnings_stability['growth_stability_winsorized'])
            
            # Calculate earnings stability score
            stability_scores = []
            if 'earnings_stability_score' in earnings_stability.columns:
                stability_scores.append('earnings_stability_score')
            if 'growth_stability_score' in earnings_stability.columns:
                stability_scores.append('growth_stability_score')
            
            if stability_scores:
                earnings_stability['earnings_stability_score_combined'] = earnings_stability[stability_scores].mean(axis=1)
                
                # Add to quality scores
                quality_scores = quality_scores.join(earnings_stability[['earnings_stability_score_combined']], how='left')
        
        # Calculate margin stability score
        if margin_stability is not None:
            # Winsorize to handle outliers
            if 'margin_stability' in margin_stability.columns:
                margin_stability['margin_stability_winsorized'] = winsorize(margin_stability['margin_stability'])
                margin_stability['margin_stability_score'] = normalize_score(margin_stability['margin_stability_winsorized'])
            
            if 'avg_profit_margin' in margin_stability.columns:
                margin_stability['avg_profit_margin_winsorized'] = winsorize(margin_stability['avg_profit_margin'])
                margin_stability['avg_profit_margin_score'] = normalize_score(margin_stability['avg_profit_margin_winsorized'])
            
            # Calculate margin stability score
            margin_scores = []
            if 'margin_stability_score' in margin_stability.columns:
                margin_scores.append('margin_stability_score')
            if 'avg_profit_margin_score' in margin_stability.columns:
                margin_scores.append('avg_profit_margin_score')
            
            if margin_scores:
                margin_stability['margin_stability_score_combined'] = margin_stability[margin_scores].mean(axis=1)
                
                # Add to quality scores
                quality_scores = quality_scores.join(margin_stability[['margin_stability_score_combined']], how='left')
        
        # Calculate return on capital score
        if return_on_capital is not None:
            # Winsorize to handle outliers
            for metric in ['roe', 'roa', 'roce']:
                if metric in return_on_capital.columns:
                    return_on_capital[f'{metric}_winsorized'] = winsorize(return_on_capital[metric])
                    return_on_capital[f'{metric}_score'] = normalize_score(return_on_capital[f'{metric}_winsorized'])
            
            # Calculate return on capital score
            roc_scores = []
            for metric in ['roe', 'roa', 'roce']:
                if f'{metric}_score' in return_on_capital.columns:
                    roc_scores.append(f'{metric}_score')
            
            if roc_scores:
                return_on_capital['return_on_capital_score'] = return_on_capital[roc_scores].mean(axis=1)
                
                # Add to quality scores
                quality_scores = quality_scores.join(return_on_capital[['return_on_capital_score']], how='left')
        
        # Fill missing values with neutral score (50)
        quality_scores = quality_scores.fillna(50)
        
        # Calculate combined quality score
        # Define weights for different components
        component_weights = {
            'profitability_score': 0.3,
            'financial_health_score': 0.2,
            'earnings_stability_score_combined': 0.2,
            'margin_stability_score_combined': 0.15,
            'return_on_capital_score': 0.15
        }
        
        # Initialize combined score
        quality_scores['quality_score'] = 0
        
        # Calculate weighted average
        total_weight = 0
        
        for component, weight in component_weights.items():
            if component in quality_scores.columns:
                quality_scores['quality_score'] += quality_scores[component] * weight
                total_weight += weight
        
        # Normalize based on available components
        if total_weight > 0:
            quality_scores['quality_score'] = quality_scores['quality_score'] / total_weight
        
        logger.info(f"Calculated combined quality scores for {len(quality_scores)} tickers")
        return quality_scores