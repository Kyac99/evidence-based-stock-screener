"""
Module for calculating Quality factor metrics.

The Quality factor focuses on:
1. Profitability metrics (ROE, ROCE, profit margins)
2. Balance sheet strength (debt levels, interest coverage)
3. Earnings stability and predictability
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
            fundamental_data (dict): Dictionary with processed fundamental data
        """
        self.fundamental_data = fundamental_data
    
    def calculate_profitability_metrics(self):
        """
        Calculate profitability metrics like ROE, ROCE, profit margins.
        
        Returns:
            pd.DataFrame: DataFrame with profitability metrics
        """
        if self.fundamental_data is None:
            logger.error("Fundamental data not set, cannot calculate profitability metrics")
            return None
        
        logger.info("Calculating profitability metrics")
        
        # Get company overview data
        company_overview = self.fundamental_data.get('company_overview')
        
        if company_overview is None or company_overview.empty:
            logger.error("No company overview data available")
            return None
        
        # Calculate profitability metrics
        try:
            # Ensure 'Symbol' is used as index if it's not already
            if 'Symbol' in company_overview.columns and company_overview.index.name != 'Symbol':
                company_overview.set_index('Symbol', inplace=True)
            
            # Initialize DataFrame for results
            metrics = pd.DataFrame(index=company_overview.index)
            
            # Extract profitability metrics from company overview
            profitability_columns = [
                'ReturnOnEquityTTM', 'ReturnOnAssetsTTM', 'ProfitMargin', 
                'OperatingMarginTTM', 'GrossProfitTTM'
            ]
            
            # Add the available metrics to the results
            for col in profitability_columns:
                if col in company_overview.columns:
                    metrics[col] = company_overview[col]
            
            # Get income statement and balance sheet data
            income_statements = self.fundamental_data.get('income_statements', {})
            balance_sheets = self.fundamental_data.get('balance_sheets', {})
            
            # Calculate additional metrics using financial statements
            for ticker in metrics.index:
                try:
                    # Get income statement and balance sheet for the ticker
                    income_data = income_statements.get(ticker, {}).get('annual')
                    balance_data = balance_sheets.get(ticker, {}).get('annual')
                    
                    if income_data is not None and not income_data.empty and balance_data is not None and not balance_data.empty:
                        # Get the most recent data
                        latest_income = income_data.iloc[0]
                        latest_balance = balance_data.iloc[0]
                        
                        # Calculate Return on Capital Employed (ROCE)
                        # ROCE = EBIT / (Total Assets - Current Liabilities)
                        if 'ebit' in latest_income and 'totalAssets' in latest_balance and 'totalCurrentLiabilities' in latest_balance:
                            ebit = latest_income['ebit']
                            total_assets = latest_balance['totalAssets']
                            current_liabilities = latest_balance['totalCurrentLiabilities']
                            
                            capital_employed = total_assets - current_liabilities
                            
                            if capital_employed > 0:
                                roce = ebit / capital_employed
                                metrics.at[ticker, 'ROCE'] = roce * 100  # Convert to percentage
                        
                        # Calculate Gross Margin
                        # Gross Margin = Gross Profit / Revenue
                        if 'grossProfit' in latest_income and 'totalRevenue' in latest_income and latest_income['totalRevenue'] > 0:
                            gross_margin = latest_income['grossProfit'] / latest_income['totalRevenue']
                            metrics.at[ticker, 'GrossMargin'] = gross_margin * 100  # Convert to percentage
                        
                        # Calculate Net Margin
                        # Net Margin = Net Income / Revenue
                        if 'netIncome' in latest_income and 'totalRevenue' in latest_income and latest_income['totalRevenue'] > 0:
                            net_margin = latest_income['netIncome'] / latest_income['totalRevenue']
                            metrics.at[ticker, 'NetMargin'] = net_margin * 100  # Convert to percentage
                        
                except Exception as e:
                    logger.error(f"Failed to calculate additional metrics for {ticker}: {e}")
            
            # Convert all values to numeric
            for col in metrics.columns:
                metrics[col] = pd.to_numeric(metrics[col], errors='coerce')
            
            logger.info(f"Calculated profitability metrics for {len(metrics)} companies")
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to calculate profitability metrics: {e}")
            return None
    
    def calculate_balance_sheet_metrics(self):
        """
        Calculate balance sheet metrics like debt-to-equity, interest coverage, etc.
        
        Returns:
            pd.DataFrame: DataFrame with balance sheet metrics
        """
        if self.fundamental_data is None:
            logger.error("Fundamental data not set, cannot calculate balance sheet metrics")
            return None
        
        logger.info("Calculating balance sheet metrics")
        
        # Get company overview data
        company_overview = self.fundamental_data.get('company_overview')
        
        if company_overview is None or company_overview.empty:
            logger.error("No company overview data available")
            return None
        
        # Calculate balance sheet metrics
        try:
            # Ensure 'Symbol' is used as index if it's not already
            if 'Symbol' in company_overview.columns and company_overview.index.name != 'Symbol':
                company_overview.set_index('Symbol', inplace=True)
            
            # Initialize DataFrame for results
            metrics = pd.DataFrame(index=company_overview.index)
            
            # Extract balance sheet metrics from company overview
            balance_sheet_columns = [
                'DebtToEquityRatio', 'CurrentRatio', 'QuickRatio'
            ]
            
            # Add the available metrics to the results
            for col in balance_sheet_columns:
                if col in company_overview.columns:
                    metrics[col] = company_overview[col]
            
            # Get income statement and balance sheet data
            income_statements = self.fundamental_data.get('income_statements', {})
            balance_sheets = self.fundamental_data.get('balance_sheets', {})
            
            # Calculate additional metrics using financial statements
            for ticker in metrics.index:
                try:
                    # Get income statement and balance sheet for the ticker
                    income_data = income_statements.get(ticker, {}).get('annual')
                    balance_data = balance_sheets.get(ticker, {}).get('annual')
                    
                    if income_data is not None and not income_data.empty and balance_data is not None and not balance_data.empty:
                        # Get the most recent data
                        latest_income = income_data.iloc[0]
                        latest_balance = balance_data.iloc[0]
                        
                        # Calculate Debt-to-Equity Ratio if not already available
                        if 'DebtToEquityRatio' not in metrics.columns or pd.isna(metrics.at[ticker, 'DebtToEquityRatio']):
                            if 'totalLiabilities' in latest_balance and 'totalShareholderEquity' in latest_balance:
                                total_liabilities = latest_balance['totalLiabilities']
                                total_equity = latest_balance['totalShareholderEquity']
                                
                                if total_equity > 0:
                                    debt_to_equity = total_liabilities / total_equity
                                    metrics.at[ticker, 'DebtToEquityRatio'] = debt_to_equity
                        
                        # Calculate Interest Coverage Ratio
                        # Interest Coverage = EBIT / Interest Expense
                        if 'ebit' in latest_income and 'interestExpense' in latest_income and latest_income['interestExpense'] != 0:
                            ebit = latest_income['ebit']
                            interest_expense = abs(latest_income['interestExpense'])  # Make sure it's positive
                            
                            interest_coverage = ebit / interest_expense
                            metrics.at[ticker, 'InterestCoverage'] = interest_coverage
                        
                        # Calculate Net Debt / EBITDA
                        # Net Debt = Total Debt - Cash
                        # Net Debt / EBITDA = Net Debt / EBITDA
                        if 'ebitda' in latest_income and 'shortLongTermDebtTotal' in latest_balance and 'cashAndShortTermInvestments' in latest_balance:
                            ebitda = latest_income['ebitda']
                            total_debt = latest_balance['shortLongTermDebtTotal']
                            cash = latest_balance['cashAndShortTermInvestments']
                            
                            net_debt = total_debt - cash
                            
                            if ebitda > 0:
                                net_debt_to_ebitda = net_debt / ebitda
                                metrics.at[ticker, 'NetDebtToEBITDA'] = net_debt_to_ebitda
                        
                except Exception as e:
                    logger.error(f"Failed to calculate balance sheet metrics for {ticker}: {e}")
            
            # Convert all values to numeric
            for col in metrics.columns:
                metrics[col] = pd.to_numeric(metrics[col], errors='coerce')
            
            logger.info(f"Calculated balance sheet metrics for {len(metrics)} companies")
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to calculate balance sheet metrics: {e}")
            return None
    
    def calculate_earnings_stability(self):
        """
        Calculate earnings stability metrics.
        
        Returns:
            pd.DataFrame: DataFrame with earnings stability metrics
        """
        if self.fundamental_data is None:
            logger.error("Fundamental data not set, cannot calculate earnings stability")
            return None
        
        logger.info("Calculating earnings stability metrics")
        
        # Get income statement data
        income_statements = self.fundamental_data.get('income_statements', {})
        
        if not income_statements:
            logger.error("No income statement data available")
            return None
        
        # Calculate earnings stability metrics
        try:
            # Initialize dictionary to store results
            stability_metrics = {}
            
            for ticker, statements in income_statements.items():
                # Get annual income statements
                annual_data = statements.get('annual')
                
                if annual_data is None or len(annual_data) < 3:
                    logger.warning(f"Not enough annual income data for {ticker}")
                    continue
                
                # Extract net income and revenue
                if 'netIncome' in annual_data.columns and 'totalRevenue' in annual_data.columns:
                    net_income = annual_data['netIncome']
                    revenue = annual_data['totalRevenue']
                    
                    # Calculate metrics
                    try:
                        # Net Income Volatility (coefficient of variation)
                        if not net_income.empty and net_income.mean() != 0:
                            ni_volatility = net_income.std() / abs(net_income.mean())
                        else:
                            ni_volatility = np.nan
                        
                        # Margin Volatility (standard deviation of net margin)
                        net_margin = net_income / revenue
                        margin_volatility = net_margin.std()
                        
                        # Earnings Growth Consistency
                        # Percentage of years with positive earnings growth
                        ni_growth = net_income.pct_change()
                        growth_consistency = (ni_growth > 0).mean() if len(ni_growth.dropna()) > 0 else np.nan
                        
                        # Store results
                        stability_metrics[ticker] = {
                            'NetIncomeVolatility': ni_volatility,
                            'MarginVolatility': margin_volatility,
                            'EarningsGrowthConsistency': growth_consistency
                        }
                        
                    except Exception as e:
                        logger.error(f"Failed to calculate earnings stability for {ticker}: {e}")
                
            # Convert to DataFrame
            stability_df = pd.DataFrame.from_dict(stability_metrics, orient='index')
            
            logger.info(f"Calculated earnings stability metrics for {len(stability_df)} companies")
            return stability_df
            
        except Exception as e:
            logger.error(f"Failed to calculate earnings stability: {e}")
            return None
    
    def calculate_combined_quality(self):
        """
        Calculate combined quality score using profitability, balance sheet, and earnings stability metrics.
        
        Returns:
            pd.DataFrame: DataFrame with combined quality scores
        """
        logger.info("Calculating combined quality score")
        
        # Calculate individual metric sets
        profitability = self.calculate_profitability_metrics()
        balance_sheet = self.calculate_balance_sheet_metrics()
        earnings_stability = self.calculate_earnings_stability()
        
        # Check if any calculations failed
        if profitability is None and balance_sheet is None and earnings_stability is None:
            logger.error("All quality metric calculations failed")
            return None
        
        # Create list of available DataFrames
        available_dfs = [df for df in [profitability, balance_sheet, earnings_stability] if df is not None]
        
        # Combine all metrics
        combined_metrics = pd.concat(available_dfs, axis=1)
        
        # Handle duplicated columns if any
        if any(combined_metrics.columns.duplicated()):
            combined_metrics = combined_metrics.loc[:, ~combined_metrics.columns.duplicated()]
        
        # Initialize quality scores
        quality_scores = pd.DataFrame(index=combined_metrics.index)
        
        try:
            # Profitability score components
            profitability_cols = ['ReturnOnEquityTTM', 'ReturnOnAssetsTTM', 'ROCE', 'ProfitMargin', 'OperatingMarginTTM']
            available_prof_cols = [col for col in profitability_cols if col in combined_metrics.columns]
            
            if available_prof_cols:
                # Higher is better for profitability metrics
                for col in available_prof_cols:
                    # Winsorize to handle outliers
                    combined_metrics[col] = winsorize(combined_metrics[col])
                    
                    # Normalize to 0-100 scale
                    quality_scores[f"{col}_score"] = normalize_score(combined_metrics[col])
                
                # Calculate weighted profitability score
                weights = {
                    'ReturnOnEquityTTM': 0.3,
                    'ROCE': 0.3,
                    'ReturnOnAssetsTTM': 0.15,
                    'ProfitMargin': 0.15,
                    'OperatingMarginTTM': 0.1
                }
                
                # Adjust weights based on available columns
                available_weights = {col: weights.get(col, 0.2) for col in available_prof_cols}
                total_weight = sum(available_weights.values())
                available_weights = {col: w/total_weight for col, w in available_weights.items()}
                
                # Calculate the weighted score
                quality_scores['profitability_score'] = 0
                for col in available_prof_cols:
                    quality_scores['profitability_score'] += quality_scores[f"{col}_score"] * available_weights[col]
            
            # Balance sheet score components
            balance_cols = ['DebtToEquityRatio', 'InterestCoverage', 'NetDebtToEBITDA', 'CurrentRatio']
            available_balance_cols = [col for col in balance_cols if col in combined_metrics.columns]
            
            if available_balance_cols:
                # For debt ratios, lower is better
                inverted_cols = ['DebtToEquityRatio', 'NetDebtToEBITDA']
                for col in available_balance_cols:
                    # Winsorize to handle outliers
                    combined_metrics[col] = winsorize(combined_metrics[col])
                    
                    # Normalize to 0-100 scale (inverting if needed)
                    if col in inverted_cols:
                        # Invert the score for metrics where lower is better
                        quality_scores[f"{col}_score"] = 100 - normalize_score(combined_metrics[col])
                    else:
                        quality_scores[f"{col}_score"] = normalize_score(combined_metrics[col])
                
                # Calculate weighted balance sheet score
                weights = {
                    'DebtToEquityRatio': 0.4,
                    'InterestCoverage': 0.3,
                    'NetDebtToEBITDA': 0.2,
                    'CurrentRatio': 0.1
                }
                
                # Adjust weights based on available columns
                available_weights = {col: weights.get(col, 0.25) for col in available_balance_cols}
                total_weight = sum(available_weights.values())
                available_weights = {col: w/total_weight for col, w in available_weights.items()}
                
                # Calculate the weighted score
                quality_scores['balance_sheet_score'] = 0
                for col in available_balance_cols:
                    quality_scores['balance_sheet_score'] += quality_scores[f"{col}_score"] * available_weights[col]
            
            # Earnings stability score components
            stability_cols = ['NetIncomeVolatility', 'MarginVolatility', 'EarningsGrowthConsistency']
            available_stability_cols = [col for col in stability_cols if col in combined_metrics.columns]
            
            if available_stability_cols:
                # For volatility metrics, lower is better
                inverted_cols = ['NetIncomeVolatility', 'MarginVolatility']
                for col in available_stability_cols:
                    # Winsorize to handle outliers
                    combined_metrics[col] = winsorize(combined_metrics[col])
                    
                    # Normalize to 0-100 scale (inverting if needed)
                    if col in inverted_cols:
                        # Invert the score for metrics where lower is better
                        quality_scores[f"{col}_score"] = 100 - normalize_score(combined_metrics[col])
                    else:
                        quality_scores[f"{col}_score"] = normalize_score(combined_metrics[col])
                
                # Calculate weighted earnings stability score
                weights = {
                    'NetIncomeVolatility': 0.4,
                    'MarginVolatility': 0.3,
                    'EarningsGrowthConsistency': 0.3
                }
                
                # Adjust weights based on available columns
                available_weights = {col: weights.get(col, 0.33) for col in available_stability_cols}
                total_weight = sum(available_weights.values())
                available_weights = {col: w/total_weight for col, w in available_weights.items()}
                
                # Calculate the weighted score
                quality_scores['earnings_stability_score'] = 0
                for col in available_stability_cols:
                    quality_scores['earnings_stability_score'] += quality_scores[f"{col}_score"] * available_weights[col]
            
            # Calculate overall quality score
            component_scores = ['profitability_score', 'balance_sheet_score', 'earnings_stability_score']
            available_components = [col for col in component_scores if col in quality_scores.columns]
            
            if available_components:
                # Default weights for overall score
                overall_weights = {
                    'profitability_score': 0.5,
                    'balance_sheet_score': 0.3,
                    'earnings_stability_score': 0.2
                }
                
                # Adjust weights based on available components
                available_overall_weights = {col: overall_weights.get(col, 1/len(available_components)) for col in available_components}
                total_weight = sum(available_overall_weights.values())
                available_overall_weights = {col: w/total_weight for col, w in available_overall_weights.items()}
                
                # Calculate the overall quality score
                quality_scores['quality_score'] = 0
                for col in available_components:
                    quality_scores['quality_score'] += quality_scores[col] * available_overall_weights[col]
            else:
                logger.warning("No component scores available for overall quality score calculation")
                quality_scores['quality_score'] = 50  # Neutral score
            
            # Keep only the final scores
            final_columns = ['quality_score'] + [col for col in component_scores if col in quality_scores.columns]
            quality_scores = quality_scores[final_columns]
            
            logger.info(f"Calculated combined quality scores for {len(quality_scores)} companies")
            return quality_scores
            
        except Exception as e:
            logger.error(f"Failed to calculate combined quality score: {e}")
            # Try to return a partial result if possible
            if 'quality_score' in quality_scores.columns:
                return quality_scores[['quality_score']]
            return None
