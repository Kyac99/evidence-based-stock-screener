"""
Module for generating charts and visualizations for the stock screener.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

from src.utils import ensure_dir_exists


class ChartGenerator:
    """
    Class for generating visualizations from stock screener results.
    """
    
    def __init__(self, output_dir='reports/visualizations'):
        """
        Initialize the chart generator.
        
        Args:
            output_dir (str, optional): Directory to save visualizations. Defaults to 'reports/visualizations'.
        """
        self.output_dir = output_dir
        ensure_dir_exists(output_dir)
        
        # Set default styles
        sns.set_style("whitegrid")
        plt.rcParams.update({
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.titlesize': 20
        })
    
    def plot_factor_scores(self, scores_df, top_n=30, title=None, save_as=None, interactive=True):
        """
        Plot the factor scores for the top N stocks.
        
        Args:
            scores_df (pd.DataFrame): DataFrame with stock scores.
            top_n (int, optional): Number of top stocks to plot. Defaults to 30.
            title (str, optional): Title for the plot. Defaults to None.
            save_as (str, optional): Filename to save the plot. Defaults to None.
            interactive (bool, optional): Whether to use Plotly (True) or Matplotlib (False). Defaults to True.
            
        Returns:
            Object: Figure object (either Plotly or Matplotlib)
        """
        # Ensure scores_df has an index name 'ticker'
        scores_df = scores_df.copy()
        if scores_df.index.name is None:
            scores_df.index.name = 'ticker'
        
        # Get the top N stocks by combined score
        if 'combined_score' in scores_df.columns:
            top_stocks = scores_df.sort_values('combined_score', ascending=False).head(top_n)
        else:
            top_stocks = scores_df.head(top_n)
        
        # Reset index to get ticker as a column
        top_stocks = top_stocks.reset_index()
        
        if interactive:
            # Create a Plotly figure
            fig = make_subplots(rows=1, cols=1)
            
            # Add momentum scores
            if 'momentum_score' in top_stocks.columns:
                fig.add_trace(
                    go.Bar(
                        y=top_stocks['ticker'],
                        x=top_stocks['momentum_score'],
                        name='Momentum',
                        marker_color='#1f77b4',
                        orientation='h'
                    )
                )
            
            # Add quality scores
            if 'quality_score' in top_stocks.columns:
                fig.add_trace(
                    go.Bar(
                        y=top_stocks['ticker'],
                        x=top_stocks['quality_score'],
                        name='Quality',
                        marker_color='#ff7f0e',
                        orientation='h'
                    )
                )
            
            # Add combined scores
            if 'combined_score' in top_stocks.columns:
                fig.add_trace(
                    go.Scatter(
                        y=top_stocks['ticker'],
                        x=top_stocks['combined_score'],
                        name='Combined',
                        mode='markers',
                        marker=dict(
                            size=12,
                            color='#2ca02c',
                            symbol='diamond'
                        )
                    )
                )
            
            # Update layout
            fig.update_layout(
                title=title or f"Top {top_n} Stocks by Factor Scores",
                xaxis_title="Score",
                yaxis_title="Ticker",
                legend_title="Factor",
                height=max(600, top_n * 20),  # Dynamic height based on number of stocks
                margin=dict(l=100, r=20, t=50, b=50),
                barmode='group'
            )
            
            # Sort y-axis by combined score or first available score
            score_cols = [col for col in ['combined_score', 'momentum_score', 'quality_score'] 
                          if col in top_stocks.columns]
            if score_cols:
                sort_col = score_cols[0]
                fig.update_layout(
                    yaxis=dict(
                        categoryorder='array',
                        categoryarray=top_stocks.sort_values(sort_col, ascending=True)['ticker']
                    )
                )
            
            # Save the figure if requested
            if save_as:
                save_path = os.path.join(self.output_dir, save_as)
                fig.write_html(save_path)
                print(f"Saved figure to {save_path}")
            
            return fig
        
        else:
            # Create a Matplotlib figure
            fig, ax = plt.subplots(figsize=(12, max(8, top_n * 0.4)))
            
            # Width of the bars
            bar_width = 0.35
            
            # Positions of the bars on the y-axis
            indices = np.arange(len(top_stocks))
            
            # Plot momentum scores
            if 'momentum_score' in top_stocks.columns:
                ax.barh(indices - bar_width/2, top_stocks['momentum_score'], bar_width, 
                        label='Momentum', color='#1f77b4', alpha=0.8)
            
            # Plot quality scores
            if 'quality_score' in top_stocks.columns:
                ax.barh(indices + bar_width/2, top_stocks['quality_score'], bar_width, 
                        label='Quality', color='#ff7f0e', alpha=0.8)
            
            # Plot combined scores
            if 'combined_score' in top_stocks.columns:
                ax.scatter(top_stocks['combined_score'], indices, color='#2ca02c', 
                           s=100, marker='D', label='Combined', zorder=3)
            
            # Add labels and title
            ax.set_xlabel('Score')
            ax.set_yticks(indices)
            ax.set_yticklabels(top_stocks['ticker'])
            ax.set_title(title or f"Top {top_n} Stocks by Factor Scores")
            ax.legend()
            
            # Add a grid for readability
            ax.grid(True, axis='x', alpha=0.3)
            
            # Tight layout
            plt.tight_layout()
            
            # Save the figure if requested
            if save_as:
                save_path = os.path.join(self.output_dir, save_as)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Saved figure to {save_path}")
            
            return fig
    
    def plot_sector_distribution(self, scores_df, top_n=50, title=None, save_as=None, interactive=True):
        """
        Plot the sector distribution of the top N stocks.
        
        Args:
            scores_df (pd.DataFrame): DataFrame with stock scores and sector information.
            top_n (int, optional): Number of top stocks to analyze. Defaults to 50.
            title (str, optional): Title for the plot. Defaults to None.
            save_as (str, optional): Filename to save the plot. Defaults to None.
            interactive (bool, optional): Whether to use Plotly (True) or Matplotlib (False). Defaults to True.
            
        Returns:
            Object: Figure object (either Plotly or Matplotlib)
        """
        # Check if sector information is available
        if 'sector' not in scores_df.columns and 'Sector' not in scores_df.columns:
            print("Sector information not available in the DataFrame")
            return None
        
        # Copy and prepare the DataFrame
        df = scores_df.copy()
        
        # Standardize column names
        if 'Sector' in df.columns:
            df.rename(columns={'Sector': 'sector'}, inplace=True)
        
        # Get the top N stocks by combined score
        if 'combined_score' in df.columns:
            top_stocks = df.sort_values('combined_score', ascending=False).head(top_n)
        else:
            top_stocks = df.head(top_n)
        
        # Count sectors
        sector_counts = top_stocks['sector'].value_counts().reset_index()
        sector_counts.columns = ['Sector', 'Count']
        
        # Calculate percentage
        total = sector_counts['Count'].sum()
        sector_counts['Percentage'] = (sector_counts['Count'] / total * 100).round(1)
        
        # Sort by count (descending)
        sector_counts = sector_counts.sort_values('Count', ascending=False)
        
        if interactive:
            # Create a Plotly figure
            fig = make_subplots(rows=1, cols=2, 
                               specs=[[{"type": "bar"}, {"type": "pie"}]],
                               column_widths=[0.6, 0.4],
                               subplot_titles=["Sector Counts", "Sector Distribution"])
            
            # Add bar chart
            fig.add_trace(
                go.Bar(
                    x=sector_counts['Sector'],
                    y=sector_counts['Count'],
                    text=sector_counts['Count'],
                    textposition='auto',
                    marker_color='#1f77b4',
                    name='Count'
                ),
                row=1, col=1
            )
            
            # Add pie chart
            fig.add_trace(
                go.Pie(
                    labels=sector_counts['Sector'],
                    values=sector_counts['Count'],
                    textinfo='label+percent',
                    hoverinfo='label+percent+value',
                    hole=0.3
                ),
                row=1, col=2
            )
            
            # Update layout
            fig.update_layout(
                title=title or f"Sector Distribution of Top {top_n} Stocks",
                height=600,
                showlegend=False
            )
            
            # Save the figure if requested
            if save_as:
                save_path = os.path.join(self.output_dir, save_as)
                fig.write_html(save_path)
                print(f"Saved figure to {save_path}")
            
            return fig
        
        else:
            # Create a Matplotlib figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # Bar chart of sector counts
            bars = ax1.bar(sector_counts['Sector'], sector_counts['Count'], color='#1f77b4')
            ax1.set_title('Sector Counts')
            ax1.set_xlabel('Sector')
            ax1.set_ylabel('Number of Stocks')
            ax1.tick_params(axis='x', rotation=45)
            
            # Add count labels
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{height:.0f}', ha='center', va='bottom')
            
            # Pie chart of sector distribution
            ax2.pie(sector_counts['Count'], labels=sector_counts['Sector'], autopct='%1.1f%%',
                   shadow=False, startangle=90)
            ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            ax2.set_title('Sector Distribution')
            
            # Overall title
            plt.suptitle(title or f"Sector Distribution of Top {top_n} Stocks", fontsize=16)
            
            # Tight layout
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
            
            # Save the figure if requested
            if save_as:
                save_path = os.path.join(self.output_dir, save_as)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Saved figure to {save_path}")
            
            return fig
    
    def plot_factor_heatmap(self, scores_df, top_n=30, title=None, save_as=None, interactive=True):
        """
        Plot a heatmap of momentum and quality scores.
        
        Args:
            scores_df (pd.DataFrame): DataFrame with stock scores.
            top_n (int, optional): Number of top stocks to plot. Defaults to 30.
            title (str, optional): Title for the plot. Defaults to None.
            save_as (str, optional): Filename to save the plot. Defaults to None.
            interactive (bool, optional): Whether to use Plotly (True) or Matplotlib (False). Defaults to True.
            
        Returns:
            Object: Figure object (either Plotly or Matplotlib)
        """
        # Check if required scores are available
        if 'momentum_score' not in scores_df.columns or 'quality_score' not in scores_df.columns:
            print("Both momentum_score and quality_score are required for this plot")
            return None
        
        # Copy and prepare the DataFrame
        df = scores_df.copy()
        
        # Get the top N stocks by combined score
        if 'combined_score' in df.columns:
            top_stocks = df.sort_values('combined_score', ascending=False).head(top_n)
        else:
            top_stocks = df.head(top_n)
        
        # Reset index to get ticker as a column if it's in the index
        if top_stocks.index.name == 'ticker':
            top_stocks = top_stocks.reset_index()
        
        # Make sure we have a ticker column
        if 'ticker' not in top_stocks.columns:
            # If no ticker column, create one from the index
            top_stocks = top_stocks.reset_index().rename(columns={'index': 'ticker'})
        
        if interactive:
            # Create a Plotly heatmap
            fig = go.Figure()
            
            # Create a heatmap-like visualization with a scatter plot
            fig.add_trace(
                go.Scatter(
                    x=top_stocks['quality_score'],
                    y=top_stocks['momentum_score'],
                    mode='markers+text',
                    marker=dict(
                        size=12,
                        color=top_stocks['combined_score'] if 'combined_score' in top_stocks.columns else None,
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title='Combined Score') if 'combined_score' in top_stocks.columns else None
                    ),
                    text=top_stocks['ticker'],
                    textposition="top center",
                    name='Stocks'
                )
            )
            
            # Add reference lines
            fig.add_shape(
                type='line',
                x0=top_stocks['quality_score'].min(),
                y0=top_stocks['momentum_score'].mean(),
                x1=top_stocks['quality_score'].max(),
                y1=top_stocks['momentum_score'].mean(),
                line=dict(color='gray', width=1, dash='dash')
            )
            
            fig.add_shape(
                type='line',
                x0=top_stocks['quality_score'].mean(),
                y0=top_stocks['momentum_score'].min(),
                x1=top_stocks['quality_score'].mean(),
                y1=top_stocks['momentum_score'].max(),
                line=dict(color='gray', width=1, dash='dash')
            )
            
            # Add quadrant labels
            fig.add_annotation(
                x=top_stocks['quality_score'].min() + (top_stocks['quality_score'].max() - top_stocks['quality_score'].min()) * 0.25,
                y=top_stocks['momentum_score'].min() + (top_stocks['momentum_score'].max() - top_stocks['momentum_score'].min()) * 0.25,
                text="Low Quality<br>Low Momentum",
                showarrow=False,
                font=dict(size=10, color="gray")
            )
            
            fig.add_annotation(
                x=top_stocks['quality_score'].min() + (top_stocks['quality_score'].max() - top_stocks['quality_score'].min()) * 0.25,
                y=top_stocks['momentum_score'].min() + (top_stocks['momentum_score'].max() - top_stocks['momentum_score'].min()) * 0.75,
                text="Low Quality<br>High Momentum",
                showarrow=False,
                font=dict(size=10, color="gray")
            )
            
            fig.add_annotation(
                x=top_stocks['quality_score'].min() + (top_stocks['quality_score'].max() - top_stocks['quality_score'].min()) * 0.75,
                y=top_stocks['momentum_score'].min() + (top_stocks['momentum_score'].max() - top_stocks['momentum_score'].min()) * 0.25,
                text="High Quality<br>Low Momentum",
                showarrow=False,
                font=dict(size=10, color="gray")
            )
            
            fig.add_annotation(
                x=top_stocks['quality_score'].min() + (top_stocks['quality_score'].max() - top_stocks['quality_score'].min()) * 0.75,
                y=top_stocks['momentum_score'].min() + (top_stocks['momentum_score'].max() - top_stocks['momentum_score'].min()) * 0.75,
                text="High Quality<br>High Momentum",
                showarrow=False,
                font=dict(size=10, color="gray")
            )
            
            # Update layout
            fig.update_layout(
                title=title or f"Factor Scores Heatmap for Top {top_n} Stocks",
                xaxis_title="Quality Score",
                yaxis_title="Momentum Score",
                height=700,
                width=900,
                template='plotly_white'
            )
            
            # Save the figure if requested
            if save_as:
                save_path = os.path.join(self.output_dir, save_as)
                fig.write_html(save_path)
                print(f"Saved figure to {save_path}")
            
            return fig
        
        else:
            # Create a Matplotlib figure
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Create a scatter plot with heatmap-like coloring
            scatter = ax.scatter(
                top_stocks['quality_score'],
                top_stocks['momentum_score'],
                c=top_stocks['combined_score'] if 'combined_score' in top_stocks.columns else None,
                cmap='viridis',
                s=100,
                alpha=0.8
            )
            
            # Add ticker labels
            for i, ticker in enumerate(top_stocks['ticker']):
                ax.annotate(ticker, 
                           (top_stocks['quality_score'].iloc[i], top_stocks['momentum_score'].iloc[i]),
                           xytext=(5, 5), textcoords='offset points')
            
            # Add reference lines
            ax.axhline(y=top_stocks['momentum_score'].mean(), color='gray', linestyle='--', alpha=0.5)
            ax.axvline(x=top_stocks['quality_score'].mean(), color='gray', linestyle='--', alpha=0.5)
            
            # Add quadrant labels
            ax.text(top_stocks['quality_score'].min() + (top_stocks['quality_score'].max() - top_stocks['quality_score'].min()) * 0.25,
                   top_stocks['momentum_score'].min() + (top_stocks['momentum_score'].max() - top_stocks['momentum_score'].min()) * 0.25,
                   "Low Quality\nLow Momentum", ha='center', va='center', color='gray')
            
            ax.text(top_stocks['quality_score'].min() + (top_stocks['quality_score'].max() - top_stocks['quality_score'].min()) * 0.25,
                   top_stocks['momentum_score'].min() + (top_stocks['momentum_score'].max() - top_stocks['momentum_score'].min()) * 0.75,
                   "Low Quality\nHigh Momentum", ha='center', va='center', color='gray')
            
            ax.text(top_stocks['quality_score'].min() + (top_stocks['quality_score'].max() - top_stocks['quality_score'].min()) * 0.75,
                   top_stocks['momentum_score'].min() + (top_stocks['momentum_score'].max() - top_stocks['momentum_score'].min()) * 0.25,
                   "High Quality\nLow Momentum", ha='center', va='center', color='gray')
            
            ax.text(top_stocks['quality_score'].min() + (top_stocks['quality_score'].max() - top_stocks['quality_score'].min()) * 0.75,
                   top_stocks['momentum_score'].min() + (top_stocks['momentum_score'].max() - top_stocks['momentum_score'].min()) * 0.75,
                   "High Quality\nHigh Momentum", ha='center', va='center', color='gray')
            
            # Add colorbar if combined score is present
            if 'combined_score' in top_stocks.columns:
                cbar = plt.colorbar(scatter)
                cbar.set_label('Combined Score')
            
            # Add labels and title
            ax.set_xlabel('Quality Score')
            ax.set_ylabel('Momentum Score')
            ax.set_title(title or f"Factor Scores Heatmap for Top {top_n} Stocks")
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            # Tight layout
            plt.tight_layout()
            
            # Save the figure if requested
            if save_as:
                save_path = os.path.join(self.output_dir, save_as)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Saved figure to {save_path}")
            
            return fig
    
    def plot_performance_comparison(self, price_data, tickers, lookback_days=365, benchmark_ticker='SPY', 
                                    title=None, save_as=None, interactive=True):
        """
        Plot the performance comparison of selected stocks against a benchmark.
        
        Args:
            price_data (dict): Dictionary with price data for each ticker.
            tickers (list): List of tickers to plot.
            lookback_days (int, optional): Number of days to look back. Defaults to 365.
            benchmark_ticker (str, optional): Ticker for the benchmark. Defaults to 'SPY'.
            title (str, optional): Title for the plot. Defaults to None.
            save_as (str, optional): Filename to save the plot. Defaults to None.
            interactive (bool, optional): Whether to use Plotly (True) or Matplotlib (False). Defaults to True.
            
        Returns:
            Object: Figure object (either Plotly or Matplotlib)
        """
        # Check if price data is available
        if not price_data:
            print("No price data available")
            return None
        
        # Check if tickers exist in price data
        valid_tickers = [ticker for ticker in tickers if ticker in price_data]
        if not valid_tickers:
            print("None of the requested tickers found in price data")
            return None
        
        # Add benchmark ticker if available
        if benchmark_ticker not in valid_tickers and benchmark_ticker in price_data:
            valid_tickers.append(benchmark_ticker)
        
        # Calculate normalized returns
        return_dfs = []
        
        for ticker in valid_tickers:
            # Get price data for this ticker
            ticker_data = price_data[ticker]
            
            # Sort by date (ascending) and get only the last lookback_days days
            ticker_data = ticker_data.sort_index()
            if lookback_days:
                ticker_data = ticker_data.tail(lookback_days)
            
            # Calculate normalized returns (start at 100)
            if 'Adj Close' in ticker_data.columns:
                price_col = 'Adj Close'
            elif 'Close' in ticker_data.columns:
                price_col = 'Close'
            else:
                continue
            
            returns = ticker_data[price_col] / ticker_data[price_col].iloc[0] * 100
            returns = pd.DataFrame(returns)
            returns.columns = [ticker]
            
            return_dfs.append(returns)
        
        # Combine all returns into a single DataFrame
        if not return_dfs:
            print("No valid price data found for the requested tickers")
            return None
        
        combined_returns = pd.concat(return_dfs, axis=1).dropna()
        
        if interactive:
            # Create a Plotly figure
            fig = go.Figure()
            
            # Add a trace for each ticker
            for ticker in combined_returns.columns:
                # Use a distinct line for the benchmark
                if ticker == benchmark_ticker:
                    fig.add_trace(
                        go.Scatter(
                            x=combined_returns.index,
                            y=combined_returns[ticker],
                            mode='lines',
                            name=ticker,
                            line=dict(color='black', width=2, dash='dash')
                        )
                    )
                else:
                    fig.add_trace(
                        go.Scatter(
                            x=combined_returns.index,
                            y=combined_returns[ticker],
                            mode='lines',
                            name=ticker
                        )
                    )
            
            # Add horizontal line at 100 (starting point)
            fig.add_shape(
                type='line',
                x0=combined_returns.index.min(),
                y0=100,
                x1=combined_returns.index.max(),
                y1=100,
                line=dict(color='gray', width=1, dash='dot')
            )
            
            # Update layout
            fig.update_layout(
                title=title or f"Performance Comparison ({lookback_days} days)",
                xaxis_title="Date",
                yaxis_title="Normalized Return (Start = 100)",
                legend_title="Ticker",
                height=600,
                template='plotly_white'
            )
            
            # Add range slider
            fig.update_layout(
                xaxis=dict(
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1, label="1m", step="month", stepmode="backward"),
                            dict(count=3, label="3m", step="month", stepmode="backward"),
                            dict(count=6, label="6m", step="month", stepmode="backward"),
                            dict(count=1, label="YTD", step="year", stepmode="todate"),
                            dict(count=1, label="1y", step="year", stepmode="backward"),
                            dict(step="all")
                        ])
                    ),
                    rangeslider=dict(visible=True),
                    type="date"
                )
            )
            
            # Save the figure if requested
            if save_as:
                save_path = os.path.join(self.output_dir, save_as)
                fig.write_html(save_path)
                print(f"Saved figure to {save_path}")
            
            return fig
        
        else:
            # Create a Matplotlib figure
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot each ticker
            for ticker in combined_returns.columns:
                # Use a distinct line for the benchmark
                if ticker == benchmark_ticker:
                    ax.plot(combined_returns.index, combined_returns[ticker], 
                            'k--', linewidth=2, label=ticker)
                else:
                    ax.plot(combined_returns.index, combined_returns[ticker], label=ticker)
            
            # Add horizontal line at 100 (starting point)
            ax.axhline(y=100, color='gray', linestyle=':', alpha=0.7)
            
            # Add labels and title
            ax.set_xlabel('Date')
            ax.set_ylabel('Normalized Return (Start = 100)')
            ax.set_title(title or f"Performance Comparison ({lookback_days} days)")
            
            # Add legend
            ax.legend()
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            # Format date ticks
            plt.xticks(rotation=45)
            fig.autofmt_xdate()
            
            # Tight layout
            plt.tight_layout()
            
            # Save the figure if requested
            if save_as:
                save_path = os.path.join(self.output_dir, save_as)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Saved figure to {save_path}")
            
            return fig
    
    def plot_rolling_correlations(self, price_data, tickers, benchmark_ticker='SPY', window=90,
                                  title=None, save_as=None, interactive=True):
        """
        Plot rolling correlations of stocks with a benchmark.
        
        Args:
            price_data (dict): Dictionary with price data for each ticker.
            tickers (list): List of tickers to plot.
            benchmark_ticker (str, optional): Ticker for the benchmark. Defaults to 'SPY'.
            window (int, optional): Rolling window size in days. Defaults to 90.
            title (str, optional): Title for the plot. Defaults to None.
            save_as (str, optional): Filename to save the plot. Defaults to None.
            interactive (bool, optional): Whether to use Plotly (True) or Matplotlib (False). Defaults to True.
            
        Returns:
            Object: Figure object (either Plotly or Matplotlib)
        """
        # Check if price data is available
        if not price_data or benchmark_ticker not in price_data:
            print(f"No price data available for benchmark {benchmark_ticker}")
            return None
        
        # Check if tickers exist in price data
        valid_tickers = [ticker for ticker in tickers if ticker in price_data and ticker != benchmark_ticker]
        if not valid_tickers:
            print("None of the requested tickers found in price data")
            return None
        
        # Get benchmark returns
        benchmark_data = price_data[benchmark_ticker].sort_index()
        if 'Adj Close' in benchmark_data.columns:
            benchmark_returns = benchmark_data['Adj Close'].pct_change().dropna()
        elif 'Close' in benchmark_data.columns:
            benchmark_returns = benchmark_data['Close'].pct_change().dropna()
        else:
            print(f"No close price data found for benchmark {benchmark_ticker}")
            return None
        
        # Calculate rolling correlations
        correlations = {}
        
        for ticker in valid_tickers:
            # Get stock returns
            stock_data = price_data[ticker].sort_index()
            
            if 'Adj Close' in stock_data.columns:
                stock_returns = stock_data['Adj Close'].pct_change().dropna()
            elif 'Close' in stock_data.columns:
                stock_returns = stock_data['Close'].pct_change().dropna()
            else:
                print(f"No close price data found for {ticker}")
                continue
            
            # Align benchmark and stock returns
            aligned_data = pd.concat([benchmark_returns, stock_returns], axis=1).dropna()
            aligned_data.columns = ['benchmark', 'stock']
            
            # Calculate rolling correlation
            rolling_corr = aligned_data['benchmark'].rolling(window=window).corr(aligned_data['stock'])
            
            # Store in dictionary
            correlations[ticker] = rolling_corr
        
        # Combine all correlations into a single DataFrame
        if not correlations:
            print("No valid correlation data calculated")
            return None
        
        combined_corrs = pd.DataFrame(correlations)
        
        if interactive:
            # Create a Plotly figure
            fig = go.Figure()
            
            # Add a trace for each ticker
            for ticker in combined_corrs.columns:
                fig.add_trace(
                    go.Scatter(
                        x=combined_corrs.index,
                        y=combined_corrs[ticker],
                        mode='lines',
                        name=ticker
                    )
                )
            
            # Add horizontal line at 0 (no correlation)
            fig.add_shape(
                type='line',
                x0=combined_corrs.index.min(),
                y0=0,
                x1=combined_corrs.index.max(),
                y1=0,
                line=dict(color='gray', width=1, dash='dot')
            )
            
            # Add horizontal lines at -0.5 and 0.5 (moderate correlation boundaries)
            fig.add_shape(
                type='line',
                x0=combined_corrs.index.min(),
                y0=0.5,
                x1=combined_corrs.index.max(),
                y1=0.5,
                line=dict(color='gray', width=1, dash='dot')
            )
            
            fig.add_shape(
                type='line',
                x0=combined_corrs.index.min(),
                y0=-0.5,
                x1=combined_corrs.index.max(),
                y1=-0.5,
                line=dict(color='gray', width=1, dash='dot')
            )
            
            # Update layout
            fig.update_layout(
                title=title or f"Rolling {window}-Day Correlation with {benchmark_ticker}",
                xaxis_title="Date",
                yaxis_title=f"Correlation with {benchmark_ticker}",
                legend_title="Ticker",
                height=600,
                template='plotly_white',
                yaxis=dict(range=[-1, 1])  # Set y-axis range to [-1, 1]
            )
            
            # Add range slider
            fig.update_layout(
                xaxis=dict(
                    rangeselector=dict(
                        buttons=list([
                            dict(count=6, label="6m", step="month", stepmode="backward"),
                            dict(count=1, label="1y", step="year", stepmode="backward"),
                            dict(count=2, label="2y", step="year", stepmode="backward"),
                            dict(step="all")
                        ])
                    ),
                    rangeslider=dict(visible=True),
                    type="date"
                )
            )
            
            # Save the figure if requested
            if save_as:
                save_path = os.path.join(self.output_dir, save_as)
                fig.write_html(save_path)
                print(f"Saved figure to {save_path}")
            
            return fig
        
        else:
            # Create a Matplotlib figure
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot each ticker
            for ticker in combined_corrs.columns:
                ax.plot(combined_corrs.index, combined_corrs[ticker], label=ticker)
            
            # Add horizontal line at 0 (no correlation)
            ax.axhline(y=0, color='gray', linestyle=':', alpha=0.7)
            
            # Add horizontal lines at -0.5 and 0.5 (moderate correlation boundaries)
            ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.7)
            ax.axhline(y=-0.5, color='gray', linestyle=':', alpha=0.7)
            
            # Set y-axis limits
            ax.set_ylim([-1, 1])
            
            # Add labels and title
            ax.set_xlabel('Date')
            ax.set_ylabel(f'Correlation with {benchmark_ticker}')
            ax.set_title(title or f"Rolling {window}-Day Correlation with {benchmark_ticker}")
            
            # Add legend
            ax.legend()
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            # Format date ticks
            plt.xticks(rotation=45)
            fig.autofmt_xdate()
            
            # Tight layout
            plt.tight_layout()
            
            # Save the figure if requested
            if save_as:
                save_path = os.path.join(self.output_dir, save_as)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Saved figure to {save_path}")
            
            return fig
    
    def plot_metric_comparison(self, fundamental_data, tickers, metric_name, 
                               title=None, save_as=None, interactive=True):
        """
        Plot a comparison of a fundamental metric across multiple stocks.
        
        Args:
            fundamental_data (dict): Dictionary with fundamental data.
            tickers (list): List of tickers to plot.
            metric_name (str): Name of the metric to plot.
            title (str, optional): Title for the plot. Defaults to None.
            save_as (str, optional): Filename to save the plot. Defaults to None.
            interactive (bool, optional): Whether to use Plotly (True) or Matplotlib (False). Defaults to True.
            
        Returns:
            Object: Figure object (either Plotly or Matplotlib)
        """
        # Check if fundamental data is available
        if not fundamental_data or 'ratios' not in fundamental_data or metric_name not in fundamental_data['ratios'].columns:
            print(f"Metric {metric_name} not found in fundamental data")
            return None
        
        # Get the ratios data
        ratios_df = fundamental_data['ratios']
        
        # Check if tickers exist in ratios data
        valid_tickers = [ticker for ticker in tickers if ticker in ratios_df['ticker'].values]
        if not valid_tickers:
            print("None of the requested tickers found in fundamental data")
            return None
        
        # Filter for the requested tickers
        metric_data = ratios_df[ratios_df['ticker'].isin(valid_tickers)][['ticker', metric_name]].copy()
        
        # Check if we have data
        if metric_data.empty:
            print(f"No data available for {metric_name} for the requested tickers")
            return None
        
        # Sort by metric value (descending)
        metric_data = metric_data.sort_values(metric_name, ascending=False)
        
        if interactive:
            # Create a Plotly figure
            fig = go.Figure()
            
            # Add bar chart
            fig.add_trace(
                go.Bar(
                    x=metric_data['ticker'],
                    y=metric_data[metric_name],
                    text=metric_data[metric_name].round(2),
                    textposition='auto',
                    marker_color='#1f77b4'
                )
            )
            
            # Update layout
            fig.update_layout(
                title=title or f"{metric_name} Comparison",
                xaxis_title="Ticker",
                yaxis_title=metric_name,
                height=500,
                template='plotly_white'
            )
            
            # Save the figure if requested
            if save_as:
                save_path = os.path.join(self.output_dir, save_as)
                fig.write_html(save_path)
                print(f"Saved figure to {save_path}")
            
            return fig
        
        else:
            # Create a Matplotlib figure
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Bar chart of metric values
            bars = ax.bar(metric_data['ticker'], metric_data[metric_name], color='#1f77b4')
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}', ha='center', va='bottom')
            
            # Add labels and title
            ax.set_xlabel('Ticker')
            ax.set_ylabel(metric_name)
            ax.set_title(title or f"{metric_name} Comparison")
            
            # Add grid
            ax.grid(True, axis='y', alpha=0.3)
            
            # Rotate x-labels if many tickers
            if len(valid_tickers) > 8:
                plt.xticks(rotation=45)
            
            # Tight layout
            plt.tight_layout()
            
            # Save the figure if requested
            if save_as:
                save_path = os.path.join(self.output_dir, save_as)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Saved figure to {save_path}")
            
            return fig
    
    def plot_score_distributions(self, scores_df, title=None, save_as=None, interactive=True):
        """
        Plot the distributions of momentum, quality, and combined scores.
        
        Args:
            scores_df (pd.DataFrame): DataFrame with stock scores.
            title (str, optional): Title for the plot. Defaults to None.
            save_as (str, optional): Filename to save the plot. Defaults to None.
            interactive (bool, optional): Whether to use Plotly (True) or Matplotlib (False). Defaults to True.
            
        Returns:
            Object: Figure object (either Plotly or Matplotlib)
        """
        # Check if scores are available
        score_cols = [col for col in ['momentum_score', 'quality_score', 'combined_score'] 
                      if col in scores_df.columns]
        
        if not score_cols:
            print("No score columns found in the DataFrame")
            return None
        
        # Copy the DataFrame
        df = scores_df.copy()
        
        if interactive:
            # Create a Plotly figure with subplots
            fig = make_subplots(rows=len(score_cols), cols=1,
                               subplot_titles=[col.replace('_', ' ').title() + ' Distribution' 
                                              for col in score_cols])
            
            # Add histograms for each score type
            for i, col in enumerate(score_cols):
                fig.add_trace(
                    go.Histogram(
                        x=df[col],
                        name=col.replace('_', ' ').title(),
                        marker_color=['#1f77b4', '#ff7f0e', '#2ca02c'][i % 3],
                        nbinsx=20,
                        histnorm='probability'
                    ),
                    row=i+1, col=1
                )
                
                # Add vertical line for mean
                mean_val = df[col].mean()
                fig.add_shape(
                    type='line',
                    x0=mean_val,
                    y0=0,
                    x1=mean_val,
                    y1=0.5,  # Adjusted based on histnorm='probability'
                    line=dict(color='red', width=2, dash='dash'),
                    row=i+1, col=1
                )
                
                # Add annotation for mean
                fig.add_annotation(
                    x=mean_val,
                    y=0.45,  # Adjusted based on histnorm='probability'
                    text=f"Mean: {mean_val:.2f}",
                    showarrow=False,
                    font=dict(color='red'),
                    row=i+1, col=1
                )
            
            # Update layout
            fig.update_layout(
                title=title or "Score Distributions",
                height=300 * len(score_cols),
                template='plotly_white',
                showlegend=False
            )
            
            # Update y-axis title for each subplot
            for i in range(len(score_cols)):
                fig.update_yaxes(title_text="Probability", row=i+1, col=1)
            
            # Update x-axis title for the last subplot
            fig.update_xaxes(title_text="Score", row=len(score_cols), col=1)
            
            # Save the figure if requested
            if save_as:
                save_path = os.path.join(self.output_dir, save_as)
                fig.write_html(save_path)
                print(f"Saved figure to {save_path}")
            
            return fig
        
        else:
            # Create a Matplotlib figure
            fig, axes = plt.subplots(len(score_cols), 1, figsize=(12, 5 * len(score_cols)))
            
            # If only one score column, wrap the axes in a list
            if len(score_cols) == 1:
                axes = [axes]
            
            # Colors for different score types
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
            
            # Plot histograms for each score type
            for i, col in enumerate(score_cols):
                # Plot histogram
                axes[i].hist(df[col], bins=20, alpha=0.7, color=colors[i % 3], density=True)
                
                # Add KDE (smooth distribution curve)
                sns.kdeplot(df[col], ax=axes[i], color='k', linewidth=2)
                
                # Add vertical line for mean
                mean_val = df[col].mean()
                axes[i].axvline(x=mean_val, color='r', linestyle='--', linewidth=2)
                
                # Add text for mean
                axes[i].text(mean_val, axes[i].get_ylim()[1] * 0.9, 
                            f"Mean: {mean_val:.2f}", 
                            horizontalalignment='center', color='r')
                
                # Add labels
                axes[i].set_xlabel("Score" if i == len(score_cols) - 1 else "")
                axes[i].set_ylabel("Probability")
                axes[i].set_title(col.replace('_', ' ').title() + ' Distribution')
                
                # Add grid
                axes[i].grid(True, alpha=0.3)
            
            # Overall title
            plt.suptitle(title or "Score Distributions", fontsize=16)
            
            # Tight layout
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
            
            # Save the figure if requested
            if save_as:
                save_path = os.path.join(self.output_dir, save_as)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Saved figure to {save_path}")
            
            return fig
