#!/usr/bin/env python
"""
Basic example of how to use the Evidence-Based Stock Screener.
This script demonstrates how to screen for high-quality stocks with momentum.
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

# Add the root directory to the path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the stock screener
from src.screener.stock_screener import StockScreener

def main():
    """
    Main function to run the stock screener example.
    """
    print("Evidence-Based Stock Screener - Basic Example")
    print("=============================================")
    
    # Initialize the screener
    print("\nInitializing stock screener...")
    screener = StockScreener()
    
    # Choose the index to screen
    index_name = choose_index()
    
    # Load data for the index
    print(f"\nLoading data for {index_name}...")
    success = screener.load_index_data(index_name)
    
    if not success:
        print(f"Failed to load data for {index_name}. Exiting.")
        return
    
    print(f"Successfully loaded data for {index_name}.")
    
    # Calculate factor scores
    print("\nCalculating factor scores...")
    success = screener.calculate_scores()
    
    if not success:
        print("Failed to calculate scores. Exiting.")
        return
    
    print("Successfully calculated factor scores.")
    
    # Get top stocks
    n_stocks = int(input("\nHow many top stocks would you like to see? (default: 20) ") or 20)
    factor = choose_sort_factor()
    
    print(f"\nGetting top {n_stocks} stocks sorted by {factor}...")
    top_stocks = screener.get_top_stocks(n=n_stocks, sort_by=factor)
    
    if top_stocks is None or top_stocks.empty:
        print("No stocks found. Exiting.")
        return
    
    # Display top stocks
    display_top_stocks(top_stocks)
    
    # Export results
    export_choice = input("\nWould you like to export the results? (y/n, default: n) ").lower() or 'n'
    
    if export_choice == 'y':
        export_results(screener, top_stocks)
    
    # Compare performance
    compare_choice = input("\nWould you like to compare performance of top 5 stocks with the index? (y/n, default: y) ").lower() or 'y'
    
    if compare_choice == 'y':
        compare_performance(screener, top_stocks, index_name)
    
    print("\nScreening complete!")

def choose_index():
    """
    Let the user choose an index to screen.
    
    Returns:
        str: Name of the chosen index
    """
    indices = {
        '1': 'SP500',
        '2': 'NASDAQ100',
        '3': 'EUROSTOXX50',
        '4': 'CAC40'
    }
    
    print("\nAvailable indices:")
    for key, index in indices.items():
        print(f"{key}. {index}")
    
    choice = input("Choose an index (1-4, default: 1): ") or '1'
    
    return indices.get(choice, 'SP500')

def choose_sort_factor():
    """
    Let the user choose a factor to sort the results by.
    
    Returns:
        str: Name of the chosen factor
    """
    factors = {
        '1': 'combined_score',
        '2': 'momentum_score',
        '3': 'quality_score'
    }
    
    print("\nSort by:")
    for key, factor in factors.items():
        print(f"{key}. {factor}")
    
    choice = input("Choose a factor (1-3, default: 1): ") or '1'
    
    return factors.get(choice, 'combined_score')

def display_top_stocks(top_stocks):
    """
    Display the top stocks in a formatted table.
    
    Args:
        top_stocks (pd.DataFrame): DataFrame with the top stocks
    """
    print("\nTop Stocks:")
    print("=" * 80)
    
    # Format the DataFrame for display
    display_df = top_stocks.copy()
    
    # Rename columns for better display
    rename_map = {
        'combined_score': 'Combined',
        'momentum_score': 'Momentum',
        'quality_score': 'Quality',
        'technical_momentum_score': 'Tech Momentum',
        'fundamental_momentum_score': 'Fund Momentum',
        'profitability_score': 'Profitability',
        'balance_sheet_score': 'Balance Sheet',
        'earnings_stability_score': 'Earnings Stability'
    }
    
    display_df.rename(columns=rename_map, inplace=True)
    
    # Round scores to 2 decimal places
    for col in display_df.columns:
        if 'score' in col.lower() or col in rename_map.values():
            display_df[col] = display_df[col].round(2)
    
    # Display the DataFrame
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 120)
    print(display_df)

def export_results(screener, top_stocks):
    """
    Export the results to a file.
    
    Args:
        screener (StockScreener): The stock screener instance
        top_stocks (pd.DataFrame): DataFrame with the top stocks
    """
    export_format = input("Export format (1 for Excel, 2 for CSV, default: 1): ") or '1'
    
    if export_format == '1':
        file_path = input("Enter file path (default: 'results.xlsx'): ") or 'results.xlsx'
        if not file_path.endswith('.xlsx'):
            file_path += '.xlsx'
    else:
        file_path = input("Enter file path (default: 'results.csv'): ") or 'results.csv'
        if not file_path.endswith('.csv'):
            file_path += '.csv'
    
    success = screener.export_results(file_path)
    
    if success:
        print(f"Results exported to {file_path}")
    else:
        print(f"Failed to export results to {file_path}")

def compare_performance(screener, top_stocks, index_name):
    """
    Compare the performance of the top 5 stocks with the index.
    
    Args:
        screener (StockScreener): The stock screener instance
        top_stocks (pd.DataFrame): DataFrame with the top stocks
        index_name (str): Name of the index
    """
    # Get the top 5 tickers
    top_5_tickers = top_stocks.index[:5].tolist()
    
    # Compare performance
    performance = screener.compare_performance(top_5_tickers, index_name)
    
    if performance is None or performance.empty:
        print("Failed to compare performance.")
        return
    
    # Plot performance
    plt.figure(figsize=(12, 8))
    for col in performance.columns:
        plt.plot(performance.index, performance[col], label=col)
    
    plt.title(f"Normalized Performance Comparison (100 = Starting Value)")
    plt.xlabel("Date")
    plt.ylabel("Performance (%)")
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plot_file = 'performance_comparison.png'
    plt.savefig(plot_file)
    plt.close()
    
    print(f"Performance comparison chart saved to {plot_file}")

if __name__ == "__main__":
    main()
