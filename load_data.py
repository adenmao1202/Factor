""" this file is for loading min k data from local repo """

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import alphalens as al
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings 
warnings.filterwarnings('ignore')


base_path = '/Users/mouyasushi/k_data/永豐'  # /Users/mouyasushi/k_data/永豐
data_path = "/Users/mouyasushi/Desktop/Factor/alpha_lens/alphalens/alphalens/my_research/data"



def is_etf(symbol):
    """
    Check if a symbol is an ETF based on Taiwan stock market patterns
    
    Parameters:
    -----------
    symbol : str
        Stock symbol to check
    
    Returns:
    --------
    bool : True if it's an ETF, False otherwise
    """
    # Convert to string if not already
    symbol = str(symbol)
    
    # ETF patterns:
    # 1. Starts with '00' and 4-5 digits long (e.g., 0050, 00631L)
    # 2. Starts with '006' (e.g., 006201, 006208)
    return symbol.startswith('00') or symbol.startswith('006')

def get_all_symbols(base_path, exclude_etfs=True):
    """
    Get list of all stock symbols from the directory, optionally excluding ETFs
    
    Parameters:
    -----------
    base_path : str
        Path to the data directory
    exclude_etfs : bool
        Whether to exclude ETF symbols
    """
    files = os.listdir(base_path)
    symbols = [f.replace('.csv', '') for f in files if f.endswith('.csv')]
    
    if exclude_etfs:
        symbols = [s for s in symbols if not is_etf(s)]
    
    return sorted(symbols)

def read_and_process_all_stocks(base_path, min_price=0, exclude_etfs=True):
    """
    Read all stock data and create master price DataFrame
    
    Parameters:
    -----------
    base_path : str
        Path to the data directory
    min_price : float
        Minimum price filter
    exclude_etfs : bool
        Whether to exclude ETF symbols
    """
    symbols = get_all_symbols(base_path, exclude_etfs=exclude_etfs)
    
    # Print summary of symbols
    print(f"Total symbols found: {len(symbols)}")
    if exclude_etfs:
        all_symbols = get_all_symbols(base_path, exclude_etfs=False)
        etf_count = len(all_symbols) - len(symbols)
        print(f"ETFs excluded: {etf_count}")
        print(f"Stocks remaining: {len(symbols)}")
    
    # Initialize empty lists for each data type
    data_dict = {
        'open': [], 'high': [], 'low': [], 'close': [], 'volume': []
    }
    
    print("\nReading stock data...")
    for symbol in tqdm(symbols):
        try:
            # Read data
            df = pd.read_csv(os.path.join(base_path, f'{symbol}.csv'))
            df['ts'] = pd.to_datetime(df['ts'])
            df.set_index('ts', inplace=True)
            
            # Group by date
            df['date'] = df.index.date
            daily = df.groupby('date').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            })
            daily.index = pd.to_datetime(daily.index)
            daily.index.name = 'ts'  # Rename the index if needed

            # Apply minimum price filter and exclude specific dates
            excluded_dates = ['2024-10-15', '2024-10-16', '2024-10-17']  # List of dates to exclude
            excluded_dates = pd.to_datetime(excluded_dates)  # Convert to datetime format

            if daily['Close'].mean() >= min_price:
                # Exclude specified dates
                daily = daily[~daily.index.isin(excluded_dates)]
                
                # Store data with symbol as column name
                for key, col in zip(['open', 'high', 'low', 'close', 'volume'],
                                    ['Open', 'High', 'Low', 'Close', 'Volume']):
                    daily_series = daily[col].copy()
                    daily_series.name = symbol
                    data_dict[key].append(daily_series)
        
        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")
    
    # Concatenate all series for each data type
    master_data = {
        key: pd.concat(series_list, axis=1)
        for key, series_list in data_dict.items()
        if series_list  # Only if there's data
    }
    
    # Print final data shape
    if 'close' in master_data:
        print(f"\nFinal data shape: {master_data['close'].shape}")
        print(f"Date range: {master_data['close'].index[0]} to {master_data['close'].index[-1]}")
    
    return master_data


# Function to validate the filtering
def validate_symbols(base_path):
    """
    Validate symbol filtering by showing examples of included and excluded symbols
    """
    all_symbols = get_all_symbols(base_path, exclude_etfs=False)
    stock_symbols = get_all_symbols(base_path, exclude_etfs=True)
    etf_symbols = list(set(all_symbols) - set(stock_symbols))
    
    print("\nSymbol Validation:")
    print("-" * 50)
    print(f"Total symbols: {len(all_symbols)}")
    print(f"Stocks: {len(stock_symbols)}")
    print(f"ETFs: {len(etf_symbols)}")
    
    print("\nSample of included stocks:")
    print(sorted(stock_symbols)[:5])
    
    print("\nSample of excluded ETFs:")
    print(sorted(etf_symbols)[:5])


def prepare_alphalens_format(factor_df, price_df):
    """Convert factor and price data to Alphalens format with improved handling"""
    # Remove any remaining NaN or infinite values
    factor_df = factor_df.replace([np.inf, -np.inf], np.nan)
    price_df = price_df.replace([np.inf, -np.inf], np.nan)
    
    # Stack factor DataFrame to get a Series with MultiIndex
    factor_data = factor_df.stack()
    factor_data.index.names = ['date', 'asset']
    factor_data.name = 'factor'
    
    # Remove any remaining NaN values
    factor_data = factor_data.dropna()
    
    return price_df, factor_data


def analyze_factor(factor_df, master_data, periods=(1, 5, 10, 21)):
    """Analyze factor performance with improved methodology"""
    try:
        # Prepare data
        price_df = master_data['close'].copy()
        price_df, factor_data = prepare_alphalens_format(factor_df, price_df)
        
        # Clean data with less aggressive filtering
        factor_data = factor_data.set_index(['date', 'asset'])['factor']
        
        cleaned_data = al.utils.get_clean_factor_and_forward_returns(
            factor=factor_data,
            prices=price_df,
            periods=periods,
            filter_zscore=3,  # Less aggressive outlier filtering
            max_loss=0.25     # Less aggressive loss threshold
        )
        
        # Calculate metrics
        ic = al.performance.factor_information_coefficient(cleaned_data)
        mean_ic = ic.mean()
        
        # Calculate returns with proper handling
        returns = al.performance.factor_returns(cleaned_data)
        mean_returns = returns.mean()
        
        # Calculate turnover
        turnover = al.performance.factor_rank_autocorrelation(cleaned_data)
        
        return {
            'cleaned_data': cleaned_data,
            'ic': ic,
            'mean_ic': mean_ic,
            'returns': returns,
            'mean_returns': mean_returns,
            'turnover': turnover
        }
    except Exception as e:
        print(f"Error in factor analysis: {str(e)}")
        return None
    
    

def load_csv_to_dict(data_path):
    """
    Load multiple CSV files from a directory and combine into a dictionary
    Each CSV should be named like 'close.csv', 'open.csv', etc.
    """
    master_data = {}
    
    # Expected file names
    expected_files = ['master_close.csv', 'master_open.csv', 'master_high.csv', 'master_low.csv']
    
    for file_name in expected_files:
        # Create the full file path
        file_path = os.path.join(data_path, file_name)
        
        if os.path.exists(file_path):
            # Read CSV file
            # Assuming first column is the date index
            df = pd.read_csv(file_path)
            # Convert first column to datetime index
            df['ts'] = pd.to_datetime(df['ts'])
            df.set_index('ts', inplace=True)
            
            # Add to dictionary using file name without .csv as key
            key = file_name.replace('.csv', '')
            master_data[key] = df
        else:
            print(f"Warning: {file_name} not found in {data_path}")
    
    return master_data

