import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import alphalens as al
import matplotlib.pyplot as plt
from tqdm import tqdm

def get_all_symbols(base_path):
    """Get list of all stock symbols from the directory"""
    files = os.listdir(base_path)
    symbols = [f.replace('.csv', '') for f in files if f.endswith('.csv')]
    return sorted(symbols)

def read_and_process_all_stocks(base_path, min_price=10):
    """
    Read all stock data and create master price DataFrame with improved performance
    """
    symbols = get_all_symbols(base_path)   # cal first func 
    
    # Init empty lists
    data_dict = {
        'open': [], 'high': [], 'low': [], 'close': [], 'volume': []
    }
    
    print("Reading stock data...")
    for symbol in tqdm(symbols):
        try:
            # Read data
            df = pd.read_csv(os.path.join(base_path, f'{symbol}.csv'))
            df['ts'] = pd.to_datetime(df['ts'])
            df.set_index('ts', inplace=True)
            
            # Resample to daily
            daily = df.resample('D').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
            
            # Apply minimum price filter
            if daily['Close'].mean() >= min_price:
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
    
    return master_data



# Testing Factor 
def calculate_factor(master_data, factor_name):
    """Calculate specified factor with improved methodology"""
    close = master_data['close'].copy()
    
    if factor_name == 'momentum_10d':
        # Improved momentum calculation with volatility adjustment
        returns = close.pct_change(10, fill_method=None)
        vol = returns.rolling(window=20).std()
        factor = (returns / vol).replace([np.inf, -np.inf], np.nan)
    
    elif factor_name == 'price_to_ma20':
        # Improved MA ratio with exponential moving average
        ema20 = close.ewm(span=20, adjust=False).mean()
        factor = (close / ema20 - 1).replace([np.inf, -np.inf], np.nan)
    
    elif factor_name == 'volume_zscore':
        # Improved volume factor with trend adjustment
        vol = master_data['volume'].copy()
        vol_ma = vol.ewm(span=20, adjust=False).mean()
        vol_std = vol.rolling(window=20).std()
        raw_factor = (vol - vol_ma) / vol_std
        
        # Add price trend filter
        price_trend = close.pct_change(5).rolling(window=5).mean()
        factor = raw_factor * np.sign(price_trend)
        factor = factor.replace([np.inf, -np.inf], np.nan)
    
    elif factor_name == 'rsi_14':
        # Improved RSI with smoothing
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
        rs = (avg_gain / avg_loss).replace([np.inf, -np.inf], np.nan)
        factor = 100 - (100 / (1 + rs))
        # Normalize RSI to [-1, 1]
        factor = (factor - 50) / 50
    
    elif factor_name == 'high_low_range':
        # Improved volatility factor with volume confirmation
        high = master_data['high']
        low = master_data['low']
        volume = master_data['volume']
        
        # Calculate normalized range
        price_range = (high - low) / close
        vol_ratio = volume / volume.rolling(window=20).mean()
        
        factor = price_range * np.sqrt(vol_ratio)
        factor = factor.replace([np.inf, -np.inf], np.nan)
    
    # Clean and normalize factor
    factor = factor.replace([np.inf, -np.inf], np.nan)
    
    # Neutralize by cross-sectional standardization
    for dt in factor.index:
        if not factor.loc[dt].isna().all():
            factor.loc[dt] = (factor.loc[dt] - factor.loc[dt].mean()) / factor.loc[dt].std()
    
    # Winsorize extreme values
    for col in factor.columns:
        series = factor[col].dropna()
        if len(series) > 0:
            lower = series.quantile(0.01)
            upper = series.quantile(0.99)
            factor.loc[:, col] = factor[col].clip(lower=lower, upper=upper)
    
    return factor


def prepare_alphalens_format(factor_df, price_df):
    """Convert factor and price data to Alphalens format with improved handling"""
    # Remove any remaining NaN or inf values
    factor_df = factor_df.replace([np.inf, -np.inf], np.nan)
    price_df = price_df.replace([np.inf, -np.inf], np.nan)
    
    # Stack factor DataFrame to get long format
    factor_data = factor_df.stack().reset_index()
    factor_data.columns = ['date', 'asset', 'factor']
    
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

def run_factor_research(base_path, factors_to_test=['momentum_10d']):
    """Run complete factor research pipeline with improved error handling"""
    # Step 1: Load all stock data
    print("Loading stock data...")
    master_data = read_and_process_all_stocks(base_path)
    
    # Step 2: Calculate and analyze each factor
    results = {}
    for factor_name in factors_to_test:
        print(f"\nAnalyzing factor: {factor_name}")
        
        try:
            # Calculate factor
            factor_df = calculate_factor(master_data, factor_name)
            
            # Analyze factor
            result = analyze_factor(factor_df, master_data)
            
            if result is not None:
                results[factor_name] = result
                
                # Print summary statistics
                print(f"\nInformation Coefficient Summary for {factor_name}:")
                print(result['mean_ic'])
                
                if len(result['returns']) > 0:
                    print(f"\nMean Factor Returns:")
                    print(result['returns'].mean())
        
        except Exception as e:
            print(f"Error analyzing {factor_name}: {str(e)}")
            continue
    
    return results, master_data


############################################################################
def main():
    """
    Main execution function for factor research
    """
    # Set base path
    base_path = '/Users/mouyasushi/k_data/永豐'
    
    # Define factors to test
    factors_to_test = [
        'momentum_10d',
        'price_to_ma20',
        'volume_zscore',
        'rsi_14',
        'high_low_range'
    ]
    
    try:
        # Run initial test with just one factor
        print("\nRunning initial test with momentum_10d factor...")
        test_results, _ = run_factor_research(base_path, factors_to_test=['momentum_10d'])
        
        if test_results:
            print("\nInitial test successful, proceeding with all factors...")
            
            # Run analysis for all factors
            results, master_data = run_factor_research(base_path, factors_to_test=factors_to_test)
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create results directory if it doesn't exist
            results_dir = 'factor_results'
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
            
            # Save summary statistics
            summary_stats = {}
            for factor_name, factor_results in results.items():
                summary_stats[factor_name] = {
                    'mean_ic': factor_results['mean_ic'].to_dict(),
                    'mean_returns': factor_results['returns'].mean().to_dict() if len(factor_results['returns']) > 0 else {}
                }
            
            # Convert to DataFrame and save
            summary_df = pd.DataFrame(summary_stats).round(4)
            summary_df.to_csv(f'{results_dir}/factor_summary_{timestamp}.csv')
            
            print("\nResults have been saved to 'factor_results' directory")
            print("\nSummary of results:")
            print(summary_df)
            
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()