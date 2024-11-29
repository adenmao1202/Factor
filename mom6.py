
""" 
- Formula: MOM_6 = ∏(1 + daily_returns_i) - 1 from i=t-126 to t-1
 
- MOM 6 : 動能  

以短期報酬作為因子來選擇股票會存在異常報酬，該異常報酬的來源有兩種解釋：

1. 第一種解釋為當短期內大量現金流入會導致該股票短期內的價格上升，此效果稱為短期價格壓力效果 

2. 第二種是市場流動性不足夠導致股價對共同因素

Note : 
1. t 為日
2. rebalance : 126 天 = 6 month 
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import alphalens as al
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings 
warnings.filterwarnings('ignore')


#######################################################################
# Start 

def calculate_mom6(master_data):
    """

    Note:
    - t is daily
    - Rebalance period: 126 days = 6 months
    - Uses close prices
    - Accounts for market pressure effect from large capital inflows
    - Considers market liquidity impact
    """
    # Get close prices
    close = master_data['close'].copy()
    
    # Calculate daily returns
    daily_returns = close.pct_change(fill_method=None)
    
    # Calculate 6-month momentum using rolling window
    def mom6_calculation(window):
        """Calculate momentum for a given window of returns"""
        # Check if we have enough valid data 
        if len(window.dropna()) < 126 :
            return np.nan
        # Calculate cumulative return using product of (1 + daily_returns)
        cum_return = (1 + window).prod() - 1
        return cum_return if np.isfinite(cum_return) else np.nan
    
    # Apply rolling calculation
    factor = daily_returns.rolling(
        window=126,  # 6 months
        min_periods=int(126)  # Require at least 80% of data
    ).apply(mom6_calculation)
    
    # Shift by 1 to avoid look-ahead bias
    factor = factor.shift(1)
    
    # Clean and standardize factor
    # 1. Remove infinite values
    factor = factor.replace([np.inf, -np.inf], np.nan)
    
    # 2. Cross-sectional standardization
    for dt in factor.index:
        valid_data = factor.loc[dt].dropna()
        if len(valid_data) >= 10:  # Need at least 10 stocks for meaningful standardization
            mean = valid_data.mean()
            std = valid_data.std()
            if std > 0:
                factor.loc[dt] = (factor.loc[dt] - mean) / std
    
    # 3. Winsorize extreme values at 1% and 99%
    valid_data = factor.stack().dropna()
    if len(valid_data) > 0:
        lower = valid_data.quantile(0.01)
        upper = valid_data.quantile(0.99)
        factor = factor.clip(lower=lower, upper=upper)
    
    return factor


# Example usage:
def test_mom6_factor(master_data):
    """Test the MOM6 factor calculation"""
    # Calculate factor
    mom6 = calculate_mom6(master_data)
    
    # Print summary statistics
    print("\nMOM6 Factor Summary:")
    print(f"Total values: {mom6.size}")
    print(f"NaN values: {mom6.isna().sum().sum()}")
    print(f"Non-NaN values: {mom6.notna().sum().sum()}")
    
    # Show sample of results for first 5 stocks
    print("\nSample factor values (first 5 stocks):")
    print(mom6.iloc[126:131, :5])
    
    # Show factor statistics
    valid_data = mom6.stack().dropna()
    print("\nFactor Statistics:")
    print(valid_data.describe())
    
    return mom6


