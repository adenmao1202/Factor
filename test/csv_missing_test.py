import pandas as pd
import numpy as np
import os
from datetime import datetime
from tqdm import tqdm

def analyze_csv_files(base_path):
    """
    Comprehensive analysis of all CSV files with missing value summary
    """
    print("\nAnalyzing CSV files in:", base_path)
    print("-" * 50)
    
    # Get all CSV files
    files = [f for f in os.listdir(base_path) if f.endswith('.csv')]
    total_files = len(files)
    print(f"\nFound {total_files} CSV files")
    
    # Initialize storage for summary statistics
    summary_stats = {
        'file_stats': {},
        'date_ranges': {},
        'missing_patterns': {}
    }
    
    # Process each file
    print("\nProcessing files...")
    for file in tqdm(sorted(files)):
        file_path = os.path.join(base_path, file)
        symbol = file.replace('.csv', '')
        
        try:
            # Read file
            df = pd.read_csv(file_path)
            
            # Convert ts to datetime
            if 'ts' in df.columns:
                df['ts'] = pd.to_datetime(df['ts'])
            
            # Store basic file statistics
            summary_stats['file_stats'][symbol] = {
                'total_rows': len(df),
                'missing_values': df.isna().sum().to_dict(),
                'total_missing': df.isna().sum().sum()
            }
            
            # Store date range
            if 'ts' in df.columns:
                summary_stats['date_ranges'][symbol] = {
                    'start': df['ts'].min(),
                    'end': df['ts'].max(),
                    'unique_dates': df['ts'].nunique()
                }
            
            # Analyze missing patterns
            for col in df.columns:
                if df[col].isna().any():
                    missing_dates = df[df[col].isna()]['ts'].tolist() if 'ts' in df.columns else []
                    if col not in summary_stats['missing_patterns']:
                        summary_stats['missing_patterns'][col] = {}
                    summary_stats['missing_patterns'][col][symbol] = {
                        'count': df[col].isna().sum(),
                        'dates': missing_dates[:5]  # Store first 5 missing dates as example
                    }
        
        except Exception as e:
            print(f"\nError processing {file}: {str(e)}")
    
    return summary_stats

def print_summary_report(summary_stats):
    """
    Print comprehensive summary report
    """
    print("\nData Quality Summary Report")
    print("=" * 50)
    
    # Overall statistics
    total_missing = sum(stats['total_missing'] for stats in summary_stats['file_stats'].values())
    total_rows = sum(stats['total_rows'] for stats in summary_stats['file_stats'].values())
    
    print(f"\n1. Overall Statistics:")
    print(f"   Total files processed: {len(summary_stats['file_stats'])}")
    print(f"   Total rows across all files: {total_rows:,}")
    print(f"   Total missing values: {total_missing:,}")
    print(f"   Overall completeness: {((1 - total_missing/total_rows) * 100):.2f}%")
    
    # Files with most missing values
    print("\n2. Top 10 Files with Most Missing Values:")
    missing_by_file = {
        symbol: stats['total_missing'] 
        for symbol, stats in summary_stats['file_stats'].items()
    }
    for symbol in sorted(missing_by_file, key=missing_by_file.get, reverse=True)[:10]:
        print(f"   {symbol}: {missing_by_file[symbol]:,} missing values "
              f"({(missing_by_file[symbol]/summary_stats['file_stats'][symbol]['total_rows']*100):.2f}%)")
    
    # Missing values by column
    print("\n3. Missing Values by Column:")
    column_missing = {}
    for symbol, stats in summary_stats['file_stats'].items():
        for col, count in stats['missing_values'].items():
            if col not in column_missing:
                column_missing[col] = 0
            column_missing[col] += count
    
    for col in sorted(column_missing, key=column_missing.get, reverse=True):
        print(f"   {col}: {column_missing[col]:,} missing values")
    
    # Date range analysis
    if summary_stats['date_ranges']:
        print("\n4. Date Range Analysis:")
        all_starts = [info['start'] for info in summary_stats['date_ranges'].values()]
        all_ends = [info['end'] for info in summary_stats['date_ranges'].values()]
        print(f"   Overall date range: {min(all_starts)} to {max(all_ends)}")
        
        # Files with irregular date ranges
        print("\n5. Files with Irregular Date Ranges:")
        expected_dates = max([info['unique_dates'] for info in summary_stats['date_ranges'].values()])
        irregular_files = {
            symbol: info['unique_dates']
            for symbol, info in summary_stats['date_ranges'].items()
            if info['unique_dates'] < expected_dates
        }
        for symbol in sorted(irregular_files, key=irregular_files.get)[:5]:
            print(f"   {symbol}: {irregular_files[symbol]} dates (expected: {expected_dates})")
    
    # Sample of missing patterns
    print("\n6. Sample Missing Patterns:")
    for col, patterns in list(summary_stats['missing_patterns'].items())[:3]:
        print(f"\n   Column: {col}")
        for symbol, info in list(patterns.items())[:3]:
            print(f"   {symbol}: {info['count']} missing values")
            if info['dates']:
                print(f"   First few missing dates: {info['dates'][:3]}")

# Run the analysis
base_path = '/Users/mouyasushi/k_data/永豐'
summary_stats = analyze_csv_files(base_path)
print_summary_report(summary_stats)