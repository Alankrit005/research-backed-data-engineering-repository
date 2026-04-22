import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_project():
    """
    Complete Pipeline for Primetrade.ai Task: 
    Sentiment-Performance Correlation & Strategy Simulation.
    """
    # --- STEP 1: SETUP & LOADING ---
    print("Step 1: Loading Datasets...")
    fg_df = pd.read_csv('fear_greed_index.csv')
    hist_df = pd.read_csv('historical_data.csv')

    # --- STEP 2: PREPROCESSING ---
    print("Step 2: Cleaning and Merging Data...")
    fg_df['date'] = pd.to_datetime(fg_df['date']).dt.date
    # Normalize Historical Data timestamps to daily dates for merging
    hist_df['date'] = pd.to_datetime(hist_df['Timestamp IST'], format='%d-%m-%Y %H:%M', errors='coerce').dt.date
    hist_df = hist_df.dropna(subset=['date'])

    # Merging Sentiment into Trading Data
    merged_df = pd.merge(hist_df, fg_df[['date', 'value', 'classification']], on='date', how='inner')
    
    # --- STEP 3: EDA & METRICS ---
    print("Step 3: Calculating Performance Metrics...")
    merged_df['is_win'] = merged_df['Closed PnL'] > 0
    
    # Grouping by sentiment to see historical win rates
    sentiment_perf = merged_df.groupby('classification').agg(
        avg_pnl=('Closed PnL', 'mean'),
        win_rate=('is_win', 'mean'),
        trade_count=('Account', 'count')
    ).reindex(['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']).reset_index()

    # --- STEP 4: TRADER SEGMENTATION ---
    print("Step 4: Segmenting Traders...")
    trader_stats = merged_df.groupby('Account').agg(
        total_pnl=('Closed PnL', 'sum'),
        trade_count=('Closed PnL', 'count')
    )
    
    # Logic: Differentiate Professional high-volume traders from casual/at-risk traders
    def segment_trader(row):
        if row['total_pnl'] > 1000 and row['trade_count'] > 50:
            return 'Pro Profitable'
        elif row['total_pnl'] < 0:
            return 'At-Risk/Loss-Making'
        else:
            return 'Moderate/Casual'
            
    trader_stats['segment'] = trader_stats.apply(segment_trader, axis=1)
    merged_df = merged_df.merge(trader_stats[['segment']], on='Account', how='left')

    # --- STEP 5: SMART STRATEGY EXECUTION ---
    print("Step 5: Simulating 'Smart Sentiment' Strategy...")
    # STRATEGY RULES:
    # 1. In Extreme Greed: Focus on SELLING (Mean Reversion/Profit Taking)
    # 2. In Extreme Fear/Fear: Focus on BUYING (Bottom Fishing)
    # 3. In Neutral: Do not trade (Avoid "Chop")
    
    strategy_mask = (
        ((merged_df['classification'] == 'Extreme Greed') & (merged_df['Side'] == 'SELL')) |
        ((merged_df['classification'] == 'Extreme Fear') & (merged_df['Side'] == 'BUY')) |
        ((merged_df['classification'] == 'Fear') & (merged_df['Side'] == 'BUY'))
    )
    
    strategy_trades = merged_df[strategy_mask].copy()
    
    benchmark_pnl = merged_df['Closed PnL'].mean()
    strategy_pnl = strategy_trades['Closed PnL'].mean()

    # --- STEP 6: VISUALIZATION ---
    print("Step 6: Generating Plots...")
    sns.set(style="whitegrid")
    
    # Plot 1: Performance Comparison
    plt.figure(figsize=(8, 5))
    comparison = pd.DataFrame({
        'Method': ['Benchmark (Total)', 'Smart Sentiment Strategy'],
        'Avg PnL ($)': [benchmark_pnl, strategy_pnl]
    })
    sns.barplot(x='Method', y='Avg PnL ($)', data=comparison, palette='magma')
    plt.title('Strategy Performance Lift: Sentiment Filtering')
    plt.savefig('strategy_comparison.png')

    # --- STEP 7: OUTPUT ---
    print("\n--- RESULTS SUMMARY ---")
    print(f"Benchmark Avg PnL: ${benchmark_pnl:.2f}")
    print(f"Strategy Avg PnL: ${strategy_pnl:.2f}")
    print(f"Performance Improvement: {((strategy_pnl/benchmark_pnl)-1)*100:.2f}%")
    
    merged_df.to_csv('final_processed_data.csv', index=False)
    print("\nProject complete. Results saved to 'final_processed_data.csv'.")

if __name__ == "__main__":
    run_project()
    