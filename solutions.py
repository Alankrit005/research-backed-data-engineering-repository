import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def run_analysis():
    fg_df = pd.read_csv('fear_greed_index.csv')
    hist_df = pd.read_csv('historical_data.csv')

    #Data Preprocessing
    #Conversion of date columns to datetime format for merging
    fg_df['date'] = pd.to_datetime(fg_df['date']).dt.date
    hist_df['date'] = pd.to_datetime(hist_df['Timestamp IST'], format='%d-%m-%Y %H:%M', errors='coerce').dt.date
    hist_df = hist_df.dropna(subset=['date'])

    #Merge the sentiment data into the historical trading data based on date
    merged_df = pd.merge(hist_df, fg_df[['date', 'value', 'classification']], on='date', how='inner')
    
    #Feature Engineering
    merged_df['is_win'] = merged_df['Closed PnL'] > 0
    
    #The'Smart Sentiment' Strategy logic
    #Rule: In Extreme Fear/Fear: Focus on BUYING (Mean Reversion/Value Investing)
    #Rule: In Extreme Greed: Focus on SELLING (Mean Reversion/Profit Taking)
    def strategy_signal(row):
        if row['classification'] in ['Extreme Fear', 'Fear'] and row['Side'] == 'BUY':
            return 'Strategy Trade'
        if row['classification'] in ['Extreme Greed'] and row['Side'] == 'SELL':
            return 'Strategy Trade'
        return 'Normal Trade'

    merged_df['Trade_Type'] = merged_df.apply(strategy_signal, axis=1)

    #Analysis of Strategy Performance
    benchmark_pnl = merged_df['Closed PnL'].mean()
    strategy_pnl = merged_df[merged_df['Trade_Type'] == 'Strategy Trade']['Closed PnL'].mean()

    #Visuaklization of PnL by Sentiment
    plt.figure(figsize=(10, 6))
    sns.barplot(x='classification', y='Closed PnL', data=merged_df, 
                order=['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed'], palette='RdYlGn')
    plt.title('Average PnL per Trade by Sentiment')
    plt.savefig('pnl_sentiment.png')
    print(f"Benchmark Avg PnL: ${benchmark_pnl:.2f}")
    print(f"Smart Strategy Avg PnL: ${strategy_pnl:.2f}")
    merged_df.to_csv('submission_results.csv', index=False)

if __name__ == "__main__":
    run_analysis()