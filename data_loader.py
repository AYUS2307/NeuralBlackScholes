import yfinance as yf
import pandas as pd
import requests
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import time

class OptionDataset(Dataset):
    def __init__(self, tickers, risk_free_rate=0.05, cache_file="options_data.csv"):
        self.data = []
        self.tickers = tickers
        self.r = risk_free_rate
        self.cache_file = cache_file
        self.fetch_data()
        
    def fetch_data(self):
        if os.path.exists(self.cache_file):
            print(f"Loading data from cache: {self.cache_file}")
            self.df = pd.read_csv(self.cache_file)
            print(f"Loaded {len(self.df)} rows from cache.")
            return

        print("Fetching option chains from API...")
        session = requests.Session()
        session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'})
        
        for i, ticker_symbol in enumerate(self.tickers):
            if i > 0:
                print("Waiting 2s to avoid rate limits...")
                time.sleep(2)
                
            print(f"Processing {ticker_symbol}...")
            tk = yf.Ticker(ticker_symbol, session=session)
            try:
                # get current price
                current_price = tk.history(period='1d')['Close'].iloc[-1]
                
                # get option expirations
                exps = tk.options
                if not exps:
                    continue
                    
                for date in exps[:2]: # Limit to first 2 expirations for speed/demonstration
                    try:
                        opt = tk.option_chain(date)
                        calls = opt.calls
                        
                        # Process Calls
                        for _, row in calls.iterrows():
                            strike = row['strike']
                            bid = row['bid']
                            ask = row['ask']
                            market_price = (bid + ask) / 2
                            if market_price <= 0: continue
                            
                            # Estimate Time to Maturity
                            mat_date = pd.to_datetime(date)
                            now = pd.Timestamp.now()
                            T = (mat_date - now).days / 365.0
                            if T <= 0: continue
                            
                            self.data.append({
                                'spot': current_price,
                                'strike': strike,
                                'time': T,
                                'rate': self.r,
                                'implied_vol': row['impliedVolatility'],
                                'market_price': market_price,
                                'type': 'call' 
                            })
                    except Exception as e:
                        print(f"Error processing expiration {date} for {ticker_symbol}: {e}")
                        
            except Exception as e:
                print(f"Error fetching data for {ticker_symbol}: {e}")
                
        self.df = pd.DataFrame(self.data)
        
        if len(self.df) > 0:
            # Clean Data
            self.df.dropna(inplace=True)
            if 'market_price' in self.df.columns:
                self.df = self.df[self.df['market_price'] > 0]
        
        if len(self.df) > 0:
            print(f"Saving {len(self.df)} rows to cache: {self.cache_file}")
            self.df.to_csv(self.cache_file, index=False)

        if len(self.df) == 0:
            print("Warning: No data fetched from yfinance. Generating synthetic data for testing.")
            # Synthetic Data (S, K, T, r, MarketPrice)
            # S=100, K=100..110, T=0.1..1.0
            for i in range(100):
                S = 100.0 * (1 + 0.2 * np.random.randn())
                K = 100.0 * (1 + 0.1 * np.random.randn())
                T = np.random.rand() * 1.0 + 0.1
                r = 0.05
                # BS Price
                # Simple approx or random
                market_price = max(S - K, 0) * 0.9 + 2.0 # dummy
                self.data.append({
                    'spot': S, 'strike': K, 'time': T, 'rate': r, 'market_price': market_price, 'type': 'call'
                })
            self.df = pd.DataFrame(self.data)
            print(f"Saving synthetic data to cache: {self.cache_file}")
            self.df.to_csv(self.cache_file, index=False)


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Inputs: Spot, Strike, Time
        # Targets: Market Price
        
        # Scale by 1000 for stability
        scale = 1000.0
        iv = row.get('implied_vol', 0.2)
        x = torch.tensor([row['spot']/scale, row['strike']/scale, row['time'], row['rate'], iv], dtype=torch.float32)
        y = torch.tensor([row['market_price']/scale], dtype=torch.float32)
        
        return x, y

def get_loader(tickers=['SPY'], batch_size=32, cache_file="options_data.csv"):
    ds = OptionDataset(tickers, cache_file=cache_file)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)

if __name__ == "__main__":
    loader = get_loader()
    for x, y in loader:
        print("Batch X:", x.shape)
        print("Batch Y:", y.shape)
        break
