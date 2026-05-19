import yfinance as yf
import pandas as pd

def check_nse_options():
    ticker = "^NSEI" # Nifty 50
    print(f"Checking option chain for {ticker}...")
    
    try:
        t = yf.Ticker(ticker)
        expirations = t.options
        if not expirations:
            print("No expirations found. yfinance might not support NSE options directly.")
            return

        print(f"Expirations found: {expirations[:3]}")
        
        # Fetch first chain
        chain = t.option_chain(expirations[0])
        print(f"Calls: {len(chain.calls)}")
        print(chain.calls.head())
        
        print("Success! NSE options are available.")
        
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    check_nse_options()
