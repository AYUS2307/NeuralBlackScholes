import torch
import numpy as np
from inference import InferenceEngine
from physics_engine import BlackScholesPricer
import scipy.stats as si

class HedgingEngine:
    def __init__(self, run_name="default"):
        print("Initializing Neural Hedging Engine...")
        self.engine = InferenceEngine(run_name=run_name)
        self.bs_pricer = BlackScholesPricer()
        self.device = self.engine.device
        
    def _bs_greeks(self, S, K, T, r, sigma, option_type='call'):
        """Calculate Black-Scholes Delta and Vega theoretically."""
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        if option_type == 'call':
            delta = si.norm.cdf(d1, 0.0, 1.0)
        else:
            delta = si.norm.cdf(d1, 0.0, 1.0) - 1
            
        vega = S * si.norm.pdf(d1, 0.0, 1.0) * np.sqrt(T)
        return delta, vega

    def compute_neural_greeks(self, S, K, T, iv=0.2, r=0.05):
        """
        Calculates Neural Delta and Neural Vega using PyTorch Automatic Differentiation.
        This forces the Neural Network to expose its internal sensitivity to Stock and Panic.
        """
        # Create tensors that require gradients
        spot_tensor = torch.tensor(S, dtype=torch.float32, device=self.device, requires_grad=True)
        iv_tensor = torch.tensor(iv, dtype=torch.float32, device=self.device, requires_grad=True)
        
        # 1. Forward Pass (Price the option but KEEP the computation graph)
        # Note: We must use a large number of paths for stable gradients, e.g., 500
        price_tensor = self.engine.predict(
            spot=spot_tensor, 
            strike=K, 
            time_to_mat=T, 
            iv=iv_tensor, 
            rate=r, 
            num_paths=500, 
            calc_greeks=True
        )
        
        # 2. Backward Pass (Autograd calculates exact derivatives)
        # We take the derivative of Price with respect to Spot (Delta) and IV (Vega)
        price_tensor.backward()
        
        neural_delta = spot_tensor.grad.item()
        neural_vega = iv_tensor.grad.item()
        
        return price_tensor.item(), neural_delta, neural_vega

    def generate_strategy_report(self, S, K, T, iv, r, position_size=100):
        print(f"\n=======================================================")
        print(f" NEURAL STOCHASTIC VOLATILITY (NSV) HEDGING STRATEGY ")
        print(f"=======================================================")
        print(f"Market State: Spot=${S}, Strike=${K}, Time={T}yr, Vol={iv*100}%")
        print(f"Position: Short {position_size} Call Options")
        print(f"-------------------------------------------------------")
        
        # Black-Scholes calculation
        bs_delta, bs_vega = self._bs_greeks(S, K, T, r, iv)
        
        # Neural calculation
        neural_price, neural_delta, neural_vega = self.compute_neural_greeks(S, K, T, iv, r)
        
        print(f"1. Delta (Stock Hedge)")
        print(f"   Black-Scholes Delta: {bs_delta:.4f}")
        print(f"   Neural NSV Delta:    {neural_delta:.4f}")
        print(f"   -> Difference:       {abs(bs_delta - neural_delta):.4f}")
        print(f"\n2. Vega (Volatility Hedge)")
        print(f"   Black-Scholes Vega:  {bs_vega:.4f}")
        print(f"   Neural NSV Vega:     {neural_vega:.4f}")
        
        print(f"\n-------------------------------------------------------")
        print(f" ACTIONABLE STRATEGY (To perfectly hedge the portfolio):")
        
        shares_to_buy_bs = int(bs_delta * position_size)
        shares_to_buy_nsv = int(neural_delta * position_size)
        
        print(f"If you use Black-Scholes: BUY {shares_to_buy_bs} shares of Stock.")
        print(f"If you use Neural NSV:    BUY {shares_to_buy_nsv} shares of Stock.")
        print(f"-------------------------------------------------------")
        if shares_to_buy_bs != shares_to_buy_nsv:
            print(f"ANALYSIS: The Neural NSV model detects the 'Leverage Effect'.")
            print(f"Because a stock crash will trigger a volatility spike, the NSV model")
            print(f"adjusts the Delta to protect you against both forces simultaneously.")
            print(f"Black-Scholes is blind to this and will under/over-hedge your book.")
        print(f"=======================================================\n")

if __name__ == "__main__":
    import argparse
    import pandas as pd
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default='default')
    args = parser.parse_args()
    
    hedger = HedgingEngine(run_name=args.run_name)
    
    csv_file = f"options_data_{args.run_name}.csv" if args.run_name != "default" else "options_data.csv"
    try:
        df = pd.read_csv(csv_file)
        # Pick a real option from the dataset (e.g. middle of the pack)
        row = df.iloc[len(df) // 2]
        
        S = row['spot']
        K = row['strike']
        T = row['time']
        iv = row['implied_vol'] if 'implied_vol' in row else 0.2
        r = row['rate']
        
        print(f"Loaded Real Option Data: Spot=${S:.2f}, Strike=${K:.2f}, Time={T:.3f}y, IV={iv:.3f}")
        hedger.generate_strategy_report(S=S, K=K, T=T, iv=iv, r=r, position_size=1000)
    except Exception as e:
        print(f"Could not load real data from {csv_file}, using defaults. Error: {e}")
        hedger.generate_strategy_report(S=400.0, K=400.0, T=0.5, iv=0.20, r=0.05, position_size=1000)
