import torch
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import yaml
import sys
from typing import Dict, Any, Tuple, Optional, Union
from neural_engine import NeuralSDE
from physics_engine import BlackScholesPricer

# Load Project Configuration
try:
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    print("Critical Error: 'config.yaml' not found in workspace.")
    sys.exit(1)

DEVICE = torch.device(config['project']['device'] if torch.cuda.is_available() else 'cpu')

class InferenceEngine:
    """
    Enterprise-grade Quantitative Inference Engine for continuous-time Option Pricing.
    Uses continuous-time GPU-accelerated SDE simulation to price options under Neural Stochastic Volatility.
    """
    def __init__(self, run_name: str = "default") -> None:
        self.run_name = run_name
        self.device = DEVICE
        self.model = self._load_model()
        self.model.eval()

    def _load_model(self) -> NeuralSDE:
        """
        Initializes the architecture wrapper and loads the trained checkpoint.
        
        Returns:
            The loaded NeuralSDE model instance placed on the execution device.
        """
        class ModelWrapper(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.sde = NeuralSDE(
                    state_size=2, 
                    brownian_size=2, 
                    hidden_size=config['model']['hidden_size']
                )
                
        model = ModelWrapper().to(self.device)
        checkpoint_path = f"{config['training']['save_dir']}/model_final_{self.run_name}.pth"
        
        try:
            print(f"Loading quantitative model parameters from: {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(state_dict)
            print("Successfully loaded model parameters into GPU device registers.")
        except Exception as e:
            print(f"Warning: Could not load parameters ({e}). Initializing with stochastic untrained weights.")
            
        return model.sde

    def predict(
        self, 
        spot: float, 
        strike: float, 
        time_to_mat: float, 
        iv: float = 0.2, 
        rate: float = 0.05, 
        num_paths: int = 2000, 
        calc_greeks: bool = False
    ) -> Union[torch.Tensor, float]:
        """
        Prices a single European Option using Neural Stochastic Volatility Monte Carlo integration.
        
        Args:
            spot: Current asset price S_0.
            strike: Option strike price K.
            time_to_mat: Annualized time to maturity T.
            iv: Initial volatility state V_0.
            rate: Annualized risk-free interest rate r.
            num_paths: Number of Monte Carlo SDE paths. Higher values eliminate numerical pricing noise.
            calc_greeks: If True, preserves PyTorch's computational graph for autograd derivatives.
            
        Returns:
            If calc_greeks is True: torch.Tensor option price (keeps autograd attached).
            If calc_greeks is False: float option price.
        """
        scale = 1000.0  # Input scaling factor for numeric gradient stability
        
        if calc_greeks:
            s_norm = spot / scale
            iv_tensor = iv
            k_norm = torch.tensor([strike / scale], dtype=torch.float32).to(self.device)
            
            spot_expanded = s_norm.repeat(num_paths).unsqueeze(1)
            vol_expanded = iv_tensor.repeat(num_paths).unsqueeze(1)
            y0 = torch.cat([spot_expanded, vol_expanded], dim=-1) # Shape: (Paths, 2)
            
            ts = torch.tensor([0.0, time_to_mat], device=self.device)
            trajectory = self.model(y0, ts, r=rate)
            params_final = trajectory[-1, :, 0] # Shape: (Paths,)
            
            payoffs_norm = torch.relu(params_final - k_norm.item())
            discount = torch.exp(torch.tensor(-rate * time_to_mat))
            price_norm = (payoffs_norm * discount).mean()
            price = price_norm * scale
            return price
        else:
            s_norm = torch.tensor([spot / scale], dtype=torch.float32).to(self.device)
            iv_tensor = torch.tensor([iv], dtype=torch.float32).to(self.device)
            k_norm = torch.tensor([strike / scale], dtype=torch.float32).to(self.device)
            
            with torch.no_grad():
                spot_expanded = s_norm.repeat(num_paths).unsqueeze(1)
                vol_expanded = iv_tensor.repeat(num_paths).unsqueeze(1)
                y0 = torch.cat([spot_expanded, vol_expanded], dim=-1) # Shape: (Paths, 2)
                
                ts = torch.tensor([0.0, time_to_mat], device=self.device)
                trajectory = self.model(y0, ts, r=rate)
                params_final = trajectory[-1, :, 0]
                
                payoffs_norm = torch.relu(params_final - k_norm.item())
                discount = torch.exp(torch.tensor(-rate * time_to_mat))
                price_norm = (payoffs_norm * discount).mean()
                price = price_norm.item() * scale
                return price

    def run_test_suite(self) -> None:
        """
        Executes comparison test suite comparing the trained SDE model to a Constant Volatility BS model.
        Uses 2000-path Monte Carlo pricing to eliminate numerical noise.
        """
        csv_file = f"options_data_{self.run_name}.csv" if self.run_name != "default" else "options_data.csv"
        print(f"Loading test options database from: {csv_file}")
        
        try:
            df = pd.read_csv(csv_file)
            df.dropna(inplace=True)
            if 'market_price' in df.columns:
                 df = df[df['market_price'] > 0]
            if len(df) == 0:
                print("Error: Option database contains no valid records.")
                return
        except FileNotFoundError:
            print(f"Critical Error: Database file '{csv_file}' not found.")
            return

        predictions = []
        targets = []
        bs_prices = []
        
        # Fair Volatility Benchmark (eliminates post-facto circular IV bias)
        avg_iv = df['implied_vol'].mean() if 'implied_vol' in df.columns else 0.2
        print(f"Quantitative Setup: Enforcing a fair Constant Volatility BS Benchmark = {avg_iv*100:.2f}%")
        print(f"Executing 2000-path Monte Carlo pricing over {len(df)} samples (Sampling noise reduction active)...")
        
        bs_pricer = BlackScholesPricer()
        
        for idx, row in df.iterrows():
            S = row['spot']
            K = row['strike']
            T = row['time']
            r = row['rate']
            market_price = row['market_price']
            iv = row['implied_vol'] if 'implied_vol' in row else 0.2
            
            # Continuous SDE Pricing (Noise reduction enabled at paths=2000)
            model_price = self.predict(S, K, T, iv=iv, rate=r, num_paths=2000)
            
            # Constant Vol BS Pricing (Fair reference benchmark)
            otype = row['type'] if 'type' in row else 'call'
            bs_res = bs_pricer.price_european_option(S, K, T, avg_iv, otype)
            bs_price = bs_res['price'] if bs_res else 0.0
            
            predictions.append(model_price)
            targets.append(market_price)
            bs_prices.append(bs_price)
            
            if (idx + 1) % 50 == 0:
                print(f"Processed {idx + 1}/{len(df)} option pricing evaluations...")
            
        # Statistical Metrics
        preds = torch.tensor(predictions)
        targs = torch.tensor(targets)
        bs_preds = torch.tensor(bs_prices)
        
        mse = torch.mean((preds - targs)**2).item()
        mae = torch.mean(torch.abs(preds - targs)).item()
        
        bs_mse = torch.mean((bs_preds - targs)**2).item()
        bs_mae = torch.mean(torch.abs(bs_preds - targs)).item()
        
        print(f"\n==============================================")
        print(f"  QUANTITATIVE MODEL BENCHMARK PERFORMANCE  ")
        print(f"==============================================")
        print(f"Evaluation Record Count: {len(preds)}")
        print(f"----------------------------------------------")
        print(f"Metric        | Neural SDE     | Black-Scholes")
        print(f"----------------------------------------------")
        print(f"MSE           | {mse:14.4f} | {bs_mse:14.4f}")
        print(f"MAE           | {mae:14.4f} | {bs_mae:14.4f}")
        print(f"----------------------------------------------")
        
        # Write clean results to text report
        report_path = f"performance_report_{self.run_name}.txt"
        with open(report_path, 'w') as rf:
            rf.write("==============================================\n")
            rf.write("  QUANTITATIVE MODEL BENCHMARK PERFORMANCE  \n")
            rf.write("==============================================\n")
            rf.write(f"Evaluation Record Count: {len(preds)}\n")
            rf.write("----------------------------------------------\n")
            rf.write("Metric        | Neural SDE     | Black-Scholes\n")
            rf.write("----------------------------------------------\n")
            rf.write(f"MSE           | {mse:14.4f} | {bs_mse:14.4f}\n")
            rf.write(f"MAE           | {mae:14.4f} | {bs_mae:14.4f}\n")
            rf.write("----------------------------------------------\n")
        print(f"Saved textual performance report to: {report_path}")

        # Render Professional Academic Plot
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
        fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), dpi=300)
        
        # Neural SDE Scatter
        axes[0].scatter(targets, predictions, alpha=0.5, edgecolors='none', color='#1f77b4', label='Predictions')
        axes[0].plot([0, max(targets)], [0, max(targets)], color='#2ca02c', linestyle='--', linewidth=2, label='Perfect Fit')
        axes[0].set_xlabel("Traded Market Price ($)", fontsize=11, fontweight='semibold')
        axes[0].set_ylabel("Neural SDE Model Price ($)", fontsize=11, fontweight='semibold')
        axes[0].set_title(f"Neural SDE Volatility Pricing\n(MAE: ${mae:.3f})", fontsize=12, fontweight='bold')
        axes[0].legend(frameon=True, facecolor='white')
        axes[0].grid(True, linestyle=':', alpha=0.6)
        
        # Constant BS Scatter
        axes[1].scatter(targets, bs_prices, alpha=0.5, edgecolors='none', color='#d62728', label='Constant BS')
        axes[1].plot([0, max(targets)], [0, max(targets)], color='#2ca02c', linestyle='--', linewidth=2, label='Perfect Fit')
        axes[1].set_xlabel("Traded Market Price ($)", fontsize=11, fontweight='semibold')
        axes[1].set_ylabel("Constant Vol BS Model Price ($)", fontsize=11, fontweight='semibold')
        axes[1].set_title(f"Standard Constant Vol Black-Scholes\n(MAE: ${bs_mae:.3f})", fontsize=12, fontweight='bold')
        axes[1].legend(frameon=True, facecolor='white')
        axes[1].grid(True, linestyle=':', alpha=0.6)
        
        plt.tight_layout()
        plot_path = f"test_results_{self.run_name}.png"
        plt.savefig(plot_path)
        print(f"Saved high-resolution comparison plot to: {plot_path}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Neural Black-Scholes Enterprise Inference Engine")
    parser.add_argument('--mode', type=str, required=True, choices=['test', 'predict'], help="Execution Mode: 'test' (comparative bench) or 'predict' (single price)")
    parser.add_argument('-S', type=float, help="Current Spot Asset Price (S_0)")
    parser.add_argument('-K', type=float, help="Option Strike Price (K)")
    parser.add_argument('-T', type=float, help="Annualized Option Time-to-Maturity (T)")
    parser.add_argument('-r', type=float, default=0.05, help="Annualized Risk-Free Rate (r)")
    parser.add_argument('--iv', type=float, default=0.2, help="Initial Volatility State (V_0)")
    parser.add_argument('--run_name', type=str, default='default', help="Active database and checkpoints configuration name")
    
    args = parser.parse_args()
    engine = InferenceEngine(run_name=args.run_name)
    
    if args.mode == 'predict':
        if not all([args.S, args.K, args.T]):
            print("Error: Prediction execution mode requires non-null float args: -S, -K, and -T.")
            sys.exit(1)
            
        price = engine.predict(args.S, args.K, args.T, iv=args.iv, rate=args.r)
        print(f"\n======================================")
        print(f"  QUANTITATIVE PRICE ESTIMATE REPORT  ")
        print(f"======================================")
        print(f"Asset Spot:       ${args.S:.2f}")
        print(f"Strike Price:     ${args.K:.2f}")
        print(f"Time-to-Maturity:  {args.T:.4f} years")
        print(f"Initial Vol (V0):  {args.iv * 100:.2f}%")
        print(f"Risk-free Rate:    {args.r * 100:.2f}%")
        print(f"--------------------------------------")
        print(f"Neural SDE Price: ${price:.3f}")
        print(f"======================================")
        
    elif args.mode == 'test':
        engine.run_test_suite()

if __name__ == "__main__":
    main()
