import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt

# Imports from our modules
from physics_engine import BlackScholesPricer
from neural_engine import NeuralSDE
from data_loader import get_loader
from utils.math_checks import penalize_arbitrage

# Load Config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

DEVICE = torch.device(config['project']['device'] if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

class ModelWrapper(nn.Module):
    def __init__(self, risk_free_rate=0.05):
        super().__init__()
        # State: [Stock Price, Volatility] -> dim=2
        self.sde = NeuralSDE(state_size=2, brownian_size=2, hidden_size=config['model']['hidden_size']).to(DEVICE)
        self.r = risk_free_rate
        
    def forward(self, spot, vol, times, rate_mean=0.05, num_paths=100):
        batch_size = spot.shape[0]
        
        # Prepare Batch for MC: [Stock, Vol]
        spot_expanded = spot.repeat_interleave(num_paths).unsqueeze(1) 
        vol_expanded = vol.repeat_interleave(num_paths).unsqueeze(1)
        y0 = torch.cat([spot_expanded, vol_expanded], dim=-1) # (B*P, 2)
        
        # Sort Times
        sorted_times, sort_idx = torch.sort(times)
        unique_times, inverse_indices = torch.unique(sorted_times, return_inverse=True)
        if unique_times[0] != 0:
            ts = torch.cat([torch.tensor([0.0], device=DEVICE), unique_times])
        else:
            ts = unique_times
            
        # SDE Solve
        # Expected output: (len(ts), batch*paths, 2)
        trajectory = self.sde(y0, ts, r=rate_mean) 
        
        # Extract S_T for each sample
        time_indices = torch.searchsorted(ts, times)
        idx_expanded = time_indices.repeat_interleave(num_paths) # (B*P,)
        
        # trajectory is (Time, Batch, State)
        # We want [idx_expanded[i], i, 0] for i in 0..BP to get the Stock Price (dim 0)
        final_spots = trajectory[idx_expanded, torch.arange(batch_size * num_paths, device=DEVICE), 0]
        final_spots = final_spots.view(batch_size, num_paths)
        
        return final_spots

def train(run_name='default'):
    # Setup
    physics = BlackScholesPricer()
    model = ModelWrapper().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])
    cache_file = f"options_data_{run_name}.csv" if run_name != 'default' else "options_data.csv"
    loader = get_loader(tickers=config['data']['tickers'], batch_size=config['training']['batch_size'], cache_file=cache_file)
    
    criterion = nn.MSELoss()
    
    print("Starting Training...")
    for epoch in range(config['training']['epochs']):
        total_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, (x, y_market) in enumerate(pbar):
            # x: [Spot, Strike, Time, Rate, IV]
            spot = x[:, 0].to(DEVICE)
            k = x[:, 1].to(DEVICE)
            t = x[:, 2].to(DEVICE)
            r = x[:, 3].to(DEVICE)
            iv = x[:, 4].to(DEVICE)
            y_market = y_market.to(DEVICE)
            
            optimizer.zero_grad()
            
            # 1. Physics Baseline
            with torch.no_grad():
                bs_prices_dict = physics.price_batch(spot.cpu(), k.cpu(), t.cpu(), iv.cpu().tolist(), ['call']*len(spot))
                bs_price = bs_prices_dict['price'].to(DEVICE).unsqueeze(1) # (B, 1)
            
            # 2. Neural Prediction (SDE Monte Carlo)
            paths = 100 # Increased to 100 to eliminate training gradient noise and maximize convergence accuracy
            final_spots = model(spot, iv, t, rate_mean=r.mean().item(), num_paths=paths) # returns (B, Paths)
            
            # Payoff: max(S_T - K, 0)
            payoffs = torch.relu(final_spots - k.unsqueeze(1))
            
            # Discount
            # e^(-rT)
            # t is (B,)
            discount = torch.exp(-r * t).unsqueeze(1)
            
            mc_prices = (payoffs * discount).mean(dim=1).unsqueeze(1) # (B, 1)
            
            predicted_price = mc_prices 
            
            # Physics-Informed Loss Function:
            # 1. loss_mse: Measures the distance between the SDE model's predictions and real-market traded prices.
            # 2. loss_arb: Arbitrage penalty to prevent negative option prices, ensuring physical and economic consistency.
            
            # Loss
            loss_mse = criterion(predicted_price, y_market)
            loss_arb = penalize_arbitrage(predicted_price)
            
            loss = loss_mse + loss_arb
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item(), 'BS_Ref': bs_price.mean().item()})

        print(f"Epoch {epoch+1} Average Loss: {total_loss / len(loader)}")
        
        # Checkpoint
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), f"{config['training']['save_dir']}/model_ep{epoch+1}_{run_name}.pth")

    # Save Final
    torch.save(model.state_dict(), f"{config['training']['save_dir']}/model_final_{run_name}.pth")
    print(f"Saved final model to {config['training']['save_dir']}/model_final_{run_name}.pth")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default='default')
    args = parser.parse_args()

    import os
    if not os.path.exists(config['training']['save_dir']):
        os.makedirs(config['training']['save_dir'])
        
    train(run_name=args.run_name)
