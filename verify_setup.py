import sys
import torch
import numpy as np

print(f"Python: {sys.version}")
print(f"Torch: {torch.__version__}, CUDA: {torch.version.cuda}, Available: {torch.cuda.is_available()}")
print(f"Numpy: {np.__version__}")

try:
    import QuantLib as ql
    print(f"QuantLib: {ql.__version__}")
except ImportError as e:
    print(f"FAILED to import QuantLib: {e}")

try:
    import torchsde
    print(f"Torchsde: {torchsde.__version__}")
except ImportError as e:
    print(f"FAILED to import torchsde: {e}")

print("\n--- Checking Modules ---")
try:
    from physics_engine import BlackScholesPricer
    print("Physics Engine: OK")
except Exception as e:
    print(f"Physics Engine: FAILED - {e}")

try:
    from neural_engine import NeuralSDE
    print("Neural Engine: OK")
except Exception as e:
    print(f"Neural Engine: FAILED - {e}")

try:
    from data_loader import get_loader
    print("Data Loader: OK")
except Exception as e:
    print(f"Data Loader: FAILED - {e}")

print("\n--- Dry Run Check ---")
try:
    # Minimal dry run
    import yaml
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    print("Config Load: OK")
    
    # Check device
    device = torch.device(config['project']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Target Device: {device}")
    
    # Init Models
    sde = NeuralSDE(state_size=1, brownian_size=1, hidden_size=config['model']['hidden_size']).to(device)
    print("Model Init: OK")
    
    # Dummy Forward
    y0 = torch.zeros(2, 1).to(device)
    ts = torch.tensor([0.0, 1.0]).to(device)
    out = sde(y0, ts)
    print(f"Model Forward: OK, Output Shape: {out.shape}")
    
except Exception as e:
    print(f"Dry Run: FAILED - {e}")
