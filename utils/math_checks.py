import torch

def check_no_arbitrage(prices, tolerance=1e-4):
    """
    Checks if prices are non-negative.
    Returns a mask of violations.
    """
    return prices < -tolerance

def check_monotonicity(prices, strikes, option_type='call'):
    """
    Call prices should decrease as strike increases.
    Put prices should increase as strike increases.
    """
    # Requires sorted strikes to check monotonicity effectively
    pass

def sanity_check_greeks(delta, option_type='call'):
    """
    Delta for Call should be in [0, 1].
    Delta for Put should be in [-1, 0].
    """
    if option_type == 'call':
        return (delta >= 0) & (delta <= 1)
    else:
        return (delta >= -1) & (delta <= 0)

def penalize_arbitrage(prices, device='cpu'):
    """
    Loss component: Penalize negative prices.
    S = ReLU(-Price)
    """
    return torch.nn.functional.relu(-prices).mean()
