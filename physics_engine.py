import QuantLib as ql
import torch
import numpy as np

class BlackScholesPricer:
    """
    Physics Engine using QuantLib for Black-Scholes Pricing and Greeks.
    This serves as the baseline model.
    """
    def __init__(self, risk_free_rate=0.05, dividend_yield=0.0):
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield
        self.calendar = ql.UnitedStates(ql.UnitedStates.NYSE)
        self.day_count = ql.Actual365Fixed()
        
    def price_european_option(self, spot_price, strike_price, time_to_maturity, volatility, option_type='call'):
        """
        Calculates Price and Greeks for a single European Option.
        """
        today = ql.Date.todaysDate()
        ql.Settings.instance().evaluationDate = today
        
        maturity_date = today + ql.Period(int(time_to_maturity * 365), ql.Days)
        
        # Market Data
        spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot_price))
        rate_handle = ql.YieldTermStructureHandle(ql.FlatForward(today, self.risk_free_rate, self.day_count))
        div_handle = ql.YieldTermStructureHandle(ql.FlatForward(today, self.dividend_yield, self.day_count))
        vol_handle = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, self.calendar, volatility, self.day_count))
        
        bsm_process = ql.BlackScholesMertonProcess(spot_handle, div_handle, rate_handle, vol_handle)
        
        # Option
        payoff = ql.PlainVanillaPayoff(ql.Option.Call if option_type.lower() == 'call' else ql.Option.Put, strike_price)
        exercise = ql.EuropeanExercise(maturity_date)
        option = ql.VanillaOption(payoff, exercise)
        
        option.setPricingEngine(ql.AnalyticEuropeanEngine(bsm_process))
        
        try:
            price = option.NPV()
            delta = option.delta()
            gamma = option.gamma()
            vega = option.vega()
            theta = option.theta()
            rho = option.rho()
            
            return {
                'price': price,
                'delta': delta,
                'gamma': gamma,
                'vega': vega,
                'theta': theta,
                'rho': rho
            }
        except RuntimeError:
            # Handle cases where QuantLib fails (e.g., negative volatility or extreme params)
            return None

    def price_batch(self, spots, strikes, times, volatilities, option_types):
        """
        Batch processing for tensors/arrays.
        Returns a dictionary of tensors.
        """
        results = []
        # Ensure inputs are iterable lists/arrays
        # This loop is CPU bound; for massive batches during training, consider 
        # implementing an analytical BS formula in PyTorch for speed if strict QuantLib accuracy isn't needed *inside* the loop.
        # But per requirements, we use QuantLib.
        
        for S, K, T, sigma, otype in zip(spots, strikes, times, volatilities, option_types):
            res = self.price_european_option(float(S), float(K), float(T), float(sigma), otype)
            if res is None:
                # Fallback or zero for invalid
                res = {'price': 0.0, 'delta': 0.0, 'gamma': 0.0, 'vega': 0.0, 'theta': 0.0, 'rho': 0.0}
            results.append(res)
            
        # Convert list of dicts to dict of tensors
        keys = results[0].keys()
        tensor_res = {k: torch.tensor([r[k] for r in results], dtype=torch.float32) for k in keys}
        return tensor_res

if __name__ == "__main__":
    # Test
    pricer = BlackScholesPricer()
    res = pricer.price_european_option(spot_price=100, strike_price=100, time_to_maturity=1.0, volatility=0.2, option_type='call')
    print("Test Result:", res)
