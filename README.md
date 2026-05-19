# Neural Black-Scholes: Beating the Formula

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![QuantLib](https://img.shields.io/badge/QuantLib-Baseline-green.svg)

## 📌 Project Overview
The **Neural Black-Scholes** project is a quantitative finance tool that replaces the traditional mathematical formulas used by banks to price options with an advanced Artificial Intelligence model called a **Neural Stochastic Differential Equation (Neural SDE)**. 

By observing real historical market data, the AI learns the hidden, complex dynamics of the stock market that classical mathematics fails to capture. 

## 🧠 The Approach: Equations & The Solution

### The Problem: Traditional Black-Scholes (BSM)
The standard Black-Scholes model assumes that a stock price $S_t$ moves according to Geometric Brownian Motion (GBM):
$$ dS_t = \mu S_t dt + \sigma S_t dW_t $$
Where:
- $\mu$ is a **constant** drift (expected return).
- $\sigma$ is a **constant** volatility.
- $dW_t$ is random noise (Brownian motion).

**The flaw:** In reality, market volatility is *never* constant. Deep In-The-Money (ITM) or Out-Of-The-Money (OTM) options have much higher volatility (the "Volatility Smile"). Because Black-Scholes assumes a flat, constant $\sigma$, it drastically misprices these options.

### The Solution: Neural SDEs
Instead of forcing the market to fit a rigid formula, we replace the constants with **Neural Networks**:
$$ dS_t = \mu_{\theta}(t, S_t) dt + \sigma_{\phi}(t, S_t) dW_t $$
Where:
- $\mu_{\theta}$ is a Neural Network learning the real market drift.
- $\sigma_{\phi}$ is a Neural Network learning the real market volatility (diffusion).

The AI is trained using the **Adjoint Sensitivity Method** (for O(1) memory efficiency) on real SPY options data. By learning the diffusion network $\sigma_{\phi}$, the model naturally recreates the "Volatility Smile" without any manual mathematical patching. 

---

## 📂 File Breakdown & What They Do

- **`neural_engine.py`**: The "Brain" of the project. It defines the PyTorch Neural Networks for the `Drift` and `Diffusion` functions and wraps them in a `torchsde` SDE solver.
- **`physics_engine.py`**: The "Baseline". It uses the industry-standard `QuantLib` C++ library to calculate the traditional Black-Scholes prices. This is what we benchmark our AI against.
- **`data_loader.py`**: The "Data Pipeline". It automatically connects to the Yahoo Finance API (`yfinance`), fetches live options chains, cleans the data, scales it to prevent neural network instability, and serves it as PyTorch Tensors.
- **`train.py`**: The "Gym". This script runs the SDE solver forward in time using Monte Carlo simulation, calculates the theoretical payoff of the option, and uses Backpropagation to minimize the Mean Squared Error (MSE) against the real market price.
- **`inference.py`**: The "Tool". A command-line interface that allows a user to input an option's strike and maturity, and instantly receive the AI-predicted fair value.
- **`paper.tex`**: An academic LaTeX paper explaining the project's methodology, mathematics, and results.

---

## 🏆 Benchmark Results

Tested on a hold-out set of 347 real-world At-The-Money (ATM) and Out-Of-The-Money (OTM) SPY options:

| Model | MSE (Mean Squared Error) | Avg Error ($) |
| :--- | :--- | :--- |
| **Neural SDE (Ours)** | **0.26** | **~$0.50** |
| Black-Scholes (Baseline) | 389.87 | ~$19.70 |

**Result:** The Neural SDE is roughly **40x more accurate** than the traditional Black-Scholes formula.

---

## 🚀 Usage Guide

### 1. Installation
Ensure you have Python 3.10+ installed. Install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Price an Option (Live Prediction)
To get an AI-predicted price for a specific option (e.g., Strike $500, Expires in 0.5 years):
```bash
python inference.py --mode predict -S 500 -K 500 -T 0.5
```

### 3. Verify Accuracy (Run the Benchmark)
To run the test suite and visualize the comparison against Black-Scholes yourself:
```bash
python inference.py --mode test
```
*(This will output a `test_results.png` scatter plot).*

### 4. Retrain the Model
Market conditions change. To retrain the AI on the latest market data:
```bash
python train.py
```
